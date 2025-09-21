#!/usr/bin/env python3
"""
Telegram signal bot (single-file MVP)
- Exchange: Bitget USDT-perp via ccxt
- Symbols: BTC/USDT, ETH/USDT, SOL/USDT (monitor all, one active position at a time)
- Indicators: Bollinger Bands (20,2), CCI(20)
- Logic: 4h filter (trend), 1h + 5m entry (band-touch + CCI turn),
         TP1 at mid band (partial), TP2 as opposite band / HTF band / fixed R
- Alerts: Telegram Bot API (long polling)
- Runtime config: /set key value, /symbols, /newsblock, /status, /reset

Requirements: pip install ccxt pandas numpy python-dotenv requests
(Optional) APScheduler is NOT required — this file uses a simple scheduler loop.

ENV:
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...   # only this chat is authorized for commands

Run:
  python telegram_signal_bot.py
"""
import os, json, time, threading, math, queue
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import ccxt

# ---------------------------
# Paths & Globals
# ---------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(DATA_DIR, "config_runtime.json")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
LOCK = threading.RLock()

KST = timezone(timedelta(hours=9))

# ---------------------------
# Default Config (can be changed at runtime via /set)
# ---------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "timeframe_5m": "5m",
    "timeframe_1h": "1h",
    "timeframe_4h": "4h",
    "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
    "primary_only": True,     # one active position at a time
    "risk_pct": 0.015,        # 1.5% of equity per trade
    "equity": 3012.0,         # update with /set equity 3500 etc.
    "tp1_ratio": 0.5,         # 50% at mid band
    "tp2_mode": "R_MULTIPLE",# OPPOSITE_BAND | HTF_BAND | R_MULTIPLE
    "tp2_r": 1.2,             # if R_MULTIPLE
    "cci_turn_threshold": 0,  # 0 or 20 (sensitivity)
    "band_touch_window": 20,  # lookback window for counting touches
    "band_touch_min": 3,      # need >=3 touches within window
    "news_blocks": [],        # list of {"start":"HH:MM","end":"HH:MM"}
    "chat_id": os.getenv("OKX_TELEGRAM_CHAT_ID", ""),
    "timezone": "Asia/Seoul",
}

# ---------------------------
# State (persisted)
# ---------------------------
DEFAULT_STATE: Dict[str, Any] = {
    "last_checked": {},           # {tf: timestamp_ms}
    "last_signal_hash": set(),    # recent signal hashes to prevent duplicates
    "open_position": None,        # {symbol, side, entry, sl, tp1, tp2, size, moved_be}
}


def load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # set() not JSON serializable, handle manually
            if path == STATE_PATH and "last_signal_hash" in data and isinstance(data["last_signal_hash"], list):
                data["last_signal_hash"] = set(data["last_signal_hash"])
            return data
    except Exception:
        return default.copy()


def save_json(path: str, data: Any) -> None:
    tmp = data.copy()
    if path == STATE_PATH and isinstance(tmp.get("last_signal_hash"), set):
        tmp["last_signal_hash"] = list(tmp["last_signal_hash"])  # serialize set
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tmp, f, ensure_ascii=False, indent=2)


CONFIG = load_json(CFG_PATH, DEFAULT_CONFIG)
STATE = load_json(STATE_PATH, DEFAULT_STATE)

# ---------------------------
# Telegram
# ---------------------------
TG_TOKEN = os.getenv("OKX_TELEGRAM_TOKEN", "")
TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"
TG_OFFSET = 0


def tg_send(text: str) -> None:
    chat = CONFIG.get("chat_id") or os.getenv("TELEGRAM_CHAT_ID", "")
    if not TG_TOKEN or not chat:
        print("[WARN] Telegram not configured. Message:", text)
        return
    try:
        requests.post(f"{TG_API}/sendMessage", json={"chat_id": chat, "text": text})
    except Exception as e:
        print("[TG ERROR]", e)


def tg_poll_loop(cmd_queue: "queue.Queue[dict]"):
    global TG_OFFSET
    chat = CONFIG.get("chat_id") or os.getenv("TELEGRAM_CHAT_ID", "")
    while True:
        try:
            resp = requests.get(f"{TG_API}/getUpdates", params={"timeout": 50, "offset": TG_OFFSET+1}, timeout=60)
            data = resp.json()
            for upd in data.get("result", []):
                TG_OFFSET = max(TG_OFFSET, upd["update_id"])
                msg = upd.get("message") or {}
                if not msg:
                    continue
                if str(msg.get("chat", {}).get("id")) != str(chat):
                    continue  # ignore unauthorized chats
                text = (msg.get("text") or "").strip()
                if text.startswith("/"):
                    cmd_queue.put({"text": text, "ts": msg.get("date")})
        except Exception:
            time.sleep(1)


# ---------------------------
# Exchange (Bitget swap via ccxt)
# ---------------------------
EX = ccxt.bitget({
    "enableRateLimit": True,
    "options": {"defaultType": "swap"},
})


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 240) -> pd.DataFrame:
    """Fetch OHLCV and return DataFrame with tz-aware datetime index (KST)."""
    o = EX.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(KST)
    df.set_index("dt", inplace=True)
    return df


# ---------------------------
# Indicators
# ---------------------------

def bollinger(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.DataFrame:
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    out = pd.DataFrame({"mid": mid, "upper": upper, "lower": lower})
    return out


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(period).mean()
    mad = (tp - sma).abs().rolling(period).mean()
    cci = (tp - sma) / (0.015 * mad)
    return cci


# ---------------------------
# Strategy helpers
# ---------------------------

def count_band_touches(df: pd.DataFrame, bb: pd.DataFrame, window: int, side: str) -> int:
    """
    Count closes beyond band within the last `window` candles for given `side`:
    side='lower' → close <= lower, side='upper' → close >= upper
    """
    sub = df.tail(window)
    if side == "lower":
        return int((sub["close"] <= bb.loc[sub.index, "lower"]).sum())
    else:
        return int((sub["close"] >= bb.loc[sub.index, "upper"]).sum())


def cci_turn(sig: pd.Series, direction: str, thr: float = 0.0) -> bool:
    """Detect CCI turning: direction='up' or 'down'. Uses last two values."""
    if len(sig) < 3:
        return False
    a, b = sig.iloc[-2], sig.iloc[-1]
    if direction == "up":
        return (a < thr) and (b > thr)
    else:
        return (a > -thr) and (b < -thr)


def h4_filter(df4: pd.DataFrame, bb4: pd.DataFrame, cci4: pd.Series) -> str:
    """Return 'Bull','Bear','Neutral' based on 4h band break & CCI extreme."""
    c = df4["close"].iloc[-1]
    up = bb4["upper"].iloc[-1]
    lo = bb4["lower"].iloc[-1]
    x = cci4.iloc[-1]
    if c >= up and x >= 100:
        return "Bull"
    if c <= lo and x <= -100:
        return "Bear"
    # Neutral range bias by ±50 CCI (optional)
    return "Neutral"


def entry_signal(df1: pd.DataFrame, bb1: pd.DataFrame, cci1: pd.Series,
                 df5: pd.DataFrame, bb5: pd.DataFrame, cci5: pd.Series,
                 trend4h: str, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Identify entry side and compute trade plan skeleton.
    Returns dict with {side, reason} or None.
    Rules (summarized):
      - Need multiple band touches on the higher intraday TF (1h) matching side
      - Confirm with 5m CCI turn in same direction
      - 4h filter: if Bull → only LONG; if Bear → only SHORT; Neutral → both allowed
    """
    win = int(cfg.get("band_touch_window", 20))
    need = int(cfg.get("band_touch_min", 3))
    thr = float(cfg.get("cci_turn_threshold", 0))

    # Check 1h band touches
    lower_touches_1h = count_band_touches(df1, bb1, win, "lower")
    upper_touches_1h = count_band_touches(df1, bb1, win, "upper")

    # LONG candidate: 1h lower touches >= need AND 1h CCI turning up, confirm 5m upturn
    long_ok_1h = (lower_touches_1h >= need) and cci_turn(cci1, "up", thr)
    long_ok_5m = cci_turn(cci5, "up", thr)

    # SHORT candidate
    short_ok_1h = (upper_touches_1h >= need) and cci_turn(cci1, "down", thr)
    short_ok_5m = cci_turn(cci5, "down", thr)

    # Trend filter
    allow_long = trend4h in ("Bull","Neutral")
    allow_short = trend4h in ("Bear","Neutral")

    if allow_long and long_ok_1h and long_ok_5m:
        return {"side": "LONG", "reason": f"4h={trend4h}, 1h lower touches={lower_touches_1h}, CCI turn up + 5m confirm"}
    if allow_short and short_ok_1h and short_ok_5m:
        return {"side": "SHORT", "reason": f"4h={trend4h}, 1h upper touches={upper_touches_1h}, CCI turn down + 5m confirm"}
    return None


def make_trade_plan(symbol: str, side: str,
                    df5: pd.DataFrame, bb5: pd.DataFrame,
                    df1: pd.DataFrame, bb1: pd.DataFrame,
                    df4: pd.DataFrame, bb4: pd.DataFrame,
                    cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute entry (last close), SL, TP1, TP2 (according to config tp2_mode)
    """
    entry = float(df5["close"].iloc[-1])
    # SL: recent swing + band concept (simple): use last 1h band beyond
    if side == "LONG":
        sl = float(min(df1["low"].tail(3).min(), bb1["lower"].iloc[-1]))
    else:
        sl = float(max(df1["high"].tail(3).max(), bb1["upper"].iloc[-1]))

    risk = abs(entry - sl)
    # TP1: 5m mid band
    tp1 = float(bb5["mid"].iloc[-1])

    mode = cfg.get("tp2_mode", "R_MULTIPLE").upper()
    if mode == "OPPOSITE_BAND":
        tp2 = float(bb5["upper"].iloc[-1] if side == "LONG" else bb5["lower"].iloc[-1])
    elif mode == "HTF_BAND":
        # Use 1h mid/upper/lower according to side
        if side == "LONG":
            tp2 = float(max(bb1["mid"].iloc[-1], bb1["upper"].iloc[-1]))
        else:
            tp2 = float(min(bb1["mid"].iloc[-1], bb1["lower"].iloc[-1]))
    else:  # R_MULTIPLE
        R = float(cfg.get("tp2_r", 1.2))
        tp2 = float(entry + R * risk if side == "LONG" else entry - R * risk)

    return {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2}


def position_size(entry: float, sl: float, equity: float, risk_pct: float) -> float:
    risk_abs = equity * risk_pct
    risk_per_unit = abs(entry - sl)
    if risk_per_unit <= 0:
        return 0.0
    qty = risk_abs / risk_per_unit
    return max(0.0, qty)


# ---------------------------
# Engine Loop
# ---------------------------

def within_news_block(now_kst: datetime, blocks: List[Dict[str,str]]) -> bool:
    hm = now_kst.strftime("%H:%M")
    for b in blocks:
        if b.get("start") <= hm <= b.get("end"):
            return True
    return False


def candle_closed_timestamp(tf: str, now_utc: datetime) -> int:
    """Return the latest fully-closed candle timestamp(ms) for timeframe tf."""
    unit = {"5m": 300, "15m":900, "1h": 3600, "4h": 14400, "1d":86400}[tf]
    secs = int(now_utc.timestamp())
    closed = secs - (secs % unit)  # align to candle start
    return (closed - 1) * 1000  # ts just before close (ms)


def hash_signal(symbol: str, side: str, ts_key: str) -> str:
    return f"{symbol}|{side}|{ts_key}"


def evaluate_once():
    with LOCK:
        cfg = CONFIG
        st = STATE
    now_utc = datetime.now(timezone.utc)
    now_kst = now_utc.astimezone(KST)

    # News block
    if within_news_block(now_kst, CONFIG.get("news_blocks", [])):
        return  # skip entries during block (positions are monitored below anyway)

    # Process each symbol for signals (but open only one position at a time)
    symbols: List[str] = cfg.get("symbols", [])

    # Fetch data per symbol
    for sym in symbols:
        try:
            df5 = fetch_ohlcv(sym, cfg["timeframe_5m"], limit=200)
            df1 = fetch_ohlcv(sym, cfg["timeframe_1h"], limit=200)
            df4 = fetch_ohlcv(sym, cfg["timeframe_4h"], limit=200)
        except Exception as e:
            print(f"[DATA] {sym} fetch error: {e}")
            continue

        bb5, cci5 = bollinger(df5), cci(df5)
        bb1, cci1 = bollinger(df1), cci(df1)
        bb4, cci4 = bollinger(df4), cci(df4)

        trend = h4_filter(df4, bb4, cci4)
        sig = entry_signal(df1, bb1, cci1, df5, bb5, cci5, trend, cfg)
        ts_key = str(df5.index[-1])  # last 5m candle closed time as key

        # Monitor existing position hits for any symbol
        with LOCK:
            pos = STATE.get("open_position")
        if pos:
            if monitor_hits(pos, df5):
                with LOCK:
                    save_json(STATE_PATH, STATE)
            # If a position is open and primary_only, do not open another.
            continue

        if not sig:
            continue

        # Avoid duplicate alert for same candle+symbol+side
        h = hash_signal(sym, sig["side"], ts_key)
        with LOCK:
            seen = STATE.get("last_signal_hash", set())
            if h in seen:
                continue
            seen.add(h)
            STATE["last_signal_hash"] = seen

        # Make trade plan (for alert only)
        plan = make_trade_plan(sym, sig["side"], df5, bb5, df1, bb1, df4, bb4, cfg)
        size = position_size(plan["entry"], plan["sl"], cfg.get("equity", 0), cfg.get("risk_pct", 0.015))
        txt = (
            f"[ENTRY SIGNAL] {sym} {sig['side']}\n"
            f"• Time: {now_kst.strftime('%Y-%m-%d %H:%M KST')}\n"
            f"• Reason: {sig['reason']}\n"
            f"• Entry: {plan['entry']:.4f}\n"
            f"• SL: {plan['sl']:.4f}  (Risk {cfg.get('risk_pct',0.015)*100:.1f}%)\n"
            f"• TP1: {plan['tp1']:.4f} (BB mid, {int(cfg.get('tp1_ratio',0.5)*100)}%)\n"
            f"• TP2: {plan['tp2']:.4f} ({cfg.get('tp2_mode')})\n"
            f"• Size est: {size:.6f} (base units)\n"
        )
        tg_send(txt)

        # Open position in state (virtual tracking)
        with LOCK:
            STATE["open_position"] = {
                "symbol": sym,
                "side": sig["side"],
                "entry": plan["entry"],
                "sl": plan["sl"],
                "tp1": plan["tp1"],
                "tp2": plan["tp2"],
                "size": size,
                "moved_be": False,
            }
            save_json(STATE_PATH, STATE)


def monitor_hits(pos: Dict[str, Any], df5: pd.DataFrame) -> bool:
    """Check TP/SL hits using last 5m candle high/low. Update TG + state. Return True if state changed."""
    changed = False
    sym, side = pos["symbol"], pos["side"]
    last = df5.iloc[-1]
    high, low, close = float(last["high"]), float(last["low"]), float(last["close"])

    # SL hit?
    if side == "LONG" and low <= pos["sl"]:
        tg_send(f"[STOP LOSS] {sym} LONG @ ~{pos['sl']:.4f}")
        with LOCK:
            STATE["open_position"] = None
            changed = True
        return changed
    if side == "SHORT" and high >= pos["sl"]:
        tg_send(f"[STOP LOSS] {sym} SHORT @ ~{pos['sl']:.4f}")
        with LOCK:
            STATE["open_position"] = None
            changed = True
        return changed

    # TP1 hit?
    if side == "LONG" and high >= pos["tp1"] and not pos.get("moved_be"):
        tg_send(f"[TP1 HIT] {sym} LONG @ {pos['tp1']:.4f} — 50% closed, SL→BE")
        with LOCK:
            STATE["open_position"]["sl"] = pos["entry"]  # move to break-even
            STATE["open_position"]["moved_be"] = True
            changed = True
    if side == "SHORT" and low <= pos["tp1"] and not pos.get("moved_be"):
        tg_send(f"[TP1 HIT] {sym} SHORT @ {pos['tp1']:.4f} — 50% closed, SL→BE")
        with LOCK:
            STATE["open_position"]["sl"] = pos["entry"]
            STATE["open_position"]["moved_be"] = True
            changed = True

    # TP2 hit?
    pos = STATE.get("open_position")
    if not pos:
        return True
    if side == "LONG" and high >= pos["tp2"]:
        tg_send(f"[FINAL TP] {sym} LONG @ {pos['tp2']:.4f} — position closed ✅")
        with LOCK:
            STATE["open_position"] = None
            changed = True
    if side == "SHORT" and low <= pos["tp2"]:
        tg_send(f"[FINAL TP] {sym} SHORT @ {pos['tp2']:.4f} — position closed ✅")
        with LOCK:
            STATE["open_position"] = None
            changed = True

    return changed


# ---------------------------
# Command handling
# ---------------------------

def cmd_status() -> str:
    with LOCK:
        cfg = CONFIG.copy()
        pos = STATE.get("open_position")
    lines = [
        f"Equity: {cfg.get('equity')} | Risk%: {cfg.get('risk_pct')} | TP1: {cfg.get('tp1_ratio')} | TP2: {cfg.get('tp2_mode')} {cfg.get('tp2_r', '')}",
        f"Symbols: {', '.join(cfg.get('symbols', []))}",
        f"News blocks: {cfg.get('news_blocks')}",
    ]
    if pos:
        lines.append(f"Position: {pos['symbol']} {pos['side']} @ {pos['entry']:.4f} | SL {pos['sl']:.4f} | TP1 {pos['tp1']:.4f} | TP2 {pos['tp2']:.4f}")
    else:
        lines.append("Position: None")
    return "\n".join(lines)


def cmd_set(key: str, value: str) -> str:
    with LOCK:
        if key in ("risk_pct","equity","tp1_ratio","tp2_r","cci_turn_threshold"):
            try:
                CONFIG[key] = float(value)
            except Exception:
                return f"Invalid float for {key}"
        elif key in ("tp2_mode","timezone","primary_only"):
            if key == "primary_only":
                CONFIG[key] = (value.lower() == "true")
            else:
                CONFIG[key] = value
        else:
            return f"Unknown key: {key}"
        save_json(CFG_PATH, CONFIG)
    return f"Updated: {key}={CONFIG[key]}"


def cmd_symbols(csv_list: str) -> str:
    arr = [s.strip() for s in csv_list.split(",") if s.strip()]
    with LOCK:
        CONFIG["symbols"] = arr
        save_json(CFG_PATH, CONFIG)
    return f"Monitoring symbols updated: {', '.join(arr)}"


def cmd_newsblock(span: str) -> str:
    try:
        start, end = [x.strip() for x in span.split("-")]
    except Exception:
        return "Usage: /newsblock HH:MM-HH:MM"
    with LOCK:
        blocks = CONFIG.get("news_blocks", [])
        blocks.append({"start": start, "end": end})
        CONFIG["news_blocks"] = blocks
        save_json(CFG_PATH, CONFIG)
    return f"Added news-block window: {start}~{end}"


def cmd_reset() -> str:
    with LOCK:
        STATE.clear()
        STATE.update(DEFAULT_STATE)
        save_json(STATE_PATH, STATE)
    return "State reset."


def command_worker(cmd_queue: "queue.Queue[dict]"):
    while True:
        item = cmd_queue.get()
        text = item.get("text","/")
        parts = text.split()
        cmd = parts[0].lower()
        if cmd == "/status":
            tg_send(cmd_status())
        elif cmd == "/set" and len(parts) >= 3:
            tg_send(cmd_set(parts[1], parts[2]))
        elif cmd == "/symbols" and len(parts) >= 2:
            tg_send(cmd_symbols(" ".join(parts[1:])))
        elif cmd == "/newsblock" and len(parts) >= 2:
            tg_send(cmd_newsblock(parts[1]))
        elif cmd == "/reset":
            tg_send(cmd_reset())
        else:
            tg_send("Commands: /status | /set <key> <value> | /symbols <csv> | /newsblock HH:MM-HH:MM | /reset")


# ---------------------------
# Main loop
# ---------------------------

def main():
    tg_send("Bot started. /status for info.")

    cmd_q: "queue.Queue[dict]" = queue.Queue()
    threading.Thread(target=tg_poll_loop, args=(cmd_q,), daemon=True).start()
    threading.Thread(target=command_worker, args=(cmd_q,), daemon=True).start()

    # evaluation loop (every 60s)
    while True:
        try:
            evaluate_once()
            # Periodic save
            with LOCK:
                save_json(CFG_PATH, CONFIG)
                save_json(STATE_PATH, STATE)
        except Exception as e:
            print("[LOOP ERROR]", e)
        time.sleep(60)


if __name__ == "__main__":
    main()
