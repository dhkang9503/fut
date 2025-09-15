# -*- coding: utf-8 -*-
"""
5m-Only BB+CCI Futures Scanner (Binance WebSocket) — Readable Telegram Logs
- Entry & management timeframe: 5m (no 1h filter)
- Symbols: BTCUSDT, ETHUSDT, SOLUSDT
- Features:
  * Multiple concurrent positions (per symbol)
  * Partial TP (50%) at +0.3% then Breakeven SL
  * Condition-based TP/SL (BB mid / bands + CCI)
  * Pre-event risk-off: CPI/PPI auto (monthly), FOMC manual — auto EXIT 5m before
  * Telegram commands: /ping /status /help
  * Chart snapshots on ENTRY/EXIT
  * Deduplicated alerts & cooldown
  * English, readable Telegram logs with emojis
"""

import os, io, json, time, threading, calendar
from collections import deque
import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from websocket import WebSocketApp

# ========= User Settings =========
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
ENTRY_INTERVAL = "5m"   # 5-minute entry/management
STREAMS = [f"{s.lower()}@kline_{ENTRY_INTERVAL}" for s in SYMBOLS]
WS_URL  = "wss://fstream.binance.com/stream?streams=" + "/".join(STREAMS)

# Indicators
BB_N, BB_K = 20, 2.0
CCI_N = 20
CCI_UP, CCI_DN = 100, -100
TOUCH_LOOKBACK = 3

# Partial TP / Breakeven
PARTIAL_TP_PCT = 0.003     # +0.3% reach -> partial 50%
PARTIAL_SIZE    = 0.5
MOVE_SL_TO_BE   = True

# Event risk-off (EXIT 5m before)
EVENT_EXIT_BEFORE_MIN = 5

# Misc
SEND_CHARTS_ON_CONFIRM = True
SEND_CHARTS_ON_EXIT    = True
MAX_KEEP = 600
KST = ZoneInfo("Asia/Seoul")

# Telegram
TELEGRAM_TOKEN   = os.environ.get("OKX_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("OKX_TELEGRAM_CHAT_ID")

# ========= Globals =========
START_TS = time.time()
LAST_LOOP_TS = time.time()
LAST_LOOP_AT = None

# Buffers (5m only)
buf_5m = {s: pd.DataFrame() for s in SYMBOLS}

# Positions: per symbol or None
# {"side": "long/short", "entry": float, "entry_time": dt, "ptp_done": bool, "be_active": bool}
positions = {s: None for s in SYMBOLS}

# Dedup / Cooldown
recent_msgs = deque(maxlen=300)
EXIT_COOLDOWN_S = 30
last_exit_at = {s: 0.0 for s in SYMBOLS}

# ========= Telegram helpers =========
def tg(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG]", text); return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
                      timeout=10)
    except Exception as e:
        print("TG error:", e)

def tg_photo(png_bytes, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG photo]", caption); return
    try:
        files = {"photo": ("chart.png", png_bytes, "image/png")}
        data  = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                      data=data, files=files, timeout=15)
    except Exception as e:
        print("TG photo error:", e)

def tg_dedup_send(text: str, key: str, ttl_s: int = 120):
    now = time.time()
    # purge old
    for k, ts in list(recent_msgs):
        if now - ts > ttl_s:
            try: recent_msgs.remove((k, ts))
            except: pass
    if any(k == key for k, _ in recent_msgs):
        return
    recent_msgs.append((key, now))
    tg(text)

# ========= Indicators / Signals =========
def compute_bb_cci(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"]
    ma = close.rolling(BB_N).mean()
    std = close.rolling(BB_N).std(ddof=0)
    d["bb_mid"] = ma
    d["bb_up"]  = ma + BB_K * std
    d["bb_dn"]  = ma - BB_K * std
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    sma_tp = tp.rolling(CCI_N).mean()
    mad = (tp - sma_tp).abs().rolling(CCI_N).mean()
    d["cci"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    d["cci"] = d["cci"].fillna(0)
    return d

def cross_up(prev, cur, level):   return (prev <= level) and (cur > level)

def cross_down(prev, cur, level): return (prev >= level) and (cur < level)

def recent_touch(series_close, series_band, lookback, mode):
    sub_c = series_close.tail(lookback)
    sub_b = series_band.tail(lookback)
    if len(sub_c) == 0 or len(sub_b) == 0: return False
    return bool((sub_c <= sub_b).any() if mode=="below" else (sub_c >= sub_b).any())

def entry_signal_live(d: pd.DataFrame):
    """5m close signal only"""
    if len(d) < max(BB_N, CCI_N)+2: return None
    last, prev = d.iloc[-1], d.iloc[-2]

    # LONG: BB lower reclaim + CCI cross up (-100)
    long_setup = recent_touch(d["close"].iloc[:-1], d["bb_dn"].iloc[:-1], TOUCH_LOOKBACK, "below")
    long_cross = cross_up(prev["cci"], last["cci"], CCI_DN)
    long_band  = (prev["close"] <= prev["bb_dn"]) and (last["close"] > last["bb_dn"])
    if long_setup and long_cross and long_band:
        return {"side":"long", "reason":"BB lower reclaim + CCI cross up (-100)"}

    # SHORT: BB upper reject + CCI cross down (+100)
    short_setup = recent_touch(d["close"].iloc[:-1], d["bb_up"].iloc[:-1], TOUCH_LOOKBACK, "above")
    short_cross = cross_down(prev["cci"], last["cci"], CCI_UP)
    short_band  = (prev["close"] >= prev["bb_up"]) and (last["close"] < last["bb_up"])
    if short_setup and short_cross and short_band:
        return {"side":"short", "reason":"BB upper reject + CCI cross down (+100)"}
    return None

def confirm_signal_close(d: pd.DataFrame):
    return entry_signal_live(d)

def exit_signal_bb_cci(d: pd.DataFrame, side: str):
    if len(d) < max(BB_N, CCI_N)+2: return None
    last, prev = d.iloc[-1], d.iloc[-2]
    if side == "long":
        if (last["close"] >= last["bb_mid"] and last["cci"] > 0):
            return {"type":"tp", "reason":"BB mid reached + CCI > 0"}
        if (last["close"] < last["bb_dn"]) or cross_down(prev["cci"], last["cci"], CCI_DN):
            return {"type":"sl", "reason":"BB lower break OR CCI < -100"}
    else:
        if (last["close"] <= last["bb_mid"] and last["cci"] < 0):
            return {"type":"tp", "reason":"BB mid reached + CCI < 0"}
        if (last["close"] > last["bb_up"]) or cross_up(prev["cci"], last["cci"], CCI_UP):
            return {"type":"sl", "reason":"BB upper break OR CCI > +100"}
    return None

# ========= PnL / Partial TP =========
def pnl_pct(side: str, entry_px: float, cur_px: float) -> float:
    p = (cur_px - entry_px) / entry_px
    return p if side == "long" else -p

def check_partial_tp(sym: str, df: pd.DataFrame, pos: dict) -> bool:
    last = df.iloc[-1]
    cur_px = last["close"]
    p = pnl_pct(pos["side"], pos["entry"], cur_px)
    if not pos["ptp_done"] and p >= PARTIAL_TP_PCT:
        msg = (
            f"🎯 [PARTIAL TP {int(PARTIAL_SIZE*100)}%] {sym} {pos['side'].upper()}\n"
            f"• Price: {cur_px:.2f}\n"
            f"• PnL: {p*100:.2f}%\n"
            f"• Status: Remaining {int((1-PARTIAL_SIZE)*100)}% → BE active ({pos['entry']:.2f})"
        )
        tg(msg)
        pos["ptp_done"] = True
        if MOVE_SL_TO_BE:
            pos["be_active"] = True
        return True
    return False

def stop_out(sym: str, df: pd.DataFrame, pos: dict):
    last = df.iloc[-1]
    px = last["close"]
    if pos.get("be_active"):
        if (pos["side"]=="long" and px <= pos["entry"]) or (pos["side"]=="short" and px >= pos["entry"]):
            return {"type":"sl_be", "reason":"BE stop triggered"}
    return exit_signal_bb_cci(df, pos["side"])

# ========= Chart =========
def make_chart_png(d: pd.DataFrame, symbol: str, entry_time_utc=None, exit_time_utc=None, entry_px=None, exit_px=None):
    dd = d.copy()
    times = pd.to_datetime(dd["close_time"], utc=True).tz_convert(KST)
    x = times.map(mdates.date2num)
    colors = ["#4DD2E6" if c>=o else "#FC495C" for o,c in zip(dd["open"], dd["close"])]
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6), sharex=True)
    for ax in (ax1, ax2):
        ax.set_facecolor("black"); ax.tick_params(colors="white"); ax.grid(True, alpha=0.2, color="white")
    fig.patch.set_facecolor("black")
    w = 0.8 / 1440.0
    for xi, o, h, l, c, col in zip(x, dd["open"], dd["high"], dd["low"], dd["close"], colors):
        ax1.plot([xi, xi], [l, h], color=col, linewidth=1)
        body_low, body_high = min(o, c), max(o, c)
        ax1.add_patch(plt.Rectangle((xi, body_low), w, body_high - body_low, color=col, alpha=0.9, linewidth=0))
    if "bb_mid" in dd:
        ax1.plot(x, dd["bb_mid"], linewidth=1.0, alpha=0.9, color="white")
        ax1.plot(x, dd["bb_up"],  linewidth=0.9, alpha=0.9, color="gray")
        ax1.plot(x, dd["bb_dn"],  linewidth=0.9, alpha=0.9, color="gray")
    if entry_time_utc and entry_px:
        ax1.scatter(mdates.date2num(entry_time_utc.astimezone(KST)), entry_px, s=80, marker="^",
                    color="#4DD2E6", edgecolors="white", linewidths=0.5, zorder=5, label="Entry")
    if exit_time_utc and exit_px:
        ax1.scatter(mdates.date2num(exit_time_utc.astimezone(KST)), exit_px, s=80, marker="v",
                    color="#FC495C", edgecolors="white", linewidths=0.5, zorder=5, label="Exit")
    ax1.legend(facecolor="black", edgecolor="white", labelcolor="white")
    ax2.plot(x, dd["cci"], linewidth=1.0)
    ax2.axhline(CCI_UP, color="white", linewidth=0.7, alpha=0.6)
    ax2.axhline(0,      color="white", linewidth=0.6, alpha=0.3, linestyle="--")
    ax2.axhline(CCI_DN, color="white", linewidth=0.7, alpha=0.6)
    ax1.set_title(f"{symbol} {ENTRY_INTERVAL}  BB({BB_N},{BB_K}) / CCI({CCI_N})", color="white")
    fmt = mdates.DateFormatter('%m-%d %H:%M', tz=KST)
    ax2.xaxis.set_major_formatter(fmt)
    plt.tight_layout()
    buf_img = io.BytesIO()
    plt.savefig(buf_img, format="png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig); buf_img.seek(0)
    return buf_img

# ========= Events (CPI/PPI monthly, FOMC manual) =========
def second_weekday(year, month, weekday):
    """Return date of the second given weekday in a month (0=Mon..6=Sun)"""
    c = calendar.Calendar(firstweekday=0)
    days = [d for d in c.itermonthdates(year, month) if d.month == month and d.weekday() == weekday]
    return days[1]

def generate_monthly_events(year=None, month=None):
    """Auto-generate CPI/PPI for the month (UTC). CPI: 2nd Tue 13:30 UTC, PPI: 2nd Wed 13:30 UTC"""
    if year is None or month is None:
        now = datetime.now(timezone.utc)
        year, month = now.year, now.month
    evs = []
    cpi_day = second_weekday(year, month, 1)  # Tue
    ppi_day = second_weekday(year, month, 2)  # Wed
    evs.append(("CPI", datetime(year, month, cpi_day.day, 13, 30, tzinfo=timezone.utc)))
    evs.append(("PPI", datetime(year, month, ppi_day.day, 13, 30, tzinfo=timezone.utc)))
    return evs

# FOMC schedule: add manually as needed (UTC)
FOMC_EVENTS = [
    # ("FOMC", datetime(2025, 9, 19, 18, 0, tzinfo=timezone.utc)),
]

def check_event_riskoff():
    """If within EVENT_EXIT_BEFORE_MIN of CPI/PPI/FOMC, force exit all open positions."""
    now_utc = datetime.now(timezone.utc)
    next_month = now_utc + timedelta(days=32)
    evs = generate_monthly_events(now_utc.year, now_utc.month) + \
          generate_monthly_events(next_month.year, next_month.month) + \
          FOMC_EVENTS
    for ev_name, ev_time in evs:
        if now_utc >= ev_time - timedelta(minutes=EVENT_EXIT_BEFORE_MIN) and now_utc < ev_time:
            # force close all open positions
            for sym in list(positions.keys()):
                pos = positions[sym]
                if not pos: continue
                df = buf_5m[sym]
                if len(df) == 0: continue
                last = df.iloc[-1]
                msg = (
                    f"⚠️ [FORCED EXIT] {sym} {pos['side'].upper()}\n"
                    f"• Price: {last['close']:.2f}\n"
                    f"• Reason: Pre-{ev_name} risk-off (5m before)\n"
                    f"• Action: Position closed"
                )
                tg_dedup_send(msg, key=f"EXIT|FORCE|{sym}|{round(last['close'],2)}|{ev_name}", ttl_s=300)
                positions[sym] = None

# ========= Status helpers =========
def now_kst():
    return datetime.now(timezone.utc).astimezone(KST).strftime("%Y-%m-%d %H:%M:%S %Z")

def format_uptime(sec: float):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    parts = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)

def build_status_text():
    now = time.time()
    age = now - LAST_LOOP_TS
    if age < 1.5: last_str = "just now (≤1s)"
    elif age < 60: last_str = f"{int(age)}s ago"
    else:
        mm, ss = divmod(int(age), 60)
        last_str = f"{mm}m {ss}s ago"
    uptime = format_uptime(now - START_TS)
    health = "OK ✅" if age < 120 else "STALE ⚠️"
    pos_lines = []
    for s, p in positions.items():
        if not p:
            pos_lines.append(f"{s}: none")
            continue
        df = buf_5m[s]
        cur = df.iloc[-1]["close"] if len(df) else p["entry"]
        pnl = pnl_pct(p["side"], p["entry"], cur)*100
        be = " BE" if p.get("be_active") else ""
        ptp = " PTP" if p.get("ptp_done") else ""
        pos_lines.append(f"{s}: {p['side']} entry {p['entry']:.2f} | cur {cur:.2f} | pnl {pnl:.2f}%{be}{ptp}")
    lines = [
        "🤖 Bot Status",
        f"- health: {health}",
        f"- uptime: {uptime}",
        f"- last tick: {last_str}",
        f"- last tick at: {LAST_LOOP_AT or 'N/A'}",
        f"- symbols: {', '.join(SYMBOLS)}",
        f"- timeframe: {ENTRY_INTERVAL} only",
        "- positions:",
        *[f"  · {ln}" for ln in pos_lines]
    ]
    return "\n".join(lines)

def tg_reply(chat_id, text):
    if not TELEGRAM_TOKEN:
        print("[TG disabled]", text); return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={"chat_id": chat_id, "text": text}, timeout=10)
    except Exception as e:
        print("TG reply error:", e)

def tg_listen_loop():
    if not TELEGRAM_TOKEN: 
        print("[TG listen disabled: no token]")
        return
    offset = None
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    while True:
        try:
            resp = requests.get(url, params={"timeout": 50, "offset": offset}, timeout=60)
            j = resp.json()
            if not j.get("ok"):
                time.sleep(2); continue
            for upd in j.get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("edited_message")
                if not msg: continue
                chat_id = msg["chat"]["id"]
                text = (msg.get("text") or "").strip()
                if not text: continue
                low = text.lower()
                if low.startswith("/ping"):
                    age = int(time.time() - LAST_LOOP_TS)
                    tg_reply(chat_id, f"pong 🏓 (last tick {age}s ago)")
                elif low.startswith("/status"):
                    tg_reply(chat_id, build_status_text())
                elif low.startswith("/help") or low.startswith("/start"):
                    tg_reply(chat_id,
                        "Commands:\n"
                        "/ping   - check heartbeat\n"
                        "/status - show health/positions\n"
                        "/help   - this help"
                    )
        except Exception as e:
            print("tg_listen_loop error:", e)
            time.sleep(3)

def start_tg_listener_thread():
    t = threading.Thread(target=tg_listen_loop, daemon=True)
    t.start()

# ========= WebSocket handlers =========
def on_message(ws, message):
    global LAST_LOOP_TS, LAST_LOOP_AT
    try:
        data = json.loads(message)
        if "data" not in data: return
        d = data["data"]
        if d.get("e") != "kline": return
        k = d["k"]
        sym = d["s"]
        interval = k["i"]
        if interval != ENTRY_INTERVAL: return

        ct_ms = k["T"]
        o,h,l,c,v = map(float, (k["o"],k["h"],k["l"],k["c"],k["v"]))
        is_final = bool(k["x"])

        LAST_LOOP_TS = time.time()
        LAST_LOOP_AT = now_kst()

        row = {"open":o,"high":h,"low":l,"close":c,"volume":v,
               "close_time": pd.to_datetime(ct_ms, unit="ms", utc=True)}

        # Update buffer
        df = buf_5m[sym]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = compute_bb_cci(df).tail(MAX_KEEP)
        buf_5m[sym] = df

        # Pre-event risk-off check (on any tick)
        check_event_riskoff()

        # EXIT (if position exists)
        pos = positions.get(sym)
        if pos:
            # cooldown against spam
            if time.time() - last_exit_at.get(sym, 0.0) >= EXIT_COOLDOWN_S:
                ex = stop_out(sym, df, pos)
                if ex:
                    last_exit_at[sym] = time.time()
                    if ex['type'] == 'sl_be':
                        msg = (
                            f"🟰 [BREAKEVEN EXIT] {sym} {pos['side'].upper()}\n"
                            f"• Price: {c:.2f}\n"
                            f"• Reason: {ex['reason']}\n"
                            f"• Action: Position closed at entry"
                        )
                        key = f"EXIT|BE|{sym}|{round(c,2)}"
                    elif ex['type'] == 'tp':
                        msg = (
                            f"💰 [FINAL TP] {sym} {pos['side'].upper()}\n"
                            f"• Price: {c:.2f}\n"
                            f"• Reason: {ex['reason']}\n"
                            f"• Action: Position closed ✅"
                        )
                        key = f"EXIT|TP|{sym}|{round(c,2)}"
                    else:  # normal stop
                        msg = (
                            f"❌ [STOP LOSS] {sym} {pos['side'].upper()}\n"
                            f"• Price: {c:.2f}\n"
                            f"• Reason: {ex['reason']}\n"
                            f"• Result: Loss realized"
                        )
                        key = f"EXIT|SL|{sym}|{round(c,2)}"
                    tg_dedup_send(msg, key=key, ttl_s=180)
                    if SEND_CHARTS_ON_EXIT:
                        png = make_chart_png(df.tail(220), sym,
                                             entry_time_utc=pos['entry_time'],
                                             exit_time_utc=row["close_time"].to_pydatetime(),
                                             entry_px=pos['entry'], exit_px=c)
                    
                        tg_photo(png, caption=f"{sym} EXIT chart")
                    positions[sym] = None
                else:
                    check_partial_tp(sym, df, pos)

        # ENTRY (5m close)
        if is_final and positions[sym] is None:
            sig = confirm_signal_close(df)
            if sig:
                positions[sym] = {
                    "side": sig["side"],
                    "entry": c,
                    "entry_time": row["close_time"],
                    "ptp_done": False,
                    "be_active": False,
                }
                key = f"CONFIRM|{sym}|{sig['side']}|{round(c,2)}|{int(ct_ms)}"
                msg = (
                    f"🚀 [ENTRY CONFIRMED] {sym} {sig['side'].upper()}\n"
                    f"• Price: {c:.2f}\n"
                    f"• Time: {datetime.now(timezone.utc).astimezone(KST).strftime('%Y-%m-%d %H:%M KST')}\n"
                    f"• Reason: {sig['reason']}"
                )
                tg_dedup_send(msg, key=key, ttl_s=300)
                if SEND_CHARTS_ON_CONFIRM:
                    png = make_chart_png(df.tail(220), sym,
                                         entry_time_utc=row["close_time"].to_pydatetime(),
                                         exit_time_utc=None,
                                         entry_px=c, exit_px=None)
                    tg_photo(png, caption=f"{sym} ENTRY chart")

    except Exception as e:
        print("on_message error:", e)

def on_open(ws): tg("5m-Only BB+CCI Scanner started (partial TP+BE, multi-position).")

def on_error(ws, error): print("ws error:", error)

def on_close(ws, code, msg): print("ws closed:", code, msg)

def run_ws():
    ws = WebSocketApp(WS_URL, on_open=on_open, on_message=on_message,
                      on_error=on_error, on_close=on_close)
    ws.run_forever(ping_interval=30, ping_timeout=10)

# ========= Main =========
if __name__ == "__main__":
    # Start Telegram command listener
    t = threading.Thread(target=tg_listen_loop, daemon=True)
    t.start()
    tg("Launching 5m-only BB+CCI (partial TP+BE, multi-position, events risk-off)…")
    run_ws()
