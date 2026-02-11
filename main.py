#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OKX USDT Perpetual (SWAP) bot
Strategy (5m):
- Universe: 24h quoteVolume TOP 50 (USDT-settled swap symbols)
- Volatility contraction: BBW (Bollinger Band Width) on closed candle <= 30th percentile of last 100 closed candles
- Trend filter: close above EMA20 => long-only; below EMA20 => short-only
- Breakout trigger (closed candle):
    Long: close > highest high of previous 20 closed candles AND volume >= 1.5 * avg(volume,20)
    Short: close < lowest low of previous 20 closed candles AND volume >= 1.5 * avg(volume,20)
- Pinbar filter: reject if (upper_wick + lower_wick) > 2 * body on breakout candle
- Risk sizing: if SL hit, lose ~3% of total USDT equity (CROSS, leverage fixed 10x)
- Execution: market entry + conditional stop-loss; TP is monitored and closed by market (reduceOnly)
- Notifications: Telegram on entry / TP / SL-detected / errors

Requirements:
  pip install ccxt pandas numpy requests

ENV:
  OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Optional:
  OKX_SANDBOX=1   (use OKX sandbox)
"""

import os
import time
import math
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import ccxt

# =========================
# Config
# =========================
TIMEFRAME = "5m"
OHLCV_LIMIT = 220  # need >= 100 bbw history + buffers
TOP_N = 50

LEVERAGE = 10.0
MARGIN_MODE = "cross"  # CROSS
RISK_PCT_EQUITY = 0.03  # lose 3% equity if stop hit
RR_TP = 1.5  # take profit at 1.5R

EMA_LEN = 20
BB_LEN = 20
BB_K = 2.0
BBW_LOOKBACK = 100
BBW_PCTL = 30  # <= 30th percentile

BOX_LEN = 20
VOL_LEN = 20
VOL_MULT = 1.5

LOOP_INTERVAL_SEC = 6
SCAN_EVERY_SEC = 30  # how often to refresh top-volume universe
STATE_PATH = "./state_okx_breakout.json"

# Safety caps (avoid oversized orders under cross)
MAX_MARGIN_FRACTION = 0.95  # don't allocate more than 95% equity as margin across a single trade
MIN_QUOTE_VOL_USDT = 1_000_000  # skip ultra-illiquid (optional)

# Telegram
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# OKX API keys
API_KEY = os.getenv("OKX_API_KEY", "")
API_SECRET = os.getenv("OKX_API_SECRET", "")
API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE", "")

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =========================
# Telegram
# =========================
def tg_send(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT_ID, "text": text}
        requests.post(url, json=payload, timeout=10)
    except Exception:
        pass


# =========================
# State
# =========================
def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {"pos": {}, "last_signal_ts": {}, "universe": [], "universe_ts": 0}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"pos": {}, "last_signal_ts": {}, "universe": [], "universe_ts": 0}


def save_state(state: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# =========================
# Exchange init
# =========================
def init_exchange() -> ccxt.okx:
    ex = ccxt.okx(
        {
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "password": API_PASSPHRASE,
            "enableRateLimit": True,
            "options": {"defaultType": "swap", "defaultSettle": "usdt"},
        }
    )
    if os.getenv("OKX_SANDBOX", "").strip() == "1":
        ex.set_sandbox_mode(True)
        logging.warning("OKX_SANDBOX=1 enabled (sandbox mode).")
    ex.load_markets()

    # one-way mode (no hedge)
    try:
        ex.set_position_mode(hedged=False)
    except Exception:
        pass

    return ex


# =========================
# Helpers
# =========================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fetch_usdt_equity(ex: ccxt.Exchange) -> float:
    bal = ex.fetch_balance()
    usdt = bal.get("USDT", {}) or {}
    total = usdt.get("total")
    if total is None:
        # fallback
        total = usdt.get("free", 0)
    return float(total or 0.0)


def contract_size(ex: ccxt.Exchange, symbol: str) -> float:
    try:
        m = ex.market(symbol)
        cs = m.get("contractSize")
        if cs:
            return float(cs)
        info = m.get("info") or {}
        if "ctVal" in info:
            return float(info["ctVal"])
    except Exception:
        pass
    return 1.0


def round_down(x: float) -> int:
    return int(math.floor(x))


def fetch_ohlcv_df(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not o:
        return None
    df = pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("dt", inplace=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # EMA
    df["ema20"] = close.ewm(span=EMA_LEN, adjust=False).mean()

    # BB + BBW
    mid = close.rolling(BB_LEN).mean()
    std = close.rolling(BB_LEN).std(ddof=0)
    upper = mid + BB_K * std
    lower = mid - BB_K * std
    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bbw"] = (upper - lower) / mid.replace(0, np.nan)

    return df


def pinbar_reject(candle: pd.Series) -> bool:
    o = float(candle["open"])
    h = float(candle["high"])
    l = float(candle["low"])
    c = float(candle["close"])
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    # Reject if very wick-y vs body
    return (upper_wick + lower_wick) > (2.0 * max(body, 1e-12))


def compute_signal(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Use CLOSED candle only: df.iloc[-2]
    Need enough history for BBW lookback and box/volume windows.
    """
    if df is None or len(df) < max(BBW_LOOKBACK + BB_LEN + 5, BOX_LEN + 5, VOL_LEN + 5):
        return None

    closed = df.iloc[-2]  # signal candle
    # reference window is also closed-only, so use up to -2 inclusive
    hist = df.iloc[:-1]  # exclude the currently forming candle

    # Volatility contraction (BBW <= 30th percentile of last 100 closed)
    bbw_hist = hist["bbw"].tail(BBW_LOOKBACK).dropna().values
    if len(bbw_hist) < int(BBW_LOOKBACK * 0.8):
        return None
    bbw_th = np.nanpercentile(bbw_hist, BBW_PCTL)
    if float(closed["bbw"]) > float(bbw_th):
        return None

    # Trend filter
    close_price = float(closed["close"])
    ema20 = float(closed["ema20"])
    if not (close_price > 0 and ema20 > 0):
        return None
    trend = "long" if close_price >= ema20 else "short"

    # Breakout box (previous 20 closed candles, excluding the signal candle itself)
    prev20 = hist.iloc[-(BOX_LEN + 1):-1]  # 20 candles before closed
    if len(prev20) < BOX_LEN:
        return None
    box_high = float(prev20["high"].max())
    box_low = float(prev20["low"].min())

    # Volume filter (avg volume of previous 20 closed, excluding signal candle)
    vprev = hist.iloc[-(VOL_LEN + 1):-1]
    if len(vprev) < VOL_LEN:
        return None
    v_avg = float(vprev["volume"].mean())
    v_now = float(closed["volume"])
    if v_avg <= 0 or v_now < (VOL_MULT * v_avg):
        return None

    # Pinbar reject on breakout candle
    if pinbar_reject(closed):
        return None

    # Trigger
    side = None
    if trend == "long" and close_price > box_high:
        side = "long"
        stop_price = box_low
    elif trend == "short" and close_price < box_low:
        side = "short"
        stop_price = box_high
    else:
        return None

    if stop_price <= 0:
        return None
    if side == "long" and stop_price >= close_price:
        return None
    if side == "short" and stop_price <= close_price:
        return None

    # TP by R multiple
    r = abs(close_price - stop_price)
    tp_price = close_price + (RR_TP * r) if side == "long" else close_price - (RR_TP * r)

    return {
        "side": side,
        "entry_price": close_price,         # reference price (close of closed candle)
        "stop_price": float(stop_price),
        "tp_price": float(tp_price),
        "signal_ts": int(closed["ts"]),
        "box_high": box_high,
        "box_low": box_low,
        "bbw": float(closed["bbw"]),
        "bbw_th": float(bbw_th),
        "vol": v_now,
        "vol_avg": v_avg,
        "ema20": ema20,
    }


def get_universe_usdt_swaps_top_volume(ex: ccxt.Exchange) -> List[str]:
    """
    Build USDT-settled swap universe, then pick top-N by quoteVolume (24h).
    """
    # Collect candidate symbols
    syms = []
    for s, m in ex.markets.items():
        try:
            if not m.get("swap"):
                continue
            if (m.get("settle") or "").lower() != "usdt":
                continue
            # ccxt okx uses symbols like "BTC/USDT:USDT"
            if not s.endswith(":USDT"):
                continue
            if "/USDT" not in s:
                continue
            if not m.get("active", True):
                continue
            syms.append(s)
        except Exception:
            continue

    if not syms:
        return []

    # Fetch tickers for candidates (may be heavy, but OKX is usually fine; enableRateLimit helps)
    # If fetch_tickers fails, fallback to sequential fetch_ticker for a smaller subset.
    tickers = {}
    try:
        tickers = ex.fetch_tickers(syms)
    except Exception:
        tickers = {}
        # fallback: sample first 300 to avoid rate-limit explosion
        for s in syms[:300]:
            try:
                tickers[s] = ex.fetch_ticker(s)
                time.sleep(0.03)
            except Exception:
                continue

    scored = []
    for s, t in (tickers or {}).items():
        try:
            qv = t.get("quoteVolume")
            if qv is None:
                # OKX info fallback
                info = t.get("info") or {}
                # on OKX, quote volume can appear as volCcy24h
                qv = info.get("volCcy24h") or info.get("quoteVol") or info.get("volCcy")
            qv = float(qv or 0.0)
            if qv < MIN_QUOTE_VOL_USDT:
                continue
            scored.append((s, qv))
        except Exception:
            continue

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:TOP_N]]


def sync_positions(ex: ccxt.Exchange, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    out = {s: {"has": False, "side": None, "contracts": 0.0, "entry": None} for s in symbols}
    try:
        poss = ex.fetch_positions()
    except Exception:
        return out

    for p in poss:
        sym = p.get("symbol")
        if sym not in out:
            continue
        contracts = float(p.get("contracts") or p.get("positionAmt") or 0.0)
        if contracts == 0.0:
            continue
        side = p.get("side")
        if not side:
            side = "long" if contracts > 0 else "short"
        entry = float(p.get("entryPrice") or p.get("avgPrice") or 0.0) or None
        out[sym] = {"has": True, "side": side, "contracts": abs(contracts), "entry": entry}
    return out


def calc_contracts_for_risk(
    ex: ccxt.Exchange,
    symbol: str,
    equity_usdt: float,
    entry_price: float,
    stop_price: float,
) -> Tuple[int, float, float]:
    """
    Returns (contracts, notional_usdt, margin_usdt_est)
    risk = equity * 0.03
    stop_pct = |entry-stop|/entry
    notional = risk / stop_pct
    margin_est = notional / leverage
    """
    if equity_usdt <= 0 or entry_price <= 0 or stop_price <= 0:
        return 0, 0.0, 0.0

    stop_pct = abs(entry_price - stop_price) / entry_price
    if stop_pct <= 0:
        return 0, 0.0, 0.0

    risk_usdt = equity_usdt * RISK_PCT_EQUITY
    target_notional = risk_usdt / stop_pct

    # cap by available margin fraction (cross)
    max_notional = equity_usdt * MAX_MARGIN_FRACTION * LEVERAGE
    notional = min(target_notional, max_notional)

    cs = contract_size(ex, symbol)
    notional_per_contract = entry_price * cs
    if notional_per_contract <= 0:
        return 0, 0.0, 0.0

    contracts = round_down(notional / notional_per_contract)
    if contracts <= 0:
        return 0, 0.0, 0.0

    actual_notional = contracts * notional_per_contract
    margin_est = actual_notional / LEVERAGE
    return contracts, actual_notional, margin_est


def place_entry_with_stop(
    ex: ccxt.Exchange,
    symbol: str,
    side: str,
    contracts: int,
    stop_price: float,
) -> Dict[str, Any]:
    """
    Market entry + conditional stop-loss (reduceOnly)
    Returns a dict with entry_order_id, stop_order_id (if created)
    """
    assert side in ("long", "short")
    order_side = "buy" if side == "long" else "sell"
    sl_side = "sell" if side == "long" else "buy"

    # set leverage fixed 10, cross
    try:
        ex.set_leverage(LEVERAGE, symbol, params={"mgnMode": MARGIN_MODE})
    except Exception:
        pass

    entry_order = ex.create_order(symbol, "market", order_side, contracts, params={"tdMode": MARGIN_MODE})

    # short delay then place stop-loss
    time.sleep(0.25)

    stop_order_id = None
    try:
        sl_order = ex.create_order(
            symbol,
            "market",
            sl_side,
            contracts,
            params={
                "tdMode": MARGIN_MODE,
                "reduceOnly": True,
                "stopLossPrice": float(stop_price),
            },
        )
        stop_order_id = sl_order.get("id")
    except Exception:
        stop_order_id = None

    return {"entry_order_id": entry_order.get("id"), "stop_order_id": stop_order_id}


def close_position_market(ex: ccxt.Exchange, symbol: str, side: str, contracts: float) -> None:
    if contracts <= 0:
        return
    close_side = "sell" if side == "long" else "buy"
    ex.create_order(symbol, "market", close_side, contracts, params={"tdMode": MARGIN_MODE, "reduceOnly": True})


# =========================
# Main
# =========================
def main():
    ex = init_exchange()
    state = load_state()

    tg_send("OKX 5m VolContraction Breakout bot started.")

    universe: List[str] = state.get("universe") or []
    universe_ts: float = float(state.get("universe_ts") or 0)

    # pos state per symbol:
    # {
    #   symbol: { side, entry, stop, tp, stop_order_id, entry_time_iso }
    # }
    pos_state: Dict[str, Dict[str, Any]] = state.get("pos") or {}
    last_signal_ts: Dict[str, int] = state.get("last_signal_ts") or {}

    last_universe_refresh = 0.0

    while True:
        try:
            now = time.time()

            # refresh universe periodically
            if (not universe) or (now - universe_ts >= SCAN_EVERY_SEC) or (now - last_universe_refresh >= SCAN_EVERY_SEC):
                universe = get_universe_usdt_swaps_top_volume(ex)
                universe_ts = now
                last_universe_refresh = now
                logging.info(f"Universe refreshed: {len(universe)} symbols (top {TOP_N} by quoteVolume).")

            if not universe:
                time.sleep(LOOP_INTERVAL_SEC)
                continue

            # sync exchange positions for universe
            exch_pos = sync_positions(ex, universe)

            # detect positions disappeared (likely SL/manual) and clean local state
            for sym in list(pos_state.keys()):
                if sym not in exch_pos:
                    continue
                if pos_state.get(sym) and not exch_pos[sym]["has"]:
                    # position gone
                    s = pos_state[sym]
                    msg = f"[EXIT DETECTED] {sym} side={s.get('side')} (position is now 0). Possible SL/manual."
                    logging.info(msg)
                    tg_send(msg)
                    pos_state.pop(sym, None)

            # manage open positions: TP monitoring + SL detection via position disappearance
            for sym in universe:
                if sym not in pos_state:
                    continue
                p = pos_state[sym]
                if not exch_pos[sym]["has"]:
                    continue

                side = p["side"]
                tp = float(p["tp"])
                # use ticker last price
                try:
                    t = ex.fetch_ticker(sym)
                    last = float(t.get("last") or 0.0)
                except Exception:
                    last = 0.0
                if last <= 0:
                    continue

                hit_tp = (side == "long" and last >= tp) or (side == "short" and last <= tp)
                if hit_tp:
                    # cancel stop order if possible
                    soid = p.get("stop_order_id")
                    if soid:
                        try:
                            ex.cancel_order(soid, sym)
                        except Exception:
                            pass

                    # close by market reduceOnly
                    close_position_market(ex, sym, side, float(exch_pos[sym]["contracts"]))
                    msg = f"[TP] {sym} {side.upper()} hit TP. last={last:.6g} tp={tp:.6g}"
                    logging.info(msg)
                    tg_send(msg)
                    pos_state.pop(sym, None)

            # entry scan: only for symbols with no position
            equity = fetch_usdt_equity(ex)
            if equity <= 0:
                time.sleep(LOOP_INTERVAL_SEC)
                continue

            for sym in universe:
                if exch_pos[sym]["has"]:
                    continue
                if sym in pos_state:
                    continue

                # fetch chart + compute signal
                df = fetch_ohlcv_df(ex, sym, TIMEFRAME, OHLCV_LIMIT)
                if df is None:
                    continue
                df = add_indicators(df)

                sig = compute_signal(df)
                if not sig:
                    continue

                sig_ts = int(sig["signal_ts"])
                if last_signal_ts.get(sym) == sig_ts:
                    continue

                side = sig["side"]
                entry_ref = float(sig["entry_price"])
                stop = float(sig["stop_price"])
                tp = float(sig["tp_price"])

                # size by risk
                contracts, notional, margin_est = calc_contracts_for_risk(ex, sym, equity, entry_ref, stop)
                if contracts <= 0:
                    last_signal_ts[sym] = sig_ts
                    continue

                # place orders
                orders = place_entry_with_stop(ex, sym, side, contracts, stop)

                # sync actual entry after placing
                time.sleep(0.35)
                pnow = sync_positions(ex, [sym]).get(sym, {})
                actual_entry = pnow.get("entry") or entry_ref

                pos_state[sym] = {
                    "side": side,
                    "entry": float(actual_entry),
                    "stop": stop,
                    "tp": tp,
                    "stop_order_id": orders.get("stop_order_id"),
                    "entry_time": now_utc().isoformat(),
                    "contracts": float(contracts),
                    "notional_est": float(notional),
                    "margin_est": float(margin_est),
                    "meta": {
                        "box_high": float(sig["box_high"]),
                        "box_low": float(sig["box_low"]),
                        "bbw": float(sig["bbw"]),
                        "bbw_th": float(sig["bbw_th"]),
                        "vol": float(sig["vol"]),
                        "vol_avg": float(sig["vol_avg"]),
                        "ema20": float(sig["ema20"]),
                    },
                }
                last_signal_ts[sym] = sig_ts

                msg = (
                    f"[ENTRY] {sym} {side.upper()} lev={LEVERAGE:.0f}x CROSS\n"
                    f"entry≈{float(actual_entry):.6g} stop={stop:.6g} tp={tp:.6g} (RR={RR_TP})\n"
                    f"contracts={contracts} notional≈{notional:.2f} margin≈{margin_est:.2f}\n"
                    f"BBW={sig['bbw']:.6g} (<= p{BBW_PCTL}:{sig['bbw_th']:.6g}), "
                    f"Vol={sig['vol']:.3g} (>= {VOL_MULT}x avg:{sig['vol_avg']:.3g})"
                )
                logging.info(msg)
                tg_send(msg)

                # small pause to be gentle to OKX rate limits after an entry
                time.sleep(0.5)

            # persist
            state["pos"] = pos_state
            state["last_signal_ts"] = last_signal_ts
            state["universe"] = universe
            state["universe_ts"] = universe_ts
            state["timestamp"] = now_utc().isoformat()
            save_state(state)

            time.sleep(LOOP_INTERVAL_SEC)

        except ccxt.BaseError as e:
            logging.warning(f"CCXT error: {e}")
            tg_send(f"[ERROR] CCXT: {e}")
            time.sleep(LOOP_INTERVAL_SEC)
        except Exception as e:
            logging.warning(f"Loop error: {e}")
            tg_send(f"[ERROR] Loop: {e}")
            time.sleep(LOOP_INTERVAL_SEC)


if __name__ == "__main__":
    main()
