#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OKX USDT Perpetual (SWAP) bot
+ Feature logging for XGB (JSONL)
Label: 1 = TP hit, 0 = SL or manual/other exit

Updated behavior:
- Entry: market
- SL: exchange-side conditional (reduceOnly)
- TP: exchange-side conditional (reduceOnly)
- Bot no longer “monitors TP and closes by market”.
  Instead, it detects position disappearance and classifies exit via:
  1) order status (TP/SL order) if available
  2) fallback to last price vs tp/stop (best-effort)
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
TOP_N = 150

LEVERAGE = 10.0
MARGIN_MODE = "cross"  # CROSS
RISK_PCT_EQUITY = 0.04  # lose ~4% equity if stop hit
RR_TP = 1.5  # take profit at 1.5R

EMA_LEN = 20
BB_LEN = 20
BB_K = 2.0
BBW_LOOKBACK = 100
BBW_PCTL = 30  # <= 30th percentile

BOX_LEN = 20
VOL_LEN = 20
VOL_MULT = 1.5

LOOP_INTERVAL_SEC = 3
SCAN_EVERY_SEC = 30  # how often to refresh top-volume universe
STATE_PATH = "/app/state/state_okx_breakout.json"
FEATURES_PATH = "/app/state/features_xgb.jsonl"  # <-- JSONL output

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

    # ATR (14, 100)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr100"] = tr.rolling(100).mean()

    return df


def pinbar_reject(candle: pd.Series) -> bool:
    o = float(candle["open"])
    h = float(candle["high"])
    l = float(candle["low"])
    c = float(candle["close"])
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return (upper_wick + lower_wick) > (2.0 * max(body, 1e-12))


def compute_signal(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if df is None or len(df) < max(BBW_LOOKBACK + BB_LEN + 25, BOX_LEN + 5, VOL_LEN + 5):
        return None

    closed = df.iloc[-2]
    hist = df.iloc[:-1]

    # 1) Volatility contraction
    bbw_hist = hist["bbw"].tail(BBW_LOOKBACK).dropna().values
    if len(bbw_hist) < int(BBW_LOOKBACK * 0.8):
        return None

    bbw_th = np.nanpercentile(bbw_hist, BBW_PCTL)
    if float(closed["bbw"]) > float(bbw_th):
        return None

    # 2) Trend
    close_price = float(closed["close"])
    ema20 = float(closed["ema20"])
    if not (close_price > 0 and ema20 > 0):
        return None

    trend = "long" if close_price >= ema20 else "short"

    # 3) Box
    prev20 = hist.iloc[-(BOX_LEN + 1) : -1]
    if len(prev20) < BOX_LEN:
        return None

    box_high = float(prev20["high"].max())
    box_low = float(prev20["low"].min())
    box_range_pct = (box_high - box_low) / close_price

    # allow only mid box regime (30~70 percentile)
    box_hist = (
        (hist["high"].rolling(BOX_LEN).max() - hist["low"].rolling(BOX_LEN).min()) / hist["close"]
    ).dropna().tail(100)

    if len(box_hist) > 20:
        q_low = np.nanpercentile(box_hist, 30)
        q_high = np.nanpercentile(box_hist, 70)
        if not (q_low <= box_range_pct <= q_high):
            return None

    # 4) Volume
    vprev = hist.iloc[-(VOL_LEN + 1) : -1]
    if len(vprev) < VOL_LEN:
        return None

    v_avg = float(vprev["volume"].mean())
    v_now = float(closed["volume"])
    if v_avg <= 0:
        return None

    vol_ratio = v_now / v_avg
    if not (2.0 <= vol_ratio <= 3.5):
        return None

    # 5) Breakout
    side = None
    if trend == "long" and close_price > box_high:
        side = "long"
        stop_price = box_low
    elif trend == "short" and close_price < box_low:
        side = "short"
        stop_price = box_high
    else:
        return None

    # 6) Momentum regime filter
    def ret_n(n):
        if len(hist) >= n + 1:
            p0 = hist["close"].iloc[-1]
            pN = hist["close"].iloc[-(n + 1)]
            if pN > 0:
                return (p0 - pN) / pN
        return 0.0

    ret_5 = ret_n(5)
    ret_20 = ret_n(20)

    ema20_prev = hist["ema20"].iloc[-6] if len(hist) >= 6 else ema20
    ema20_slope_pct = (ema20 - ema20_prev) / ema20_prev if ema20_prev > 0 else 0.0

    if side == "long":
        if not (ret_5 > 0 and ret_20 > 0 and ema20_slope_pct > 0):
            return None
    else:
        if not (ret_5 < 0 and ret_20 < 0 and ema20_slope_pct < 0):
            return None

    # 7) Pinbar reject
    if pinbar_reject(closed):
        return None

    if stop_price <= 0:
        return None
    if side == "long" and stop_price >= close_price:
        return None
    if side == "short" and stop_price <= close_price:
        return None

    r = abs(close_price - stop_price)
    tp_price = close_price + (RR_TP * r) if side == "long" else close_price - (RR_TP * r)

    return {
        "side": side,
        "entry_price": close_price,
        "stop_price": float(stop_price),
        "tp_price": float(tp_price),
        "signal_ts": int(closed["ts"]),
        "box_high": box_high,
        "box_low": box_low,
        "bbw": float(closed["bbw"]),
        "bbw_th": float(bbw_th),
        "vol": v_now,
        "vol_avg": v_avg,
        "vol_ratio": vol_ratio,
        "ema20": ema20,
        "ret_5": ret_5,
        "ret_20": ret_20,
        "ema20_slope_pct": ema20_slope_pct,
    }


def get_universe_usdt_swaps_top_volume(ex: ccxt.Exchange) -> List[str]:
    syms = []
    for s, m in ex.markets.items():
        try:
            if not m.get("swap"):
                continue
            if (m.get("settle") or "").lower() != "usdt":
                continue
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

    try:
        tickers = ex.fetch_tickers(syms)
    except Exception:
        tickers = {}
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
                info = t.get("info") or {}
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
    if equity_usdt <= 0 or entry_price <= 0 or stop_price <= 0:
        return 0, 0.0, 0.0

    stop_pct = abs(entry_price - stop_price) / entry_price
    if stop_pct <= 0:
        return 0, 0.0, 0.0

    risk_usdt = equity_usdt * RISK_PCT_EQUITY
    target_notional = risk_usdt / stop_pct

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


# =========================
# Orders: Entry + SL + TP
# =========================
def _place_reduceonly_trigger_order_best_effort(
    ex: ccxt.Exchange,
    symbol: str,
    side: str,
    contracts: int,
    trigger_price: float,
    kind: str,  # "sl" or "tp"
) -> Optional[str]:
    """
    Best-effort to place exchange-side conditional reduceOnly order.

    Primary attempt uses ccxt unified params:
      - stopLossPrice / takeProfitPrice inside a market order payload
    Fallback: try OKX algo endpoint if available.

    Returns order/algo id if known, else None.
    """
    assert kind in ("sl", "tp")
    assert side in ("long", "short")

    # To CLOSE a long -> SELL ; to CLOSE a short -> BUY
    close_side = "sell" if side == "long" else "buy"

    # 1) Unified attempt
    params = {"tdMode": MARGIN_MODE, "reduceOnly": True}
    if kind == "sl":
        params["stopLossPrice"] = float(trigger_price)
    else:
        params["takeProfitPrice"] = float(trigger_price)

    try:
        o = ex.create_order(symbol, "market", close_side, contracts, params=params)
        oid = (o or {}).get("id")
        return oid
    except Exception:
        pass

    # 2) OKX algo endpoint fallback (ccxt exposes raw endpoints on okx instance)
    #    We'll try a minimal order-algo placement.
    try:
        m = ex.market(symbol)
        inst_id = (m.get("info") or {}).get("instId") or m.get("id") or symbol

        # OKX expects close order side:
        okx_side = "sell" if side == "long" else "buy"
        # ordType: "conditional"
        # triggerPx: trigger price
        # orderPx: "-1" market
        body = {
            "instId": inst_id,
            "tdMode": MARGIN_MODE,
            "side": okx_side,
            "ordType": "conditional",
            "sz": str(int(contracts)),
            "reduceOnly": "true",
            "triggerPx": str(float(trigger_price)),
            "orderPx": "-1",
        }

        # OKX distinguishes TP/SL with "tpTriggerPx" / "slTriggerPx" in some modes.
        # We'll add hint fields to improve compatibility:
        if kind == "sl":
            body["slTriggerPx"] = body["triggerPx"]
            body["slOrdPx"] = "-1"
        else:
            body["tpTriggerPx"] = body["triggerPx"]
            body["tpOrdPx"] = "-1"

        resp = ex.privatePostTradeOrderAlgo(body)
        # Response shape varies; try to find algoId
        if isinstance(resp, dict):
            data = resp.get("data") or []
            if isinstance(data, list) and data:
                algo_id = data[0].get("algoId") or data[0].get("id")
                if algo_id:
                    return str(algo_id)
        return None
    except Exception:
        return None


def place_entry_with_sl_tp(
    ex: ccxt.Exchange,
    symbol: str,
    side: str,
    contracts: int,
    stop_price: float,
    tp_price: float,
) -> Dict[str, Any]:
    """
    Entry: market
    Then place:
      - SL reduceOnly trigger order
      - TP reduceOnly trigger order
    """
    assert side in ("long", "short")
    order_side = "buy" if side == "long" else "sell"

    try:
        ex.set_leverage(LEVERAGE, symbol, params={"mgnMode": MARGIN_MODE})
    except Exception:
        pass

    entry_order = ex.create_order(symbol, "market", order_side, contracts, params={"tdMode": MARGIN_MODE})
    time.sleep(0.25)

    sl_id = _place_reduceonly_trigger_order_best_effort(ex, symbol, side, contracts, stop_price, kind="sl")
    time.sleep(0.15)
    tp_id = _place_reduceonly_trigger_order_best_effort(ex, symbol, side, contracts, tp_price, kind="tp")

    return {
        "entry_order_id": (entry_order or {}).get("id"),
        "stop_order_id": sl_id,
        "tp_order_id": tp_id,
    }


def safe_cancel_order(ex: ccxt.Exchange, order_id: Optional[str], symbol: str) -> None:
    if not order_id:
        return
    try:
        ex.cancel_order(order_id, symbol)
    except Exception:
        # could be algo order or already filled/canceled
        pass


def fetch_last_price(ex: ccxt.Exchange, symbol: str) -> float:
    try:
        t = ex.fetch_ticker(symbol)
        return float(t.get("last") or 0.0)
    except Exception:
        return 0.0


def try_fetch_order_status(ex: ccxt.Exchange, order_id: Optional[str], symbol: str) -> Optional[str]:
    """
    Try to get order status by id.
    Returns: "closed" / "open" / "canceled" / None (unknown)
    """
    if not order_id:
        return None
    try:
        o = ex.fetch_order(order_id, symbol)
        st = (o or {}).get("status")
        if st:
            return str(st)
    except Exception:
        return None
    return None


def classify_exit_label_best_effort(
    ex: ccxt.Exchange,
    symbol: str,
    pos_rec: Dict[str, Any],
) -> int:
    """
    When position disappeared, decide label:
    - Prefer order status: if TP order is closed => 1; if SL order closed => 0
    - Else fallback using last price relative to tp/stop (best-effort)
    - Default to 0
    """
    tp_id = pos_rec.get("tp_order_id")
    sl_id = pos_rec.get("stop_order_id")
    tp = float(pos_rec.get("tp") or 0.0)
    sl = float(pos_rec.get("stop") or 0.0)
    side = pos_rec.get("side")

    tp_status = try_fetch_order_status(ex, tp_id, symbol)
    sl_status = try_fetch_order_status(ex, sl_id, symbol)

    if tp_status == "closed":
        return 1
    if sl_status == "closed":
        return 0

    # If both unknown or not closed: fallback
    last = fetch_last_price(ex, symbol)
    if last > 0 and tp > 0 and sl > 0 and side in ("long", "short"):
        if side == "long":
            if last >= tp:
                return 1
            if last <= sl:
                return 0
        else:
            if last <= tp:
                return 1
            if last >= sl:
                return 0

    return 0


# =========================
# Feature engineering (XGB)
# =========================
def compute_features(df: pd.DataFrame, sig: Dict[str, Any]) -> Dict[str, float]:
    """
    Build 20 features (scale-free)
    Uses CLOSED candle (df.iloc[-2])
    """
    c = df.iloc[-2]
    hist = df.iloc[:-1]

    entry = float(sig["entry_price"])
    stop = float(sig["stop_price"])
    tp = float(sig["tp_price"])
    box_high = float(sig["box_high"])
    box_low = float(sig["box_low"])

    side = 1.0 if sig["side"] == "long" else 0.0

    rr = abs(tp - entry) / max(abs(entry - stop), 1e-12)
    stop_dist_pct = abs(entry - stop) / entry
    tp_dist_pct = abs(tp - entry) / entry
    box_range_pct = (box_high - box_low) / entry

    if sig["side"] == "long":
        breakout_pct = max((entry - box_high) / entry, 0.0)
    else:
        breakout_pct = max((box_low - entry) / entry, 0.0)

    bbw = float(c["bbw"])
    bbw_hist = hist["bbw"].tail(BBW_LOOKBACK).dropna().values
    bbw_percentile = float((bbw_hist <= bbw).mean()) if len(bbw_hist) > 0 else 0.0

    atr14 = float(c.get("atr14") or np.nan)
    atr100 = float(c.get("atr100") or np.nan)
    atr14_pct = (atr14 / entry) if entry > 0 and not np.isnan(atr14) else 0.0
    atr_ratio_14_100 = (atr14 / atr100) if (atr100 and not np.isnan(atr14) and not np.isnan(atr100)) else 0.0

    ema20 = float(c["ema20"])
    ema20_dist_pct = (entry - ema20) / ema20 if ema20 > 0 else 0.0
    ema20_n = hist["ema20"].iloc[-6] if len(hist) >= 6 else ema20
    ema20_slope_pct = (ema20 - ema20_n) / ema20_n if ema20_n > 0 else 0.0

    def ret_n(n):
        if len(hist) >= n + 1:
            p0 = hist["close"].iloc[-1]
            pN = hist["close"].iloc[-(n + 1)]
            if pN > 0:
                return (p0 - pN) / pN
        return 0.0

    ret_1 = ret_n(1)
    ret_5 = ret_n(5)
    ret_20 = ret_n(20)

    vol_now = float(c["volume"])
    vprev = hist["volume"].tail(VOL_LEN)
    vol_avg_20 = float(vprev.mean()) if len(vprev) > 0 else 0.0
    vol_std_20 = float(vprev.std(ddof=0)) if len(vprev) > 1 else 0.0
    vol_ratio = (vol_now / vol_avg_20) if vol_avg_20 > 0 else 0.0
    vol_z_20 = ((vol_now - vol_avg_20) / vol_std_20) if vol_std_20 > 0 else 0.0

    o = float(c["open"])
    h = float(c["high"])
    l = float(c["low"])
    cl = float(c["close"])
    body = abs(cl - o)
    rng = max(h - l, 1e-12)
    upper_wick = h - max(o, cl)
    lower_wick = min(o, cl) - l
    body_ratio = body / rng
    wick_ratio = (upper_wick + lower_wick) / max(body, 1e-12)
    range_pct = rng / entry if entry > 0 else 0.0

    return {
        "side": side,
        "rr": rr,
        "stop_dist_pct": stop_dist_pct,
        "tp_dist_pct": tp_dist_pct,
        "box_range_pct": box_range_pct,
        "breakout_pct": breakout_pct,
        "bbw": bbw,
        "bbw_percentile_100": bbw_percentile,
        "atr14_pct": atr14_pct,
        "atr_ratio_14_100": atr_ratio_14_100,
        "ema20_dist_pct": ema20_dist_pct,
        "ema20_slope_pct": ema20_slope_pct,
        "ret_1": ret_1,
        "ret_5": ret_5,
        "ret_20": ret_20,
        "vol_ratio": vol_ratio,
        "vol_z_20": vol_z_20,
        "body_ratio": body_ratio,
        "wick_ratio": wick_ratio,
        "range_pct": range_pct,
    }


def append_feature_record(features: Dict[str, float], label: int, meta: Dict[str, Any]) -> None:
    rec = {}
    rec.update(features)
    rec["label"] = int(label)
    rec["symbol"] = meta.get("symbol")
    rec["side_str"] = meta.get("side")
    rec["entry_time"] = meta.get("entry_time")
    rec["exit_time"] = meta.get("exit_time")
    try:
        os.makedirs(os.path.dirname(FEATURES_PATH) or ".", exist_ok=True)
        with open(FEATURES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


# =========================
# Main
# =========================
def main():
    ex = init_exchange()
    state = load_state()

    tg_send("OKX 5m VolContraction Breakout bot started. (TP+SL on exchange)")

    universe: List[str] = state.get("universe") or []
    universe_ts: float = float(state.get("universe_ts") or 0)

    pos_state: Dict[str, Dict[str, Any]] = state.get("pos") or {}
    last_signal_ts: Dict[str, int] = state.get("last_signal_ts") or {}

    last_universe_refresh = 0.0

    while True:
        try:
            now = time.time()

            # refresh universe
            if (not universe) or (now - universe_ts >= SCAN_EVERY_SEC) or (now - last_universe_refresh >= SCAN_EVERY_SEC):
                universe = get_universe_usdt_swaps_top_volume(ex)
                universe_ts = now
                last_universe_refresh = now

            if not universe:
                time.sleep(LOOP_INTERVAL_SEC)
                continue

            exch_pos = sync_positions(ex, universe)

            # detect positions disappeared => classify TP/SL/manual and log
            for sym in list(pos_state.keys()):
                if sym not in exch_pos:
                    continue
                if pos_state.get(sym) and not exch_pos[sym]["has"]:
                    s = pos_state[sym]

                    label = classify_exit_label_best_effort(ex, sym, s)

                    feats = s.get("features")
                    if feats:
                        append_feature_record(
                            feats,
                            label=label,
                            meta={
                                "symbol": sym,
                                "side": s.get("side"),
                                "entry_time": s.get("entry_time"),
                                "exit_time": now_utc().isoformat(),
                            },
                        )

                    # cancel the other protection order best-effort (might already be filled/canceled)
                    safe_cancel_order(ex, s.get("tp_order_id"), sym)
                    safe_cancel_order(ex, s.get("stop_order_id"), sym)

                    tag = "TP" if label == 1 else "SL/MANUAL"
                    msg = f"[EXIT DETECTED] {sym} side={s.get('side')} => label={label} ({tag})"
                    logging.info(msg)
                    tg_send(msg)
                    pos_state.pop(sym, None)

            # entry scan
            equity = fetch_usdt_equity(ex)
            if equity <= 0:
                time.sleep(LOOP_INTERVAL_SEC)
                continue

            for sym in universe:
                if exch_pos[sym]["has"]:
                    continue
                if sym in pos_state:
                    continue

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

                contracts, notional, margin_est = calc_contracts_for_risk(ex, sym, equity, entry_ref, stop)
                if contracts <= 0:
                    last_signal_ts[sym] = sig_ts
                    continue

                orders = place_entry_with_sl_tp(ex, sym, side, contracts, stop, tp)

                time.sleep(0.35)
                pnow = sync_positions(ex, [sym]).get(sym, {})
                actual_entry = pnow.get("entry") or entry_ref

                # compute and store features at entry
                features = compute_features(df, sig)

                pos_state[sym] = {
                    "side": side,
                    "entry": float(actual_entry),
                    "stop": stop,
                    "tp": tp,
                    "stop_order_id": orders.get("stop_order_id"),
                    "tp_order_id": orders.get("tp_order_id"),
                    "entry_order_id": orders.get("entry_order_id"),
                    "entry_time": now_utc().isoformat(),
                    "contracts": float(contracts),
                    "notional_est": float(notional),
                    "margin_est": float(margin_est),
                    "features": features,
                }
                last_signal_ts[sym] = sig_ts

                msg = (
                    f"[ENTRY] {sym} {side.upper()} lev={LEVERAGE:.0f}x CROSS\n"
                    f"entry≈{float(actual_entry):.6g} stop={stop:.6g} tp={tp:.6g} (RR={RR_TP})\n"
                    f"contracts={contracts} notional≈{notional:.2f} margin≈{margin_est:.2f}\n"
                    f"SL_id={orders.get('stop_order_id')} TP_id={orders.get('tp_order_id')}"
                )
                logging.info(msg)
                tg_send(msg)

                time.sleep(0.5)

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
