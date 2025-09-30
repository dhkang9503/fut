
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triangle Breakout Detector — Multi (ETH/XRP/SOL 5m) + Retest + Leverage + Fake-breakout filters
-----------------------------------------------------------------------------------------------
Adds *fake breakout* filters before confirming a breakout:
- Require N confirming bars to also close beyond the trendline (same direction)
- Minimum breakout candle body relative to ATR
- Minimum body-to-range ratio to avoid long-wick spikes

Only after confirmation do we send the breakout alert and arm the *retest* watcher.
"""
import os
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp  # pip install aiohttp
import websockets  # pip install websockets
import numpy as np  # pip install numpy

REST_BASE = "https://fapi.binance.com"
WS_BASE = "wss://fstream.binance.com/stream"
KLINES_ENDPOINT = "/fapi/v1/klines"

SYMBOLS = ["ethusdt", "xrpusdt", "solusdt"]

# --------------------------- Config --------------------------- #

@dataclass
class Config:
    interval: str = "5m"
    seed_candles: int = 600
    ws_reconnect_delay: int = 5

    # Triangle detection params
    pivot_span: int = 3
    max_pivots: int = 30
    min_pivots_required: int = 4
    lookback_bars: int = 200

    # Triangle validation thresholds
    min_contraction_ratio: float = 0.35
    atr_window: int = 14
    atr_contraction_factor: float = 0.7

    # Breakout rules
    breakout_buffer_pct: float = 0.001
    vol_ma_window: int = 20
    vol_confirm_factor: float = 1.5

    # Fake-breakout filters
    confirm_bars: int = 1               # require this many *subsequent* bars also closing beyond the line
    min_body_atr: float = 0.25          # breakout candle body >= 0.25 * ATR
    min_body_to_range: float = 0.40     # breakout candle body / range >= 0.40 (limits long wicks)

    # Retest entry rules
    retest_max_bars: int = 10
    retest_tolerance_pct: float = 0.002
    retest_confirm_close: bool = True
    stop_buffer_pct: float = 0.002
    tp_rr_list: Tuple[float, ...] = (1.0, 2.0)
    use_atr_targets: bool = False
    atr_tp_multipliers: Tuple[float, ...] = (1.0, 1.5, 2.0)

    # Direction filter
    use_direction_filter: bool = True
    ema_period: int = 200
    htf_interval: str = "15m"
    htf_ema_period: int = 200
    htf_slope_lookback: int = 10

    # Leverage targeting
    loss_pct_at_sl: float = 0.10
    leverage_cap: float = 50.0

    # Telegram
    enable_telegram: bool = True
    telegram_bot_token: str = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID')
    telegram_parse_mode: str = "Markdown"

cfg = Config()

# --------------------------- Types --------------------------- #

@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int

@dataclass
class TriangleState:
    upper_slope: float
    upper_intercept: float
    lower_slope: float
    lower_intercept: float
    start_idx: int
    end_idx: int

@dataclass
class PendingRetest:
    direction: str           # 'UP' or 'DOWN'
    breakout_idx: int
    line_at_breakout: float
    upper_slope: float
    upper_intercept: float
    lower_slope: float
    lower_intercept: float

@dataclass
class PendingBreakoutConfirm:
    direction: str
    breakout_idx: int
    confirms_needed: int
    confirms_got: int
    upper_slope: float
    upper_intercept: float
    lower_slope: float
    lower_intercept: float
    line_at_breakout: float

@dataclass
class SymbolState:
    symbol: str
    candles: List[Candle] = field(default_factory=list)
    last_alert_time: float = 0.0
    pending: Optional[PendingRetest] = None
    pending_confirm: Optional[PendingBreakoutConfirm] = None

# --------------------------- Utils --------------------------- #

def pivot_high(candles: List[Candle], i: int, span: int) -> bool:
    if i - span < 0 or i + span >= len(candles): 
        return False
    h = candles[i].high
    for j in range(i - span, i + span + 1):
        if candles[j].high > h:
            return False
    return True

def pivot_low(candles: List[Candle], i: int, span: int) -> bool:
    if i - span < 0 or i + span >= len(candles): 
        return False
    l = candles[i].low
    for j in range(i - span, i + span + 1):
        if candles[j].low < l:
            return False
    return True

def linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    if len(xs) < 2: 
        return 0.0, ys[-1] if ys else 0.0
    m, b = np.polyfit(xs, ys, 1)
    return float(m), float(b)

def atr(candles: List[Candle], n: int) -> float:
    if len(candles) < n + 1:
        return 0.0
    trs = []
    for i in range(1, n+1):
        high = candles[-i].high
        low = candles[-i].low
        prev_close = candles[-i-1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return float(sum(trs) / n)

def moving_average(vals: List[float], n: int) -> float:
    if len(vals) < n or n <= 0: 
        return 0.0
    return float(sum(vals[-n:]) / n)

def ema(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2/(period+1)
    ema_vals = [values[0]]
    for v in values[1:]:
        ema_vals.append(ema_vals[-1] + k*(v - ema_vals[-1]))
    return ema_vals

def price_vs_trendlines(state: TriangleState, idx: int) -> Tuple[float, float]:
    upper = state.upper_slope * idx + state.upper_intercept
    lower = state.lower_slope * idx + state.lower_intercept
    return upper, lower

# --------------------------- Core Logic --------------------------- #

def compute_triangle(candles: List[Candle], start_idx: int, end_idx: int, span: int, max_pivots: int, min_piv_req: int) -> Optional[TriangleState]:
    highs_x, highs_y = [], []
    lows_x, lows_y = [], []

    for i in range(start_idx, end_idx+1):
        if pivot_high(candles, i, span):
            highs_x.append(i); highs_y.append(candles[i].high)
        if pivot_low(candles, i, span):
            lows_x.append(i); lows_y.append(candles[i].low)

    highs_x, highs_y = highs_x[-max_pivots:], highs_y[-max_pivots:]
    lows_x, lows_y   = lows_x[-max_pivots:], lows_y[-max_pivots:]

    if len(highs_x) < min_piv_req or len(lows_x) < min_piv_req:
        return None

    m_up, b_up = linear_regression(highs_x, highs_y)
    m_lo, b_lo = linear_regression(lows_x, lows_y)

    if not (m_up < 0 and m_lo > 0):
        return None

    gap_start = (m_up*start_idx + b_up) - (m_lo*start_idx + b_lo)
    gap_end   = (m_up*end_idx   + b_up) - (m_lo*end_idx   + b_lo)
    if gap_start <= 0 or gap_end <= 0:
        return None
    if gap_end >= gap_start:
        return None

    return TriangleState(m_up, b_up, m_lo, b_lo, start_idx, end_idx)

def contraction_ok(candles: List[Candle], state: TriangleState, atr_win: int, atr_factor: float, contraction_ratio: float) -> bool:
    window = candles[state.start_idx: state.end_idx+1]
    if len(window) < atr_win + 10:
        return False

    cut = len(window) // 2
    early = window[:cut]
    late = window[cut:]
    early_range = max(c.high for c in early) - min(c.low for c in early)
    late_range  = max(c.high for c in late)  - min(c.low for c in late)
    if early_range <= 0:
        return False
    ratio = late_range / early_range
    if ratio > contraction_ratio:
        return False

    atr_recent = atr(window, atr_win)
    atr_long   = atr(candles[max(0, state.start_idx-atr_win): state.end_idx+1], atr_win)
    if atr_recent <= 0 or atr_long <= 0:
        return False
    if atr_recent / atr_long > atr_factor:
        return False

    return True

def breakout_seed_signal(candles: List[Candle], state: TriangleState, vol_ma_win: int, vol_factor: float, buffer_pct: float) -> Optional[Dict]:
    """Initial breakout hit (first close beyond line with volume)."""
    i = len(candles) - 1
    last = candles[i]
    upper, lower = price_vs_trendlines(state, i)
    vols = [c.volume for c in candles]
    vol_ma = moving_average(vols, vol_ma_win)

    if last.close > upper * (1 + buffer_pct) and last.volume >= vol_ma * vol_factor:
        return {"direction": "UP", "close": last.close, "open": last.open, "high": last.high, "low": last.low,
                "line": upper, "volume": last.volume, "vol_ma": vol_ma, "idx": i}
    if last.close < lower * (1 - buffer_pct) and last.volume >= vol_ma * vol_factor:
        return {"direction": "DOWN", "close": last.close, "open": last.open, "high": last.high, "low": last.low,
                "line": lower, "volume": last.volume, "vol_ma": vol_ma, "idx": i}
    return None

def passes_body_filters(sig: Dict, atr_val: float) -> bool:
    body = abs(sig["close"] - sig["open"])
    rng = max(1e-12, sig["high"] - sig["low"])
    if atr_val <= 0:
        return False
    # Body relative to ATR
    if body < cfg.min_body_atr * atr_val:
        return False
    # Body relative to candle range (limits long-wick)
    if (body / rng) < cfg.min_body_to_range:
        return False
    return True

def line_value_for_direction(state: TriangleState, direction: str, idx: int) -> float:
    if direction == "UP":
        return state.upper_slope * idx + state.upper_intercept
    else:
        return state.lower_slope * idx + state.lower_intercept

def bar_confirms_breakout(direction: str, close_price: float, line_value: float, buffer_pct: float) -> bool:
    if direction == "UP":
        return close_price > line_value * (1 + buffer_pct)
    else:
        return close_price < line_value * (1 - buffer_pct)

# --------------------------- Direction & Leverage --------------------------- #

async def compute_direction_context(symbol: str, latest_close: float) -> Dict[str, str]:
    if not cfg.use_direction_filter:
        return {"tag": "No filter", "ctx": ""}
    async with aiohttp.ClientSession() as session:
        stf_kl = await fetch_klines(session, symbol, cfg.interval, max(cfg.seed_candles, cfg.ema_period+20))
        stf_closes = [c.close for c in stf_kl]
        stf_ema = ema(stf_closes, cfg.ema_period)[-1] if len(stf_closes) >= cfg.ema_period else 0.0

        htf_kl = await fetch_klines(session, symbol, cfg.htf_interval, max(400, cfg.htf_ema_period+cfg.htf_slope_lookback+5))
        htf_closes = [c.close for c in htf_kl]
        htf_emas = ema(htf_closes, cfg.htf_ema_period)
        if len(htf_emas) < cfg.htf_slope_lookback+1:
            return {"tag": "Insufficient HTF data", "ctx": ""}
        htf_ema_last = htf_emas[-1]
        htf_ema_prev = htf_emas[-1 - cfg.htf_slope_lookback]
        htf_slope_up = htf_ema_last > htf_ema_prev
        htf_slope_down = htf_ema_last < htf_ema_prev

    long_ok  = latest_close > stf_ema and htf_slope_up
    short_ok = latest_close < stf_ema and htf_slope_down

    if long_ok and not short_ok:
        tag = "LONG preferred"
    elif short_ok and not long_ok:
        tag = "SHORT preferred"
    else:
        tag = "Counter-trend (caution)"

    ctx = f"STF EMA200: {stf_ema:.4f} | HTF {cfg.htf_interval} EMA{cfg.htf_ema_period} slope: {'UP' if htf_slope_up else ('DOWN' if htf_slope_down else 'FLAT')}"
    return {"tag": tag, "ctx": ctx}

def leverage_for_loss(entry: float, sl: float) -> float:
    if entry <= 0 or sl <= 0:
        return 1.0
    distance_pct = abs(entry - sl) / entry
    if distance_pct == 0:
        return 1.0
    L = cfg.loss_pct_at_sl / distance_pct
    return max(1.0, min(cfg.leverage_cap, L))

# --------------------------- IO: Binance & Telegram --------------------------- #

async def fetch_klines(session: aiohttp.ClientSession, symbol: str, interval: str, limit: int) -> List[Candle]:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    url = REST_BASE + KLINES_ENDPOINT
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        resp.raise_for_status()
        data = await resp.json()
    candles = []
    for k in data:
        candles.append(Candle(
            open_time=int(k[0]),
            open=float(k[1]),
            high=float(k[2]),
            low=float(k[3]),
            close=float(k[4]),
            volume=float(k[5]),
            close_time=int(k[6])
        ))
    return candles

def parse_ws_kline(msg: Dict) -> Optional[Tuple[str, Candle]]:
    try:
        stream = msg.get("stream", "")
        symbol = stream.split("@")[0]
        k = msg["data"]["k"]
        if not k["x"]:
            return None
        candle = Candle(
            open_time=int(k["t"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["q"]),
            close_time=int(k["T"]),
        )
        return symbol, candle
    except Exception:
        return None

async def send_telegram(text: str):
    if not cfg.enable_telegram:
        return
    token = cfg.telegram_bot_token.strip()
    chat_id = cfg.telegram_chat_id.strip()
    if not token or not chat_id or "PUT_YOUR" in token or "PUT_YOUR" in chat_id:
        print("[WARN] Telegram not configured. Set telegram_bot_token and telegram_chat_id.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": cfg.telegram_parse_mode, "disable_web_page_preview": True}
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"[Telegram] HTTP {resp.status}: {body}")
        except Exception as e:
            print(f"[Telegram] Error: {e}")

# --------------------------- Streams & Handlers --------------------------- #

async def seed_all(states: Dict[str, SymbolState]):
    async with aiohttp.ClientSession() as session:
        for sym in SYMBOLS:
            kl = await fetch_klines(session, sym, cfg.interval, cfg.seed_candles)
            states[sym].candles = kl
            await asyncio.sleep(0.1)

async def multiplex_stream(symbols: List[str], interval: str):
    streams = "/".join([f"{s}@kline_{interval}" for s in symbols])
    url = f"{WS_BASE}?streams={streams}"
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=5) as ws:
                async for message in ws:
                    yield json.loads(message)
        except Exception:
            await asyncio.sleep(cfg.ws_reconnect_delay)
            continue

def within_tolerance(price: float, line: float, tol_pct: float) -> bool:
    return abs(price - line) <= line * tol_pct

def compute_sl_tp(direction: str, entry_price: float, retest_extreme: float, line_at_retest: float, atr_val: float) -> Dict[str, float]:
    if direction == "UP":
        sl = min(retest_extreme, line_at_retest) * (1 - cfg.stop_buffer_pct)
        risk = max(1e-9, entry_price - sl)
        targets = {}
        if cfg.use_atr_targets and atr_val > 0:
            for m in cfg.atr_tp_multipliers:
                targets[f"TP_ATR_{m}x"] = entry_price + m * atr_val
        else:
            for r in cfg.tp_rr_list:
                targets[f"TP_{r:.1f}R"] = entry_price + r * risk
        return {"SL": sl, **targets}
    else:
        sl = max(retest_extreme, line_at_retest) * (1 + cfg.stop_buffer_pct)
        risk = max(1e-9, sl - entry_price)
        targets = {}
        if cfg.use_atr_targets and atr_val > 0:
            for m in cfg.atr_tp_multipliers:
                targets[f"TP_ATR_{m}x"] = entry_price - m * atr_val
        else:
            for r in cfg.tp_rr_list:
                targets[f"TP_{r:.1f}R"] = entry_price - r * risk
        return {"SL": sl, **targets}

# --------------------------- Main Loop --------------------------- #

async def main():
    send_telegram(f"Starting Multi-Symbol Triangle Breakout (5m) with Fake-breakout filters for: {', '.join([s.upper() for s in SYMBOLS])}")
    states: Dict[str, SymbolState] = {s: SymbolState(symbol=s) for s in SYMBOLS}
    await seed_all(states)

    async for raw in multiplex_stream(SYMBOLS, cfg.interval):
        parsed = parse_ws_kline(raw)
        if not parsed:
            continue
        sym, candle = parsed
        st = states.get(sym)
        if not st:
            continue
        st.candles.append(candle)
        if len(st.candles) > max(cfg.seed_candles, cfg.lookback_bars*3):
            st.candles = st.candles[-max(cfg.seed_candles, cfg.lookback_bars*3):]

        i = len(st.candles) - 1
        end_idx = i
        start_idx = max(0, end_idx - cfg.lookback_bars)
        tri = compute_triangle(st.candles, start_idx, end_idx, cfg.pivot_span, cfg.max_pivots, cfg.min_pivots_required)

        # If we are waiting for breakout confirmations, process them first
        if st.pending_confirm:
            # still within same triangle slopes to compute line
            direction = st.pending_confirm.direction
            line_here = line_value_for_direction(tri if tri else TriangleState(st.pending_confirm.upper_slope, st.pending_confirm.upper_intercept, st.pending_confirm.lower_slope, st.pending_confirm.lower_intercept, 0, 0), direction, i)
            if bar_confirms_breakout(direction, st.candles[i].close, line_here, cfg.breakout_buffer_pct):
                st.pending_confirm.confirms_got += 1
                if st.pending_confirm.confirms_got >= st.pending_confirm.confirms_needed:
                    # confirmation achieved: send breakout alert & arm retest
                    price = st.candles[i].close
                    vols = [c.volume for c in st.candles]
                    vol_ma = moving_average(vols, cfg.vol_ma_window)
                    dir_ctx = await compute_direction_context(sym, price)
                    tag = dir_ctx["tag"]; ctx_str = dir_ctx["ctx"]
                    text = (
                        f"*{sym.upper()} {cfg.interval}* — *TRIANGLE BREAKOUT (confirmed)* `{direction}`\n"
                        f"• Close: `{price:.6f}` vs Line: `{line_here:.6f}`\n"
                        f"• Volume MA({cfg.vol_ma_window}): `{vol_ma:.0f}`\n"
                        f"• Direction: *{tag}*\n"
                        f"• TrendCtx: `{ctx_str}`\n"
                        f"• Next: watching for *retest* within `{cfg.retest_max_bars}` bars (±{cfg.retest_tolerance_pct*100:.2f}%)"
                    )
                    print(text.replace("*","").replace("`",""))
                    await send_telegram(text)

                    st.pending = PendingRetest(
                        direction=direction,
                        breakout_idx=i,
                        line_at_breakout=line_here,
                        upper_slope=tri.upper_slope if tri else st.pending_confirm.upper_slope,
                        upper_intercept=tri.upper_intercept if tri else st.pending_confirm.upper_intercept,
                        lower_slope=tri.lower_slope if tri else st.pending_confirm.lower_slope,
                        lower_intercept=tri.lower_intercept if tri else st.pending_confirm.lower_intercept,
                    )
                    st.pending_confirm = None
            else:
                # invalidated
                await send_telegram(f"*{sym.upper()} {cfg.interval}* — Breakout invalidated during confirmation phase.")
                st.pending_confirm = None

            # continue to next loop (avoid double-processing as new triangle may form)
            continue

        # Fresh triangle & initial breakout seed
        if tri and contraction_ok(st.candles, tri, cfg.atr_window, cfg.atr_contraction_factor, cfg.min_contraction_ratio):
            sig = breakout_seed_signal(st.candles, tri, cfg.vol_ma_window, cfg.vol_confirm_factor, cfg.breakout_buffer_pct)
            now = time.time()
            if sig and now - st.last_alert_time > 10:
                # Candlestick quality filters (anti-fake)
                atr_val = atr(st.candles, cfg.atr_window)
                if not passes_body_filters(sig, atr_val):
                    # soft notify? For now, just skip.
                    continue

                st.last_alert_time = now
                direction = sig["direction"]
                line_at_break = sig["line"]

                # If confirmation bars requested, stage confirmation
                if cfg.confirm_bars > 0:
                    st.pending_confirm = PendingBreakoutConfirm(
                        direction=direction,
                        breakout_idx=sig["idx"],
                        confirms_needed=cfg.confirm_bars,
                        confirms_got=0,
                        upper_slope=tri.upper_slope,
                        upper_intercept=tri.upper_intercept,
                        lower_slope=tri.lower_slope,
                        lower_intercept=tri.lower_intercept,
                        line_at_breakout=line_at_break,
                    )
                    await send_telegram(f"*{sym.upper()} {cfg.interval}* — Breakout spotted, waiting `{cfg.confirm_bars}` confirm bar(s) to filter fake-outs.")
                else:
                    # No confirmation needed: alert immediately and arm retest
                    price = sig["close"]
                    dir_ctx = await compute_direction_context(sym, price)
                    tag = dir_ctx["tag"]; ctx_str = dir_ctx["ctx"]
                    text = (
                        f"*{sym.upper()} {cfg.interval}* — *TRIANGLE BREAKOUT* `{direction}`\n"
                        f"• Close: `{price:.6f}` vs Line: `{line_at_break:.6f}`\n"
                        f"• Direction: *{tag}*\n"
                        f"• TrendCtx: `{ctx_str}`\n"
                        f"• Next: watching for *retest* within `{cfg.retest_max_bars}` bars (±{cfg.retest_tolerance_pct*100:.2f}%)"
                    )
                    print(text.replace("*","").replace("`",""))
                    await send_telegram(text)
                    st.pending = PendingRetest(
                        direction=direction,
                        breakout_idx=i,
                        line_at_breakout=line_at_break,
                        upper_slope=tri.upper_slope,
                        upper_intercept=tri.upper_intercept,
                        lower_slope=tri.lower_slope,
                        lower_intercept=tri.lower_intercept,
                    )

        # Retest handling
        if st.pending:
            bars_since = i - st.pending.breakout_idx
            if bars_since > cfg.retest_max_bars:
                await send_telegram(f"*{sym.upper()} {cfg.interval}* — Retest window expired (>{cfg.retest_max_bars} bars).")
                st.pending = None
                continue

            if st.pending.direction == "UP":
                line_here = st.pending.upper_slope * i + st.pending.upper_intercept
                touched = within_tolerance(st.candles[i].low, line_here, cfg.retest_tolerance_pct)
                confirm = st.candles[i].close > line_here if cfg.retest_confirm_close else touched
                if touched and confirm:
                    entry = st.candles[i].close
                    retest_extreme = st.candles[i].low
                    atr_val = atr(st.candles, cfg.atr_window)
                    levels = compute_sl_tp("UP", entry, retest_extreme, line_here, atr_val)
                    sl = levels["SL"]; L = leverage_for_loss(entry, sl)
                    lev_note = f"Lev≈`{L:.2f}x` to risk ~{cfg.loss_pct_at_sl*100:.0f}% at SL"
                    level_str = "\n".join([f"• {k}: `{v:.6f}`" for k, v in levels.items() if k != "SL"])
                    msg = (
                        f"*{sym.upper()} {cfg.interval}* — ✅ *Retest LONG Entry*\n"
                        f"• Entry: `{entry:.6f}` (retested upper line `{line_here:.6f}`)\n"
                        f"• SL: `{sl:.6f}`  ({lev_note})\n"
                        f"{level_str}"
                    )
                    print(msg.replace("*","").replace("`",""))
                    await send_telegram(msg)
                    st.pending = None
            else:
                line_here = st.pending.lower_slope * i + st.pending.lower_intercept
                touched = within_tolerance(st.candles[i].high, line_here, cfg.retest_tolerance_pct)
                confirm = st.candles[i].close < line_here if cfg.retest_confirm_close else touched
                if touched and confirm:
                    entry = st.candles[i].close
                    retest_extreme = st.candles[i].high
                    atr_val = atr(st.candles, cfg.atr_window)
                    levels = compute_sl_tp("DOWN", entry, retest_extreme, line_here, atr_val)
                    sl = levels["SL"]; L = leverage_for_loss(entry, sl)
                    lev_note = f"Lev≈`{L:.2f}x` to risk ~{cfg.loss_pct_at_sl*100:.0f}% at SL"
                    level_str = "\n".join([f"• {k}: `{v:.6f}`" for k, v in levels.items() if k != "SL"])
                    msg = (
                        f"*{sym.upper()} {cfg.interval}* — ✅ *Retest SHORT Entry*\n"
                        f"• Entry: `{entry:.6f}` (retested lower line `{line_here:.6f}`)\n"
                        f"• SL: `{sl:.6f}`  ({lev_note})\n"
                        f"{level_str}"
                    )
                    print(msg.replace("*","").replace("`",""))
                    await send_telegram(msg)
                    st.pending = None

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")
