
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triangle/Wedge Breakout Detector — ETH/XRP/SOL with MTF + Pattern Options
-------------------------------------------------------------------------
Supports detecting:
- Symmetrical triangle (upper slope < 0, lower slope > 0, contracting)
- Ascending triangle (flat/near-flat upper, rising lower, contracting)
- Descending triangle (falling upper, flat/near-flat lower, contracting)
- Rising wedge   (upper > 0, lower > 0, contracting; upper slope < lower slope)
- Falling wedge  (upper < 0, lower < 0, contracting; upper slope > lower slope)

(Optionally you can enable broadening patterns, but disabled by default.)

Includes:
- Anti-fake filters (confirm bars, body/ATR, body/range)
- Retest-based entry with SL/TP
- Leverage hint (SL hit ≈ 10% notional)
- MTF fallback 5m→3m→15m→30m→1h
- Telegram alerts

NOTE: Educational reference. Paper trade/backtest before live usage.
"""
import os
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp
import websockets
import numpy as np

REST_BASE = "https://fapi.binance.com"
WS_BASE = "wss://fstream.binance.com/stream"
KLINES_ENDPOINT = "/fapi/v1/klines"

SYMBOLS = ["ethusdt", "xrpusdt", "solusdt"]
TF_PRIORITY = ["5m", "3m", "15m", "30m", "1h"]

# --------------------------- Config --------------------------- #

@dataclass
class Config:
    seed_candles: int = 600
    ws_reconnect_delay: int = 5

    # Triangle detection
    pivot_span: int = 3
    max_pivots: int = 30
    min_pivots_required: int = 4
    lookback_bars: int = 200

    # Acceptable patterns
    patterns: Tuple[str, ...] = ("symmetrical", "ascending", "descending", "rising_wedge", "falling_wedge")
    slope_flat_threshold: float = 1e-6    # near-0 slope considered flat; adjust if index spacing is large
    # If using timestamps as x, slope scale changes. Here x=index, so threshold is tiny.

    # Contraction checks
    min_contraction_ratio: float = 0.35
    atr_window: int = 14
    atr_contraction_factor: float = 0.7
    allow_broadening: bool = False  # if True, also allow gap_end > gap_start (broadening patterns)

    # Breakout rules
    breakout_buffer_pct: float = 0.001
    vol_ma_window: int = 20
    vol_confirm_factor: float = 1.5

    # Anti-fake
    confirm_bars: int = 1
    min_body_atr: float = 0.25
    min_body_to_range: float = 0.40

    # Retest
    retest_max_bars: int = 10
    retest_tolerance_pct: float = 0.002
    retest_confirm_close: bool = True
    stop_buffer_pct: float = 0.002
    tp_rr_list: Tuple[float, ...] = (1.0, 2.0)

    # Leverage target
    loss_pct_at_sl: float = 0.10
    leverage_cap: float = 50.0

    # Telegram
    enable_telegram: bool = True
    telegram_bot_token: str = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID')
    telegram_parse_mode: str = "Markdown"

cfg = Config()

# --------------------------- Data Types --------------------------- #

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
    up_m: float
    up_b: float
    lo_m: float
    lo_b: float
    start_idx: int
    end_idx: int
    pattern: str

@dataclass
class PendingRetest:
    direction: str
    breakout_idx: int
    up_m: float
    up_b: float
    lo_m: float
    lo_b: float

@dataclass
class PendingConfirm:
    direction: str
    confirms_needed: int
    confirms_got: int
    up_m: float
    up_b: float
    lo_m: float
    lo_b: float

@dataclass
class TFState:
    candles: List[Candle] = field(default_factory=list)
    vol_ma: List[float] = field(default_factory=list)
    atr: List[float] = field(default_factory=list)
    last_alert_time: float = 0.0
    pending_confirm: Optional[PendingConfirm] = None
    pending_retest: Optional[PendingRetest] = None

@dataclass
class SymbolState:
    tfs: Dict[str, TFState] = field(default_factory=lambda: {tf: TFState() for tf in TF_PRIORITY})

# --------------------------- Math helpers --------------------------- #

def ma(values: List[float], period: int) -> float:
    if len(values) < period or period <= 0:
        return float('nan')
    return float(sum(values[-period:]) / period)

def atr_calc(candles: List[Candle], n: int) -> float:
    if len(candles) < n+1:
        return float('nan')
    trs = []
    for i in range(1, n+1):
        hi = candles[-i].high
        lo = candles[-i].low
        prev_close = candles[-i-1].close
        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        trs.append(tr)
    return float(sum(trs)/n)

def pivot_high(candles: List[Candle], i: int, span: int) -> bool:
    if i - span < 0 or i + span >= len(candles): return False
    h = candles[i].high
    for j in range(i-span, i+span+1):
        if candles[j].high > h: return False
    return True

def pivot_low(candles: List[Candle], i: int, span: int) -> bool:
    if i - span < 0 or i + span >= len(candles): return False
    l = candles[i].low
    for j in range(i-span, i+span+1):
        if candles[j].low < l: return False
    return True

def linreg(xs: List[int], ys: List[float]) -> Tuple[float,float]:
    if len(xs) < 2: return 0.0, ys[-1] if ys else 0.0
    m, b = np.polyfit(xs, ys, 1)
    return float(m), float(b)

# --------------------------- Pattern logic --------------------------- #

def slope_is_flat(m: float) -> bool:
    return abs(m) <= cfg.slope_flat_threshold

def classify_pattern(m_up: float, m_lo: float) -> Optional[str]:
    up, lo = m_up, m_lo
    if "symmetrical" in cfg.patterns and (up < 0 and lo > 0):
        return "symmetrical"
    if "ascending" in cfg.patterns and slope_is_flat(up) and lo > 0:
        return "ascending"
    if "descending" in cfg.patterns and up < 0 and slope_is_flat(lo):
        return "descending"
    if "rising_wedge" in cfg.patterns and (up > 0 and lo > 0 and up < lo):
        return "rising_wedge"
    if "falling_wedge" in cfg.patterns and (up < 0 and lo < 0 and up > lo):
        return "falling_wedge"
    return None

def compute_triangle(candles: List[Candle]) -> Optional[TriangleState]:
    i_end = len(candles)-1
    i_start = max(0, i_end - cfg.lookback_bars)
    highs_x, highs_y, lows_x, lows_y = [], [], [], []
    for i in range(i_start, i_end+1):
        if pivot_high(candles, i, cfg.pivot_span):
            highs_x.append(i); highs_y.append(candles[i].high)
        if pivot_low(candles, i, cfg.pivot_span):
            lows_x.append(i); lows_y.append(candles[i].low)
    highs_x, highs_y = highs_x[-cfg.max_pivots:], highs_y[-cfg.max_pivots:]
    lows_x,  lows_y  = lows_x[-cfg.max_pivots:],  lows_y[-cfg.max_pivots:]
    if len(highs_x) < cfg.min_pivots_required or len(lows_x) < cfg.min_pivots_required:
        return None
    up_m, up_b = linreg(highs_x, highs_y)
    lo_m, lo_b = linreg(lows_x, lows_y)

    pat = classify_pattern(up_m, lo_m)
    if pat is None:
        return None

    gap_start = (up_m*i_start + up_b) - (lo_m*i_start + lo_b)
    gap_end   = (up_m*i_end   + up_b) - (lo_m*i_end   + lo_b)
    if not cfg.allow_broadening:
        if gap_start <= 0 or gap_end <= 0: 
            return None
        if gap_end >= gap_start: 
            return None  # must be contracting
    else:
        # If broadening allowed: accept either contracting (gap_end < gap_start) or broadening (gap_end > gap_start)
        # But keep gap positive to avoid crossing lines.
        if gap_start <= 0 or gap_end <= 0:
            return None

    return TriangleState(up_m, up_b, lo_m, lo_b, i_start, i_end, pat)

# --------------------------- Breakout pipeline --------------------------- #

def line_vals(state: TriangleState, i: int) -> Tuple[float,float]:
    return state.up_m*i + state.up_b, state.lo_m*i + state.lo_b

def contraction_ok(candles: List[Candle], tri: TriangleState, atr_val_recent: float, atr_vals: List[float]) -> bool:
    if tri is None: return False
    window = candles[tri.start_idx:tri.end_idx+1]
    if len(window) < cfg.atr_window + 10: return False
    cut = len(window)//2
    early = window[:cut]
    late  = window[cut:]
    early_range = max(c.high for c in early) - min(c.low for c in early)
    late_range  = max(c.high for c in late)  - min(c.low for c in late)
    if early_range <= 0: return False
    if (late_range/early_range) > cfg.min_contraction_ratio: return False
    if len(atr_vals) < cfg.atr_window: return False
    atr_long = float(np.nanmean(atr_vals[max(0, tri.start_idx - cfg.atr_window):tri.end_idx+1]))
    if not (atr_val_recent>0 and atr_long>0): return False
    if (atr_val_recent/atr_long) > cfg.atr_contraction_factor: return False
    return True

def breakout_seed(candles: List[Candle], tri: TriangleState, vol_ma_value: float) -> Optional[Dict]:
    i = len(candles)-1
    last = candles[i]
    up, lo = line_vals(tri, i)
    if np.isnan(vol_ma_value): return None
    if last.close > up*(1+cfg.breakout_buffer_pct) and last.volume >= vol_ma_value*cfg.vol_confirm_factor:
        return {"direction":"UP", "idx":i, "open":last.open,"high":last.high,"low":last.low,"close":last.close,"line":up, "pattern": tri.pattern}
    if last.close < lo*(1-cfg.breakout_buffer_pct) and last.volume >= vol_ma_value*cfg.vol_confirm_factor:
        return {"direction":"DOWN","idx":i, "open":last.open,"high":last.high,"low":last.low,"close":last.close,"line":lo, "pattern": tri.pattern}
    return None

def body_filters(sig: Dict, atr_val: float) -> bool:
    body = abs(sig["close"] - sig["open"])
    rng  = max(1e-12, sig["high"] - sig["low"])
    if atr_val <= 0: return False
    if body < cfg.min_body_atr * atr_val: return False
    if (body / rng) < cfg.min_body_to_range: return False
    return True

def bar_confirms(direction: str, close_price: float, line_value: float) -> bool:
    if direction == "UP":
        return close_price > line_value*(1+cfg.breakout_buffer_pct)
    else:
        return close_price < line_value*(1-cfg.breakout_buffer_pct)

def within_tolerance(price: float, line: float, tol_pct: float) -> bool:
    return abs(price - line) <= line * tol_pct

def compute_sl_tp(direction: str, entry: float, retest_extreme: float, line_at_retest: float) -> Dict[str, float]:
    if direction == "UP":
        sl = min(retest_extreme, line_at_retest) * (1 - cfg.stop_buffer_pct)
        risk = max(1e-9, entry - sl)
        return {"SL": sl, "TP_1.0R": entry + 1.0*risk, "TP_2.0R": entry + 2.0*risk}
    else:
        sl = max(retest_extreme, line_at_retest) * (1 + cfg.stop_buffer_pct)
        risk = max(1e-9, sl - entry)
        return {"SL": sl, "TP_1.0R": entry - 1.0*risk, "TP_2.0R": entry - 2.0*risk}

def leverage_for_loss(entry: float, sl: float) -> float:
    if entry <= 0 or sl <= 0: return 1.0
    dist_pct = abs(entry - sl) / entry
    if dist_pct <= 0: return 1.0
    L = cfg.loss_pct_at_sl / dist_pct
    return max(1.0, min(cfg.leverage_cap, L))

# --------------------------- IO --------------------------- #

async def fetch_seed_klines(session: aiohttp.ClientSession, symbol: str, interval: str, limit: int) -> List[Candle]:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    url = REST_BASE + KLINES_ENDPOINT
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        resp.raise_for_status()
        data = await resp.json()
    out = []
    for k in data:
        out.append(Candle(
            open_time=int(k[0]), open=float(k[1]), high=float(k[2]), low=float(k[3]),
            close=float(k[4]), volume=float(k[5]), close_time=int(k[6])
        ))
    return out

def parse_ws_kline(msg: Dict) -> Optional[Tuple[str,str,Candle]]:
    try:
        stream = msg.get("stream","")               # e.g., ethusdt@kline_5m
        sym, rest = stream.split("@", 1)
        tf = rest.split("_")[1]
        k = msg["data"]["k"]
        if not k["x"]:
            return None
        candle = Candle(
            open_time=int(k["t"]), open=float(k["o"]), high=float(k["h"]),
            low=float(k["l"]), close=float(k["c"]), volume=float(k["q"]), close_time=int(k["T"])
        )
        return sym, tf, candle
    except Exception:
        return None

async def telegram(text: str):
    if not cfg.enable_telegram: 
        return
    token = cfg.telegram_bot_token.strip()
    chat_id = cfg.telegram_chat_id.strip()
    if not token or not chat_id or "PUT_YOUR" in token or "PUT_YOUR" in chat_id:
        print("[WARN] Telegram not configured.")
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

# --------------------------- Stream orchestration --------------------------- #

async def seed_all(states: Dict[str, SymbolState]):
    async with aiohttp.ClientSession() as session:
        for sym in SYMBOLS:
            for tf in TF_PRIORITY:
                kl = await fetch_seed_klines(session, sym, tf, cfg.seed_candles)
                st = states[sym].tfs[tf]
                st.candles = kl
                # rolling vol MA
                volumes = [c.volume for c in st.candles]
                st.vol_ma = [float('nan')]*len(volumes)
                for i in range(len(volumes)):
                    if i+1 >= cfg.vol_ma_window:
                        st.vol_ma[i] = float(sum(volumes[i+1-cfg.vol_ma_window:i+1])/cfg.vol_ma_window)
                # rolling ATR
                st.atr = [float('nan')]*len(st.candles)
                for i in range(len(st.candles)):
                    if i+1 >= cfg.atr_window+1:
                        trs = []
                        for j in range(i-cfg.atr_window+1, i+1):
                            hi = st.candles[j].high
                            lo = st.candles[j].low
                            prev_close = st.candles[j-1].close if j-1 >= 0 else st.candles[j].open
                            trs.append(max(hi-lo, abs(hi-prev_close), abs(lo-prev_close)))
                        st.atr[i] = float(sum(trs)/cfg.atr_window)
            await asyncio.sleep(0.1)

async def multiplex_stream(symbols: List[str], tfs: List[str]):
    streams = "/".join([f"{s}@kline_{tf}" for s in symbols for tf in tfs])
    url = f"{WS_BASE}?streams={streams}"
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=5) as ws:
                async for message in ws:
                    yield json.loads(message)
        except Exception:
            await asyncio.sleep(cfg.ws_reconnect_delay)
            continue

# --------------------------- Processing --------------------------- #

def process_breakout_pipeline(sym: str, tf: str, state: TFState) -> Optional[str]:
    i = len(state.candles)-1
    if i < max(cfg.lookback_bars, cfg.atr_window+5, cfg.vol_ma_window):
        return None

    tri = compute_triangle(state.candles)
    atr_recent = state.atr[i]
    if tri is None or not contraction_ok(state.candles, tri, atr_recent, state.atr):
        return None

    vol_ma_val = state.vol_ma[i]
    seed = breakout_seed(state.candles, tri, vol_ma_val)
    if seed is None:
        return None

    if not body_filters(seed, atr_recent if atr_recent==atr_recent else float('nan')):
        return None

    if cfg.confirm_bars > 0:
        state.pending_confirm = PendingConfirm(
            direction=seed["direction"],
            confirms_needed=cfg.confirm_bars,
            confirms_got=0,
            up_m=tri.up_m, up_b=tri.up_b,
            lo_m=tri.lo_m, lo_b=tri.lo_b
        )
        return "STAGED"
    else:
        state.pending_retest = PendingRetest(
            direction=seed["direction"],
            breakout_idx=i,
            up_m=tri.up_m, up_b=tri.up_b,
            lo_m=tri.lo_m, lo_b=tri.lo_b
        )
        return "TRIGGERED"

async def handle_confirmation_and_alert(sym: str, tf: str, st: TFState):
    i = len(st.candles)-1
    pc = st.pending_confirm
    if not pc:
        return
    up_line = pc.up_m*i + pc.up_b
    lo_line = pc.lo_m*i + pc.lo_b
    line_here = up_line if pc.direction=="UP" else lo_line
    if bar_confirms(pc.direction, st.candles[i].close, line_here):
        pc.confirms_got += 1
        if pc.confirms_got >= pc.confirms_needed:
            price = st.candles[i].close
            now = time.time()
            if now - st.last_alert_time > 10:
                st.last_alert_time = now
                msg = (
                    f"*{sym.upper()} {tf}* — *BREAKOUT (confirmed)* `{pc.direction}`\n"
                    f"• Close: `{price:.6f}` vs Line: `{line_here:.6f}`\n"
                    f"• Volume MA({cfg.vol_ma_window}): `{(st.vol_ma[i] if i<len(st.vol_ma) else float('nan')):.0f}`\n"
                    f"• Next: watching for *retest* within `{cfg.retest_max_bars}` bars (±{cfg.retest_tolerance_pct*100:.2f}%)"
                )
                print(msg.replace("*","").replace("`",""))
                await telegram(msg)
            st.pending_retest = PendingRetest(pc.direction, i, pc.up_m, pc.up_b, pc.lo_m, pc.lo_b)
            st.pending_confirm = None
    else:
        st.pending_confirm = None

async def handle_retest(sym: str, tf: str, st: TFState):
    pr = st.pending_retest
    if not pr:
        return
    i = len(st.candles)-1
    bars_since = i - pr.breakout_idx
    if bars_since > cfg.retest_max_bars:
        await telegram(f"*{sym.upper()} {tf}* — Retest window expired (>{cfg.retest_max_bars} bars).")
        st.pending_retest = None
        return

    up_line = pr.up_m*i + pr.up_b
    lo_line = pr.lo_m*i + pr.lo_b

    if pr.direction == "UP":
        line_here = up_line
        touched = within_tolerance(st.candles[i].low, line_here, cfg.retest_tolerance_pct)
        confirm = st.candles[i].close > line_here if cfg.retest_confirm_close else touched
        if touched and confirm:
            entry = st.candles[i].close
            retest_extreme = st.candles[i].low
            lv = compute_sl_tp("UP", entry, retest_extreme, line_here)
            sl = lv["SL"]; L = leverage_for_loss(entry, sl)
            msg = (
                f"*{sym.upper()} {tf}* — ✅ *Retest LONG Entry*\n"
                f"• Entry: `{entry:.6f}` (line `{line_here:.6f}`)\n"
                f"• SL: `{sl:.6f}`  (Lev≈`{L:.2f}x` to risk ~{cfg.loss_pct_at_sl*100:.0f}% at SL)\n"
                f"• TP_1.0R: `{lv['TP_1.0R']:.6f}`\n"
                f"• TP_2.0R: `{lv['TP_2.0R']:.6f}`"
            )
            print(msg.replace("*","").replace("`",""))
            await telegram(msg)
            st.pending_retest = None
    else:
        line_here = lo_line
        touched = within_tolerance(st.candles[i].high, line_here, cfg.retest_tolerance_pct)
        confirm = st.candles[i].close < line_here if cfg.retest_confirm_close else touched
        if touched and confirm:
            entry = st.candles[i].close
            retest_extreme = st.candles[i].high
            lv = compute_sl_tp("DOWN", entry, retest_extreme, line_here)
            sl = lv["SL"]; L = leverage_for_loss(entry, sl)
            msg = (
                f"*{sym.upper()} {tf}* — ✅ *Retest SHORT Entry*\n"
                f"• Entry: `{entry:.6f}` (line `{line_here:.6f}`)\n"
                f"• SL: `{sl:.6f}`  (Lev≈`{L:.2f}x` to risk ~{cfg.loss_pct_at_sl*100:.0f}% at SL)\n"
                f"• TP_1.0R: `{lv['TP_1.0R']:.6f}`\n"
                f"• TP_2.0R: `{lv['TP_2.0R']:.6f}`"
            )
            print(msg.replace("*","").replace("`",""))
            await telegram(msg)
            st.pending_retest = None

# --------------------------- Main --------------------------- #

async def seed_all_states(states: Dict[str, SymbolState]):
    await seed_all(states)

async def multiplex_stream(symbols: List[str], tfs: List[str]):
    streams = "/".join([f"{s}@kline_{tf}" for s in symbols for tf in tfs])
    url = f"{WS_BASE}?streams={streams}"
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=5) as ws:
                async for message in ws:
                    yield json.loads(message)
        except Exception:
            await asyncio.sleep(cfg.ws_reconnect_delay)
            continue

async def main():
    await telegram("Starting MTF Triangle/Wedge Breakout with Pattern Options")
    states: Dict[str, SymbolState] = {s: SymbolState() for s in SYMBOLS}
    await seed_all_states(states)

    async for raw in multiplex_stream(SYMBOLS, TF_PRIORITY):
        parsed = parse_ws_kline(raw)
        if not parsed:
            continue
        sym, tf, candle = parsed
        st = states[sym].tfs[tf]
        st.candles.append(candle)
        # rolling metrics
        vols = [c.volume for c in st.candles]
        st.vol_ma.append(ma(vols, cfg.vol_ma_window))
        st.atr.append(atr_calc(st.candles, cfg.atr_window))

        # maintenance per TF (confirmation, retest) in priority order
        for tf_c in TF_PRIORITY:
            tf_state = states[sym].tfs[tf_c]
            await handle_confirmation_and_alert(sym, tf_c, tf_state)
            await handle_retest(sym, tf_c, tf_state)

        # try to stage/trigger in priority order
        for tf_c in TF_PRIORITY:
            tf_state = states[sym].tfs[tf_c]
            res = process_breakout_pipeline(sym, tf_c, tf_state)
            if res in ("STAGED", "TRIGGERED"):
                break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")
