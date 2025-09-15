# -*- coding: utf-8 -*-
"""
Multi-Timeframe BB+CCI Scanner
- Entry: 5m Bollinger + CCI
- Filter: 1h CCI 방향성 확인
- Symbols: BTCUSDT, ETHUSDT, SOLUSDT
- Features: Partial TP + Breakeven SL, Multiple concurrent positions
"""

import os, io, json, time
import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from websocket import WebSocketApp

# ===== Telegram =====
TELEGRAM_TOKEN   = os.environ.get("OKX_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("OKX_TELEGRAM_CHAT_ID")

def tg(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG]", msg); return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        print("TG error:", e)

# ===== Settings =====
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

STREAMS = [f"{s.lower()}@kline_5m" for s in SYMBOLS] + \
          [f"{s.lower()}@kline_1h" for s in SYMBOLS]
WS_URL  = "wss://fstream.binance.com/stream?streams=" + "/".join(STREAMS)

BB_N, BB_K = 20, 2.0
CCI_N = 20
CCI_UP, CCI_DN = 100, -100
TOUCH_LOOKBACK = 3
KST = ZoneInfo("Asia/Seoul")

SEND_CHARTS_ON_CONFIRM = True
MAX_KEEP = 500

# ===== Buffers =====
buf_5m = {s: pd.DataFrame() for s in SYMBOLS}
buf_1h = {s: pd.DataFrame() for s in SYMBOLS}

# ===== Position State (multi) =====
# positions = {symbol: {"side":..., "entry":..., "ptp_done":..., "be_active":...}}
positions = {s: None for s in SYMBOLS}

# ===== Indicators =====
def compute_bb_cci(df):
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

# ===== Signals (5m) =====
def entry_signal_live(d):
    if len(d) < max(BB_N, CCI_N)+2: return None
    last, prev = d.iloc[-1], d.iloc[-2]
    long_setup = recent_touch(d["close"].iloc[:-1], d["bb_dn"].iloc[:-1], TOUCH_LOOKBACK, "below")
    long_cross = cross_up(prev["cci"], last["cci"], CCI_DN)
    long_band  = (prev["close"] <= prev["bb_dn"]) and (last["close"] > last["bb_dn"])
    if long_setup and long_cross and long_band:
        return {"side":"long", "reason":"BB lower reclaim + CCI↑-100"}
    short_setup = recent_touch(d["close"].iloc[:-1], d["bb_up"].iloc[:-1], TOUCH_LOOKBACK, "above")
    short_cross = cross_down(prev["cci"], last["cci"], CCI_UP)
    short_band  = (prev["close"] >= prev["bb_up"]) and (last["close"] < last["bb_up"])
    if short_setup and short_cross and short_band:
        return {"side":"short", "reason":"BB upper reject + CCI↓+100"}
    return None

def confirm_signal_close(d):
    return entry_signal_live(d)

# ===== Filter (1h) =====
def hourly_filter(sym, side):
    df = buf_1h[sym]
    if len(df) < CCI_N+2: return False
    last = df.iloc[-1]
    if side == "long" and last["cci"] > 0: return True
    if side == "short" and last["cci"] < 0: return True
    return False

# ===== Partial TP & BE =====
def pnl_pct(side, entry_px, cur_px):
    p = (cur_px - entry_px) / entry_px
    return p if side == "long" else -p

def check_partial_tp(sym, df, pos):
    last = df.iloc[-1]
    cur_px = last["close"]
    p = pnl_pct(pos["side"], pos["entry"], cur_px)
    # 기준: +0.3% 도달 시 부분익절
    if not pos["ptp_done"] and p >= 0.003:
        tg(f"[[PARTIAL-TP 50%]] {sym} {pos['side'].upper()} @ {cur_px:.2f}\n"
           f"PnL: {p*100:.2f}%")
        pos["ptp_done"] = True
        pos["be_active"] = True
        return True
    return False

def stop_out(sym, df, pos):
    last = df.iloc[-1]
    px = last["close"]
    entry = pos["entry"]
    # 본전 방어
    if pos["be_active"]:
        if (pos["side"]=="long" and px <= entry) or (pos["side"]=="short" and px >= entry):
            return {"type":"sl", "reason":"breakeven stop"}
    # 기본 SL/TP
    ex = exit_signal_bb_cci(df, pos["side"])
    return ex

def exit_signal_bb_cci(d, side):
    if len(d) < max(BB_N, CCI_N)+2: return None
    last, prev = d.iloc[-1], d.iloc[-2]
    if side == "long":
        if (last["close"] >= last["bb_mid"] and last["cci"] > 0):
            return {"type":"tp", "reason":"BB mid reached & CCI>0"}
        if (last["close"] < last["bb_dn"]) or cross_down(prev["cci"], last["cci"], CCI_DN):
            return {"type":"sl", "reason":"BB lower loss or CCI<-100"}
    else:
        if (last["close"] <= last["bb_mid"] and last["cci"] < 0):
            return {"type":"tp", "reason":"BB mid reached & CCI<0"}
        if (last["close"] > last["bb_up"]) or cross_up(prev["cci"], last["cci"], CCI_UP):
            return {"type":"sl", "reason":"BB upper break or CCI>+100"}
    return None

# ===== WebSocket Callback =====
def on_message(ws, message):
    global positions
    try:
        data = json.loads(message)
        if "data" not in data: return
        d = data["data"]
        if d.get("e") != "kline": return
        k = d["k"]
        sym = d["s"]
        ct_ms = k["T"]
        o,h,l,c,v = map(float, (k["o"],k["h"],k["l"],k["c"],k["v"]))
        is_final = bool(k["x"])
        row = {"open":o,"high":h,"low":l,"close":c,"volume":v,
               "close_time": pd.to_datetime(ct_ms, unit="ms", utc=True)}

        interval = k["i"]
        if interval == "5m":
            df = buf_5m[sym]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df = compute_bb_cci(df).tail(MAX_KEEP)
            buf_5m[sym] = df

            # EXIT 체크
            if positions[sym]:
                ex = stop_out(sym, df, positions[sym])
                if ex:
                    tg(f"[EXIT-{ex['type'].upper()}] {sym} {positions[sym]['side'].upper()} @ {c:.2f}\n"
                       f"{datetime.now(timezone.utc).astimezone(KST)}\nreason: {ex['reason']}")
                    positions[sym] = None
                    return
                else:
                    check_partial_tp(sym, df, positions[sym])

            # ENTRY 체크
            sig = entry_signal_live(df)
            if sig and is_final and positions[sym] is None:
                if hourly_filter(sym, sig["side"]):
                    positions[sym] = {
                        "side": sig["side"],
                        "entry": c,
                        "entry_time": row["close_time"],
                        "ptp_done": False,
                        "be_active": False
                    }
                    tg(f"[CONFIRM] {sym} {sig['side'].upper()} @ {c:.2f}\n"
                       f"{datetime.now(timezone.utc).astimezone(KST)}\nreason: {sig['reason']} (1h OK)")
                else:
                    tg(f"[IGNORE] {sym} {sig['side'].upper()} @ {c:.2f}\n"
                       f"{datetime.now(timezone.utc).astimezone(KST)}\nreason: {sig['reason']} (1h FAIL)")

        elif interval == "1h":
            df = buf_1h[sym]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df = compute_bb_cci(df).tail(MAX_KEEP)
            buf_1h[sym] = df

    except Exception as e:
        print("on_message error:", e)

def on_open(ws): tg("Multi-TF BB+CCI Scanner started (5m entry + 1h filter, partial TP+BE).")
def on_error(ws, error): print("ws error:", error)
def on_close(ws, code, msg): print("ws closed:", code, msg)

def run_ws():
    ws = WebSocketApp(WS_URL, on_open=on_open, on_message=on_message,
                      on_error=on_error, on_close=on_close)
    ws.run_forever(ping_interval=30, ping_timeout=10)

if __name__ == "__main__":
    tg("Launching multi-TF BB+CCI (5m entry + 1h filter, partial TP+BE, multi-position)…")
    run_ws()
