# -*- coding: utf-8 -*-
import os, json, time, threading, requests
import numpy as np
import pandas as pd
from binance import ThreadedWebsocketManager
from datetime import timezone

SYMBOL = "btcusdt"
TZ = "Asia/Seoul"
K = 2.0
THR_MIN, THR_MAX = 0.002, 0.03       # 0.2%~3%
ATR_LEN, ATR_WIN = 14, 30
TELEGRAM_BOT=os.getenv('TELEGRAM_BOT_TOKEN'), CHAT_ID=os.getenv('TELEGRAM_CHAT_ID')          # 선택

# ---- 상태 변수 ----
direction = 0       # +1 up-leg, -1 down-leg, 0 init
ext_price = None    # 현재 파동 극값
ext_time  = None
thr_pct   = None
hist_4h   = pd.DataFrame(columns=["open","high","low","close"])  # 종가로 계산

def ta_atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"]-df["low"]).abs(),
                    (df["high"]-pc).abs(),
                    (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def notify(msg):
    if not TELEGRAM_BOT: 
        print(msg); return
    try:
        requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT}/sendMessage",
                     params={"chat_id": CHAT_ID, "text": msg})
    except Exception as e:
        print("notify error:", e)

def on_4h_close(bar):
    """bar: dict with open,high,low,close, close_time(ms)"""
    global hist_4h, thr_pct, direction, ext_price, ext_time

    # 1) 시계열에 append
    ts = pd.to_datetime(bar["t"], unit="ms", utc=True)
    row = pd.Series({"open":float(bar["o"]), "high":float(bar["h"]),
                     "low":float(bar["l"]), "close":float(bar["c"])}, name=ts)
    hist_4h.loc[ts] = row

    # 2) 임계값 계산/업데이트
    if len(hist_4h) < ATR_WIN + ATR_LEN + 1:
        return
    atr = ta_atr(hist_4h, ATR_LEN)
    atr_pct = (atr / hist_4h["close"]).tail(ATR_WIN)
    thr_pct = float(np.clip(np.median(atr_pct) * K, THR_MIN, THR_MAX))

    close = row["close"]

    # 초기화
    if ext_price is None:
        ext_price, ext_time, direction = close, ts, 0
        return

    signal = None
    # 3) 파동/되돌림 판정 (종가 기준 → 재페인트 없음)
    if direction >= 0:
        # 상승파: 극대 갱신
        if close > ext_price:
            ext_price, ext_time = close, ts
        retrace = (ext_price - close) / ext_price
        if retrace >= thr_pct:
            signal = ("SHORT", ts, ext_time, ext_price)
            direction = -1
            ext_price, ext_time = close, ts
    else:
        # 하락파: 극소 갱신
        if close < ext_price:
            ext_price, ext_time = close, ts
        retrace = (close - ext_price) / ext_price
        if retrace >= thr_pct:
            signal = ("LONG", ts, ext_time, ext_price)
            direction = +1
            ext_price, ext_time = close, ts

    if signal:
        side, sig_time, piv_time, piv_px = signal
        kst = pd.Timestamp(sig_time, tz="UTC").tz_convert(TZ)
        msg = (f"[4H ZigZag] {side} 확정\n"
               f"신호시각: {kst}\n"
               f"Pivot@{piv_px:.2f}\n"
               f"thr≈{thr_pct*100:.2f}%")
        notify(msg)
        print(msg)

def start_ws():
    twm = ThreadedWebsocketManager()
    twm.start()
    msg = "start"
    notify(msg)

    # 4시간봉 Kline 구독
    def handle_4h(msg):
        if msg.get("e") != "kline": return
        k = msg["k"]
        if not k["x"]:   # bar close가 아니면 무시
            return
        bar = {"t": k["T"], "o": k["o"], "h": k["h"], "l": k["l"], "c": k["c"]}
        on_4h_close(bar)

    twm.start_kline_socket(callback=handle_4h, symbol=SYMBOL, interval="4h")

    # 블로킹 유지
    while True:
        time.sleep(60)

if __name__ == "__main__":
    start_ws()
