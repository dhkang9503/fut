#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_signal_bot.py

Binance USDT-M 선물( fapi )에서 BTC/ETH/XRP/SOL의 15m/4h 차트를 실시간 폴링하여
'진입 타점'이 감지되면 Telegram Bot으로 즉시 알림을 전송합니다.

- 4H 바이어스(롱): [EMA9>EMA21, MACD>Signal, Hist>0, RSI>=55] 중 3개 이상 참 (마지막 '종가 확정된' 4h 캔들 기준)
- 15m 숏-스캘프 타점: EMA9<EMA21 & MACD<Signal & Hist<0  (마지막 '종가 확정된' 15m 캔들 기준)
- 15m 롱 타점(옵션):  EMA9>EMA21 & MACD>Signal & Hist>0

알림 정책(기본):
- 롱/숏 모두 알림 (설정으로 끌 수 있음)
- 엔트리 체결 가정: '다음 15m 시가(next open)' — 알림 메시지에 명시

환경변수(필수):
  TELEGRAM_BOT_TOKEN : 텔레그램 봇 토큰
  TELEGRAM_CHAT_ID   : 수신 채팅 ID

실행:
  python live_signal_bot.py
"""

import os
import time
import math
import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import requests
import pandas as pd
from dateutil import tz

# =========================
# 설정
# =========================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]
FUTURES_BASE = "https://fapi.binance.com"
INTERVAL_15M = "15m"
INTERVAL_4H  = "4h"

# 텔레그램
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# 알림 토글
ALERT_SHORT = True
ALERT_LONG  = True   # False로 두면 숏만 알림

# 폴링 주기(초) — 10~15초 권장
POLL_SECONDS = 10

# 4H 바이어스 설정
BIAS_RSI_MIN = 55
BIAS_MIN_TRUE = 3  # 위 4개 조건 중 몇 개 이상이면 롱 바이어스로 간주

# 로컬 타임존 (표시용)
TZ_SEOUL = tz.gettz("Asia/Seoul")


# =========================
# 유틸 & 인디케이터
# =========================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rs = up.ewm(com=window-1, adjust=False).mean() / dn.ewm(com=window-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)

def macd(close: pd.Series, n_fast=12, n_slow=26, n_signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ef = close.ewm(span=n_fast, adjust=False).mean()
    es = close.ewm(span=n_slow, adjust=False).mean()
    m = ef - es
    sig = m.ewm(span=n_signal, adjust=False).mean()
    hist = m - sig
    return m, sig, hist

def now_ms() -> int:
    return int(time.time() * 1000)

def ts_to_str_kr(ms: int) -> str:
    dt = datetime.fromtimestamp(ms/1000, tz=timezone.utc).astimezone(TZ_SEOUL)
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


# =========================
# Binance API
# =========================
def get_klines(symbol: str, interval: str, limit: int = 200) -> List[list]:
    """fapi/v1/klines"""
    url = f"{FUTURES_BASE}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def klines_to_df(kl: List[list]) -> pd.DataFrame:
    """
    Binance futures kline format:
    [ open_time, open, high, low, close, volume,
      close_time, quote_volume, trades, taker_base, taker_quote, ignore ]
    """
    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","number_of_trades",
            "taker_buy_base_volume","taker_buy_quote_volume","ignore"]
    df = pd.DataFrame(kl, columns=cols)
    num_cols = ["open","high","low","close","volume",
                "quote_asset_volume","taker_buy_base_volume","taker_buy_quote_volume"]
    df[num_cols] = df[num_cols].astype(float)
    # ms -> UTC datetime index
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("close_time").sort_index()
    return df

def last_closed_index(df: pd.DataFrame) -> pd.Timestamp:
    """마지막으로 '종가 확정'된 캔들의 close_time 인덱스 반환(UTC)"""
    # Binance는 진행 중 캔들도 반환하므로, 현재 시각보다 close_time이 지난 것만 '확정'
    utcnow = pd.Timestamp.utcnow().tz_localize("UTC")
    closed = df.index[df.index <= utcnow]
    if len(closed) == 0:
        return None
    return closed[-1]


# =========================
# 신호 계산
# =========================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV dataframe (index=close_time) -> 인디케이터 추가"""
    out = df.copy()
    out["ema9"]  = ema(out["close"], 9)
    out["ema21"] = ema(out["close"], 21)
    m, s, h = macd(out["close"])
    out["macd"] = m; out["macd_signal"] = s; out["macd_hist"] = h
    out["rsi14"] = rsi(out["close"], 14)
    return out

def is_long_bias_4h(row: pd.Series) -> bool:
    """4H 바이어스(롱) 판단: 4개 중 3개 이상 참"""
    checks = [
        (row["ema9"]  > row["ema21"]),
        (row["macd"]  > row["macd_signal"]),
        (row["macd_hist"] > 0),
        (row["rsi14"] >= BIAS_RSI_MIN),
    ]
    return sum(bool(x) for x in checks) >= BIAS_MIN_TRUE

def short_trigger_15m(row: pd.Series) -> bool:
    return (row["ema9"] < row["ema21"]) and (row["macd"] < row["macd_signal"]) and (row["macd_hist"] < 0)

def long_trigger_15m(row: pd.Series) -> bool:
    return (row["ema9"] > row["ema21"]) and (row["macd"] > row["macd_signal"]) and (row["macd_hist"] > 0)


# =========================
# Telegram
# =========================
def tg_send(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram token/chat_id not set. Message would be:\n", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            print("[TG ERROR]", resp.text)
    except Exception as e:
        print("[TG EXCEPTION]", repr(e))


# =========================
# 메인 루프
# =========================
def main_loop():
    print("Starting live signal bot...")
    print("Symbols:", SYMBOLS)
    print("Alert SHORT:", ALERT_SHORT, " / LONG:", ALERT_LONG)
    last_alerted: Dict[Tuple[str, str], pd.Timestamp] = {}  # (symbol, side) -> last close_time we alerted on

    while True:
        loop_start = time.time()
        try:
            for sym in SYMBOLS:
                # 1) 15m klines
                k15 = get_klines(sym, INTERVAL_15M, limit=200)
                df15 = klines_to_df(k15)
                df15i = compute_indicators(df15)

                # 마지막 확정 15m 캔들(=진입 판단 캔들)
                last15 = last_closed_index(df15i)
                if last15 is None or last15 not in df15i.index:
                    continue

                row15 = df15i.loc[last15]

                # 2) 4h klines (바이어스는 '직전 확정 4h'로 계산 → 룩어헤드 방지)
                k4 = get_klines(sym, INTERVAL_4H, limit=200)
                df4 = klines_to_df(k4)
                df4i = compute_indicators(df4)

                last4_all = df4i.index[df4i.index <= last15]  # 15m 캔들 시점 이전/같은 4h 중 확정된 것
                if len(last4_all) == 0:
                    continue
                last4 = last4_all[-1]
                row4 = df4i.loc[last4]
                bias_long = is_long_bias_4h(row4)

                # 3) 트리거 판정(15m)
                candidates = []
                if ALERT_SHORT and short_trigger_15m(row15):
                    candidates.append("SHORT")
                if ALERT_LONG  and long_trigger_15m(row15):
                    candidates.append("LONG")

                # 4) 중복 알림 방지: 같은 심볼/사이드의 같은 15m close_time에서 한 번만
                for side in candidates:
                    key = (sym, side)
                    if last_alerted.get(key) == last15:
                        continue  # 이미 알림 전송

                    # 엔트리 체결 가정: 다음 15m 시가 (close_time + 1)
                    # 다음 캔들의 open_time = last15 + 1ms ~ 실제로는 다음 15m 오픈
                    # 메시지에 'next open'이라고 명시
                    open_price_next = None
                    # 안전하게 다음 봉이 존재하면 그 시가를 보여주고, 없으면 None
                    idxs = df15.index.tolist()
                    if last15 in idxs:
                        pos = idxs.index(last15)
                        if pos+1 < len(idxs):
                            next_idx = idxs[pos+1]
                            # 다음 봉의 open은 해당 row의 'open'(주의: 우리 df 인덱스는 close_time이라 open은 다음 행의 open이 맞음)
                            # df15의 open 컬럼은 각 행의 open 가격(그 행의 close_time에 해당하는 15m의 open이 아님) -> 주의 필요
                            # 하지만 klines_to_df에서 인덱스를 close_time으로 잡았기 때문에, '다음 행'의 'open'이 next open과 일치합니다.
                            open_price_next = float(df15.loc[next_idx, "open"])

                    msg = []
                    msg.append(f"📈 <b>{sym}</b> | 15m <b>{side}</b> signal")
                    msg.append(f"• 15m close: <code>{ts_to_str_kr(int(last15.timestamp()*1000))}</code>")
                    msg.append(f"• 4h bias: <b>{'LONG' if bias_long else 'NEUTRAL/SHORT'}</b>")
                    msg.append(f"• Entry: <i>next 15m open</i>{(' ≈ ' + str(open_price_next)) if open_price_next else ''}")
                    # 지표 간단 요약
                    msg.append(f"• 15m ema9/ema21: {row15['ema9']:.4f} / {row15['ema21']:.4f}")
                    msg.append(f"• 15m MACD / Sig / Hist: {row15['macd']:.5f} / {row15['macd_signal']:.5f} / {row15['macd_hist']:.5f}")
                    msg.append(f"• 4h RSI: {row4['rsi14']:.2f} | 4h ema9/ema21: {row4['ema9']:.2f}/{row4['ema21']:.2f}")
                    msg.append("—")
                    msg.append("체결/위험관리, TP/SL은 별도 로직에서 처리 예정")

                    tg_send("\n".join(msg))
                    last_alerted[key] = last15

        except Exception as e:
            print("[ERROR]", repr(e))

        # 폴링 간격 유지
        elapsed = time.time() - loop_start
        sleep_s = max(1.0, POLL_SECONDS - elapsed)
        time.sleep(sleep_s)


if __name__ == "__main__":
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("환경변수 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 를 설정하세요.")
        print("예) export TELEGRAM_BOT_TOKEN=xxxx; export TELEGRAM_CHAT_ID=123456789")
    main_loop()
