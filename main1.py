"""
BTC/ETH/SOL 모니터링용 시그널 알림 봇 (OKX Public API + Telegram)
- 실제 주문 없음. 텔레그램으로 매매 제안만 전송
- 전략: 4H 레짐(BB 20SMA+기울기) + 15m %B 필터 + 5m 밴드외→재진입 & CCI 역전
- TP/SL: 진입가 기준 ±10% (요청사항)
- 레버리지 안내: 50x ~ 100x (요청사항)
- 투자금 안내: 시드의 10% 사용 권장 (SEED_USDT 설정 시 금액/명목가 계산해서 표시)
"""

import os
import time
import json
import math
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import numpy as np

# =========================
# 환경 변수 / 상수
# =========================
BASE_URL = "https://www.okx.com"
HTTP_TIMEOUT = 10
LOOP_SLEEP_SEC = 300              # 5분마다 체크
COOLDOWN_MIN = 60                 # 동일 심볼/방향 알림 최소 간격(분)

# 텔레그램
TELEGRAM_TOKEN = os.getenv("OKX_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("OKX_TELEGRAM_CHAT_ID")

# 심볼 고정 (요청사항)
WATCH_SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]

# 레버리지/시드 안내 (요청사항)
MIN_LEVERAGE = 50
MAX_LEVERAGE = 100
try:
    SEED_USDT = float(os.getenv("SEED_USDT") or "nan")
    if not np.isfinite(SEED_USDT):
        SEED_USDT = None
except Exception:
    SEED_USDT = None

# 로컬 기준시 (KST)
KST = timezone(timedelta(hours=9))

# 내부 상태 (중복 알림 방지)
last_alert_at = {}        # key: (symbol, side) -> datetime
last_alert_bar = {}       # key: (symbol, side) -> pandas.Timestamp (5m 마지막 캔들 ts)

# =========================
# 유틸/텔레그램
# =========================
def send_telegram(message: str):
    prefix = "[OKX signal] "
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(prefix + message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": prefix + message}
    try:
        requests.post(url, data=payload, timeout=HTTP_TIMEOUT)
        print(prefix + message)
    except Exception as e:
        print("텔레그램 전송 실패:", e)

def format_price(val: float) -> str:
    if val is None or not np.isfinite(val):
        return "N/A"
    if val >= 100:
        return f"{val:,.2f}"
    elif val >= 1:
        return f"{val:,.4f}"
    elif val >= 0.01:
        return f"{val:,.6f}"
    elif val >= 0.0001:
        return f"{val:,.8f}"
    else:
        return f"{val:,.10f}"

# =========================
# OKX 퍼블릭 마켓
# =========================
def get_candles(symbol: str, bar: str, limit: int = 300) -> pd.DataFrame:
    url = f"{BASE_URL}/api/v5/market/candles?instId={symbol}&bar={bar}&limit={limit}"
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df.columns = ["ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"]
    df = df.iloc[::-1].reset_index(drop=True)
    for col in ["o", "h", "l", "c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
    return df

def get_instrument_meta(symbol: str):
    """tickSz 등 메타 조회 (가격 라운딩용)"""
    url = f"{BASE_URL}/api/v5/public/instruments?instType=SWAP&instId={symbol}"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None
        info = data[0]
        return {
            "tickSz": float(info.get("tickSz") or 0.0),
            "lotSz": float(info.get("lotSz") or 0.0),
            "minSz": float(info.get("minSz") or 0.0),
        }
    except Exception:
        return None

_meta_cache = {}
def get_tick(symbol: str) -> float:
    meta = _meta_cache.get(symbol)
    if not meta:
        meta = get_instrument_meta(symbol)
        if meta:
            _meta_cache[symbol] = meta
    return (meta or {}).get("tickSz", 0.0) if meta else 0.0

def quantize_price(x: float, tick: float) -> float:
    if tick <= 0 or not np.isfinite(x):
        return x
    precision = max(-int(math.floor(math.log10(tick))), 0)
    return round(round(x / tick) * tick, precision)

# =========================
# 지표
# =========================
def bollinger(df: pd.DataFrame, period: int = 20, mult: float = 2.0):
    mid = df["c"].rolling(period).mean()
    std = df["c"].rolling(period).std(ddof=0)
    ub = mid + mult * std
    lb = mid - mult * std
    percB = (df["c"] - lb) / (ub - lb)
    bandwidth = (ub - lb) / mid
    return mid, ub, lb, percB, bandwidth

def cci(df: pd.DataFrame, period: int = 20, c: float = 0.015):
    tp = (df["h"] + df["l"] + df["c"]) / 3.0
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    cci_series = (tp - ma) / (c * md)
    return cci_series

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["h"] - df["l"]
    high_close = (df["h"] - df["c"].shift()).abs()
    low_close = (df["l"] - df["c"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.bfill()

def slope(series: pd.Series, lookback: int = 5):
    if len(series) < lookback + 1:
        return 0.0
    return float(series.iloc[-1] - series.iloc[-1 - lookback])

# =========================
# 시그널 (BB + CCI)
# =========================
def generate_signal_bb_cci(symbol: str):
    """
    반환: (side, entry_price, last_5m_ts) 또는 (None, None, None)
    side: 'long' or 'short'
    """
    df_4h  = get_candles(symbol, "4H", 300)
    df_15m = get_candles(symbol, "15m", 300)
    df_5m  = get_candles(symbol, "5m", 300)

    if df_4h.empty or df_15m.empty or df_5m.empty:
        return None, None, None

    # 4H 레짐
    mid4, ub4, lb4, pB4, bw4 = bollinger(df_4h, 20, 2.0)
    sma4 = mid4
    up_regime   = (df_4h["c"].iloc[-1] > sma4.iloc[-1]) and (slope(sma4, 5) > 0)
    down_regime = (df_4h["c"].iloc[-1] < sma4.iloc[-1]) and (slope(sma4, 5) < 0)
    if not (up_regime or down_regime):
        return None, None, None

    # 15m 중기 확인
    mid15, ub15, lb15, pB15, bw15 = bollinger(df_15m, 20, 2.0)
    pB15_last = pB15.iloc[-1]
    bw15_last = bw15.iloc[-1]
    if not np.isfinite(pB15_last) or not np.isfinite(bw15_last):
        return None, None, None
    # 스퀴즈 회피: 너무 좁은 밴드면 패스
    if bw15_last < 0.015:
        return None, None, None

    # 5m 실행 신호
    mid5, ub5, lb5, pB5, bw5 = bollinger(df_5m, 20, 2.0)
    cci5 = cci(df_5m, 20)
    close = float(df_5m["c"].iloc[-1])
    last_ts = df_5m["ts"].iloc[-1]  # 마지막 5m 캔들 타임스탬프(UTC)

    # 최근 1~2봉 내 외밴드 터치 확인
    last2_high = df_5m["h"].iloc[-2:].max()
    last2_low  = df_5m["l"].iloc[-2:].min()
    ub_now = float(ub5.iloc[-1])
    lb_now = float(lb5.iloc[-1])

    long_trigger = (
        up_regime and
        (0.20 <= float(pB15_last) <= 0.55) and
        (last2_low <= lb_now) and (close > lb_now) and
        (cci5.iloc[-2] < -100 and cci5.iloc[-1] > -100)
    )
    short_trigger = (
        down_regime and
        (0.45 <= float(pB15_last) <= 0.80) and
        (last2_high >= ub_now) and (close < ub_now) and
        (cci5.iloc[-2] > 100 and cci5.iloc[-1] < 100)
    )

    if long_trigger:
        return "long", close, last_ts
    if short_trigger:
        return "short", close, last_ts
    return None, None, None

# =========================
# 알림 메시지 구성
# =========================
def build_alert(symbol: str, side: str, entry_price: float) -> str:
    tick = get_tick(symbol) or 0.0

    if side == "long":
        sl = entry_price * 0.90      # -10%
        tp = entry_price * 1.10      # +10%
    else:
        sl = entry_price * 1.10
        tp = entry_price * 0.90

    # 틱 사이즈 정렬(메시지용)
    sl = quantize_price(sl, tick)
    tp = quantize_price(tp, tick)
    entry_q = quantize_price(entry_price, tick)

    # 시드 금액 기반 안내(선택)
    seed_line = "권장 투자금: 시드의 10%"
    if SEED_USDT is not None and SEED_USDT > 0:
        margin = SEED_USDT * 0.10
        notional_min = margin * MIN_LEVERAGE
        notional_max = margin * MAX_LEVERAGE
        seed_line = (
            f"권장 투자금: 시드의 10% ≈ {format_price(margin)} USDT\n"
            f"예상 포지션 명목가: {format_price(notional_min)} ~ {format_price(notional_max)} USDT"
        )

    msg = (
        f"📊 매매 신호 발생\n"
        f"━━━━━━━━━━━━━━━\n"
        f"종목: {symbol}\n"
        f"방향: {side.upper()}\n"
        f"진입가(참고): {format_price(entry_q)} USDT\n"
        f"손절가(SL): {format_price(sl)} USDT (-10%)\n"
        f"익절가(TP): {format_price(tp)} USDT (+10%)\n"
        f"권장 레버리지: {MIN_LEVERAGE}x ~ {MAX_LEVERAGE}x\n"
        f"{seed_line}"
    )
    return msg

# =========================
# 메인 루프
# =========================
if __name__ == "__main__":
    send_telegram("✅ 시그널 알림 봇 시작됨 (BTC/ETH/SOL, 주문 없음)")
    try:
        while True:
            try:
                now = datetime.now(KST)
                for symbol in WATCH_SYMBOLS:
                    side, price, last5_ts = generate_signal_bb_cci(symbol)
                    if not side:
                        continue

                    key = (symbol, side)
                    # 같은 5m 봉에서 중복 알림 방지
                    prev_bar = last_alert_bar.get(key)
                    if prev_bar is not None and pd.Timestamp(last5_ts) == prev_bar:
                        continue

                    # 쿨다운(분) 체크
                    prev_time = last_alert_at.get(key)
                    if prev_time is not None:
                        minutes = (now - prev_time).total_seconds() / 60.0
                        if minutes < COOLDOWN_MIN:
                            continue

                    # 알림 전송
                    msg = build_alert(symbol, side, price)
                    send_telegram(msg)

                    # 상태 업데이트
                    last_alert_at[key] = now
                    last_alert_bar[key] = pd.Timestamp(last5_ts)

                time.sleep(LOOP_SLEEP_SEC)

            except Exception as loop_err:
                send_telegram(f"[루프 오류] {loop_err}")
                time.sleep(60)

    except KeyboardInterrupt:
        send_telegram("🛑 수동 종료됨")
    except Exception as e:
        send_telegram(f"[치명 오류] {e}")
        raise
