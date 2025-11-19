#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OKX USDT Perpetual Futures (BTC/USDT:USDT) 자동매매 봇

전략 요약:
- 차트: 5분봉
- 지표: MA50, MA200 (종가 기준 단순이동평균)

- 진입 (롱만):
    1) MA50 < MA200  (중기 하락 구간)
    2) MA50(i) > MA50(i-1)  → MA50 우상향
    3) 종가(i) > MA50(i)
    4) 포지션 없음
   → 위 조건이 "막 닫힌 5분봉"에서 성립하면, 다음에 시장가 롱 진입

- 포지션 크기:
    - 계좌 USDT Equity 100% 기준
    - 레버리지 6배 고정
    - notional = equity_total * 6
    - amount = notional / entry_price

- 손절:
    - 진입가 기준 -0.5% 미만 (entry_price * 0.995 이하)에서
      시장가 전량 손절

- 익절:
    - MA50이 MA200을 골든크로스 (위로 돌파)하면
      시장가 전량 익절

⚠️ 주의:
- 반드시 OKX Demo(샌드박스) / 소액으로 먼저 테스트
- API 키는 Demo/실계정 환경에 맞게 별도 발급해야 함
"""

import os
import time
import math
import logging
from datetime import datetime, timezone

import ccxt
import pandas as pd


# ============== 설정값 ============== #

API_KEY = os.getenv("OKX_API_KEY", "")
API_SECRET = os.getenv("OKX_API_SECRET", "")
API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE", "")

# 선물 심볼 (OKX USDT 무기한: BTC/USDT:USDT)
SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "5m"

# 전략 파라미터
MA_SHORT = 50
MA_LONG = 200

STOP_PCT = 0.005      # 0.5% 손절
LEVERAGE = 6          # 6배 레버리지 고정
LOOP_INTERVAL = 5     # 몇 초마다 루프 돌릴지 (초 단위)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ============== OKX 초기화 ============== #

def init_exchange():
    exchange = ccxt.okx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "password": API_PASSPHRASE,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",  # 선물/스왑
        },
    })

    # 🔹 데모 트레이딩(모의 거래) 사용할 때는 sandbox 모드를 켜야 함
    exchange.set_sandbox_mode(True)

    # 포지션 모드: net (롱/숏 합산 모드)
    try:
        exchange.set_position_mode(hedged=False)
        logging.info("포지션 모드: net 설정 완료")
    except Exception as e:
        logging.warning(f"포지션 모드 설정 실패 (무시 가능): {e}")

    # 레버리지 / 마진모드 설정 (cross)
    try:
        exchange.set_leverage(LEVERAGE, SYMBOL, params={"mgnMode": "cross"})
        logging.info(f"레버리지 {LEVERAGE}배, cross 마진 설정 완료")
    except Exception as e:
        logging.warning(f"레버리지/마진 설정 실패 (무시 가능): {e}")

    return exchange


# ============== 유틸 함수들 ============== #

def fetch_ohlcv_df(exchange, symbol, timeframe, limit=300):
    """OHLCV 데이터를 pandas DataFrame으로 변환."""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        return None
    df = pd.DataFrame(
        ohlcv,
        columns=["ts", "open", "high", "low", "close", "volume"],
    )
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("dt", inplace=True)
    return df


def calculate_indicators(df: pd.DataFrame):
    """MA50, MA200 계산."""
    df["ma50"] = df["close"].rolling(MA_SHORT).mean()
    df["ma200"] = df["close"].rolling(MA_LONG).mean()
    return df


def get_last_closed_candles(df: pd.DataFrame):
    """
    마지막 캔들은 진행 중일 수 있으니,
    -3, -2 인덱스를 '완전히 닫힌 두 개의 캔들'로 사용.
    prev: 이전 캔들, curr: 현재 막 닫힌 캔들
    """
    if len(df) < MA_LONG + 3:
        return None, None
    prev = df.iloc[-3]
    curr = df.iloc[-2]
    return prev, curr


def fetch_futures_equity(exchange):
    """
    선물(USDT-M) 계좌에서 USDT equity 추정.
    단순화: fetch_balance()['USDT']['total'] 사용.
    실제로는 account 유형에 따라 세부 조정 필요할 수 있음.
    """
    balance = exchange.fetch_balance()
    usdt = balance.get("USDT", {})
    total = float(usdt.get("total", 0.0))
    free = float(usdt.get("free", 0.0))
    return free, total


def compute_order_size_futures(entry_price, equity_total):
    """
    계좌 equity 100%를 기준으로 6배 레버리지 포지션 크기 계산.

    notional = equity_total * LEVERAGE
    amount = notional / entry_price

    예)
    - equity_total = 1000 USDT
    - LEVERAGE = 6
    - notional = 6000 USDT
    - entry_price = 60000 USDT라면,
      amount = 6000 / 60000 = 0.1 BTC
    """
    if entry_price <= 0 or equity_total <= 0:
        return 0.0

    notional = equity_total * LEVERAGE
    amount = notional / entry_price

    # BTC 수량 소수점 자리 조정 (OKX 규칙에 맞게 0.001 단위로 내림)
    amount = math.floor(amount * 1000) / 1000
    return max(amount, 0.0)


def get_current_price(exchange, symbol):
    """실시간 현재가(마지막 체결 가격) 가져오기."""
    ticker = exchange.fetch_ticker(symbol)
    last = ticker.get("last")
    if last is None:
        last = ticker.get("close")
    return float(last)


# ============== 전략 조건 함수들 ============== #

def check_entry_signal(prev, curr):
    """
    진입 조건:
    - MA50 < MA200 (하락 구간)
    - MA50 우상향 (현재 MA50 > 이전 MA50)
    - 종가 > MA50
    """
    if any(pd.isna([prev["ma50"], prev["ma200"], curr["ma50"], curr["ma200"]])):
        return False

    cond1 = curr["ma50"] < curr["ma200"]
    cond2 = curr["ma50"] > prev["ma50"]
    cond3 = curr["close"] > curr["ma50"]

    return cond1 and cond2 and cond3


def check_exit_signal(prev, curr):
    """
    익절 조건:
    - 직전: MA50 <= MA200
    - 현재: MA50 > MA200 (골든크로스)
    """
    if any(pd.isna([prev["ma50"], prev["ma200"], curr["ma50"], curr["ma200"]])):
        return False

    was_below = prev["ma50"] <= prev["ma200"]
    now_above = curr["ma50"] > curr["ma200"]
    return was_below and now_above


# ============== 메인 루프 ============== #

def main():
    exchange = init_exchange()
    logging.info("OKX 선물 자동매매 봇 시작")

    in_position = False
    entry_price = None
    position_size = 0.0
    stop_price = None
    entry_time = None
    last_signal_candle_ts = None  # 같은 캔들 재진입 방지용

    while True:
        try:
            df = fetch_ohlcv_df(exchange, SYMBOL, TIMEFRAME, limit=MA_LONG + 10)
            if df is None or df.empty:
                logging.warning("캔들 데이터를 가져오지 못했습니다.")
                time.sleep(LOOP_INTERVAL)
                continue

            df = calculate_indicators(df)
            prev, curr = get_last_closed_candles(df)
            if prev is None or curr is None:
                logging.info("MA 계산에 필요한 캔들이 부족합니다. 대기.")
                time.sleep(LOOP_INTERVAL)
                continue

            curr_ts = int(curr["ts"])
            current_price = get_current_price(exchange, SYMBOL)

            # ---------------- 포지션 있는 경우: 손절 / 익절 ---------------- #
            if in_position:
                # 1) 손절: 현재가가 stop_price 이하이면 시장가 전량 청산
                if stop_price is not None and current_price <= stop_price:
                    logging.info(
                        f"[STOP] 현재가 {current_price:.2f} <= 스탑 {stop_price:.2f} → 시장가 손절"
                    )
                    try:
                        order = exchange.create_order(
                            SYMBOL,
                            type="market",
                            side="sell",
                            amount=position_size,
                            params={
                                "tdMode": "cross",
                                "reduceOnly": True,
                            },
                        )
                        logging.info(f"손절 주문 체결: {order}")
                    except Exception as e:
                        logging.error(f"손절 주문 실패: {e}")

                    in_position = False
                    entry_price = None
                    position_size = 0.0
                    stop_price = None
                    entry_time = None

                else:
                    # 2) 익절: MA50 / MA200 골든크로스
                    if check_exit_signal(prev, curr):
                        logging.info("[TP] MA50/MA200 골든크로스 → 시장가 익절")
                        try:
                            order = exchange.create_order(
                                SYMBOL,
                                type="market",
                                side="sell",
                                amount=position_size,
                                params={
                                    "tdMode": "cross",
                                    "reduceOnly": True,
                                },
                            )
                            logging.info(f"익절 주문 체결: {order}")
                        except Exception as e:
                            logging.error(f"익절 주문 실패: {e}")

                        in_position = False
                        entry_price = None
                        position_size = 0.0
                        stop_price = None
                        entry_time = None

            # ---------------- 포지션 없는 경우: 진입 신호 체크 ---------------- #
            else:
                if last_signal_candle_ts is not None and curr_ts == last_signal_candle_ts:
                    # 이미 이 캔들에서 처리함
                    pass
                else:
                    if check_entry_signal(prev, curr):
                        logging.info("[ENTRY] 진입 신호 발생")

                        free_eq, total_eq = fetch_futures_equity(exchange)
                        logging.info(f"USDT Equity (free={free_eq}, total={total_eq})")

                        est_entry_price = float(curr["close"])
                        amount = compute_order_size_futures(est_entry_price, total_eq)
                        if amount <= 0:
                            logging.warning("포지션 수량이 0 이하입니다. 진입 스킵.")
                        else:
                            try:
                                order = exchange.create_order(
                                    SYMBOL,
                                    type="market",
                                    side="buy",
                                    amount=amount,
                                    params={
                                        "tdMode": "cross",   # 교차 마진
                                    },
                                )
                                logging.info(f"진입 주문 체결: {order}")

                                # 실제 체결 평균가를 쓰고 싶다면:
                                # entry_price = float(order.get("average", est_entry_price))
                                entry_price = est_entry_price
                                position_size = amount
                                in_position = True
                                entry_time = datetime.now(timezone.utc)

                                # 진입가 기준 -0.5% 손절
                                stop_price = entry_price * (1.0 - STOP_PCT)
                                logging.info(
                                    f"진입가={entry_price:.2f}, 수량={position_size}, "
                                    f"스탑로스={stop_price:.2f} (레버리지 {LEVERAGE}x, "
                                    f"계좌 100% 기준 진입)"
                                )

                                last_signal_candle_ts = curr_ts

                            except Exception as e:
                                logging.error(f"진입 주문 실패: {e}")

            time.sleep(LOOP_INTERVAL)

        except Exception as e:
            logging.error(f"메인 루프 에러: {e}")
            time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    main()
