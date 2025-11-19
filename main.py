#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OKX USDT Perpetual Futures (BTC/USDT:USDT) 자동매매 봇

- 계좌 Equity 100% 기준, 레버리지 6배
- 진입가 기준 -0.5% 손절
- 포지션 진입 시 거래소에 스탑로스 주문을 함께 걸어둠
- MA50/MA200 골든크로스 시 시장가 익절 + 스탑로스 주문 취소
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

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "5m"

MA_SHORT = 50
MA_LONG = 200

STOP_PCT = 0.005      # 0.5% 손절
LEVERAGE = 6
LOOP_INTERVAL = 5

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
            "defaultType": "swap",
        },
    })

    # 데모 트레이딩이면 켜기
    exchange.set_sandbox_mode(True)

    try:
        exchange.set_position_mode(hedged=False)
        logging.info("포지션 모드: net 설정 완료")
    except Exception as e:
        logging.warning(f"포지션 모드 설정 실패 (무시 가능): {e}")

    try:
        exchange.set_leverage(LEVERAGE, SYMBOL, params={"mgnMode": "cross"})
        logging.info(f"레버리지 {LEVERAGE}배, cross 마진 설정 완료")
    except Exception as e:
        logging.warning(f"레버리지/마진 설정 실패 (무시 가능): {e}")

    return exchange


# ============== 유틸 함수들 ============== #

def fetch_ohlcv_df(exchange, symbol, timeframe, limit=300):
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
    df["ma50"] = df["close"].rolling(MA_SHORT).mean()
    df["ma200"] = df["close"].rolling(MA_LONG).mean()
    return df


def get_last_closed_candles(df: pd.DataFrame):
    if len(df) < MA_LONG + 3:
        return None, None
    prev = df.iloc[-3]
    curr = df.iloc[-2]
    return prev, curr


def fetch_futures_equity(exchange):
    balance = exchange.fetch_balance()
    usdt = balance.get("USDT", {})
    total = float(usdt.get("total", 0.0))
    free = float(usdt.get("free", 0.0))
    return free, total


def compute_order_size_futures(entry_price, equity_total):
    if entry_price <= 0 or equity_total <= 0:
        return 0.0
    notional = equity_total * LEVERAGE
    amount = notional / entry_price
    amount = math.floor(amount * 1000) / 1000  # BTC 0.001 단위 내림
    return max(amount, 0.0)


def get_current_price(exchange, symbol):
    ticker = exchange.fetch_ticker(symbol)
    last = ticker.get("last") or ticker.get("close")
    return float(last)


# ============== 전략 조건 ============== #

def check_entry_signal(prev, curr):
    if any(pd.isna([prev["ma50"], prev["ma200"], curr["ma50"], curr["ma200"]])):
        return False
    cond1 = curr["ma50"] < curr["ma200"]
    cond2 = curr["ma50"] > prev["ma50"]
    cond3 = curr["close"] > curr["ma50"]
    return cond1 and cond2 and cond3


def check_exit_signal(prev, curr):
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
    last_signal_candle_ts = None
    stop_order_id = None   # ✅ 스탑로스 주문 ID 저장

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

            # ---------------- 포지션 보유 중: 익절만 체크 (손절은 거래소 스탑 주문이 담당) ---------------- #
            if in_position:
                # 골든크로스 익절
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

                    # ✅ 걸려 있던 스탑로스 주문 취소
                    if stop_order_id is not None:
                        try:
                            exchange.cancel_order(stop_order_id, SYMBOL)
                            logging.info(f"스탑로스 주문 취소: {stop_order_id}")
                        except Exception as e:
                            logging.warning(f"스탑로스 주문 취소 실패(이미 체결/취소됐을 수 있음): {e}")

                    in_position = False
                    entry_price = None
                    position_size = 0.0
                    stop_price = None
                    stop_order_id = None
                    entry_time = None

            # ---------------- 포지션 없을 때: 진입 신호 ---------------- #
            else:
                if last_signal_candle_ts is not None and curr_ts == last_signal_candle_ts:
                    # 같은 캔들에서 중복 진입 방지
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
                                # 1) 시장가 롱 진입
                                order = exchange.create_order(
                                    SYMBOL,
                                    type="market",
                                    side="buy",
                                    amount=amount,
                                    params={
                                        "tdMode": "cross",
                                    },
                                )
                                logging.info(f"진입 주문 체결: {order}")

                                # 체결가 근사치 (원하면 order['average'] 참고)
                                entry_price = est_entry_price
                                position_size = amount
                                in_position = True
                                entry_time = datetime.now(timezone.utc)

                                # 손절 가격 계산
                                stop_price = entry_price * (1.0 - STOP_PCT)

                                # 2) ✅ 스탑로스 트리거 주문(감소 전용) 걸기
                                # OKX + ccxt: price = 실제 청산 가격, stopLossPrice = 트리거 가격
                                # 여기서는 둘 다 stop_price로 설정 (트리거=실행 모두 같은 가격)
                                try:
                                    sl_order = exchange.create_order(
                                        SYMBOL,
                                        type="limit",          # 트리거 발동 시 limit sell
                                        side="sell",
                                        amount=position_size,
                                        price=stop_price,
                                        params={
                                            "tdMode": "cross",
                                            "reduceOnly": True,
                                            "stopLossPrice": stop_price,  # ccxt 통합 스탑로스 파라미터
                                        },
                                    )
                                    stop_order_id = sl_order.get("id")
                                    logging.info(
                                        f"스탑로스 주문 생성: id={stop_order_id}, "
                                        f"트리거/체결가={stop_price:.2f}"
                                    )
                                except Exception as e:
                                    logging.error(f"스탑로스 주문 생성 실패! 백업으로 수동 모니터링 필요: {e}")
                                    stop_order_id = None

                                logging.info(
                                    f"진입가={entry_price:.2f}, 수량={position_size}, "
                                    f"스탑로스={stop_price:.2f} (레버리지 {LEVERAGE}x, 계좌 100% 기준)"
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
