#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OKX USDT Perpetual Futures (BTC/USDT:USDT) 자동매매 봇

전략 요약:

[롱 전략]
- 조건:
    1) MA50 < MA200
    2) MA50(i) > MA50(i-1) (MA50 우상향)
    3) close(i) > MA50(i)
- 진입: 위 조건 만족 & 포지션 없을 때, 다음 봉 시가에 시장가 롱 진입
- 손절: 진입가 -0.5% (조건부 스탑마켓, reduceOnly)
- 익절: MA50이 MA200을 위로 골든크로스할 때 시장가 전량 청산

[숏 전략 - LH 필터]
- 조건:
    1) MA50 > MA200
    2) MA50(i) < MA50(i-1) (MA50 우하향)
    3) close(i) < MA50(i)
    4) Lower High 필터:
       - high(i) < high(i-1)
       - high(i-1) > high(i-2)
- 진입: 위 조건 만족 & 포지션 없을 때, 다음 봉 시가에 시장가 숏 진입
- 손절: 진입가 +0.5% (조건부 스탑마켓, reduceOnly)
- 익절: MA50이 MA200을 아래로 데드크로스할 때 시장가 전량 청산

레버리지: 6배 (cross, net mode)
계좌 equity 100% 기준으로 포지션 크기 계산
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

SYMBOL = "BTC/USDT:USDT"   # OKX USDT 무기한
TIMEFRAME = "5m"

MA_SHORT = 50
MA_LONG = 200

STOP_PCT = 0.005      # 0.5% 손절
LEVERAGE = 6          # 6배 레버리지
LOOP_INTERVAL = 5     # 루프 주기(초)

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
            "defaultSettle": "usdt",
        },
    })

    # 🔹 Demo(모의거래) 환경이면 켜기
    exchange.set_sandbox_mode(True)

    # 포지션 모드: net
    try:
        exchange.set_position_mode(hedged=False)
        logging.info("포지션 모드: net 설정 완료")
    except Exception as e:
        logging.warning(f"포지션 모드 설정 실패 (무시 가능): {e}")

    # 레버리지 / 마진모드 설정
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
    """선물(USDT-M) 계좌에서 USDT equity 추정."""
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
    """
    if entry_price <= 0 or equity_total <= 0:
        return 0.0

    notional = equity_total * LEVERAGE
    amount = notional / entry_price

    # BTC 수량 소수점 자리 조정 (0.001 단위 내림)
    amount = math.floor(amount * 1000) / 1000
    return max(amount, 0.0)


def get_current_position(exchange, symbol):
    """
    OKX 선물 포지션 조회.
    - 리턴: (has_position, side, size, entry_price)
    """
    try:
        positions = exchange.fetch_positions([symbol])
    except Exception as e:
        logging.warning(f"포지션 조회 실패: {e}")
        return False, None, 0.0, None

    for p in positions:
        if p.get("symbol") != symbol:
            continue

        contracts = float(p.get("contracts") or 0)
        side = None
        if contracts > 0:
            side = "long"
        elif contracts < 0:
            side = "short"

        entry_price = float(p.get("entryPrice") or 0)

        if side is not None and abs(contracts) > 0:
            return True, side, abs(contracts), entry_price

    return False, None, 0.0, None


# ============== 전략 조건 함수들 ============== #

def check_long_entry(prev, curr):
    """롱 진입 조건."""
    if any(pd.isna([prev["ma50"], prev["ma200"], curr["ma50"], curr["ma200"]])):
        return False
    return (
        (curr["ma50"] < curr["ma200"]) and
        (curr["ma50"] > prev["ma50"]) and
        (curr["close"] > curr["ma50"])
    )


def check_short_entry_lh(prev2, prev, curr):
    """숏 진입 조건 + Lower High 필터."""
    if any(pd.isna([prev["ma50"], prev["ma200"], curr["ma50"], curr["ma200"]])):
        return False

    base = (
        (curr["ma50"] > curr["ma200"]) and
        (curr["ma50"] < prev["ma50"]) and
        (curr["close"] < curr["ma50"])
    )

    lh = (curr["high"] < prev["high"]) and (prev["high"] > prev2["high"])

    return base and lh


def check_long_tp(prev, curr):
    """롱 익절: MA50 / MA200 골든크로스."""
    if any(pd.isna([prev["ma50"], prev["ma200"], curr["ma50"], curr["ma200"]])):
        return False
    return (prev["ma50"] <= prev["ma200"]) and (curr["ma50"] > curr["ma200"])


def check_short_tp(prev, curr):
    """숏 익절: MA50 / MA200 데드크로스."""
    if any(pd.isna([prev["ma50"], prev["ma200"], curr["ma50"], curr["ma200"]])):
        return False
    return (prev["ma50"] >= prev["ma200"]) and (curr["ma50"] < curr["ma200"])


# ============== 메인 루프 ============== #

def main():
    exchange = init_exchange()
    logging.info("OKX 롱/숏 자동매매 봇 시작")

    in_position = False
    pos_side = None            # "long" or "short"
    entry_price = None
    position_size = 0.0
    stop_price = None
    stop_order_id = None
    entry_time = None
    last_signal_candle_ts = None

    while True:
        try:
            # --- 캔들/지표 업데이트 --- #
            df = fetch_ohlcv_df(exchange, SYMBOL, TIMEFRAME, limit=MA_LONG + 10)
            if df is None or df.empty:
                logging.warning("캔들 데이터를 가져오지 못했습니다.")
                time.sleep(LOOP_INTERVAL)
                continue

            df = calculate_indicators(df)
            if len(df) < MA_LONG + 3:
                logging.info("MA 계산에 필요한 캔들이 부족합니다. 대기.")
                time.sleep(LOOP_INTERVAL)
                continue

            prev2 = df.iloc[-4]
            prev = df.iloc[-3]
            curr = df.iloc[-2]
            curr_ts = int(curr["ts"])

            # --- 실제 포지션 상태 동기화 --- #
            has_pos, exch_side, exch_size, exch_entry = get_current_position(exchange, SYMBOL)

            if not has_pos:
                if in_position:
                    logging.info("거래소 포지션이 사라짐 → 로컬 상태 초기화 (스탑로스 or 수동 청산)")
                in_position = False
                pos_side = None
                position_size = 0.0
                entry_price = None
                stop_price = None
                stop_order_id = None
            else:
                in_position = True
                pos_side = exch_side
                position_size = exch_size
                if exch_entry > 0:
                    entry_price = exch_entry

            # ---------------- 포지션 있는 경우: 익절만 관리 ---------------- #
            if in_position:
                if pos_side == "long":
                    if check_long_tp(prev, curr):
                        logging.info("[TP LONG] MA50/MA200 골든크로스 → 시장가 롱 익절")
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
                            logging.info(f"롱 익절 주문 체결: {order}")
                        except Exception as e:
                            logging.error(f"롱 익절 주문 실패: {e}")

                        if stop_order_id is not None:
                            try:
                                exchange.cancel_order(stop_order_id, SYMBOL)
                                logging.info(f"롱 스탑 주문 취소: {stop_order_id}")
                            except Exception as e:
                                logging.warning(f"롱 스탑 취소 실패(이미 체결/취소됐을 수 있음): {e}")

                        in_position = False
                        pos_side = None
                        position_size = 0.0
                        entry_price = None
                        stop_price = None
                        stop_order_id = None
                        entry_time = None

                elif pos_side == "short":
                    if check_short_tp(prev, curr):
                        logging.info("[TP SHORT] MA50/MA200 데드크로스 → 시장가 숏 익절")
                        try:
                            order = exchange.create_order(
                                SYMBOL,
                                type="market",
                                side="buy",
                                amount=position_size,
                                params={
                                    "tdMode": "cross",
                                    "reduceOnly": True,
                                },
                            )
                            logging.info(f"숏 익절 주문 체결: {order}")
                        except Exception as e:
                            logging.error(f"숏 익절 주문 실패: {e}")

                        if stop_order_id is not None:
                            try:
                                exchange.cancel_order(stop_order_id, SYMBOL)
                                logging.info(f"숏 스탑 주문 취소: {stop_order_id}")
                            except Exception as e:
                                logging.warning(f"숏 스탑 취소 실패(이미 체결/취소됐을 수 있음): {e}")

                        in_position = False
                        pos_side = None
                        position_size = 0.0
                        entry_price = None
                        stop_price = None
                        stop_order_id = None
                        entry_time = None

            # ---------------- 포지션 없는 경우: 롱/숏 진입 체크 ---------------- #
            else:
                if last_signal_candle_ts is not None and curr_ts == last_signal_candle_ts:
                    # 같은 캔들에서 중복 진입 방지
                    pass
                else:
                    long_signal = check_long_entry(prev, curr)
                    short_signal = check_short_entry_lh(prev2, prev, curr)

                    # MA50<MA200와 MA50>MA200는 동시에 참일 수 없어서 충돌 X
                    if long_signal or short_signal:
                        free_eq, total_eq = fetch_futures_equity(exchange)
                        logging.info(f"USDT Equity (free={free_eq}, total={total_eq})")

                        est_entry_price = float(curr["close"])
                        amount = compute_order_size_futures(est_entry_price, total_eq)
                        if amount <= 0:
                            logging.warning("포지션 수량이 0 이하입니다. 진입 스킵.")
                        else:
                            try:
                                if long_signal:
                                    side = "buy"
                                    pos_side = "long"
                                    log_side = "LONG"
                                else:
                                    side = "sell"
                                    pos_side = "short"
                                    log_side = "SHORT"

                                logging.info(f"[ENTRY {log_side}] 진입 신호 발생")
                                order = exchange.create_order(
                                    SYMBOL,
                                    type="market",
                                    side=side,
                                    amount=amount,
                                    params={
                                        "tdMode": "cross",
                                    },
                                )
                                logging.info(f"{log_side} 진입 주문 체결: {order}")

                                in_position = True
                                position_size = amount
                                entry_time = datetime.now(timezone.utc)
                                entry_price = est_entry_price

                                # 손절 가격 계산
                                if pos_side == "long":
                                    stop_price = entry_price * (1.0 - STOP_PCT)
                                    sl_side = "sell"
                                else:
                                    stop_price = entry_price * (1.0 + STOP_PCT)
                                    sl_side = "buy"

                                # 조건부 스탑마켓 주문
                                try:
                                    sl_order = exchange.create_order(
                                        SYMBOL,
                                        type="market",
                                        side=sl_side,
                                        amount=position_size,
                                        params={
                                            "tdMode": "cross",
                                            "reduceOnly": True,
                                            "stopLossPrice": stop_price,
                                        },
                                    )
                                    stop_order_id = sl_order.get("id")
                                    logging.info(
                                        f"{log_side} 스탑로스 주문 생성: id={stop_order_id}, "
                                        f"트리거 가격={stop_price:.2f}"
                                    )
                                except Exception as e:
                                    logging.error(f"{log_side} 스탑로스 주문 생성 실패! 수동 확인 필요: {e}")
                                    stop_order_id = None

                                logging.info(
                                    f"{log_side} 진입가={entry_price:.2f}, 수량={position_size}, "
                                    f"스탑로스={stop_price:.2f} (레버리지 {LEVERAGE}x)"
                                )

                                last_signal_candle_ts = curr_ts

                            except Exception as e:
                                logging.error(f"{log_side} 진입 주문 실패: {e}")

            time.sleep(LOOP_INTERVAL)

        except Exception as e:
            logging.error(f"메인 루프 에러: {e}")
            time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    main()
