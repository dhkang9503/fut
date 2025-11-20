#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OKX USDT Perpetual Futures (BTC/USDT:USDT) 자동매매 봇

전략 요약:
- 차트: 5분봉
- 지표: MA50, MA200 (종가 기준 SMA)
- 진입 (롱만):
    1) MA50 < MA200
    2) MA50(i) > MA50(i-1)  → MA50 우상향
    3) 종가(i) > MA50(i)
    4) 포지션 없음
   → 다음에 시장가 롱 진입

- 포지션 크기:
    - 계좌 USDT Equity 100% 기준
    - 레버리지 6배 (cross)
    - notional = equity_total * 6
    - amount = notional / entry_price

- 손절:
    - 진입가 기준 -0.5% (entry_price * 0.995)
    - 진입 시점에 OKX에 조건부 스탑마켓 주문 걸어둠

- 익절:
    - MA50이 MA200을 골든크로스할 때
      시장가 전량 익절 + 스탑로스 주문 취소

⚠️ 반드시 Demo(모의거래)에서 먼저 테스트할 것!
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
            "defaultType": "swap",   # 선물/스왑
            "defaultSettle": "usdt"
        },
    })

    # 🔹 Demo(모의거래) 환경이면 꼭 켜기
    exchange.set_sandbox_mode(True)

    # 포지션 모드: net (롱/숏 합산)
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
    """
    선물(USDT-M) 계좌에서 USDT equity 추정.
    여기서는 단순히 balance['USDT']['total'] 사용.
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
    """
    if entry_price <= 0 or equity_total <= 0:
        return 0.0

    notional = equity_total * LEVERAGE
    amount = notional / entry_price

    # BTC 수량 소수점 자리 조정 (OKX: 보통 0.001 단위 가능)
    amount = math.floor(amount * 1000) / 1000
    return max(amount, 0.0)


def get_current_price(exchange, symbol):
    """실시간 현재가(마지막 체결 가격) 가져오기."""
    ticker = exchange.fetch_ticker(symbol)
    last = ticker.get("last") or ticker.get("close")
    return float(last)


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


# ============== 전략 조건 ============== #

def check_entry_signal(prev, curr):
    """롱 진입 조건."""
    if any(pd.isna([prev["ma50"], prev["ma200"], curr["ma50"], curr["ma200"]])):
        return False

    cond1 = curr["ma50"] < curr["ma200"]       # 하락 구간
    cond2 = curr["ma50"] > prev["ma50"]        # MA50 우상향
    cond3 = curr["close"] > curr["ma50"]       # 종가 > MA50

    return cond1 and cond2 and cond3


def check_exit_signal(prev, curr):
    """익절 조건: MA50 / MA200 골든크로스."""
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
    stop_order_id = None   # 스탑로스 주문 ID

    while True:
        try:
            # --- 캔들/지표 업데이트 --- #
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

            # --- 거래소 실제 포지션 상태 동기화 --- #
            has_pos, pos_side, pos_size_exch, pos_entry_price_exch = get_current_position(exchange, SYMBOL)

            if not has_pos:
                if in_position:
                    logging.info("거래소 포지션이 사라짐 → 로컬 상태 초기화 (스탑로스 or 수동 청산)")
                in_position = False
                position_size = 0.0
                entry_price = None
                stop_price = None
                # 스탑 주문은 거래소에서 이미 체결/취소되었을 수 있음
                stop_order_id = None
            else:
                if pos_side == "long":
                    in_position = True
                    position_size = pos_size_exch
                    if pos_entry_price_exch > 0:
                        entry_price = pos_entry_price_exch
                else:
                    # 숏 포지션은 이 전략에선 사용하지 않음
                    in_position = False
                    position_size = 0.0
                    entry_price = None
                    stop_price = None
                    stop_order_id = None

            # ---------------- 포지션 있는 경우: 익절만 관리 ---------------- #
            if in_position:
                # 손절은 거래소 조건부 주문이 처리하므로 여기선 X
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

                    # 스탑로스 조건부 주문 취소 시도
                    if stop_order_id is not None:
                        try:
                            exchange.cancel_order(stop_order_id, SYMBOL)
                            logging.info(f"스탑로스 주문 취소: {stop_order_id}")
                        except Exception as e:
                            logging.warning(f"스탑로스 주문 취소 실패(이미 체결/취소됐을 수 있음): {e}")

                    in_position = False
                    position_size = 0.0
                    entry_price = None
                    stop_price = None
                    stop_order_id = None
                    entry_time = None

            # ---------------- 포지션 없는 경우: 진입 신호 체크 ---------------- #
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

                                in_position = True
                                position_size = amount
                                entry_time = datetime.now(timezone.utc)
                                # 간단하게 현재 캔들 종가를 진입가로 사용
                                entry_price = est_entry_price
                                stop_price = entry_price * (1.0 - STOP_PCT)

                                # 2) 조건부 스탑마켓 주문 생성 (reduceOnly)
                                try:
                                    sl_order = exchange.create_order(
                                        SYMBOL,
                                        type="market",          # 조건부 스탑마켓
                                        side="sell",
                                        amount=position_size,
                                        params={
                                            "tdMode": "cross",
                                            "reduceOnly": True,
                                            # 트리거 가격: stop_price
                                            "stopLossPrice": stop_price,
                                        },
                                    )
                                    stop_order_id = sl_order.get("id")
                                    logging.info(
                                        f"스탑로스 주문 생성: id={stop_order_id}, "
                                        f"트리거 가격={stop_price:.2f}"
                                    )
                                except Exception as e:
                                    logging.error(f"스탑로스 주문 생성 실패! 수동 확인 필요: {e}")
                                    stop_order_id = None

                                logging.info(
                                    f"진입가={entry_price:.2f}, 수량={position_size}, "
                                    f"스탑로스={stop_price:.2f} "
                                    f"(레버리지 {LEVERAGE}x, 계좌 100% 기준)"
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
