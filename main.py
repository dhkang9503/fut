#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OKX USDT Perpetual Futures 자동매매 봇 (멀티심볼: BTC + XRP + DOGE)

전략 요약:

[공통 환경]
- 거래소: OKX
- 심볼: BTC/USDT:USDT, XRP/USDT:USDT, DOGE/USDT:USDT (USDT 무기한)
- 타임프레임: 5분봉
- 포지션: 세 심볼 통틀어 항상 1개만 보유
- 포지션 크기: 손절 도달 시 계좌의 약 3% 손실로 고정 (ma_gap 기반 동적 스탑)

[롱 전략]
- 조건 (최근 닫힌 캔들 기준):
    1) MA50 < MA200
    2) MA50(i) > MA50(i-1)  (MA50 우상향)
    3) close(i) > MA50(i)
- 진입: 위 조건 만족 & 무포지션일 때, 다음 봉 시가에 시장가 롱 진입
- 손절: ma_gap = |MA50 - MA200| / close (진입 직전 캔들 기준, 0.3%~2%로 조정)
         stop_pct = ma_gap, stop = "실제 진입가" * (1 - stop_pct)
         포지션 크기는 stop까지 손실이 계좌의 3%가 되도록 계산
- 익절: MA50이 MA200을 위로 골든크로스할 때 시장가 전량 익절

[숏 전략 - LH 필터]
- 조건:
    1) MA50 > MA200
    2) MA50(i) < MA50(i-1)  (MA50 우하향)
    3) close(i) < MA50(i)
    4) Lower High (LH) 필터:
       - high(i) < high(i-1)
       - high(i-1) > high(i-2)
- 진입: 위 조건 만족 & 무포지션일 때, 다음 봉 시가에 시장가 숏 진입
- 손절: ma_gap = |MA50 - MA200| / close (진입 직전 캔들 기준, 0.3%~2%로 조정)
         stop_pct = ma_gap, stop = "실제 진입가" * (1 + stop_pct)
         포지션 크기는 stop까지 손실이 계좌의 3%가 되도록 계산
- 익절: MA50이 MA200을 아래로 데드크로스할 때 시장가 전량 익절
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

SYMBOLS = [
    "BTC/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
]

TIMEFRAME = "5m"

MA_SHORT = 50
MA_LONG = 200

# 리스크 및 레버리지 관련
RISK_PER_TRADE = 0.03      # 손절 도달 시 계좌의 3% 손실 목표
MAX_LEVERAGE   = 10        # 최대 레버리지(실제 포지션 노출 / equity 상한)

# ma_gap 기반 최소/최대 손절 폭 (비율)
MIN_STOP_PCT = 0.003       # 0.3%
MAX_STOP_PCT = 0.02        # 2.0%

LOOP_INTERVAL = 5          # 루프 주기(초)

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

    # 🔹 데모(모의거래)면 켜기
    exchange.set_sandbox_mode(True)

    # 마켓 정보 미리 로드
    exchange.load_markets()

    # 포지션 모드: net
    try:
        exchange.set_position_mode(hedged=False)
        logging.info("포지션 모드: net 설정 완료")
    except Exception as e:
        logging.warning(f"포지션 모드 설정 실패 (무시 가능): {e}")

    # 심볼별 레버리지 / 마진모드 설정
    for sym in SYMBOLS:
        try:
            # 여기서 설정하는 레버리지는 '최대 허용 레버리지' 느낌으로 사용
            exchange.set_leverage(MAX_LEVERAGE, sym, params={"mgnMode": "cross"})
            logging.info(f"{sym} 레버리지 {MAX_LEVERAGE}배, cross 마진 설정 완료")
        except Exception as e:
            logging.warning(f"{sym} 레버리지/마진 설정 실패 (무시 가능): {e}")

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
    """MA50, MA200 및 ma_gap 계산."""
    df["ma50"] = df["close"].rolling(MA_SHORT).mean()
    df["ma200"] = df["close"].rolling(MA_LONG).mean()
    df["ma_gap"] = (df["ma50"] - df["ma200"]).abs() / df["close"]
    return df


def fetch_futures_equity(exchange):
    """선물(USDT-M) 계좌에서 USDT equity 추정."""
    balance = exchange.fetch_balance()
    usdt = balance.get("USDT", {})
    total = float(usdt.get("total", 0.0))
    free = float(usdt.get("free", 0.0))
    return free, total


def calc_ma_gap_pct_from_row(row):
    """
    ma_gap = |MA50 - MA200| / close
    MIN_STOP_PCT ~ MAX_STOP_PCT 사이로 클리핑.
    NaN/이상치면 기본값 1% 사용.
    """
    ma50 = row.get("ma50")
    ma200 = row.get("ma200")
    close = row.get("close")
    if any(pd.isna([ma50, ma200, close])) or close <= 0:
        return 0.01

    gap = abs(ma50 - ma200) / close
    if not math.isfinite(gap) or gap <= 0:
        return 0.01

    return max(MIN_STOP_PCT, min(MAX_STOP_PCT, float(gap)))


def compute_order_size_risk_based(exchange, symbol, entry_price, equity_total, stop_pct):
    """
    리스크 3% 고정 포지션 크기 계산.

    - risk_value      = equity_total * RISK_PER_TRADE
    - target_notional = risk_value / stop_pct
    - max_notional    = equity_total * MAX_LEVERAGE
    - notional        = min(target_notional, max_notional)
    - amount(contracts) = floor( notional / (entry_price * contract_size) )
    """
    if entry_price <= 0 or equity_total <= 0 or stop_pct <= 0:
        return 0.0, 0.0  # amount, effective_leverage

    risk_value = equity_total * RISK_PER_TRADE
    target_notional = risk_value / stop_pct
    max_notional = equity_total * MAX_LEVERAGE

    notional = min(target_notional, max_notional)

    market = exchange.market(symbol)
    contract_size = market.get("contractSize")
    if contract_size is None:
        info = market.get("info", {})
        contract_size = float(info.get("ctVal", 1))

    notional_per_contract = entry_price * contract_size
    if notional_per_contract <= 0:
        return 0.0, 0.0

    amount = notional / notional_per_contract
    amount = math.floor(amount)  # 정수 계약 수

    if amount <= 0:
        return 0.0, 0.0

    effective_leverage = (amount * notional_per_contract) / equity_total
    return amount, effective_leverage


def sync_position(exchange, symbols):
    """
    OKX 선물 포지션 조회.
    세 심볼 중 하나라도 포지션이 있을 경우:
    - 리턴: (has_position, symbol, side, size, entry_price)
    """
    try:
        positions = exchange.fetch_positions()
    except Exception as e:
        logging.warning(f"포지션 조회 실패: {e}")
        return False, None, None, 0.0, None

    active = []
    for p in positions:
        sym = p.get("symbol")
        if sym not in symbols:
            continue
        contracts = float(p.get("contracts") or 0)
        if abs(contracts) <= 0:
            continue
        side = "long" if contracts > 0 else "short"
        entry_price = float(p.get("entryPrice") or 0)
        active.append((sym, side, abs(contracts), entry_price))

    if len(active) == 0:
        return False, None, None, 0.0, None
    if len(active) > 1:
        logging.warning(f"여러 심볼에 동시에 포지션이 있습니다: {active} (전략은 1포지션만 가정)")
    sym, side, size, entry_price = active[0]
    return True, sym, side, size, entry_price


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
    """롱 익절: MA50 / MA200 골든크로스 이후 구간."""
    if any(pd.isna([prev["ma50"], prev["ma200"], curr["ma50"], curr["ma200"]])):
        return False
    return curr["ma50"] > curr["ma200"]


def check_short_tp(prev, curr):
    """숏 익절: MA50 / MA200 데드크로스 이후 구간."""
    if any(pd.isna([prev["ma50"], prev["ma200"], curr["ma50"], curr["ma200"]])):
        return False
    return curr["ma50"] < curr["ma200"]


# ============== 메인 루프 ============== #

def main():
    exchange = init_exchange()
    logging.info("OKX BTC+XRP+DOGE 롱/숏 자동매매 봇 시작 (ma_gap 기반 + 리스크 3% 고정)")

    in_position = False
    pos_symbol = None
    pos_side = None          # "long" or "short"
    entry_price = None
    position_size = 0.0
    stop_price = None
    stop_order_id = None
    entry_time = None
    last_signal_candle_ts = {}  # 심볼별 마지막 신호 캔들 ts

    while True:
        try:
            # --- 각 심볼별 캔들/지표 업데이트 --- #
            data = {}  # symbol -> (df, prev2, prev, curr)
            for sym in SYMBOLS:
                df = fetch_ohlcv_df(exchange, sym, TIMEFRAME, limit=MA_LONG + 10)
                if df is None or df.empty:
                    logging.warning(f"{sym} 캔들 데이터를 가져오지 못했습니다.")
                    continue
                df = calculate_indicators(df)
                if len(df) < MA_LONG + 3:
                    logging.info(f"{sym}: MA 계산에 필요한 캔들이 부족합니다.")
                    continue
                prev2 = df.iloc[-4]
                prev = df.iloc[-3]
                curr = df.iloc[-2]
                data[sym] = (df, prev2, prev, curr)

            if not data:
                logging.warning("어느 심볼에서도 유효한 데이터가 없습니다. 대기.")
                time.sleep(LOOP_INTERVAL)
                continue

            # --- 실제 포지션 상태 동기화 --- #
            has_pos, exch_sym, exch_side, exch_size, exch_entry = sync_position(exchange, SYMBOLS)

            if not has_pos:
                if in_position:
                    logging.info("거래소 포지션이 사라짐 → 로컬 상태 초기화 (스탑로스 or 수동 청산)")
                in_position = False
                pos_symbol = None
                pos_side = None
                position_size = 0.0
                entry_price = None
                stop_price = None
                stop_order_id = None
            else:
                in_position = True
                pos_symbol = exch_sym
                pos_side = exch_side
                position_size = exch_size
                if exch_entry > 0:
                    entry_price = exch_entry

            # ---------------- 포지션 있는 경우: 익절만 관리 ---------------- #
            if in_position:
                if pos_symbol not in data:
                    logging.warning(f"{pos_symbol} 데이터가 없어 익절 체크 불가. 대기.")
                else:
                    _, prev2, prev, curr = data[pos_symbol]
                    if pos_side == "long":
                        if check_long_tp(prev, curr):
                            logging.info(f"[TP LONG] {pos_symbol} 골든크로스 → 시장가 롱 익절")
                            try:
                                order = exchange.create_order(
                                    pos_symbol,
                                    type="market",
                                    side="sell",
                                    amount=position_size,
                                    params={
                                        "tdMode": "cross",
                                        "reduceOnly": True,
                                    },
                                )
                                logging.info(f"{pos_symbol} 롱 익절 주문 체결: {order}")
                            except Exception as e:
                                logging.error(f"{pos_symbol} 롱 익절 주문 실패: {e}")

                            if stop_order_id is not None:
                                try:
                                    exchange.cancel_order(stop_order_id, pos_symbol)
                                    logging.info(f"{pos_symbol} 롱 스탑 주문 취소: {stop_order_id}")
                                except Exception as e:
                                    logging.warning(f"{pos_symbol} 롱 스탑 취소 실패(이미 체결/취소됐을 수 있음): {e}")

                            in_position = False
                            pos_symbol = None
                            pos_side = None
                            position_size = 0.0
                            entry_price = None
                            stop_price = None
                            stop_order_id = None
                            entry_time = None

                    elif pos_side == "short":
                        if check_short_tp(prev, curr):
                            logging.info(f"[TP SHORT] {pos_symbol} 데드크로스 → 시장가 숏 익절")
                            try:
                                order = exchange.create_order(
                                    pos_symbol,
                                    type="market",
                                    side="buy",
                                    amount=position_size,
                                    params={
                                        "tdMode": "cross",
                                        "reduceOnly": True,
                                    },
                                )
                                logging.info(f"{pos_symbol} 숏 익절 주문 체결: {order}")
                            except Exception as e:
                                logging.error(f"{pos_symbol} 숏 익절 주문 실패: {e}")

                            if stop_order_id is not None:
                                try:
                                    exchange.cancel_order(stop_order_id, pos_symbol)
                                    logging.info(f"{pos_symbol} 숏 스탑 주문 취소: {stop_order_id}")
                                except Exception as e:
                                    logging.warning(f"{pos_symbol} 숏 스탑 취소 실패(이미 체결/취소됐을 수 있음): {e}")

                            in_position = False
                            pos_symbol = None
                            pos_side = None
                            position_size = 0.0
                            entry_price = None
                            stop_price = None
                            stop_order_id = None
                            entry_time = None

            # ---------------- 포지션 없는 경우: 각 심볼 신호 체크 후 하나만 진입 ---------------- #
            else:
                # 심볼 순서: BTC → XRP → DOGE
                for sym in SYMBOLS:
                    if sym not in data:
                        continue
                    df_sym, prev2, prev, curr = data[sym]
                    curr_ts = int(curr["ts"])

                    # 같은 심볼의 같은 캔들에서 중복 진입 방지
                    if sym in last_signal_candle_ts and last_signal_candle_ts[sym] == curr_ts:
                        continue

                    long_signal = check_long_entry(prev, curr)
                    short_signal = check_short_entry_lh(prev2, prev, curr)

                    if not (long_signal or short_signal):
                        continue

                    free_eq, total_eq = fetch_futures_equity(exchange)
                    logging.info(f"[{sym}] USDT Equity (free={free_eq}, total={total_eq})")

                    if total_eq <= 0:
                        logging.warning(f"[{sym}] equity가 0 이하입니다. 진입 스킵.")
                        continue

                    est_entry_price = float(curr["close"])
                    if est_entry_price <= 0:
                        logging.warning(f"[{sym}] 유효하지 않은 추정 진입가입니다. 진입 스킵.")
                        continue

                    # ma_gap 기반 stop_pct 계산
                    ma_gap_pct = calc_ma_gap_pct_from_row(curr)

                    # 리스크 3% 기반 포지션 크기 계산
                    amount, eff_lev = compute_order_size_risk_based(
                        exchange,
                        sym,
                        est_entry_price,
                        total_eq,
                        ma_gap_pct
                    )
                    if amount <= 0:
                        logging.warning(f"[{sym}] 포지션 수량이 0입니다. 진입 스킵.")
                        continue

                    try:
                        if long_signal:
                            side = "buy"
                            pos_side = "long"
                            log_side = "LONG"
                        else:
                            side = "sell"
                            pos_side = "short"
                            log_side = "SHORT"

                        logging.info(
                            f"[ENTRY {log_side}] {sym} 진입 신호 발생 / stop_pct={ma_gap_pct*100:.3f}%%, "
                            f"target_lev≈{RISK_PER_TRADE/ma_gap_pct:.2f}x, eff_lev≈{eff_lev:.2f}x"
                        )

                        order = exchange.create_order(
                            sym,
                            type="market",
                            side=side,
                            amount=amount,
                            params={
                                "tdMode": "cross",
                            },
                        )
                        logging.info(f"[{sym}] {log_side} 진입 주문 체결: {order}")

                        # 🔹 실제 포지션 진입가/사이즈를 다시 조회해서 SL 기준으로 사용
                        actual_entry_price = est_entry_price
                        actual_size = amount

                        # 체결 반영 기다렸다가 포지션 조회 (최대 5번 재시도)
                        time.sleep(0.5)
                        for _ in range(5):
                            has_pos2, sym2, side2, size2, entry2 = sync_position(exchange, SYMBOLS)
                            if has_pos2 and sym2 == sym and size2 > 0 and entry2 and entry2 > 0:
                                actual_entry_price = entry2
                                actual_size = size2
                                pos_side = side2  # 거래소 기준으로 덮어쓰기
                                break
                            time.sleep(0.3)

                        in_position = True
                        pos_symbol = sym
                        position_size = actual_size
                        entry_time = datetime.now(timezone.utc)
                        entry_price = actual_entry_price

                        # 손절 가격 계산 (실제 진입가 기준, ma_gap_pct 사용)
                        if pos_side == "long":
                            stop_price = entry_price * (1.0 - ma_gap_pct)
                            sl_side = "sell"
                        else:
                            stop_price = entry_price * (1.0 + ma_gap_pct)
                            sl_side = "buy"

                        # 조건부 스탑마켓 주문 (reduceOnly)
                        try:
                            sl_order = exchange.create_order(
                                sym,
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
                                f"[{sym}] {log_side} 스탑로스 주문 생성: id={stop_order_id}, "
                                f"트리거 가격={stop_price:.6f}, stop_pct={ma_gap_pct*100:.3f}%%"
                            )
                        except Exception as e:
                            logging.error(f"[{sym}] {log_side} 스탑로스 주문 생성 실패! 수동 확인 필요: {e}")
                            stop_order_id = None

                        logging.info(
                            f"[{sym}] {log_side} 실제 진입가={entry_price:.6f}, 수량={position_size}, "
                            f"스탑로스={stop_price:.6f} (stop_pct={ma_gap_pct*100:.3f}%%)"
                        )

                        last_signal_candle_ts[sym] = curr_ts

                        # 포지션 하나만 들고가므로, 진입 후 다른 심볼은 이번 턴에 보지 않음
                        break

                    except Exception as e:
                        logging.error(f"[{sym}] {log_side} 진입 주문 실패: {e}")

            time.sleep(LOOP_INTERVAL)

        except Exception as e:
            logging.error(f"메인 루프 에러: {e}")
            time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    main()
