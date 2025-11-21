#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
from typing import Optional, Dict, Any, Tuple

import ccxt
import pandas as pd

# ======================
# 설정
# ======================
API_KEY = os.getenv("OKX_API_KEY", "YOUR_API_KEY")
SECRET = os.getenv("OKX_API_SECRET", "YOUR_SECRET")
PASSWORD = os.getenv("OKX_API_PASSPHRASE", "YOUR_PASSWORD")  # OKX passphrase

# 데모 / 실계정 선택
USE_TESTNET = True  # 데모 트레이딩이면 True

SYMBOLS = [
    "BTC/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
]

TIMEFRAME = "5m"
MA_FAST = 50
MA_SLOW = 200

LEVERAGE = 6          # 고정 레버리지
STOP_PCT = 0.005      # -0.5% 가격 움직임에 손절 (진입가 기준)
POLL_INTERVAL = 5    # 초

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ======================
# OKX 초기화
# ======================
def create_exchange() -> ccxt.okx:
    exchange = ccxt.okx({
        "apiKey": API_KEY,
        "secret": SECRET,
        "password": PASSWORD,
        "enableRateLimit": True,
    })
    if USE_TESTNET:
        exchange.set_sandbox_mode(True)
    return exchange


# ======================
# 유틸
# ======================
def fetch_ohlcv_df(ex: ccxt.okx, symbol: str, limit: int = 250) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
    df = pd.DataFrame(
        ohlcv,
        columns=["ts", "open", "high", "low", "close", "volume"],
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    df["ma_fast"] = df["close"].rolling(MA_FAST).mean()
    df["ma_slow"] = df["close"].rolling(MA_SLOW).mean()
    df.dropna(inplace=True)
    return df


def get_usdt_equity(ex: ccxt.okx) -> Tuple[float, float]:
    """
    cross 선물 USDT 잔고 조회: (free, total)
    """
    balance = ex.fetch_balance({"type": "swap"})
    info = balance.get("USDT", {})
    free = float(info.get("free", 0.0))
    total = float(info.get("total", 0.0))
    return free, total


def get_current_position(
    ex: ccxt.okx,
    symbol: str,
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    현재 심볼 포지션 조회
    return: (side, entry_price, size)
        side: 'long' / 'short' / None
    """
    positions = ex.fetch_positions([symbol])
    for pos in positions:
        if float(pos.get("contracts", 0) or 0) == 0:
            continue
        side = pos.get("side")  # ccxt unified: 'long'/'short'
        entry_price = pos.get("entryPrice")
        if entry_price is None and "avgPx" in pos.get("info", {}):
            entry_price = float(pos["info"]["avgPx"])
        size = float(pos.get("contracts", 0) or 0)
        return side, float(entry_price), size
    return None, None, None


def cancel_all_orders(ex: ccxt.okx, symbol: str):
    try:
        open_orders = ex.fetch_open_orders(symbol)
        for o in open_orders:
            try:
                ex.cancel_order(o["id"], symbol=symbol)
            except Exception as e:
                logging.warning(f"[{symbol}] 주문 취소 실패: {e}")
    except Exception as e:
        logging.warning(f"[{symbol}] 오픈 주문 조회 실패: {e}")


# ======================
# 전략 신호 계산
# ======================
def get_signal(df: pd.DataFrame) -> Optional[str]:
    """
    기존 전략 + LH 필터 버전
    return: 'long' / 'short' / None
    """
    if len(df) < 3:
        return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # LONG 조건
    long_cond = (
        (curr.ma_fast < curr.ma_slow) and
        (curr.ma_fast > prev.ma_fast) and
        (curr.close > curr.ma_fast)
    )

    # SHORT 조건 + LH
    short_base = (
        (curr.ma_fast > curr.ma_slow) and
        (curr.ma_fast < prev.ma_fast) and
        (curr.close < curr.ma_fast)
    )
    lh = (curr.high < prev.high) and (prev.high > prev2.high)
    short_cond = short_base and lh

    if long_cond:
        return "long"
    if short_cond:
        return "short"
    return None


# ======================
# 포지션 진입 + 실제 진입가 기반 스탑로스 설정
# ======================
def open_position_and_set_sl(
    ex: ccxt.okx,
    symbol: str,
    direction: str,
):
    """
    direction: 'long' or 'short'
    1) 시장가 진입
    2) 실제 포지션 진입가(avgPx/entryPrice)를 API로 조회
    3) 그 가격 기준으로 -0.5% SL 주문(reduce-only) 세팅
    """
    assert direction in ("long", "short")

    free, total = get_usdt_equity(ex)
    if total <= 0:
        logging.warning(f"[{symbol}] USDT 잔고 없음, 진입 불가")
        return

    # 풀시드 기준: 계좌 equity 전부 × 레버리지
    notional = total * LEVERAGE * 0.92
    ticker = ex.fetch_ticker(symbol)
    price = ticker["last"]
    amount = notional / price
    amount = float(ex.amount_to_precision(symbol, amount))

    side = "buy" if direction == "long" else "sell"

    logging.info(
        f"[ENTRY {direction.upper()}] {symbol} 시장가 진입 시도 "
        f"(notional≈{notional:.2f} USDT, amount={amount})"
    )

    cancel_all_orders(ex, symbol)

    try:
        ex.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params={
                "tdMode": "cross",
                "lever": str(LEVERAGE),
                # 필요하면 "posSide": "long"/"short" 추가
            },
        )
    except Exception as e:
        logging.error(f"[{symbol}] {direction} 진입 주문 실패: {e}")
        return

    # 체결 반영 기다리기
    time.sleep(1.0)

    pos_side, entry_price, pos_size = get_current_position(ex, symbol)
    if pos_side is None or entry_price is None or pos_size is None:
        logging.error(f"[{symbol}] 포지션 조회 실패 (진입 후) - SL 설정 불가")
        return

    logging.info(
        f"[{symbol}] 실제 포지션 진입가 = {entry_price}, size={pos_size}, side={pos_side}"
    )

    # 진입가 기준 SL 계산
    if pos_side == "long":
        sl_price = entry_price * (1.0 - STOP_PCT)
        sl_side = "sell"
    else:  # short
        sl_price = entry_price * (1.0 + STOP_PCT)
        sl_side = "buy"

    sl_price = float(ex.price_to_precision(symbol, sl_price))

    logging.info(
        f"[{symbol}] 스탑로스 주문 설정: side={sl_side}, "
        f"trigger={sl_price} (진입가 대비 {STOP_PCT*100:.2f}%)"
    )

    try:
        # OKX TP/SL용 조건부 주문 – reduceOnly / closePosition 활용
        ex.create_order(
            symbol=symbol,
            type="market",          # 트리거 후 시장가 청산
            side=sl_side,
            amount=pos_size,
            params={
                "tdMode": "cross",
                "reduceOnly": True,
                "stopLossPrice": sl_price,   # ccxt unified 필드 (OKX는 slTriggerPx에 맵핑)
                # "closePosition": True,     # 필요하면 이 방식으로도 사용 가능
            },
        )
    except Exception as e:
        logging.error(f"[{symbol}] 스탑로스 주문 설정 실패: {e}")


# ======================
# 메인 루프
# ======================
def main():
    ex = create_exchange()

    logging.info("OKX 선물 자동매매 봇 시작")

    # 포지션은 한 번에 하나만
    current_symbol: Optional[str] = None

    while True:
        try:
            # 현재 실제 포지션 상태 동기화
            any_pos = None
            for sym in SYMBOLS:
                side, entry, size = get_current_position(ex, sym)
                if side is not None and size and size > 0:
                    any_pos = (sym, side, entry, size)
                    break

            if any_pos:
                sym, side, entry, size = any_pos
                if current_symbol != sym:
                    logging.info(
                        f"[{sym}] 기존 포지션 감지됨: side={side}, "
                        f"entry={entry}, size={size}"
                    )
                current_symbol = sym
                # 포지션 들고 있는 동안엔 시그널만 모니터링 (추가 진입 없음)
            else:
                current_symbol = None

            if current_symbol is None:
                # 무포지션일 때만 새 진입 시그널 탐색
                for sym in SYMBOLS:
                    try:
                        df = fetch_ohlcv_df(ex, sym, limit=MA_SLOW + 5)
                    except Exception as e:
                        logging.warning(f"[{sym}] 캔들 조회 실패: {e}")
                        continue

                    signal = get_signal(df)
                    if signal is None:
                        continue

                    logging.info(f"[ENTRY {signal.upper()}] {sym} 진입 신호 발생")
                    open_position_and_set_sl(ex, sym, signal)
                    current_symbol = sym
                    break  # 포지션 하나만

        except Exception as e:
            logging.error(f"메인 루프 에러: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
