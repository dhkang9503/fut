#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import logging
from datetime import datetime, timezone
import json
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

TIMEFRAME = "1h"

RISK_PER_TRADE = 0.03
MAX_LEVERAGE   = 10
LOOP_INTERVAL  = 3

CCI_PERIOD = 14
BB_PERIOD  = 20
BB_K       = 2.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ============== JSON 직렬화 ============== #
def _serialize_pos_state(pos_state: dict):
    out = {}
    for sym, s in pos_state.items():
        if s is None:
            out[sym] = None
            continue
        d = dict(s)
        if isinstance(d.get("entry_time"), datetime):
            d["entry_time"] = d["entry_time"].isoformat()
        out[sym] = d
    return out


def save_state(pos_state, entry_restrict, last_signal, equity, ohlcv):
    state = {
        "pos_state": _serialize_pos_state(pos_state),
        "entry_restrict": entry_restrict,
        "last_signal": last_signal,
        "equity": equity,
        "ohlcv": ohlcv,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open("/app/bot_state.json", "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============== OKX 초기화 ============== #
def init_exchange():
    exchange = ccxt.okx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "password": API_PASSPHRASE,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "defaultSettle": "usdt"},
    })
    exchange.set_sandbox_mode(True)
    exchange.load_markets()

    try:
        exchange.set_position_mode(hedged=False)
    except Exception:
        pass

    for sym in SYMBOLS:
        try:
            exchange.set_leverage(MAX_LEVERAGE, sym, params={"mgnMode": "cross"})
        except Exception:
            pass

    return exchange


# ============== 유틸 ============== #
def fetch_ohlcv_df(exchange, symbol, timeframe, limit=200):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("dt", inplace=True)
    return df


def calculate_indicators(df):
    # CCI
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma_tp = tp.rolling(CCI_PERIOD).mean()
    mad = (tp - ma_tp).abs().rolling(CCI_PERIOD).mean()
    df["cci"] = (tp - ma_tp) / (0.015 * mad)

    # Bollinger Bands
    ma = df["close"].rolling(BB_PERIOD).mean()
    std = df["close"].rolling(BB_PERIOD).std()
    df["bb_mid"]   = ma
    df["bb_upper"] = ma + BB_K * std
    df["bb_lower"] = ma - BB_K * std
    return df


def fetch_futures_equity(exchange):
    bal = exchange.fetch_balance()
    usdt = bal.get("USDT", {})
    return float(usdt.get("free", 0)), float(usdt.get("total", 0))


def compute_order_size_risk_based(exchange, symbol, entry_price, equity_total, stop_pct):
    if entry_price <= 0 or stop_pct <= 0 or equity_total <= 0:
        return 0, 0

    risk_value = equity_total * RISK_PER_TRADE
    target_notional = risk_value / stop_pct
    max_notional = equity_total * MAX_LEVERAGE
    notional = min(target_notional, max_notional)

    market = exchange.market(symbol)
    contract_size = market.get("contractSize") or float(market["info"].get("ctVal", 1))
    notional_per_contract = entry_price * contract_size

    amount = math.floor(notional / notional_per_contract)
    if amount <= 0:
        return 0, 0

    effective_leverage = (amount * notional_per_contract) / equity_total
    return amount, effective_leverage


def sync_positions(exchange, symbols):
    result = {sym: {"has_position": False, "side": None, "size": 0, "entry_price": None} for sym in symbols}

    try:
        positions = exchange.fetch_positions()
    except Exception:
        return result

    for p in positions:
        sym = p.get("symbol")
        if sym not in symbols:
            continue

        contracts = float(p.get("contracts") or 0)
        if contracts == 0:
            continue

        side = p.get("side") or ("long" if contracts > 0 else "short")
        entry_price = float(p.get("entryPrice") or 0)

        result[sym] = {
            "has_position": True,
            "side": side,
            "size": abs(contracts),
            "entry_price": entry_price,
        }
    return result


# ============== CCI 신호 ============== #
def detect_cci_signal(df):
    if df is None or len(df) < CCI_PERIOD + 3:
        return None

    curr = df.iloc[-2]
    prev = df.iloc[-3]

    cci_curr = float(curr.get("cci", float("nan")))
    cci_prev = float(prev.get("cci", float("nan")))

    if math.isnan(cci_curr) or math.isnan(cci_prev):
        return None

    entry_price = float(curr["close"])
    if entry_price <= 0:
        return None

    side = None
    stop_price = None

    if cci_prev > 100 and 70 <= cci_curr <= 99:   # 숏 진입
        side = "short"
        stop_price = float(curr["high"])

    elif cci_prev < -100 and -70 >= cci_curr >= -99:  # 롱 진입
        side = "long"
        stop_price = float(curr["low"])

    if side is None or stop_price <= 0:
        return None

    return {
        "side": side,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "signal_ts": int(curr["ts"]),
    }


def _safe_float(v):
    """
    NaN / inf / None 등을 JSON에 안 들어가게 None으로 치환
    """
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


# ============== 메인 루프 ============== #
def main():
    exchange = init_exchange()
    logging.info("CCI + Bollinger 자동매매 (동적 TP + 마커 표시 + BB/CCI 대시보드) 시작")

    pos_state = {
        sym: {
            "side": None,
            "size": 0,
            "entry_price": None,
            "stop_price": None,
            "tp_price": None,
            "entry_candle_ts": None,
            "stop_order_id": None,
            "entry_time": None,
        }
        for sym in SYMBOLS
    }

    entry_restrict = {sym: None for sym in SYMBOLS}
    last_signal_candle_ts = {}

    while True:
        try:
            # --- OHLCV & 지표 --- #
            data = {}
            for sym in SYMBOLS:
                df = fetch_ohlcv_df(exchange, sym, TIMEFRAME, 200)
                if df is None:
                    continue
                df = calculate_indicators(df)
                if len(df) < BB_PERIOD + 3:
                    continue
                data[sym] = (df, df.iloc[-3], df.iloc[-2])

            if not data:
                time.sleep(LOOP_INTERVAL)
                continue

            # --- 포지션 동기화 --- #
            exch_positions = sync_positions(exchange, SYMBOLS)

            for sym in SYMBOLS:
                if not exch_positions[sym]["has_position"]:
                    pos_state[sym] = {
                        "side": None,
                        "size": 0,
                        "entry_price": None,
                        "stop_price": None,
                        "tp_price": None,
                        "entry_candle_ts": None,
                        "stop_order_id": None,
                        "entry_time": None,
                    }
                else:
                    pos_state[sym]["side"] = exch_positions[sym]["side"]
                    pos_state[sym]["size"] = exch_positions[sym]["size"]
                    pos_state[sym]["entry_price"] = exch_positions[sym]["entry_price"]

            # --- TP 관리 (동적) --- #
            for sym in SYMBOLS:
                if sym not in data:
                    continue
                df, prev, curr = data[sym]

                side = pos_state[sym]["side"]
                size = pos_state[sym]["size"]
                if side is None or size <= 0:
                    pos_state[sym]["tp_price"] = None
                    continue

                bb_upper = float(curr["bb_upper"])
                bb_lower = float(curr["bb_lower"])
                high = float(curr["high"])
                low = float(curr["low"])

                # 대시보드용 tp_price = 현재 볼린저
                pos_state[sym]["tp_price"] = bb_upper if side == "long" else bb_lower

                # 실제 익절
                if side == "long" and high >= bb_upper:
                    # 롱 익절
                    if pos_state[sym]["stop_order_id"]:
                        try:
                            exchange.cancel_order(pos_state[sym]["stop_order_id"], sym)
                        except Exception:
                            pass

                    exch_now = sync_positions(exchange, SYMBOLS)[sym]
                    if exch_now["has_position"]:
                        exchange.create_order(sym, "market", "sell", exch_now["size"], params={"tdMode": "cross"})

                    entry_restrict[sym] = "short_only"
                    pos_state[sym] = {
                        "side": None, "size": 0, "entry_price": None,
                        "stop_price": None, "tp_price": None,
                        "entry_candle_ts": None, "stop_order_id": None,
                        "entry_time": None,
                    }

                elif side == "short" and low <= bb_lower:
                    # 숏 익절
                    if pos_state[sym]["stop_order_id"]:
                        try:
                            exchange.cancel_order(pos_state[sym]["stop_order_id"], sym)
                        except Exception:
                            pass

                    exch_now = sync_positions(exchange, SYMBOLS)[sym]
                    if exch_now["has_position"]:
                        exchange.create_order(sym, "market", "buy", exch_now["size"], params={"tdMode": "cross"})

                    entry_restrict[sym] = "long_only"
                    pos_state[sym] = {
                        "side": None, "size": 0, "entry_price": None,
                        "stop_price": None, "tp_price": None,
                        "entry_candle_ts": None, "stop_order_id": None,
                        "entry_time": None,
                    }

            # --- 신규 진입 --- #
            for sym in SYMBOLS:
                if sym not in data:
                    continue

                df, prev, curr = data[sym]
                curr_ts = int(curr["ts"])
                if last_signal_candle_ts.get(sym) == curr_ts:
                    continue

                if pos_state[sym]["side"] is not None:
                    continue

                signal = detect_cci_signal(df)
                if not signal:
                    continue

                side_signal = signal["side"]
                entry_price = signal["entry_price"]
                stop_price = signal["stop_price"]

                # entry 제한 체크
                if entry_restrict[sym] == "long_only" and side_signal != "long":
                    continue
                if entry_restrict[sym] == "short_only" and side_signal != "short":
                    continue

                free, total = fetch_futures_equity(exchange)
                if total <= 0:
                    continue

                stop_pct = abs(entry_price - stop_price) / entry_price
                amount, eff_lev = compute_order_size_risk_based(
                    exchange, sym, entry_price, total, stop_pct
                )
                if amount <= 0:
                    continue

                # 주문
                order_side = "buy" if side_signal == "long" else "sell"
                sl_side = "sell" if side_signal == "long" else "buy"

                exchange.create_order(
                    sym,
                    "market",
                    order_side,
                    amount,
                    params={"tdMode": "cross"},
                )

                # 체결 확인
                time.sleep(0.3)
                after = sync_positions(exchange, SYMBOLS)[sym]
                actual_entry = after["entry_price"] or entry_price
                actual_size = after["size"]

                # pos_state 업데이트
                pos_state[sym]["side"] = side_signal
                pos_state[sym]["size"] = actual_size
                pos_state[sym]["entry_price"] = actual_entry
                pos_state[sym]["stop_price"] = stop_price
                pos_state[sym]["entry_time"] = datetime.now(timezone.utc)
                pos_state[sym]["entry_candle_ts"] = curr_ts  # ★ 마커 표시용 저장

                # SL 주문
                try:
                    sl_order = exchange.create_order(
                        sym,
                        "market",
                        sl_side,
                        actual_size,
                        params={"tdMode": "cross", "reduceOnly": True, "stopLossPrice": stop_price},
                    )
                    pos_state[sym]["stop_order_id"] = sl_order.get("id")
                except Exception:
                    pos_state[sym]["stop_order_id"] = None

                last_signal_candle_ts[sym] = curr_ts
                entry_restrict[sym] = None

            # --- 대시보드용 OHLCV + 인디케이터 저장 --- #
            ohlcv_state = {}
            for sym in SYMBOLS:
                if sym not in data:
                    continue
                df, _, _ = data[sym]
                tail = df.tail(100)
                candles = []
                for row in tail.itertuples():
                    candles.append({
                        "time": int(row.ts // 1000),
                        "open": float(row.open),
                        "high": float(row.high),
                        "low": float(row.low),
                        "close": float(row.close),
                        # 인디케이터 값들 같이 전달
                        "bb_upper": _safe_float(getattr(row, "bb_upper", None)),
                        "bb_lower": _safe_float(getattr(row, "bb_lower", None)),
                        "bb_mid": _safe_float(getattr(row, "bb_mid", None)),
                        "cci": _safe_float(getattr(row, "cci", None)),
                    })
                ohlcv_state[sym] = candles

            _, total = fetch_futures_equity(exchange)
            save_state(pos_state, entry_restrict, last_signal_candle_ts, total, ohlcv_state)

            time.sleep(LOOP_INTERVAL)

        except Exception as e:
            logging.warning(f"메인 루프 에러: {e}")
            time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    main()
