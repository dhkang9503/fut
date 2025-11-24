import os
import time
import math
import logging
from datetime import datetime, timezone

import ccxt
import pandas as pd
import json

# ============== 설정값 ============== #

API_KEY = os.getenv("OKX_API_KEY", "")
API_SECRET = os.getenv("OKX_API_SECRET", "")
API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE", "")

SYMBOLS = [
    "BTC/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
]

TIMEFRAME = "1h"   # CCI + Bollinger 전략

# 리스크 및 레버리지 관련
RISK_PER_TRADE = 0.03
MAX_LEVERAGE   = 10

LOOP_INTERVAL = 5

# CCI / 볼린저 파라미터
CCI_PERIOD = 14
BB_PERIOD  = 20
BB_K       = 2.0

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ============== 상태 저장 ============== #

def save_state(pos_state, entry_restrict, last_signal, equity):
    state = {
        "pos_state": pos_state,
        "entry_restrict": entry_restrict,
        "last_signal": last_signal,
        "equity": equity,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open("/app/bot_state.json", "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# ============== 거래소 초기화 ============== #

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

    exchange.set_sandbox_mode(True)
    exchange.load_markets()

    try:
        exchange.set_position_mode(hedged=False)
        logging.info("포지션 모드: net 설정 완료")
    except Exception as e:
        logging.warning(f"포지션 모드 설정 실패: {e}")

    for sym in SYMBOLS:
        try:
            exchange.set_leverage(MAX_LEVERAGE, sym, params={"mgnMode": "cross"})
        except:
            pass

    return exchange

# ============== 유틸 ============== #

def fetch_ohlcv_df(exchange, symbol, timeframe, limit=200):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        return None
    df = pd.DataFrame(
        ohlcv, columns=["ts","open","high","low","close","volume"]
    )
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("dt", inplace=True)
    return df

def calculate_indicators(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma_tp = tp.rolling(CCI_PERIOD).mean()
    mean_dev = (tp - ma_tp).abs().rolling(CCI_PERIOD).mean()
    df["cci"] = (tp - ma_tp) / (0.015 * mean_dev)

    ma = df["close"].rolling(BB_PERIOD).mean()
    std = df["close"].rolling(BB_PERIOD).std()
    df["bb_mid"]   = ma
    df["bb_upper"] = ma + BB_K * std
    df["bb_lower"] = ma - BB_K * std

    return df

def fetch_futures_equity(exchange):
    balance = exchange.fetch_balance()
    usdt = balance.get("USDT", {})
    total = float(usdt.get("total", 0.0))
    free = float(usdt.get("free", 0.0))
    return free, total

def compute_order_size_risk_based(exchange, symbol, entry_price, equity_total, stop_pct):
    if entry_price <= 0 or equity_total <= 0 or stop_pct <= 0:
        return 0.0, 0.0

    risk_value = equity_total * RISK_PER_TRADE
    target_notional = risk_value / stop_pct
    max_notional = equity_total * MAX_LEVERAGE

    notional = min(target_notional, max_notional)

    market = exchange.market(symbol)
    contract_size = market.get("contractSize") or float(market.get("info", {}).get("ctVal", 1))

    amount = math.floor(notional / (entry_price * contract_size))
    if amount <= 0:
        return 0.0, 0.0

    effective_leverage = (amount * entry_price * contract_size) / equity_total
    return amount, effective_leverage

def sync_positions(exchange, symbols):
    result = {sym: {"has_position": False, "side":None, "size":0.0, "entry_price":None} for sym in symbols}

    try:
        positions = exchange.fetch_positions()
    except:
        return result

    for p in positions:
        sym = p.get("symbol")
        if sym not in symbols:
            continue

        side = (p.get("side") or "").lower()
        contracts = float(p.get("contracts") or 0)

        if not side:
            if abs(contracts) <= 0:
                continue
            side = "long" if contracts > 0 else "short"

        size = abs(contracts)
        if size <= 0:
            continue

        entry_price = float(p.get("entryPrice") or 0)

        result[sym] = {
            "has_position": True,
            "side": side,
            "size": size,
            "entry_price": entry_price or None,
        }

    return result

# ============== CCI 엔트리 ============== #

def detect_cci_signal(df):
    if df is None or len(df) < CCI_PERIOD + 3:
        return None

    curr = df.iloc[-2]
    prev = df.iloc[-3]

    cci_curr = float(curr["cci"])
    cci_prev = float(prev["cci"])
    entry_price = float(curr["close"])

    side = None
    stop_price = None

    # 숏
    if cci_prev > 100 and cci_curr <= 99:
        side = "short"
        stop_price = float(curr["high"])

    # 롱
    elif cci_prev < -100 and cci_curr >= -99:
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

# ============== 메인 루프 ============== #

def main():
    exchange = init_exchange()
    logging.info("OKX CCI+Bollinger 자동매매 시작")

    pos_state = {
        sym: {"side":None,"size":0.0,"entry_price":None,"stop_price":None,
              "stop_order_id":None,"entry_time":None}
        for sym in SYMBOLS
    }

    entry_restrict = {sym: None for sym in SYMBOLS}
    last_signal_candle_ts = {}

    while True:
        try:
            # ===== 데이터 업데이트 =====
            data = {}
            for sym in SYMBOLS:
                df = fetch_ohlcv_df(exchange, sym, TIMEFRAME, limit=CCI_PERIOD+BB_PERIOD+20)
                if df is None:
                    continue
                df = calculate_indicators(df)
                if len(df) < max(CCI_PERIOD, BB_PERIOD)+3:
                    continue
                prev = df.iloc[-3]
                curr = df.iloc[-2]
                data[sym] = (df, prev, curr)

            if not data:
                time.sleep(LOOP_INTERVAL)
                continue

            exch_positions = sync_positions(exchange, SYMBOLS)

            # ===== 포지션 동기화 =====
            for sym in SYMBOLS:
                exch_pos = exch_positions[sym]
                if not exch_pos["has_position"]:
                    if pos_state[sym]["side"] is not None and pos_state[sym]["size"] > 0:
                        last_side = pos_state[sym]["side"]
                        if last_side == "short":
                            entry_restrict[sym] = "long_only"
                        elif last_side == "long":
                            entry_restrict[sym] = "short_only"

                    pos_state[sym] = {
                        "side":None,"size":0.0,"entry_price":None,"stop_price":None,
                        "stop_order_id":None,"entry_time":None
                    }

                else:
                    pos_state[sym]["side"] = exch_pos["side"]
                    pos_state[sym]["size"] = exch_pos["size"]
                    if exch_pos["entry_price"]:
                        pos_state[sym]["entry_price"] = exch_pos["entry_price"]

            # ===== TP 처리 =====
            for sym in SYMBOLS:
                if sym not in data:
                    continue

                side = pos_state[sym]["side"]
                size = pos_state[sym]["size"]
                if not side or size <= 0:
                    continue

                df_sym, prev, curr = data[sym]

                bb_upper = float(curr["bb_upper"])
                bb_lower = float(curr["bb_lower"])

                high = float(curr["high"])
                low = float(curr["low"])

                # 롱 TP
                if side == "long" and high >= bb_upper:
                    logging.info(f"[TP LONG] {sym} 볼린저 상단 터치 → 청산")
                    stop_id = pos_state[sym]["stop_order_id"]
                    if stop_id:
                        try:
                            exchange.cancel_order(stop_id, sym)
                        except:
                            pass

                    exch_now = sync_positions(exchange, SYMBOLS)[sym]
                    if exch_now["has_position"]:
                        try:
                            exchange.create_order(sym, "market", "sell",
                                exch_now["size"], params={"tdMode":"cross"})
                        except:
                            pass

                    pos_state[sym] = {
                        "side":None,"size":0.0,"entry_price":None,"stop_price":None,
                        "stop_order_id":None,"entry_time":None
                    }
                    entry_restrict[sym] = "short_only"

                # 숏 TP
                elif side == "short" and low <= bb_lower:
                    logging.info(f"[TP SHORT] {sym} 볼린저 하단 터치 → 청산")

                    stop_id = pos_state[sym]["stop_order_id"]
                    if stop_id:
                        try:
                            exchange.cancel_order(stop_id, sym)
                        except:
                            pass

                    exch_now = sync_positions(exchange, SYMBOLS)[sym]
                    if exch_now["has_position"]:
                        try:
                            exchange.create_order(sym, "market", "buy",
                                exch_now["size"], params={"tdMode":"cross"})
                        except:
                            pass

                    pos_state[sym] = {
                        "side":None,"size":0.0,"entry_price":None,"stop_price":None,
                        "stop_order_id":None,"entry_time":None
                    }
                    entry_restrict[sym] = "long_only"

            # ===== 신규 진입 =====
            for sym in SYMBOLS:
                if sym not in data:
                    continue

                if pos_state[sym]["side"]:
                    continue

                df_sym, prev, curr = data[sym]
                curr_ts = int(curr["ts"])

                if last_signal_candle_ts.get(sym) == curr_ts:
                    continue

                signal = detect_cci_signal(df_sym)
                if not signal:
                    continue

                side_signal = signal["side"]
                entry_px = signal["entry_price"]
                stop_px = signal["stop_price"]

                restrict = entry_restrict[sym]
                if restrict == "long_only" and side_signal != "long":
                    continue
                if restrict == "short_only" and side_signal != "short":
                    continue

                stop_pct = abs(entry_px - stop_px) / entry_px

                free_eq, total_eq = fetch_futures_equity(exchange)
                if total_eq <= 0:
                    continue

                amount, eff_lev = compute_order_size_risk_based(
                    exchange, sym, entry_px, total_eq, stop_pct
                )
                if amount <= 0:
                    continue

                try:
                    if side_signal == "long":
                        order = exchange.create_order(sym, "market", "buy", amount,
                            params={"tdMode":"cross"})
                        sl_side = "sell"
                        pos_side = "long"
                    else:
                        order = exchange.create_order(sym, "market", "sell", amount,
                            params={"tdMode":"cross"})
                        sl_side = "buy"
                        pos_side = "short"

                    time.sleep(0.5)
                    for _ in range(5):
                        p = sync_positions(exchange, SYMBOLS)[sym]
                        if p["has_position"] and p["entry_price"]:
                            entry_px = p["entry_price"]
                            amount = p["size"]
                            break
                        time.sleep(0.3)

                    pos_state[sym]["side"] = pos_side
                    pos_state[sym]["size"] = amount
                    pos_state[sym]["entry_price"] = entry_px
                    pos_state[sym]["stop_price"] = stop_px
                    pos_state[sym]["entry_time"] = datetime.now(timezone.utc)

                    try:
                        sl_order = exchange.create_order(
                            sym, "market", sl_side, amount,
                            params={"tdMode":"cross","reduceOnly":True,
                                    "stopLossPrice":stop_px}
                        )
                        pos_state[sym]["stop_order_id"] = sl_order.get("id")
                    except:
                        pos_state[sym]["stop_order_id"] = None

                    last_signal_candle_ts[sym] = curr_ts
                    entry_restrict[sym] = None

                except Exception as e:
                    logging.error(f"[{sym}] 진입 실패: {e}")

            # ============================
            # ⭐ save_state 호출부 수정
            # ============================
            free_eq, total_eq = fetch_futures_equity(exchange)
            total_equity = total_eq

            save_state(pos_state, entry_restrict, last_signal_candle_ts, total_equity)

            time.sleep(LOOP_INTERVAL)

        except Exception as e:
            logging.error(f"메인 루프 에러: {e}")
            time.sleep(LOOP_INTERVAL)

if __name__ == "__main__":
    main()
