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

TIMEFRAME = "1h"   # CCI + Bollinger 전략: 1시간봉

# 리스크 및 레버리지 관련
RISK_PER_TRADE = 0.03      # 손절 도달 시 계좌의 3% 손실 목표
MAX_LEVERAGE   = 10        # 최대 레버리지(실제 포지션 노출 / equity 상한)

LOOP_INTERVAL = 5          # 루프 주기(초)

# CCI / 볼린저 파라미터
CCI_PERIOD = 14
BB_PERIOD  = 20
BB_K       = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ============== 상태 저장 ============== #

def _serialize_pos_state(pos_state: dict) -> dict:
    """datetime 등 JSON 안 되는 타입을 문자열로 바꿔서 저장용으로 변환."""
    out = {}
    for sym, s in pos_state.items():
        if s is None:
            out[sym] = None
            continue
        d = dict(s)  # shallow copy
        et = d.get("entry_time")
        if isinstance(et, datetime):
            d["entry_time"] = et.isoformat()
        out[sym] = d
    return out


def save_state(pos_state, entry_restrict, last_signal, equity, ohlcv):
    state = {
        "pos_state": _serialize_pos_state(pos_state),
        "entry_restrict": entry_restrict,
        "last_signal": last_signal,
        "equity": equity,
        "ohlcv": ohlcv,  # 심볼별 1h OHLCV
        "timestamp": datetime.utcnow().isoformat()
    }
    try:
        with open("/app/bot_state.json", "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"상태 저장 실패: {e}")


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

    # 데모(모의거래)면 True, 실계정이면 False
    exchange.set_sandbox_mode(True)

    exchange.load_markets()

    try:
        exchange.set_position_mode(hedged=False)
        logging.info("포지션 모드: net 설정 완료")
    except Exception as e:
        logging.warning(f"포지션 모드 설정 실패 (무시 가능): {e}")

    for sym in SYMBOLS:
        try:
            exchange.set_leverage(MAX_LEVERAGE, sym, params={"mgnMode": "cross"})
            logging.info(f"{sym} 레버리지 {MAX_LEVERAGE}배, cross 마진 설정 완료")
        except Exception as e:
            logging.warning(f"{sym} 레버리지/마진 설정 실패 (무시 가능): {e}")

    return exchange


# ============== 유틸 함수들 ============== #

def fetch_ohlcv_df(exchange, symbol, timeframe, limit=200):
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
    """
    CCI, Bollinger Bands 계산.
    - CCI: period = CCI_PERIOD
    - Bollinger: close 기준, period = BB_PERIOD, K = BB_K
    """
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
    """선물(USDT-M) 계좌에서 USDT equity 추정."""
    balance = exchange.fetch_balance()
    usdt = balance.get("USDT", {})
    total = float(usdt.get("total", 0.0))
    free = float(usdt.get("free", 0.0))
    return free, total


def compute_order_size_risk_based(exchange, symbol, entry_price, equity_total, stop_pct):
    """
    리스크 RISK_PER_TRADE 고정 포지션 크기 계산.

    - risk_value      = equity_total * RISK_PER_TRADE
    - target_notional = risk_value / stop_pct
    - max_notional    = equity_total * MAX_LEVERAGE
    - notional        = min(target_notional, max_notional)
    - amount(contracts) = floor( notional / (entry_price * contract_size) )
    """
    if entry_price <= 0 or equity_total <= 0 or stop_pct <= 0:
        return 0.0, 0.0

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

    amount = math.floor(notional / notional_per_contract)
    if amount <= 0:
        return 0.0, 0.0

    effective_leverage = (amount * notional_per_contract) / equity_total
    return amount, effective_leverage


def sync_positions(exchange, symbols):
    """
    OKX 선물 포지션 조회.
    리턴: { symbol: { "has_position", "side", "size", "entry_price" }, ... }
    """
    result = {
        sym: {
            "has_position": False,
            "side": None,
            "size": 0.0,
            "entry_price": None,
        }
        for sym in symbols
    }

    try:
        positions = exchange.fetch_positions()
    except Exception as e:
        logging.warning(f"포지션 조회 실패: {e}")
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
            "entry_price": entry_price if entry_price > 0 else None,
        }

    return result


# ============== CCI + Bollinger 엔트리 로직 ============== #

def detect_cci_signal(df: pd.DataFrame):
    """
    마지막으로 닫힌 캔들 기준 CCI 신호 탐지.
    - 직전 캔들의 CCI, 현재(막 닫힌) 캔들의 CCI 비교
    - 숏: prev_cci > +100 이고 curr_cci <= +99
    - 롱: prev_cci < -100 이고 curr_cci >= -99
    """
    if df is None or len(df) < CCI_PERIOD + 3:
        return None

    curr = df.iloc[-2]   # 막 닫힌 캔들
    prev = df.iloc[-3]   # 그 이전 캔들

    cci_curr = float(curr.get("cci", float("nan")))
    cci_prev = float(prev.get("cci", float("nan")))
    if math.isnan(cci_curr) or math.isnan(cci_prev):
        return None

    entry_price = float(curr["close"])
    if entry_price <= 0:
        return None

    side = None
    stop_price = None

    # 숏 신호: 과매수(+100 이상) 후 꺾여서 +99 이하로 복귀
    if cci_prev > 100 and (70 <= cci_curr <= 99):
        side = "short"
        stop_price = float(curr["high"])

    # 롱 신호: 과매도(-100 이하) 후 꺾여서 -99 이상으로 복귀
    elif cci_prev < -100 and (-70 >= cci_curr >= -99):
        side = "long"
        stop_price = float(curr["low"])

    if side is None or stop_price is None or stop_price <= 0:
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
    logging.info("OKX CCI + Bollinger 자동매매 봇 시작 (1h, 리스크 3% 고정, 익절/손절 후 방향 대칭 제한)")

    pos_state = {
        sym: {
            "side": None,
            "size": 0.0,
            "entry_price": None,
            "stop_price": None,
            "stop_order_id": None,
            "entry_time": None,
        }
        for sym in SYMBOLS
    }

    # 방향 제한: None / "long_only" / "short_only"
    entry_restrict = {sym: None for sym in SYMBOLS}
    last_signal_candle_ts = {}

    while True:
        try:
            # --- 캔들/지표 업데이트 --- #
            data = {}
            for sym in SYMBOLS:
                df = fetch_ohlcv_df(exchange, sym, TIMEFRAME, limit=CCI_PERIOD + BB_PERIOD + 50)
                if df is None or df.empty:
                    logging.warning(f"{sym} 캔들 데이터를 가져오지 못했습니다.")
                    continue
                df = calculate_indicators(df)
                if len(df) < max(CCI_PERIOD, BB_PERIOD) + 3:
                    logging.info(f"{sym}: 지표 계산에 필요한 캔들이 부족합니다.")
                    continue

                prev = df.iloc[-3]
                curr = df.iloc[-2]
                data[sym] = (df, prev, curr)

            if not data:
                logging.warning("어느 심볼에서도 유효한 데이터가 없습니다. 대기.")
                time.sleep(LOOP_INTERVAL)
                continue

            # --- 포지션 동기화 --- #
            exch_positions = sync_positions(exchange, SYMBOLS)

            for sym in SYMBOLS:
                exch_pos = exch_positions.get(sym, {})
                has_pos = exch_pos.get("has_position", False)

                if not has_pos:
                    if pos_state[sym]["side"] is not None and pos_state[sym]["size"] > 0:
                        last_side = pos_state[sym]["side"]
                        logging.info(f"[{sym}] 포지션 종료 감지, last_side={last_side}")
                        if last_side == "short":
                            entry_restrict[sym] = "long_only"
                        elif last_side == "long":
                            entry_restrict[sym] = "short_only"

                    pos_state[sym] = {
                        "side": None,
                        "size": 0.0,
                        "entry_price": None,
                        "stop_price": None,
                        "stop_order_id": None,
                        "entry_time": None,
                    }
                else:
                    pos_state[sym]["side"] = exch_pos.get("side")
                    pos_state[sym]["size"] = exch_pos.get("size", 0.0)
                    entry_price = exch_pos.get("entry_price")
                    if entry_price and entry_price > 0:
                        pos_state[sym]["entry_price"] = entry_price

            # --- TP 관리 (볼린저 터치) --- #
            for sym in SYMBOLS:
                if sym not in data:
                    continue
                side = pos_state[sym]["side"]
                size = pos_state[sym]["size"]
                if side is None or size <= 0:
                    continue

                df_sym, prev, curr = data[sym]

                bb_upper = float(curr.get("bb_upper", float("nan")))
                bb_lower = float(curr.get("bb_lower", float("nan")))
                high = float(curr["high"])
                low = float(curr["low"])

                # 롱 TP: 상단 터치
                if side == "long" and not math.isnan(bb_upper) and high >= bb_upper:
                    logging.info(f"[TP LONG] {sym} 볼린저 상단 터치 → 롱 익절")
                    stop_order_id = pos_state[sym]["stop_order_id"]
                    if stop_order_id:
                        try:
                            exchange.cancel_order(stop_order_id, sym)
                        except Exception as e:
                            logging.warning(f"{sym} 롱 스탑 취소 실패: {e}")
                    pos_state[sym]["stop_order_id"] = None
                    pos_state[sym]["stop_price"] = None

                    exch_now = sync_positions(exchange, SYMBOLS).get(sym, {})
                    if exch_now.get("has_position") and exch_now.get("size", 0) > 0:
                        try:
                            order = exchange.create_order(
                                sym,
                                type="market",
                                side="sell",
                                amount=exch_now["size"],
                                params={"tdMode": "cross"},
                            )
                            logging.info(f"{sym} 롱 익절 주문: {order}")
                        except Exception as e:
                            logging.error(f"{sym} 롱 익절 주문 실패: {e}")

                    pos_state[sym] = {
                        "side": None,
                        "size": 0.0,
                        "entry_price": None,
                        "stop_price": None,
                        "stop_order_id": None,
                        "entry_time": None,
                    }
                    entry_restrict[sym] = "short_only"

                # 숏 TP: 하단 터치
                elif side == "short" and not math.isnan(bb_lower) and low <= bb_lower:
                    logging.info(f"[TP SHORT] {sym} 볼린저 하단 터치 → 숏 익절")
                    stop_order_id = pos_state[sym]["stop_order_id"]
                    if stop_order_id:
                        try:
                            exchange.cancel_order(stop_order_id, sym)
                        except Exception as e:
                            logging.warning(f"{sym} 숏 스탑 취소 실패: {e}")
                    pos_state[sym]["stop_order_id"] = None
                    pos_state[sym]["stop_price"] = None

                    exch_now = sync_positions(exchange, SYMBOLS).get(sym, {})
                    if exch_now.get("has_position") and exch_now.get("size", 0) > 0:
                        try:
                            order = exchange.create_order(
                                sym,
                                type="market",
                                side="buy",
                                amount=exch_now["size"],
                                params={"tdMode": "cross"},
                            )
                            logging.info(f"{sym} 숏 익절 주문: {order}")
                        except Exception as e:
                            logging.error(f"{sym} 숏 익절 주문 실패: {e}")

                    pos_state[sym] = {
                        "side": None,
                        "size": 0.0,
                        "entry_price": None,
                        "stop_price": None,
                        "stop_order_id": None,
                        "entry_time": None,
                    }
                    entry_restrict[sym] = "long_only"

            # --- 신규 진입 --- #
            for sym in SYMBOLS:
                if sym not in data:
                    continue

                if pos_state[sym]["side"] is not None and pos_state[sym]["size"] > 0:
                    continue

                df_sym, prev, curr = data[sym]
                curr_ts = int(curr["ts"])

                if last_signal_candle_ts.get(sym) == curr_ts:
                    continue

                signal = detect_cci_signal(df_sym)
                if not signal:
                    continue

                side_signal = signal["side"]
                est_entry_price = signal["entry_price"]
                stop_price = signal["stop_price"]

                restrict = entry_restrict.get(sym)
                if restrict == "long_only" and side_signal != "long":
                    logging.info(f"[{sym}] 진입 제한: long_only → 숏 신호 무시")
                    continue
                if restrict == "short_only" and side_signal != "short":
                    logging.info(f"[{sym}] 진입 제한: short_only → 롱 신호 무시")
                    continue

                if est_entry_price <= 0 or stop_price <= 0:
                    continue

                stop_pct = abs(est_entry_price - stop_price) / est_entry_price
                if stop_pct <= 0:
                    continue

                free_eq, total_eq = fetch_futures_equity(exchange)
                logging.info(f"[{sym}] USDT Equity (free={free_eq}, total={total_eq})")

                if total_eq <= 0:
                    logging.warning(f"[{sym}] equity가 0 이하입니다. 진입 스킵.")
                    continue

                amount, eff_lev = compute_order_size_risk_based(
                    exchange, sym, est_entry_price, total_eq, stop_pct
                )
                if amount <= 0:
                    logging.warning(f"[{sym}] 포지션 수량이 0입니다. 진입 스킵.")
                    continue

                try:
                    if side_signal == "long":
                        side = "buy"
                        pos_side = "long"
                        log_side = "LONG"
                        sl_side = "sell"
                    else:
                        side = "sell"
                        pos_side = "short"
                        log_side = "SHORT"
                        sl_side = "buy"

                    logging.info(
                        f"[ENTRY {log_side}] {sym} CCI 신호 진입 / "
                        f"stop_pct={stop_pct*100:.3f}%%, "
                        f"target_lev≈{RISK_PER_TRADE/stop_pct:.2f}x, eff_lev≈{eff_lev:.2f}x, "
                        f"entry≈{est_entry_price:.6f}, SL={stop_price:.6f}"
                    )

                    order = exchange.create_order(
                        sym,
                        type="market",
                        side=side,
                        amount=amount,
                        params={"tdMode": "cross"},
                    )
                    logging.info(f"[{sym}] {log_side} 진입 주문 체결: {order}")

                    actual_entry_price = est_entry_price
                    actual_size = amount

                    time.sleep(0.5)
                    for _ in range(5):
                        exch_positions_after = sync_positions(exchange, SYMBOLS)
                        p = exch_positions_after.get(sym, {})
                        if p.get("has_position") and p.get("size", 0) > 0 and p.get("entry_price"):
                            actual_entry_price = p["entry_price"]
                            actual_size = p["size"]
                            break
                        time.sleep(0.3)

                    pos_state[sym]["side"] = pos_side
                    pos_state[sym]["size"] = actual_size
                    pos_state[sym]["entry_price"] = actual_entry_price
                    pos_state[sym]["entry_time"] = datetime.now(timezone.utc)
                    pos_state[sym]["stop_price"] = stop_price

                    stop_order_id = None
                    try:
                        sl_order = exchange.create_order(
                            sym,
                            type="market",
                            side=sl_side,
                            amount=actual_size,
                            params={
                                "tdMode": "cross",
                                "reduceOnly": True,
                                "stopLossPrice": stop_price,
                            },
                        )
                        stop_order_id = sl_order.get("id")
                        pos_state[sym]["stop_order_id"] = stop_order_id
                        logging.info(
                            f"[{sym}] {log_side} 스탑로스 주문 생성: id={stop_order_id}, "
                            f"트리거 가격={stop_price:.6f}"
                        )
                    except Exception as e:
                        logging.error(f"[{sym}] {log_side} 스탑로스 주문 생성 실패: {e}")
                        pos_state[sym]["stop_order_id"] = None

                    logging.info(
                        f"[{sym}] {log_side} 실제 진입가={actual_entry_price:.6f}, 수량={actual_size}, "
                        f"SL={stop_price:.6f}, stop_pct={stop_pct*100:.3f}%%"
                    )

                    last_signal_candle_ts[sym] = curr_ts
                    entry_restrict[sym] = None

                except Exception as e:
                    logging.error(f"[{sym}] {log_side} 진입 주문 실패: {e}")

            # --- 대시보드용 OHLCV + equity 저장 --- #
            ohlcv_state = {}
            for sym in SYMBOLS:
                if sym not in data:
                    continue
                df_sym, _, _ = data[sym]
                tail = df_sym.tail(120)  # 최근 120개 1h 캔들

                candles = []
                for row in tail.itertuples():
                    candles.append({
                        "time": int(row.ts // 1000),
                        "open": float(row.open),
                        "high": float(row.high),
                        "low": float(row.low),
                        "close": float(row.close),
                    })
                ohlcv_state[sym] = candles

            free_eq, total_eq = fetch_futures_equity(exchange)
            total_equity = total_eq

            save_state(pos_state, entry_restrict, last_signal_candle_ts, total_equity, ohlcv_state)

            time.sleep(LOOP_INTERVAL)

        except Exception as e:
            logging.error(f"메인 루프 에러: {e}")
            time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    main()
