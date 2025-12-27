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
    "AVAX/USDT:USDT",
    "OKB/USDT:USDT",
    "SOL/USDT:USDT",
]

TIMEFRAME = "1h"

RISK_PER_TRADE = 0.02
MAX_LEVERAGE   = 13
LOOP_INTERVAL  = 3

CCI_PERIOD = 14
BB_PERIOD  = 20
BB_K       = 2.0

SL_OFFSET  = 0.01
TP_OFFSET  = 0.004

R_THRESHOLD = 1.2
MIN_DELTA   = 16.0

MARGIN_DIVISOR = 3.5

# === BE 스탑 전환 설정 ===
R_BE_THRESHOLD = 0.6
BE_PROFIT_PCT  = 0.01   # +1% (수수료 커버)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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


def init_exchange():
    exchange = ccxt.okx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "password": API_PASSPHRASE,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "defaultSettle": "usdt"},
    })
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


def fetch_ohlcv_df(exchange, symbol, timeframe, limit=200):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("dt", inplace=True)
    return df


def calculate_indicators(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(CCI_PERIOD).mean()

    def _mean_dev(window):
        m = window.mean()
        return (window.sub(m).abs().sum()) / len(window)

    md = tp.rolling(CCI_PERIOD).apply(_mean_dev, raw=False)
    df["cci"] = (tp - sma_tp) / (0.015 * md)

    ma = df["close"].rolling(BB_PERIOD).mean()
    std = df["close"].rolling(BB_PERIOD).std(ddof=0)
    df["bb_mid"]   = ma
    df["bb_upper"] = ma + BB_K * std
    df["bb_lower"] = ma - BB_K * std
    return df


def fetch_futures_equity(exchange):
    bal = exchange.fetch_balance()
    usdt = bal.get("USDT", {})
    return float(usdt.get("free", 0)), float(usdt.get("total", 0))


def compute_order_size_equal_margin_and_risk(exchange, symbol, entry_price, equity_total, stop_pct):
    if entry_price <= 0 or stop_pct <= 0 or equity_total <= 0:
        return 0, 0.0, 0.0

    margin_per_pos = equity_total / MARGIN_DIVISOR
    risk_per_pos = equity_total * RISK_PER_TRADE
    target_notional_for_risk = risk_per_pos / stop_pct

    min_notional = margin_per_pos
    max_notional = margin_per_pos * MAX_LEVERAGE
    notional = max(min(target_notional_for_risk, max_notional), min_notional)

    market = exchange.market(symbol)
    contract_size = market.get("contractSize") or float(market["info"].get("ctVal", 1))
    notional_per_contract = entry_price * contract_size

    amount = math.floor(notional / notional_per_contract)
    if amount <= 0:
        return 0, 0.0, 0.0

    actual_notional = amount * notional_per_contract
    actual_leverage = actual_notional / margin_per_pos
    actual_leverage = min(max(actual_leverage, 1.0), MAX_LEVERAGE)

    effective_leverage = actual_notional / equity_total
    return amount, actual_leverage, effective_leverage


def sync_positions(exchange, symbols):
    result = {sym: {"has_position": False} for sym in symbols}
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

        market = exchange.market(sym)
        contract_size = market.get("contractSize") or float(market["info"].get("ctVal", 1))
        entry_price = float(p.get("entryPrice") or 0)

        result[sym] = {
            "has_position": True,
            "side": p.get("side"),
            "size": abs(contracts),
            "entry_price": entry_price,
            "margin": float(p.get("margin") or 0),
            "leverage": float(p.get("leverage") or 0),
            "notional": abs(contracts) * contract_size * entry_price,
        }
    return result


def detect_cci_signal(df):
    if len(df) < CCI_PERIOD + 3:
        return None

    curr, prev = df.iloc[-1], df.iloc[-2]
    entry_price = float(curr["close"])

    if prev["cci"] < -100 and curr["cci"] > prev["cci"] and curr["cci"] - prev["cci"] >= MIN_DELTA:
        return {
            "side": "long",
            "entry_price": entry_price,
            "stop_price": float(prev["low"]) * (1 - SL_OFFSET),
        }

    if prev["cci"] > 100 and curr["cci"] < prev["cci"] and prev["cci"] - curr["cci"] >= MIN_DELTA:
        return {
            "side": "short",
            "entry_price": entry_price,
            "stop_price": float(prev["high"]) * (1 + SL_OFFSET),
        }
    return None


def main():
    exchange = init_exchange()
    logging.info("CCI + Bollinger 자동매매 시작")

    pos_state = {
        sym: {
            "side": None,
            "size": 0,
            "entry_price": None,
            "stop_price": None,
            "tp_price": None,
            "stop_order_id": None,
            "be_applied": False,
        }
        for sym in SYMBOLS
    }

    while True:
        try:
            data = {}
            for sym in SYMBOLS:
                df = fetch_ohlcv_df(exchange, sym, TIMEFRAME)
                if df is None:
                    continue
                data[sym] = calculate_indicators(df)

            positions = sync_positions(exchange, SYMBOLS)

            for sym in SYMBOLS:
                if not positions[sym]["has_position"]:
                    pos_state[sym] = {
                        "side": None,
                        "size": 0,
                        "entry_price": None,
                        "stop_price": None,
                        "tp_price": None,
                        "stop_order_id": None,
                        "be_applied": False,
                    }
                    continue

                pos = pos_state[sym]
                curr = data[sym].iloc[-1]
                price = float(curr["close"])

                side = pos["side"]
                entry = pos["entry_price"]
                stop = pos["stop_price"]

                bb_upper = float(curr["bb_upper"])
                bb_lower = float(curr["bb_lower"])
                tp = bb_upper * (1 - TP_OFFSET) if side == "long" else bb_lower * (1 + TP_OFFSET)
                pos["tp_price"] = tp

                # === BE 스탑 전환 ===
                if not pos["be_applied"]:
                    if side == "long":
                        in_profit = price > entry
                        remaining_risk = price - stop
                        remaining_reward = tp - price
                        be_price = entry * (1 + BE_PROFIT_PCT)
                        sl_side = "sell"
                    else:
                        in_profit = price < entry
                        remaining_risk = stop - price
                        remaining_reward = price - tp
                        be_price = entry * (1 - BE_PROFIT_PCT)
                        sl_side = "buy"

                    if in_profit and remaining_risk > 0:
                        R = remaining_reward / remaining_risk if remaining_reward > 0 else 0
                        if R < R_BE_THRESHOLD:
                            if pos["stop_order_id"]:
                                exchange.cancel_order(pos["stop_order_id"], sym)

                            sl = exchange.create_order(
                                sym,
                                "market",
                                sl_side,
                                pos["size"],
                                params={
                                    "tdMode": "cross",
                                    "reduceOnly": True,
                                    "stopLossPrice": be_price,
                                },
                            )
                            pos["stop_order_id"] = sl.get("id")
                            pos["stop_price"] = be_price
                            pos["be_applied"] = True
                            logging.info(f"[{sym}] BE 전환 R={R:.2f}")

            time.sleep(LOOP_INTERVAL)

        except Exception as e:
            logging.warning(f"메인 루프 에러: {e}")
            time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    main()
