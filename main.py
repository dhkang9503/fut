# okx_live_trader_full.py
# - Incremental OHLCV (fixed window)
# - Feature engineering (내장)
# - Train-or-load (force_load_model / retrain_on_start 지원)
# - Consistency: min_confidence, cooldown, trigger-only logging
# - Telegram notify
# - OKX trading: single position, 10% balance notional, 25x leverage (config)
# - Env var expansion in YAML (${VAR} / ${VAR:default})

import os, re, csv, math, time, json, traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yaml
import ccxt
import joblib

from lightgbm import LGBMClassifier
from sklearn.exceptions import NotFittedError

# ---------- OPTIONAL: .env 지원 ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# =========================
# Utils & Config
# =========================

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)

def timeframe_to_millis(tf: str) -> int:
    if tf.endswith("m"): return int(tf[:-1]) * 60_000
    if tf.endswith("h"): return int(tf[:-1]) * 3_600_000
    if tf.endswith("d"): return int(tf[:-1]) * 86_400_000
    raise ValueError(f"Unsupported timeframe: {tf}")

_ENV_RE = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")  # ${VAR} or ${VAR:default}

def _expand_env(value):
    if isinstance(value, str):
        def repl(m):
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else "")
        return _ENV_RE.sub(repl, value)
    elif isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value

def load_config(path="config_okx.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _expand_env(cfg)

# =========================
# Exchange helpers (OKX)
# =========================

def create_okx(cfg):
    ex_cfg = cfg.get("okx", {})
    exchange = ccxt.okx({
        "apiKey": ex_cfg.get("apiKey", ""),
        "secret": ex_cfg.get("secret", ""),
        "password": ex_cfg.get("password", ""),
        "enableRateLimit": ex_cfg.get("enableRateLimit", True),
        "options": ex_cfg.get("options", {"defaultType": "swap"}),
    })
    return exchange

def resolve_symbol(exchange, desired_symbol):
    try:
        markets = exchange.load_markets()
        if desired_symbol in markets:
            return desired_symbol
    except Exception:
        pass
    return desired_symbol

def market_info(exchange, symbol):
    m = exchange.market(symbol)
    amount_prec = m.get("precision", {}).get("amount", None)  # int(소수 자릿수)일 때 多
    price_prec  = m.get("precision", {}).get("price", None)
    min_amt     = m.get("limits", {}).get("amount", {}).get("min", None)
    min_cost    = m.get("limits", {}).get("cost", {}).get("min", None)
    return amount_prec, price_prec, min_amt, min_cost, m

def round_amount_by_prec(amount, amount_prec):
    if amount_prec is None:
        return float(amount)
    if isinstance(amount_prec, int):
        return float(f"{amount:.{amount_prec}f}")
    # 틱/스텝 케이스는 보수적으로 내림
    return math.floor(amount / amount_prec) * amount_prec

def set_leverage_okx(exchange, symbol, lev=25, mgn_mode="cross"):
    try:
        exchange.set_leverage(lev, symbol, params={"mgnMode": mgn_mode, "posSide": "long"})
    except Exception as e:
        print("[WARN] set_leverage long:", e)
    try:
        exchange.set_leverage(lev, symbol, params={"mgnMode": mgn_mode, "posSide": "short"})
    except Exception as e:
        print("[WARN] set_leverage short:", e)

def get_free_usdt(exchange):
    bal = exchange.fetch_balance()
    # 통합 계정/선물 계정에서 모두 커버 시도
    if "USDT" in bal and isinstance(bal["USDT"], dict):
        if "free" in bal["USDT"]:
            return float(bal["USDT"]["free"])
        if "total" in bal["USDT"]:
            total = float(bal["USDT"]["total"])
            used  = float(bal["USDT"].get("used", 0))
            return max(0.0, total - used)
    if "free" in bal and "USDT" in bal["free"]:
        return float(bal["free"]["USDT"])
    return 0.0

def fetch_net_position(exchange, symbol):
    """
    OKX linear swap + one-way(net) 가정.
    return: ("LONG"/"SHORT"/"FLAT", size_base(float))
    """
    try:
        positions = exchange.fetch_positions([symbol])
        for p in positions:
            if p.get("symbol") == symbol:
                amt = float(p.get("contracts", 0) or p.get("amount", 0) or 0)
                side = "LONG" if amt > 0 else ("SHORT" if amt < 0 else "FLAT")
                return side, float(amt)
    except Exception as e:
        print("[WARN] fetch_positions:", e)
    return "FLAT", 0.0

def place_reduce_only_market(exchange, symbol, side, amount, amount_prec=None):
    amt = round_amount_by_prec(amount, amount_prec)
    params = {"reduceOnly": True, "tdMode": "cross"}
    return exchange.create_order(symbol, "market", side, amt, None, params)

def place_entry_with_tp_sl(exchange, symbol, direction, price, tp, sl,
                           notional_target, amount_prec, inline_params=True):
    """
    direction: "LONG" / "SHORT"
    notional_target: USDT 명목가 (잔고의 10% 등)
    수량 = notional_target / price  (BTC 수량)
    """
    amount = max(1e-9, notional_target / max(price, 1e-9))
    amount = round_amount_by_prec(amount, amount_prec)
    side = "buy" if direction == "LONG" else "sell"

    # 시도1) 인라인 TP/SL (계정 설정/상품에 따라 실패 가능)
    if inline_params:
        params = {
            "reduceOnly": False,
            "tdMode": "cross",
            "tpTriggerPx": f"{tp}",
            "tpOrdPx": "-1",  # market
            "slTriggerPx": f"{sl}",
            "slOrdPx": "-1",  # market
        }
        try:
            order = exchange.create_order(symbol, "market", side, amount, None, params)
            return {"entry": order, "tp": None, "sl": None}
        except Exception as e:
            print("[WARN] entry with inline TP/SL failed; fallback:", e)

    # 시도2) 진입 후 separate reduceOnly 트리거 주문
    entry = exchange.create_order(symbol, "market", side, amount, None, {"reduceOnly": False, "tdMode": "cross"})
    try:
        # NOTE: OKX 조건주문 파라미터는 계정/상품마다 상이 → 일반적 형태로 시도
        # TP
        tp_params = {"reduceOnly": True, "tdMode": "cross", "trigger": True, "stopPrice": tp}
        exchange.create_order(symbol, "market", "sell" if direction == "LONG" else "buy", amount, None, tp_params)
    except Exception as e:
        print("[WARN] TP separate order failed:", e)
    try:
        # SL
        sl_params = {"reduceOnly": True, "tdMode": "cross", "trigger": True, "stopPrice": sl}
        exchange.create_order(symbol, "market", "sell" if direction == "LONG" else "buy", amount, None, sl_params)
    except Exception as e:
        print("[WARN] SL separate order failed:", e)
    return {"entry": entry, "tp": None, "sl": None}

# =========================
# Telegram
# =========================

def send_telegram(token: str, chat_id: str, text: str):
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=5,
        )
    except Exception as e:
        print(f"[WARN] Telegram send failed: {e}")

# =========================
# Feature Engineering (내장)
# =========================
import ta

def _add_ta_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    rsi = ta.momentum.RSIIndicator(df["close"], window=14, fillna=True).rsi()
    ema_fast = ta.trend.EMAIndicator(df["close"], window=12, fillna=True).ema_indicator()
    ema_slow = ta.trend.EMAIndicator(df["close"], window=26, fillna=True).ema_indicator()
    macd = ema_fast - ema_slow
    bb = ta.volatility.BollingerBands(df["close"], window=20, fillna=True)
    bb_h = bb.bollinger_hband()
    bb_l = bb.bollinger_lband()
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14, fillna=True).average_true_range()
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14, fillna=True).adx()
    obv = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"], fillna=True).on_balance_volume()
    vol_ratio = df["volume"] / (df["volume"].rolling(20, min_periods=1).mean())
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    upper_wick = (df["high"] - df[["open","close"]].max(axis=1)).clip(lower=0)
    lower_wick = (df[["open","close"]].min(axis=1) - df["low"]).clip(lower=0)
    shadow_ratio = (upper_wick + lower_wick) / (rng.replace(0, np.nan))

    out = pd.DataFrame({
        f"{prefix}_close": df["close"].astype(float),
        f"{prefix}_volume": df["volume"].astype(float),
        f"{prefix}_rsi": rsi.astype(float),
        f"{prefix}_ema_fast": ema_fast.astype(float),
        f"{prefix}_ema_slow": ema_slow.astype(float),
        f"{prefix}_macd": macd.astype(float),
        f"{prefix}_bb_h": bb_h.astype(float),
        f"{prefix}_bb_l": bb_l.astype(float),
        f"{prefix}_atr": atr.astype(float),
        f"{prefix}_adx": adx.astype(float),
        f"{prefix}_obv": obv.astype(float),
        f"{prefix}_vol_ratio": vol_ratio.astype(float),
        f"{prefix}_body": body.astype(float),
        f"{prefix}_range": rng.astype(float),
        f"{prefix}_shadow_ratio": shadow_ratio.astype(float),
    }, index=df.index)
    return out

def _tf_seconds(tf: str) -> int:
    if tf.endswith("m"): return int(tf[:-1]) * 60
    if tf.endswith("h"): return int(tf[:-1]) * 3600
    if tf.endswith("d"): return int(tf[:-1]) * 86400
    raise ValueError("unsupported timeframe")

def build_multitimeframe_features_asof(dfs_list: list[pd.DataFrame], prefixes: list[str], base_tf: str) -> pd.DataFrame:
    """
    dfs_list와 prefixes는 같은 순서(예: ["5m","15m"])로 전달.
    """
    proc = []
    for df, pref in zip(dfs_list, prefixes):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        f = _add_ta_features(df, pref)
        f["timestamp"] = df["timestamp"].values
        proc.append(f.sort_values("timestamp"))

    base = proc[0][["timestamp"]].drop_duplicates().copy()
    out = base.rename(columns={"timestamp":"t_base"})
    tol = pd.Timedelta(seconds=2 * _tf_seconds(base_tf))

    for f in proc:
        out = pd.merge_asof(
            out.sort_values("t_base"),
            f.sort_values("timestamp"),
            left_on="t_base", right_on="timestamp",
            direction="backward", tolerance=tol
        )
        out = out.drop(columns=["timestamp"])
    out = out.rename(columns={"t_base":"timestamp"}).dropna().reset_index(drop=True)

    # cross-TF simple interactions
    if {"5m_close","15m_close"}.issubset(out.columns):
        out["ratio_5m_15m_close"] = out["5m_close"] / out["15m_close"]
    if {"5m_rsi","15m_rsi"}.issubset(out.columns):
        out["delta_rsi_5m_15m"] = out["5m_rsi"] - out["15m_rsi"]

    out = out.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)
    return out

# =========================
# Data Fetch (initial/incremental)
# =========================

def fetch_ohlcv_initial(exchange, symbol, timeframe, target_rows: int):
    tf_ms = timeframe_to_millis(timeframe)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    limit = min(1000, target_rows)
    rows = []
    since = now_ms - tf_ms * limit
    loops = 0
    while len(rows) < target_rows and loops < 400:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        oldest = batch[0][0]
        since = oldest - tf_ms * limit
        loops += 1
    if not rows:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").tail(target_rows).reset_index(drop=True)
    return df

def fetch_ohlcv_incremental(exchange, symbol, timeframe, cache_df: pd.DataFrame | None, target_rows: int):
    tf_ms = timeframe_to_millis(timeframe)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    limit = 1000
    if cache_df is not None and len(cache_df) > 0:
        since = int(cache_df["timestamp"].iloc[-1].timestamp() * 1000) + 1
    else:
        since = now_ms - tf_ms * (target_rows + 1)
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    if not data:
        return cache_df if cache_df is not None else pd.DataFrame(columns=["timestamp","open","high","low","close","volume"]), 0
    df_new = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms", utc=True)
    if cache_df is None:
        df = df_new
    else:
        df = pd.concat([cache_df, df_new], ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.tail(target_rows).reset_index(drop=True)
    return df, len(df_new)

# =========================
# Model (train-or-load)
# =========================

def _labeling(cfg, feats, base_tf):
    horizon = int(cfg["labeling"]["horizon"])
    threshold = float(cfg["labeling"]["threshold"])
    price_col = f"{base_tf}_close"

    df = feats.copy()
    if price_col not in df.columns:
        raise ValueError(f"price_col {price_col} not in features")
    df["future"] = df[price_col].shift(-horizon)
    df.dropna(inplace=True)
    df["ret"] = (df["future"] - df[price_col]) / df[price_col]
    df["label"] = 1  # FLAT
    df.loc[df["ret"] > threshold, "label"] = 2   # LONG
    df.loc[df["ret"] < -threshold, "label"] = 0  # SHORT

    X = df.drop(columns=["timestamp","future","ret","label"], errors="ignore")
    y = df["label"].astype(int)
    return X, y

def train_model(cfg, feats, base_tf):
    print("[INFO] Training LightGBM model...")
    X, y = _labeling(cfg, feats, base_tf)
    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.02,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    print(f"[INFO] Training complete on {len(X)} samples.")
    return model

def train_or_load_model(cfg, feats, base_tf):
    model_path = cfg["model_path"]
    force_load = bool(cfg.get("force_load_model", True))
    retrain = bool(cfg.get("train", {}).get("retrain_on_start", False))
    min_rows = int(cfg.get("min_train_rows", 1000))

    if feats is None or len(feats) < min_rows:
        raise SystemExit(f"[INFO] Not enough rows to train ({0 if feats is None else len(feats)}). Need >= {min_rows}")

    # 로드 우선
    if os.path.exists(model_path) and not retrain and force_load:
        print(f"[INFO] loading model from {model_path}")
        try:
            return joblib.load(model_path)
        except Exception as e:
            print("[WARN] failed to load model, retraining:", e)

    # 새 학습
    print("[INFO] training new model...")
    model = train_model(cfg, feats, base_tf)
    ensure_dir(os.path.dirname(model_path))
    joblib.dump(model, model_path)
    print(f"[INFO] model saved to {model_path}")
    return model

# =========================
# Logging (CSV)
# =========================

CSV_HEADERS = [
    "timestamp","symbol","raw_direction","raw_confidence","price",
    "enforced_direction","cooldown_remaining","tp","sl"
]

def ensure_logfile(path):
    ensure_dir(os.path.dirname(path))
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS)

def read_last_state(path):
    if not os.path.exists(path):
        return {"enforced_direction": "FLAT", "cooldown_remaining": 0}
    try:
        last = None
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                last = row
        if last is None:
            return {"enforced_direction": "FLAT", "cooldown_remaining": 0}
        return {
            "enforced_direction": last.get("enforced_direction", "FLAT"),
            "cooldown_remaining": int(float(last.get("cooldown_remaining", "0") or 0)),
        }
    except Exception:
        return {"enforced_direction": "FLAT", "cooldown_remaining": 0}

def append_log_row(path, row: dict):
    ensure_logfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row)

# =========================
# Strategy helpers
# =========================

def predict_dir_conf(model, x_last: pd.DataFrame):
    proba = model.predict_proba(x_last)[0]
    idx2dir = {0: -1, 1: 0, 2: 1}  # label 매핑: SHORT(0), FLAT(1), LONG(2)
    pred_dir = idx2dir[int(np.argmax(proba))]
    conf = float(np.max(proba))
    return pred_dir, conf

def dir_to_str(d):
    return "LONG" if d == 1 else "SHORT" if d == -1 else "FLAT"

# =========================
# Main
# =========================

def main():
    cfg = load_config("config_okx.yaml")

    # Signal policy
    min_conf = float(cfg.get("signal", {}).get("min_confidence", 0.75))
    cooldown_bars = int(cfg["signal"]["cooldown_bars"])
    tf_list = cfg["timeframes"]
    base_tf = tf_list[0]
    price_col = f"{base_tf}_close"
    take_profit = float(cfg["risk"]["take_profit"])   # 0.007
    stop_loss   = float(cfg["risk"]["stop_loss"])     # 0.004

    # Trading params
    leverage = int(cfg.get("trading", {}).get("leverage", 25))  # 25x
    position_risk_fraction = float(cfg.get("trading", {}).get("position_risk_fraction", 0.10))  # 잔고 10%

    # Telegram
    tg_cfg = cfg.get("telegram", {})
    tg_enabled = bool(tg_cfg.get("enabled", False))
    tg_token = tg_cfg.get("bot_token", "")
    tg_chat_id = tg_cfg.get("chat_id", "")

    # Logs
    log_path = cfg["log_path"]
    ensure_logfile(log_path)
    state = read_last_state(log_path)  # enforced_direction / cooldown_remaining

    # Exchange / Market / Leverage
    exchange = create_okx(cfg)
    symbol = resolve_symbol(exchange, cfg["symbol"])
    set_leverage_okx(exchange, symbol, leverage, mgn_mode="cross")
    amount_prec, price_prec, min_amt, min_cost, market = market_info(exchange, symbol)

    # Initial data fetch
    cache = {}
    for tf in tf_list:
        n = int(cfg["window_rows"][tf])
        df0 = fetch_ohlcv_initial(exchange, symbol, tf, n)
        cache[tf] = df0
        print(f"[INIT] {symbol} {tf} rows={len(df0)}  range={df0['timestamp'].min()} → {df0['timestamp'].max()}")

    # Initial features
    dfs_list = [cache[tf] for tf in tf_list]
    feats = build_multitimeframe_features_asof(dfs_list, prefixes=tf_list, base_tf=base_tf)
    if feats is None or len(feats) == 0:
        raise SystemExit("[INFO] Not enough feature rows after initial build. Increase window_rows or check market/timeframes.")

    # Train-or-Load model
    model = train_or_load_model(cfg, feats, base_tf)

    msg = "OKX LIVE Trader (Incremental + TrainOrLoad + TriggerOnly + Telegram) started. Ctrl+C to stop."
    send_telegram(tg_token, tg_chat_id, msg)

    while True:
        try:
            # Incremental updates
            adds_msg, dfs_list = [], []
            for tf in tf_list:
                n = int(cfg["window_rows"][tf])
                cache[tf], added = fetch_ohlcv_incremental(exchange, symbol, tf, cache[tf], n)
                adds_msg.append(f"{tf}(+{added})")
                dfs_list.append(cache[tf])
            # print(f"[INFO] incremental update: {', '.join(adds_msg)}")

            if len(dfs_list[0]) == 0:
                time.sleep(int(cfg["poll_interval_sec"])); continue

            feats = build_multitimeframe_features_asof(dfs_list, prefixes=tf_list, base_tf=base_tf)
            if len(feats) == 0:
                time.sleep(int(cfg["poll_interval_sec"])); continue

            X = feats.drop(columns=["timestamp"], errors="ignore")
            x_last = X.iloc[[-1]]

            # Price
            ticker = exchange.fetch_ticker(symbol)
            live_price = float(ticker["last"])

            # Predict
            raw_dir, raw_conf = predict_dir_conf(model, x_last)

            # Consistency / cooldown
            prev_dir = state.get("enforced_direction", "FLAT")
            cd_remain = int(state.get("cooldown_remaining", 0))
            final_dir = prev_dir

            if cd_remain > 0:
                cd_remain -= 1
            else:
                if raw_conf >= min_conf:
                    if prev_dir == "FLAT":
                        if raw_dir == 1:
                            final_dir = "LONG"; cd_remain = cooldown_bars
                        elif raw_dir == -1:
                            final_dir = "SHORT"; cd_remain = cooldown_bars
                    else:
                        if (prev_dir == "LONG" and raw_dir == -1) or (prev_dir == "SHORT" and raw_dir == 1):
                            final_dir = "LONG" if raw_dir == 1 else "SHORT"
                            cd_remain = cooldown_bars

            # Trigger-only: ENF 변경시에만 실행(주문/로그/텔레그램)
            if final_dir != prev_dir:
                # TP/SL 계산 (정보/주문용)
                if final_dir == "LONG":
                    tp = live_price * (1 + take_profit)
                    sl = live_price * (1 - stop_loss)
                elif final_dir == "SHORT":
                    tp = live_price * (1 - take_profit)
                    sl = live_price * (1 + stop_loss)
                else:
                    tp = None; sl = None

                # 1) 기존 포지션 정리
                cur_side, cur_amt = fetch_net_position(exchange, symbol)
                if cur_side != "FLAT":
                    side_to_close = "sell" if cur_side == "LONG" else "buy"
                    amt_to_close = abs(cur_amt)
                    try:
                        place_reduce_only_market(exchange, symbol, side_to_close, amt_to_close, amount_prec=amount_prec)
                        print(f"[TRADE] Closed existing position {cur_side} amt={amt_to_close}")
                    except Exception as e:
                        print("[ERROR] close position failed:", e)

                # 2) 새 포지션 진입
                if final_dir in ("LONG", "SHORT"):
                    free_usdt = get_free_usdt(exchange)
                    if free_usdt <= 0:
                        print("[WARN] No free USDT to enter.")
                    else:
                        notional_target = free_usdt * position_risk_fraction  # 잔고 10%
                        try:
                            res = place_entry_with_tp_sl(
                                exchange, symbol, final_dir, live_price, tp, sl,
                                notional_target, amount_prec, inline_params=True
                            )
                            print(f"[TRADE] Enter {final_dir} notional≈{notional_target:.2f} USDT @~{live_price:.2f}")
                        except Exception as e:
                            print("[ERROR] entry failed:", e)

                # 3) 로그 & 텔레그램
                msg = (
                    f"{now_iso()}  {symbol:>12}  TRADE_TRIGGER  "
                    f"RAW={dir_to_str(raw_dir):>5} p={raw_conf:.2f}  "
                    f"→ ENF={final_dir:<5}  cd={cd_remain:2d}  price={live_price:.4f}"
                )
                print(msg)
                append_log_row(cfg["log_path"], {
                    "timestamp": now_iso(),
                    "symbol": symbol,
                    "raw_direction": dir_to_str(raw_dir),
                    "raw_confidence": f"{raw_conf:.6f}",
                    "price": f"{live_price:.6f}",
                    "enforced_direction": final_dir,
                    "cooldown_remaining": cd_remain,
                    "tp": f"{tp:.6f}" if tp is not None else "",
                    "sl": f"{sl:.6f}" if sl is not None else "",
                })
                if tg_enabled:
                    send_telegram(tg_token, tg_chat_id, msg)

            # 상태 저장
            state = {"enforced_direction": final_dir, "cooldown_remaining": cd_remain}

            time.sleep(int(cfg["poll_interval_sec"]))
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except NotFittedError:
            print("[WARN] model not fitted. Training...")
            model = train_or_load_model(cfg, feats, base_tf)
        except Exception as e:
            print("Loop error:", e)
            traceback.print_exc()
            time.sleep(int(cfg["poll_interval_sec"]))

if __name__ == "__main__":
    main()
