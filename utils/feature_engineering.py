import pandas as pd
import numpy as np
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

def build_multitimeframe_features_asof(dfs: list, prefixes: list, base_tf: str) -> pd.DataFrame:
    proc = []
    for df, pref in zip(dfs, prefixes):
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
        out = pd.merge_asof(out.sort_values("t_base"), f.sort_values("timestamp"),
                            left_on="t_base", right_on="timestamp",
                            direction="backward", tolerance=tol)
        out = out.drop(columns=["timestamp"])
    out = out.rename(columns={"t_base":"timestamp"}).dropna().reset_index(drop=True)
    if {"5m_close","15m_close"}.issubset(out.columns):
        out["ratio_5m_15m_close"] = out["5m_close"] / out["15m_close"]
    if {"5m_rsi","15m_rsi"}.issubset(out.columns):
        out["delta_rsi_5m_15m"] = out["5m_rsi"] - out["15m_rsi"]
    out = out.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)
    return out
