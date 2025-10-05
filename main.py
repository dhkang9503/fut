# ob_fvg_telegram_http_bot.py
# --- Telegram Bot API(HTTP)로 알림 보내는 OB+FVG 신호 봇 ---
# Strategy: Signal 45m, Confirm 10m + sliding 5m, ATR×1.2, N_break=3, FVG invalidation 90%, max 3 entries/OB
# Exchange: Binance USDT Perp via ccxt (read-only)

import os
import time
import json
import math
import requests
from datetime import datetime, timezone, timedelta

import ccxt
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
CONFIG = {
    "exchange": "binanceusdm",          # Binance Futures (USDT-M)
    "symbol": "XRP/USDT:USDT",
    "poll_seconds": 60,                 # 폴링 주기
    "history_days": 7,                  # 초기 로드 일수(5m 기준)
    "signal_tf": "45T",                 # 45분
    "confirm_tf": "10T",                # 10분
    "risk_fraction": 0.05,              # 5% 리스크
    "account_equity_usd": 1000.0,       # 사이징 기준 자본(추정)
    "max_reentries_per_ob": 3,          # OB당 재진입 한도
    "fvg_invalidation": 0.90,           # FVG 90% 메워지면 무효
    "atr_mult": 1.2,
    "n_break": 3,
    "tp1_R": 1.5,
    "tp2_R": 3.0,
    "state_file": "bot_state.json",     # 중복 방지 상태 저장
    # Telegram Bot API
    "telegram_token_env": "TELEGRAM_BOT_TOKEN",
    "telegram_chat_env": "TELEGRAM_CHAT_ID",
    "telegram_timeout": 10,             # HTTP 타임아웃(초)
    "parse_mode": "Markdown"            # 또는 "MarkdownV2"
}

# =========================
# Utils
# =========================
def utc_now():
    return datetime.now(timezone.utc)

def ohlcv_to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()

def resample_ohlc(df, rule):
    return df.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

def load_state(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"seen_entries": {}, "ob_counts": {}}

def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)

def fmt_ts(ts: pd.Timestamp) -> str:
    # 텔레그램 알림은 KST로 보여주기
    return ts.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S KST")

# =========================
# Strategy blocks
# =========================
def find_fvg(h: pd.DataFrame) -> pd.DataFrame:
    rows = []
    idx = h.index
    for i in range(len(h)-2):
        hi0 = h["high"].iloc[i]
        lo2 = h["low"].iloc[i+2]
        if lo2 > hi0:  # bullish FVG
            rows.append({"time": idx[i+2], "type":"bull", "low": hi0, "high": lo2, "origin_time": idx[i]})
        lo0 = h["low"].iloc[i]
        hi2 = h["high"].iloc[i+2]
        if hi2 < lo0:  # bearish FVG
            rows.append({"time": idx[i+2], "type":"bear", "low": hi2, "high": lo0, "origin_time": idx[i]})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["time","type","low","high","origin_time"])

def build_ob(h: pd.DataFrame, n_break=3, atr_mult=1.2):
    tr = h["high"] - h["low"]
    out = h.copy()
    out["atr"] = tr.rolling(14, min_periods=14).mean()
    out["prior_max_high"] = out["high"].rolling(n_break).max().shift(1)
    out["prior_min_low"]  = out["low"].rolling(n_break).min().shift(1)
    out["is_bull_disp"] = (out["close"] > out["prior_max_high"]) & ((out["high"]-out["low"]) > atr_mult*out["atr"])
    out["is_bear_disp"] = (out["close"] < out["prior_min_low"])  & ((out["high"]-out["low"]) > atr_mult*out["atr"])

    rows = []
    for i in range(1, len(out)):
        t = out.index[i]
        if out["is_bull_disp"].iloc[i]:
            for k in [i-1, i-2]:
                if k >= 0 and out["close"].iloc[k] < out["open"].iloc[k]:
                    rows.append({"time": t, "dir":"bull",
                                 "ob_start": out["low"].iloc[k],
                                 "ob_end":   out["open"].iloc[k],
                                 "ob_time":  out.index[k]})
                    break
        if out["is_bear_disp"].iloc[i]:
            for k in [i-1, i-2]:
                if k >= 0 and out["close"].iloc[k] > out["open"].iloc[k]:
                    rows.append({"time": t, "dir":"bear",
                                 "ob_start": out["open"].iloc[k],
                                 "ob_end":   out["high"].iloc[k],
                                 "ob_time":  out.index[k]})
                    break
    ob = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["time","dir","ob_start","ob_end","ob_time"])
    return out, ob

def pair_ob_fvg(ob: pd.DataFrame, fvg: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for _, r in ob.iterrows():
        same = fvg[fvg["type"] == ("bull" if r["dir"]=="bull" else "bear")]
        window = same[(same["time"] >= r["time"] - pd.Timedelta(hours=6)) & (same["time"] <= r["time"] + pd.Timedelta(hours=6))].copy()
        if window.empty:
            continue
        if r["dir"]=="bull":
            cands = window[window["low"] >= r["ob_end"]].copy()
            if cands.empty: 
                continue
            cands["dist"] = cands["low"] - r["ob_end"]
            chosen = cands.sort_values("dist").iloc[0]
        else:
            cands = window[window["high"] <= r["ob_start"]].copy()
            if cands.empty:
                continue
            cands["dist"] = r["ob_start"] - cands["high"]
            chosen = cands.sort_values("dist").iloc[0]
        pairs.append({**r.to_dict(),
                      "fvg_low": chosen["low"], "fvg_high": chosen["high"],
                      "fvg_time": chosen["time"], "fvg_origin": chosen["origin_time"]})
    return pd.DataFrame(pairs) if pairs else pd.DataFrame(columns=list(ob.columns)+["fvg_low","fvg_high","fvg_time","fvg_origin"])

def touched_zone(bar_high, bar_low, z_low, z_high):
    return (bar_low <= z_high) and (bar_high >= z_low)

def fvg_fill_pct(dirn, f_low, f_high, prices_max, prices_min):
    height = f_high - f_low
    if height <= 0: return 1.0
    if dirn == "bull":
        filled = max(0.0, prices_max - f_low)
        return min(1.0, filled/height)
    else:
        filled = max(0.0, f_high - prices_min)
        return min(1.0, filled/height)

# =========================
# Telegram (HTTP)
# =========================
def tg_send_message(text: str):
    token = os.getenv(CONFIG["telegram_token_env"])
    chat_id = os.getenv(CONFIG["telegram_chat_env"])
    print(token, chat_id)
    if not token or not chat_id:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars.")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": CONFIG["parse_mode"],
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=CONFIG["telegram_timeout"])
        if r.status_code != 200:
            print("Telegram send error:", r.text)
    except Exception as e:
        print("Telegram send exception:", e)

# =========================
# Main live loop
# =========================
def main():
    ex = getattr(ccxt, CONFIG["exchange"])({"enableRateLimit": True})
    symbol = CONFIG["symbol"]
    poll_s = CONFIG["poll_seconds"]

    state = load_state(CONFIG["state_file"])

    since = int((utc_now() - timedelta(days=CONFIG["history_days"])).timestamp() * 1000)
    ohlcv = ex.fetch_ohlcv(symbol, timeframe="5m", since=since, limit=None)
    df = ohlcv_to_df(ohlcv)

    tg_send_message(f"✅ Bot started for {symbol}\nSignal:45m / Confirm:10m+5m\nRisk:{int(CONFIG['risk_fraction']*100)}%  FVG invalid:{int(CONFIG['fvg_invalidation']*100)}%")

    while True:
        try:
            # 최근 2시간만 갱신
            since_ms = int((utc_now() - timedelta(hours=2)).timestamp() * 1000)
            new_ohlcv = ex.fetch_ohlcv(symbol, timeframe="5m", since=since_ms, limit=None)
            dfn = ohlcv_to_df(new_ohlcv)
            if len(dfn) > 0:
                df = pd.concat([df[df.index < dfn.index[0]], dfn]).drop_duplicates().sort_index()

            m5  = df.copy()
            m10 = resample_ohlc(df, CONFIG["confirm_tf"])
            m45 = resample_ohlc(df, CONFIG["signal_tf"])

            s45, ob45 = build_ob(m45, n_break=CONFIG["n_break"], atr_mult=CONFIG["atr_mult"])
            fvg45 = find_fvg(m45)
            pairs = pair_ob_fvg(ob45, fvg45)
            if pairs.empty:
                time.sleep(poll_s); continue

            # 최근 24시간 신호만 스캔
            recent_pairs = pairs[pairs["time"] >= (m45.index[-1] - pd.Timedelta(hours=24))]
            for _, p in recent_pairs.iterrows():
                dirn = p["dir"]
                z_low, z_high = (min(p["ob_start"], p["ob_end"]), max(p["ob_start"], p["ob_end"]))
                start_time = p["time"]
                future = m45.loc[start_time + pd.Timedelta(minutes=45):]
                if future.empty:
                    continue

                max_seen, min_seen = -np.inf, np.inf
                post_fvg = m45.loc[p["fvg_time"]:] if p["fvg_time"] in m45.index else future

                ob_key = f"{symbol}|{fmt_ts(p['ob_time'])}|{dirn}"
                count = state["ob_counts"].get(ob_key, 0)
                if count >= CONFIG["max_reentries_per_ob"]:
                    continue

                # 가장 최신 3개(약 135분) 윈도를 검사
                for t, bar in future.iloc[-4:].iterrows():
                    if t in post_fvg.index:
                        max_seen = max(max_seen, bar["high"])
                        min_seen = min(min_seen, bar["low"] if np.isfinite(min_seen) else bar["low"])
                    fill = fvg_fill_pct(dirn, p["fvg_low"], p["fvg_high"], max_seen, min_seen) if np.isfinite(max_seen) and np.isfinite(min_seen) else 0.0
                    if fill >= CONFIG["fvg_invalidation"]:
                        break

                    if not touched_zone(bar["high"], bar["low"], z_low, z_high):
                        continue

                    # 10m OR 5m 슬라이딩 컨펌
                    win_start, win_end = t, t + pd.Timedelta(minutes=45)
                    seg10 = m10.loc[(m10.index >= win_start) & (m10.index < win_end)]
                    seg5  = m5.loc[(m5.index  >= win_start) & (m5.index  < win_end)]
                    confirm = False

                    if len(seg10) > 0:
                        rng10 = (seg10["high"] - seg10["low"]).replace(0, np.nan)
                        if dirn == "bull":
                            strong10 = seg10[(seg10["close"] > seg10["open"]) & ((seg10["close"] - seg10["low"]) > 0.75*rng10)]
                        else:
                            strong10 = seg10[(seg10["close"] < seg10["open"]) & ((seg10["high"] - seg10["close"]) > 0.75*rng10)]
                        confirm = len(strong10) > 0
                    if not confirm and len(seg5) > 0:
                        rng5 = (seg5["high"] - seg5["low"]).replace(0, np.nan)
                        if dirn == "bull":
                            strong5 = seg5[(seg5["close"] > seg5["open"]) & ((seg5["close"] - seg5["low"]) > 0.75*rng5)]
                        else:
                            strong5 = seg5[(seg5["close"] < seg5["open"]) & ((seg5["high"] - seg5["close"]) > 0.75*rng5)]
                        confirm = len(strong5) > 0
                    if not confirm:
                        continue

                    # Entry/SL/TP/Size/Lev
                    if dirn == "bull":
                        entry, stop = z_high, z_low
                        rpu = entry - stop
                        tp1 = entry + CONFIG["tp1_R"]*rpu
                        tp2 = entry + CONFIG["tp2_R"]*rpu
                        side = "LONG"
                    else:
                        entry, stop = z_low, z_high
                        rpu = stop - entry
                        tp1 = entry - CONFIG["tp1_R"]*rpu
                        tp2 = entry - CONFIG["tp2_R"]*rpu
                        side = "SHORT"

                    if rpu <= 0:
                        continue

                    risk_usd = CONFIG["risk_fraction"] * CONFIG["account_equity_usd"]
                    size_units = risk_usd / rpu
                    notional = size_units * entry
                    implied_leverage = notional / CONFIG["account_equity_usd"]

                    # 중복 방지
                    uniq_key = f"{side}|{round(float(entry),5)}|{fmt_ts(t)}"
                    seen = state["seen_entries"].get(ob_key, [])
                    if uniq_key in seen:
                        continue

                    state["seen_entries"].setdefault(ob_key, []).append(uniq_key)
                    state["ob_counts"][ob_key] = state["ob_counts"].get(ob_key, 0) + 1
                    save_state(CONFIG["state_file"], state)

                    text = (
                        f"📣 *OB+FVG 진입 신호* ({CONFIG['symbol']})\n"
                        f"• 방향: *{side}*\n"
                        f"• 시그널(45m): {fmt_ts(p['time'])}\n"
                        f"• OB 형성: {fmt_ts(p['ob_time'])}\n"
                        f"• 리테스트 감지: {fmt_ts(t)}\n"
                        f"\n"
                        f"• 진입가: `{entry:.5f}`\n"
                        f"• 손절가: `{stop:.5f}`\n"
                        f"• 익절1(1.5R): `{tp1:.5f}`\n"
                        f"• 익절2(3.0R): `{tp2:.5f}`\n"
                        f"\n"
                        f"• 계좌위험: `{risk_usd:.2f} USDT` ({int(CONFIG['risk_fraction']*100)}%)\n"
                        f"• 포지션 수량: `{size_units:.2f}` {CONFIG['symbol'].split('/')[0]}\n"
                        f"• 추정 레버리지: `x{implied_leverage:.2f}`\n"
                        f"\n"
                        f"_ATR×{CONFIG['atr_mult']}, N={CONFIG['n_break']}, FVG 무효 {int(CONFIG['fvg_invalidation']*100)}%, 재진입 최대 {CONFIG['max_reentries_per_ob']}회_"
                    )
                    tg_send_message(text)

            time.sleep(poll_s)

        except Exception as e:
            tg_send_message(f"⚠️ Bot 오류: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
