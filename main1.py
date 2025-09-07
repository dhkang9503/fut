import os, time, io, math, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# =========================
# 환경 / 텔레그램
# =========================
TELEGRAM_TOKEN = os.environ.get("OKX_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("OKX_TELEGRAM_CHAT_ID")

def tg(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=15
        )
    except Exception as e:
        print("TG error:", e)

def tg_photo(png_bytes, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        files = {"photo": ("chart.png", png_bytes, "image/png")}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files, timeout=20)
    except Exception as e:
        print("TG photo error:", e)

# =========================
# 설정값
# =========================
SYMBOLS = ["BTCUSDT", "ETHUSDT"]   # 동시 진입 금지
INTERVAL = "1m"
LIMIT = 180                        # 최근 180분 사용

# --- 엔트리(방장 성향 기본값)
VOL_SMA_N = 20                     # 거래량 기준선 길이
VOL_SPIKE_MULT = 3.5               # "거래량 터짐" 배수 기준
SR_TOL = 0.002                     # 지지/저항 근접 허용(±0.2%)
WICK_RATIO_MIN = 0.45              # 윗/아랫꼬리 비율 최소

# --- SR/STOP(기본/보수 모드 값: 아래 Adaptive에서 자동 전환)
SR_LOOKBACK_BASE = 60              # 기본: 60분 SR
SR_LOOKBACK_CONS = 180             # 보수: 180분 SR
STOP_PAD_BASE = 0.0015             # 기본: ±0.15%
STOP_PAD_CONS = 0.0030             # 보수: ±0.30%

# --- 수익 극대화(핵심 추가)
PARTIAL_SIZE = 0.5                 # 부분익절 비중(신호 봇이므로 안내만)
PARTIAL_TP_R_MULT = 1.0            # 1R 도달 시 부분익절
PARTIAL_TP_FALLBACK_PCT = 0.003    # R 계산 불가 시 +0.3%에서 부분익절
MOVE_SL_TO_BE = True               # 부분익절 후 본전 방어 가동
BE_PAD = 0.0                       # 본전 방어 여유

FINAL_TP_R = 2.5                   # 최종 TP 목표(RR). 트레일과 병행
TRAIL_N = 7                        # 최근 N봉 저/고 기반 트레일
USE_EMA_TRAIL = True               # EMA 트레일 사용
EMA_TRAIL = 20                     # EMA 트레일 길이

# --- Adaptive(자동 보수화 + 돌파 감지)
EMA_FAST = 20
EMA_SLOW = 50
ATR_N = 14
STRONG_TREND_SLOPE = 0.0006        # 분당 기울기 임계치(0.06%)
HIGH_ATR_PCT = 0.004               # ATR/price >= 0.4%면 고변동
CONSEC_VOL_SPIKES = 2              # 연속 스파이크 N개면 돌파 성격
SL_COOLDOWN_MIN = 5                # 손절 직후 N분 진입 금지
RECLAIM_CONFIRM = True             # 보수 모드에서 재전유 확인

# --- 돌파형(continuation) 판정 파라미터
BODY_EXPANSION = 0.60              # 바디/전체(range) 비율이 크면 돌파형
HOLD_FOR_BREAKOUT_BARS = 2         # 돌파형 감지 직후 초기 TP 스킵할 최소 봉 수

# --- 시각화
GREEN = "#4DD2E6"  # 양봉
RED   = "#FC495C"  # 음봉
KST   = ZoneInfo("Asia/Seoul")

# =========================
# 전역(Adaptive 동적 적용)
# =========================
SR_LOOKBACK = SR_LOOKBACK_BASE
STOP_PAD    = STOP_PAD_BASE

# =========================
# 상태
# =========================
state = {
    "open_symbol": None,
    "side": None,
    "entry_price": None,
    "entry_time_utc": None,
    "ref_low": None,
    "ref_high": None,

    # RR/부분익절/트레일
    "risk_R": None,
    "ptp_done": False,
    "be_active": False,
    "hold_bars": 0,                # 보유한 봉 수(진입 후)
    "breakout_mode": False,        # 돌파형 보유 모드(초기 TP 회피)
    "breakout_armed_bars": 0,      # 돌파 보유 모드 경과 봉 수

    # 쿨다운
    "last_sl_time": None
}

# =========================
# 유틸
# =========================
def now_kst():
    return datetime.now(timezone.utc).astimezone(KST).strftime("%Y-%m-%d %H:%M:%S %Z")

def fmt_price(symbol, p):
    return f"{p:,.2f}" if symbol.endswith("USDT") else f"{p}"

def near(x, ref, tol=SR_TOL):
    return abs(x - ref) / ref <= tol

# =========================
# 데이터 수집/지표
# =========================
def fetch_klines(symbol, limit=LIMIT):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": INTERVAL, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    arr = r.json()
    df = pd.DataFrame(arr, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","n_trades","tbav","tbqv","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df

def add_indicators(df):
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_N, adjust=False).mean()
    return df

def fetch_range_1m(symbol, start_utc, end_utc):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol, "interval": "1m",
        "startTime": int(start_utc.timestamp() * 1000),
        "endTime":   int(end_utc.timestamp()   * 1000)
    }
    r = requests.get(url, params=params, timeout=10); r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","n_trades","tbav","tbqv","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df[["open_time","open","high","low","close","volume","close_time"]]

# =========================
# Adaptive 판단
# =========================
def slope_per_min(series):
    n = min(len(series), EMA_FAST)
    if n < 3: return 0.0
    y = series.tail(n).values
    x = np.arange(n)
    k = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean())**2).sum()
    return 0.0 if y.mean() == 0 else k / y.mean()  # 비율/분

def conservative_mode(df):
    """강한 추세/고변동/연속 스파이크 → 보수 모드 True"""
    last = df.iloc[-1]
    s = slope_per_min(df["ema_fast"])
    strong_trend = (abs(s) >= STRONG_TREND_SLOPE) and \
                   ((last["ema_fast"] > last["ema_slow"]) or (last["ema_fast"] < last["ema_slow"]))
    high_vol = (last["atr"] / last["close"]) >= HIGH_ATR_PCT

    v_sma = df["volume"].rolling(VOL_SMA_N).mean().shift(1)
    spikes = (df["volume"] > (v_sma * VOL_SPIKE_MULT)).astype(int)
    consec = int(spikes.tail(CONSEC_VOL_SPIKES).sum() >= CONSEC_VOL_SPIKES)

    return bool(strong_trend or high_vol or consec)

def can_enter(sym_df):
    """손절 직후 쿨다운"""
    if state["last_sl_time"] is not None:
        if sym_df.iloc[-1]["close_time"].to_pydatetime() < state["last_sl_time"] + timedelta(minutes=SL_COOLDOWN_MIN):
            return False
    return True

def reclaim_confirm(df, side, level):
    """보수 모드에서만 사용: 레벨 재전유 확인"""
    if len(df) < 3: return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if side == "long":
        return (last["close"] > level) and (last["high"] >= max(prev["high"] * 0.999, prev["high"]))
    else:
        return (last["close"] < level) and (last["low"]  <= min(prev["low"]  * 1.001, prev["low"]))

def detect_breakout_candle(row):
    """단일 봉이 돌파형인지: 바디 확장 + 볼륨 스파이크"""
    rng = max(row["high"] - row["low"], 1e-12)
    body = abs(row["close"] - row["open"])
    body_ratio = body / rng
    return (body_ratio >= BODY_EXPANSION)

# =========================
# 시그널 로직
# =========================
def entry_signal(df):
    """마감된 최신봉 기준 역추세 진입 신호"""
    if len(df) < max(VOL_SMA_N+1, SR_LOOKBACK+5):
        return None
    last = df.iloc[-1]
    prevN = df.iloc[-(VOL_SMA_N+1):-1]
    v_sma = prevN["volume"].mean()
    if v_sma == 0: return None
    v_mult = last["volume"] / v_sma

    high = last["high"]; low = last["low"]; open_ = last["open"]; close = last["close"]
    rng = max(high - low, 1e-9)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    up_ratio = upper_wick / rng
    lo_ratio = lower_wick / rng
    is_green = close >= open_

    prior = df.iloc[-(SR_LOOKBACK+1):-1]
    prior_high = prior["high"].max()
    prior_low  = prior["low"].min()

    short_ok = (v_mult >= VOL_SPIKE_MULT) and ((high >= prior_high) or near(high, prior_high)) \
               and (up_ratio >= WICK_RATIO_MIN) and is_green
    long_ok  = (v_mult >= VOL_SPIKE_MULT) and ((low  <= prior_low)  or near(low, prior_low)) \
               and (lo_ratio >= WICK_RATIO_MIN) and (not is_green)

    if short_ok:
        return {"side":"short","reason":f"vol x{v_mult:.1f}, near R({prior_high:.2f}), upper wick {up_ratio:.2f}",
                "v_mult": v_mult, "prior_high": prior_high, "prior_low": prior_low,
                "is_breakout": detect_breakout_candle(last)}
    if long_ok:
        return {"side":"long","reason":f"vol x{v_mult:.1f}, near S({prior_low:.2f}), lower wick {lo_ratio:.2f}",
                "v_mult": v_mult, "prior_high": prior_high, "prior_low": prior_low,
                "is_breakout": detect_breakout_candle(last)}
    return None

def compute_risk_R(entry_px, side, ref_low, ref_high):
    """진입가 vs 스윙 고/저 기반 1R(%) 계산"""
    if side == "long":
        stop = ref_low * (1 - STOP_PAD)
        R = (entry_px - stop) / entry_px
    else:
        stop = ref_high * (1 + STOP_PAD)
        R = (stop - entry_px) / entry_px
    return R if R and R > 0 else None

def pnl_pct(side, entry_px, cur_px):
    p = (cur_px - entry_px) / entry_px
    return p if side == "long" else -p

def partial_tp_needed(side, entry_px, cur_px, risk_R):
    p = pnl_pct(side, entry_px, cur_px)
    if risk_R and p >= PARTIAL_TP_R_MULT * risk_R: return True
    if (not risk_R) and p >= PARTIAL_TP_FALLBACK_PCT: return True
    return False

def dynamic_stop_price_for_be(entry_px, side):
    return entry_px * (1 - BE_PAD) if side == "long" else entry_px * (1 + BE_PAD)

def trail_stop_hit(df, side, entry_px):
    """트레일링 스탑: 최근 N봉 저/고 + EMA 교차 둘 중 하나라도 이탈하면 True"""
    if len(df) < max(TRAIL_N+1, EMA_TRAIL+1): return False
    sub = df.iloc[-TRAIL_N:]
    last = df.iloc[-1]
    if side == "long":
        bar_trail = sub["low"].min()
        cond_bar = last["close"] < bar_trail
        if USE_EMA_TRAIL:
            ema = df["close"].ewm(span=EMA_TRAIL, adjust=False).mean().iloc[-1]
            cond_ema = last["close"] < ema
            return bool(cond_bar or cond_ema)
        return bool(cond_bar)
    else:
        bar_trail = sub["high"].max()
        cond_bar = last["close"] > bar_trail
        if USE_EMA_TRAIL:
            ema = df["close"].ewm(span=EMA_TRAIL, adjust=False).mean().iloc[-1]
            cond_ema = last["close"] > ema
            return bool(cond_bar or cond_ema)
        return bool(cond_bar)

def final_tp_hit_by_R(side, entry_px, cur_px, risk_R):
    if not risk_R: return False
    p = pnl_pct(side, entry_px, cur_px)
    return p >= FINAL_TP_R * risk_R

def stop_out(df, side, ref_low, ref_high, entry_px=None, be_active=False):
    """본전 방어 우선 → 스윙 무효화"""
    last = df.iloc[-1]
    px = last["close"]
    if be_active and entry_px is not None:
        be_stop = dynamic_stop_price_for_be(entry_px, side)
        if (side == "long" and px < be_stop) or (side == "short" and px > be_stop):
            return {"type":"sl", "reason": f"breakeven stop {be_stop:.2f}"}
    if side == "long":
        stop = ref_low * (1 - STOP_PAD)
        if px < stop: return {"type":"sl","reason":f"lost swing low {ref_low:.2f}"}
    else:
        stop = ref_high * (1 + STOP_PAD)
        if px > stop: return {"type":"sl","reason":f"broke swing high {ref_high:.2f}"}
    return None

def exit_signal_spike(df, side, ptp_done=False, allow_early_tp=True):
    """볼륨 스파이크 기반 TP (초기 돌파 보유모드면 early TP 금지)"""
    if len(df) < VOL_SMA_N+1: return None
    last = df.iloc[-1]
    prevN = df.iloc[-(VOL_SMA_N+1):-1]
    v_sma = prevN["volume"].mean()
    if v_sma == 0: return None
    v_mult = last["volume"] / v_sma
    open_, close = last["open"], last["close"]
    is_green = close >= open_
    if not allow_early_tp:
        return None  # 돌파 보유 모드에서는 초반 스파이크 TP 비활성화
    if ptp_done:
        if v_mult >= VOL_SPIKE_MULT:
            return {"type":"tp","reason":f"vol x{v_mult:.1f} spike (post-PTP)"}
        return None
    if side == "long" and v_mult >= VOL_SPIKE_MULT and is_green:
        return {"type":"tp","reason":f"vol x{v_mult:.1f} green spike"}
    if side == "short" and v_mult >= VOL_SPIKE_MULT and not is_green:
        return {"type":"tp","reason":f"vol x{v_mult:.1f} red spike"}
    return None

# =========================
# 차트(KST, 거래량=캔들 색)
# =========================
def make_chart_png(df, symbol, entry_time_utc, exit_time_utc, entry_px, exit_px):
    df = df.copy()
    df["kst_time"] = df["open_time"].dt.tz_convert(KST)
    entry_kst = entry_time_utc.astimezone(KST)
    exit_kst  = exit_time_utc.astimezone(KST)

    x = df["kst_time"].map(mdates.date2num)
    colors = [(GREEN if c >= o else RED) for o, c in zip(df["open"], df["close"])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for ax in (ax1, ax2):
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, color="white")
    fig.patch.set_facecolor("black")

    w = 0.8 / 1440.0
    for xi, o, h, l, c, col in zip(x, df["open"], df["high"], df["low"], df["close"], colors):
        ax1.plot([xi, xi], [l, h], color=col, linewidth=1)
        body_low, body_high = min(o, c), max(o, c)
        ax1.add_patch(plt.Rectangle((xi, body_low), w, body_high - body_low,
                                    color=col, alpha=0.9, linewidth=0))

    ax2.bar(x, df["volume"], width=w, align="center", alpha=0.9, color=colors)

    ax1.scatter(mdates.date2num(entry_kst), entry_px, s=80, marker="^",
                color=GREEN, edgecolors="white", linewidths=0.5, zorder=5, label="Entry")
    ax1.scatter(mdates.date2num(exit_kst),  exit_px,  s=80, marker="v",
                color=RED,   edgecolors="white", linewidths=0.5, zorder=5, label="Exit")
    ax1.legend(facecolor="black", edgecolor="white", labelcolor="white")

    ax1.set_title(f"{symbol} 1m (KST)", color="white")
    ax1.xaxis_date(); ax2.xaxis_date()
    formatter = mdates.DateFormatter('%H:%M', tz=KST)
    ax2.xaxis.set_major_formatter(formatter)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

# =========================
# 메인 루프
# =========================
def main():
    tg("running scanner with extended profit mode...")
    global SR_LOOKBACK, STOP_PAD

    while True:
        # 다음 분 00초까지 대기(정각 동기화)
        now = datetime.now(timezone.utc)
        next_min = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
        time.sleep(max((next_min - now).total_seconds(), 0.0))

        try:
            dfs = {sym: add_indicators(fetch_klines(sym)) for sym in SYMBOLS}

            # 포지션 없으면: 한 종목만 진입
            if state["open_symbol"] is None:
                candidates = []
                for sym in SYMBOLS:
                    df = dfs[sym]
                    if not can_enter(df):
                        continue

                    cons = conservative_mode(df)
                    SR_LOOKBACK = SR_LOOKBACK_CONS if cons else SR_LOOKBACK_BASE
                    STOP_PAD    = STOP_PAD_CONS    if cons else STOP_PAD_BASE

                    sig = entry_signal(df)
                    if sig:
                        # 보수 모드면 재전유 확인
                        if cons and RECLAIM_CONFIRM:
                            level = sig["prior_low"] if sig["side"] == "long" else sig["prior_high"]
                            if not reclaim_confirm(df, sig["side"], level):
                                continue
                        last = df.iloc[-1]
                        candidates.append((sig["v_mult"], sym, sig, last, cons))

                if candidates:
                    candidates.sort(reverse=True, key=lambda x: x[0])
                    _, sym, sig, last, cons = candidates[0]

                    # 진입 시점의 파라미터 고정
                    SR_LOOKBACK = SR_LOOKBACK_CONS if cons else SR_LOOKBACK_BASE
                    STOP_PAD    = STOP_PAD_CONS    if cons else STOP_PAD_BASE

                    side = sig["side"]; px = last["close"]
                    # 돌파형 보유 모드 조건: 연속 스파이크 환경 + 큰 바디
                    breakout_env = cons  # 연속 스파이크/강추세 포함
                    breakout_candle = bool(sig.get("is_breakout", False))
                    breakout_mode = bool(breakout_env and breakout_candle)

                    state.update({
                        "open_symbol": sym,
                        "side": side,
                        "entry_price": px,
                        "entry_time_utc": last["close_time"].to_pydatetime(),
                        "ref_low": sig["prior_low"],
                        "ref_high": sig["prior_high"],

                        "risk_R": compute_risk_R(px, side, sig["prior_low"], sig["prior_high"]),
                        "ptp_done": False,
                        "be_active": False,
                        "hold_bars": 0,

                        "breakout_mode": breakout_mode,
                        "breakout_armed_bars": 0
                    })
                    tg(f"*[ENTRY]* {sym} {side.upper()} @ {fmt_price(sym, px)}\n"
                       f"{now_kst()}\nreason: {sig['reason']}"
                       + (f"\nmode: breakout-hold" if breakout_mode else ""))
                    print("ENTRY:", sym, side, px, sig["reason"], "breakout:", breakout_mode)

            # 포지션 보유: SL/부분TP/트레일/최종TP
            else:
                sym = state["open_symbol"]
                df = dfs.get(sym)
                if df is None: 
                    continue

                # 보유 봉 카운트 갱신
                state["hold_bars"] += 1
                if state["breakout_mode"] and state["breakout_armed_bars"] < HOLD_FOR_BREAKOUT_BARS:
                    state["breakout_armed_bars"] += 1

                # Adaptive: 보유 중에도 STOP_PAD 동적
                cons_hold = conservative_mode(df)
                STOP_PAD = STOP_PAD_CONS if cons_hold else STOP_PAD_BASE

                # 1) 손절(본전 방어 포함)
                so = stop_out(df, state["side"], state["ref_low"], state["ref_high"],
                              entry_px=state["entry_price"], be_active=state["be_active"])
                if so:
                    px = df.iloc[-1]["close"]
                    tg(f"*[EXIT-SL]* {sym} {state['side'].upper()} @ {fmt_price(sym, px)}\n"
                       f"{now_kst()}\nreason: {so['reason']}")
                    # 차트 전송
                    start_utc = state["entry_time_utc"] - timedelta(minutes=10)
                    end_utc   = df.iloc[-1]["close_time"].to_pydatetime() + timedelta(minutes=1)
                    cut = fetch_range_1m(sym, start_utc, end_utc)
                    png = make_chart_png(cut, sym, state["entry_time_utc"],
                                         df.iloc[-1]["close_time"].to_pydatetime(),
                                         state["entry_price"], px)
                    tg_photo(png, caption=f"{sym} EXIT-SL chart")
                    print("EXIT-SL:", sym, px, so["reason"])

                    state["last_sl_time"] = df.iloc[-1]["close_time"].to_pydatetime()
                    state.update({"open_symbol":None,"side":None,"entry_price":None,
                                  "entry_time_utc":None,"ref_low":None,"ref_high":None,
                                  "risk_R":None,"ptp_done":False,"be_active":False,
                                  "hold_bars":0,"breakout_mode":False,"breakout_armed_bars":0})
                    continue

                # 2) 부분익절 + 본전 이동
                cur_px = df.iloc[-1]["close"]
                if not state["ptp_done"] and partial_tp_needed(
                        state["side"], state["entry_price"], cur_px, state["risk_R"]):
                    tg(f"*[[PARTIAL-TP {int(PARTIAL_SIZE*100)}%]]* {sym} {state['side'].upper()} @ {fmt_price(sym, cur_px)}\n"
                       f"{now_kst()}\nPnL: {pnl_pct(state['side'], state['entry_price'], cur_px)*100:.2f}%"
                       + (f" | R_used: {state['risk_R']:.4f}" if state["risk_R"] else " | no-R"))
                    print("PARTIAL-TP:", sym, cur_px)
                    if MOVE_SL_TO_BE:
                        state["be_active"] = True
                    state["ptp_done"] = True

                # 3) 트레일링 스탑 (PTP 이후 추세 수익 극대화)
                trail_hit = state["ptp_done"] and trail_stop_hit(df, state["side"], state["entry_price"])
                # 4) RR 기반 최종 TP
                rr_hit = final_tp_hit_by_R(state["side"], state["entry_price"], cur_px, state["risk_R"])

                if trail_hit or rr_hit:
                    reason = "trail stop" if trail_hit else f"{FINAL_TP_R:.1f}R target"
                    px = df.iloc[-1]["close"]
                    tg(f"*[[EXIT-TP]]* {sym} {state['side'].upper()} @ {fmt_price(sym, px)}\n"
                       f"{now_kst()}\nreason: {reason}")
                    start_utc = state["entry_time_utc"] - timedelta(minutes=10)
                    end_utc   = df.iloc[-1]["close_time"].to_pydatetime() + timedelta(minutes=1)
                    cut = fetch_range_1m(sym, start_utc, end_utc)
                    png = make_chart_png(cut, sym, state["entry_time_utc"],
                                         df.iloc[-1]["close_time"].to_pydatetime(),
                                         state["entry_price"], px)
                    tg_photo(png, caption=f"{sym} EXIT-TP chart")
                    print("EXIT-TP:", sym, px, reason)

                    state.update({"open_symbol":None,"side":None,"entry_price":None,
                                  "entry_time_utc":None,"ref_low":None,"ref_high":None,
                                  "risk_R":None,"ptp_done":False,"be_active":False,
                                  "hold_bars":0,"breakout_mode":False,"breakout_armed_bars":0})
                    continue

                # 5) (옵션) 초기 스파이크 기반 TP — 돌파형이면 일정 봉수 동안 비활성화
                allow_early_tp = not (state["breakout_mode"] and state["breakout_armed_bars"] < HOLD_FOR_BREAKOUT_BARS)
                ex = exit_signal_spike(df, state["side"], ptp_done=state["ptp_done"], allow_early_tp=allow_early_tp)
                if ex:
                    px = df.iloc[-1]["close"]
                    tg(f"*[[EXIT-TP]]* {sym} {state['side'].upper()} @ {fmt_price(sym, px)}\n"
                       f"{now_kst()}\nreason: {ex['reason']}")
                    start_utc = state["entry_time_utc"] - timedelta(minutes=10)
                    end_utc   = df.iloc[-1]["close_time"].to_pydatetime() + timedelta(minutes=1)
                    cut = fetch_range_1m(sym, start_utc, end_utc)
                    png = make_chart_png(cut, sym, state["entry_time_utc"],
                                         df.iloc[-1]["close_time"].to_pydatetime(),
                                         state["entry_price"], px)
                    tg_photo(png, caption=f"{sym} EXIT-TP chart")
                    print("EXIT-TP:", sym, px, ex["reason"])

                    state.update({"open_symbol":None,"side":None,"entry_price":None,
                                  "entry_time_utc":None,"ref_low":None,"ref_high":None,
                                  "risk_R":None,"ptp_done":False,"be_active":False,
                                  "hold_bars":0,"breakout_mode":False,"breakout_armed_bars":0})

        except Exception as e:
            print("loop error:", e)

if __name__ == "__main__":
    main()
