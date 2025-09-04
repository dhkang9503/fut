import os, time, requests, math, io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# =========================
# 환경/텔레그램
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
SYMBOLS = ["BTCUSDT", "ETHUSDT"]  # 동시 진입 금지(한 번에 1종목만)
INTERVAL = "1m"
LIMIT = 180                   # 최근 180분 사용

# 엔트리 로직(방장 성향)
VOL_SMA_N = 20                # 거래량 기준선
VOL_SPIKE_MULT = 3.5          # "거래량 터짐" 배수 기준
SR_LOOKBACK = 60              # 지지/저항 탐색 분
SR_TOL = 0.002                # 지지/저항 근접 허용(±0.2%)
WICK_RATIO_MIN = 0.45         # 윗/아랫꼬리 비율 최소
STOP_PAD = 0.0015             # 스윙 고/저 무효화 버퍼(±0.15%)

# 단타형 청산
PARTIAL_TP_R_MULT = 1.0       # 1R 수익 도달 시 부분 익절
PARTIAL_TP_FALLBACK_PCT = 0.003  # R 계산 불가 시 +0.3%에서 부분 익절
MOVE_SL_TO_BE = True          # 부분 익절 후 본전 방어 활성화
BE_PAD = 0.0                  # 본전 방어 여유(0이면 정확히 본전)

# 시각화 색
GREEN = "#4DD2E6"  # 양봉
RED   = "#FC495C"  # 음봉

# 시간대
KST = ZoneInfo("Asia/Seoul")

# =========================
# 상태
# =========================
state = {
    "open_symbol": None,
    "side": None,                 # "long" or "short"
    "entry_price": None,
    "entry_time_utc": None,
    "ref_low": None,              # 진입 근거 스윙저/스윙고
    "ref_high": None,

    # 단타형 청산용
    "risk_R": None,               # 진입 시 계산한 1R(%) – 없으면 None
    "ptp_done": False,            # 부분 익절 완료 여부
    "be_active": False            # 본전 방어 활성화 여부
}

# =========================
# 도우미
# =========================
def now_kst():
    return datetime.now(timezone.utc).astimezone(KST).strftime("%Y-%m-%d %H:%M:%S %Z")

def fmt_price(symbol, p):
    return f"{p:,.2f}" if symbol.endswith("USDT") else f"{p}"

def near(x, ref, tol=SR_TOL):
    return abs(x - ref) / ref <= tol

# =========================
# 데이터 수집
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

def fetch_range_1m(symbol, start_utc, end_utc):
    """start_utc~end_utc 구간 1분봉 재조회"""
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
# 시그널 로직
# =========================
def entry_signal(df):
    """마감된 최신봉 기준으로 역추세 진입 시그널 계산"""
    if len(df) < max(VOL_SMA_N+1, SR_LOOKBACK+5):
        return None
    last = df.iloc[-1]
    prevN = df.iloc[-(VOL_SMA_N+1):-1]
    v_sma = prevN["volume"].mean()
    if v_sma == 0:
        return None
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

    # SHORT: 거래량 피크 + 저항 근접 + 윗꼬리 두드러짐 + 양봉
    short_ok = (
        v_mult >= VOL_SPIKE_MULT and
        (high >= prior_high or near(high, prior_high)) and
        up_ratio >= WICK_RATIO_MIN and
        is_green
    )
    # LONG: 거래량 피크 + 지지 근접 + 아랫꼬리 두드러짐 + 음봉
    long_ok = (
        v_mult >= VOL_SPIKE_MULT and
        (low <= prior_low or near(low, prior_low)) and
        lo_ratio >= WICK_RATIO_MIN and
        not is_green
    )

    if short_ok:
        return {"side":"short","reason":f"vol x{v_mult:.1f}, near R({prior_high:.2f}), upper wick {up_ratio:.2f}",
                "v_mult": v_mult, "prior_high": prior_high, "prior_low": prior_low}
    if long_ok:
        return {"side":"long","reason":f"vol x{v_mult:.1f}, near S({prior_low:.2f}), lower wick {lo_ratio:.2f}",
                "v_mult": v_mult, "prior_high": prior_high, "prior_low": prior_low}
    return None

def compute_risk_R(entry_px, side, ref_low, ref_high):
    """
    진입가와 스윙 고/저 기반 손절선으로 1R(%) 계산. 불능(음수/0)이면 None.
    """
    if side == "long":
        stop = ref_low * (1 - STOP_PAD)
        R = (entry_px - stop) / entry_px
    else:  # short
        stop = ref_high * (1 + STOP_PAD)
        R = (stop - entry_px) / entry_px
    return R if R and R > 0 else None

def pnl_pct(side, entry_px, cur_px):
    """현재 PnL 퍼센트(진입가 기준)를 +면 이익, -면 손실로 반환"""
    p = (cur_px - entry_px) / entry_px
    return p if side == "long" else -p

def partial_tp_needed(side, entry_px, cur_px, risk_R):
    """
    부분 익절 조건:
      - risk_R 있으면 PnL >= PARTIAL_TP_R_MULT * risk_R
      - 없으면 PnL >= PARTIAL_TP_FALLBACK_PCT
    """
    p = pnl_pct(side, entry_px, cur_px)
    if risk_R and p >= PARTIAL_TP_R_MULT * risk_R:
        return True
    if (not risk_R) and p >= PARTIAL_TP_FALLBACK_PCT:
        return True
    return False

def dynamic_stop_price_for_be(entry_px, side):
    """본전 방어용 스탑 가격 산출"""
    if side == "long":
        return entry_px * (1 - BE_PAD)
    else:
        return entry_px * (1 + BE_PAD)

def stop_out(df, side, ref_low, ref_high, entry_px=None, be_active=False):
    """
    무효화(손절) 로직:
      1) be_active면 '본전 방어' 우선
      2) 아니면 스윙 고/저 기반 무효화
    """
    last = df.iloc[-1]
    px = last["close"]

    # 본전 방어
    if be_active and entry_px is not None:
        be_stop = dynamic_stop_price_for_be(entry_px, side)
        if (side == "long" and px < be_stop) or (side == "short" and px > be_stop):
            return {"type":"sl", "reason": f"breakeven stop {be_stop:.2f}"}

    # 기본 무효화
    if side == "long":
        stop = ref_low * (1 - STOP_PAD)
        if px < stop:
            return {"type":"sl","reason":f"lost swing low {ref_low:.2f}"}
    else:
        stop = ref_high * (1 + STOP_PAD)
        if px > stop:
            return {"type":"sl","reason":f"broke swing high {ref_high:.2f}"}
    return None

def exit_signal(df, side, ptp_done=False):
    """
    TP 로직:
      - 기본: 기존과 동일(거래량 x 배수 + 방향색 일치)
      - 부분 익절 이후(ptp_done=True): 색깔 무시, v_spike만으로 청산
    """
    if len(df) < VOL_SMA_N+1:
        return None
    last = df.iloc[-1]
    prevN = df.iloc[-(VOL_SMA_N+1):-1]
    v_sma = prevN["volume"].mean()
    if v_sma == 0:
        return None
    v_mult = last["volume"] / v_sma

    open_, close = last["open"], last["close"]
    is_green = close >= open_

    if ptp_done:
        if v_mult >= VOL_SPIKE_MULT:
            return {"type":"tp", "reason": f"vol x{v_mult:.1f} spike (post-PTP)"}
        return None

    if side == "long" and v_mult >= VOL_SPIKE_MULT and is_green:
        return {"type":"tp","reason":f"vol x{v_mult:.1f} green spike"}
    if side == "short" and v_mult >= VOL_SPIKE_MULT and not is_green:
        return {"type":"tp","reason":f"vol x{v_mult:.1f} red spike"}
    return None

# =========================
# 차트 생성 (KST, 거래량 색상=캔들 색상)
# =========================
def make_chart_png(df, symbol, entry_time_utc, exit_time_utc, entry_px, exit_px):
    """
    - 배경: 검정
    - 캔들: 양봉 #4DD2E6, 음봉 #FC495C
    - 거래량: 캔들과 동일 색상
    - x축: 한국시간(KST) HH:MM
    """
    df = df.copy()
    df["kst_time"] = df["open_time"].dt.tz_convert(KST)
    entry_kst = entry_time_utc.astimezone(KST)
    exit_kst  = exit_time_utc.astimezone(KST)

    x = df["kst_time"].map(mdates.date2num)
    colors = [(GREEN if c >= o else RED) for o, c in zip(df["open"], df["close"])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 배경/그리드/축색
    for ax in (ax1, ax2):
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, color="white")
    fig.patch.set_facecolor("black")

    # 캔들
    w = 0.8 / 1440.0  # 0.8분 폭(일 단위 눈금)
    for xi, o, h, l, c, col in zip(x, df["open"], df["high"], df["low"], df["close"], colors):
        ax1.plot([xi, xi], [l, h], color=col, linewidth=1)
        body_low, body_high = min(o, c), max(o, c)
        ax1.add_patch(plt.Rectangle((xi, body_low), w, body_high - body_low,
                                    color=col, alpha=0.9, linewidth=0))

    # 거래량 (캔들과 동일 색)
    ax2.bar(x, df["volume"], width=w, align="center", alpha=0.9, color=colors)

    # 진입/청산 마커
    ax1.scatter(mdates.date2num(entry_kst), entry_px, s=80, marker="^",
                color=GREEN, edgecolors="white", linewidths=0.5, zorder=5, label="Entry")
    ax1.scatter(mdates.date2num(exit_kst),  exit_px,  s=80, marker="v",
                color=RED,   edgecolors="white", linewidths=0.5, zorder=5, label="Exit")
    ax1.legend(facecolor="black", edgecolor="white", labelcolor="white")

    # 축/포맷
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
    tg("running scanner...")
    while True:
        # 다음 분 00초까지 대기(정각 동기화)
        now = datetime.now(timezone.utc)
        next_min = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
        time.sleep(max((next_min - now).total_seconds(), 0.0))

        try:
            # 데이터 수집
            dfs = {sym: fetch_klines(sym) for sym in SYMBOLS}

            # 포지션 없으면: 두 코인 중 가장 강한 신호(거래량 배수 큰 것) 1개만 진입
            if state["open_symbol"] is None:
                candidates = []
                for sym in SYMBOLS:
                    sig = entry_signal(dfs[sym])
                    if sig:
                        last = dfs[sym].iloc[-1]
                        candidates.append((sig["v_mult"], sym, sig, last))
                if candidates:
                    candidates.sort(reverse=True, key=lambda x: x[0])
                    _, sym, sig, last = candidates[0]
                    side = sig["side"]; px = last["close"]
                    state.update({
                        "open_symbol": sym,
                        "side": side,
                        "entry_price": px,
                        "entry_time_utc": last["close_time"].to_pydatetime(),
                        "ref_low": sig["prior_low"],
                        "ref_high": sig["prior_high"],
                        # 단타형
                        "risk_R": compute_risk_R(px, side, sig["prior_low"], sig["prior_high"]),
                        "ptp_done": False,
                        "be_active": False
                    })
                    tg(f"*[ENTRY]* {sym} {side.upper()} @ {fmt_price(sym, px)}\n"
                       f"{now_kst()}\nreason: {sig['reason']}")
                    print("ENTRY:", sym, side, px, sig["reason"])

            # 포지션 있으면: SL/부분TP/TP 감시
            else:
                sym = state["open_symbol"]
                df = dfs.get(sym)
                if df is None:
                    continue

                # 1) 손절(본전 방어 포함)
                so = stop_out(df, state["side"], state["ref_low"], state["ref_high"],
                              entry_px=state["entry_price"], be_active=state["be_active"])
                if so:
                    px = df.iloc[-1]["close"]
                    tg(f"*[EXIT-SL]* {sym} {state['side'].upper()} @ {fmt_price(sym, px)}\n"
                       f"{now_kst()}\nreason: {so['reason']}")
                    # 차트 전송: 진입 10분 전 ~ 청산 1분 후
                    start_utc = state["entry_time_utc"] - timedelta(minutes=10)
                    end_utc   = df.iloc[-1]["close_time"].to_pydatetime() + timedelta(minutes=1)
                    cut = fetch_range_1m(sym, start_utc, end_utc)
                    png = make_chart_png(cut, sym, state["entry_time_utc"],
                                         df.iloc[-1]["close_time"].to_pydatetime(),
                                         state["entry_price"], px)
                    tg_photo(png, caption=f"{sym} EXIT-SL chart")
                    print("EXIT-SL:", sym, px, so["reason"])
                    # 상태 리셋
                    state.update({"open_symbol":None,"side":None,"entry_price":None,
                                  "entry_time_utc":None,"ref_low":None,"ref_high":None,
                                  "risk_R":None,"ptp_done":False,"be_active":False})
                    continue

                # 2) 부분 익절 + 본전 이동
                cur_px = df.iloc[-1]["close"]
                if not state["ptp_done"] and partial_tp_needed(
                        state["side"], state["entry_price"], cur_px, state["risk_R"]):
                    tg(f"*[[PARTIAL-TP]]* {sym} {state['side'].upper()} @ {fmt_price(sym, cur_px)}\n"
                       f"{now_kst()}\nPnL: {pnl_pct(state['side'], state['entry_price'], cur_px)*100:.2f}%"
                       + (f" | R_used: {state['risk_R']:.4f}" if state["risk_R"] else " | no-R"))
                    print("PARTIAL-TP:", sym, cur_px)
                    if MOVE_SL_TO_BE:
                        state["be_active"] = True
                    state["ptp_done"] = True

                # 3) 최종 익절
                ex = exit_signal(df, state["side"], ptp_done=state["ptp_done"])
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
                    # 상태 리셋
                    state.update({"open_symbol":None,"side":None,"entry_price":None,
                                  "entry_time_utc":None,"ref_low":None,"ref_high":None,
                                  "risk_R":None,"ptp_done":False,"be_active":False})

        except Exception as e:
            tg(f"loop error: {e}")

if __name__ == "__main__":
    main()
