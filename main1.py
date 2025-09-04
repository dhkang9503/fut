import os, time, requests, math
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import io
import matplotlib.pyplot as plt

# ===== 텔레그램 =====
TELEGRAM_TOKEN = os.environ.get("OKX_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("OKX_TELEGRAM_CHAT_ID")
def tg(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
        )
    except Exception as e:
        print("TG error:", e)

GREEN = "#4DD2E6"  # 양봉
RED   = "#FC495C"  # 음봉

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
    return df[["open_time","open","high","low","close","volume"]]

def make_chart_png(df, symbol, entry_time_utc, exit_time_utc, entry_px, exit_px):
    """검정 배경 + 지정 색상 캔들, 진입/청산 마커 포함 PNG bytes 반환"""
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6), sharex=True)
    # 배경
    for ax in (ax1, ax2):
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, color="white")
    fig.patch.set_facecolor("black")

    # 캔들
    for _, r in df.iterrows():
        color = GREEN if r["close"] >= r["open"] else RED
        ax1.plot([r["open_time"], r["open_time"]], [r["low"], r["high"]], color=color, linewidth=1)
        body_low  = min(r["open"], r["close"])
        body_high = max(r["open"], r["close"])
        ax1.add_patch(plt.Rectangle((r["open_time"], body_low),
                                    pd.Timedelta(minutes=0.8),
                                    body_high - body_low,
                                    color=color, alpha=0.9, linewidth=0))

    # 진입/청산 마커
    ax1.scatter(entry_time_utc, entry_px, s=80, marker="^", color=GREEN, edgecolors="white", linewidths=0.5, zorder=5, label="Entry")
    ax1.scatter(exit_time_utc,  exit_px,  s=80, marker="v", color=RED,   edgecolors="white", linewidths=0.5, zorder=5, label="Exit")
    ax1.legend(facecolor="black", edgecolor="white", labelcolor="white")

    # 거래량
    ax2.bar(df["open_time"], df["volume"], width=0.8/1440, align="center", alpha=0.8, color="white")
    ax1.set_title(f"{symbol} 1m  (Entry~Exit view)", color="white")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

def tg_photo(png_bytes, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    files = {"photo": ("chart.png", png_bytes, "image/png")}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files, timeout=15)

# ===== 설정값 (필요시 조정) =====
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1m"
LIMIT = 180                 # 최근 180분 사용
VOL_SMA_N = 20              # 거래량 기준선
VOL_SPIKE_MULT = 3.5        # "거래량 터짐" 배수 기준
SR_LOOKBACK = 60            # 지지/저항 탐색 분
SR_TOL = 0.002              # 지지/저항 근접 허용(0.2%)
WICK_RATIO_MIN = 0.45       # 윗/아랫꼬리 비율 최소
STOP_PAD = 0.0015           # 무효화(손절) 버퍼 0.15%
KST = ZoneInfo("Asia/Seoul")

# ===== 상태 =====
state = {
    "open_symbol": None,        # 포지션 보유 코인
    "side": None,               # "long" or "short"
    "entry_price": None,
    "entry_time_utc": None,
    "ref_low": None,            # 진입 근거 스윙저/스윙고
    "ref_high": None
}

# ===== 데이터 수집 =====
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

# ===== 시그널 로직 =====
def near(x, ref, tol=SR_TOL):
    return abs(x - ref) / ref <= tol

def entry_signal(df):
    """마감된 최신봉 기준으로 역추세 진입 시그널 계산"""
    if len(df) < max(VOL_SMA_N+1, SR_LOOKBACK+5): return None
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

def exit_signal(df, side):
    """보유중일 때 청산: 해당 방향 수익 확정용 거래량 피크 봉"""
    if len(df) < VOL_SMA_N+1: return None
    last = df.iloc[-1]
    prevN = df.iloc[-(VOL_SMA_N+1):-1]
    v_sma = prevN["volume"].mean()
    if v_sma == 0: return None
    v_mult = last["volume"] / v_sma

    open_, close = last["open"], last["close"]
    is_green = close >= open_
    # 롱이면 대량거래 *상승* 봉에서 청산, 숏이면 대량거래 *하락* 봉에서 청산
    if side == "long" and v_mult >= VOL_SPIKE_MULT and is_green:
        return {"type":"tp","reason":f"vol x{v_mult:.1f} green spike"}
    if side == "short" and v_mult >= VOL_SPIKE_MULT and not is_green:
        return {"type":"tp","reason":f"vol x{v_mult:.1f} red spike"}
    return None

def stop_out(df, side, ref_low, ref_high):
    """무효화(손절): 롱은 스윙저 하회, 숏은 스윙고 상회"""
    last = df.iloc[-1]
    px = last["close"]
    if side == "long":
        stop = ref_low * (1 - STOP_PAD)
        if px < stop: return {"type":"sl","reason":f"lost swing low {ref_low:.2f}"}
    if side == "short":
        stop = ref_high * (1 + STOP_PAD)
        if px > stop: return {"type":"sl","reason":f"broke swing high {ref_high:.2f}"}
    return None

def fmt_price(symbol, p):
    return f"{p:,.2f}" if symbol.endswith("USDT") else f"{p}"

def now_kst():
    return datetime.now(timezone.utc).astimezone(KST).strftime("%Y-%m-%d %H:%M:%S %Z")

# ===== 메인 루프 =====
def main():
    tg("running scanner...")
    while True:
        # ---- 다음 분 00초까지 대기 (정각 동기화) ----
        now = datetime.now(timezone.utc)
        next_min = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
        time.sleep(max((next_min - now).total_seconds(), 0.0))

        try:
            # 데이터 수집
            dfs = {sym: fetch_klines(sym) for sym in SYMBOLS}

            # 포지션 없으면: 두 코인 신호 중 가장 강한(spike 배수 큰) 한 개만 채택
            if state["open_symbol"] is None:
                cands = []
                for sym in SYMBOLS:
                    sig = entry_signal(dfs[sym])
                    if sig:
                        last = dfs[sym].iloc[-1]
                        cands.append((sig["v_mult"], sym, sig, last))
                if cands:
                    cands.sort(reverse=True, key=lambda x: x[0])  # 가장 강한 신호
                    _, sym, sig, last = cands[0]
                    side = sig["side"]; px = last["close"]
                    state.update({
                        "open_symbol": sym,
                        "side": side,
                        "entry_price": px,
                        "entry_time_utc": last["close_time"].to_pydatetime(),
                        "ref_low": sig["prior_low"],
                        "ref_high": sig["prior_high"]
                    })
                    kst_time = now_kst()
                    tg(f"*[ENTRY]* {sym} {side.upper()} @ {fmt_price(sym, px)}\n"
                       f"{kst_time}\n"
                       f"reason: {sig['reason']}")
                    print("ENTRY:", sym, side, px, sig["reason"])

            # 포지션 있으면: 청산(TP/SL) 감시
            else:
                sym = state["open_symbol"]
                df = dfs.get(sym)
                if df is None: continue
                # 우선 손절 체크
                so = stop_out(df, state["side"], state["ref_low"], state["ref_high"])
                if so:
                    px = df.iloc[-1]["close"]
                    tg(f"*[EXIT-SL]* {sym} {state['side'].upper()} @ {fmt_price(sym, px)}\n"
                       f"{now_kst()}\nreason: {so['reason']}")
                    start_utc = state["entry_time_utc"] - timedelta(minutes=10)
                    end_utc   = df.iloc[-1]["close_time"].to_pydatetime() + timedelta(minutes=1)
                    cut = fetch_range_1m(sym, start_utc, end_utc)
                    png = make_chart_png(cut, sym, state["entry_time_utc"], df.iloc[-1]["close_time"].to_pydatetime(),
                        state["entry_price"], px)
                    tg_photo(png, caption=f"{sym} EXIT-SL chart")
                    print("EXIT-SL:", sym, px, so["reason"])
                    state.update({"open_symbol":None,"side":None,"entry_price":None,
                                  "entry_time_utc":None,"ref_low":None,"ref_high":None})
                    continue
                # 이익실현 체크
                ex = exit_signal(df, state["side"])
                if ex:
                    px = df.iloc[-1]["close"]
                    tg(f"*[[EXIT-TP]]* {sym} {state['side'].upper()} @ {fmt_price(sym, px)}\n"
                       f"{now_kst()}\nreason: {ex['reason']}")
                    start_utc = state["entry_time_utc"] - timedelta(minutes=10)
                    end_utc   = df.iloc[-1]["close_time"].to_pydatetime() + timedelta(minutes=1)
                    cut = fetch_range_1m(sym, start_utc, end_utc)
                    png = make_chart_png(cut, sym, state["entry_time_utc"], df.iloc[-1]["close_time"].to_pydatetime(),
                        state["entry_price"], px)
                    tg_photo(png, caption=f"{sym} EXIT-TP chart")
                    print("EXIT-TP:", sym, px, ex["reason"])
                    state.update({"open_symbol":None,"side":None,"entry_price":None,
                                  "entry_time_utc":None,"ref_low":None,"ref_high":None})

        except Exception as e:
            tg(f"loop error: {e}")

if __name__ == "__main__":
    main()
