# pip install python-binance pandas numpy requests
from binance.client import Client
from binance import ThreadedWebsocketManager
import pandas as pd, numpy as np, time, requests

# ✅ 텔레그램 알림 완성형 예시
import requests, os

# ───────── 설정 ─────────
TELEGRAM_BOT = os.getenv('TELEGRAM_BOT_TOKEN')  # 봇 토큰
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')                # 채팅방 ID (혹은 @channelusername)
# ────────────────────────

def notify(msg: str):
    """텔레그램 메시지 전송"""
    if not TELEGRAM_BOT or not CHAT_ID:
        print("[알림 생략] " + msg)
        return
    try:
        api = f"https://api.telegram.org/bot{TELEGRAM_BOT}/sendMessage"
        params = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        r = requests.get(api, params=params, timeout=5)
        if r.status_code != 200:
            print(f"텔레그램 전송 실패: {r.text}")
    except Exception as e:
        print("텔레그램 전송 오류:", e)

SYMBOL = "BTCUSDT"
INTERVAL_4H = Client.KLINE_INTERVAL_4HOUR
INIT_LIMIT = 60           # 시작 시 백필할 4H 캔들 개수 (ATR 14 + median 30 충분)
MAX_WIN    = 80           # 롤링 윈도우(여유 버퍼)

K = 2.0
THR_MIN, THR_MAX = 0.002, 0.03
ATR_LEN, ATR_WIN = 14, 30

hist_4h = pd.DataFrame(columns=["open","high","low","close"])
direction, ext_price, ext_time, thr_pct = 0, None, None, None

def ta_atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"]-df["low"]).abs(),
                    (df["high"]-pc).abs(),
                    (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def backfill_4h(client):
    global hist_4h, direction, ext_price, ext_time, thr_pct
    kl = client.get_klines(symbol=SYMBOL, interval=INTERVAL_4H, limit=INIT_LIMIT)
    rows = []
    for o in kl:
        ts = pd.to_datetime(o[0], unit="ms", utc=True)
        rows.append([ts, float(o[1]), float(o[2]), float(o[3]), float(o[4])])
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close"]).set_index("ts")
    hist_4h = df.copy()

    # 임계값 계산
    if len(hist_4h) >= ATR_WIN + ATR_LEN:
        atr = ta_atr(hist_4h, ATR_LEN)
        atr_pct = (atr / hist_4h["close"]).tail(ATR_WIN)
        thr = np.clip(np.median(atr_pct) * K, THR_MIN, THR_MAX)
        thr_pct = float(thr)

    # 상태 시드: 마지막 종가 기준으로 초기 극값 설정
    last_close = hist_4h["close"].iloc[-1]
    # 직전 구간의 간단한 방향 추정(직전 5개 평균 기울기)
    slope = (hist_4h["close"].iloc[-5:].diff().mean())
    direction = 1 if slope > 0 else (-1 if slope < 0 else 0)
    ext_price = last_close
    ext_time  = hist_4h.index[-1]

def on_4h_close(bar):  # bar dict: {t,o,h,l,c} strings/numbers
    global hist_4h, direction, ext_price, ext_time, thr_pct
    ts = pd.to_datetime(bar["t"], unit="ms", utc=True)
    row = pd.Series({"open":float(bar["o"]), "high":float(bar["h"]),
                     "low":float(bar["l"]), "close":float(bar["c"])}, name=ts)
    hist_4h.loc[ts] = row

    # 롤링 윈도우 유지
    if len(hist_4h) > MAX_WIN:
        hist_4h = hist_4h.iloc[-MAX_WIN:]

    # 임계값 갱신
    if len(hist_4h) >= ATR_WIN + ATR_LEN:
        atr = ta_atr(hist_4h, ATR_LEN)
        atr_pct = (atr / hist_4h["close"]).tail(ATR_WIN)
        thr_pct = float(np.clip(np.median(atr_pct)*K, THR_MIN, THR_MAX))
    else:
        return  # 아직 데이터 부족

    close = row["close"]
    signal = None
    if ext_price is None:
        ext_price, ext_time, direction = close, ts, 0
        return

    if direction >= 0:
        if close > ext_price:
            ext_price, ext_time = close, ts
        retrace = (ext_price - close) / ext_price
        if retrace >= thr_pct:
            signal = ("SHORT", ts, ext_time, ext_price)
            direction = -1
            ext_price, ext_time = close, ts
    else:
        if close < ext_price:
            ext_price, ext_time = close, ts
        retrace = (close - ext_price) / ext_price
        if retrace >= thr_pct:
            signal = ("LONG", ts, ext_time, ext_price)
            direction = 1
            ext_price, ext_time = close, ts

    if signal:
        side, sig_ts, piv_ts, piv_px = signal
        notify(f"[4H ZigZag] {side} | signal={sig_ts} | pivot@{piv_px:.2f} | thr≈{thr_pct*100:.2f}%")

def main():
    client = Client()  # API 키 없이도 퍼블릭 klines 조회 가능(제한적)
    backfill_4h(client)  # ✅ 시작 시 최근 60개 4H 백필
    twm = ThreadedWebsocketManager()
    twm.start()
    notify('🤩')

    def handle_4h(msg):
        if msg.get("e") != "kline": return
        k = msg["k"]
        if not k["x"]:  # 미마감 봉은 무시
            return
        bar = {"t": k["T"], "o": k["o"], "h": k["h"], "l": k["l"], "c": k["c"]}
        on_4h_close(bar)

    twm.start_kline_socket(callback=handle_4h, symbol=SYMBOL.lower(), interval="4h")
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
