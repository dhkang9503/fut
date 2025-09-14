# -*- coding: utf-8 -*-
"""
WebSocket LIVE BB+CCI Scanner + Telegram Healthcheck (/ping, /status)
- Exchange: Binance Futures WebSocket (fstream.binance.com)
- Signals: [EARLY] (진행중 봉), [CONFIRM] (봉 마감), [EXIT-TP]/[EXIT-SL]
- Healthcheck: /ping, /status (롱-폴링 getUpdates)
- Asia/Seoul timestamps

환경변수:
  OKX_TELEGRAM_TOKEN   = 텔레그램 봇 토큰
  OKX_TELEGRAM_CHAT_ID = 채팅 또는 그룹 ID
"""

import os, io, json, time, threading
import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from websocket import WebSocketApp

# ===== 텔레그램 =====
TELEGRAM_TOKEN   = os.environ.get("OKX_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("OKX_TELEGRAM_CHAT_ID")

def tg(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG]", msg); return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        print("TG error:", e)

def tg_photo(png_bytes, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG photo]", caption); return
    try:
        files = {"photo": ("chart.png", png_bytes, "image/png")}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                      data=data, files=files, timeout=15)
    except Exception as e:
        print("TG photo error:", e)

# ====== 상태/헬스체크 ======
START_TS = time.time()
LAST_LOOP_TS = time.time()   # 웹소켓 이벤트 수신 시 갱신

def tg_reply(chat_id: str, text: str):
    if not TELEGRAM_TOKEN:
        print("[TG disabled]", text); return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10
        )
    except Exception as e:
        print("TG reply error:", e)

def format_uptime(sec: float):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    parts = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)

def build_status_text(extra_chat_id=None):
    now = time.time()
    uptime = format_uptime(now - START_TS)
    last_loop_age = now - LAST_LOOP_TS
    health = "OK ✅" if last_loop_age < 120 else "STALE ⚠️ (loop idle >120s)"
    lines = [
        "🤖 Bot Status",
        f"- health: {health}",
        f"- uptime: {uptime}",
        f"- last tick: {int(last_loop_age)}s ago",
        f"- symbols: {', '.join(SYMBOLS)}",
        f"- timeframe: {INTERVAL}",
    ]
    if extra_chat_id and str(extra_chat_id) != str(TELEGRAM_CHAT_ID or ""):
        lines.append(f"- reply chat: {extra_chat_id}")
    return "\n".join(lines)

def tg_listen_loop():
    """텔레그램 명령 리스너 (/ping, /status, /help)"""
    if not TELEGRAM_TOKEN:
        print("[TG listen disabled: no token]")
        return
    offset = None
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    while True:
        try:
            resp = requests.get(url, params={"timeout": 50, "offset": offset}, timeout=60)
            j = resp.json()
            if not j.get("ok"):
                time.sleep(2); continue
            for upd in j.get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("edited_message")
                if not msg: continue
                chat_id = msg["chat"]["id"]
                text = (msg.get("text") or "").strip()
                if not text: continue

                low = text.lower()
                if low.startswith("/ping"):
                    age = int(time.time() - LAST_LOOP_TS)
                    tg_reply(chat_id, f"pong 🏓 (last tick {age}s ago)")
                elif low.startswith("/status"):
                    tg_reply(chat_id, build_status_text(extra_chat_id=chat_id))
                elif low.startswith("/help") or low.startswith("/start"):
                    tg_reply(chat_id,
                        "Commands:\n"
                        "/ping   - check heartbeat\n"
                        "/status - show health/uptime/last tick\n"
                        "/help   - this help"
                    )
        except Exception as e:
            print("tg_listen_loop error:", e)
            time.sleep(3)

def start_tg_listener_thread():
    t = threading.Thread(target=tg_listen_loop, daemon=True)
    t.start()

# ===== 설정 =====
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1m"  # 1m 기준 (웹소켓 스트림도 1m)
STREAMS = [f"{s.lower()}@kline_{INTERVAL}" for s in SYMBOLS]
WS_URL  = "wss://fstream.binance.com/stream?streams=" + "/".join(STREAMS)

BB_N, BB_K = 20, 2.0
CCI_N = 20
CCI_UP, CCI_DN = 100, -100
TOUCH_LOOKBACK = 3
KST = ZoneInfo("Asia/Seoul")

SEND_CHARTS_ON_CONFIRM = True  # 확정/청산 시 차트 전송

# ===== 데이터 버퍼 =====
MAX_KEEP = 300
buf = {s: pd.DataFrame(columns=["open","high","low","close","volume","close_time",
                                "bb_mid","bb_up","bb_dn","cci"])
            for s in SYMBOLS}

early_sent = {}   # (symbol, bar_open_ms, side) -> True
pos_state  = {"open_symbol": None, "side": None, "entry_price": None, "entry_time_utc": None}

# ===== 지표 =====
def compute_bb_cci(df):
    d = df.copy()
    close = d["close"]
    ma = close.rolling(BB_N).mean()
    std = close.rolling(BB_N).std(ddof=0)
    d["bb_mid"] = ma
    d["bb_up"]  = ma + BB_K * std
    d["bb_dn"]  = ma - BB_K * std
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    sma_tp = tp.rolling(CCI_N).mean()
    mad = (tp - sma_tp).abs().rolling(CCI_N).mean()
    d["cci"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    d["cci"] = d["cci"].fillna(0)
    return d

def cross_up(prev, cur, level):   return (prev <= level) and (cur > level)
def cross_down(prev, cur, level): return (prev >= level) and (cur < level)

def recent_touch(series_close, series_band, lookback, mode):
    sub_c = series_close.tail(lookback)
    sub_b = series_band.tail(lookback)
    if len(sub_c) == 0 or len(sub_b) == 0: return False
    return bool((sub_c <= sub_b).any() if mode=="below" else (sub_c >= sub_b).any())

# ===== 시그널 (실시간: 최신 진행중 봉 포함) =====
def entry_signal_live(d):
    if len(d) < max(BB_N, CCI_N)+2: return None
    last, prev = d.iloc[-1], d.iloc[-2]
    # LONG
    long_setup = recent_touch(d["close"].iloc[:-1], d["bb_dn"].iloc[:-1], TOUCH_LOOKBACK, "below")
    long_cross = cross_up(prev["cci"], last["cci"], CCI_DN)
    long_band  = (prev["close"] <= prev["bb_dn"]) and (last["close"] > last["bb_dn"])
    if long_setup and long_cross and long_band:
        return {"side":"long", "reason":"EARLY: BB lower reclaim + CCI↑-100"}
    # SHORT
    short_setup = recent_touch(d["close"].iloc[:-1], d["bb_up"].iloc[:-1], TOUCH_LOOKBACK, "above")
    short_cross = cross_down(prev["cci"], last["cci"], CCI_UP)
    short_band  = (prev["close"] >= prev["bb_up"]) and (last["close"] < last["bb_up"])
    if short_setup and short_cross and short_band:
        return {"side":"short", "reason":"EARLY: BB upper reject + CCI↓+100"}
    return None

def confirm_signal_close(d):
    # 봉 마감 시점에도 동일 조건 확인
    return entry_signal_live(d)

def exit_signal_bb_cci(d, side):
    if len(d) < max(BB_N, CCI_N)+2: return None
    last, prev = d.iloc[-1], d.iloc[-2]
    if side == "long":
        if (last["close"] >= last["bb_mid"] and last["cci"] > 0):
            return {"type":"tp", "reason":"BB mid reached & CCI>0"}
        if (last["close"] < last["bb_dn"]) or cross_down(prev["cci"], last["cci"], CCI_DN):
            return {"type":"sl", "reason":"BB lower loss or CCI<-100"}
    else:
        if (last["close"] <= last["bb_mid"] and last["cci"] < 0):
            return {"type":"tp", "reason":"BB mid reached & CCI<0"}
        if (last["close"] > last["bb_up"]) or cross_up(prev["cci"], last["cci"], CCI_UP):
            return {"type":"sl", "reason":"BB upper break or CCI>+100"}
    return None

# ===== 차트 (확정/청산 시만) =====
def make_chart_png(d, symbol, entry_time_utc=None, exit_time_utc=None, entry_px=None, exit_px=None):
    dd = d.copy()
    # dd에는 index가 open_time이 아닌 일반 행 인덱스이므로 시간 축 생성
    # 여기서는 close_time을 사용
    kst = KST
    times = pd.to_datetime(dd["close_time"], utc=True).tz_convert(kst)
    x = times.map(mdates.date2num)
    colors = ["#4DD2E6" if c>=o else "#FC495C" for o,c in zip(dd["open"], dd["close"])]
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6), sharex=True)
    for ax in (ax1, ax2):
        ax.set_facecolor("black"); ax.tick_params(colors="white"); ax.grid(True, alpha=0.2, color="white")
    fig.patch.set_facecolor("black")
    w = 0.8 / 1440.0
    for xi, o, h, l, c, col in zip(x, dd["open"], dd["high"], dd["low"], dd["close"], colors):
        ax1.plot([xi, xi], [l, h], color=col, linewidth=1)
        body_low, body_high = min(o, c), max(o, c)
        ax1.add_patch(plt.Rectangle((xi, body_low), w, body_high - body_low, color=col, alpha=0.9, linewidth=0))
    ax1.plot(x, dd["bb_mid"], linewidth=1.0, alpha=0.9, color="white")
    ax1.plot(x, dd["bb_up"],  linewidth=0.9, alpha=0.9, color="gray")
    ax1.plot(x, dd["bb_dn"],  linewidth=0.9, alpha=0.9, color="gray")
    if entry_time_utc and entry_px:
        ax1.scatter(mdates.date2num(entry_time_utc.astimezone(kst)), entry_px, s=80, marker="^",
                    color="#4DD2E6", edgecolors="white", linewidths=0.5, zorder=5, label="Entry")
    if exit_time_utc and exit_px:
        ax1.scatter(mdates.date2num(exit_time_utc.astimezone(kst)), exit_px, s=80, marker="v",
                    color="#FC495C", edgecolors="white", linewidths=0.5, zorder=5, label="Exit")
    ax1.legend(facecolor="black", edgecolor="white", labelcolor="white")
    ax2.plot(x, dd["cci"], linewidth=1.0)
    ax2.axhline(CCI_UP, color="white", linewidth=0.7, alpha=0.6)
    ax2.axhline(0,      color="white", linewidth=0.6, alpha=0.3, linestyle="--")
    ax2.axhline(CCI_DN, color="white", linewidth=0.7, alpha=0.6)
    ax1.set_title(f"{symbol} {INTERVAL}  BB({BB_N},{BB_K}) / CCI({CCI_N})", color="white")
    fmt = mdates.DateFormatter('%H:%M', tz=KST)
    ax2.xaxis.set_major_formatter(fmt)
    plt.tight_layout()
    buf_img = io.BytesIO()
    plt.savefig(buf_img, format="png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig); buf_img.seek(0)
    return buf_img

def now_kst():
    return datetime.now(timezone.utc).astimezone(KST).strftime("%Y-%m-%d %H:%M:%S %Z")

# ===== WebSocket 콜백 =====
def on_message(ws, message):
    global buf, early_sent, pos_state, LAST_LOOP_TS
    try:
        LAST_LOOP_TS = time.time()   # 하트비트 갱신
        data = json.loads(message)
        if "data" not in data: return
        d = data["data"]
        if d.get("e") != "kline": return
        k = d["k"]
        sym = d["s"]
        ot_ms = k["t"]  # bar open time (ms)
        ct_ms = k["T"]  # bar close time (ms)
        o,h,l,c,v = map(float, (k["o"],k["h"],k["l"],k["c"],k["v"]))
        is_final = bool(k["x"])

        # 버퍼 업데이트(진행중 봉 덮어쓰기)
        df = buf[sym]
        row = {
            "open":o,"high":h,"low":l,"close":c,"volume":v,
            "close_time": pd.to_datetime(ct_ms, unit="ms", utc=True)
        }
        # open_time을 키로 쓰진 않지만, 같은 바를 덮어쓰기 위해 마지막 행 검사
        # 간단하게: 같은 close_time이면 업데이트, 아니면 append
        if len(df) and int(df.iloc[-1]["close_time"].value/1e6) == ct_ms:
            for kf,vf in row.items(): df.iloc[-1, df.columns.get_loc(kf)] = vf
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        # 지표 갱신 & 슬라이딩 창
        df = compute_bb_cci(df).tail(MAX_KEEP)
        buf[sym] = df

        # 보유 중이면 exit 우선 체크
        if pos_state["open_symbol"] == sym:
            ex = exit_signal_bb_cci(df, pos_state["side"])
            if ex:
                last = df.iloc[-1]
                tg(f"[EXIT-{ex['type'].upper()}] {sym} {pos_state['side'].upper()} @ {last['close']:.2f}\n{now_kst()}\nreason: {ex['reason']}")
                if SEND_CHARTS_ON_CONFIRM:
                    png = make_chart_png(df.tail(90), sym,
                                         entry_time_utc=pos_state["entry_time_utc"],
                                         exit_time_utc=last["close_time"].to_pydatetime(),
                                         entry_px=pos_state["entry_price"], exit_px=last["close"])
                    tg_photo(png, caption=f"{sym} {ex['type'].upper()} chart")
                pos_state = {"open_symbol": None, "side": None, "entry_price": None, "entry_time_utc": None}
            return

        # 포지션 없으면 실시간 엔트리 평가(EARLY)
        dsig = entry_signal_live(df)
        if dsig:
            key = (sym, ot_ms, dsig["side"])
            if not early_sent.get(key):
                last = df.iloc[-1]
                tg(f"[EARLY] {sym} {dsig['side'].upper()} @ {last['close']:.2f}\n{now_kst()}\n{dsig['reason']}")
                early_sent[key] = True

        # 봉 마감 시 CONFIRM → 포지션 오픈(알림용)
        if is_final:
            csig = confirm_signal_close(df)
            if csig:
                last = df.iloc[-1]
                pos_state = {"open_symbol": sym, "side": csig["side"],
                             "entry_price": last["close"], "entry_time_utc": last["close_time"].to_pydatetime()}
                tg(f"[CONFIRM] {sym} {csig['side'].upper()} @ {last['close']:.2f}\n(lev x100 alert)\n{now_kst()}\nreason: {csig['reason']}")
                if SEND_CHARTS_ON_CONFIRM:
                    png = make_chart_png(df.tail(90), sym,
                                         entry_time_utc=pos_state["entry_time_utc"],
                                         exit_time_utc=None,
                                         entry_px=pos_state["entry_price"], exit_px=None)
                    tg_photo(png, caption=f"{sym} CONFIRM chart")

            # EARLY 키 청소(메모리/중복 방지)
            for kkey in list(early_sent.keys()):
                _, k_ot, _ = kkey
                if k_ot <= ot_ms - 3*60_000:
                    early_sent.pop(kkey, None)

    except Exception as e:
        print("on_message error:", e)

def on_error(ws, error): print("ws error:", error)
def on_close(ws, code, msg): print("ws closed:", code, msg)
def on_open(ws): tg("BB+CCI 1m LIVE started (WebSocket).")

def run_ws():
    ws = WebSocketApp(WS_URL, on_open=on_open, on_message=on_message,
                      on_error=on_error, on_close=on_close)
    ws.run_forever(ping_interval=15, ping_timeout=7)

# ===== 실행 =====
if __name__ == "__main__":
    # 텔레그램 헬스체크 리스너 시작
    start_tg_listener_thread()
    # 웹소켓 실시간 스캐너 시작(메인 스레드)
    tg("Launching BB+CCI 1m live scanner…")
    run_ws()
