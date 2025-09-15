# -*- coding: utf-8 -*-
"""
WebSocket LIVE BB+CCI Scanner (1h candles) + Telegram Healthcheck + Exit De-Dup
- Exchange: Binance Futures WebSocket (fstream.binance.com)
- Symbols: BTCUSDT, ETHUSDT, SOLUSDT
- Signals: [EARLY], [CONFIRM], [EXIT-TP]/[EXIT-SL]/[EXIT-FORCE]
- Healthcheck: /ping, /status
- Event force-exit: CPI/PPI 자동 반복, FOMC 수동 지정
"""

import os, io, json, time, threading, calendar
from collections import deque
import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
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
LAST_LOOP_TS = time.time()
LAST_LOOP_AT = None

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
    age = now - LAST_LOOP_TS
    if age < 1.5:
        last_str = "just now (≤1s)"
    elif age < 60:
        last_str = f"{int(age)}s ago"
    else:
        mm, ss = divmod(int(age), 60)
        last_str = f"{mm}m {ss}s ago"
    uptime = format_uptime(now - START_TS)
    health = "OK ✅" if age < 600 else "STALE ⚠️ (loop idle >10m)"
    lines = [
        "🤖 Bot Status",
        f"- health: {health}",
        f"- uptime: {uptime}",
        f"- last tick: {last_str}",
        f"- last tick at: {LAST_LOOP_AT or 'N/A'}",
        f"- symbols: {', '.join(SYMBOLS)}",
        f"- timeframe: {INTERVAL}",
    ]
    if extra_chat_id and str(extra_chat_id) != str(TELEGRAM_CHAT_ID or ""):
        lines.append(f"- reply chat: {extra_chat_id}")
    return "\n".join(lines)

def tg_listen_loop():
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
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL = "1h"  # 1시간봉
STREAMS = [f"{s.lower()}@kline_{INTERVAL}" for s in SYMBOLS]
WS_URL  = "wss://fstream.binance.com/stream?streams=" + "/".join(STREAMS)

BB_N, BB_K = 20, 2.0
CCI_N = 20
CCI_UP, CCI_DN = 100, -100
TOUCH_LOOKBACK = 3
KST = ZoneInfo("Asia/Seoul")

SEND_CHARTS_ON_CONFIRM = True

# ===== 데이터 버퍼 =====
MAX_KEEP = 500
buf = {s: pd.DataFrame(columns=["open","high","low","close","volume","close_time",
                                "bb_mid","bb_up","bb_dn","cci"])
            for s in SYMBOLS}

# ===== 중복/스팸 방지 =====
EXIT_COOLDOWN_S = 60
last_exit_at = {}
recent_msgs = deque(maxlen=200)

def tg_dedup_send(text: str, key: str, ttl_s: int = 300):
    now = time.time()
    for k, ts in list(recent_msgs):
        if now - ts > ttl_s:
            try: recent_msgs.remove((k, ts))
            except: pass
    if any(k == key for k, _ in recent_msgs):
        return
    recent_msgs.append((key, now))
    tg(text)

# ===== 포지션 상태 =====
pos_state  = {
    "open_symbol": None,
    "side": None,
    "entry_price": None,
    "entry_time_utc": None,
    "trade_id": None,
    "exit_sent": False,
}

# ===== 이벤트 계산 =====
EVENT_EXIT_BEFORE_MIN = 5

def second_weekday(year, month, weekday):
    """해당 월의 두번째 weekday 날짜 (weekday: 0=월 .. 6=일)"""
    c = calendar.Calendar(firstweekday=0)
    days = [d for d in c.itermonthdates(year, month) if d.month == month and d.weekday() == weekday]
    return days[1]

def generate_monthly_events(year=None, month=None):
    """해당 월의 CPI/PPI UTC 시각 반환"""
    if year is None or month is None:
        now = datetime.now(timezone.utc)
        year, month = now.year, now.month
    events = []
    cpi_day = second_weekday(year, month, 1)  # 화요일
    ppi_day = second_weekday(year, month, 2)  # 수요일
    events.append(("CPI", datetime(year, month, cpi_day.day, 13, 30, tzinfo=timezone.utc)))
    events.append(("PPI", datetime(year, month, ppi_day.day, 13, 30, tzinfo=timezone.utc)))
    return events

# FOMC 일정은 수동으로 직접 넣어야 함
FOMC_EVENTS = [
    ("FOMC", datetime(2025, 9, 19, 18, 0, tzinfo=timezone.utc)),  # 예시
]

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

# ===== 시그널 =====
def entry_signal_live(d):
    if len(d) < max(BB_N, CCI_N)+2: return None
    last, prev = d.iloc[-1], d.iloc[-2]
    long_setup = recent_touch(d["close"].iloc[:-1], d["bb_dn"].iloc[:-1], TOUCH_LOOKBACK, "below")
    long_cross = cross_up(prev["cci"], last["cci"], CCI_DN)
    long_band  = (prev["close"] <= prev["bb_dn"]) and (last["close"] > last["bb_dn"])
    if long_setup and long_cross and long_band:
        return {"side":"long", "reason":"EARLY: BB lower reclaim + CCI↑-100"}
    short_setup = recent_touch(d["close"].iloc[:-1], d["bb_up"].iloc[:-1], TOUCH_LOOKBACK, "above")
    short_cross = cross_down(prev["cci"], last["cci"], CCI_UP)
    short_band  = (prev["close"] >= prev["bb_up"]) and (last["close"] < last["bb_up"])
    if short_setup and short_cross and short_band:
        return {"side":"short", "reason":"EARLY: BB upper reject + CCI↓+100"}
    return None

def confirm_signal_close(d):
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

# ===== 차트 =====
def make_chart_png(d, symbol, entry_time_utc=None, exit_time_utc=None, entry_px=None, exit_px=None):
    dd = d.copy()
    times = pd.to_datetime(dd["close_time"], utc=True).tz_convert(KST)
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
        ax1.scatter(mdates.date2num(entry_time_utc.astimezone(KST)), entry_px, s=80, marker="^",
                    color="#4DD2E6", edgecolors="white", linewidths=0.5, zorder=5, label="Entry")
    if exit_time_utc and exit_px:
        ax1.scatter(mdates.date2num(exit_time_utc.astimezone(KST)), exit_px, s=80, marker="v",
                    color="#FC495C", edgecolors="white", linewidths=0.5, zorder=5, label="Exit")
    ax1.legend(facecolor="black", edgecolor="white", labelcolor="white")
    ax2.plot(x, dd["cci"], linewidth=1.0)
    ax2.axhline(CCI_UP, color="white", linewidth=0.7, alpha=0.6)
    ax2.axhline(0,      color="white", linewidth=0.6, alpha=0.3, linestyle="--")
    ax2.axhline(CCI_DN, color="white", linewidth=0.7, alpha=0.6)
    ax1.set_title(f"{symbol} {INTERVAL}  BB({BB_N},{BB_K}) / CCI({CCI_N})", color="white")
    fmt = mdates.DateFormatter('%m-%d %H:%M', tz=KST)
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
    global buf, pos_state, LAST_LOOP_TS, LAST_LOOP_AT
    try:
        LAST_LOOP_TS = time.time()
        LAST_LOOP_AT = now_kst()
        data = json.loads(message)
        if "data" not in data: return
        d = data["data"]
        if d.get("e") != "kline": return
        k = d["k"]
        sym = d["s"]
        ct_ms = k["T"]
        o,h,l,c,v = map(float, (k["o"],k["h"],k["l"],k["c"],k["v"]))
        is_final = bool(k["x"])

        # 버퍼 업데이트
        df = buf[sym]
        row = {"open":o,"high":h,"low":l,"close":c,"volume":v,
               "close_time": pd.to_datetime(ct_ms, unit="ms", utc=True)}
        if len(df) and int(df.iloc[-1]["close_time"].value/1e6) == ct_ms:
            for kf,vf in row.items(): df.iloc[-1, df.columns.get_loc(kf)] = vf
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = compute_bb_cci(df).tail(MAX_KEEP)
        buf[sym] = df

        # ===== 이벤트 강제 청산 =====
        if pos_state["open_symbol"]:
            now_utc = datetime.now(timezone.utc)
            evs = generate_monthly_events(now_utc.year, now_utc.month) + \
                  generate_monthly_events((now_utc + timedelta(days=32)).year,
                                          (now_utc + timedelta(days=32)).month) + \
                  FOMC_EVENTS
            for ev_name, ev_time in evs:
                if now_utc >= ev_time - timedelta(minutes=EVENT_EXIT_BEFORE_MIN) and now_utc < ev_time:
                    last = buf[pos_state["open_symbol"]].iloc[-1]
                    tg(f"[EXIT-FORCE] {pos_state['open_symbol']} {pos_state['side'].upper()} @ {last['close']:.2f}\n"
                       f"{now_kst()}\nreason: Pre-{ev_name} event risk-off")
                    pos_state.update({"open_symbol": None, "side": None, "entry_price": None,
                                      "entry_time_utc": None, "trade_id": None, "exit_sent": False})
                    return

        # ===== EXIT =====
        if pos_state["open_symbol"] == sym:
            ex = exit_signal_bb_cci(df, pos_state["side"])
            if ex and not pos_state.get("exit_sent", False):
                now_ts = time.time()
                if now_ts - last_exit_at.get(sym, 0) >= EXIT_COOLDOWN_S:
                    last = df.iloc[-1]
                    msg_key = f"EXIT|{sym}|{pos_state['side']}|{ex['type']}|{round(last['close'],2)}"
                    msg_txt = (f"[EXIT-{ex['type'].upper()}] {sym} {pos_state['side'].upper()} @ {last['close']:.2f}\n"
                               f"{now_kst()}\nreason: {ex['reason']}")
                    tg_dedup_send(msg_txt, key=msg_key, ttl_s=600)
                    if SEND_CHARTS_ON_CONFIRM:
                        png = make_chart_png(df.tail(200), sym,
                                             entry_time_utc=pos_state["entry_time_utc"],
                                             exit_time_utc=last["close_time"].to_pydatetime(),
                                             entry_px=pos_state["entry_price"], exit_px=last["close"])
                        tg_photo(png, caption=f"{sym} {ex['type'].upper()} chart")
                    pos_state = {"open_symbol": None, "side": None, "entry_price": None,
                                 "entry_time_utc": None, "trade_id": None, "exit_sent": False}
                    last_exit_at[sym] = now_ts
            return

        # ===== EARLY =====
        dsig = entry_signal_live(df)
        if dsig:
            key = f"EARLY|{sym}|{dsig['side']}|{round(df.iloc[-1]['close'],2)}"
            msg = (f"[EARLY] {sym} {dsig['side'].upper()} @ {df.iloc[-1]['close']:.2f}\n"
                   f"{now_kst()}\n{dsig['reason']}")
            tg_dedup_send(msg, key=key, ttl_s=1800)

        # ===== CONFIRM =====
        if is_final:
            csig = confirm_signal_close(df)
            if csig:
                last = df.iloc[-1]
                msg_key = f"CONFIRM|{sym}|{csig['side']}|{round(last['close'],2)}"
                msg_txt = (f"[CONFIRM] {sym} {csig['side'].upper()} @ {last['close']:.2f}\n"
                           f"(lev x100 alert)\n{now_kst()}\nreason: {csig['reason']}")
                tg_dedup_send(msg_txt, key=msg_key, ttl_s=1800)
                pos_state.update({
                    "open_symbol": sym,
                    "side": csig["side"],
                    "entry_price": last["close"],
                    "entry_time_utc": last["close_time"].to_pydatetime(),
                    "trade_id": f"{sym}-{int(last['close_time'].timestamp())}",
                    "exit_sent": False,
                })
                if SEND_CHARTS_ON_CONFIRM:
                    png = make_chart_png(df.tail(200), sym,
                                         entry_time_utc=pos_state["entry_time_utc"],
                                         exit_time_utc=None,
                                         entry_px=pos_state["entry_price"], exit_px=None)
                    tg_photo(png, caption=f"{sym} CONFIRM chart")

    except Exception as e:
        print("on_message error:", e)

def on_error(ws, error): print("ws error:", error)
def on_close(ws, code, msg): print("ws closed:", code, msg)
def on_open(ws): tg("BB+CCI 1h LIVE started (WebSocket).")

def run_ws():
    ws = WebSocketApp(WS_URL, on_open=on_open, on_message=on_message,
                      on_error=on_error, on_close=on_close)
    ws.run_forever(ping_interval=30, ping_timeout=10)

if __name__ == "__main__":
    start_tg_listener_thread()
    tg("Launching BB+CCI 1h live scanner with monthly CPI/PPI risk-off…")
    run_ws()
