# ✅ OKX 자동매매 봇 전체 코드 (시장가 진입 + 체결가 기준 TP/SL 설정)

import os
import requests
import time
from datetime import datetime, timezone, timedelta
import hmac
import hashlib
import base64
import json
import pandas as pd
import numpy as np

# === 환경변수 ===
API_KEY = os.getenv("OKX_API_KEY")
API_SECRET = os.getenv("OKX_API_SECRET")
API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE")
BASE_URL = "https://www.okx.com"
TELEGRAM_TOKEN = os.getenv("OKX_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("OKX_TELEGRAM_CHAT_ID")
LEVERAGE = 3
RISK_PER_TRADE = 0.01
TARGET_COINS = 3
SLIPPAGE = 0.002  # 지정가 주문 시 슬리피지 비율

# === 상태 저장 ===
open_positions = {}
min_sizes = {}
daily_start_balance = None
daily_loss_limit_triggered = False
loss_limit = 0.05
report_sent = False

# === Telegram ===
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("텔레그램 전송 실패:", e)

# === 포맷 함수 ===
def format_price(val):
    if val >= 100:
        return f"{val:,.2f}"
    elif val >= 1:
        return f"{val:,.4f}"
    elif val >= 0.01:
        return f"{val:,.6f}"
    elif val >= 0.0001:
        return f"{val:,.8f}"
    else:
        return f"{val:,.10f}"

# === OKX API ===
def get_timestamp():
    return datetime.now(timezone.utc).isoformat("T", "milliseconds").replace("+00:00", "Z")

def sign_request(method, path, body):
    timestamp = get_timestamp()
    body_str = json.dumps(body) if body else ""
    message = f"{timestamp}{method}{path}{body_str}"
    mac = hmac.new(bytes(API_SECRET, 'utf-8'), msg=message.encode('utf-8'), digestmod='sha256')
    sign = base64.b64encode(mac.digest()).decode()
    headers = {
        'OK-ACCESS-KEY': API_KEY,
        'OK-ACCESS-SIGN': sign,
        'OK-ACCESS-TIMESTAMP': timestamp,
        'OK-ACCESS-PASSPHRASE': API_PASSPHRASE,
        'Content-Type': 'application/json'
    }
    return headers

def send_request(method, path, body=None):
    url = BASE_URL + path
    headers = sign_request(method, path, body)
    if method == "GET":
        res = requests.get(url, headers=headers, params=body)
    else:
        res = requests.post(url, headers=headers, data=json.dumps(body))
    return res.json()

# === 최소 수량 로드 ===
def load_min_sizes():
    url = f"{BASE_URL}/api/v5/public/instruments?instType=SWAP"
    res = requests.get(url).json()
    mapping = {}
    for item in res.get("data", []):
        symbol = item["instId"]
        try:
            mapping[symbol] = float(item["minSz"])
        except:
            continue
    return mapping

# === 계좌/포지션 ===
def get_balance():
    res = send_request("GET", "/api/v5/account/balance", {})
    for asset in res.get("data", [])[0].get("details", []):
        if asset["ccy"] == "USDT":
            return float(asset["cashBal"])
    return 0

def get_top_symbols(limit=TARGET_COINS):
    url = f"{BASE_URL}/api/v5/market/tickers?instType=SWAP"
    res = requests.get(url).json()
    df = pd.DataFrame(res["data"])
    df = df[~df["instId"].str.contains("BTC|ETH")]
    df["vol"] = df["volCcy24h"].astype(float)
    return df.sort_values("vol", ascending=False).head(limit)["instId"].tolist()

def has_open_position(symbol):
    res = send_request("GET", "/api/v5/account/positions", {"instType": "SWAP"})
    for pos in res.get("data", []):
        if pos["instId"] == symbol and float(pos["pos"] or 0) != 0:
            return True
    return False

def get_position_price(symbol):
    res = send_request("GET", "/api/v5/account/positions", {"instType": "SWAP"})
    for pos in res.get("data", []):
        if pos["instId"] == symbol and float(pos["pos"] or 0) != 0:
            return float(pos["avgPx"])
    return None

# === 차트 및 지표 ===
def get_candles(symbol, bar, limit=100):
    url = f"{BASE_URL}/api/v5/market/candles?instId={symbol}&bar={bar}&limit={limit}"
    res = requests.get(url)
    df = pd.DataFrame(res.json()["data"])
    df.columns = ["ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"]
    df = df.iloc[::-1]
    df[["o", "h", "l", "c"]] = df[["o", "h", "l", "c"]].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms")
    return df

def calculate_ema(df, period):
    return df['c'].ewm(span=period, adjust=False).mean()

def calculate_rsi(df, period=14):
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high_low = df['h'] - df['l']
    high_close = np.abs(df['h'] - df['c'].shift())
    low_close = np.abs(df['l'] - df['c'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def generate_signal(symbol):
    df_4h = get_candles(symbol, "4H")
    df_15m = get_candles(symbol, "15m")
    df_5m = get_candles(symbol, "5m")

    ema200_4h = calculate_ema(df_4h, 200).iloc[-1]
    last_close_4h = df_4h['c'].iloc[-1]
    trend_up = last_close_4h > ema200_4h

    ema50_15m = calculate_ema(df_15m, 50).iloc[-1]
    ema200_15m = calculate_ema(df_15m, 200).iloc[-1]
    trend_15m_up = ema50_15m > ema200_15m

    rsi_5m = calculate_rsi(df_5m).iloc[-1]

    if trend_up and trend_15m_up and rsi_5m < 35:
        return "long", df_5m['c'].iloc[-1], calculate_atr(df_5m).iloc[-1]
    elif not trend_up and not trend_15m_up and rsi_5m > 65:
        return "short", df_5m['c'].iloc[-1], calculate_atr(df_5m).iloc[-1]
    else:
        return None, None, None

# === 진입 및 TP/SL ===
def place_order(symbol, side, size):
    direction = "buy" if side == "long" else "sell"
    order = {
        "instId": symbol,
        "tdMode": "isolated",
        "side": direction,
        "ordType": "market",
        "sz": str(round(size, 3))
    }
    res = send_request("POST", "/api/v5/trade/order", order)
    print(json.dumps(res, indent=4))

    if res.get("code") != "0":
        reason = res.get("msg", "Unknown error")
        send_telegram(f"❌ 주문 실패 (시장가 진입)\n━━━━━━━━━━━━━━━\n종목: {symbol}\n방향: {side.upper()}\n수량: {format_price(size)}\n사유: {reason}")
        return None

    time.sleep(1)
    entry_price = get_position_price(symbol)
    if entry_price is None:
        send_telegram(f"❗️ 진입가 조회 실패: {symbol}")
        return None

    tp = entry_price * (1 + 0.025) if side == "long" else entry_price * (1 - 0.025)
    sl = entry_price * (1 - 0.015) if side == "long" else entry_price * (1 + 0.015)

    algo_order = {
        "instId": symbol,
        "tdMode": "isolated",
        "side": "sell" if side == "long" else "buy",
        "ordType": "oco",
        "sz": str(round(size, 3)),
        "tpTriggerPx": str(round(tp, 9)),
        "tpOrdPx": "-1",
        "slTriggerPx": str(round(sl, 9)),
        "slOrdPx": "-1"
    }
    send_request("POST", "/api/v5/trade/order-algo", algo_order)

    send_telegram(f"📥 포지션 진입 ({side.upper()})\n━━━━━━━━━━━━━━━\n종목: {symbol}\n진입가: {format_price(entry_price)}\n수량: {format_price(size)}\n익절가 (TP): {format_price(tp)}\n손절가 (SL): {format_price(sl)}")

    return entry_price

# === 메인 루프 ===
if __name__ == "__main__":
    send_telegram("✅ 자동매매 봇 시작됨.")
    min_sizes = load_min_sizes()
    daily_start_balance = get_balance()
    last_date = datetime.now().date()

    while True:
        try:
            now = datetime.now()
            if now.date() != last_date:
                # 자정 이후 초기화
                daily_start_balance = get_balance()
                daily_loss_limit_triggered = False
                report_sent = False
                last_date = now.date()

            if daily_loss_limit_triggered:
                time.sleep(60)
                continue

            current_balance = get_balance()
            if (current_balance - daily_start_balance) / daily_start_balance <= -loss_limit:
                daily_loss_limit_triggered = True
                send_telegram("⛔️ 손실 한도 초과로 당일 거래 정지됨.")
                continue

            top_symbols = get_top_symbols()

            for sym in list(open_positions):
                if not has_open_position(sym):
                    entry_price = open_positions[sym]['entry_price']
                    size = open_positions[sym]['size']
                    direction = open_positions[sym]['direction']
                    last_price = get_candles(sym, "1m", 1)['c'].iloc[-1]
                    pnl = (last_price - entry_price) if direction == 'long' else (entry_price - last_price)
                    profit = pnl * size
                    percent = (pnl / entry_price) * 100 * LEVERAGE
                    status = "익절" if profit > 0 else "손절"
                    emoji = "✅" if profit > 0 else "❌"
                    send_telegram(
                        f"{emoji} 포지션 종료 ({direction.upper()})\n━━━━━━━━━━━━━━━\n종목: {sym}\n진입가: {format_price(entry_price)}\n종료가: {format_price(last_price)}\n수익금: {format_price(profit)} USDT ({percent:.2f}%)\n현재 잔고: {format_price(current_balance)} USDT"
                    )
                    del open_positions[sym]

            for symbol in top_symbols:
                if has_open_position(symbol):
                    continue
                signal, price, atr = generate_signal(symbol)
                if not signal:
                    continue
                capital = get_balance()
                stop_loss_distance = 1.5 * atr
                size = (capital * RISK_PER_TRADE) / stop_loss_distance
                if symbol in min_sizes and size < min_sizes[symbol]:
                    send_telegram(f"⚠️ 최소 수량 미달로 스킵됨: {symbol} ({format_price(size)} < {min_sizes[symbol]})")
                    continue
                entry = place_order(symbol, signal, size)
                if entry:
                    open_positions[symbol] = {"entry_price": entry, "direction": signal, "size": size}

            if not report_sent and now.hour == 23 and now.minute >= 55:
                profit = current_balance - daily_start_balance
                percent = (profit / daily_start_balance) * 100
                emoji = "✅" if profit >= 0 else "❌"
                send_telegram(f"{emoji} 오늘의 수익률 요약\n━━━━━━━━━━━━━━━\n수익금: {format_price(profit)} USDT\n수익률: {percent:.2f}%")
                report_sent = True

            time.sleep(300)
        except Exception as e:
            send_telegram(f"[오류 발생]\n{e}")
            time.sleep(60)
