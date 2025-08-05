import os
import requests
import time
from datetime import datetime, timezone
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

# === Telegram ===
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("텔레그램 전송 실패:", e)

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

# === 계좌 및 종목 ===
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
    top_symbols = df.sort_values("vol", ascending=False).head(limit)["instId"].tolist()
    return top_symbols

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

def place_order(symbol, side, size, stop_loss, take_profit):
    try:
        direction = "buy" if side == "long" else "sell"

        order = {
            "instId": symbol,
            "tdMode": "isolated",
            "side": direction,
            "ordType": "market",
            "sz": str(round(size, 3))
        }
        res = send_request("POST", "/api/v5/trade/order", order)

        if res.get("code") != "0":
            reason = res.get("msg", "Unknown error")
            send_telegram(
                f"""❌ 주문 실패 (시장가 진입)
━━━━━━━━━━━━━━━
종목: {symbol}
방향: {side.upper()}
사유: {reason}"""
            )
            return

        algo_order = {
            "instId": symbol,
            "tdMode": "isolated",
            "side": "sell" if side == "long" else "buy",
            "ordType": "oco",
            "sz": str(round(size, 3)),
            "tpTriggerPx": str(round(take_profit, 9)),
            "tpOrdPx": "-1",
            "slTriggerPx": str(round(stop_loss, 9)),
            "slOrdPx": "-1"
        }
        res2 = send_request("POST", "/api/v5/trade/order-algo", algo_order)

        if res2.get("code") != "0":
            reason = res2.get("msg", "Unknown error")
            send_telegram(
                f"""⚠️ OCO 주문 실패
━━━━━━━━━━━━━━━
종목: {symbol}
방향: {side.upper()}
TP: {take_profit:.9f} / SL: {stop_loss:.9f}
사유: {reason}"""
            )
            return

        send_telegram(
            f"""📥 포지션 진입 ({side.upper()})
━━━━━━━━━━━━━━━
종목: {symbol}
진입가: {price:.9f}
수량: {size:,.9f}
익절가 (TP): {take_profit:.9f}
손절가 (SL): {stop_loss:.9f}"""
        )

    except Exception as e:
        send_telegram(
            f"""❗️ 예외 발생 (주문 시)
━━━━━━━━━━━━━━━
종목: {symbol}
방향: {side.upper()}
에러: {str(e)}"""
        )

# === 초기화 ===
send_telegram("✅ OKX 자동매매 봇이 시작되었습니다.")
open_positions = {}
daily_start_balance = get_balance()
current_day = datetime.now().date()
trading_paused = False
daily_loss_limit_percent = 5

# === 메인 루프 ===
while True:
    try:
        now = datetime.now().date()

        # 자정이면 리포트 + 초기화
        if now != current_day:
            daily_end_balance = get_balance()
            profit = daily_end_balance - daily_start_balance
            percent = (profit / daily_start_balance) * 100
            icon = "✅" if profit >= 0 else "❌"
            send_telegram(
                f"""{icon} 하루 거래 요약 리포트
━━━━━━━━━━━━━━━
🗓 날짜: {current_day}
시작 잔고: {daily_start_balance:.9f} USDT
종료 잔고: {daily_end_balance:.9f} USDT
수익금: {profit:.9f} USDT
수익률: {percent:.9f}%"""
            )
            current_day = now
            trading_paused = False
            daily_start_balance = daily_end_balance
            send_telegram("🔄 새로운 하루가 시작되어 거래가 재개됩니다.")

        if trading_paused:
            time.sleep(60)
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
                percent = (pnl / entry_price) * 100
                status = "익절" if profit > 0 else "손절"
                current_balance = get_balance()
                send_telegram(
                    f"""📤 포지션 종료 ({status})
━━━━━━━━━━━━━━━
종목: {sym}
진입가: {entry_price:.9f}
종료가: {last_price:.9f}
수익금: {profit:.9f} USDT
수익률: {percent:.9f}%
잔고: {current_balance:.9f} USDT"""
                )
                del open_positions[sym]

                # 손실 한도 확인
                daily_loss = ((daily_start_balance - current_balance) / daily_start_balance) * 100
                if daily_loss >= daily_loss_limit_percent:
                    if not trading_paused:
                        trading_paused = True
                        send_telegram(
                            f"""⛔️ 당일 손실 한도 초과로 거래 정지
━━━━━━━━━━━━━━━
손실률: {daily_loss:.2f}%
기준 한도: {daily_loss_limit_percent:.2f}%
오늘은 더 이상 거래하지 않습니다."""
                        )

        for symbol in top_symbols:
            if has_open_position(symbol) or symbol in open_positions:
                continue

            if trading_paused:
                break

            signal, price, atr = generate_signal(symbol)
            if signal:
                capital = get_balance()
                stop_loss_distance = 1.5 * atr
                position_size = (capital * RISK_PER_TRADE) / stop_loss_distance
                take_profit_price = price + 2.5 * atr if signal == "long" else price - 2.5 * atr
                stop_loss_price = price - 1.5 * atr if signal == "long" else price + 1.5 * atr
                place_order(symbol, signal, position_size, stop_loss_price, take_profit_price)
                open_positions[symbol] = {
                    "entry_price": price,
                    "direction": signal,
                    "size": position_size
                }

        time.sleep(300)

    except Exception as e:
        print(f"[ERROR] {e}")
        send_telegram(f"[오류 발생] {e}")
        time.sleep(60)
