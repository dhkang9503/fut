import os
import time
import json
import math
import hmac
import base64
import hashlib
import traceback
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode

import requests
import pandas as pd
import numpy as np

# =========================
# 환경 변수 / 상수
# =========================
API_KEY = os.getenv("OKX_API_KEY")
API_SECRET = os.getenv("OKX_API_SECRET")
API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE")
BASE_URL = "https://www.okx.com"
TELEGRAM_TOKEN = os.getenv("OKX_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("OKX_TELEGRAM_CHAT_ID")

LEVERAGE = 3
RISK_PER_TRADE = 0.005            # 일반 코인: 계좌 대비 0.5% 리스크
RISK_PER_TRADE_MICRO = 0.002      # 마이크로코인: 계좌 대비 0.2% 리스크

TARGET_COINS = 3
DAILY_LOSS_LIMIT = 0.05           # 일간 손실 한도(5%)
LOOP_SLEEP_SEC = 300              # 루프 슬립(초)
HTTP_TIMEOUT = 10

# 마이크로코인 판정/손절폭 하한(% of price)
MICRO_PRICE_THRESHOLD = 0.01      # 이 가격 미만이면 마이크로로 간주 (USDT 기준)
DEFAULT_MIN_SL_PCT = 0.0025       # 일반: 최소 손절폭 하한 0.25%
MICRO_MIN_SL_PCT = 0.005          # 마이크로: 최소 손절폭 하한 0.5%

# 로컬 기준시 (KST)
KST = timezone(timedelta(hours=9))

# =========================
# 상태
# =========================
open_positions = {}         # { symbol: {entry_price, size, direction} }
daily_start_balance = None
daily_loss_limit_triggered = False
report_sent = False

# =========================
# 유틸/텔레그램
# =========================
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[텔레그램 비활성]\n" + message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload, timeout=HTTP_TIMEOUT)
    except Exception as e:
        print("텔레그램 전송 실패:", e)

def format_price(val: float) -> str:
    if val is None:
        return "N/A"
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

# =========================
# OKX 시간/서명/요청
# =========================
def get_timestamp() -> str:
    """OKX 서버 시각(UTC, 밀리초) ISO8601 문자열"""
    r = requests.get(f"{BASE_URL}/api/v5/public/time", timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    ts_ms = int(r.json()["data"][0]["ts"])
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

def sign_request(method: str, path: str, body: dict | None, params: dict | None):
    timestamp = get_timestamp()

    # requestPath(+query) 규격 반영
    request_path = path
    if method.upper() == "GET" and params:
        qs = urlencode(params, doseq=True)
        if qs:
            request_path = f"{path}?{qs}"

    body_str = json.dumps(body) if (body and method.upper() != "GET") else ""
    message = f"{timestamp}{method.upper()}{request_path}{body_str}"

    mac = hmac.new(API_SECRET.encode("utf-8"), msg=message.encode("utf-8"), digestmod=hashlib.sha256)
    sign = base64.b64encode(mac.digest()).decode()

    return {
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": API_PASSPHRASE,
        "Content-Type": "application/json",
    }

def send_request(method: str, path: str, body: dict | None = None, timeout: int = HTTP_TIMEOUT) -> dict:
    url = BASE_URL + path
    params = body if method.upper() == "GET" else None
    headers = sign_request(method, path, body if method.upper() != "GET" else None, params)
    try:
        if method.upper() == "GET":
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
        else:
            r = requests.post(url, headers=headers, data=json.dumps(body or {}), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print("[HTTP ERROR]", msg)
        return {"code": "error", "msg": msg, "data": []}

# =========================
# 마켓 메타/정밀도
# =========================
def get_instrument_meta(symbol: str):
    res = send_request("GET", "/api/v5/public/instruments", {"instType": "SWAP", "instId": symbol})
    if res.get("code") == "0" and res.get("data"):
        info = res["data"][0]
        try:
            return {
                "lotSz": float(info["lotSz"]),
                "tickSz": float(info["tickSz"]),
                "minSz": float(info["minSz"]),
                "settleCcy": info.get("settleCcy", ""),
                "maxMktSz": float(info.get("maxMktSz") or 0),   # 최대 시장가 수량
                "maxLmtSz": float(info.get("maxLmtSz") or 0),   # 최대 지정가 수량(참고)
            }
        except Exception:
            pass
    return None

def round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    precision = max(-int(math.floor(math.log10(step))), 0)
    return round(math.floor(x / step) * step, precision)

def adjust_size_to_lot(size: float, lot_size: float) -> float:
    if lot_size <= 0:
        return 0.0
    precision = max(-int(math.floor(math.log10(lot_size))), 0)
    adjusted = math.floor(size / lot_size) * lot_size
    return round(adjusted, precision)

# =========================
# 계정/포지션/한도
# =========================
def get_balance() -> float:
    res = send_request("GET", "/api/v5/account/balance", {})
    try:
        for asset in res.get("data", [])[0].get("details", []):
            if asset["ccy"] == "USDT":
                return float(asset["cashBal"])
    except Exception:
        pass
    return 0.0

def has_open_position(symbol: str) -> bool:
    res = send_request("GET", "/api/v5/account/positions", {"instType": "SWAP"})
    for pos in res.get("data", []):
        if pos.get("instId") == symbol and float(pos.get("pos") or 0.0) != 0.0:
            return True
    return False

def get_position_price(symbol: str) -> float | None:
    res = send_request("GET", "/api/v5/account/positions", {"instType": "SWAP"})
    for pos in res.get("data", []):
        if pos.get("instId") == symbol and float(pos.get("pos") or 0.0) != 0.0:
            try:
                return float(pos["avgPx"])
            except Exception:
                return None
    return None

def get_max_tradable_size(symbol: str, side: str, price: float, td_mode: str = "isolated", ccy: str = "USDT") -> float:
    """
    OKX가 산출한 현재 계정 상태 기준 최대 가능 수량.
    side == 'long' -> buy 기준, 'short' -> sell 기준.
    보수적으로 쓰기 위해 0이거나 실패하면 0.0 반환.
    """
    pos_side = "long" if side == "long" else "short"
    params = {"instId": symbol, "tdMode": td_mode, "ccy": ccy, "px": str(price), "posSide": pos_side}
    res = send_request("GET", "/api/v5/account/max-size", params)
    if res.get("code") != "0" or not res.get("data"):
        return 0.0
    info = res["data"][0]
    try:
        val = info.get("maxBuy") if side == "long" else info.get("maxSell")
        if val is None:
            val = info.get("maxSz")
        return float(val or 0.0)
    except Exception:
        return 0.0

# =========================
# 시세/지표
# =========================
def get_candles(symbol: str, bar: str, limit: int = 100) -> pd.DataFrame:
    url = f"{BASE_URL}/api/v5/market/candles?instId={symbol}&bar={bar}&limit={limit}"
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()["data"]
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df.columns = ["ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"]
    df = df.iloc[::-1].reset_index(drop=True)
    for col in ["o", "h", "l", "c"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
    return df

def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["c"].ewm(span=period, adjust=False).mean()

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["c"].diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean().replace(0, np.nan)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill().fillna(50)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["h"] - df["l"]
    high_close = np.abs(df["h"] - df["c"].shift())
    low_close = np.abs(df["l"] - df["c"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.bfill()

def generate_signal(symbol: str):
    df_4h = get_candles(symbol, "4H", 300)
    df_15m = get_candles(symbol, "15m", 300)
    df_5m = get_candles(symbol, "5m", 300)

    if df_4h.empty or df_15m.empty or df_5m.empty:
        return None, None, None

    ema200_4h = calculate_ema(df_4h, 200).iloc[-1]
    last_close_4h = df_4h["c"].iloc[-1]
    trend_up = last_close_4h > ema200_4h

    ema50_15m = calculate_ema(df_15m, 50).iloc[-1]
    ema200_15m = calculate_ema(df_15m, 200).iloc[-1]
    trend_15m_up = ema50_15m > ema200_15m

    rsi_5m = calculate_rsi(df_5m).iloc[-1]
    atr_5m = calculate_atr(df_5m).iloc[-1]

    if trend_up and trend_15m_up and rsi_5m < 35:
        return "long", df_5m["c"].iloc[-1], atr_5m
    elif (not trend_up) and (not trend_15m_up) and rsi_5m > 65:
        return "short", df_5m["c"].iloc[-1], atr_5m
    else:
        return None, None, None

# =========================
# 종목 선택
# =========================
def get_top_symbols(limit: int = TARGET_COINS) -> list[str]:
    url = f"{BASE_URL}/api/v5/market/tickers?instType=SWAP"
    res = requests.get(url, timeout=HTTP_TIMEOUT).json()
    df = pd.DataFrame(res.get("data", []))
    if df.empty:
        return []
    df["vol"] = pd.to_numeric(df.get("volCcy24h", 0), errors="coerce").fillna(0.0)
    if "settleCcy" in df.columns:
        df = df[df["settleCcy"] == "USDT"]
    return df.sort_values("vol", ascending=False).head(limit)["instId"].tolist()

# =========================
# 계정 설정
# =========================
def set_leverage(symbol: str, leverage: int, mode: str = "isolated", pos_side: str = "long"):
    body = {"instId": symbol, "lever": str(leverage), "mgnMode": mode, "posSide": pos_side}
    res = send_request("POST", "/api/v5/account/set-leverage", body)
    if res.get("code") != "0":
        reason = res.get("msg", "Unknown")
        send_telegram(f"⚠️ 레버리지 설정 실패: {symbol} ({pos_side})\n사유: {reason}")
    else:
        print(f"✅ 레버리지 설정 완료: {symbol} [{pos_side}] → {leverage}x")
    return res

# =========================
# 마이크로코인 보정
# =========================
def is_micro_price(price: float) -> bool:
    try:
        return float(price) < MICRO_PRICE_THRESHOLD
    except Exception:
        return False

def pick_risk_per_trade(price: float) -> float:
    return RISK_PER_TRADE_MICRO if is_micro_price(price) else RISK_PER_TRADE

# =========================
def place_order(symbol: str, side: str, atr: float):
    """
    side: 'long' | 'short'
    atr : 최근 ATR(가격 단위)
    """
    meta = get_instrument_meta(symbol)
    if not meta:
        send_telegram(f"❌ 주문 실패: 종목 메타 조회 실패 - {symbol}")
        return None

    lotSz    = float(meta["lotSz"])
    tickSz   = float(meta["tickSz"])
    minSz    = float(meta["minSz"])
    maxMktSz = float(meta.get("maxMktSz") or 0)

    balance = get_balance()
    candles = get_candles(symbol, "1m", 1)
    if candles.empty:
        send_telegram(f"❌ 진입 실패: 캔들 데이터 없음 - {symbol}")
        return None

    price = float(candles["c"].iloc[-1])

    # --- 리스크 기반 수량 ---
    min_sl_pct   = MICRO_MIN_SL_PCT if is_micro_price(price) else DEFAULT_MIN_SL_PCT
    stop_loss_dist = max(1.5 * float(atr), price * min_sl_pct)
    if stop_loss_dist <= 0:
        send_telegram(f"❌ ATR 기반 손절폭 계산 실패 - {symbol}")
        return None

    rpt = pick_risk_per_trade(price)
    raw_size    = (balance * rpt) / stop_loss_dist
    target_size = adjust_size_to_lot(raw_size, lotSz)

    # --- 거래소 산출 최대 가능 수량으로 보수 캡 (여유 10%) ---
    max_tradable = get_max_tradable_size(symbol, side, price, td_mode="isolated", ccy="USDT")
    if max_tradable > 0:
        cap_by_ex = adjust_size_to_lot(max_tradable * 0.90, lotSz)   # 10% 버퍼
        target_size = min(target_size, cap_by_ex)

    # --- 종목 자체의 최대 '시장가' 수량으로 보수 캡 (여유 5%) ---
    if maxMktSz > 0:
        cap_by_mkt = adjust_size_to_lot(maxMktSz * 0.95, lotSz)      # 5% 버퍼
        target_size = min(target_size, cap_by_mkt)

    # 최소/lot 체크
    if target_size < max(minSz, lotSz):
        send_telegram(f"⚠️ 최소 주문 수량 미달: {symbol} ({format_price(target_size)} < {max(minSz, lotSz)})")
        return None

    # 증거금(보수적) 체크(수수료/오차 버퍼 3%)
    est_cost = price * target_size / LEVERAGE
    if est_cost * 1.03 > balance:
        send_telegram(
            "⚠️ 증거금 부족(보수 계산)으로 주문 스킵\n"
            f"━━━━━━━━━━━━━━━\n종목: {symbol}\n필요 증거금(예상+버퍼): {format_price(est_cost*1.03)} > 잔고: {format_price(balance)}"
        )
        return None

    # --- 분할 초기 크기 산정 ---
    # 1) 거래소 시장가 최대치
    # 2) 거래소 가용 수량 cap
    # 3) 타겟 사이즈
    chunk_sz = target_size
    if maxMktSz > 0:
        chunk_sz = min(chunk_sz, adjust_size_to_lot(maxMktSz * 0.95, lotSz))
    if max_tradable > 0:
        chunk_sz = min(chunk_sz, adjust_size_to_lot(max_tradable * 0.90, lotSz))
    chunk_sz = adjust_size_to_lot(chunk_sz, lotSz)

    if chunk_sz < max(minSz, lotSz):
        send_telegram(f"⚠️ 거래소 한도 반영 후 분할 수량이 최소 미만: {symbol} ({format_price(chunk_sz)})")
        return None

    # 레버리지 설정
    set_leverage(symbol, LEVERAGE, mode="isolated", pos_side=side)

    # --- 시장가 분할 체결 (+ 적응형 축소) ---
    filled_any = False
    side_str = "buy" if side == "long" else "sell"
    remaining = target_size
    safety_counter = 0

    while remaining >= max(minSz, lotSz) - 1e-12:
        safety_counter += 1
        if safety_counter > 25:  # 과도 루프 방지
            break

        this_sz = min(remaining, chunk_sz)
        this_sz = adjust_size_to_lot(this_sz, lotSz)
        if this_sz < max(minSz, lotSz):
            break

        order = {
            "instId": symbol,
            "tdMode": "isolated",
            "side": side_str,
            "ordType": "market",
            "posSide": side,
            "sz": str(this_sz),
        }
        print("[ORDER REQUEST]", json.dumps(order, ensure_ascii=False))
        res = send_request("POST", "/api/v5/trade/order", order)
        print("[ORDER RESPONSE]", json.dumps(res, ensure_ascii=False))

        if res.get("code") != "0":
            # 오류별 적응형 처리
            d0    = (res.get("data") or [{}])[0]
            s_code = d0.get("sCode")
            s_msg  = d0.get("sMsg", res.get("msg", "Unknown error"))

            # 51202: 주문당 최대치 초과 -> 분할 크기 추가 축소 후 재시도
            if s_code == "51202" or "exceeds the maximum amount" in s_msg:
                chunk_sz = adjust_size_to_lot(max(lotSz, chunk_sz * 0.6), lotSz)  # 40% 축소
                print(f"[INFO] 51202/최대치 → chunk_sz 축소: {chunk_sz}")
                if chunk_sz < max(minSz, lotSz):
                    send_telegram(
                        "❌ 주문 실패: 시장가 최대치 제한으로 더 이상 분할 불가\n"
                        f"종목:{symbol} / 마지막 분할:{format_price(chunk_sz)}"
                    )
                    break
                time.sleep(0.35)
                continue

            # 증거금 부족류 메시지 -> 남은/분할 크기 모두 축소 후 재시도
            if "insufficient" in s_msg.lower() or "margin" in s_msg.lower():
                # 거래소 최대 가용 수량을 다시 조회해 반영
                max_tradable_now = get_max_tradable_size(symbol, side, price, td_mode="isolated", ccy="USDT")
                if max_tradable_now > 0:
                    # 즉시 가능한 크기의 80%로 재설정
                    cap_now = adjust_size_to_lot(max_tradable_now * 0.80, lotSz)
                    chunk_sz = min(chunk_sz, cap_now)
                    remaining = min(remaining, cap_now)
                    print(f"[INFO] margin 부족 → cap_now={cap_now}, chunk_sz={chunk_sz}, remaining={remaining}")

                # 그래도 크면 추가 축소
                chunk_sz = adjust_size_to_lot(max(lotSz, chunk_sz * 0.7), lotSz)
                remaining = adjust_size_to_lot(max(lotSz, remaining * 0.85), lotSz)
                if chunk_sz < max(minSz, lotSz) or remaining < max(minSz, lotSz):
                    send_telegram(
                        "❌ 주문 실패: 증거금 부족으로 더 이상 안전한 분할 불가\n"
                        f"종목:{symbol} / 마지막 분할:{format_price(chunk_sz)} / 남은:{format_price(remaining)}"
                    )
                    break
                time.sleep(0.35)
                continue

            # 그 외 오류: 중단
            send_telegram(
                "❌ 주문 실패 (시장가 진입)\n"
                f"━━━━━━━━━━━━━━━\n종목: {symbol}\n방향: {side.upper()}\n수량: {format_price(this_sz)}\n사유: {s_msg}"
            )
            break

        filled_any = True
        remaining = max(0.0, remaining - this_sz)
        time.sleep(0.18)  # 너무 빠른 연속 주문 방지

    if not filled_any:
        return None

    # 체결 대기 후 평균가 조회
    time.sleep(1.0)
    entry_price = get_position_price(symbol)
    if entry_price is None:
        send_telegram(f"❗️ 진입가 조회 실패: {symbol}")
        return None

    # --- 포지션 현재 수량 재조회 후 OCO 한 번만 ---
    res_pos = send_request("GET", "/api/v5/account/positions", {"instType": "SWAP"})
    pos_sz = None
    for pos in res_pos.get("data", []):
        if pos.get("instId") == symbol and pos.get("posSide") == side:
            try:
                pos_sz = float(pos.get("pos") or 0)
            except:
                pass
    if pos_sz is None or pos_sz <= 0:
        # 분할 체결 누계 추정
        pos_sz = adjust_size_to_lot(target_size - remaining, lotSz)

    if pos_sz < max(minSz, lotSz):
        send_telegram(f"⚠️ 포지션 수량이 최소 미만으로 OCO 스킵: {symbol} ({format_price(pos_sz)})")
        return {"entry_price": entry_price, "size": pos_sz}

    # OCO TP/SL (tickSz 반영)
    tp = entry_price * (1 + 0.025) if side == "long" else entry_price * (1 - 0.025)
    sl = entry_price * (1 - 0.015) if side == "long" else entry_price * (1 + 0.015)
    tp = round_to_step(tp, tickSz)
    sl = round_to_step(sl, tickSz)

    algo_order = {
        "instId": symbol,
        "tdMode": "isolated",
        "side": "sell" if side == "long" else "buy",
        "posSide": side,
        "ordType": "oco",
        "sz": str(pos_sz),
        "tpTriggerPx": f"{tp}",
        "tpOrdPx": "-1",
        "slTriggerPx": f"{sl}",
        "slOrdPx": "-1",
    }
    _ = send_request("POST", "/api/v5/trade/order-algo", algo_order)

    send_telegram(
        f"📥 포지션 진입 ({side.upper()})\n"
        f"━━━━━━━━━━━━━━━\n종목: {symbol}\n진입가: {format_price(entry_price)}\n"
        f"수량: {format_price(pos_sz)}\n익절(TP): {format_price(tp)}\n손절(SL): {format_price(sl)}"
    )
    return {"entry_price": entry_price, "size": pos_sz}


# =========================
# 메인 루프
# =========================
if __name__ == "__main__":
    try:
        start_balance = get_balance()
        daily_start_balance = start_balance
        last_date = datetime.now(KST).date()
        send_telegram(f"✅ 자동매매 봇 시작됨, 잔고: {format_price(start_balance)} USDT")

        while True:
            try:
                now = datetime.now(KST)

                # 날짜 변경(자정 이후) 초기화
                if now.date() != last_date:
                    daily_start_balance = get_balance()
                    daily_loss_limit_triggered = False
                    report_sent = False
                    last_date = now.date()

                # 손실 한도 확인
                current_balance = get_balance()
                if daily_start_balance > 0:
                    daily_dd = (current_balance - daily_start_balance) / daily_start_balance
                    if daily_dd <= -DAILY_LOSS_LIMIT:
                        if not daily_loss_limit_triggered:
                            daily_loss_limit_triggered = True
                            send_telegram("⛔️ 손실 한도 초과로 당일 거래 정지됨.")
                        time.sleep(60)
                        continue

                # 포지션 종료 감지 및 리포트(근사치)
                for sym in list(open_positions):
                    if not has_open_position(sym):
                        info = open_positions[sym]
                        entry_price = info["entry_price"]
                        size = info["size"]
                        direction = info["direction"]
                        # 종료가 근사: 최신 1m 종가
                        last_price_df = get_candles(sym, "1m", 1)
                        close_px = float(last_price_df["c"].iloc[-1]) if not last_price_df.empty else None

                        pnl = None
                        profit = None
                        pct = None
                        if close_px is not None:
                            pnl = (close_px - entry_price) if direction == "long" else (entry_price - close_px)
                            profit = pnl * size
                            pct = (pnl / entry_price) * 100 * LEVERAGE

                        lines = [
                            f"{'✅' if (profit or 0) > 0 else '❌'} 포지션 종료 ({direction.upper()})",
                            "━━━━━━━━━━━━━━━",
                            f"종목: {sym}",
                            f"진입가: {format_price(entry_price)}",
                            f"종료가(근사): {format_price(close_px) if close_px is not None else 'N/A'}",
                            f"수익금(근사): {format_price(profit) if profit is not None else 'N/A'} USDT"
                            + (f" ({pct:.2f}%)" if pct is not None else ""),
                            f"현재 잔고: {format_price(current_balance)} USDT",
                        ]
                        send_telegram("\n".join(lines))
                        del open_positions[sym]

                # 거래 대상 심볼
                top_symbols = get_top_symbols()
                if not top_symbols:
                    time.sleep(60)
                    continue

                # 시그널 탐색/진입
                for symbol in top_symbols:
                    if has_open_position(symbol):
                        continue
                    signal, price, atr = generate_signal(symbol)
                    if not signal:
                        continue
                    entry = place_order(symbol, signal, atr)
                    if entry:
                        open_positions[symbol] = {
                            "entry_price": entry["entry_price"],
                            "direction": signal,
                            "size": entry["size"],
                        }

                # 일일 요약(23:55~)
                if not report_sent and now.hour == 23 and now.minute >= 55:
                    current_balance = get_balance()
                    profit = current_balance - daily_start_balance
                    percent = (profit / daily_start_balance) * 100 if daily_start_balance > 0 else 0.0
                    emoji = "✅" if profit >= 0 else "❌"
                    send_telegram(
                        f"{emoji} 오늘의 수익률 요약\n"
                        f"━━━━━━━━━━━━━━━\n수익금: {format_price(profit)} USDT\n수익률: {percent:.2f}%"
                    )
                    report_sent = True

                time.sleep(LOOP_SLEEP_SEC)

            except Exception as loop_e:
                send_telegram(f"[오류 발생]\n{loop_e}")
                print(traceback.format_exc())
                time.sleep(60)

    except Exception as e:
        print("초기화 실패:", e)
        print(traceback.format_exc())
        send_telegram(f"[치명적 오류] 초기화 실패\n{e}")
