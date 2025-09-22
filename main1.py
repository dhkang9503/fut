#!/usr/bin/env python3
"""
Bitget Auto-Trader (Multi-Symbol Live Orders)
- Symbols: BTC/USDT, ETH/USDT @ 50x; SOL/USDT @ 30x  (USDT-M, Isolated)
- Opens/closes REAL positions via ccxt.create_order()
- Strategy: 1m trigger + 5m filter + 15m/1h trend filter + volume spike
- Min TP: +4% (covers taker ~0.08% * 50)
- Risk gate: skip if potential loss > 1.5% equity
- Protections: daily -5% stop, 3 consecutive losses cooldown, (optional) news block
- Telegram alerts

Requirements:
  pip install ccxt pandas numpy python-dotenv requests pytz
"""

import os, time, traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import ccxt
from dotenv import load_dotenv
import requests
from zoneinfo import ZoneInfo

# ========= CONFIG =========
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
LEVERAGE_MAP = {"BTC": 50, "ETH": 50, "SOL": 30}
MARGIN_FRAC = 0.01
RISK_PCT = 0.015
TP2_R = 1.2
MIN_PROFIT_PCT = 0.04
MAX_DAILY_LOSS_PCT = 0.05
COOLDOWN_AFTER_3_LOSSES_MIN = 60
LOOP_SLEEP_SEC = 5
PRICE_POLL_SEC = 3

BB_PERIOD = 20
BB_MULT = 2.0
CCI_PERIOD = 20
VOL_SPIKE_MULT = 1.5

KST = ZoneInfo("Asia/Seoul")
NYT = ZoneInfo("America/New_York")

# ========= TELEGRAM =========
load_dotenv()
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

def tg_send(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        print("[TG]", msg)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg}
        )
    except Exception as e:
        print("[TG ERROR]", e)

# ========= INDICATORS =========
def bollinger(close: pd.Series, period=BB_PERIOD, mult=BB_MULT):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    return mid, mid + mult*std, mid - mult*std

def cci(df: pd.DataFrame, period=CCI_PERIOD):
    tp = (df['high'] + df['low'] + df['close'])/3
    sma = tp.rolling(period).mean()
    mad = (tp - sma).abs().rolling(period).mean()
    return (tp - sma)/(0.015*mad)

# ========= NEWS BLOCKS (optional; quick fixed-times) =========
def build_news_blocks():
    blocks = []
    now = datetime.now(NYT).date()
    today = datetime.combine(now, datetime.min.time(), tzinfo=NYT)
    cpi = today.replace(hour=8, minute=30).astimezone(KST)
    fomc= today.replace(hour=14, minute=0).astimezone(KST)
    blocks.append((cpi - timedelta(minutes=30), cpi + timedelta(minutes=30), "CPI/PPI"))
    blocks.append((fomc - timedelta(minutes=30), fomc + timedelta(minutes=30), "FOMC"))
    return blocks

def in_news_block():
    now = datetime.now(KST)
    for s,e,label in build_news_blocks():
        if s <= now <= e:
            return True, label
    return False, None

# ========= STATE =========
@dataclass
class Position:
    symbol: str
    side: str  # long/short
    entry: float
    sl: float
    tp2: float
    size: float

STATE = {
    'open': None,  # Position | None
    'daily_start_equity': None,
    'daily_loss_usd': 0.0,
    'consec_losses': 0,
    'cooldown_until': 0.0,
    'day': None,
}

# ========= EXCHANGE =========
def build_exchange():
    key = os.getenv('BITGET_API_KEY'); secret = os.getenv('BITGET_API_SECRET'); password = os.getenv('BITGET_API_PASSWORD')
    if not (key and secret and password):
        raise RuntimeError("Missing BITGET_API_* env vars")
    return ccxt.bitget({
        'apiKey': key,
        'secret': secret,
        'password': password,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })

EX = None

def _base(sym: str) -> str:
    return sym.split('/')[0]

def get_leverage(sym: str) -> int:
    base = _base(sym)
    return LEVERAGE_MAP.get(base, 50)

def ensure_isolated_and_leverage(symbol: str):
    lev = get_leverage(symbol)
    try:
        EX.set_margin_mode('isolated', symbol, params={'productType': 'USDT-FUTURES'})
    except Exception:
        pass
    try:
        EX.set_leverage(lev, symbol, params={'productType': 'USDT-FUTURES'})
    except Exception:
        pass

# ========= DATA FETCH =========
def fetch_ohlcv(symbol, timeframe, limit=200):
    o = EX.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(o, columns=['ts','open','high','low','close','volume'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_convert(KST)
    df.set_index('dt', inplace=True)
    return df

def fetch_equity_usdt():
    bal = EX.fetch_balance({'type': 'swap'})
    usdt = bal.get('USDT') or {}
    return float(usdt.get('total') or usdt.get('free') or 0.0)

# ========= STRATEGY =========
def check_signal(df1, df5, df15, df60):
    mid1, up1, lo1 = bollinger(df1['close']); cci1 = cci(df1)
    mid5, up5, lo5 = bollinger(df5['close']); cci5 = cci(df5)
    cci15 = cci(df15); cci60 = cci(df60)
    if len(df1)<BB_PERIOD+2 or len(df5)<BB_PERIOD+2: return None
    if not(-50 <= cci15.iloc[-1] <= 50 and -50 <= cci60.iloc[-1] <= 50):
        return None
    c1_prev, c1 = df1['close'].iloc[-2], df1['close'].iloc[-1]
    cci1_prev, cci1_now = cci1.iloc[-2], cci1.iloc[-1]
    c5, mid5_now, cci5_prev, cci5_now = df5['close'].iloc[-1], mid5.iloc[-1], cci5.iloc[-2], cci5.iloc[-1]
    bull5 = (c5>=mid5_now and cci5_now>cci5_prev)
    bear5 = (c5<=mid5_now and cci5_now<cci5_prev)
    vol_mean = df1['volume'].iloc[-20:].mean(); vol_now = df1['volume'].iloc[-1]
    vol_ok = vol_now >= VOL_SPIKE_MULT*vol_mean
    long_cond = (c1_prev<lo1.iloc[-1] and c1>=lo1.iloc[-1]) and (cci1_prev<-100 and cci1_now>cci1_prev)
    short_cond= (c1_prev>up1.iloc[-1] and c1<=up1.iloc[-1]) and (cci1_prev>100 and cci1_now<cci1_prev)
    if bull5 and long_cond and vol_ok: return {'side':'long','price':float(c1)}
    if bear5 and short_cond and vol_ok: return {'side':'short','price':float(c1)}
    return None

# ========= ORDERS =========
def open_position(symbol: str, sig: dict, equity_usdt: float):
    ensure_isolated_and_leverage(symbol)
    price = sig['price']; side = sig['side']
    lev = get_leverage(symbol)
    margin = equity_usdt * MARGIN_FRAC
    nominal = margin * lev

    # SL distance (example 0.3%) — you can switch to swing/BB-based
    sl_dist = price * 0.003
    sl = (price - sl_dist) if side=='long' else (price + sl_dist)

    # Risk gate
    expected_loss = nominal * (abs(price - sl) / price)
    if expected_loss > equity_usdt * RISK_PCT:
        tg_send(f"[SKIP] Risk gate: exp loss {expected_loss:.4f} > {equity_usdt*RISK_PCT:.4f}")
        return None

    # TP target (max of 1.2R and +4%)
    if side=='long':
        tp2 = price + TP2_R * (price - sl)
        min_tp = price * (1 + MIN_PROFIT_PCT)
        tp2 = max(tp2, min_tp)
    else:
        tp2 = price - TP2_R * (sl - price)
        min_tp = price * (1 - MIN_PROFIT_PCT)
        tp2 = min(tp2, min_tp)

    # Size in base currency
    size = nominal / price

    # --- PLACE MARKET ORDER ---
    side_type = 'buy' if side=='long' else 'sell'
    try:
        order = EX.create_order(symbol=symbol, type='market', side=side_type, amount=size,
                                params={'marginMode': 'isolated', 'reduceOnly': False})
    except Exception as e:
        tg_send(f"[OPEN ERROR] {symbol} {side} size={size:.6f}: {e}")
        return None

    pos = Position(symbol, side, price, sl, tp2, size)
    STATE['open'] = pos
    tg_send(f"[OPEN] {symbol} {side.upper()} @ {price:.2f} size={size:.6f} | SL {sl:.2f} | TP {tp2:.2f} | lev {lev}x")
    return pos


def close_position_market(pos: Position, tag: str):
    side_type = 'sell' if pos.side=='long' else 'buy'
    try:
        EX.create_order(symbol=pos.symbol, type='market', side=side_type, amount=pos.size,
                        params={'reduceOnly': True, 'marginMode': 'isolated'})
        tg_send(f"[{tag}] {pos.symbol} {pos.side.upper()} closed")
    except Exception as e:
        tg_send(f"[CLOSE ERROR] {pos.symbol}: {e}")

# ========= POSITION MONITOR =========
def monitor_position():
    pos: Position = STATE['open']
    if not pos:
        return
    try:
        ticker = EX.fetch_ticker(pos.symbol)
        last = float(ticker['last'])
        if pos.side=='long':
            if last <= pos.sl:
                close_position_market(pos, 'STOP LOSS')
                STATE['open']=None; STATE['consec_losses']+=1
                if STATE['consec_losses']>=3:
                    STATE['cooldown_until']=time.time()+COOLDOWN_AFTER_3_LOSSES_MIN*60
                return
            if last >= pos.tp2:
                close_position_market(pos, 'TAKE PROFIT')
                STATE['open']=None; STATE['consec_losses']=0
                return
        else:
            if last >= pos.sl:
                close_position_market(pos, 'STOP LOSS')
                STATE['open']=None; STATE['consec_losses']+=1
                if STATE['consec_losses']>=3:
                    STATE['cooldown_until']=time.time()+COOLDOWN_AFTER_3_LOSSES_MIN*60
                return
            if last <= pos.tp2:
                close_position_market(pos, 'TAKE PROFIT')
                STATE['open']=None; STATE['consec_losses']=0
                return
    except Exception as e:
        tg_send(f"[MONITOR ERROR] {e}")

# ========= MAIN LOOP =========

def fetch_ohlcv_bundle(sym: str):
    df1 = fetch_ohlcv(sym, "1m", limit=BB_PERIOD+50)
    df5 = fetch_ohlcv(sym, "5m", limit=BB_PERIOD+50)
    df15= fetch_ohlcv(sym, "15m",limit=BB_PERIOD+50)
    df60= fetch_ohlcv(sym, "1h", limit=BB_PERIOD+50)
    return df1, df5, df15, df60


def main():
    global EX
    EX = build_exchange()
    tg_send("Bot started (Multi-Symbol LIVE)")

    while True:
        try:
            # Day init
            today = datetime.now(KST).date()
            if STATE['daily_start_equity'] is None or STATE['day'] != str(today):
                eq = fetch_equity_usdt()
                STATE['daily_start_equity']=eq; STATE['daily_loss_usd']=0.0; STATE['consec_losses']=0; STATE['day']=str(today)
                tg_send(f"[DAY START] equity={eq}")

            # Cooldown & daily stop
            if time.time() < STATE['cooldown_until']:
                time.sleep(LOOP_SLEEP_SEC); continue
            eq_now = fetch_equity_usdt(); daily_pnl = eq_now - STATE['daily_start_equity']
            if daily_pnl < -MAX_DAILY_LOSS_PCT * STATE['daily_start_equity']:
                tg_send("[DAILY STOP] exceeded -5% — pausing until tomorrow")
                STATE['cooldown_until']= time.time()+3600*24
                time.sleep(60); continue

            # Optional news block
            blocked, label = in_news_block()
            if blocked and STATE['open'] is None:
                tg_send(f"[NEWS BLOCK] {label}, skip new entries")
                time.sleep(60); continue

            # If open, monitor; else scan symbols in order
            if STATE['open']:
                monitor_position()
            else:
                for sym in SYMBOLS:
                    df1, df5, df15, df60 = fetch_ohlcv_bundle(sym)
                    sig = check_signal(df1, df5, df15, df60)
                    if sig:
                        pos = open_position(sym, sig, eq_now)
                        if pos:  # opened successfully
                            break
        except Exception as e:
            tg_send(f"[ERROR] {e}\n{traceback.format_exc()}")
            time.sleep(10)
        time.sleep(LOOP_SLEEP_SEC)

if __name__ == "__main__":
    main()
