#!/usr/bin/env python3
"""
Bitget Auto-Trader (Multi-Symbol: BTC, ETH, SOL)
- 기본 전략은 동일 (볼밴+CCI, 다중 타임프레임, 리스크 게이트)
- 심볼: BTC/USDT, ETH/USDT, SOL/USDT (USDT-M, isolated)
- 우선순위: BTC > ETH > SOL
- 포지션이 없으면 3개 심볼을 모두 스캔하고 신호가 뜬 심볼에서 진입
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
LEVERAGE = 50
MARGIN_FRAC = 0.10
RISK_PCT = 0.015
TP2_R = 1.2
MIN_PROFIT_PCT = 0.04
MAX_DAILY_LOSS_PCT = 0.05
COOLDOWN_AFTER_3_LOSSES_MIN = 60

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

# ========= STATE =========
@dataclass
class Position:
    symbol: str
    side: str
    entry: float
    sl: float
    tp2: float
    size: float
    moved_be: bool = False

STATE = {
    'open': None,
    'daily_start_equity': None,
    'daily_loss_usd': 0.0,
    'consec_losses': 0,
    'cooldown_until': 0.0,
}

# ========= EXCHANGE =========
def build_exchange():
    key = os.getenv('BITGET_API_KEY')
    secret = os.getenv('BITGET_API_SECRET')
    password = os.getenv('BITGET_API_PASSWORD')
    return ccxt.bitget({
        'apiKey': key,
        'secret': secret,
        'password': password,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })

EX = None

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
    mid1, up1, lo1 = bollinger(df1['close'])
    cci1 = cci(df1)
    mid5, up5, lo5 = bollinger(df5['close'])
    cci5 = cci(df5)
    cci15 = cci(df15)
    cci60 = cci(df60)
    if len(df1)<BB_PERIOD+2 or len(df5)<BB_PERIOD+2: return None
    if not(-50 <= cci15.iloc[-1] <= 50 and -50 <= cci60.iloc[-1] <= 50):
        return None
    c1_prev, c1 = df1['close'].iloc[-2], df1['close'].iloc[-1]
    cci1_prev, cci1_now = cci1.iloc[-2], cci1.iloc[-1]
    c5, mid5_now, cci5_prev, cci5_now = df5['close'].iloc[-1], mid5.iloc[-1], cci5.iloc[-2], cci5.iloc[-1]
    bull5 = (c5>=mid5_now and cci5_now>cci5_prev)
    bear5 = (c5<=mid5_now and cci5_now<cci5_prev)
    vol_mean = df1['volume'].iloc[-20:].mean()
    vol_now = df1['volume'].iloc[-1]
    vol_ok = vol_now >= VOL_SPIKE_MULT*vol_mean
    long_cond = (c1_prev<lo1.iloc[-1] and c1>=lo1.iloc[-1]) and (cci1_prev<-100 and cci1_now>cci1_prev)
    short_cond= (c1_prev>up1.iloc[-1] and c1<=up1.iloc[-1]) and (cci1_prev>100 and cci1_now<cci1_prev)
    if bull5 and long_cond and vol_ok: return {'side':'long','price':c1}
    if bear5 and short_cond and vol_ok: return {'side':'short','price':c1}
    return None

# ========= ORDER EXECUTION =========
def open_position(symbol, sig, eq):
    price = sig['price']
    side = sig['side']
    margin = eq*MARGIN_FRAC
    nominal = margin*LEVERAGE
    sl_dist = price*0.003
    if side=='long': sl = price - sl_dist
    else: sl = price + sl_dist
    tp2 = price + (price-sl)*TP2_R if side=='long' else price - (sl-price)*TP2_R
    min_tp = price*(1+MIN_PROFIT_PCT) if side=='long' else price*(1-MIN_PROFIT_PCT)
    if side=='long' and tp2<min_tp: tp2=min_tp
    if side=='short' and tp2>min_tp: tp2=min_tp
    size = nominal/price
    pos = Position(symbol, side, price, sl, tp2, size)
    STATE['open']=pos
    tg_send(f"[OPEN] {symbol} {side} {size:.4f} @ {price}, SL {sl}, TP2 {tp2}")

# ========= POSITION MONITOR =========
def monitor_position():
    pos: Position = STATE['open']
    if not pos: return
    try:
        ticker = EX.fetch_ticker(pos.symbol)
        last = ticker['last']
        if pos.side=='long':
            if last<=pos.sl:
                tg_send(f"[STOP LOSS] {pos.symbol} long closed @ {last}")
                STATE['open']=None; STATE['consec_losses']+=1
                if STATE['consec_losses']>=3:
                    STATE['cooldown_until']=time.time()+COOLDOWN_AFTER_3_LOSSES_MIN*60
                return
            if last>=pos.tp2:
                tg_send(f"[TAKE PROFIT] {pos.symbol} long closed @ {last}")
                STATE['open']=None; STATE['consec_losses']=0
                return
        else:
            if last>=pos.sl:
                tg_send(f"[STOP LOSS] {pos.symbol} short closed @ {last}")
                STATE['open']=None; STATE['consec_losses']+=1
                if STATE['consec_losses']>=3:
                    STATE['cooldown_until']=time.time()+COOLDOWN_AFTER_3_LOSSES_MIN*60
                return
            if last<=pos.tp2:
                tg_send(f"[TAKE PROFIT] {pos.symbol} short closed @ {last}")
                STATE['open']=None; STATE['consec_losses']=0
                return
    except Exception as e:
        tg_send(f"[MONITOR ERROR] {e}")

# ========= MAIN LOOP =========
def main():
    global EX
    EX = build_exchange()
    tg_send("Bot started (Multi-Symbol Auto)")
    while True:
        try:
            today = datetime.now(KST).date()
            if STATE['daily_start_equity'] is None or STATE.get('day')!=str(today):
                eq = fetch_equity_usdt()
                STATE['daily_start_equity']=eq; STATE['daily_loss_usd']=0.0; STATE['consec_losses']=0; STATE['day']=str(today)
                tg_send(f"[DAY START] equity={eq}")
            if time.time()<STATE['cooldown_until']: time.sleep(5); continue
            eq_now=fetch_equity_usdt(); daily_pnl=eq_now-STATE['daily_start_equity']
            if daily_pnl< -MAX_DAILY_LOSS_PCT*STATE['daily_start_equity']:
                tg_send("[DAILY STOP] exceeded -5%")
                STATE['cooldown_until']=time.time()+3600*24
                time.sleep(60); continue
            if not STATE['open']:
                for sym in SYMBOLS:
                    df1=fetch_ohlcv(sym,"1m",limit=BB_PERIOD+50)
                    df5=fetch_ohlcv(sym,"5m",limit=BB_PERIOD+50)
                    df15=fetch_ohlcv(sym,"15m",limit=BB_PERIOD+50)
                    df60=fetch_ohlcv(sym,"1h",limit=BB_PERIOD+50)
                    sig=check_signal(df1,df5,df15,df60)
                    if sig:
                        open_position(sym, sig, eq_now)
                        break
            else:
                monitor_position()
        except Exception as e:
            tg_send(f"[ERROR] {e}\n{traceback.format_exc()}")
            time.sleep(10)
        time.sleep(5)

if __name__=="__main__":
    main()
