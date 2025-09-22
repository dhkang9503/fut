#!/usr/bin/env python3
"""
Bitget Auto-Trader (LIVE + Auto Mode Switch + Risk Modes)
- Symbols: BTC/USDT, ETH/USDT @ 50x; SOL/USDT @ 30x  (USDT-M, Isolated)
- Strategy Modes:
  • REVERSION (default): BB(20,2) + CCI(20) 역추세 스캘핑
  • TREND (auto): CCI 극단 + 볼밴 확장 + 거래량 폭발 → 돌파 추종 (ATR SL/트레일)
- **Risk Modes (Telegram /risk)**
  • fast   : 복구 모드 (신호 완화, 진입 기회↑)
  • normal : 기본 모드 (현재 기본값)
  • safe   : 보존 모드 (4h 필터+엄격 조건)
- Auto mode switch: REVERSION↔TREND 자동 전환 (+ 5분 히스테리시스)
- REAL orders via ccxt.create_order(), reduceOnly=True on exits
- Protections: min TP +4%, risk gate (≤1.5% equity), daily -5% stop, 3-loss cooldown, (optional) news block
- Telegram: alerts + commands (/ping, /status, /position, /set margin x, /mode [auto|reversion|trend], **/risk [fast|normal|safe]**)

Requirements:
  pip install ccxt pandas numpy python-dotenv requests pytz
"""

import os, time, traceback, threading
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
from dotenv import load_dotenv
import requests
from zoneinfo import ZoneInfo

# ========= CONFIG =========
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
LEVERAGE_MAP = {"BTC": 50, "ETH": 50, "SOL": 30}
MARGIN_FRAC = 0.10
RISK_PCT = 0.015
TP2_R = 1.2
MIN_PROFIT_PCT = 0.04
MAX_DAILY_LOSS_PCT = 0.05
COOLDOWN_AFTER_3_LOSSES_MIN = 60
LOOP_SLEEP_SEC = 5
PRICE_POLL_SEC = 3

# Base indicators
BB_PERIOD = 20
BB_MULT = 2.0
CCI_PERIOD = 20
ATR_LEN = 14
ATR_SL_MULT = 0.8
ATR_TP_ANCHOR = 1.5
ATR_TRAIL_STEP = 0.5

# Auto-mode switching
MODE_LOCK_SEC = 300  # 5m

KST = ZoneInfo("Asia/Seoul")
NYT = ZoneInfo("America/New_York")

# ========= TELEGRAM =========
load_dotenv()
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"
_last_update_id = None

def tg_send(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        print("[TG]", msg)
        return
    try:
        requests.post(f"{TG_API}/sendMessage", json={"chat_id": TG_CHAT, "text": msg})
    except Exception as e:
        print("[TG ERROR]", e)

def tg_get_updates(timeout=10):
    global _last_update_id
    if not TG_TOKEN:
        return []
    params = {"timeout": timeout}
    if _last_update_id is not None:
        params["offset"] = _last_update_id + 1
    try:
        r = requests.get(f"{TG_API}/getUpdates", params=params, timeout=timeout+5)
        data = r.json()
        if not data.get("ok"):
            return []
        updates = data.get("result", [])
        if updates:
            _last_update_id = updates[-1]["update_id"]
        return updates
    except Exception:
        return []

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

def true_range(df: pd.DataFrame):
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n=ATR_LEN):
    tr = true_range(df)
    return tr.rolling(n).mean()

def bb_bandwidth(df: pd.DataFrame):
    mid, up, lo = bollinger(df['close'])
    midv = float(mid.iloc[-1]) if not np.isnan(mid.iloc[-1]) else 0.0
    if midv == 0.0:
        return 0.0
    return float((up.iloc[-1] - lo.iloc[-1]) / midv)

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
    mode: str  # REVERSION | TREND

STATE = {
    'open': None,
    'daily_start_equity': None,
    'daily_loss_usd': 0.0,
    'consec_losses': 0,
    'cooldown_until': 0.0,
    'day': None,
    'mode': 'REVERSION',
    'auto_mode': True,
    'last_mode_switch_ts': 0.0,
    'risk_mode': 'normal',  # fast | normal | safe
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

# ========= RISK MODE PARAMS =========

def get_params():
    rm = STATE.get('risk_mode','normal')
    # defaults (normal)
    p = {
        'VOL_SPIKE_MULT_REV': 1.5,
        'VOL_SPIKE_MULT_TR' : 2.0,
        'BB_EXPAND_BW_TR'   : 0.010,  # 1.0%
        'BB_COOL_BW_RV'     : 0.007,  # 0.7%
        'CCI_TREND_ON'      : 200,
        'CCI_TREND_OFF'     : 100,
        'USE_4H_FILTER'     : False,
    }
    if rm == 'fast':
        p.update({
            'VOL_SPIKE_MULT_REV': 1.2,
            'VOL_SPIKE_MULT_TR' : 1.5,
            'BB_EXPAND_BW_TR'   : 0.008,
            'BB_COOL_BW_RV'     : 0.008,  # 조금 느슨
            'CCI_TREND_ON'      : 180,
            'CCI_TREND_OFF'     : 80,
            'USE_4H_FILTER'     : False,
        })
    elif rm == 'safe':
        p.update({
            'VOL_SPIKE_MULT_REV': 1.6,
            'VOL_SPIKE_MULT_TR' : 2.2,
            'BB_EXPAND_BW_TR'   : 0.012,
            'BB_COOL_BW_RV'     : 0.006,
            'CCI_TREND_ON'      : 220,
            'CCI_TREND_OFF'     : 100,
            'USE_4H_FILTER'     : True,
        })
    return p

# ========= SIGNALS =========

def check_signal_reversion(df1, df5, df15, df60, df4h=None):
    P = get_params()
    # 4h 필터 (safe 모드에서만 활성)
    if P['USE_4H_FILTER'] and df4h is not None:
        c4h = float(cci(df4h).iloc[-1])
        if abs(c4h) > 200:
            return None  # 강추세 역추세 금지
    mid1, up1, lo1 = bollinger(df1['close']); cci1 = cci(df1)
    mid5, up5, lo5 = bollinger(df5['close']); cci5 = cci(df5)
    cci15 = cci(df15); cci60 = cci(df60)
    if len(df1)<BB_PERIOD+2 or len(df5)<BB_PERIOD+2: return None
    # 추세 중립 필터 (1h/15m)
    if not(-50 <= cci15.iloc[-1] <= 50 and -50 <= cci60.iloc[-1] <= 50):
        return None
    c1_prev, c1 = df1['close'].iloc[-2], df1['close'].iloc[-1]
    cci1_prev, cci1_now = cci1.iloc[-2], cci1.iloc[-1]
    c5, mid5_now, cci5_prev, cci5_now = df5['close'].iloc[-1], mid5.iloc[-1], cci5.iloc[-2], cci5.iloc[-1]
    bull5 = (c5>=mid5_now and cci5_now>cci5_prev)
    bear5 = (c5<=mid5_now and cci5_now<cci5_prev)
    vol_mean = df1['volume'].iloc[-20:].mean(); vol_now = df1['volume'].iloc[-1]
    vol_ok = vol_now >= P['VOL_SPIKE_MULT_REV']*vol_mean
    long_cond = (c1_prev<lo1.iloc[-1] and c1>=lo1.iloc[-1]) and (cci1_prev<-P['CCI_TREND_OFF'] and cci1_now>cci1_prev)
    short_cond= (c1_prev>up1.iloc[-1] and c1<=up1.iloc[-1]) and (cci1_prev> P['CCI_TREND_OFF'] and cci1_now<cci1_prev)
    if bull5 and long_cond and vol_ok: return {'side':'long','price':float(c1)}
    if bear5 and short_cond and vol_ok: return {'side':'short','price':float(c1)}
    return None


def check_signal_trend(df1, df5, df15, df60, df4h=None):
    P = get_params()
    cci5_last = float(cci(df5).iloc[-1]); cci15_last = float(cci(df15).iloc[-1])
    bw = bb_bandwidth(df5)
    vol_ok = df1['volume'].iloc[-1] >= P['VOL_SPIKE_MULT_TR'] * df1['volume'].iloc[-20:].mean()
    if not ((abs(cci5_last)>=P['CCI_TREND_ON'] or abs(cci15_last)>=P['CCI_TREND_ON']) and bw>=P['BB_EXPAND_BW_TR'] and vol_ok):
        return None
    # 4h 방향 정합(안전 모드): 4h와 같은 방향만
    if P['USE_4H_FILTER'] and df4h is not None:
        c4h = float(cci(df4h).iloc[-1])
        if (c4h>200 and cci5_last<0) or (c4h<-200 and cci5_last>0):
            return None
    mid5, up5, lo5 = bollinger(df5['close'])
    above = (df5['close'].iloc[-1] >= up5.iloc[-1]) and (df5['close'].iloc[-2] >= up5.iloc[-2])
    below = (df5['close'].iloc[-1] <= lo5.iloc[-1]) and (df5['close'].iloc[-2] <= lo5.iloc[-2])
    if above and cci5_last >= float(cci(df5).iloc[-2]):
        return {'side':'long','price':float(df1['close'].iloc[-1]), 'trend': True}
    if below and cci5_last <= float(cci(df5).iloc[-2]):
        return {'side':'short','price':float(df1['close'].iloc[-1]), 'trend': True}
    return None

# ========= MODE SWITCH =========

def should_switch_to_trend(df1, df5, df15):
    P = get_params()
    c5 = float(cci(df5).iloc[-1]); c15 = float(cci(df15).iloc[-1])
    bw = bb_bandwidth(df5)
    vol_ok = df1['volume'].iloc[-1] >= P['VOL_SPIKE_MULT_TR'] * df1['volume'].iloc[-20:].mean()
    out_2sigma = (
        df5['high'].iloc[-1] >= bollinger(df5['close'])[1].iloc[-1] or
        df5['low'].iloc[-1]  <= bollinger(df5['close'])[2].iloc[-1]
    )
    return (abs(c5)>=P['CCI_TREND_ON'] or abs(c15)>=P['CCI_TREND_ON']) and (bw>=P['BB_EXPAND_BW_TR']) and vol_ok and out_2sigma

def should_switch_to_reversion(df5, df15):
    P = get_params()
    c5 = float(cci(df5).iloc[-1]); c15 = float(cci(df15).iloc[-1])
    bw = bb_bandwidth(df5)
    return (abs(c5)<=P['CCI_TREND_OFF'] and abs(c15)<=P['CCI_TREND_OFF']) and (bw < P['BB_COOL_BW_RV'])

def maybe_switch_mode(df1, df5, df15):
    if not STATE['auto_mode']:
        return
    now = time.time()
    if now - STATE.get('last_mode_switch_ts', 0) < MODE_LOCK_SEC:
        return
    if STATE['mode']=='REVERSION' and should_switch_to_trend(df1, df5, df15):
        STATE['mode']='TREND'; STATE['last_mode_switch_ts']=now
        tg_send(f"[MODE] switched to TREND (risk={STATE['risk_mode']})")
    elif STATE['mode']=='TREND' and should_switch_to_reversion(df5, df15):
        STATE['mode']='REVERSION'; STATE['last_mode_switch_ts']=now
        tg_send(f"[MODE] back to REVERSION (risk={STATE['risk_mode']})")

# ========= ORDERS =========

def open_position(symbol: str, sig: dict, equity_usdt: float):
    global MARGIN_FRAC
    ensure_isolated_and_leverage(symbol)
    price = float(sig['price']); side = sig['side']
    mode = STATE['mode']
    lev = get_leverage(symbol)
    margin = equity_usdt * MARGIN_FRAC
    nominal = margin * lev

    if mode=='REVERSION':
        sl_dist = price * 0.003
        sl = (price - sl_dist) if side=='long' else (price + sl_dist)
        expected_loss = nominal * (abs(price - sl) / price)
        if expected_loss > equity_usdt * RISK_PCT:
            tg_send(f"[SKIP] Risk gate: exp loss {expected_loss:.4f} > {equity_usdt*RISK_PCT:.4f}")
            return None
        if side=='long':
            tp2 = max(price + TP2_R*(price - sl), price*(1+MIN_PROFIT_PCT))
        else:
            tp2 = min(price - TP2_R*(sl - price), price*(1-MIN_PROFIT_PCT))
    else:
        df5 = fetch_ohlcv(symbol, "5m", limit=max(BB_PERIOD, ATR_LEN)+50)
        a = float(atr(df5, ATR_LEN).iloc[-1])
        if a == 0 or np.isnan(a):
            return None
        sl = (price - ATR_SL_MULT*a) if side=='long' else (price + ATR_SL_MULT*a)
        expected_loss = nominal * (abs(price - sl) / price)
        if expected_loss > equity_usdt * RISK_PCT:
            tg_send(f"[SKIP] Risk gate: exp loss {expected_loss:.4f} > {equity_usdt*RISK_PCT:.4f}")
            return None
        tp_anchor = ATR_TP_ANCHOR * a
        tp2 = (price + tp_anchor) if side=='long' else (price - tp_anchor)
        min_tp = price*(1+MIN_PROFIT_PCT) if side=='long' else price*(1-MIN_PROFIT_PCT)
        if (side=='long' and tp2 < min_tp): tp2 = min_tp
        if (side=='short' and tp2 > min_tp): tp2 = min_tp

    size = nominal / price
    side_type = 'buy' if side=='long' else 'sell'
    try:
        EX.create_order(symbol=symbol, type='market', side=side_type, amount=size,
                        params={'marginMode': 'isolated', 'reduceOnly': False})
    except Exception as e:
        tg_send(f"[OPEN ERROR] {symbol} {side} size={size:.6f}: {e}")
        return None

    pos = Position(symbol, side, price, sl, tp2, size, mode)
    STATE['open'] = pos
    tg_send(f"[OPEN] {symbol} {side.upper()} @ {price:.2f} size={size:.6f} | SL {sl:.2f} | TP {tp2:.2f} | lev {lev}x | margin_frac {MARGIN_FRAC} | mode {mode} | risk {STATE['risk_mode']}")
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
        last = float(EX.fetch_ticker(pos.symbol)['last'])
        if pos.mode=='REVERSION':
            if pos.side=='long':
                if last <= pos.sl:
                    close_position_market(pos, 'STOP LOSS'); STATE['open']=None; STATE['consec_losses']+=1
                    if STATE['consec_losses']>=3: STATE['cooldown_until']=time.time()+COOLDOWN_AFTER_3_LOSSES_MIN*60
                    return
                if last >= pos.tp2:
                    close_position_market(pos, 'TAKE PROFIT'); STATE['open']=None; STATE['consec_losses']=0; return
            else:
                if last >= pos.sl:
                    close_position_market(pos, 'STOP LOSS'); STATE['open']=None; STATE['consec_losses']+=1
                    if STATE['consec_losses']>=3: STATE['cooldown_until']=time.time()+COOLDOWN_AFTER_3_LOSSES_MIN*60
                    return
                if last <= pos.tp2:
                    close_position_market(pos, 'TAKE PROFIT'); STATE['open']=None; STATE['consec_losses']=0; return
        else:
            df5 = fetch_ohlcv(pos.symbol, "5m", limit=max(BB_PERIOD, ATR_LEN)+50)
            a = float(atr(df5, ATR_LEN).iloc[-1])
            if pos.side=='long':
                trail = last - ATR_TRAIL_STEP*a
                if trail > pos.sl: pos.sl = trail
                if last <= pos.sl:
                    close_position_market(pos, 'STOP LOSS'); STATE['open']=None; STATE['consec_losses']+=1
                    if STATE['consec_losses']>=3: STATE['cooldown_until']=time.time()+COOLDOWN_AFTER_3_LOSSES_MIN*60
                    return
                if last >= pos.tp2:
                    close_position_market(pos, 'TAKE PROFIT'); STATE['open']=None; STATE['consec_losses']=0; return
            else:
                trail = last + ATR_TRAIL_STEP*a
                if trail < pos.sl: pos.sl = trail
                if last >= pos.sl:
                    close_position_market(pos, 'STOP LOSS'); STATE['open']=None; STATE['consec_losses']+=1
                    if STATE['consec_losses']>=3: STATE['cooldown_until']=time.time()+COOLDOWN_AFTER_3_LOSSES_MIN*60
                    return
                if last <= pos.tp2:
                    close_position_market(pos, 'TAKE PROFIT'); STATE['open']=None; STATE['consec_losses']=0; return
    except Exception as e:
        tg_send(f"[MONITOR ERROR] {e}")

# ========= TELEGRAM COMMANDS =========

def _fmt_pos() -> str:
    pos: Position = STATE['open']
    if not pos:
        return "No open position"
    try:
        last = float(EX.fetch_ticker(pos.symbol)['last'])
    except Exception:
        last = None
    lines = [
        f"symbol: {pos.symbol}",
        f"side: {pos.side}",
        f"entry: {pos.entry}",
        f"sl: {pos.sl}",
        f"tp: {pos.tp2}",
        f"size: {pos.size}",
        f"mode: {pos.mode}",
        f"risk: {STATE['risk_mode']}",
    ]
    if last is not None:
        pnl = (last - pos.entry) / pos.entry * (1 if pos.side=='long' else -1)
        lines.append(f"last: {last} (pnl: {pnl*100:.2f}%)")
    return "\n".join(lines)


def process_command(cmd_text: str):
    global MARGIN_FRAC
    t = cmd_text.strip()
    if t.startswith('/ping'):
        tg_send('pong ✅'); return
    if t.startswith('/status'):
        try: eq = fetch_equity_usdt()
        except Exception: eq = None
        msg = ["[STATUS]",
               f"equity={eq}",
               f"cooldown_until={STATE['cooldown_until']}",
               f"consec_losses={STATE['consec_losses']}",
               f"margin_frac={MARGIN_FRAC}",
               f"mode={STATE['mode']}",
               f"auto_mode={STATE['auto_mode']}",
               f"risk_mode={STATE['risk_mode']}" ]
        tg_send("\n".join(msg)); return
    if t.startswith('/position') or t.startswith('/pos'):
        tg_send(_fmt_pos()); return
    if t.startswith('/set'):
        parts = t.split()
        if len(parts)>=3 and parts[1].lower()=='margin':
            try:
                val = float(parts[2])
                if 0.001 <= val <= 0.25:
                    MARGIN_FRAC = val; tg_send(f"[SET] margin_frac={MARGIN_FRAC}")
                else:
                    tg_send("[SET ERROR] margin must be 0.001~0.25")
            except Exception:
                tg_send("[SET ERROR] usage: /set margin 0.01")
        else:
            tg_send("[SET] unknown or missing parameter. try: /set margin 0.01")
        return
    if t.startswith('/mode'):
        parts = t.split()
        if len(parts)==1:
            tg_send(f"mode={STATE['mode']} auto_mode={STATE['auto_mode']}"); return
        arg = parts[1].lower()
        if arg=='auto':
            STATE['auto_mode']=True; tg_send('[MODE] auto on'); return
        if arg in ('reversion','trend'):
            STATE['auto_mode']=False; STATE['mode']=arg.upper(); STATE['last_mode_switch_ts']=time.time()
            tg_send(f"[MODE] forced to {STATE['mode']} (auto off)"); return
        tg_send('usage: /mode [auto|reversion|trend]'); return
    if t.startswith('/risk'):
        parts = t.split()
        if len(parts)==1:
            tg_send(f"risk_mode={STATE['risk_mode']}"); return
        arg = parts[1].lower()
        if arg in ('fast','normal','safe'):
            STATE['risk_mode'] = arg
            tg_send(f"[RISK] set to {arg}")
            return
        tg_send('usage: /risk [fast|normal|safe]'); return


def command_poller():
    while True:
        try:
            ups = tg_get_updates(timeout=10)
            for u in ups:
                msg = u.get('message') or {}
                chat_id = str(msg.get('chat',{}).get('id',''))
                text = msg.get('text','')
                if not text:
                    continue
                if TG_CHAT and chat_id != str(TG_CHAT):
                    continue
                process_command(text)
        except Exception:
            pass
        time.sleep(1)

# ========= MAIN LOOP =========

def fetch_ohlcv_bundle(sym: str):
    df1 = fetch_ohlcv(sym, "1m", limit=BB_PERIOD+50)
    df5 = fetch_ohlcv(sym, "5m", limit=BB_PERIOD+50)
    df15= fetch_ohlcv(sym, "15m",limit=BB_PERIOD+50)
    df60= fetch_ohlcv(sym, "1h", limit=BB_PERIOD+50)
    df4h= fetch_ohlcv(sym, "4h", limit=BB_PERIOD+50)
    return df1, df5, df15, df60, df4h


def main():
    global EX
    EX = build_exchange()
    tg_send("Bot started (LIVE + Auto Mode + Risk Modes)")

    if TG_TOKEN:
        threading.Thread(target=command_poller, daemon=True).start()

    while True:
        try:
            today = datetime.now(KST).date()
            if STATE['daily_start_equity'] is None or STATE['day'] != str(today):
                eq = fetch_equity_usdt()
                STATE['daily_start_equity']=eq; STATE['daily_loss_usd']=0.0; STATE['consec_losses']=0; STATE['day']=str(today)
                tg_send(f"[DAY START] equity={eq}")

            if time.time() < STATE['cooldown_until']:
                time.sleep(LOOP_SLEEP_SEC); continue
            eq_now = fetch_equity_usdt(); daily_pnl = eq_now - STATE['daily_start_equity']
            if daily_pnl < -MAX_DAILY_LOSS_PCT * STATE['daily_start_equity']:
                tg_send("[DAILY STOP] exceeded -5% — pausing until tomorrow")
                STATE['cooldown_until']= time.time()+3600*24
                time.sleep(60); continue

            #blocked, label = in_news_block()
            #if blocked and STATE['open'] is None:
                #tg_send(f"[NEWS BLOCK] {label}, skip new entries")
                #time.sleep(60); continue

            if STATE['open']:
                monitor_position()
            else:
                for sym in SYMBOLS:
                    df1, df5, df15, df60, df4h = fetch_ohlcv_bundle(sym)
                    maybe_switch_mode(df1, df5, df15)
                    if STATE['mode']=='REVERSION':
                        sig = check_signal_reversion(df1, df5, df15, df60, df4h)
                    else:
                        sig = check_signal_trend(df1, df5, df15, df60, df4h)
                    if sig:
                        pos = open_position(sym, sig, eq_now)
                        if pos:
                            break
        except Exception as e:
            tg_send(f"[ERROR] {e}\n{traceback.format_exc()}")
            time.sleep(10)
        time.sleep(LOOP_SLEEP_SEC)

if __name__ == "__main__":
    main()
