#!/usr/bin/env python3
"""
Bitget 자동매매 봇 (BTC+SOL 전용) — 1순위 리스크 + SL-ONLY 서버사이드 + BE(수수료 포함)

핵심 변경
  • 진입 직후: **서버사이드 SL만** 생성(반드시 reduceOnly)
  • TP1/TP2: 로직으로 체결 (부분청산/완전청산)
  • TP1 체결 시: SL을 **진입가 ± (2*FEE_RATE)** 로 이동(수수료까지 BE)  
    - 롱: new_stop = entry * (1 + 2*FEE_RATE)  
    - 숏: new_stop = entry * (1 - 2*FEE_RATE)
    ※ 레버리지는 BE 가격 오프셋(%)에 **영향 없음**. 수수료는 노치널 기준이므로, 오프셋 %는 대략 2*fee로 충분.

환경변수
  - BITGET_API_KEY, BITGET_API_SECRET, BITGET_API_PASSWORD
  - TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  - (선택) FEE_RATE  (기본 0.0008 = 0.08%/사이드)

리스크(1순위)
  1) 서버사이드 SL
  2) 일일 하드락 -4% (수동 /resume 필요)
  3) 실계좌 동기화(fetch_balance)
  4) TG 킬스위치: /panic /pause /resume
"""
import os, json, time, logging, math, requests
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Optional, List

# ---- Lean logger: send most messages to Telegram; print only critical ----
class _LeanLogger:
    def info(self, msg: str):
        try:
            tg_send(str(msg))
        except Exception:
            pass
    def warning(self, msg: str):
        try:
            tg_send("⚠️ " + str(msg))
        except Exception:
            pass
    def error(self, msg: str, critical: bool = True):
        try:
            tg_send("❌ " + str(msg))
        except Exception:
            pass
        if critical:
            try:
                print(f"[CRITICAL] {msg}")
            except Exception:
                pass

# replace std logging with lean logger (shadow the module)
logging = _LeanLogger()

# ---------------------- ENV ----------------------
BITGET_API_KEY      = os.getenv("BITGET_API_KEY", "")
BITGET_API_SECRET   = os.getenv("BITGET_API_SECRET", "")
BITGET_API_PASSWORD = os.getenv("BITGET_API_PASSWORD", "")
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
FEE_RATE = float(os.getenv("FEE_RATE", "0.0008"))  # per side

SYMBOLS = {"BTCUSDT": {"leverage": 100}, "SOLUSDT": {"leverage": 60}}
INTERVAL = "1h"; BB_PERIOD=20; BB_NSTD=2.0; CCI_PERIOD=20
SL_PCT=0.02; TP1_PCT=0.04; TP2_PCT=0.06; DAILY_STOP=0.04
STATE_FILE="state.json"; LOG_FILE=None

# logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------- DATA ----------------------
@dataclass
class Position:
    symbol: str; side: str; entry: float; notional: float
    remaining_frac: float = 1.0; tp1_hit: bool=False; stop: float=0.0
    opened_at: str = ""; sl_order_id: Optional[str] = None

@dataclass
class State:
    equity: float = 2000.0; current_day: str = ""; day_start_equity: float = 2000.0
    open_pos: Optional[Position] = None; last_update_id: int = 0
    sizing_mode: str = "percent"; sizing_value: float = 0.10
    hard_paused: bool = False

# ---------------------- UTIL ----------------------
def now_utc(): return datetime.now(timezone.utc)

def load_state()->State:
    if os.path.exists(STATE_FILE):
        d=json.load(open(STATE_FILE)); pos=d.get("open_pos")
        return State(d.get("equity",2000.0), d.get("current_day", now_utc().date().isoformat()),
                     d.get("day_start_equity",2000.0), Position(**pos) if pos else None,
                     d.get("last_update_id",0), d.get("sizing_mode","percent"), d.get("sizing_value",0.10),
                     d.get("hard_paused", False))
    s=State(); s.current_day=now_utc().date().isoformat(); s.day_start_equity=s.equity; save_state(s); return s

def save_state(s:State):
    d=asdict(s); d["open_pos"]=asdict(s.open_pos) if s.open_pos else None
    json.dump(d, open(STATE_FILE,"w"), indent=2)

# ---------------------- Telegram ----------------------
TG_BASE=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

def tg_send(text:str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    try: requests.post(f"{TG_BASE}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e: logging.error(f"Telegram send error: {e}")

def parse_size_command(text:str,s:State)->Optional[str]:
    parts=text.strip().split();
    if len(parts)<3: return "사용법: /size percent <숫자> 또는 /size fixed <USDT>"
    _,mode,val=parts[:3]
    try: num=float(val)
    except: return "숫자를 올바르게 입력하세요. 예) /size percent 10 또는 /size fixed 100"
    if mode.lower()=="percent":
        if not(0<num<=100): return "percent 값은 0~100 사이여야 합니다."
        s.sizing_mode="percent"; s.sizing_value=num/100.0
    elif mode.lower()=="fixed":
        if num<=0: return "fixed 금액은 0보다 커야 합니다."
        s.sizing_mode="fixed"; s.sizing_value=num
    else: return "mode는 percent 또는 fixed"
    save_state(s); return f"포지션 크기: {s.sizing_mode}={(s.sizing_value if s.sizing_mode=='fixed' else str(int(s.sizing_value*100))+'%')}"

def tg_poll_and_handle(s: State):
    if not TELEGRAM_BOT_TOKEN:
        return
    try:
        r = requests.get(
            f"{TG_BASE}/getUpdates",
            params={"timeout": 5, "offset": s.last_update_id + 1}
        )
        data = r.json()
        if not data.get("ok"):
            return
        for upd in data.get("result", []):
            s.last_update_id = upd["update_id"]  # <<< 최신 ID 갱신
            msg = upd.get("message") or {}
            chat_id = str(msg.get("chat", {}).get("id"))
            text = (msg.get("text") or "").strip()
            if TELEGRAM_CHAT_ID and chat_id != str(TELEGRAM_CHAT_ID):
                continue
            if not text:
                continue

            if text.startswith("/size"):
                tg_send(parse_size_command(text, s) or "설정 반영 완료")
            elif text.startswith("/status") or text.startswith("/state"):
                tg_send(
                    f"equity={s.equity:.2f}, day_start={s.day_start_equity:.2f}, open={'Y' if s.open_pos else 'N'}\n"
                    f"size={s.sizing_mode} {(s.sizing_value if s.sizing_mode=='fixed' else str(int(s.sizing_value*100))+'%')}\n"
                    f"hard_paused={s.hard_paused}"
                )
            elif text.startswith("/help"):
                tg_send("/size percent <x> | /size fixed <usdt> | /status | /panic | /pause | /resume")
            elif text.startswith("/panic"):
                s.hard_paused = True
                s.__dict__["panic_now"] = True
                tg_send("PANIC: 전량 청산 + 하드락")
            elif text.startswith("/pause"):
                s.hard_paused = True
                tg_send("일시정지 설정")
            elif text.startswith("/resume"):
                s.hard_paused = False
                s.__dict__.pop("panic_now", None)
                tg_send("재개")

        # <<< 여기! 전체 루프 처리 후 반드시 저장
        save_state(s)

    except Exception as e:
        logging.error(f"Telegram poll error: {e}")


# ---------------------- Indicators ----------------------
def bollinger_and_cci(df:List[Dict], period=20, nstd=2.0)->List[Dict]:
    closes=[x['close'] for x in df]; highs=[x['high'] for x in df]; lows=[x['low'] for x in df]
    ma=[]; sd=[]
    for i in range(len(closes)):
        if i+1<period: ma.append(float('nan')); sd.append(float('nan'))
        else:
            w=closes[i+1-period:i+1]; m=sum(w)/period; var=sum((v-m)**2 for v in w)/period
            ma.append(m); sd.append(var**0.5)
    bb_mid=ma; bb_up=[m+nstd*s if not math.isnan(m) else float('nan') for m,s in zip(ma,sd)]
    bb_low=[m-nstd*s if not math.isnan(m) else float('nan') for m,s in zip(ma,sd)]
    cci=[]; tps=[(h+l+c)/3.0 for h,l,c in zip(highs,lows,closes)]
    for i in range(len(tps)):
        if i+1<period: cci.append(float('nan'))
        else:
            w=tps[i+1-period:i+1]; m=sum(w)/period; mad=sum(abs(x-m) for x in w)/period
            cci.append((tps[i]-m)/(0.015*mad) if mad>0 else 0.0)
    out=[]
    for i,row in enumerate(df):
        r=dict(row); r.update({'bb_mid':bb_mid[i],'bb_upper':bb_up[i],'bb_lower':bb_low[i],'cci20':cci[i]}); out.append(r)
    return [r for r in out if not math.isnan(r['bb_mid']) and not math.isnan(r['cci20'])]

# ---------------------- CCXT Adapter ----------------------
class CcxtBitgetAdapter:
    def __init__(self):
        import ccxt
        self.ccxt=ccxt.bitget({'apiKey':BITGET_API_KEY,'secret':BITGET_API_SECRET,'password':BITGET_API_PASSWORD,'enableRateLimit':True,'options':{'defaultType':'swap','defaultSubType':'linear'}})
        self.symbol_map={'BTCUSDT':'BTC/USDT:USDT','SOLUSDT':'SOL/USDT:USDT'}
        for sym,cfg in SYMBOLS.items():
            try: self.ccxt.setLeverage(cfg.get('leverage',20), self.symbol_map[sym], params={'marginMode':'cross'})
            except Exception as e: logging.warning(f"setLeverage fail {sym}: {e}")
    def _cc(self,s): return self.symbol_map.get(s,s)
    def fetch_ohlcv(self,symbol,interval="1h",limit=300)->List[Dict]:
        try:
            ohl=self.ccxt.fetch_ohlcv(self._cc(symbol), timeframe=interval, limit=limit)
            return [{'time':datetime.fromtimestamp(t/1000,tz=timezone.utc),'open':float(o),'high':float(h),'low':float(l),'close':float(c),'volume':float(v)} for t,o,h,l,c,v in ohl]
        except Exception as e: logging.error(f"fetch_ohlcv {symbol}: {e}"); return []
    def fetch_equity(self)->float:
        try:
            bal=self.ccxt.fetch_balance(params={'type':'swap'}); usdt=bal.get('USDT') or {}
            return float(usdt.get('total') or usdt.get('free') or 0.0)
        except Exception as e: logging.error(f"fetch_equity: {e}"); return 0.0
    def _last(self,ccs):
        try: t=self.ccxt.fetch_ticker(ccs); return float(t.get('last')) if t else float('nan')
        except Exception as e: logging.error(f"ticker {ccs}: {e}"); return float('nan')
    def _amt_from_notional(self, ccs, notional, ref_px=None):
        px=ref_px if (ref_px and ref_px>0) else self._last(ccs)
        if not px or math.isnan(px) or px<=0: raise RuntimeError("가격 조회 실패")
        amt=notional/px
        try:
            m=self.ccxt.market(ccs); prec=m.get('precision',{}).get('amount');
            if prec is not None:
                step=10**(-prec); amt=math.floor(amt/step)*step
        except Exception: pass
        return max(amt,0.0)
    def place_market(self,symbol,side,notional):
        ccs=self._cc(symbol); amt=self._amt_from_notional(ccs,notional)
        order=self.ccxt.create_order(ccs,'market','buy' if side=='long' else 'sell',amt,params={'reduceOnly':False})
        price=float(order.get('average') or order.get('price') or self._last(ccs) or 0.0)
        return {'price':price,'amount':amt,'raw':order}
    def place_stop(self,symbol,side,stop_price,notional):
        """서버사이드 SL만 생성 (reduceOnly, stopPrice)
        반환: order_id 또는 None
        """
        ccs=self._cc(symbol); amt=self._amt_from_notional(ccs,notional,ref_px=stop_price)
        try:
            o=self.ccxt.create_order(ccs,'market','sell' if side=='long' else 'buy',amt,params={'reduceOnly':True,'stopPrice':stop_price})
            return o.get('id') if isinstance(o,dict) else getattr(o,'id',None)
        except Exception as e:
            logging.error(f"place_stop fail {symbol}: {e}"); return None
    def cancel_order(self, symbol, order_id):
        try: self.ccxt.cancel_order(order_id, self._cc(symbol))
        except Exception as e: logging.warning(f"cancel_order fail {symbol}:{order_id}: {e}")

# ---------------------- Signals ----------------------
def long_signal(r): return (r['close']<r['bb_mid']) and (r['close']>r['bb_lower']) and (r['cci20']>-100)

def short_signal(r):
    if r['close']>=r['bb_mid'] and r['cci20']>=100: return False
    return (r['close']<r['bb_mid']) and (r['cci20']<100)

# ---------------------- Sizing ----------------------
def compute_margin(s:State)->float:
    return max(0.0, s.equity*(s.sizing_value) if s.sizing_mode=='percent' else float(s.sizing_value))

# ---------------------- Main ----------------------
def run_once(ex:CcxtBitgetAdapter, s:State):
    # sync equity
    try:
        real=ex.fetch_equity()
        if real>0: s.equity=real
    except Exception as e: logging.warning(f"equity sync fail: {e}")
    # roll day
    today=now_utc().date().isoformat()
    if s.current_day!=today: s.current_day=today; s.day_start_equity=s.equity
    # daily hard lock
    if (s.day_start_equity - s.equity)/max(1e-9,s.day_start_equity) >= DAILY_STOP:
        s.hard_paused=True; save_state(s); tg_send(f"Daily hard lock {-int(DAILY_STOP*100)}% reached"); return
    # panic
    if getattr(s,'panic_now',False) and s.open_pos:
        try:
            pos=s.open_pos; ex.place_market(pos.symbol, 'short' if pos.side=='long' else 'long', pos.notional*pos.remaining_frac)
        except Exception as e: logging.error(f"panic close fail: {e}")
        s.open_pos=None; s.hard_paused=True; s.__dict__.pop('panic_now',None); save_state(s); tg_send("Panic closed & locked"); return
    # manage (TP/SL 로직)
    if s.open_pos:
        # 서버사이드 SL이 있으므로 여기선 TP1/TP2 로직만 수행(봉 확정 후 판단)
        # OHLCV 1개만 가져와 최근봉로직 적용
        ind = None
        try:
            ohlcv = ex.fetch_ohlcv(s.open_pos.symbol, INTERVAL, limit=1)
            ind = bollinger_and_cci(ohlcv, BB_PERIOD, BB_NSTD) if ohlcv else None
        except Exception as e:
            logging.warning(f"manage fetch ohlcv fail: {e}")
        if ind:
            r = ind[-1]
            pos = s.open_pos
            entry = pos.entry
            side = pos.side
            tp1 = entry*(1+TP1_PCT if side=='long' else 1-TP1_PCT)
            tp2 = entry*(1+TP2_PCT if side=='long' else 1-TP2_PCT)
            # TP1
            if (r['high']>=tp1 and side=='long') or (r['low']<=tp1 and side=='short'):
                # half close at market
                try:
                    ex.place_market(pos.symbol, 'short' if side=='long' else 'long', pos.notional*0.5)
                except Exception as e:
                    logging.error(f"TP1 close fail: {e}")
                pos.remaining_frac = 0.5
                pos.tp1_hit = True
                # MOVE SL -> BE + fees (≈ entry ± 2*fee)
                be_adj = 2.0 * FEE_RATE
                new_stop = entry*(1+be_adj) if side=='long' else entry*(1-be_adj)
                # cancel old SL & place new one
                if pos.sl_order_id:
                    ex.cancel_order(pos.symbol, pos.sl_order_id)
                pos.sl_order_id = ex.place_stop(pos.symbol, side, new_stop, pos.notional*pos.remaining_frac)
                pos.stop = new_stop
                save_state(s)
                tg_send(f"TP1 {pos.symbol} {side}. SL->BE(+fees) at {new_stop:.2f}")
                return
            # TP2 (full close)
            if (r['high']>=tp2 and side=='long') or (r['low']<=tp2 and side=='short'):
                try:
                    ex.place_market(pos.symbol, 'short' if side=='long' else 'long', pos.notional*pos.remaining_frac)
                except Exception as e:
                    logging.error(f"TP2 close fail: {e}")
                # cancel SL
                if pos.sl_order_id: ex.cancel_order(pos.symbol, pos.sl_order_id)
                s.open_pos=None; save_state(s); tg_send(f"TP2 {pos.symbol} {side} closed")
                return
        save_state(s); return
    # blocked?
    if s.hard_paused: return
    # entries
    for sym,cfg in SYMBOLS.items():
        ohlcv=ex.fetch_ohlcv(sym, INTERVAL, limit=300)
        if not ohlcv or len(ohlcv)<50: continue
        ind=bollinger_and_cci(ohlcv, BB_PERIOD, BB_NSTD); r=ind[-1]
        if (r['close']>=r['bb_mid'] and r['cci20']>=100): continue
        go_long=long_signal(r); go_short=(not go_long) and short_signal(r)
        if not(go_long or go_short): continue
        margin = s.equity*s.sizing_value if s.sizing_mode=='percent' else s.sizing_value
        if margin<=0: tg_send("포지션 크기 0. /size 로 설정"); break
        notional = margin * cfg['leverage']
        # entry
        try:
            res=ex.place_market(sym, 'long' if go_long else 'short', notional)
            entry_price=res['price']
            # place SL only
            stop_price = entry_price*(1-SL_PCT) if go_long else entry_price*(1+SL_PCT)
            sl_id = ex.place_stop(sym, 'long' if go_long else 'short', stop_price, notional)
            s.open_pos = Position(symbol=sym, side='long' if go_long else 'short', entry=entry_price,
                                  notional=notional, remaining_frac=1.0, tp1_hit=False, stop=stop_price, opened_at=now_utc().isoformat(), sl_order_id=sl_id)
            save_state(s)
            tg_send(f"OPEN {sym} {'long' if go_long else 'short'} @ {entry_price:.2f} | SL {stop_price:.2f}")
            break
        except Exception as e:
            tg_send(f"Entry failed {sym}: {e}")

if __name__ == "__main__":
    ex = CcxtBitgetAdapter()
    last_traded_hour = None

    from datetime import datetime, timezone, timedelta
    POLL_EVERY_SEC = 3
    BUFFER_MIN = 2  # 정각 +2분 실행

    while True:
        try:
            # 1) 텔레그램 커맨드 상시 처리
            st = load_state()
            tg_poll_and_handle(st)

            # 2) 매 시 정각+버퍼에 run_once 실행 (시간당 1회만)
            now = datetime.now(timezone.utc)
            target = now.replace(minute=BUFFER_MIN, second=0, microsecond=0)
            if now.minute == BUFFER_MIN and now.second < POLL_EVERY_SEC:
                if last_traded_hour != now.hour:
                    run_once(ex, st)
                    last_traded_hour = now.hour

            time.sleep(POLL_EVERY_SEC)
        except Exception as e:
            print(f"[CRITICAL] main loop error: {e}")
            time.sleep(5)
