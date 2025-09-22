#!/usr/bin/env python3
"""
Bitget 자동매매 봇 (BTC+SOL 전용, 텔레그램 제거 버전)
- 전략: 1h Bollinger(20,2) + CCI(20)
- 진입: (Long) BB_lower<Close<BB_mid & CCI>-100 / (Short) Close<BB_mid & CCI<+100 (단 Close≥BB_mid & CCI≥100 숏 금지)
- 청산: SL=2% (서버사이드-필수), TP1=+4%(0.5 청산 후 SL→BE(+수수료)), TP2=+6%(전량)
- 리스크(1순위): 서버사이드 SL, 일일 하드락 -4%(자동 중지), 실계좌 동기화
- 실행: 상시 루프, 매 시 정각+2분(UTC)만 트레이딩 체크
"""
import os, json, time, math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Optional, List

# ===================== 환경설정 =====================
BITGET_API_KEY      = os.getenv("BITGET_API_KEY", "")
BITGET_API_SECRET   = os.getenv("BITGET_API_SECRET", "")
BITGET_API_PASSWORD = os.getenv("BITGET_API_PASSWORD", "")
FEE_RATE = float(os.getenv("FEE_RATE", "0.0008"))  # per side (0.08%)

SYMBOLS = {"BTCUSDT": {"leverage": 100}, "SOLUSDT": {"leverage": 60}}
INTERVAL = "1h"; BB_PERIOD=20; BB_NSTD=2.0; CCI_PERIOD=20
SL_PCT=0.02; TP1_PCT=0.04; TP2_PCT=0.06
DAILY_STOP=0.04  # -4%

STATE_FILE="state.json"

# ===================== 유틸/로깅 =====================
def now_utc():
    return datetime.now(timezone.utc)

def log(msg: str, critical: bool=False):
    ts = now_utc().strftime('%Y-%m-%d %H:%M:%S')
    print(("[CRITICAL] " if critical else "[INFO] ") + ts + " " + msg)

# ===================== 데이터 모델 =====================
@dataclass
class Position:
    symbol: str
    side: str  # 'long' | 'short'
    entry: float
    notional: float
    remaining_frac: float = 1.0
    tp1_hit: bool = False
    stop: float = 0.0
    opened_at: str = ""
    sl_order_id: Optional[str] = None

@dataclass
class State:
    equity: float = 0.0
    current_day: str = ""
    day_start_equity: float = 0.0
    open_pos: Optional[Position] = None
    sizing_mode: str = "percent"
    sizing_value: float = 0.10  # 10%
    hard_paused: bool = False   # 일일 하드락 상태

# ===================== 상태 IO =====================
def load_state() -> State:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            d = json.load(f)
        pos = d.get("open_pos")
        position = Position(**pos) if pos else None
        return State(
            equity=d.get("equity", 0.0),
            current_day=d.get("current_day", ""),
            day_start_equity=d.get("day_start_equity", 0.0),
            open_pos=position,
            sizing_mode=d.get("sizing_mode", "percent"),
            sizing_value=d.get("sizing_value", 0.10),
            hard_paused=d.get("hard_paused", False),
        )
    s = State()
    save_state(s)
    return s

def save_state(s: State):
    d = asdict(s)
    if s.open_pos:
        d["open_pos"] = asdict(s.open_pos)
    with open(STATE_FILE, "w") as f:
        json.dump(d, f, indent=2)

# ===================== 인디케이터 =====================
def bollinger_and_cci(df: List[Dict], period=20, nstd=2.0) -> List[Dict]:
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

# ===================== CCXT 어댑터 =====================
class CcxtBitgetAdapter:
    def __init__(self):
        import ccxt
        self.ccxt=ccxt.bitget({'apiKey':BITGET_API_KEY,'secret':BITGET_API_SECRET,'password':BITGET_API_PASSWORD,'enableRateLimit':True,'options':{'defaultType':'swap','defaultSubType':'linear'}})
        self.symbol_map={'BTCUSDT':'BTC/USDT:USDT','SOLUSDT':'SOL/USDT:USDT'}
        for sym,cfg in SYMBOLS.items():
            try: self.ccxt.setLeverage(cfg.get('leverage',20), self.symbol_map[sym], params={'marginMode':'cross'})
            except Exception as e: log(f"setLeverage fail {sym}: {e}")
    def _cc(self,s): return self.symbol_map.get(s,s)
    def fetch_ohlcv(self,symbol,interval="1h",limit=300)->List[Dict]:
        try:
            ohl=self.ccxt.fetch_ohlcv(self._cc(symbol), timeframe=interval, limit=limit)
            return [{'time':datetime.fromtimestamp(t/1000,tz=timezone.utc),'open':float(o),'high':float(h),'low':float(l),'close':float(c),'volume':float(v)} for t,o,h,l,c,v in ohl]
        except Exception as e:
            log(f"fetch_ohlcv {symbol}: {e}")
            return []
    def fetch_equity(self)->float:
        try:
            bal=self.ccxt.fetch_balance(params={'type':'swap'})
            usdt=bal.get('USDT') or {}
            return float(usdt.get('total') or usdt.get('free') or 0.0)
        except Exception as e:
            log(f"fetch_equity: {e}")
            return 0.0
    def _last(self,ccs):
        try:
            t=self.ccxt.fetch_ticker(ccs); return float(t.get('last')) if t else float('nan')
        except Exception as e:
            log(f"ticker {ccs}: {e}")
            return float('nan')
    def _amt_from_notional(self, ccs, notional, ref_px=None):
        px=ref_px if (ref_px and ref_px>0) else self._last(ccs)
        if not px or math.isnan(px) or px<=0: raise RuntimeError("가격 조회 실패")
        amt=notional/px
        try:
            m=self.ccxt.market(ccs); prec=m.get('precision',{}).get('amount')
            if prec is not None:
                step=10**(-prec); amt=math.floor(amt/step)*step
        except Exception:
            pass
        return max(amt,0.0)
    def place_market(self,symbol,side,notional):
        ccs=self._cc(symbol); amt=self._amt_from_notional(ccs,notional)
        order=self.ccxt.create_order(ccs,'market','buy' if side=='long' else 'sell',amt,params={'reduceOnly':False})
        price=float(order.get('average') or order.get('price') or self._last(ccs) or 0.0)
        return {'price':price,'amount':amt,'raw':order}
    def place_stop(self,symbol,side,stop_price,notional):
        """서버사이드 SL만 생성 (reduceOnly). Bitget 호환 파라미터를 순차 시도"""
        ccs=self._cc(symbol); amt=self._amt_from_notional(ccs,notional,ref_px=stop_price)
        params_list=[
            {'reduceOnly':True,'stopPrice':stop_price},
            {'reduceOnly':True,'triggerPrice':stop_price},
            {'reduceOnly':True,'stopLossPrice':stop_price},
        ]
        side_o = 'sell' if side=='long' else 'buy'
        last_err=None
        for p in params_list:
            try:
                o=self.ccxt.create_order(ccs,'market', side_o, amt, params=p)
                return o.get('id') if isinstance(o,dict) else getattr(o,'id',None)
            except Exception as e:
                last_err=e
                continue
        log(f"place_stop fail {symbol}: {last_err}")
        return None
    def cancel_order(self, symbol, order_id):
        try:
            self.ccxt.cancel_order(order_id, self._cc(symbol))
        except Exception as e:
            log(f"cancel_order fail {symbol}:{order_id}: {e}")

# ===================== 시그널 =====================
def long_signal(r):
    return (r['close']<r['bb_mid']) and (r['close']>r['bb_lower']) and (r['cci20']>-100)

def short_signal(r):
    if r['close']>=r['bb_mid'] and r['cci20']>=100: return False
    return (r['close']<r['bb_mid']) and (r['cci20']<100)

# ===================== 사이징 =====================
def compute_margin(s: State) -> float:
    return max(0.0, s.equity * s.sizing_value if s.sizing_mode=='percent' else float(s.sizing_value))

# ===================== 코어 루틴 =====================
def run_once(ex: CcxtBitgetAdapter, s: State):
    # 1) 실계좌 동기화 (0이라도 반영)
    s.equity = ex.fetch_equity()

    # 2) 일자 롤링/초기화
    if not s.current_day:
        s.current_day = now_utc().date().isoformat()
    if s.day_start_equity == 0.0:
        s.day_start_equity = s.equity
    today = now_utc().date().isoformat()
    if s.current_day != today:
        s.current_day = today
        s.day_start_equity = s.equity

    # 3) 일일 하드락
    if s.day_start_equity > 0 and (s.day_start_equity - s.equity) / s.day_start_equity >= DAILY_STOP:
        s.hard_paused = True
        save_state(s)
        log(f"Daily hard lock {int(DAILY_STOP*100)}% reached")
        return

    # 4) 포지션 관리 (TP1/TP2)
    if s.open_pos:
        ohlcv = ex.fetch_ohlcv(s.open_pos.symbol, INTERVAL, limit=1)
        if ohlcv:
            r = ohlcv[-1]
            pos = s.open_pos
            entry = pos.entry; side = pos.side
            tp1 = entry*(1+TP1_PCT if side=='long' else 1-TP1_PCT)
            tp2 = entry*(1+TP2_PCT if side=='long' else 1-TP2_PCT)
            # TP1
            if ((r['high']>=tp1) if side=='long' else (r['low']<=tp1)) and not pos.tp1_hit:
                ex.place_market(pos.symbol, 'short' if side=='long' else 'long', pos.notional*0.5)
                pos.remaining_frac = 0.5; pos.tp1_hit = True
                be_adj = 2.0 * FEE_RATE
                new_stop = entry*(1+be_adj) if side=='long' else entry*(1-be_adj)
                if pos.sl_order_id: ex.cancel_order(pos.symbol, pos.sl_order_id)
                pos.sl_order_id = ex.place_stop(pos.symbol, side, new_stop, pos.notional*pos.remaining_frac)
                pos.stop = new_stop
                save_state(s)
                log(f"TP1 {pos.symbol} {side}. SL->BE(+fees) {new_stop:.2f}")
                return
            # TP2
            if (r['high']>=tp2 if side=='long' else r['low']<=tp2):
                ex.place_market(pos.symbol, 'short' if side=='long' else 'long', pos.notional*pos.remaining_frac)
                if pos.sl_order_id: ex.cancel_order(pos.symbol, pos.sl_order_id)
                s.open_pos = None
                save_state(s)
                log(f"TP2 {pos.symbol} {side} closed")
                return
        save_state(s)
        return

    # 5) 신규 진입 차단
    if s.hard_paused:
        return

    # 6) 진입 (BTC -> SOL)
    for sym, cfg in SYMBOLS.items():
        ohlcv = ex.fetch_ohlcv(sym, INTERVAL, limit=300)
        if not ohlcv or len(ohlcv) < 50:
            continue
        ind = bollinger_and_cci(ohlcv, BB_PERIOD, BB_NSTD)
        r = ind[-1]
        if (r['close']>=r['bb_mid'] and r['cci20']>=100):
            continue
        go_long = long_signal(r)
        go_short = (not go_long) and short_signal(r)
        if not (go_long or go_short):
            continue
        margin = compute_margin(s)
        if margin <= 0:
            log("포지션 크기 0. sizing_value를 확인하세요")
            break
        notional = margin * cfg['leverage']
        res = ex.place_market(sym, 'long' if go_long else 'short', notional)
        entry_price = res['price']
        stop_price = entry_price*(1-SL_PCT) if go_long else entry_price*(1+SL_PCT)
        sl_id = ex.place_stop(sym, 'long' if go_long else 'short', stop_price, notional)
        s.open_pos = Position(symbol=sym, side='long' if go_long else 'short', entry=entry_price,
                              notional=notional, remaining_frac=1.0, tp1_hit=False, stop=stop_price,
                              opened_at=now_utc().isoformat(), sl_order_id=sl_id)
        save_state(s)
        log(f"OPEN {sym} {'long' if go_long else 'short'} @ {entry_price:.2f} | SL {stop_price:.2f}")
        break

# ===================== 메인 루프 =====================
if __name__ == "__main__":
    # 어댑터 초기화
    ex = CcxtBitgetAdapter()
    tg_send(f"{ex.fetch_equity()}")
    # 매시 정각+2분마다 1회 실행
    BUFFER_MIN = 21
    last_hour = None
    while True:
        try:
            st = load_state()
            now = now_utc()
            if now.minute == BUFFER_MIN:
                if last_hour != now.hour:
                    run_once(ex, st)
                    last_hour = now.hour
            time.sleep(2)
        except Exception as e:
            log(f"main loop error: {e}", critical=True)
            time.sleep(5)
