#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (File header omitted for brevity; see previous cell's docstring.)

import os, asyncio, json, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import aiohttp, websockets, numpy as np

REST_BASE = "https://fapi.binance.com"
WS_BASE = "wss://fstream.binance.com/stream"
KLINES_ENDPOINT = "/fapi/v1/klines"

SYMBOLS = ["btcusdt", "ethusdt", "xrpusdt", "solusdt"]
TF_PRIORITY = ["5m", "15m", "1h"]

@dataclass
class Config:
    seed_candles:int=900; ws_reconnect_delay:int=5
    pivot_span:int=3; max_pivots:int=30; min_pivots_required:int=4; lookback_bars:int=220
    patterns_tri:Tuple[str,...]=("symmetrical","ascending","descending","rising_wedge","falling_wedge")
    slope_flat_threshold:float=1e-6; allow_broadening:bool=False
    box_window:int=80; box_flat_tol_pct:float=0.004; box_break_buffer_pct:float=0.001
    dbl_pivot_min_gap:int=5; dbl_price_tolerance:float=0.004; dbl_confirm_buffer_pct:float=0.001
    hs_min_gap:int=4; hs_price_tolerance:float=0.01; hs_confirm_buffer_pct:float=0.001
    min_contraction_ratio:float=0.35; atr_window:int=14; atr_contraction_factor:float=0.7
    breakout_buffer_pct:float=0.001; vol_ma_window:int=20; vol_confirm_factor:float=1.5
    confirm_bars:int=1; min_body_atr:float=0.25; min_body_to_range:float=0.40
    retest_max_bars:int=10; retest_tolerance_pct:float=0.002; retest_confirm_close:bool=True
    stop_buffer_pct:float=0.002; tp_rr_list:Tuple[float,...]=(1.0,2.0)
    loss_pct_at_sl:float=0.10; leverage_cap:float=50.0
    atr_guard_ratio:float=0.7; ema_slope_eps:float=0.0001
    telegram_bot_token:str=os.getenv("TELEGRAM_BOT_TOKEN","").strip()
    telegram_chat_id:str=os.getenv("TELEGRAM_CHAT_ID","").strip()
    telegram_parse_mode:str="Markdown"
cfg=Config()

@dataclass
class Candle: open_time:int; open:float; high:float; low:float; close:float; volume:float; close_time:int
@dataclass
class TriangleState: up_m:float; up_b:float; lo_m:float; lo_b:float; start_idx:int; end_idx:int; pattern:str
@dataclass
class PendingRetest: direction:str; breakout_idx:int; up_m:float; up_b:float; lo_m:float; lo_b:float; label:str
@dataclass
class PendingConfirm: direction:str; confirms_needed:int; confirms_got:int; up_m:float; up_b:float; lo_m:float; lo_b:float; label:str
@dataclass
class TFState:
    candles:List[Candle]=field(default_factory=list)
    vol_ma:List[float]=field(default_factory=list)
    atr:List[float]=field(default_factory=list)
    last_alert_time:float=0.0
    pending_confirm:Optional[PendingConfirm]=None
    pending_retest:Optional[PendingRetest]=None
@dataclass
class SymbolState: tfs:Dict[str,TFState]=field(default_factory=lambda:{tf:TFState() for tf in TF_PRIORITY})

def ma(values:List[float], period:int)->float:
    if len(values)<period or period<=0: return float('nan')
    return float(sum(values[-period:])/period)
def ema(series:List[float], period:int)->float:
    if len(series)<period: return float('nan')
    alpha=2.0/(period+1.0); e=series[-period]
    for v in series[-period+1:]: e=alpha*v+(1-alpha)*e
    return float(e)
def atr_calc(candles:List[Candle], n:int)->float:
    if len(candles)<n+1: return float('nan')
    trs=[]; 
    for i in range(1,n+1):
        hi=candles[-i].high; lo=candles[-i].low; prev_close=candles[-i-1].close
        trs.append(max(hi-lo,abs(hi-prev_close),abs(lo-prev_close)))
    return float(sum(trs)/n)
def pivot_high(candles:List[Candle], i:int, span:int)->bool:
    if i-span<0 or i+span>=len(candles): return False
    h=candles[i].high
    for j in range(i-span,i+span+1):
        if candles[j].high>h: return False
    return True
def pivot_low(candles:List[Candle], i:int, span:int)->bool:
    if i-span<0 or i+span>=len(candles): return False
    l=candles[i].low
    for j in range(i-span,i+span+1):
        if candles[j].low<l: return False
    return True
def linreg(xs:List[int], ys:List[float]):
    if len(xs)<2: return 0.0, ys[-1] if ys else 0.0
    m,b=np.polyfit(xs,ys,1); return float(m),float(b)
def slope_is_flat(m:float)->bool: return abs(m)<=cfg.slope_flat_threshold
def classify_triangle(m_up:float,m_lo:float)->Optional[str]:
    up,lo=m_up,m_lo
    if "symmetrical" in cfg.patterns_tri and (up<0 and lo>0): return "triangle_sym"
    if "ascending" in cfg.patterns_tri and slope_is_flat(up) and lo>0: return "triangle_asc"
    if "descending" in cfg.patterns_tri and up<0 and slope_is_flat(lo): return "triangle_desc"
    if "rising_wedge" in cfg.patterns_tri and (up>0 and lo>0 and up<lo): return "wedge_rising"
    if "falling_wedge" in cfg.patterns_tri and (up<0 and lo<0 and up>lo): return "wedge_falling"
    return None
def compute_triangle(candles:List[Candle])->Optional[TriangleState]:
    i_end=len(candles)-1; i_start=max(0,i_end-cfg.lookback_bars)
    hx,hy,lx,ly=[],[],[],[]
    for i in range(i_start,i_end+1):
        if pivot_high(candles,i,cfg.pivot_span): hx.append(i); hy.append(candles[i].high)
        if pivot_low(candles,i,cfg.pivot_span):  lx.append(i); ly.append(candles[i].low)
    hx,hy=hx[-cfg.max_pivots:],hy[-cfg.max_pivots:]; lx,ly=lx[-cfg.max_pivots:],ly[-cfg.max_pivots:]
    if len(hx)<cfg.min_pivots_required or len(lx)<cfg.min_pivots_required: return None
    up_m,up_b=linreg(hx,hy); lo_m,lo_b=linreg(lx,ly)
    pat=classify_triangle(up_m,lo_m)
    if pat is None: return None
    gap_start=(up_m*i_start+up_b)-(lo_m*i_start+lo_b); gap_end=(up_m*i_end+up_b)-(lo_m*i_end+lo_b)
    if not cfg.allow_broadening:
        if gap_start<=0 or gap_end<=0: return None
        if gap_end>=gap_start: return None
    else:
        if gap_start<=0 or gap_end<=0: return None
    return TriangleState(up_m,up_b,lo_m,lo_b,i_start,i_end,pat)
def tri_contraction_ok(candles:List[Candle], tri:TriangleState, atr_recent:float, atr_vals:List[float])->bool:
    if tri is None: return False
    window=candles[tri.start_idx:tri.end_idx+1]
    if len(window)<cfg.atr_window+10: return False
    cut=len(window)//2; early=window[:cut]; late=window[cut:]
    er=max(c.high for c in early)-min(c.low for c in early); lr=max(c.high for c in late)-min(c.low for c in late)
    if er<=0: return False
    if (lr/er)>cfg.min_contraction_ratio: return False
    if len(atr_vals)<cfg.atr_window: return False
    atr_long=float(np.nanmean(atr_vals[max(0,tri.start_idx-cfg.atr_window):tri.end_idx+1]))
    if not (atr_recent>0 and atr_long>0): return False
    if (atr_recent/atr_long)>cfg.atr_contraction_factor: return False
    return True
def detect_box(candles:List[Candle])->Optional[Tuple[float,float]]:
    if len(candles)<cfg.box_window+5: return None
    window=candles[-cfg.box_window-1:-1]; highs=[c.high for c in window]; lows=[c.low for c in window]
    rh=max(highs); rl=min(lows); tol_h=rh*cfg.box_flat_tol_pct; tol_l=rl*cfg.box_flat_tol_pct
    near_top=sum(1 for x in highs if rh-x<=tol_h); near_bot=sum(1 for x in lows if x-rl<=tol_l)
    if near_top>=max(3,cfg.box_window//8) and near_bot>=max(3,cfg.box_window//8): return rh,rl
    return None
def find_recent_pivots(candles:List[Candle], span:int, limit:int=60):
    idx_end=len(candles)-1; idx_start=max(0,idx_end-limit)
    ph=[i for i in range(idx_start,idx_end+1) if pivot_high(candles,i,span)]
    pl=[i for i in range(idx_start,idx_end+1) if pivot_low(candles,i,span)]
    return ph,pl
def detect_double_top(candles:List[Candle])->Optional[Dict]:
    ph,pl=find_recent_pivots(candles,cfg.pivot_span,limit=80)
    if len(ph)<2 or len(pl)<1: return None
    h1,h2=ph[-2],ph[-1]
    if h2-h1<cfg.dbl_pivot_min_gap: return None
    p1=candles[h1].high; p2=candles[h2].high
    if abs(p1-p2)/((p1+p2)/2)>cfg.dbl_price_tolerance: return None
    mid_lows=[candles[i].low for i in range(h1,h2+1)]
    if not mid_lows: return None
    neckline=min(mid_lows); last=candles[-1]
    if last.close<neckline*(1-cfg.dbl_confirm_buffer_pct): return {"label":"double_top","neckline":neckline,"dir":"DOWN"}
    return None
def detect_double_bottom(candles:List[Candle])->Optional[Dict]:
    ph,pl=find_recent_pivots(candles,cfg.pivot_span,limit=80)
    if len(pl)<2 or len(ph)<1: return None
    l1,l2=pl[-2],pl[-1]
    if l2-l1<cfg.dbl_pivot_min_gap: return None
    p1=candles[l1].low; p2=candles[l2].low
    if abs(p1-p2)/((p1+p2)/2)>cfg.dbl_price_tolerance: return None
    mid_highs=[candles[i].high for i in range(l1,l2+1)]
    if not mid_highs: return None
    neckline=max(mid_highs); last=candles[-1]
    if last.close>neckline*(1+cfg.dbl_confirm_buffer_pct): return {"label":"double_bottom","neckline":neckline,"dir":"UP"}
    return None
def detect_head_shoulders(candles:List[Candle])->Optional[Dict]:
    ph,pl=find_recent_pivots(candles,cfg.pivot_span,limit=120)
    if len(ph)<3 or len(pl)<2: return None
    hL,hH,hR=ph[-3],ph[-2],ph[-1]
    if not (hH-hL>=cfg.hs_min_gap and hR-hH>=cfg.hs_min_gap): return None
    Lh=candles[hL].high; Hh=candles[hH].high; Rh=candles[hR].high
    if not (Hh>Lh and Hh>Rh): return None
    if abs(Lh-Rh)/((Lh+Rh)/2)>cfg.hs_price_tolerance: return None
    lows_between=[candles[i].low for i in range(hL,hR+1)]
    if not lows_between: return None
    neckline=min(lows_between); last=candles[-1]
    if last.close<neckline*(1-cfg.hs_confirm_buffer_pct): return {"label":"head_shoulders","neckline":neckline,"dir":"DOWN"}
    return None
def detect_inverse_head_shoulders(candles:List[Candle])->Optional[Dict]:
    ph,pl=find_recent_pivots(candles,cfg.pivot_span,limit=120)
    if len(pl)<3 or len(ph)<2: return None
    lL,lH,lR=pl[-3],pl[-2],pl[-1]
    if not (lH-lL>=cfg.hs_min_gap and lR-lH>=cfg.hs_min_gap): return None
    Ll=candles[lL].low; Hl=candles[lH].low; Rl=candles[lR].low
    if not (Hl<Ll and Hl<Rl): return None
    if abs(Ll-Rl)/((Ll+Rl)/2)>cfg.hs_price_tolerance: return None
    highs_between=[candles[i].high for i in range(lL,lR+1)]
    if not highs_between: return None
    neckline=max(highs_between); last=candles[-1]
    if last.close>neckline*(1+cfg.hs_confirm_buffer_pct): return {"label":"inverse_head_shoulders","neckline":neckline,"dir":"UP"}
    return None
def breakout_seed_triangle(candles,tri,vol_ma_value,allowed_dir):
    i=len(candles)-1; last=candles[i]; up=tri.up_m*i+tri.up_b; lo=tri.lo_m*i+tri.lo_b
    if np.isnan(vol_ma_value): return None
    if last.close>up*(1+cfg.breakout_buffer_pct) and last.volume>=vol_ma_value*cfg.vol_confirm_factor:
        sig={"direction":"UP","idx":i,"open":last.open,"high":last.high,"low":last.low,"close":last.close,"line":up,"label":tri.pattern}
        if allowed_dir and sig["direction"]!=allowed_dir: return None
        return sig
    if last.close<lo*(1-cfg.breakout_buffer_pct) and last.volume>=vol_ma_value*cfg.vol_confirm_factor:
        sig={"direction":"DOWN","idx":i,"open":last.open,"high":last.high,"low":last.low,"close":last.close,"line":lo,"label":tri.pattern}
        if allowed_dir and sig["direction"]!=allowed_dir: return None
        return sig
    return None
def breakout_seed_box(candles,rng,vol_ma_value,allowed_dir):
    i=len(candles)-1; last=candles[i]; rh,rl=rng
    if np.isnan(vol_ma_value): return None
    if last.close>rh*(1+cfg.box_break_buffer_pct) and last.volume>=vol_ma_value*cfg.vol_confirm_factor:
        sig={"direction":"UP","idx":i,"open":last.open,"high":last.high,"low":last.low,"close":last.close,"line":rh,"label":"box_breakout_up"}
        if allowed_dir and sig["direction"]!=allowed_dir: return None
        return sig
    if last.close<rl*(1-cfg.box_break_buffer_pct) and last.volume>=vol_ma_value*cfg.vol_confirm_factor:
        sig={"direction":"DOWN","idx":i,"open":last.open,"high":last.high,"low":last.low,"close":last.close,"line":rl,"label":"box_breakout_down"}
        if allowed_dir and sig["direction"]!=allowed_dir: return None
        return sig
    return None
def body_filters(sig, atr_val):
    body=abs(sig["close"]-sig["open"]); rng=max(1e-12,sig["high"]-sig["low"])
    if atr_val<=0: return False
    if body<cfg.min_body_atr*atr_val: return False
    if (body/rng)<cfg.min_body_to_range: return False
    return True
def bar_confirms(direction, close_price, line_value):
    return close_price>line_value*(1+cfg.breakout_buffer_pct) if direction=="UP" else close_price<line_value*(1-cfg.breakout_buffer_pct)
def within_tolerance(price,line,tol_pct): return abs(price-line)<=line*tol_pct
def compute_sl_tp(direction,entry,retest_extreme,line_at_retest):
    if direction=="UP":
        sl=min(retest_extreme,line_at_retest)*(1-cfg.stop_buffer_pct); risk=max(1e-9,entry-sl)
        return {"SL":sl,"TP_1.0R":entry+1.0*risk,"TP_2.0R":entry+2.0*risk}
    else:
        sl=max(retest_extreme,line_at_retest)*(1+cfg.stop_buffer_pct); risk=max(1e-9,sl-entry)
        return {"SL":sl,"TP_1.0R":entry-1.0*risk,"TP_2.0R":entry-2.0*risk}
def leverage_for_loss(entry,sl):
    if entry<=0 or sl<=0: return 1.0
    dist_pct=abs(entry-sl)/entry
    if dist_pct<=0: return 1.0
    return max(1.0,min(cfg.leverage_cap,cfg.loss_pct_at_sl/dist_pct))
async def fetch_seed_klines(session,symbol,interval,limit):
    params={"symbol":symbol.upper(),"interval":interval,"limit":limit}; url=REST_BASE+KLINES_ENDPOINT
    async with session.get(url,params=params,timeout=aiohttp.ClientTimeout(total=15)) as resp:
        resp.raise_for_status(); data=await resp.json()
    out=[]
    for k in data:
        out.append(Candle(open_time=int(k[0]),open=float(k[1]),high=float(k[2]),low=float(k[3]),close=float(k[4]),volume=float(k[5]),close_time=int(k[6])))
    return out
def parse_ws_kline(msg):
    try:
        stream=msg.get("stream",""); sym,rest=stream.split("@",1); tf=rest.split("_")[1]; k=msg["data"]["k"]
        if not k["x"]: return None
        candle=Candle(open_time=int(k["t"]),open=float(k["o"]),high=float(k["h"]),low=float(k["l"]),close=float(k["c"]),volume=float(k["q"]),close_time=int(k["T"]))
        return sym,tf,candle
    except Exception:
        return None
async def telegram(text:str):
    token=cfg.telegram_bot_token; chat_id=cfg.telegram_chat_id
    if not token or not chat_id:
        print("[WARN] Telegram ENV not set. Skipping Telegram send."); return
    url=f"https://api.telegram.org/bot{token}/sendMessage"
    payload={"chat_id":chat_id,"text":text,"parse_mode":cfg.telegram_parse_mode,"disable_web_page_preview":True}
    timeout=aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url,json=payload) as resp:
                if resp.status!=200:
                    body=await resp.text(); print(f"[Telegram] HTTP {resp.status}: {body}")
        except Exception as e:
            print(f"[Telegram] Error: {e}")
async def seed_all(states:Dict[str,SymbolState]):
    async with aiohttp.ClientSession() as session:
        for sym in SYMBOLS:
            for tf in TF_PRIORITY:
                kl=await fetch_seed_klines(session,sym,tf,cfg.seed_candles)
                st=states[sym].tfs[tf]; st.candles=kl
                vols=[c.volume for c in st.candles]; st.vol_ma=[float('nan')]*len(vols)
                for i in range(len(vols)):
                    if i+1>=cfg.vol_ma_window: st.vol_ma[i]=float(sum(vols[i+1-cfg.vol_ma_window:i+1])/cfg.vol_ma_window)
                st.atr=[float('nan')]*len(st.candles)
                for i in range(len(st.candles)):
                    if i+1>=cfg.atr_window+1:
                        trs=[]
                        for j in range(i-cfg.atr_window+1,i+1):
                            hi=st.candles[j].high; lo=st.candles[j].low
                            prev_close=st.candles[j-1].close if j-1>=0 else st.candles[j].open
                            trs.append(max(hi-lo,abs(hi-prev_close),abs(lo-prev_close)))
                        st.atr[i]=float(sum(trs)/cfg.atr_window)
            await asyncio.sleep(0.05)
async def multiplex_stream(symbols:List[str],tfs:List[str]):
    streams="/".join([f"{s}@kline_{tf}" for s in symbols for tf in tfs]); url=f"{WS_BASE}?streams={streams}"
    while True:
        try:
            async with websockets.connect(url,ping_interval=20,ping_timeout=20,close_timeout=5) as ws:
                async for message in ws: yield json.loads(message)
        except Exception:
            await asyncio.sleep(cfg.ws_reconnect_delay); continue
def htf_trend(states:Dict[str,SymbolState],sym:str)->str:
    st=states[sym].tfs["1h"]; closes=[c.close for c in st.candles]
    if len(closes)<205: return "NONE"
    e200_now=ema(closes,200); e200_prev=ema(closes[:-1],200); price_now=closes[-1]
    slope_up=e200_now>e200_prev*(1+cfg.ema_slope_eps); slope_dn=e200_now<e200_prev*(1-cfg.ema_slope_eps)
    if price_now>e200_now and slope_up: return "UP"
    if price_now<e200_now and slope_dn: return "DOWN"
    return "NONE"
def mtf_confluence(states:Dict[str,SymbolState],sym:str,want_dir:str)->bool:
    st15=states[sym].tfs["15m"]; closes=[c.close for c in st15.candles]
    if len(closes)<205: return False
    e50=ema(closes,50); e200=ema(closes,200)
    if want_dir=="UP": return e50>e200
    if want_dir=="DOWN": return e50<e200
    return False
def atr_guard_ok(states:Dict[str,SymbolState],sym:str)->bool:
    st5=states[sym].tfs["5m"]
    if len(st5.atr)<210: return False
    atr_recent=st5.atr[-1]; atr_slice=[x for x in st5.atr[-200:] if x==x]
    if not atr_slice or atr_recent!=atr_recent: return False
    atr_long=float(np.mean(atr_slice)); return atr_recent>=cfg.atr_guard_ratio*atr_long
def stage_or_trigger(state:TFState,direction:str,idx:int,up_m:float,up_b:float,lo_m:float,lo_b:float,line_val:float,label:str)->str:
    if cfg.confirm_bars>0:
        state.pending_confirm=PendingConfirm(direction,cfg.confirm_bars,0,up_m,up_b,lo_m,lo_b,label); return "STAGED"
    else:
        state.pending_retest=PendingRetest(direction,idx,up_m,up_b,lo_m,lo_b,label); return "TRIGGERED"
def process_triangle(sym,tf,st,allowed_dir): 
    i=len(st.candles)-1
    if i<max(cfg.lookback_bars,cfg.atr_window+5,cfg.vol_ma_window): return None
    tri=compute_triangle(st.candles)
    if tri is None: return None
    if not tri_contraction_ok(st.candles,tri,st.atr[i],st.atr): return None
    seed=breakout_seed_triangle(st.candles,tri,st.vol_ma[i],allowed_dir)
    if seed is None: return None
    atr_recent=st.atr[i]
    if np.isnan(atr_recent) or not body_filters(seed,atr_recent): return None
    return stage_or_trigger(st,seed["direction"],seed["idx"],tri.up_m,tri.up_b,tri.lo_m,tri.lo_b,seed["line"],f"{seed['label']}")
def process_box(sym,tf,st,allowed_dir):
    i=len(st.candles)-1
    if i<max(cfg.box_window+5,cfg.vol_ma_window): return None
    rng=detect_box(st.candles)
    if not rng: return None
    seed=breakout_seed_box(st.candles,rng,st.vol_ma[i],allowed_dir)
    if seed is None: return None
    atr_recent=st.atr[i]
    if np.isnan(atr_recent) or not body_filters(seed,atr_recent): return None
    up_m=0.0; up_b=rng[0]; lo_m=0.0; lo_b=rng[1]
    return stage_or_trigger(st,seed["direction"],seed["idx"],up_m,up_b,lo_m,lo_b,seed["line"],seed["label"])
def process_double_patterns(sym,tf,st,allowed_dir):
    i=len(st.candles)-1
    if i<40: return None
    dt=detect_double_top(st.candles)
    if dt and (allowed_dir in (None,dt["dir"])):
        up_m=0.0; up_b=dt["neckline"]; lo_m=0.0; lo_b=dt["neckline"]
        return stage_or_trigger(st,dt["dir"],i,up_m,up_b,lo_m,lo_b,dt["neckline"],dt["label"])
    db=detect_double_bottom(st.candles)
    if db and (allowed_dir in (None,db["dir"])):
        up_m=0.0; up_b=db["neckline"]; lo_m=0.0; lo_b=db["neckline"]
        return stage_or_trigger(st,db["dir"],i,up_m,up_b,lo_m,lo_b,db["neckline"],db["label"])
    return None
def process_head_shoulders(sym,tf,st,allowed_dir):
    i=len(st.candles)-1
    if i<60: return None
    hs=detect_head_shoulders(st.candles)
    if hs and (allowed_dir in (None,hs["dir"])):
        up_m=0.0; up_b=hs["neckline"]; lo_m=0.0; lo_b=hs["neckline"]
        return stage_or_trigger(st,hs["dir"],i,up_m,up_b,lo_m,lo_b,hs["neckline"],hs["label"])
    inv=detect_inverse_head_shoulders(st.candles)
    if inv and (allowed_dir in (None,inv["dir"])):
        up_m=0.0; up_b=inv["neckline"]; lo_m=0.0; lo_b=inv["neckline"]
        return stage_or_trigger(st,inv["dir"],i,up_m,up_b,lo_m,lo_b,inv["neckline"],inv["label"])
    return None
async def handle_confirmation_and_alert(sym,tf,st:TFState):
    i=len(st.candles)-1; pc=st.pending_confirm
    if not pc: return
    up_line=pc.up_m*i+pc.up_b; lo_line=pc.lo_m*i+pc.lo_b; line_here=up_line if pc.direction=="UP" else lo_line
    if bar_confirms(pc.direction,st.candles[i].close,line_here):
        pc.confirms_got+=1
        if pc.confirms_got>=pc.confirms_needed:
            price=st.candles[i].close; now=time.time()
            if now-st.last_alert_time>10:
                st.last_alert_time=now
                msg=(f"*{sym.upper()} {tf}* — *BREAKOUT (confirmed)* `{pc.direction}` [{pc.label}]\n"
                     f"• Close: `{price:.6f}` vs Line: `{line_here:.6f}`\n"
                     f"• Volume MA({cfg.vol_ma_window}): `{(st.vol_ma[i] if i<len(st.vol_ma) else float('nan')):.0f}`\n"
                     f"• Next: watching for *retest* within `{cfg.retest_max_bars}` bars (±{cfg.retest_tolerance_pct*100:.2f}%)")
                print(msg.replace("*","").replace("`","")); await telegram(msg)
            st.pending_retest=PendingRetest(pc.direction,i,pc.up_m,pc.up_b,pc.lo_m,pc.lo_b,pc.label); st.pending_confirm=None
    else:
        st.pending_confirm=None
async def handle_retest(sym,tf,st:TFState):
    pr=st.pending_retest
    if not pr: return
    i=len(st.candles)-1; bars_since=i-pr.breakout_idx
    if bars_since>cfg.retest_max_bars:
        await telegram(f"*{sym.upper()} {tf}* — Retest window expired (>{cfg.retest_max_bars} bars)."); st.pending_retest=None; return
    up_line=pr.up_m*i+pr.up_b; lo_line=pr.lo_m*i+pr.lo_b
    if pr.direction=="UP":
        line_here=up_line; touched=within_tolerance(st.candles[i].low,line_here,cfg.retest_tolerance_pct)
        confirm=st.candles[i].close>line_here if cfg.retest_confirm_close else touched
        if touched and confirm:
            entry=st.candles[i].close; retest_extreme=st.candles[i].low
            lv=compute_sl_tp("UP",entry,retest_extreme,line_here); sl=lv["SL"]; L=leverage_for_loss(entry,sl)
            msg=(f"*{sym.upper()} {tf}* — ✅ *Retest LONG Entry* [{pr.label}]\n"
                 f"• Entry: `{entry:.6f}` (line `{line_here:.6f}`)\n"
                 f"• SL: `{sl:.6f}`  (Lev≈`{L:.2f}x` to risk ~{cfg.loss_pct_at_sl*100:.0f}% at SL)\n"
                 f"• TP_1.0R: `{lv['TP_1.0R']:.6f}`\n"
                 f"• TP_2.0R: `{lv['TP_2.0R']:.6f}`")
            print(msg.replace("*","").replace("`","")); await telegram(msg); st.pending_retest=None
    else:
        line_here=lo_line; touched=within_tolerance(st.candles[i].high,line_here,cfg.retest_tolerance_pct)
        confirm=st.candles[i].close<line_here if cfg.retest_confirm_close else touched
        if touched and confirm:
            entry=st.candles[i].close; retest_extreme=st.candles[i].high
            lv=compute_sl_tp("DOWN",entry,retest_extreme,line_here); sl=lv["SL"]; L=leverage_for_loss(entry,sl)
            msg=(f"*{sym.upper()} {tf}* — ✅ *Retest SHORT Entry* [{pr.label}]\n"
                 f"• Entry: `{entry:.6f}` (line `{line_here:.6f}`)\n"
                 f"• SL: `{sl:.6f}`  (Lev≈`{L:.2f}x` to risk ~{cfg.loss_pct_at_sl*100:.0f}% at SL)\n"
                 f"• TP_1.0R: `{lv['TP_1.0R']:.6f}`\n"
                 f"• TP_2.0R: `{lv['TP_2.0R']:.6f}`")
            print(msg.replace("*","").replace("`","")); await telegram(msg); st.pending_retest=None
def process_all_patterns(sym,tf,st,allowed_dir):
    res=process_triangle(sym,tf,st,allowed_dir)
    if res: return res
    res=process_box(sym,tf,st,allowed_dir)
    if res: return res
    res=process_double_patterns(sym,tf,st,allowed_dir)
    if res: return res
    res=process_head_shoulders(sym,tf,st,allowed_dir)
    if res: return res
    return None
    
async def main():
    await telegram("ping")
    start_msg=("🚀 Top-Down Multi-Pattern Bot started\n"
               f"• Symbols: {', '.join([s.upper() for s in SYMBOLS])}\n"
               f"• TFs: 5m(entry), 15m(filter), 1h(trend)\n"
               f"• Filters: 1h EMA200(slope+side), 15m EMA50↔EMA200, 5m ATR≥{cfg.atr_guard_ratio:.2f}×ATR200\n"
               f"• Patterns: triangles({', '.join(cfg.patterns_tri)}), box, double_top/bottom, head&shoulders")
    print(start_msg); await telegram(start_msg)
    if not cfg.telegram_bot_token or not cfg.telegram_chat_id:
        print("[WARN] Set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID to receive Telegram messages.")
    states:Dict[str,SymbolState]={s:SymbolState() for s in SYMBOLS}
    await seed_all(states)
    async for raw in multiplex_stream(SYMBOLS,TF_PRIORITY):
        parsed=parse_ws_kline(raw)
        if not parsed: continue
        sym,tf,candle=parsed
        st=states[sym].tfs[tf]; st.candles.append(candle)
        vols=[c.volume for c in st.candles]; st.vol_ma.append(ma(vols,cfg.vol_ma_window))
        st.atr.append(atr_calc(st.candles,cfg.atr_window))
        for tf_c in TF_PRIORITY:
            tf_state=states[sym].tfs[tf_c]
            await handle_confirmation_and_alert(sym,tf_c,tf_state)
            await handle_retest(sym,tf_c,tf_state)
        if tf=="5m":
            trend=htf_trend(states,sym)
            if trend=="NONE": continue
            if not mtf_confluence(states,sym,trend): continue
            if not atr_guard_ok(states,sym): continue
            _=process_all_patterns(sym,"5m",states[sym].tfs["5m"],allowed_dir=trend)
if __name__=="__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")
