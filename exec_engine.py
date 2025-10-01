import pandas as pd
from .config import MODE, SYMBOL, RR, TIMEOUT_BARS, TRAIL_ATR_MULT, PT1_SHARE
from .indicators import prepare_ohlcv
from .strategy import generate_signal_row
from .risk import position_size_and_leverage
from .telegram import notify
from .rest_client import BitgetREST

class Engine:
    def __init__(self):
        self.df=pd.DataFrame(columns=["time","open","high","low","close","volume"]).astype({"time":"int64","open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"})
        self.equity=1000.0 if MODE=="paper" else None
        self.open_positions=[]
        self.rest=BitgetREST()

    async def close(self): await self.rest.close()

    async def on_bar_close(self, bar:dict):
        self.df=pd.concat([self.df,pd.DataFrame([bar])], ignore_index=True)
        if len(self.df)<120: return
        df=prepare_ohlcv(self.df.copy())
        i=len(df)-1
        if MODE=="paper":
            await self._paper_manage(df,i)
        sig=generate_signal_row(df,i-1)
        if sig:
            await self._handle_signal(sig, df, i)

    async def _handle_signal(self, sig, df, i):
        side=sig["side"]; entry=sig["entry"]; stop=sig["stop"]; target=sig["target"]; pt1=sig["pt1"]
        qty, lev=position_size_and_leverage(SYMBOL, entry, stop, self.equity or 1000.0)
        risk_d=0.01*(self.equity if self.equity is not None else 1000.0)
        await notify(f"[SIGNAL] {side} e={entry:.2f} sl={stop:.2f} 2R={target:.2f} pt1={pt1:.2f} qty={qty} lev={lev:.1f} risk=${risk_d:.2f}")
        if MODE=="paper":
            pos={"side":side,"entry":entry,"stop":stop,"pt1":pt1,"target":target,"qty":qty,"risk$":risk_d,"entry_index":i,"hit_pt1":False,"trail":entry,"timeout":TIMEOUT_BARS,"lockedR":0.0}
            self.open_positions.append(pos)
        else:
            resp=await self.rest.place_market_order(side, qty, reduce_only=False)
            await notify(f"[LIVE] Market order resp: {resp}")
            # TODO: place stop/pt1/2R plan orders & update trail periodically

    async def _paper_manage(self, df, i):
        cur=df.iloc[i]; hi=cur["high"]; lo=cur["low"]; atr=cur["atr"]
        to_close=[]
        for idx,pos in enumerate(self.open_positions):
            life=i-pos["entry_index"]
            if life>pos["timeout"]:
                r=self._r_from_close(pos, cur["close"]); await self._paper_close(idx,r); to_close.append(idx); continue
            if pos["side"]=="LONG":
                if lo<=pos["stop"]:
                    r=-1.0 if not pos["hit_pt1"] else -0.5; await self._paper_close(idx,r); to_close.append(idx); continue
                if (not pos["hit_pt1"]) and hi>=pos["pt1"]:
                    pos["hit_pt1"]=True; pos["lockedR"] += PT1_SHARE*1.0; pos["trail"]=max(pos["trail"], cur["close"]-TRAIL_ATR_MULT*atr)
                if pos["hit_pt1"]:
                    if lo<=pos["trail"]:
                        r=pos["lockedR"]; await self._paper_close(idx,r); to_close.append(idx); continue
                    if hi>=pos["target"]:
                        r=pos["lockedR"]+PT1_SHARE*RR; await self._paper_close(idx,r); to_close.append(idx); continue
                    pos["trail"]=max(pos["trail"], cur["close"]-TRAIL_ATR_MULT*atr)
            else:
                if hi>=pos["stop"]:
                    r=-1.0 if not pos["hit_pt1"] else -0.5; await self._paper_close(idx,r); to_close.append(idx); continue
                if (not pos["hit_pt1"]) and lo<=pos["pt1"]:
                    pos["hit_pt1"]=True; pos["lockedR"] += PT1_SHARE*1.0; pos["trail"]=min(pos["trail"], cur["close"]+TRAIL_ATR_MULT*atr)
                if pos["hit_pt1"]:
                    if hi>=pos["trail"]:
                        r=pos["lockedR"]; await self._paper_close(idx,r); to_close.append(idx); continue
                    if lo<=pos["target"]:
                        r=pos["lockedR"]+PT1_SHARE*RR; await self._paper_close(idx,r); to_close.append(idx); continue
                    pos["trail"]=min(pos["trail"], cur["close"]+TRAIL_ATR_MULT*atr)
        for j in sorted(to_close, reverse=True):
            self.open_positions.pop(j)

    def _r_from_close(self, pos, close):
        risk=abs(pos["entry"]-pos["stop"])
        return (close-pos["entry"])/risk if pos["side"]=="LONG" else (pos["entry"]-close)/risk

    async def _paper_close(self, idx, R):
        pos=self.open_positions[idx]; pnl=R*pos["risk$"]; self.equity += pnl
        await notify(f"[PAPER] Exit {pos['side']} R={R:.2f} PnL=${pnl:.2f} Equity=${self.equity:.2f}")
