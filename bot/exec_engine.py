import pandas as pd
from .config import MODE, SYMBOL, RR, TIMEOUT_BARS
from .indicators import prepare_ohlcv
from .strategy import generate_signal_row
from .risk import position_size_and_leverage
from .telegram import notify
from .rest_client import BitgetREST

class Engine:
    def __init__(self):
        self.df=pd.DataFrame(columns=["time","open","high","low","close","volume"])
        self.equity=1000.0
        self.rest=BitgetREST()

    async def close(self): await self.rest.close()

    async def on_new_bar(self, bar:dict):
        self.df=pd.concat([self.df,pd.DataFrame([bar])], ignore_index=True)
        if len(self.df)<120: return
        df=prepare_ohlcv(self.df.copy())
        i=len(df)-2  # evaluate just-closed bar
        sig=generate_signal_row(df,i)
        if not sig: return
        side=sig["side"]; entry=sig["entry"]; stop=sig["stop"]; target=sig["target"]; pt1=sig["pt1"]
        qty, lev=position_size_and_leverage(SYMBOL, entry, stop, self.equity)
        risk_d=0.01*self.equity
        await notify(f"[SIGNAL] {side} e={entry:.2f} sl={stop:.2f} t2R={target:.2f} pt1={pt1:.2f} qty={qty} lev={lev:.1f} risk=${risk_d:.2f}")
        if MODE=="paper":
            # Here you would simulate PT1 + trailing; simplified placeholder:
            R=0.5  # assume PT1 hit (for demo skeleton)
            self.equity += R*risk_d
            await notify(f"[PAPER] Closed R={R:.2f}; equity={self.equity:.2f}")
        else:
            resp=await self.rest.place_market_order(side, qty, reduce_only=False)
            await notify(f"[LIVE] Market entry placed: {resp}")
