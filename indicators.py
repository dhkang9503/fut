import numpy as np, pandas as pd
from .config import ATR_LEN, SWING_K

def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def atr(high, low, close, n=14):
    h_l=high-low; h_pc=(high-close.shift()).abs(); l_pc=(low-close.shift()).abs()
    tr=pd.concat([h_l,h_pc,l_pc],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def swings(df,k=SWING_K):
    n=len(df); sh=np.zeros(n,dtype=bool); sl=np.zeros(n,dtype=bool)
    for i in range(n):
        lo=max(0,i-k); hi=min(n-1,i+k)
        sh[i]=df.loc[i,"high"]==df.loc[lo:hi,"high"].max()
        sl[i]=df.loc[i,"low"]==df.loc[lo:hi,"low"].min()
    return sh,sl

def detect_fvg_ob(df):
    n=len(df)
    bull=np.zeros(n,dtype=bool); bear=np.zeros(n,dtype=bool)
    fvl=np.full(n,np.nan); fvh=np.full(n,np.nan)
    for i in range(2,n):
        c1h=df.loc[i-2,"high"]; c1l=df.loc[i-2,"low"]
        c3h=df.loc[i,"high"];   c3l=df.loc[i,"low"]
        if c1h<c3l: bull[i]=True; fvl[i]=c1h; fvh[i]=c3l
        elif c1l>c3h: bear[i]=True; fvl[i]=c3h; fvh[i]=c1l
    df["bull_fvg"]=bull; df["bear_fvg"]=bear; df["fvg_low"]=fvl; df["fvg_high"]=fvh
    abfl=np.full(n,np.nan); abfh=np.full(n,np.nan); abel=np.full(n,np.nan); abeh=np.full(n,np.nan)
    last_b=(np.nan,np.nan); last_s=(np.nan,np.nan)
    for i in range(n):
        if not np.isnan(last_b[0]) and (df.loc[i,"low"]<=last_b[0]) and (df.loc[i,"high"]>=last_b[1]): last_b=(np.nan,np.nan)
        if not np.isnan(last_s[0]) and (df.loc[i,"low"]<=last_s[0]) and (df.loc[i,"high"]>=last_s[1]): last_s=(np.nan,np.nan)
        if df.loc[i,"bull_fvg"]: last_b=(df.loc[i,"fvg_low"],df.loc[i,"fvg_high"])
        if df.loc[i,"bear_fvg"]: last_s=(df.loc[i,"fvg_low"],df.loc[i,"fvg_high"])
        abfl[i],abfh[i]=last_b; abel[i],abeh[i]=last_s
    df["active_bull_fvg_low"]=abfl; df["active_bull_fvg_high"]=abfh
    df["active_bear_fvg_low"]=abel; df["active_bear_fvg_high"]=abeh
    df["bull_ob_low"]=np.nan; df["bull_ob_high"]=np.nan; df["bear_ob_low"]=np.nan; df["bear_ob_high"]=np.nan
    look=6
    sh,sl=swings(df)
    df["swing_high"]=sh; df["swing_low"]=sl
    lsh=np.full(n,np.nan); lsl=np.full(n,np.nan); curh=np.nan; curl=np.nan
    for i in range(n):
        if sh[i]: curh=df.loc[i,"high"]
        if sl[i]: curl=df.loc[i,"low"]
        lsh[i]=curh; lsl[i]=curl
    df["last_swing_high_val"]=lsh; df["last_swing_low_val"]=lsl
    for i in range(max(20,look+5),n):
        ph=df.loc[i-1,"last_swing_high_val"]; pl=df.loc[i-1,"last_swing_low_val"]
        if np.isnan(ph) or np.isnan(pl): continue
        body=abs(df.loc[i,"close"]-df.loc[i,"open"]); atrv=df.loc[i,"atr"]
        if (df.loc[i,"high"]>ph) and (body>atrv):
            w=df.loc[i-look:i-1]; red=w[w["close"]<w["open"]]
            if len(red): c=red.tail(1).iloc[0]; df.loc[i,"bull_ob_low"]=c["low"]; df.loc[i,"bull_ob_high"]=c["high"]
        if (df.loc[i,"low"]<pl) and (body>atrv):
            w=df.loc[i-look:i-1]; green=w[w["close"]>w["open"]]
            if len(green): c=green.tail(1).iloc[0]; df.loc[i,"bear_ob_low"]=c["low"]; df.loc[i,"bear_ob_high"]=c["high"]
    abobl=np.full(n,np.nan); abobh=np.full(n,np.nan); aeobl=np.full(n,np.nan); aeobh=np.full(n,np.nan)
    bull=(np.nan,np.nan); bear=(np.nan,np.nan)
    for i in range(n):
        if not np.isnan(bull[0]) and df.loc[i,"close"]<bull[0]: bull=(np.nan,np.nan)
        if not np.isnan(bear[0]) and df.loc[i,"close"]>bear[1]: bear=(np.nan,np.nan)
        if not np.isnan(df.loc[i,"bull_ob_low"]): bull=(df.loc[i,"bull_ob_low"], df.loc[i,"bull_ob_high"])
        if not np.isnan(df.loc[i,"bear_ob_low"]): bear=(df.loc[i,"bear_ob_low"], df.loc[i,"bear_ob_high"])
        abobl[i],abobh[i]=bull; aeobl[i],aeobh[i]=bear
    df["active_bull_ob_low"]=abobl; df["active_bull_ob_high"]=abobh
    df["active_bear_ob_low"]=aeobl; df["active_bear_ob_high"]=aeobh
    return df

def prepare_ohlcv(df):
    df=df.copy()
    df["ema50"]=ema(df["close"],50); df["ema100"]=ema(df["close"],100)
    df["ema50_slope"]=df["ema50"].diff(); df["ema100_slope"]=df["ema100"].diff()
    df["uptrend50"]=(df["close"]>df["ema50"])&(df["ema50_slope"]>0)
    df["downtrend50"]=(df["close"]<df["ema50"])&(df["ema50_slope"]<0)
    df["uptrend100"]=(df["close"]>df["ema100"])&(df["ema100_slope"]>0)
    df["downtrend100"]=(df["close"]<df["ema100"])&(df["ema100_slope"]<0)
    df["atr"]=atr(df["high"],df["low"],df["close"],n=ATR_LEN).fillna(method="bfill")
    df["vol_ma50"]=df["volume"].rolling(50,min_periods=1).mean()
    df=detect_fvg_ob(df); return df

def overlap(a_low,a_high,b_low,b_high):
    import pandas as pd
    for x in (a_low,a_high,b_low,b_high):
        if pd.isna(x): return False
    return max(a_low,b_low) < min(a_high,b_high)
