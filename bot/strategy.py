import numpy as np, pandas as pd
from .config import (FVG_MIN_ATR, OB_MIN_ATR, OB_MAX_ATR, VOL_SPIKE, RR, PT1_SHARE, TRAIL_ATR_MULT)
from .indicators import overlap

def generate_signal_row(df, i:int):
    row=df.iloc[i]
    bf_l,bf_h=row["active_bull_fvg_low"],row["active_bull_fvg_high"]
    bb_l,bb_h=row["active_bear_fvg_low"],row["active_bear_fvg_high"]
    bo_l,bo_h=row["active_bull_ob_low"],row["active_bull_ob_high"]
    so_l,so_h=row["active_bear_ob_low"],row["active_bear_ob_high"]
    fvg_b_ok = (pd.notna(bf_l) and pd.notna(bf_h) and (bf_h-bf_l)>=FVG_MIN_ATR*row["atr"])
    fvg_s_ok = (pd.notna(bb_l) and pd.notna(bb_h) and (bb_h-bb_l)>=FVG_MIN_ATR*row["atr"])
    ob_b_ok = (pd.notna(bo_l) and pd.notna(bo_h) and (OB_MIN_ATR*row["atr"] <= (bo_h-bo_l) <= OB_MAX_ATR*row["atr"]))
    ob_s_ok = (pd.notna(so_l) and pd.notna(so_h) and (OB_MIN_ATR*row["atr"] <= (so_h-so_l) <= OB_MAX_ATR*row["atr"]))
    vol_ok = row["volume"] > (1+VOL_SPIKE)*row["vol_ma50"]

    if row["uptrend50"] and row["uptrend100"] and fvg_b_ok and ob_b_ok and vol_ok and overlap(bf_l,bf_h,bo_l,bo_h):
        touched=(row["low"]<=bo_h) and (row["high"]>=bo_l)
        ob_mid=(bo_l+bo_h)/2 if (pd.notna(bo_l) and pd.notna(bo_h)) else np.nan
        trigger=pd.notna(ob_mid) and (row["close"]>ob_mid)
        if touched and trigger:
            entry=df.iloc[i+1]["open"] if i+1<len(df) else row["close"]
            stop=bo_l; risk=entry-stop; target=entry+RR*risk; pt1=bf_l
            return dict(side="LONG",entry=entry,stop=stop,target=target,pt1=pt1)
    if row["downtrend50"] and row["downtrend100"] and fvg_s_ok and ob_s_ok and vol_ok and overlap(bb_l,bb_h,so_l,so_h):
        touched=(row["low"]<=so_h) and (row["high"]>=so_l)
        ob_mid=(so_l+so_h)/2 if (pd.notna(so_l) and pd.notna(so_h)) else np.nan
        trigger=pd.notna(ob_mid) and (row["close"]<ob_mid)
        if touched and trigger:
            entry=df.iloc[i+1]["open"] if i+1<len(df) else row["close"]
            stop=so_h; risk=stop-entry; target=entry-RR*risk; pt1=bb_h
            return dict(side="SHORT",entry=entry,stop=stop,target=target,pt1=pt1)
    return None
