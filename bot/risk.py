import math
from .config import RISK_PCT, MARGIN_PCT, SYMBOL_INFO

def round_to_step(v, step): return math.floor(v/step)*step

def position_size_and_leverage(symbol, entry, stop, equity):
    meta=SYMBOL_INFO.get(symbol, {"size_step":0.001,"min_size":0.001,"max_leverage":50})
    risk_d = RISK_PCT*equity
    margin_alloc = MARGIN_PCT*equity
    price_diff = abs(entry-stop)
    if price_diff<=0: return 0.0,1.0
    qty = risk_d/price_diff
    qty = max(qty, meta["min_size"])
    qty = round_to_step(qty, meta["size_step"])
    notional = entry*qty
    lev = min(int(notional/max(1e-8,margin_alloc)), meta["max_leverage"])
    return qty, lev
