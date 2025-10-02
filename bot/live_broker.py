import os, time, hmac, hashlib, base64, json, httpx, math
from typing import Optional, Dict, Any
from .config import SYMBOL, PRODUCT_TYPE, MARGIN_PCT, RISK_PCT, SYMBOL_INFO

REST_BASE = os.getenv("REST_BASE", "https://api.bitget.com")
API_KEY = os.getenv("BITGET_API_KEY", "")
API_SECRET = os.getenv("BITGET_API_SECRET", "")
API_PASSPHRASE = os.getenv("BITGET_API_PASSPHRASE", "")

MARGIN_MODE = "isolated"
MARGIN_COIN = "USDT"

def _ts() -> str:
    return str(int(time.time() * 1000))

def _sign(ts: str, method: str, path: str, query: str = "", body: str = "") -> str:
    method = method.upper()
    prehash = f"{ts}{method}{path}{'?' + query if query else ''}{body}"
    digest = hmac.new(API_SECRET.encode(), prehash.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()

async def _req(method: str, path: str, params: Optional[Dict[str, Any]]=None, body: Optional[Dict[str, Any]]=None):
    if params is None: params = {}
    if body is None: body = {}
    ts = _ts()
    query = "&".join([f"{k}={params[k]}" for k in sorted(params)]) if params else ""
    body_s = json.dumps(body) if body else ""
    sign = _sign(ts, method, path, query, body_s)
    headers = {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": sign,
        "ACCESS-PASSPHRASE": API_PASSPHRASE,
        "ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json",
    }
    url = REST_BASE + path + (f"?{query}" if query else "")
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.request(method, url, headers=headers, content=body_s if body_s else None)
        r.raise_for_status()
        js = r.json()
        if js.get("code") not in ("00000", "0", 0, None):
            raise Exception(f"Bitget error: {js}")
        return js

def _round_step(x: float, step: float) -> float:
    return math.floor(x / step) * step

async def compute_qty_and_leverage(entry: float, stop: float, equity: float) -> Dict[str, Any]:
    info = SYMBOL_INFO.get(SYMBOL, {"size_step": 0.001, "min_size": 0.001, "max_leverage": 100})
    risk_amount = equity * RISK_PCT
    per_unit = abs(entry - stop)
    if per_unit <= 0:
        return {"qty": 0.0, "lev": 1.0}
    raw_qty = risk_amount / per_unit
    qty = max(info["min_size"], _round_step(raw_qty, info["size_step"]))
    notional = entry * qty
    alloc = max(1e-9, equity * MARGIN_PCT)
    lev = min(max(1.0, notional / alloc), info.get("max_leverage", 100))
    return {"qty": qty, "lev": lev}

async def ensure_leverage(lev: float, hold_side: Optional[str] = None):
    product = PRODUCT_TYPE.replace("umcbl","USDT-FUTURES").upper()
    body = {"symbol": SYMBOL, "productType": product, "marginCoin": MARGIN_COIN, "leverage": str(int(math.ceil(lev)))}
    if hold_side: body["holdSide"] = hold_side
    return await _req("POST", "/api/v2/mix/account/set-leverage", body=body)

async def open_with_server_sl(side: str, qty: float, preset_sl: Optional[float] = None):
    """Market entry with exchange-managed SL; no server TP (PT1/Trail are local)."""
    product = PRODUCT_TYPE.replace("umcbl","USDT-FUTURES").upper()
    body = {
        "symbol": SYMBOL,
        "productType": product,
        "marginMode": MARGIN_MODE,
        "marginCoin": MARGIN_COIN,
        "orderType": "market",
        "side": side,  # 'buy' or 'sell'
        "size": f"{qty:.10f}",
    }
    if preset_sl is not None:
        body["presetStopLossPrice"] = f"{preset_sl}"
    body["clientOid"] = f"live#{int(time.time()*1000)}"
    return await _req("POST", "/api/v2/mix/order/place-order", body=body)

async def reduce_only(side: str, qty: float):
    product = PRODUCT_TYPE.replace("umcbl","USDT-FUTURES").upper()
    body = {
        "symbol": SYMBOL,
        "productType": product,
        "marginMode": MARGIN_MODE,
        "marginCoin": MARGIN_COIN,
        "orderType": "market",
        "side": side,  # opposite of entry for one-way
        "size": f"{qty:.10f}",
        "reduceOnly": "YES"
    }
    body["clientOid"] = f"close#{int(time.time()*1000)}"
    return await _req("POST", "/api/v2/mix/order/place-order", body=body)

def close_side_for(signal_side: str) -> str:
    # signal_side: "LONG"/"SHORT"
    return "sell" if signal_side == "LONG" else "buy"
