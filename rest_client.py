import time, hmac, hashlib, base64, json, httpx
from .config import REST_BASE, BITGET_API_KEY, BITGET_API_SECRET, BITGET_API_PASSPHRASE, PRODUCT_TYPE, SYMBOL

def _sign(ts, method, path, body=""):
    pre=f"{ts}{method}{path}{body}"
    mac=hmac.new(BITGET_API_SECRET.encode(), pre.encode(), hashlib.sha256).digest()
    return base64.b64encode(mac).decode()

def _headers(ts, sign):
    return {
        "ACCESS-KEY": BITGET_API_KEY,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": BITGET_API_PASSPHRASE,
        "Content-Type": "application/json"
    }

class BitgetREST:
    def __init__(self):
        self.client=httpx.AsyncClient(timeout=10, base_url=REST_BASE)
    async def close(self): await self.client.aclose()

    async def place_market_order(self, side:str, size:float, reduce_only=False):
        ts=str(int(time.time()*1000)); path="/api/mix/v1/order/placeOrder"
        body=json.dumps({
            "symbol": SYMBOL, "marginCoin":"USDT", "size": f"{size}",
            "side":"open_long" if side=="LONG" else "open_short",
            "orderType":"market", "productType": PRODUCT_TYPE, "reduceOnly": reduce_only
        })
        sign=_sign(ts,"POST",path,body)
        r=await self.client.post(path, data=body, headers=_headers(ts,sign))
        return r.json()
