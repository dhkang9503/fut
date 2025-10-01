import asyncio, json, websockets
from typing import Callable, Awaitable, Dict, Any
from .config import WS_PUBLIC, SYMBOL, PRODUCT_TYPE, TIMEFRAME
from .telegram import notify

async def stream_kline_5m(on_bar_close: Callable[[Dict[str, Any]], Awaitable[None]]):
    sub = {
        "op": "subscribe",
        "args": [{
            "instType": "mc",
            "channel": f"kline.{TIMEFRAME}",
            "instId": f"{SYMBOL}_{PRODUCT_TYPE.upper()}"
        }]
    }
    while True:
        try:
            async with websockets.connect(WS_PUBLIC, ping_interval=20, ping_timeout=20, close_timeout=10) as ws:
                await ws.send(json.dumps(sub))
                await notify(f"WS connected and subscribed.")
                async for raw in ws:
                    try:
                        msg=json.loads(raw)
                    except Exception:
                        continue
                    data=msg.get("data")
                    if not data: continue
                    for item in data:
                        bar=_normalize(item)
                        if bar and bar.get("closed"):
                            await on_bar_close({
                                "time": bar["ts"],
                                "open": bar["o"],
                                "high": bar["h"],
                                "low":  bar["l"],
                                "close":bar["c"],
                                "volume":bar.get("v",0.0)
                            })
        except Exception as e:
            await notify(f"WS error, reconnecting: {e}")
            await asyncio.sleep(3)

def _normalize(item: dict)->dict:
    if isinstance(item, dict):
        ts=item.get("ts") or item.get("t") or item.get("time")
        o=item.get("o") or item.get("open"); h=item.get("h") or item.get("high")
        l=item.get("l") or item.get("low");  c=item.get("c") or item.get("close")
        v=item.get("v") or item.get("volume") or 0.0
        closed=item.get("confirm", True)
        if ts and o is not None and c is not None:
            ts=int(ts); ts=ts//1000 if ts>1e12 else ts
            return {"ts":ts,"o":float(o),"h":float(h),"l":float(l),"c":float(c),"v":float(v),"closed":bool(closed)}
    if isinstance(item, list) and len(item)>=6:
        ts,o,h,l,c,v=item[:6]; confirm=item[6] if len(item)>6 else True
        ts=int(ts); ts=ts//1000 if ts>1e12 else ts
        return {"ts":ts,"o":float(o),"h":float(h),"l":float(l),"c":float(c),"v":float(v),"closed":bool(confirm)}
    return {}
