import httpx
from typing import List, Dict

API_URL = "https://api.bitget.com/api/v2/mix/market/candles"

async def fetch_recent_candles(symbol: str, limit: int = 240, product_type: str = "usdt-futures") -> List[Dict]:
    """
    Fetch recent 5m futures candles (oldest->newest) using Bitget V2 Mix endpoint.
    Returns list of dicts: {time, open, high, low, close, volume}
    """
    params = {
        "symbol": symbol,
        "granularity": "5m",
        "limit": str(limit),
        "productType": product_type
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(API_URL, params=params)
        r.raise_for_status()
        js = r.json()
        data = js.get("data", [])
        rows = []
        # Response format is a list of arrays: [ts_ms, o, h, l, c, volBase, volQuote]
        for it in data:
            ts_ms = int(it[0])
            o, h, l, c = map(float, it[1:5])
            v_base = float(it[5]) if len(it) > 5 else 0.0
            ts = ts_ms // 1000
            rows.append({"time": ts, "open": o, "high": h, "low": l, "close": c, "volume": v_base})
        rows.sort(key=lambda x: x["time"])
        return rows

async def warmup_engine(engine, symbol: str, limit: int):
    """Warm up engine with historical bars without generating signals/trades."""
    bars = await fetch_recent_candles(symbol, limit=limit)
    await engine.seed_bars(bars)
    return len(bars)
