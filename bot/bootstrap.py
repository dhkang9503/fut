import httpx, asyncio
from typing import List, Dict

API_URL = "https://api.bitget.com/api/v3/market/candles"

async def fetch_recent_candles(symbol: str, limit: int = 240):
    params = {
        "category": "USDT-FUTURES",
        "symbol": symbol,
        "granularity": "5m",
        "type": "MARKET",
        "limit": str(limit)
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(API_URL, params=params)
        r.raise_for_status()
        js = r.json()
        data = js.get("data", [])
        rows = []
        for it in data:
            ts, o, h, l, c, v = it[:6]
            ts = int(ts); ts = ts//1000 if ts > 1_000_000_000_000 else ts
            rows.append({"time": ts, "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": float(v)})
        rows.sort(key=lambda x: x["time"])
        return rows

async def warmup_engine(engine, symbol: str, limit: int):
    bars = await fetch_recent_candles(symbol, limit=limit)
    await engine.seed_bars(bars)
    return len(bars)
