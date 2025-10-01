import asyncio, json, random, time, websockets, httpx
from typing import Callable, Awaitable, Dict, Any
from .config import WS_PUBLIC, SYMBOL, TIMEFRAME
from .telegram import notify

INST_TYPE = "USDT-FUTURES"   # 선물
CHANNEL   = "candle5m"       # 5분봉 채널명 (V2)
PING_INTERVAL = 30
STALE_SEC = 65
MAX_BACKOFF = 60

_last_msg_ts = 0
_last_kline_ts = None

async def _rest_backfill(symbol: str, since_ts: int):
    # Bitget V3 REST candles
    url = "https://api.bitget.com/api/v3/market/candles"
    params = {
        "category": "USDT-FUTURES",
        "symbol": symbol,
        "granularity": "5m",
        "type": "MARKET",
        "limit": "3"
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json().get("data", [])
        # 표준화: 서버는 최신->과거 순일 수 있음. 정렬하고 since 이후만 반환
        rows = []
        for it in data:
            # V3 포맷 예: [ts, o, h, l, c, vol, ...]
            ts, o, h, l, c, v = it[:6]
            ts = int(ts) // 1000 if int(ts) > 1e12 else int(ts)
            rows.append({"time": ts, "open": float(o), "high": float(h),
                         "low": float(l), "close": float(c), "volume": float(v)})
        rows.sort(key=lambda x: x["time"])
        if since_ts:
            rows = [r for r in rows if r["time"] > since_ts]
        return rows

async def stream_kline_5m(on_bar_close: Callable[[Dict[str, Any]], Awaitable[None]]):
    global _last_msg_ts, _last_kline_ts
    backoff = 1
    while True:
        try:
            sub = {
                "op": "subscribe",
                "args": [{
                    "instType": INST_TYPE,
                    "channel": CHANNEL,
                    "instId": SYMBOL
                }]
            }
            async with websockets.connect(
                WS_PUBLIC, ping_interval=None, ping_timeout=None, close_timeout=30
            ) as ws:
                await ws.send(json.dumps(sub))
                await notify("WS connected and subscribed (V2).")
                _last_msg_ts = time.time()

                # 핑 루프
                async def _pinger():
                    while True:
                        await asyncio.sleep(PING_INTERVAL)
                        try:
                            await ws.send("ping")
                        except Exception:
                            return

                # 수신 루프
                async def _receiver():
                    nonlocal backoff
                    async for raw in ws:
                        _last_msg_ts = time.time()
                        if raw == "pong":
                            continue
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        # 데이터 패킷 처리
                        data = msg.get("data")
                        if not data:
                            continue
                        for item in data:
                            bar = _normalize(item)
                            if not bar:
                                continue
                            _last_kline_ts = bar["ts"]
                            if bar.get("closed"):  # 봉 마감 신호
                                await on_bar_close({
                                    "time": bar["ts"],
                                    "open": bar["o"], "high": bar["h"],
                                    "low": bar["l"], "close": bar["c"],
                                    "volume": bar.get("v", 0.0)
                                })
                    # 소켓 종료 시 루프 탈출
                # 스테일 감지 루프
                async def _stale_watch():
                    while True:
                        await asyncio.sleep(5)
                        if time.time() - _last_msg_ts > STALE_SEC:
                            raise asyncio.TimeoutError("stale-connection")

                await asyncio.gather(_pinger(), _receiver(), _stale_watch())
        except Exception as e:
            await notify(f"WS error, reconnecting: {e}")
            # 백필: 마지막 봉 이후 2~3개 보정
            try:
                rows = await _rest_backfill(SYMBOL, _last_kline_ts or 0)
                for r in rows:
                    await on_bar_close(r)  # backfill은 마감봉으로 간주
            except Exception:
                pass
            # 지수 백오프 + 지터
            await asyncio.sleep(min(backoff, MAX_BACKOFF) * (1 + random.uniform(-0.1, 0.1)))
            backoff = min(backoff * 2, MAX_BACKOFF)

def _normalize(item: dict) -> dict:
    # V2 candle payload 표준화: dict 또는 list 지원
    if isinstance(item, dict):
        ts = item.get("ts") or item.get("t") or item.get("time")
        o = item.get("o") or item.get("open"); h = item.get("h") or item.get("high")
        l = item.get("l") or item.get("low");  c = item.get("c") or item.get("close")
        v = item.get("v") or item.get("volume") or 0.0
        confirm = item.get("confirm", True)
        if ts and o is not None and c is not None:
            ts = int(ts); ts = ts//1000 if ts > 1_000_000_000_000 else ts
            return {"ts": ts, "o": float(o), "h": float(h), "l": float(l), "c": float(c),
                    "v": float(v), "closed": bool(confirm)}
    if isinstance(item, list) and len(item) >= 6:
        ts,o,h,l,c,v = item[:6]
        confirm = item[6] if len(item) > 6 else True
        ts = int(ts); ts = ts//1000 if ts > 1_000_000_000_000 else ts
        return {"ts": ts, "o": float(o), "h": float(h), "l": float(l), "c": float(c),
                "v": float(v), "closed": bool(confirm)}
    return {}
