import asyncio, json, random, time, websockets
from typing import Callable, Awaitable, Dict, Any
from .config import WS_PUBLIC, SYMBOL, TIMEFRAME
from .telegram import notify

# WS stability knobs
PING_INTERVAL = 25         # seconds between native ping frames
PONG_TIMEOUT  = 12         # must receive pong within this after ping
STALE_SEC     = 190        # consider data stale if no messages for ~3+ minutes
MAX_BACKOFF   = 60         # max reconnect backoff (seconds)

INST_TYPE = "USDT-FUTURES"
CANDLE_CHANNEL = "candle5m"  # Bitget V2 channel name for 5m
ENABLE_TICKER = True         # co-subscribe to ticker for lightweight heartbeat

_last_msg_ts  = 0.0
_last_pong_ts = 0.0
_last_kline_ts = None

def _normalize(item: dict) -> dict:
    # Normalize V2 candle payload into {ts, o, h, l, c, v, closed}
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

async def stream_kline_5m(on_bar_close: Callable[[Dict[str, Any]], Awaitable[None]]):
    global _last_msg_ts, _last_pong_ts, _last_kline_ts
    backoff = 1
    while True:
        try:
            args = [{
                "instType": INST_TYPE,
                "channel": CANDLE_CHANNEL,
                "instId": SYMBOL
            }]
            if ENABLE_TICKER:
                args.append({
                    "instType": INST_TYPE,
                    "channel": "ticker",
                    "instId": SYMBOL
                })
            sub = {"op": "subscribe", "args": args}

            async with websockets.connect(
                WS_PUBLIC, ping_interval=None, ping_timeout=None, close_timeout=30
            ) as ws:
                await ws.send(json.dumps(sub))
                await notify("WS connected and subscribed (V2).")
                _last_msg_ts = time.time()
                _last_pong_ts = _last_msg_ts

                async def _pinger():
                    while True:
                        await asyncio.sleep(PING_INTERVAL)
                        try:
                            pong_waiter = await ws.ping()  # native ping frame
                            await asyncio.wait_for(pong_waiter, timeout=PONG_TIMEOUT)
                            # pong OK
                            nonlocal _last_pong_ts
                            _last_pong_ts = time.time()
                        except asyncio.TimeoutError:
                            raise asyncio.TimeoutError("pong-timeout")
                        except Exception:
                            raise

                async def _receiver():
                    async for raw in ws:
                        _last_msg_ts = time.time()
                        if raw == "pong":
                            _last_pong_ts = time.time()
                            continue
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue

                        # Ignore non-data frames
                        if not isinstance(msg, dict):
                            continue

                        # Candle/ticker multiplexer
                        data = msg.get("data")
                        channel = (msg.get("arg") or {}).get("channel", "")

                        # Use ticker only as heartbeat; ignore its content
                        if channel == "ticker":
                            continue

                        if not data:
                            continue
                        for item in data:
                            bar = _normalize(item)
                            if not bar:
                                continue
                            _last_kline_ts = bar["ts"]
                            if bar.get("closed"):
                                await on_bar_close({
                                    "time": bar["ts"],
                                    "open": bar["o"],
                                    "high": bar["h"],
                                    "low":  bar["l"],
                                    "close":bar["c"],
                                    "volume":bar.get("v", 0.0)
                                })

                async def _stale_watch():
                    while True:
                        await asyncio.sleep(5)
                        now = time.time()
                        # if pong is overdue → reconnect
                        if (now - _last_pong_ts) > (PING_INTERVAL + PONG_TIMEOUT + 5):
                            raise asyncio.TimeoutError("pong-missed")
                        # if overall quiet beyond STALE_SEC → soft reconnect
                        if (now - _last_msg_ts) > STALE_SEC:
                            raise asyncio.TimeoutError("stale-connection")

                await asyncio.gather(_pinger(), _receiver(), _stale_watch())

        except Exception as e:
            await notify(f"WS error, reconnecting: {e}")
            # Exponential backoff with jitter
            delay = min(backoff, MAX_BACKOFF) * (1 + random.uniform(-0.1, 0.1))
            await asyncio.sleep(delay)
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue
