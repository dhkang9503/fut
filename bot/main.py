import asyncio
from .ws_public import stream_kline_5m
from .exec_engine import Engine
from .storage import init_db
from .telegram import notify_system
from .config import SYMBOL, BACKFILL_BARS
from .bootstrap import warmup_engine

async def main():
    await init_db()
    eng = Engine()
    await notify_system("Bot starting (real-time WS).")
    try:
        loaded = await warmup_engine(eng, SYMBOL, BACKFILL_BARS)
        await notify_system(f"Warm-up complete: {loaded} bars preloaded.")
    except Exception as e:
        await notify_system(f"Warm-up fetch failed (continuing without it): {e}")
    try:
        await stream_kline_5m(eng.on_bar_close)
    finally:
        await eng.close()

if __name__ == "__main__":
    asyncio.run(main())
