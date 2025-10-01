import asyncio
from .ws_public import stream_kline_5m
from .exec_engine import Engine
from .storage import init_db
from .telegram import notify

async def main():
    await init_db()
    eng=Engine()
    await notify("Bot starting (real-time WS).")
    try:
        await stream_kline_5m(eng.on_bar_close)
    finally:
        await eng.close()

if __name__=="__main__":
    asyncio.run(main())
