import asyncio, os, pandas as pd
from .exec_engine import Engine
from .storage import init_db
from .telegram import notify

async def fake_feed(engine:Engine):
    path=os.getenv("BACKTEST_CSV","")
    if not path:
        await notify("Set BACKTEST_CSV to feed historical data into the bot for a dry run.")
        return
    df=pd.read_csv(path); df["time"]=pd.to_datetime(df["time"])
    for _,r in df.iterrows():
        bar=dict(time=int(r["time"].timestamp()), open=float(r["open"]), high=float(r["high"]), low=float(r["low"]), close=float(r["close"]), volume=float(r.get("volume",0)))
        await engine.on_new_bar(bar)
        await asyncio.sleep(0.01)

async def main():
    await init_db()
    eng=Engine()
    await notify("Bot starting.")
    try:
        await fake_feed(eng)  # Replace with real WS listener for production
    finally:
        await eng.close()

if __name__=="__main__":
    asyncio.run(main())
