import asyncio, time
import pandas as pd
from .config import MODE, SYMBOL, PRODUCT_TYPE, RR, TIMEOUT_BARS, TRAIL_ATR_MULT, PT1_SHARE, MAX_BARS
try:
    from .config import EQUITY_REFRESH_SEC
except Exception:
    EQUITY_REFRESH_SEC = 10

from .indicators import prepare_ohlcv
from .strategy import generate_signal_row
from .risk import position_size_and_leverage
from .telegram import notify, notify_exit, notify_system, notify_order
from . import live_broker

class Engine:
    def __init__(self):
        self.df = pd.DataFrame(columns=["time","open","high","low","close","volume"]).astype({
            "time":"int64","open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"
        })
        self.equity = 1000.0 if MODE=="paper" else None
        self._last_equity_fetch_ts = 0
        self.open_positions = []
        self.last_processed_ts = 0

    async def close(self):
        pass

    def _trim_df(self):
        try:
            if len(self.df) > MAX_BARS:
                self.df = self.df.iloc[-MAX_BARS:].reset_index(drop=True)
        except Exception:
            pass

    async def _refresh_equity(self, force: bool = False):
        if MODE != "live":
            return
        now = time.time()
        if (not force) and (now - self._last_equity_fetch_ts < EQUITY_REFRESH_SEC):
            return
        try:
            eq = await live_broker.fetch_equity(SYMBOL, PRODUCT_TYPE, margin_coin="USDT")
            if eq and eq > 0:
                self.equity = eq
                await notify_system(f"[LIVE] equity updated: ${eq:.2f}")
            else:
                await notify_system(f"[LIVE] equity fetch returned 0; keeping cache {self.equity}")
            self._last_equity_fetch_ts = now
        except Exception as e:
            await notify_system(f"[LIVE] equity fetch failed: {e}; cache={self.equity}")

    async def seed_bars(self, bars):
        if not bars: return
        self.df = pd.concat([self.df, pd.DataFrame(bars)], ignore_index=True)
        await notify_system(f"Warm-up loaded {len(bars)} bars. Ready for real-time.")
        self.last_processed_ts = int(bars[-1]["time"])
        await self._refresh_equity(force=True)

    async def on_bar_close(self, bar: dict):
        ts = int(bar["time"]); now=int(time.time()); interval=300
        if ts < now - 2*interval: return
        if ts <= self.last_processed_ts: return
        self.last_processed_ts = ts

        self.df = pd.concat([self.df, pd.DataFrame([bar])], ignore_index=True)
        self._trim_df()
        if len(self.df) < 120: return
        df = prepare_ohlcv(self.df.copy()); i=len(df)-1

        # Always manage positions
        await self._manage_positions(df, i)

        sig = generate_signal_row(df, i-1)
        if sig:
            await self._handle_signal(sig, df, i, ts)

    async def _handle_signal(self, sig, df, i, ts):
        side=sig["side"]; entry=sig["entry"]; stop=sig["stop"]; target=sig["target"]; pt1=sig["pt1"]
        ts_str=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))

        if MODE=="paper":
            qty, lev=position_size_and_leverage(SYMBOL, entry, stop, self.equity or 1000.0)
            risk_d=0.01*(self.equity if self.equity is not None else 1000.0)
            await notify(f"[SIGNAL] {side} @{ts_str} e={entry:.2f} sl={stop:.2f} 2R={target:.2f} pt1={pt1:.2f} qty={qty} lev={lev:.1f} risk=${risk_d:.2f}")
            pos={"side":side,"entry":entry,"stop":stop,"pt1":pt1,"target":target,
                 "qty_total":qty,"qty_rem":qty,"entry_index":i,"entry_time":ts,
                 "hit_pt1":False,"trail":entry,"timeout":TIMEOUT_BARS,"lockedR":0.0,"live":False}
            self.open_positions.append(pos)
        else:
            await self._refresh_equity(force=True)
            equity_used = self.equity or 1000.0
            sizing = await live_broker.compute_qty_and_leverage(entry, stop, equity_used)
            qty = sizing["qty"]; lev = sizing["lev"]
            await notify(f"[SIGNAL] {side} @{ts_str} e={entry:.2f} sl={stop:.2f} 2R={target:.2f} pt1={pt1:.2f} qty={qty} lev=x{lev:.1f} eq=${equity_used:.2f}")
            if qty <= 0:
                await notify_system("[LIVE] sizing produced qty<=0; skip")
                return
            await live_broker.ensure_leverage(lev, hold_side=("long" if side=="LONG" else "short"))
            resp = await live_broker.open_with_server_sl(("buy" if side=="LONG" else "sell"), qty, preset_sl=stop)
            await notify(f"[LIVE] Market order resp: {resp}", event="order")
            pos={"side":side,"entry":entry,"stop":stop,"pt1":pt1,"target":target,
                 "qty_total":qty,"qty_rem":qty,"entry_index":i,"entry_time":ts,
                 "hit_pt1":False,"trail":entry,"timeout":TIMEOUT_BARS,"lockedR":0.0,"live":True}
            self.open_positions.append(pos)

    async def _manage_positions(self, df: pd.DataFrame, i: int):
        cur=df.iloc[i]; hi=cur["high"]; lo=cur["low"]; atr=cur["atr"]
        to_close=[]
        for idx,pos in enumerate(self.open_positions):
            life=i-pos["entry_index"]
            if life>pos["timeout"]:
                if pos.get("live"):
                    if pos["qty_rem"]>0:
                        side_close = live_broker.close_side_for(pos["side"])
                        await live_broker.reduce_only(side_close, pos["qty_rem"])
                    to_close.append(idx); continue
                else:
                    r=self._r_from_close(pos, cur["close"]); await self._paper_close(idx,r); to_close.append(idx); continue

            if pos["side"]=="LONG":
                if lo<=pos["stop"]:
                    if pos.get("live"):
                        to_close.append(idx); continue
                    else:
                        r=-1.0 if not pos["hit_pt1"] else -0.5; await self._paper_close(idx,r); to_close.append(idx); continue

                if (not pos["hit_pt1"]) and hi>=pos["pt1"]:
                    if pos.get("live"):
                        qty_pt1 = pos["qty_total"]*PT1_SHARE
                        if qty_pt1>0:
                            await live_broker.reduce_only(live_broker.close_side_for("LONG"), qty_pt1)
                            pos["qty_rem"] = max(0.0, pos["qty_rem"] - qty_pt1)
                    pos["hit_pt1"]=True; pos["lockedR"] += PT1_SHARE*1.0; pos["trail"]=max(pos["trail"], cur["close"]-TRAIL_ATR_MULT*atr)

                if pos["hit_pt1"]:
                    if lo<=pos["trail"]:
                        if pos.get("live"):
                            if pos["qty_rem"]>0:
                                await live_broker.reduce_only(live_broker.close_side_for("LONG"), pos["qty_rem"])
                            to_close.append(idx); continue
                        else:
                            r=pos["lockedR"]; await self._paper_close(idx,r); to_close.append(idx); continue
                    if hi>=pos["target"]:
                        if pos.get("live"):
                            if pos["qty_rem"]>0:
                                await live_broker.reduce_only(live_broker.close_side_for("LONG"), pos["qty_rem"])
                            to_close.append(idx); continue
                        else:
                            r=pos["lockedR"]+PT1_SHARE*RR; await self._paper_close(idx,r); to_close.append(idx); continue
                    pos["trail"]=max(pos["trail"], cur["close"]-TRAIL_ATR_MULT*atr)

            else:
                if hi>=pos["stop"]:
                    if pos.get("live"):
                        to_close.append(idx); continue
                    else:
                        r=-1.0 if not pos["hit_pt1"] else -0.5; await self._paper_close(idx,r); to_close.append(idx); continue

                if (not pos["hit_pt1"]) and lo<=pos["pt1"]:
                    if pos.get("live"):
                        qty_pt1 = pos["qty_total"]*PT1_SHARE
                        if qty_pt1>0:
                            await live_broker.reduce_only(live_broker.close_side_for("SHORT"), qty_pt1)
                            pos["qty_rem"] = max(0.0, pos["qty_rem"] - qty_pt1)
                    pos["hit_pt1"]=True; pos["lockedR"] += PT1_SHARE*1.0; pos["trail"]=min(pos["trail"], cur["close"]+TRAIL_ATR_MULT*atr)

                if pos["hit_pt1"]:
                    if hi>=pos["trail"]:
                        if pos.get("live"):
                            if pos["qty_rem"]>0:
                                await live_broker.reduce_only(live_broker.close_side_for("SHORT"), pos["qty_rem"])
                            to_close.append(idx); continue
                        else:
                            r=pos["lockedR"]; await self._paper_close(idx,r); to_close.append(idx); continue
                    if lo<=pos["target"]:
                        if pos.get("live"):
                            if pos["qty_rem"]>0:
                                await live_broker.reduce_only(live_broker.close_side_for("SHORT"), pos["qty_rem"])
                            to_close.append(idx); continue
                        else:
                            r=pos["lockedR"]+PT1_SHARE*RR; await self._paper_close(idx,r); to_close.append(idx); continue
                    pos["trail"]=min(pos["trail"], cur["close"]+TRAIL_ATR_MULT*atr)

        for j in sorted(to_close, reverse=True):
            self.open_positions.pop(j)

    def _r_from_close(self, pos, close):
        risk=abs(pos["entry"]-pos["stop"])
        return (close-pos["entry"])/risk if pos["side"]=="LONG" else (pos["entry"]-close)/risk

    async def _paper_close(self, idx, R):
        pos=self.open_positions[idx]; risk_d=0.01*(self.equity if self.equity is not None else 1000.0)
        pnl=R*risk_d; self.equity=(self.equity if self.equity is not None else 1000.0)+pnl
        await notify_exit(f"[PAPER] Exit {pos['side']} R={R:.2f} PnL=${pnl:.2f} Equity=${self.equity:.2f}")
