#!/usr/bin/env python3
"""
Bitget AutoTrader — Single File / No State Files / Telegram Alerts
Strategy: 1h Bollinger(20,2) + CCI(20)
Entry:  Long  if bb_low < close < bb_mid and CCI>-100
        Short if close < bb_mid and CCI<+100  (but NOT when close>=bb_mid & CCI>=+100)
Exit:   Server-side SL = 2%
        TP1 = +4% (close 50%, then move SL to BE(+fees) = entry ± 2*fee)
        TP2 = +6% (close remainder)
Risk:   Daily hard lock at 4% drawdown from day_start (no new entries that day)
Run:    Loop; trade only at hh:ENTRY_BUFFER_MIN (default :02) UTC
"""

import os, time, math, json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, List

# ========================= ENV =========================
BITGET_API_KEY      = os.getenv("BITGET_API_KEY", "")
BITGET_API_SECRET   = os.getenv("BITGET_API_SECRET", "")
BITGET_API_PASSWORD = os.getenv("BITGET_API_PASSWORD", "")
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")

SIZING_MODE  = os.getenv("SIZING_MODE", "percent").lower()   # 'percent' or 'fixed'
SIZING_VALUE = float(os.getenv("SIZING_VALUE", "0.10"))      # 10% equity or fixed USDT
FEE_RATE     = float(os.getenv("FEE_RATE", "0.0008"))        # per side, 0.08%
DAILY_STOP   = float(os.getenv("DAILY_STOP", "0.04"))        # 4%
ENTRY_BUFFER_MIN = int(os.getenv("ENTRY_BUFFER_MIN", "2"))   # run at HH:02 UTC

# Symbols & leverage
SYMBOLS = {
    "BTCUSDT": {"lev": 100, "cc": "BTC/USDT:USDT"},
    "SOLUSDT": {"lev": 60,  "cc": "SOL/USDT:USDT"},
}

# Strategy params
INTERVAL   = "1h"
BB_PERIOD  = 20
BB_NSTD    = 2.0
CCI_PERIOD = 20
SL_PCT   = 0.02
TP1_PCT  = 0.04
TP2_PCT  = 0.06

# ========================= Utils / Notify =========================
def now_utc():
    return datetime.now(timezone.utc)

def notify(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[INFO]", text)
        return
    import requests
    TG_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    try:
        requests.post(f"{TG_BASE}/sendMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
                      timeout=10)
    except Exception as e:
        print(f"[CRITICAL][TG] send error: {e} | msg={text[:120]}")

def pct(a, b):
    return 0.0 if b == 0 else (a / b - 1.0)

# ========================= Indicators =========================
def indicators_from_ohlcv(ohlcv: List[List[float]]):
    """
    ohlcv: list of [ts_ms, open, high, low, close, volume]
    returns list of dict with keys: time, open, high, low, close, bb_mid, bb_upper, bb_lower, cci20
    """
    from statistics import mean
    out = []
    closes = []
    highs = []
    lows = []
    tps = []
    for i, (t,o,h,l,c,v) in enumerate(ohlcv):
        closes.append(float(c)); highs.append(float(h)); lows.append(float(l))
        tps.append((float(h)+float(l)+float(c))/3.0)
        # BB
        if len(closes) >= BB_PERIOD:
            window = closes[-BB_PERIOD:]
            m = sum(window)/BB_PERIOD
            var = sum((x-m)**2 for x in window)/BB_PERIOD
            sd = var**0.5
            bb_mid = m; bb_up = m + BB_NSTD*sd; bb_low = m - BB_NSTD*sd
        else:
            bb_mid = bb_up = bb_low = float("nan")
        # CCI
        if len(tps) >= CCI_PERIOD:
            w = tps[-CCI_PERIOD:]
            m = sum(w)/CCI_PERIOD
            mad = sum(abs(x-m) for x in w)/CCI_PERIOD
            cci = (tps[-1]-m)/(0.015*mad) if mad>0 else 0.0
        else:
            cci = float("nan")
        if not (math.isnan(bb_mid) or math.isnan(cci)):
            out.append({
                "time": datetime.fromtimestamp(t/1000, tz=timezone.utc),
                "open": float(o), "high": float(h), "low": float(l), "close": float(c),
                "bb_mid": bb_mid, "bb_upper": bb_up, "bb_lower": bb_low, "cci20": cci
            })
    return out

def long_signal(r):   # bb_low < close < bb_mid & cci>-100
    return (r["close"] < r["bb_mid"]) and (r["close"] > r["bb_lower"]) and (r["cci20"] > -100)

def short_signal(r):  # close<bb_mid & cci<+100, but forbid (close>=bb_mid & cci>=+100)
    if r["close"] >= r["bb_mid"] and r["cci20"] >= 100:
        return False
    return (r["close"] < r["bb_mid"]) and (r["cci20"] < 100)

# ========================= CCXT / Bitget =========================
class Bitget:
    def __init__(self):
        import ccxt
        self.x = ccxt.bitget({
            "apiKey": BITGET_API_KEY,
            "secret": BITGET_API_SECRET,
            "password": BITGET_API_PASSWORD,
            "enableRateLimit": True,
            "options": {"defaultType": "swap", "defaultSubType": "linear"},
        })
        # set leverage
        for sym, meta in SYMBOLS.items():
            try:
                self.x.setLeverage(meta["lev"], meta["cc"], params={"marginMode":"cross"})
            except Exception:
                pass

    def fetch_equity(self) -> float:
        try:
            bal = self.x.fetch_balance(params={"type":"swap"})
            usdt = bal.get("USDT") or {}
            total = float(usdt.get("total") or usdt.get("free") or 0.0)
            return total
        except Exception as e:
            notify(f"❌ fetch_equity error: {e}")
            return 0.0

    def fetch_ohlcv(self, cc_symbol: str, timeframe="1h", limit=300):
        try:
            return self.x.fetch_ohlcv(cc_symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            notify(f"❌ fetch_ohlcv error {cc_symbol}: {e}")
            return []

    def _amount_from_notional(self, cc_symbol: str, notional: float, price_hint: Optional[float]=None) -> float:
        px = price_hint
        if not px or px <= 0:
            try:
                t = self.x.fetch_ticker(cc_symbol)
                px = float(t.get("last"))
            except Exception:
                pass
        if not px or px <= 0:
            raise RuntimeError("가격 조회 실패")
        amt = notional / px
        try:
            m = self.x.market(cc_symbol)
            prec = m.get("precision", {}).get("amount")
            if prec is not None:
                step = 10 ** (-prec)
                amt = math.floor(amt / step) * step
        except Exception:
            pass
        return max(amt, 0.0)

    def market_open(self, cc_symbol: str, side: str, notional: float) -> Dict:
        amt = self._amount_from_notional(cc_symbol, notional)
        ord_side = "buy" if side == "long" else "sell"
        o = self.x.create_order(cc_symbol, type="market", side=ord_side, amount=amt, params={"reduceOnly": False})
        price = float(o.get("average") or o.get("price") or 0.0)
        if price <= 0:
            t = self.x.fetch_ticker(cc_symbol); price = float(t.get("last"))
        return {"amount": amt, "price": price, "raw": o}

    def place_stop_loss(self, cc_symbol: str, side: str, stop_price: float, notional: float) -> Optional[str]:
        amt = self._amount_from_notional(cc_symbol, notional, price_hint=stop_price)
        reduce_side = "sell" if side=="long" else "buy"
        params_list = [
            {"reduceOnly": True, "stopPrice": stop_price},
            {"reduceOnly": True, "triggerPrice": stop_price},
            {"reduceOnly": True, "stopLossPrice": stop_price},
        ]
        last_err = None
        for p in params_list:
            try:
                o = self.x.create_order(cc_symbol, type="market", side=reduce_side, amount=amt, params=p)
                return o.get("id") if isinstance(o, dict) else getattr(o, "id", None)
            except Exception as e:
                last_err = e
                continue
        notify(f"⚠️ SL order failed ({cc_symbol}) : {last_err}")
        return None

    def cancel_order(self, cc_symbol: str, order_id: str):
        try:
            self.x.cancel_order(order_id, cc_symbol)
        except Exception as e:
            notify(f"⚠️ cancel_order fail {cc_symbol}:{order_id}: {e}")

# ========================= Runtime (no persisted state) =========================
@dataclass
class Position:
    sym: str
    side: str
    entry: float
    notional: float
    remaining_frac: float = 1.0
    tp1_hit: bool = False
    sl_price: float = 0.0
    sl_order_id: Optional[str] = None

class Engine:
    def __init__(self, ex: Bitget):
        self.ex = ex
        self.equity = 0.0
        self.day_start_equity = 0.0
        self.current_day = None
        self.hard_paused = False
        self.last_traded_hour = None
        self.pos: Optional[Position] = None

    # --------- Sizing ---------
    def compute_margin(self) -> float:
        if SIZING_MODE == "percent":
            return max(0.0, self.equity * SIZING_VALUE)
        else:
            return max(0.0, float(SIZING_VALUE))

    # --------- Core Tick ---------
    def tick(self):
        # 1) sync equity
        self.equity = self.ex.fetch_equity()
        # 2) roll day
        d = now_utc().date()
        if (self.current_day is None) or (self.current_day != d):
            self.current_day = d
            self.day_start_equity = self.equity
            self.hard_paused = False

        # 3) daily hard lock
        if self.day_start_equity > 0 and (self.day_start_equity - self.equity) / self.day_start_equity >= DAILY_STOP:
            if not self.hard_paused:
                self.hard_paused = True
                notify(f"🧯 Daily hard lock {int(DAILY_STOP*100)}% reached. New entries disabled for UTC {d}.")
            return

        # 4) manage existing position
        if self.pos:
            cc = SYMBOLS[self.pos.sym]["cc"]
            ohl = self.ex.fetch_ohlcv(cc, timeframe=INTERVAL, limit=1)
            if ohl:
                _, o, h, l, c, v = ohl[-1]
                side = self.pos.side
                entry = self.pos.entry
                notional = self.pos.notional
                rem = self.pos.remaining_frac

                tp1 = entry*(1+TP1_PCT if side=='long' else 1-TP1_PCT)
                sl  = self.pos.sl_price if self.pos.sl_price>0 else entry*(1-SL_PCT if side=='long' else 1+SL_PCT)
                tp2 = entry*(1+TP2_PCT if side=='long' else 1-TP2_PCT) if self.pos.tp1_hit else None

                # Priority: SL -> TP1 -> TP2
                hit = None
                if (l <= sl and side=='long') or (h >= sl and side=='short'):
                    hit = "SL"
                elif ((h>=tp1 and side=='long') or (l<=tp1 and side=='short')):
                    hit = "TP1" if not self.pos.tp1_hit else ("TP2" if tp2 and ((h>=tp2 and side=='long') or (l<=tp2 and side=='short')) else "TP1")
                elif tp2 and ((h>=tp2 and side=='long') or (l<=tp2 and side=='short')):
                    hit = "TP2"

                if hit == "SL":
                    # close remainder at SL (reduceOnly market)
                    self._close_market(entry, sl, rem, reason="SL")
                    self.pos = None
                    return

                if hit == "TP1" and not self.pos.tp1_hit:
                    # close 50% at tp1
                    self._close_market(entry, tp1, 0.5, reason="TP1")
                    self.pos.remaining_frac = 0.5
                    self.pos.tp1_hit = True
                    # move SL -> BE(+fees)
                    be_adj = 2.0 * FEE_RATE
                    new_stop = entry*(1+be_adj) if side=='long' else entry*(1-be_adj)
                    # cancel prev SL & place new
                    if self.pos.sl_order_id:
                        self.ex.cancel_order(cc, self.pos.sl_order_id)
                    self.pos.sl_order_id = self.ex.place_stop_loss(cc, side, new_stop, notional*self.pos.remaining_frac)
                    self.pos.sl_price = new_stop
                    notify(f"🔧 Move SL to BE(+fees) {self.pos.sym} {side} @ {new_stop:.2f}")
                    return

                if hit == "TP2":
                    self._close_market(entry, tp2, rem, reason="TP2")
                    if self.pos.sl_order_id:
                        self.ex.cancel_order(cc, self.pos.sl_order_id)
                    self.pos = None
                    return

        # 5) entries
        if self.pos or self.hard_paused:
            return

        # BTC -> SOL 순서로 검사
        for sym in ["BTCUSDT", "SOLUSDT"]:
            cc = SYMBOLS[sym]["cc"]
            lev = SYMBOLS[sym]["lev"]
            ohl = self.ex.fetch_ohlcv(cc, timeframe=INTERVAL, limit=300)
            if len(ohl) < 50:
                continue
            rows = indicators_from_ohlcv(ohl)
            r = rows[-1]
            if (r["close"]>=r["bb_mid"] and r["cci20"]>=100):
                continue
            go_long = long_signal(r)
            go_short = (not go_long) and short_signal(r)
            if not (go_long or go_short):
                continue

            margin = self.compute_margin()
            if margin <= 0:
                notify("⚠️ margin=0: check SIZING_MODE/SIZING_VALUE or equity")
                break
            notional = margin * lev
            # open position
            res = self.ex.market_open(cc, "long" if go_long else "short", notional)
            entry_price = res["price"]
            # place SL server-side
            stop_price = entry_price*(1-SL_PCT) if go_long else entry_price*(1+SL_PCT)
            sl_id = self.ex.place_stop_loss(cc, "long" if go_long else "short", stop_price, notional)
            self.pos = Position(sym=sym, side="long" if go_long else "short",
                                entry=entry_price, notional=notional,
                                remaining_frac=1.0, tp1_hit=False,
                                sl_price=stop_price, sl_order_id=sl_id)
            notify(f"🟢 OPEN {sym} {self.pos.side} @ {entry_price:.2f} | lev {lev}x | SL {stop_price:.2f}")
            break

    # --------- Helpers ---------
    def _close_market(self, entry: float, px: float, frac: float, reason: str):
        """Simulate PnL / fees for notification only; actual exit is through reduce-only market."""
        side = self.pos.side
        cc = SYMBOLS[self.pos.sym]["cc"]
        notional = self.pos.notional * frac
        # place reduce-only market
        try:
            self.ex.market_open(cc, "short" if side=="long" else "long", notional)
        except Exception as e:
            notify(f"❌ close market error ({reason}) {self.pos.sym}: {e}")
        pnl = notional * ( (px/entry - 1) if side=="long" else (1 - px/entry) )
        fee = notional * FEE_RATE
        self.equity += pnl - fee
        notify(f"🔴 {reason} {self.pos.sym} {side} @ {px:.2f} | pnl(after fee) {pnl-fee:.2f} | equity≈{self.equity:.2f}")

# ========================= Main Loop =========================
def main():
    # Checks
    if not (BITGET_API_KEY and BITGET_API_SECRET and BITGET_API_PASSWORD):
        raise SystemExit("Set BITGET_API_KEY/BITGET_API_SECRET/BITGET_API_PASSWORD")
    ex = Bitget()
    eng = Engine(ex)

    notify("🤖 Bitget AutoTrader started. Strategy=BB(20,2)+CCI(20). SL server-side; TP1/TP2 logic; daily hard lock.")

    while True:
        try:
            now = now_utc()
            # trade only once per hour at HH:ENTRY_BUFFER_MIN
            if now.minute == ENTRY_BUFFER_MIN:
                if eng.last_traded_hour != now.hour:
                    eng.last_traded_hour = now.hour
                    eng.tick()
            else:
                # manage open position also on non-trade minutes? -> keep hourly only per spec
                pass
            time.sleep(2)
        except Exception as e:
            print(f"[CRITICAL] main loop error: {e}")
            notify(f"❌ main loop error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
