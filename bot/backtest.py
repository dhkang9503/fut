import argparse, os, pandas as pd, numpy as np, json
from bot.indicators import prepare_ohlcv
from bot.strategy import generate_signal_row
from bot.config import TIMEOUT_BARS, RR, PT1_SHARE, TRAIL_ATR_MULT

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize time
    if np.issubdtype(df["time"].dtype, np.number):
        t = df["time"].astype(np.int64)
        df["time"] = (t // 1000).astype(np.int64) if (t > 1e12).any() else t.astype(np.int64)
    else:
        df["time"] = pd.to_datetime(df["time"]).astype("int64") // 10**9
    df = df.sort_values("time").reset_index(drop=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df

def paper_sim(df: pd.DataFrame):
    df_feat = prepare_ohlcv(df.copy())
    trades = []; open_positions = []; equity = 1000.0

    def position_size(entry, stop, equity_now):
        risk_d = 0.01 * equity_now
        risk = abs(entry - stop)
        if risk <= 0: return 0.0, risk_d
        qty = risk_d / risk  # linear contracts approximation
        return qty, risk_d

    for i in range(120, len(df_feat)-1):
        # manage open positions on bar i
        cur = df_feat.iloc[i]
        hi, lo, close, atr = cur["high"], cur["low"], cur["close"], cur["atr"]
        to_close = []
        for idx, pos in enumerate(open_positions):
            life = i - pos["entry_idx"]
            if life > TIMEOUT_BARS:
                r = (close - pos["entry"])/abs(pos["entry"]-pos["stop"]) if pos["side"]=="LONG" else (pos["entry"]-close)/abs(pos["entry"]-pos["stop"])
                pnl = r * pos["risk$"]; equity += pnl
                trades.append({"entry_time": pos["entry_time"], "exit_time": int(df_feat.iloc[i]["time"]), "side": pos["side"], "R": r, "pnl$": pnl})
                to_close.append(idx); continue
            if pos["side"]=="LONG":
                if lo <= pos["stop"]:
                    r = -1.0 if not pos["hit_pt1"] else -0.5
                    pnl = r * pos["risk$"]; equity += pnl
                    trades.append({"entry_time": pos["entry_time"], "exit_time": int(df_feat.iloc[i]["time"]), "side":"LONG","R":r,"pnl$":pnl})
                    to_close.append(idx); continue
                if (not pos["hit_pt1"]) and hi >= pos["pt1"]:
                    pos["hit_pt1"] = True
                    pos["lockedR"] = pos.get("lockedR", 0.0) + PT1_SHARE * 1.0
                    pos["trail"] = max(pos["trail"], close - TRAIL_ATR_MULT * atr)
                if pos["hit_pt1"]:
                    if lo <= pos["trail"]:
                        r = pos.get("lockedR", 0.0)
                        pnl = r * pos["risk$"]; equity += pnl
                        trades.append({"entry_time": pos["entry_time"], "exit_time": int(df_feat.iloc[i]["time"]), "side":"LONG","R":r,"pnl$":pnl})
                        to_close.append(idx); continue
                    if hi >= pos["target"]:
                        r = pos.get("lockedR", 0.0) + PT1_SHARE * RR
                        pnl = r * pos["risk$"]; equity += pnl
                        trades.append({"entry_time": pos["entry_time"], "exit_time": int(df_feat.iloc[i]["time"]), "side":"LONG","R":r,"pnl$":pnl})
                        to_close.append(idx); continue
                    pos["trail"] = max(pos["trail"], close - TRAIL_ATR_MULT * atr)
            else:
                if hi >= pos["stop"]:
                    r = -1.0 if not pos["hit_pt1"] else -0.5
                    pnl = r * pos["risk$"]; equity += pnl
                    trades.append({"entry_time": pos["entry_time"], "exit_time": int(df_feat.iloc[i]["time"]), "side":"SHORT","R":r,"pnl$":pnl})
                    to_close.append(idx); continue
                if (not pos["hit_pt1"]) and lo <= pos["pt1"]:
                    pos["hit_pt1"] = True
                    pos["lockedR"] = pos.get("lockedR", 0.0) + PT1_SHARE * 1.0
                    pos["trail"] = min(pos["trail"], close + TRAIL_ATR_MULT * atr) if pos["trail"] is not None else close + TRAIL_ATR_MULT * atr
                if pos["hit_pt1"]:
                    if hi >= pos["trail"]:
                        r = pos.get("lockedR", 0.0)
                        pnl = r * pos["risk$"]; equity += pnl
                        trades.append({"entry_time": pos["entry_time"], "exit_time": int(df_feat.iloc[i]["time"]), "side":"SHORT","R":r,"pnl$":pnl})
                        to_close.append(idx); continue
                    if lo <= pos["target"]:
                        r = pos.get("lockedR", 0.0) + PT1_SHARE * RR
                        pnl = r * pos["risk$"]; equity += pnl
                        trades.append({"entry_time": pos["entry_time"], "exit_time": int(df_feat.iloc[i]["time"]), "side":"SHORT","R":r,"pnl$":pnl})
                        to_close.append(idx); continue
                    pos["trail"] = min(pos["trail"], close + TRAIL_ATR_MULT * atr) if pos["trail"] is not None else close + TRAIL_ATR_MULT * atr
        for j in sorted(to_close, reverse=True):
            open_positions.pop(j)

        # signal from i-1, entry at i open
        sig = generate_signal_row(df_feat, i-1)
        if sig:
            entry = df_feat.iloc[i]["open"]
            stop  = sig["stop"]
            target= sig["target"]
            pt1   = sig["pt1"]
            side  = sig["side"]
            qty, risk = position_size(entry, stop, equity)
            pos = {"side": side, "entry": entry, "stop": stop, "pt1": pt1, "target": target,
                   "qty": qty, "risk$": risk, "entry_idx": i, "entry_time": int(df_feat.iloc[i]["time"]),
                   "hit_pt1": False, "trail": entry, "lockedR": 0.0}
            open_positions.append(pos)

    tr = pd.DataFrame(trades)
    if len(tr)==0:
        return tr, {"trades":0,"win_rate":0.0,"final_equity":1000.0,"net_profit":0.0,"avg_R":0.0,"cum_R":0.0,"mdd_pct":0.0}
    eq = 1000.0; curve=[]
    for _, r in tr.iterrows():
        eq += r["pnl$"]; curve.append(eq)
    s = pd.Series(curve); mdd = ((s - s.cummax())/s.cummax()).min()*100.0
    wr = float((tr["R"]>0).mean())
    return tr, {"trades": int(len(tr)), "win_rate": wr, "final_equity": float(curve[-1]), "net_profit": float(curve[-1]-1000.0),
                "avg_R": float(tr["R"].mean()), "cum_R": float(tr["R"].sum()), "mdd_pct": float(mdd)}

def main():
    ap = argparse.ArgumentParser(description="Backtest fixed strategy on a 5m CSV (time, open, high, low, close, volume).")
    ap.add_argument("--csv", required=True, help="Path to CSV file")
    ap.add_argument("--outdir", default="backtest_out", help="Output directory")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = load_csv(args.csv)
    trades, stats = paper_sim(df)
    trades_path = os.path.join(args.outdir, "trades.csv"); trades.to_csv(trades_path, index=False)
    stats_path  = os.path.join(args.outdir, "stats.json"); open(stats_path,"w").write(json.dumps(stats, indent=2))
    print("Saved:", trades_path); print("Saved:", stats_path); print("Summary:", stats)

if __name__ == "__main__":
    main()
