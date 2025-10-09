import pandas as pd
import numpy as np
from typing import Dict
from .metrics import compute_equity_curve, max_drawdown, sharpe_ratio

class SimpleBacktester:
    def __init__(self, take_profit=0.007, stop_loss=0.004, fee_bps=6, max_hold_bars=150,
                 min_confidence=0.55, cooldown_bars=10):
        self.tp = take_profit
        self.sl = stop_loss
        self.fee = fee_bps / 10000.0
        self.max_hold = max_hold_bars
        self.min_conf = min_confidence
        self.cooldown = cooldown_bars

    def run(self, df: pd.DataFrame, preds: np.ndarray, probas: np.ndarray = None, price_col: str = "close") -> Dict:
        """Run a naive one-position-at-a-time backtest following discrete predictions.
           preds: 1 = long, -1 = short, 0 = flat
           probas: predicted probability for chosen class (max softmax). If provided, we gate by min_conf.
        """
        price = df[price_col].values
        times = df["timestamp"].values
        n = len(df)

        position = 0
        entry_price = None
        entry_idx = None
        trades = []
        cooldown_counter = 0

        for i in range(n):
            signal = int(preds[i])
            conf_ok = True
            if probas is not None:
                conf_ok = probas[i] >= self.min_conf

            # decrement cooldown
            if cooldown_counter > 0:
                cooldown_counter -= 1

            # manage open position
            if position != 0:
                hold = i - entry_idx
                if position == 1:
                    if price[i] >= entry_price * (1 + self.tp) or price[i] <= entry_price * (1 - self.sl) or hold >= self.max_hold:
                        pnl = (price[i]/entry_price - 1) - 2*self.fee
                        trades.append({"dir":"long","entry":entry_price,"exit":price[i],
                                       "pnl":pnl,"entry_time":times[entry_idx],"exit_time":times[i]})
                        position = 0; entry_price=None; entry_idx=None; cooldown_counter = self.cooldown
                else:
                    if price[i] <= entry_price * (1 - self.tp) or price[i] >= entry_price * (1 + self.sl) or hold >= self.max_hold:
                        pnl = (entry_price/price[i] - 1) - 2*self.fee
                        trades.append({"dir":"short","entry":entry_price,"exit":price[i],
                                       "pnl":pnl,"entry_time":times[entry_idx],"exit_time":times[i]})
                        position = 0; entry_price=None; entry_idx=None; cooldown_counter = self.cooldown

            # open new position if flat, not cooling down, signal !=0 and passes confidence gate
            if position == 0 and cooldown_counter == 0 and signal != 0 and conf_ok:
                position = signal
                entry_price = price[i]
                entry_idx = i

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            return {"trades": trades_df, "equity": None,
                    "win_rate": 0.0, "avg_pnl": 0.0, "sharpe": 0.0, "mdd": 0.0, "final_return": 0.0}

        win_rate = float((trades_df["pnl"] > 0).mean())
        avg_pnl = float(trades_df["pnl"].mean())

        eq = compute_equity_curve(trades, initial_balance=10000.0)
        mdd = float(max_drawdown(eq["equity"]))
        total_return = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1)
        sharpe = float(sharpe_ratio(trades_df["pnl"].values))

        return {"trades": trades_df, "equity": eq, "win_rate": win_rate,
                "avg_pnl": avg_pnl, "sharpe": sharpe, "mdd": mdd, "final_return": total_return}
