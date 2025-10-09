import numpy as np
import pandas as pd

def compute_equity_curve(trades: list, initial_balance: float = 10000.0):
    balance = initial_balance
    curve = []
    for t in trades:
        balance *= (1 + t["pnl"])
        curve.append({"timestamp": t["exit_time"], "equity": balance})
    return pd.DataFrame(curve)

def max_drawdown(series: pd.Series):
    cummax = series.cummax()
    dd = series / cummax - 1.0
    return dd.min()

def sharpe_ratio(returns: np.ndarray, eps=1e-9):
    if len(returns) < 2:
        return 0.0
    return float(np.mean(returns) / (np.std(returns) + eps)) * np.sqrt(252)
