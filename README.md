# OKX Signal Generator – Incremental Version

This version supports true incremental updates (only fetches new candles).
- Keeps a fixed window size (e.g. 5m=6000, 15m=4000)
- Oldest candles are dropped automatically
- Loads pre-trained model.pkl (force_load_model=true)
- Real-time OKX price via fetch_ticker()

Run:
```bash
pip install ccxt pandas numpy ta lightgbm scikit-learn joblib pyyaml
python okx_signal_generator_live_incremental.py
```
