import os
from dotenv import load_dotenv
load_dotenv()

SYMBOL = os.getenv("SYMBOL","BTCUSDT")
PRODUCT_TYPE = os.getenv("PRODUCT_TYPE","umcbl")
TIMEFRAME = os.getenv("TIMEFRAME","5m")
MODE = os.getenv("MODE","paper")

BITGET_API_KEY=os.getenv("BITGET_API_KEY","")
BITGET_API_SECRET=os.getenv("BITGET_API_SECRET","")
BITGET_API_PASSPHRASE=os.getenv("BITGET_API_PASSPHRASE","")

TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN","")
TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID","")

RISK_PCT=float(os.getenv("RISK_PCT","0.01"))
MARGIN_PCT=float(os.getenv("MARGIN_PCT","0.10"))
DB_PATH=os.getenv("DB_PATH","bot.db")

FVG_MIN_ATR=0.5
OB_MIN_ATR=0.2
OB_MAX_ATR=2.0
VOL_SPIKE=0.2
RR=2.0
TIMEOUT_BARS=48

PT1_SHARE=0.5
TRAIL_ATR_MULT=0.5

SWING_K=2
ATR_LEN=14

REST_BASE="https://api.bitget.com"
WS_PUBLIC = "wss://ws.bitget.com/v2/ws/public"  # V2 public

SYMBOL_INFO={"BTCUSDT":{"price_tick":0.1,"size_step":0.001,"min_size":0.001,"max_leverage":100}}
