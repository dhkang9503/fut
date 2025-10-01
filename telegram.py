import httpx
from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

async def notify(text:str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"HTML"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url,json=payload)
    except Exception:
        pass
