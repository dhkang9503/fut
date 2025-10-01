import os, httpx

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

_ALLOWED = os.getenv(
    "TELEGRAM_NOTIFY_EVENTS",
    "signal,order,fill,exit,pnl,system"
).lower().replace(" ", "").split(",")

def _allowed(event: str) -> bool:
    return (event or "").lower() in _ALLOWED

async def _send_text(text: str):
    if not TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text}
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(url, data=data)

async def notify(text: str, event: str = "signal"):
    """Send Telegram message only if event is whitelisted via TELEGRAM_NOTIFY_EVENTS."""
    if not _allowed(event):
        return
    await _send_text(text)

# Convenience wrappers
async def notify_signal(text: str): return await notify(text, event="signal")
async def notify_order(text: str):  return await notify(text, event="order")
async def notify_fill(text: str):   return await notify(text, event="fill")
async def notify_exit(text: str):   return await notify(text, event="exit")
async def notify_pnl(text: str):    return await notify(text, event="pnl")
# Default-filtered (not sent unless enabled in TELEGRAM_NOTIFY_EVENTS)
async def notify_ws(text: str):     return await notify(text, event="ws")
async def notify_system(text: str): return await notify(text, event="system")
async def notify_debug(text: str):  return await notify(text, event="debug")
