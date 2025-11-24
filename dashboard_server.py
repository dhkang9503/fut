# dashboard_server.py
import json
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

STATE_PATH = "/app/bot_state.json"

app = FastAPI()

# CORS allow all (브라우저 접근용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", StaticFiles(directory="dashboard", html=True), name="static")

@app.get("/state")
def get_state():
    """최초 로딩용 JSON"""
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"error": "state_not_found"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        try:
            with open(STATE_PATH, "r") as f:
                state = json.load(f)
        except:
            state = {"error": "state_not_found"}

        await ws.send_json(state)
        await asyncio.sleep(1)
