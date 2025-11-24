import json
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # ✅ 이미 추가했을 거야

STATE_PATH = "/app/bot_state.json"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ❌ 기존 (문제 있는 버전)
# app.mount("/", StaticFiles(directory="dashboard", html=True), name="static")

# ✅ 새 버전: /dashboard 아래로 정적 파일 서빙
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")


@app.get("/state")
def get_state():
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
