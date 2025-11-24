import json
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

STATE_PATH = "/app/bot_state.json"

app = FastAPI()

# CORS (모바일 브라우저 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /static → /app/dashboard 폴더 매핑 (dashboard.js 등)
app.mount("/static", StaticFiles(directory="dashboard"), name="static")


# 루트 요청시 dashboard.html 반환
@app.get("/")
async def index():
    return FileResponse("dashboard/dashboard.html")


# 상태 조회용 REST (디버깅/확인용)
@app.get("/state")
def get_state():
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"error": "state_not_found"}


# WebSocket → 대시보드에 실시간 state push
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        try:
            with open(STATE_PATH, "r") as f:
                state = json.load(f)
        except Exception:
            state = {"error": "state_not_found"}

        await ws.send_json(state)
        await asyncio.sleep(1)
