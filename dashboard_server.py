import json
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

STATE_PATH = "/app/bot_state.json"

app = FastAPI()

# CORS 허용 (폰 브라우저 접근용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 /static 경로로 정적 파일 서빙 (JS 등)
#    /static/dashboard.js → /app/dashboard/dashboard.js
app.mount("/static", StaticFiles(directory="dashboard"), name="static")


# 🔹 /  요청 들어오면 dashboard.html 그대로 반환
@app.get("/")
async def index():
    return FileResponse("dashboard/dashboard.html")


# 🔹 상태 조회용 REST
@app.get("/state")
def get_state():
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"error": "state_not_found"}


# 🔹 WebSocket (대시보드 실시간 업데이트)
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
