#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
대시보드 백엔드 서버

- /           : dashboard/dashboard.html 반환
- /static/... : dashboard/ 디렉토리 정적 파일
- /state      : bot_state.json을 1회 반환 (디버그용)
- /ws         : WebSocket. 1초마다 bot_state.json 읽어서 클라이언트로 push

여기서 Bollinger Bands + CCI 를 계산해서
state["ohlcv"][심볼][i] 에
  bb_mid, bb_upper, bb_lower, cci
를 넣어 프론트로 전달한다.
"""

import asyncio
import json
from math import sqrt
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# --------------------
# 설정
# --------------------

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BASE_DIR / "dashboard"
BOT_STATE_PATH = BASE_DIR / "bot_state.json"

# 인디케이터 파라미터 (트레이딩 봇과 맞춰줌)
CCI_PERIOD = 14
BB_PERIOD = 20
BB_K = 2.0

app = FastAPI()

# /static 으로 dashboard 디렉토리 서빙
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")


# --------------------
# 유틸 함수
# --------------------

def read_raw_state() -> Dict[str, Any]:
    """
    bot_state.json 읽어서 dict로 반환.
    파일이 없거나 파싱 실패하면 {} 반환.
    """
    if not BOT_STATE_PATH.exists():
        return {}
    try:
        with BOT_STATE_PATH.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def compute_cci(tp: List[float], period: int) -> List[Optional[float]]:
    """
    단순 Python 리스트 기반 CCI 계산
    tp: typical price 배열
    period: 윈도우 길이
    반환: len(tp) 와 같은 길이의 CCI 리스트 (초기 구간은 None)
    """
    n = period
    res: List[Optional[float]] = [None] * len(tp)
    if len(tp) < n:
        return res

    for i in range(n - 1, len(tp)):
        window = tp[i - n + 1 : i + 1]
        mean = sum(window) / n
        mad = sum(abs(x - mean) for x in window) / n
        if mad == 0:
            res[i] = None
        else:
            res[i] = (tp[i] - mean) / (0.015 * mad)
    return res


def compute_bb(closes: List[float], period: int, k: float) -> Dict[str, List[Optional[float]]]:
    """
    Bollinger Bands 계산
    반환: {"mid": [...], "upper": [...], "lower": [...]}
    """
    n = period
    mid: List[Optional[float]] = [None] * len(closes)
    upper: List[Optional[float]] = [None] * len(closes)
    lower: List[Optional[float]] = [None] * len(closes)

    if len(closes) < n:
        return {"mid": mid, "upper": upper, "lower": lower}

    for i in range(n - 1, len(closes)):
        window = closes[i - n + 1 : i + 1]
        mean = sum(window) / n
        var = sum((x - mean) ** 2 for x in window) / n
        std = sqrt(var)

        mid[i] = mean
        upper[i] = mean + k * std
        lower[i] = mean - k * std

    return {"mid": mid, "upper": upper, "lower": lower}


def add_indicators_to_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    state["ohlcv"][심볼] 의 각 캔들에
      bb_mid, bb_upper, bb_lower, cci
    필드를 추가해서 반환.
    원본 state를 in-place 로 수정한 뒤 그대로 리턴한다.
    """
    ohlcv_all = state.get("ohlcv")
    if not isinstance(ohlcv_all, dict):
        return state

    for sym, candles in ohlcv_all.items():
        if not isinstance(candles, list) or len(candles) == 0:
            continue

        # 값이 없는 캔들 걸러내지 않고 그대로 두되,
        # 계산 가능한 것만 계산해서 채워준다.
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []
        tps: List[float] = []

        for c in candles:
            try:
                h = float(c["high"])
                l = float(c["low"])
                cl = float(c["close"])
            except Exception:
                h = l = cl = float("nan")
            highs.append(h)
            lows.append(l)
            closes.append(cl)
            tps.append((h + l + cl) / 3.0)

        # CCI / BB 계산
        cci_vals = compute_cci(tps, CCI_PERIOD)
        bb = compute_bb(closes, BB_PERIOD, BB_K)

        # 다시 캔들 dict에 넣어준다.
        for i, c in enumerate(candles):
            cci_val = cci_vals[i]
            mid_val = bb["mid"][i]
            up_val = bb["upper"][i]
            low_val = bb["lower"][i]

            c["cci"] = float(cci_val) if cci_val is not None else None
            c["bb_mid"] = float(mid_val) if mid_val is not None else None
            c["bb_upper"] = float(up_val) if up_val is not None else None
            c["bb_lower"] = float(low_val) if low_val is not None else None

    return state


# --------------------
# 라우팅
# --------------------

@app.get("/", response_class=HTMLResponse)
async def get_dashboard() -> HTMLResponse:
    """
    메인 대시보드 HTML 반환
    """
    html_path = DASHBOARD_DIR / "dashboard.html"
    return FileResponse(html_path)


@app.get("/state")
async def get_state_once() -> Dict[str, Any]:
    """
    디버그용: 현재 상태를 한 번만 반환
    (인디케이터 포함)
    """
    raw = read_raw_state()
    state = add_indicators_to_state(raw)
    return state


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket 엔드포인트
    - 1초마다 bot_state.json 읽어서
      인디케이터를 붙인 후 클라이언트로 전송
    """
    await ws.accept()
    try:
        while True:
            raw = read_raw_state()
            state = add_indicators_to_state(raw)
            await ws.send_text(json.dumps(state))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        # 클라이언트가 정상적으로 연결을 끊은 경우
        return
    except Exception:
        # 기타 예외는 조용히 종료 (로그는 uvicorn이 찍어줌)
        return
