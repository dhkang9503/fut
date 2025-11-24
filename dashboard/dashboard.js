// =====================
// 기본 설정
// =====================

// 심볼 리스트
const SYMBOLS = ["BTC/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT"];

// 심볼 → canvas ID 매핑 (HTML의 <canvas id="...">와 맞춰야 함)
const CHART_IDS = {
    "BTC/USDT:USDT": "chart-btc",
    "XRP/USDT:USDT": "chart-xrp",
    "DOGE/USDT:USDT": "chart-doge",
};

// 차트 인스턴스 저장
const charts = {};

// DOM 엘리먼트
const equityEl = document.getElementById("equity");
const entryRestrictEl = document.getElementById("entry_restrict");
const posEl = document.getElementById("position");
const logsEl = document.getElementById("logs");

// =====================
// 유틸 함수
// =====================

// 숫자를 소수 n째 자리까지 포맷
function fmtNum(value, digits = 3) {
    if (value === null || value === undefined || isNaN(value)) return "-";
    return Number(value).toFixed(digits);
}

// USDT 표시
function fmtUSDT(value) {
    if (value === null || value === undefined || isNaN(value)) return "-";
    return `${Number(value).toFixed(3)} USDT`;
}

// =====================
// Entry Restriction 출력
// =====================
function renderEntryRestriction(entryRestrict) {
    if (!entryRestrictEl) return;
    if (!entryRestrict) {
        entryRestrictEl.textContent = "-";
        return;
    }

    const lines = SYMBOLS.map(sym => {
        const v = entryRestrict[sym];
        return `${sym}:  ${v ? v : "-"}`;
    });

    entryRestrictEl.textContent = lines.join("\n");
}

// =====================
// 현재 포지션 출력
// =====================
function renderPosition(posState) {
    if (!posEl) return;
    if (!posState) {
        posEl.textContent = "{}";
        return;
    }

    // 예쁘게 포맷해서 보여주기
    const formatted = {};
    for (const sym of SYMBOLS) {
        const p = posState[sym] || {};
        formatted[sym] = {
            side: p.side || null,
            size: p.size || 0,
            entry_price: p.entry_price || null,
            stop_price: p.stop_price || null,
            stop_order_id: p.stop_order_id || null,
            entry_time: p.entry_time || null,
        };
    }

    posEl.textContent = JSON.stringify(formatted, null, 2);
}

// =====================
// 로그 출력 (지금은 last_signal이나 나중에 확장용)
// =====================
function renderLogs(state) {
    if (!logsEl) return;
    const lastSignal = state.last_signal || {};
    if (Object.keys(lastSignal).length === 0) {
        logsEl.textContent = "{}";
        return;
    }
    logsEl.textContent = JSON.stringify(lastSignal, null, 2);
}

// =====================
// Chart.js 캔들 차트 생성/업데이트
// =====================

// Chart.js candlestick가 기대하는 포맷:
// { x: Date(or ms), o: number, h: number, l: number, c: number }

// 백엔드의 ohlcv 한 개를 위 포맷으로 변환
function mapCandleForChart(raw) {
    if (!raw) return null;

    // bot_state.json의 time은 "초 단위 Unix timestamp" 이므로 ms로 변환
    const t = raw.time ? raw.time * 1000 : null;
    if (!t) return null;

    const o = Number(raw.open);
    const h = Number(raw.high);
    const l = Number(raw.low);
    const c = Number(raw.close);

    if ([o, h, l, c].some(v => isNaN(v))) return null;

    return {
        x: new Date(t),
        o,
        h,
        l,
        c,
    };
}

// 차트 초기화
function initChart(symbol) {
    const canvasId = CHART_IDS[symbol];
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.warn("Canvas element not found for", symbol, canvasId);
        return null;
    }

    const ctx = canvas.getContext("2d");

    const chart = new Chart(ctx, {
        type: "candlestick", // chartjs-chart-financial 플러그인 기준
        data: {
            datasets: [
                {
                    label: symbol,
                    data: [],
                    barThickness: 4,
                    barPercentage: 0.6
                },
            ],
        },
        options: {
            parsing: false, // 우리가 직접 {x,o,h,l,c} 포맷으로 넣을 거라서
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: "time",
                    time: {
                        tooltipFormat: "yyyy-MM-dd HH:mm",
                    },
                    ticks: {
                        // 너무 빽빽하면 자동으로 줄여줌
                        source: "auto",
                    },
                },
                y: {
                    position: "right",
                },
            },
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const v = ctx.raw;
                            if (!v) return "";
                            return `O:${v.o} H:${v.h} L:${v.l} C:${v.c}`;
                        },
                    },
                },
            },
        },
    });

    charts[symbol] = chart;
    return chart;
}

// 차트 데이터 갱신
function updateChart(symbol, rawCandles) {
    if (!Array.isArray(rawCandles)) return;

    let chart = charts[symbol];
    if (!chart) {
        chart = initChart(symbol);
        if (!chart) return;
    }

    // rawCandles → {x,o,h,l,c} 배열로 변환
    const mapped = rawCandles
        .map(mapCandleForChart)
        .filter(c => c !== null);

    chart.data.datasets[0].data = mapped;
    chart.update();
}

// =====================
// WebSocket 연결
// =====================

let ws = null;
let reconnectTimer = null;

function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${protocol}://${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log("WebSocket connected:", wsUrl);
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
    };

    ws.onclose = () => {
        console.warn("WebSocket closed. Reconnecting in 3s...");
        if (!reconnectTimer) {
            reconnectTimer = setTimeout(connectWebSocket, 3000);
        }
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        ws.close();
    };

    ws.onmessage = (event) => {
        try {
            const state = JSON.parse(event.data);
            handleStateUpdate(state);
        } catch (e) {
            console.error("Failed to parse WS message:", e);
        }
    };
}

// =====================
// 상태 업데이트 핸들러
// =====================
function handleStateUpdate(state) {
    if (!state) return;

    // 1) Equity
    if (equityEl) {
        equityEl.textContent = fmtUSDT(state.equity);
    }

    // 2) Entry Restriction
    renderEntryRestriction(state.entry_restrict);

    // 3) Position
    renderPosition(state.pos_state);

    // 4) Logs / last_signal
    renderLogs(state);

    // 5) 캔들 차트 (state.ohlcv 사용)
    const ohlcv = state.ohlcv || {};
    for (const sym of SYMBOLS) {
        const candles = ohlcv[sym];
        if (!Array.isArray(candles) || candles.length === 0) {
            // 데이터가 없는 경우, 해당 차트는 그대로 두거나 나중에 클리어할 수 있음
            continue;
        }
        updateChart(sym, candles);
    }
}

// =====================
// 초기 실행
// =====================

window.addEventListener("load", () => {
    // 차트 먼저 초기화 (데이터 없어도 canvas 생성)
    SYMBOLS.forEach(sym => initChart(sym));

    // WebSocket 연결 시작
    connectWebSocket();
});
