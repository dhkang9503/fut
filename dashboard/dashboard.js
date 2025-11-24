// =====================
// 기본 설정
// =====================

// 심볼 리스트
const SYMBOLS = ["BTC/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT"];

// 캔들 차트 canvas ID
const CHART_IDS = {
    "BTC/USDT:USDT": "chart-btc",
    "XRP/USDT:USDT": "chart-xrp",
    "DOGE/USDT:USDT": "chart-doge",
};

// CCI 차트 canvas ID
const CCI_CHART_IDS = {
    "BTC/USDT:USDT": "chart-btc-cci",
    "XRP/USDT:USDT": "chart-xrp-cci",
    "DOGE/USDT:USDT": "chart-doge-cci",
};

// 차트 인스턴스 저장
const charts = {};
const cciCharts = {};

// DOM 엘리먼트
const equityEl = document.getElementById("equity");
const entryRestrictEl = document.getElementById("entry_restrict");
const posEl = document.getElementById("position");
const logsEl = document.getElementById("logs");

// =====================
// 유틸 함수
// =====================

function fmtNum(value, digits = 3) {
    if (value === null || value === undefined || isNaN(value)) return "-";
    return Number(value).toFixed(digits);
}

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
// 로그 출력
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
// 데이터 매핑
// =====================

// OHLCV + BB + CCI 한 캔들을 {x,o,h,l,c,bb_mid,bb_upper,bb_lower,cci}로 변환
function mapCandleForChart(raw) {
    if (!raw) return null;

    const t = raw.time ? raw.time * 1000 : null;
    if (!t) return null;

    const o = Number(raw.open);
    const h = Number(raw.high);
    const l = Number(raw.low);
    const c = Number(raw.close);

    if ([o, h, l, c].some(v => isNaN(v))) return null;

    const bb_mid = raw.bb_mid !== null && raw.bb_mid !== undefined ? Number(raw.bb_mid) : null;
    const bb_upper = raw.bb_upper !== null && raw.bb_upper !== undefined ? Number(raw.bb_upper) : null;
    const bb_lower = raw.bb_lower !== null && raw.bb_lower !== undefined ? Number(raw.bb_lower) : null;
    const cci = raw.cci !== null && raw.cci !== undefined ? Number(raw.cci) : null;

    return {
        x: new Date(t),
        o,
        h,
        l,
        c,
        bb_mid,
        bb_upper,
        bb_lower,
        cci,
    };
}

// CCI용 데이터만 뽑기
function mapCciForChart(raw) {
    if (!raw) return null;
    const t = raw.time ? raw.time * 1000 : null;
    if (!t) return null;

    const cci = raw.cci;
    if (cci === null || cci === undefined || isNaN(Number(cci))) return null;

    return {
        x: new Date(t),
        y: Number(cci),
    };
}

// =====================
// 캔들 + BB 차트 생성/업데이트
// =====================

function initChart(symbol) {
    const canvasId = CHART_IDS[symbol];
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.warn("Canvas element not found for", symbol, canvasId);
        return null;
    }

    const ctx = canvas.getContext("2d");

    const chart = new Chart(ctx, {
        type: "candlestick",
        data: {
            datasets: [
                {
                    // 캔들
                    label: symbol,
                    type: "candlestick",
                    data: [],
                    parsing: false,  // {x,o,h,l,c} 직접 사용
                    barThickness: 4,
                    barPercentage: 0.6,
                },
                {
                    // BB mid
                    label: "BB Mid",
                    type: "line",
                    data: [],
                    borderWidth: 1,
                    pointRadius: 0,
                    borderDash: [2, 2],
                    parsing: { xAxisKey: "x", yAxisKey: "bb_mid" },
                },
                {
                    // BB upper
                    label: "BB Upper",
                    type: "line",
                    data: [],
                    borderWidth: 1,
                    pointRadius: 0,
                    parsing: { xAxisKey: "x", yAxisKey: "bb_upper" },
                },
                {
                    // BB lower
                    label: "BB Lower",
                    type: "line",
                    data: [],
                    borderWidth: 1,
                    pointRadius: 0,
                    parsing: { xAxisKey: "x", yAxisKey: "bb_lower" },
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: "time",
                    time: {
                        tooltipFormat: "yyyy-MM-dd HH:mm",
                    },
                    ticks: {
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
                            if (ctx.dataset.type === "candlestick") {
                                return `O:${v.o} H:${v.h} L:${v.l} C:${v.c}`;
                            }
                            return `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y, 2)}`;
                        },
                    },
                },
            },
        },
    });

    charts[symbol] = chart;
    return chart;
}

function updateChart(symbol, rawCandles) {
    if (!Array.isArray(rawCandles)) return;

    let chart = charts[symbol];
    if (!chart) {
        chart = initChart(symbol);
        if (!chart) return;
    }

    const mapped = rawCandles
        .map(mapCandleForChart)
        .filter(c => c !== null);

    // 모든 dataset에 같은 data 배열을 공유시킴
    chart.data.datasets[0].data = mapped; // 캔들
    chart.data.datasets[1].data = mapped; // BB mid
    chart.data.datasets[2].data = mapped; // BB upper
    chart.data.datasets[3].data = mapped; // BB lower

    chart.update();
}

// =====================
// CCI 차트 생성/업데이트
// =====================

function initCciChart(symbol) {
    const canvasId = CCI_CHART_IDS[symbol];
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.warn("CCI canvas not found for", symbol, canvasId);
        return null;
    }

    const ctx = canvas.getContext("2d");

    const chart = new Chart(ctx, {
        type: "line",
        data: {
            datasets: [
                {
                    label: "CCI",
                    data: [],
                    borderWidth: 1,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: "time",
                    time: {
                        tooltipFormat: "yyyy-MM-dd HH:mm",
                    },
                },
                y: {
                    position: "right",
                    ticks: {
                        // 보통 ±200 정도가 많이 보이는 영역
                        callback: (value) => value,
                    },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `CCI: ${fmtNum(ctx.parsed.y, 2)}`,
                    },
                },
            },
        },
    });

    cciCharts[symbol] = chart;
    return chart;
}

function updateCciChart(symbol, rawCandles) {
    if (!Array.isArray(rawCandles)) return;

    let chart = cciCharts[symbol];
    if (!chart) {
        chart = initCciChart(symbol);
        if (!chart) return;
    }

    const mapped = rawCandles
        .map(mapCciForChart)
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

    if (equityEl) {
        equityEl.textContent = fmtUSDT(state.equity);
    }

    renderEntryRestriction(state.entry_restrict);
    renderPosition(state.pos_state);
    renderLogs(state);

    const ohlcv = state.ohlcv || {};
    for (const sym of SYMBOLS) {
        const candles = ohlcv[sym];
        if (!Array.isArray(candles) || candles.length === 0) continue;
        updateChart(sym, candles);
        updateCciChart(sym, candles);
    }
}

// =====================
// 초기 실행
// =====================

window.addEventListener("load", () => {
    SYMBOLS.forEach(sym => {
        initChart(sym);
        initCciChart(sym);
    });
    connectWebSocket();
});
