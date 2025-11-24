// =====================
// Chart.js + Financial Plugin 캔들 차트 버전
// =====================

// 심볼 리스트
const SYMBOLS = ["BTC/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT"];

// 심볼 → canvas ID 매핑
const CHART_IDS = {
    "BTC/USDT:USDT": "chart-btc",
    "XRP/USDT:USDT": "chart-xrp",
    "DOGE/USDT:USDT": "chart-doge",
};

let charts = {};
let equityEl = document.getElementById("equity");
let entryRestrictEl = document.getElementById("entry_restrict");
let posEl = document.getElementById("position");
let logsEl = document.getElementById("logs");

// =====================
// Entry Restriction 출력
// =====================
function renderEntryRestriction(entryRestrict) {
    if (!entryRestrict) return "-";
    let text = "";
    for (const sym of SYMBOLS) {
        const r = entryRestrict[sym];
        text += `${sym}: ${r === null ? "-" : r}\n`;
    }
    return text.trim();
}

// =====================
// Chart.js 캔들 차트 초기화
// =====================
function initChart(sym) {
    const canvasId = CHART_IDS[sym];
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    // 기존 차트 제거
    if (charts[sym]) {
        charts[sym].destroy();
    }

    charts[sym] = new Chart(ctx, {
        type: "candlestick",
        data: {
            datasets: [
                {
                    label: `${sym} OHLC`,
                    data: [],  // 차후 setData 로 채울 것
                    borderColor: "#00bcd4",
                    color: {
                        up: "#22c55e",
                        down: "#ef4444",
                        unchanged: "#e5e7eb",
                    },
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,   // div 높이를 그대로 사용
            animation: false,

            scales: {
                x: {
                    type: "timeseries",
                    time: {
                        unit: "hour"
                    },
                    grid: { color: "#1f2937" },
                    ticks: { color: "#9ca3af" }
                },
                y: {
                    grid: { color: "#1f2937" },
                    ticks: { color: "#9ca3af" }
                }
            },

            plugins: {
                legend: { display: false },
            }
        }
    });
}

// =====================
// 차트에 캔들 데이터 반영
// =====================
function updateCandleChart(sym, ohlcv) {
    if (!charts[sym]) {
        initChart(sym);
    }

    const chart = charts[sym];
    if (!chart) return;

    const dataset = chart.data.datasets[0];

    dataset.data = ohlcv.map(c => ({
        x: Number(c.time) * 1000,
        o: Number(c.open),
        h: Number(c.high),
        l: Number(c.low),
        c: Number(c.close),
    }));

    chart.update();
}

// =====================
// 포지션 라인 오버레이
// =====================
function updateOverlayLines(sym, position, ohlcv) {
    if (!charts[sym]) return;
    const chart = charts[sym];

    // 기존 보조선 제거
    chart.data.datasets = chart.data.datasets.filter(d => d.type === "candlestick");

    const first = Number(ohlcv[0].time) * 1000;
    const last = Number(ohlcv[ohlcv.length - 1].time) * 1000;

    function makeLine(label, price, color) {
        return {
            label,
            type: "line",
            borderColor: color,
            borderWidth: 1,
            fill: false,
            data: [
                { x: first, y: price },
                { x: last, y: price }
            ]
        };
    }

    if (position && position.entry_price) {
        chart.data.datasets.push(
            makeLine("Entry", position.entry_price, "#eab308")
        );
    }
    if (position && position.stop_price) {
        chart.data.datasets.push(
            makeLine("Stop", position.stop_price, "#ef4444")
        );
    }
    if (position && position.tp_price) {
        chart.data.datasets.push(
            makeLine("TP", position.tp_price, "#22c55e")
        );
    }

    chart.update();
}

// =====================
// Dashboard 업데이트
// =====================
function updateDashboard(state) {
    if (state.equity != null) {
        equityEl.innerText = Number(state.equity).toLocaleString() + " USDT";
    }

    entryRestrictEl.innerText = renderEntryRestriction(state.entry_restrict);
    posEl.innerText = JSON.stringify(state.pos_state || {}, null, 2);
    logsEl.innerText = JSON.stringify(state.last_signal || {}, null, 2);

    const ohlcv = state.ohlcv || {};

    for (const sym of SYMBOLS) {
        if (!ohlcv[sym] || ohlcv[sym].length === 0) continue;

        // 캔들 반영
        updateCandleChart(sym, ohlcv[sym]);

        // 라인 오버레이
        const posData = (state.pos_state || {})[sym];
        updateOverlayLines(sym, posData, ohlcv[sym]);
    }
}

// =====================
// WebSocket 연결
// =====================
function connectWS() {
    const wsUrl =
        (location.protocol === "https:" ? "wss://" : "ws://") +
        window.location.hostname +
        ":8000/ws";

    const socket = new WebSocket(wsUrl);

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };

    socket.onclose = () => {
        setTimeout(connectWS, 2000);
    };

    socket.onerror = () => {};
}

connectWS();
