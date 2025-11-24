// =====================
// 기본 설정
// =====================

const SYMBOLS = ["BTC/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT"];

const CHART_IDS = {
    "BTC/USDT:USDT": "chart-btc",
    "XRP/USDT:USDT": "chart-xrp",
    "DOGE/USDT:USDT": "chart-doge",
};

const charts = {};

const equityEl = document.getElementById("equity");
const entryRestrictEl = document.getElementById("entry_restrict");
const posEl = document.getElementById("position");
const logsEl = document.getElementById("logs");


// =====================
// 유틸
// =====================

function fmtNum(v, d = 3) {
    return (v === null || v === undefined || isNaN(v)) ? "-" : Number(v).toFixed(d);
}

function fmtUSDT(v) {
    return (v === null || v === undefined || isNaN(v)) ? "-" : `${Number(v).toFixed(3)} USDT`;
}


// =====================
// render 포맷
// =====================

function renderEntryRestriction(entryRestrict) {
    if (!entryRestrict) { entryRestrictEl.textContent = "-"; return; }
    const lines = SYMBOLS.map(sym => `${sym}: ${entryRestrict[sym] ?? "-"}`);
    entryRestrictEl.textContent = lines.join("\n");
}

function renderPosition(posState) {
    if (!posState) return posEl.textContent = "{}";

    const formatted = {};
    for (const sym of SYMBOLS) {
        const p = posState[sym] || {};
        formatted[sym] = {
            side: p.side || null,
            size: p.size || 0,
            entry_price: p.entry_price || null,
            stop_price: p.stop_price || null,
            tp_price: p.tp_price || null,
            entry_candle_ts: p.entry_candle_ts || null,
            stop_order_id: p.stop_order_id || null,
            entry_time: p.entry_time || null,
        };
    }
    posEl.textContent = JSON.stringify(formatted, null, 2);
}

function renderLogs(state) {
    if (!logsEl) return;
    logsEl.textContent = JSON.stringify(state.last_signal || {}, null, 2);
}


// =====================
// Chart.js
// =====================

function mapCandle(raw) {
    if (!raw) return null;
    const t = raw.time ? raw.time * 1000 : null;
    if (!t) return null;

    return {
        x: new Date(t),
        o: Number(raw.open),
        h: Number(raw.high),
        l: Number(raw.low),
        c: Number(raw.close),
    };
}

function initChart(symbol) {
    const canvas = document.getElementById(CHART_IDS[symbol]);
    if (!canvas) return null;

    const ctx = canvas.getContext("2d");

    const chart = new Chart(ctx, {
        type: "candlestick",
        data: {
            datasets: [
                {
                    label: symbol,
                    type: "candlestick",
                    data: [],
                },
                {
                    label: "Entry",
                    type: "line",
                    borderColor: "gray",
                    borderWidth: 1,
                    borderDash: [4, 2],
                    pointRadius: 0,
                    data: [],
                },
                {
                    label: "TP",
                    type: "line",
                    borderColor: "blue",
                    borderWidth: 1,
                    borderDash: [2, 2],
                    pointRadius: 0,
                    data: [],
                },
                {
                    label: "SL",
                    type: "line",
                    borderColor: "red",
                    borderWidth: 1,
                    borderDash: [2, 4],
                    pointRadius: 0,
                    data: [],
                },
                {
                    label: "Entry Marker",
                    type: "scatter",
                    data: [],
                    pointStyle: [], // 각 점별로 style 지정
                    pointRadius: 8,
                    showLine: false,
                },
            ],
        },
        options: {
            parsing: false,
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { type: "time" },
                y: { position: "right" },
            },
            plugins: {
                legend: { display: false },
            },
        },
    });
    charts[symbol] = chart;
    return chart;
}


// ========== 차트 업데이트 ==========

function updateChart(symbol, rawCandles, pos) {
    let chart = charts[symbol];
    if (!chart) chart = initChart(symbol);

    const mapped = rawCandles.map(mapCandle).filter(v => v);

    // 캔들
    chart.data.datasets[0].data = mapped;

    const hasPos = pos && pos.side && pos.size > 0;

    let entryLine = [], tpLine = [], slLine = [], markerData = [], markerStyles = [];

    if (hasPos && mapped.length > 0) {
        const firstX = mapped[0].x;
        const lastX = mapped[mapped.length - 1].x;

        const entry = pos.entry_price;
        const stop  = pos.stop_price;
        const tp    = pos.tp_price;
        const ts    = pos.entry_candle_ts;

        if (entry > 0) {
            entryLine = [
                { x: firstX, y: entry },
                { x: lastX, y: entry },
            ];
        }
        if (tp > 0) {
            tpLine = [
                { x: firstX, y: tp },
                { x: lastX, y: tp },
            ];
        }
        if (stop > 0) {
            slLine = [
                { x: firstX, y: stop },
                { x: lastX, y: stop },
            ];
        }

        // ======== 삼각형 마커 표시 ========
        if (ts) {
            const candle = mapped.find(c => c.x.getTime() === ts * 1000);
            if (candle) {
                let y, style;

                if (pos.side === "short") {
                    y = candle.h * 1.0015;
                    style = "triangle-down";
                } else {
                    y = candle.l * 0.9985;
                    style = "triangle";
                }

                markerData.push({ x: candle.x, y });
                markerStyles.push(style);
            }
        }
    }

    // 라인 및 마커 반영
    chart.data.datasets[1].data = entryLine;
    chart.data.datasets[2].data = tpLine;
    chart.data.datasets[3].data = slLine;
    chart.data.datasets[4].data = markerData;
    chart.data.datasets[4].pointStyle = markerStyles;

    chart.update();
}


// =====================
// WebSocket
// =====================

let ws;
function connectWS() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${proto}://${location.host}/ws`;
    ws = new WebSocket(wsUrl);

    ws.onmessage = e => {
        const state = JSON.parse(e.data);
        renderState(state);
    };

    ws.onclose = () => setTimeout(connectWS, 2000);
}
function renderState(state) {
    if (equityEl) equityEl.textContent = fmtUSDT(state.equity);

    renderEntryRestriction(state.entry_restrict);
    renderPosition(state.pos_state);
    renderLogs(state);

    const ohlcv = state.ohlcv || {};
    for (const sym of SYMBOLS) {
        const candles = ohlcv[sym];
        if (!candles) continue;
        const p = (state.pos_state || {})[sym] || null;
        updateChart(sym, candles, p);
    }
}


// =====================
// 초기 실행
// =====================
window.addEventListener("load", () => {
    SYMBOLS.forEach(initChart);
    connectWS();
});
