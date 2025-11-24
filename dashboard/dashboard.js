// 모니터링 심볼
const SYMBOLS = ["BTC/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT"];

const CHART_IDS = {
    "BTC/USDT:USDT": "chart-btc",
    "XRP/USDT:USDT": "chart-xrp",
    "DOGE/USDT:USDT": "chart-doge",
};

const charts = {};
const candleSeries = {};
const entryLines = {};
const stopLines = {};
const tpLines = {};

const equityEl = document.getElementById("equity");
const entryRestrictEl = document.getElementById("entry_restrict");
const posEl = document.getElementById("position");
const logsEl = document.getElementById("logs");

function initChartForSymbol(sym) {
    const containerId = CHART_IDS[sym];
    const el = document.getElementById(containerId);
    if (!el) return;

    const chart = LightweightCharts.createChart(el, {
        layout: {
            background: { color: "#111827" },
            textColor: "white",
        },
        grid: {
            vertLines: { color: "#1f2937" },
            horzLines: { color: "#1f2937" },
        },
        width: el.clientWidth,
        height: 220,
        timeScale: {
            timeVisible: true,
            secondsVisible: false,
        },
    });

    const candles = chart.addCandlestickSeries({
        upColor: "#22c55e",
        borderUpColor: "#22c55e",
        wickUpColor: "#22c55e",
        downColor: "#ef4444",
        borderDownColor: "#ef4444",
        wickDownColor: "#ef4444",
    });

    const entry = chart.addLineSeries({ color: "yellow", lineWidth: 1 });
    const stop = chart.addLineSeries({ color: "red", lineWidth: 1 });
    const tp = chart.addLineSeries({ color: "lime", lineWidth: 1 });

    charts[sym] = chart;
    candleSeries[sym] = candles;
    entryLines[sym] = entry;
    stopLines[sym] = stop;
    tpLines[sym] = tp;
}

function renderEntryRestriction(entryRestrict) {
    if (!entryRestrict) return "-";
    let text = "";
    for (const sym of SYMBOLS) {
        const r = entryRestrict[sym];
        text += `${sym}: ${r === null ? "-" : r}\n`;
    }
    return text.trim();
}

function updateDashboard(state) {
    // Equity
    if (state.equity != null) {
        equityEl.innerText = Number(state.equity).toLocaleString() + " USDT";
    } else {
        equityEl.innerText = "-";
    }

    // Entry Restriction
    entryRestrictEl.innerText = renderEntryRestriction(state.entry_restrict);

    // Raw 상태 표시
    posEl.innerText = JSON.stringify(state.pos_state || {}, null, 2);
    logsEl.innerText = JSON.stringify(state.last_signal || {}, null, 2);

    const ohlcv = state.ohlcv || {};
    const posState = state.pos_state || {};

    for (const sym of SYMBOLS) {
        const candles = ohlcv[sym];
        if (!candles || candles.length === 0) continue;

        if (!charts[sym]) {
            initChartForSymbol(sym);
        }

        candleSeries[sym].setData(candles);

        const p = posState[sym] || {};
        const hasPosition = p.side && p.size > 0 && p.entry_price != null;

        const firstTime = candles[0].time;
        const lastTime = candles[candles.length - 1].time;

        if (hasPosition) {
            const entryPrice = p.entry_price;
            const stopPrice = p.stop_price;
            const tpPrice = p.tp_price; // 없으면 undefined라 자동 무시됨

            const lineData = (price) => [
                { time: firstTime, value: price },
                { time: lastTime, value: price },
            ];

            entryLines[sym].setData(entryPrice ? lineData(entryPrice) : []);
            stopLines[sym].setData(stopPrice ? lineData(stopPrice) : []);
            tpLines[sym].setData(tpPrice ? lineData(tpPrice) : []);
        } else {
            entryLines[sym].setData([]);
            stopLines[sym].setData([]);
            tpLines[sym].setData([]);
        }
    }
}

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
        console.log("WS Closed. Reconnecting in 3s...");
        setTimeout(connectWS, 3000);
    };

    socket.onerror = (e) => {
        console.log("WS Error:", e);
    };
}

connectWS();
function updateDashboard(state) {
    // Equity
    document.getElementById("equity").innerText =
        state.equity ? state.equity.toLocaleString() + " USDT" : "-";

    // Entry Restriction
    const restrict = state.entry_restrict;
    let text = "";

    for (const sym in restrict) {
        const r = restrict[sym];
        text += `${sym}: ${r === null ? "-" : r}\n`;
    }

// entryRestrictElement.innerText = text;

    document.getElementById("entry_restrict").innerText =
        text || "none";

    // Position
    document.getElementById("position").innerText =
        JSON.stringify(state.pos_state, null, 2);

    // Logs
    document.getElementById("logs").innerText =
        JSON.stringify(state.last_signal, null, 2);

    // 현재 포지션 가진 심볼 찾기
    const activeSymbols = Object.keys(state.pos_state || {}).filter(sym => {
        const p = state.pos_state[sym];
        return p && p.side && p.size > 0;
    });

    if (activeSymbols.length === 0) {
        document.getElementById("chart").innerHTML =
            "<p class='text-gray-400'>현재 포지션 없음</p>";
        return;
    }

    const sym = activeSymbols[0]; // 첫 번째 포지션 심볼만 표시

    const pos = state.pos_state[sym];
    const price = pos.entry_price || 0;

    if (!chart) initChart();

    // SL / TP / Entry 라인 업데이트
    const now = Math.floor(Date.now() / 1000);

    entryLine.setData([{ time: now, value: pos.entry_price }]);
    stopLine.setData([{ time: now, value: pos.stop_price }]);
    tpLine.setData([{ time: now, value: pos.tp_price }]);
}

function connectWS() {
    socket = new WebSocket("ws://" + window.location.hostname + ":8000/ws");

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };

    socket.onclose = () => {
        console.log("WS Closed. Reconnecting in 3s...");
        setTimeout(connectWS, 3000);
    };
}

connectWS();
