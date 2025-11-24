let socket = null;
let chart = null;
let candleSeries = null;
let entryLine = null;
let stopLine = null;
let tpLine = null;

function initChart() {
    const chartElement = document.getElementById("chart");
    chartElement.innerHTML = "";

    chart = LightweightCharts.createChart(chartElement, {
        layout: { background: { color: "#1f2937" }, textColor: "white" },
        grid: {
            vertLines: { color: "#2d3748" },
            horzLines: { color: "#2d3748" },
        },
        width: chartElement.clientWidth,
        height: 400,
    });

    candleSeries = chart.addCandlestickSeries({
        upColor: "#22c55e",
        borderUpColor: "#22c55e",
        wickUpColor: "#22c55e",
        downColor: "#ef4444",
        borderDownColor: "#ef4444",
        wickDownColor: "#ef4444",
    });

    entryLine = chart.addLineSeries({ color: "yellow", lineWidth: 2 });
    stopLine = chart.addLineSeries({ color: "red", lineWidth: 2 });
    tpLine = chart.addLineSeries({ color: "green", lineWidth: 2 });
}


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
