// 모니터링 심볼 (JSON의 키 그대로)
const SYMBOLS = ["BTC/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT"];

// 심볼 → canvas id 매핑
const CHART_IDS = {
    "BTC/USDT:USDT": "chart-btc",
    "XRP/USDT:USDT": "chart-xrp",
    "DOGE/USDT:USDT": "chart-doge",
};

const charts = {};  // 심볼별 Chart.js 인스턴스

const equityEl = document.getElementById("equity");
const entryRestrictEl = document.getElementById("entry_restrict");
const posEl = document.getElementById("position");
const logsEl = document.getElementById("logs");

// Entry Restriction 예쁘게
function renderEntryRestriction(entryRestrict) {
    if (!entryRestrict) return "-";
    let text = "";
    for (const sym of SYMBOLS) {
        const r = entryRestrict[sym];
        text += `${sym}: ${r === null ? "-" : r}\n`;
    }
    return text.trim();
}

// 심볼별 차트 초기화
function initChart(sym, labels, closeData, entryPrice, stopPrice, tpPrice) {
    const canvasId = CHART_IDS[sym];
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const datasets = [];

    // 메인 가격 라인
    datasets.push({
        label: `${sym} Close`,
        data: closeData,
        borderColor: "#38bdf8",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
    });

    const makeLineDataset = (label, color, price) => ({
        label,
        data: labels.map(() => price),
        borderColor: color,
        borderWidth: 1,
        borderDash: [4, 4],
        pointRadius: 0,
        tension: 0,
    });

    if (entryPrice != null) {
        datasets.push(makeLineDataset("Entry", "#eab308", entryPrice));
    }
    if (stopPrice != null) {
        datasets.push(makeLineDataset("Stop", "#f97373", stopPrice));
    }
    if (tpPrice != null) {
        datasets.push(makeLineDataset("TP", "#4ade80", tpPrice));
    }

    const chart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
            },
            scales: {
                x: {
                    ticks: {
                        maxTicksLimit: 6,
                        color: "#9ca3af",
                        font: { size: 10 },
                    },
                    grid: { color: "#1f2937" },
                },
                y: {
                    ticks: {
                        color: "#9ca3af",
                        font: { size: 10 },
                    },
                    grid: { color: "#1f2937" },
                },
            },
        },
    });

    charts[sym] = chart;
}

// 차트 업데이트
function updateChart(sym, candles, posInfo) {
    if (!candles || candles.length === 0) return;

    const labels = candles.map(c => {
        const d = new Date(c.time * 1000);
        // 'MM-DD HH:mm' 정도로 간단하게
        return `${(d.getMonth() + 1).toString().padStart(2, "0")}-${d
            .getDate()
            .toString()
            .padStart(2, "0")} ${d.getHours().toString().padStart(2, "0")}:00`;
    });

    const closeData = candles.map(c => c.close);

    const hasPosition =
        posInfo &&
        posInfo.side &&
        posInfo.size > 0 &&
        posInfo.entry_price != null;

    const entryPrice = hasPosition ? posInfo.entry_price : null;
    const stopPrice = hasPosition ? posInfo.stop_price : null;
    const tpPrice = hasPosition ? posInfo.tp_price : null;

    if (!charts[sym]) {
        // 최초 생성
        initChart(sym, labels, closeData, entryPrice, stopPrice, tpPrice);
        return;
    }

    const chart = charts[sym];

    // dataset[0] = close, [1] 이후는 entry/stop/tp (있을 경우)
    chart.data.labels = labels;
    chart.data.datasets[0].data = closeData;

    // 기존 수평선 datasets는 싹 갈아엎자
    chart.data.datasets = chart.data.datasets.slice(0, 1);

    const makeLineDataset = (label, color, price) => ({
        label,
        data: labels.map(() => price),
        borderColor: color,
        borderWidth: 1,
        borderDash: [4, 4],
        pointRadius: 0,
        tension: 0,
    });

    if (entryPrice != null) {
        chart.data.datasets.push(
            makeLineDataset("Entry", "#eab308", entryPrice)
        );
    }
    if (stopPrice != null) {
        chart.data.datasets.push(
            makeLineDataset("Stop", "#f97373", stopPrice)
        );
    }
    if (tpPrice != null) {
        chart.data.datasets.push(
            makeLineDataset("TP", "#4ade80", tpPrice)
        );
    }

    chart.update("none");
}

// 대시보드 전체 업데이트
function updateDashboard(state) {
    // 상단 숫자/텍스트들
    if (state.equity != null) {
        equityEl.innerText = Number(state.equity).toLocaleString() + " USDT";
    } else {
        equityEl.innerText = "-";
    }

    entryRestrictEl.innerText = renderEntryRestriction(state.entry_restrict);
    posEl.innerText = JSON.stringify(state.pos_state || {}, null, 2);
    logsEl.innerText = JSON.stringify(state.last_signal || {}, null, 2);

    const ohlcv = state.ohlcv || {};
    const posState = state.pos_state || {};

    for (const sym of SYMBOLS) {
        const candles = ohlcv[sym];
        if (!candles || candles.length === 0) continue;

        const posInfo = posState[sym] || {};
        updateChart(sym, candles, posInfo);
    }
}

// WebSocket 연결 및 자동 재접속
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
