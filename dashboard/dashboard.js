// 모니터링 심볼 (JSON의 키 그대로)
const SYMBOLS = ["BTC/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT"];

// 심볼 → HTML div id 매핑
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

function updateDashboard(state) {
    // 1) 위쪽 숫자 영역들 먼저 업데이트
    if (state.equity != null) {
        equityEl.innerText = Number(state.equity).toLocaleString() + " USDT";
    } else {
        equityEl.innerText = "-";
    }

    entryRestrictEl.innerText = renderEntryRestriction(state.entry_restrict);
    posEl.innerText = JSON.stringify(state.pos_state || {}, null, 2);
    logsEl.innerText = JSON.stringify(state.last_signal || {}, null, 2);

    // 2) 차트용 데이터
    const ohlcv = state.ohlcv || {};
    const posState = state.pos_state || {};

    for (const sym of SYMBOLS) {
        let raw = ohlcv[sym];
        if (!raw) continue;

        // 🔹 raw 는 JSON에서 이미 배열 형태지만,
        // 혹시라도 객체로 들어와도 대응하도록 방어 코드 추가
        let candles;
        if (Array.isArray(raw)) {
            candles = raw;
        } else {
            candles = Object.values(raw);
        }

        if (!candles || candles.length === 0) continue;

        // 차트 미생성 시 초기화
        if (!charts[sym]) {
            initChartForSymbol(sym);
        }

        try {
            candleSeries[sym].setData(candles);
            charts[sym].timeScale().fitContent();
        } catch (e) {
            console.log("chart error for", sym, e);
        }

        // 3) 포지션 엔트리/SL/TP 라인 오버레이
        const p = posState[sym] || {};
        const hasPosition = p.side && p.size > 0 && p.entry_price != null;

        const firstTime = candles[0].time;
        const lastTime = candles[candles.length - 1].time;

        if (hasPosition) {
            const entryPrice = p.entry_price;
            const stopPrice = p.stop_price;
            const tpPrice = p.tp_price;

            const lineData = (price) => [
                { time: firstTime, value: price },
                { time: lastTime, value: price },
            ];

            entryLines[sym].setData(entryPrice ? lineData(entryPrice) : []);
            stopLines[sym].setData(stopPrice ? lineData(stopPrice) : []);
            tpLines[sym].setData(tpPrice ? lineData(tpPrice) : []);
        } else {
            // 포지션 없으면 라인 제거
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
