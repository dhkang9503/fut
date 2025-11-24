// 모니터링 심볼 (JSON의 키 그대로)
const SYMBOLS = ["BTC/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT"];

// 심볼 → div id 매핑
const CHART_IDS = {
    "BTC/USDT:USDT": "chart-btc",
    "XRP/USDT:USDT": "chart-xrp",
    "DOGE/USDT:USDT": "chart-doge",
};

const charts = {};        // 심볼별 chart 객체
const candleSeries = {};  // 심볼별 캔들 시리즈
const entryLines = {};    // 엔트리 라인
const stopLines = {};     // 스탑 라인
const tpLines = {};       // TP 라인

const equityEl = document.getElementById("equity");
const entryRestrictEl = document.getElementById("entry_restrict");
const posEl = document.getElementById("position");
const logsEl = document.getElementById("logs");

// Entry Restriction 텍스트 렌더링
function renderEntryRestriction(entryRestrict) {
    if (!entryRestrict) return "-";
    let text = "";
    for (const sym of SYMBOLS) {
        const r = entryRestrict[sym];
        text += `${sym}: ${r === null ? "-" : r}\n`;
    }
    return text.trim();
}

// 심볼별 차트 생성
function initChartForSymbol(sym) {
    if (typeof LightweightCharts === "undefined") {
        // 라이브러리 로딩 실패하면 차트는 포기
        return;
    }

    const containerId = CHART_IDS[sym];
    const el = document.getElementById(containerId);
    if (!el) return;

    // 🔹 높이는 고정 220px, width만 컨테이너 기준
    const rect = el.getBoundingClientRect();
    const chart = LightweightCharts.createChart(el, {
        width: rect.width || el.clientWidth || 600,
        height: 220,
        layout: {
            background: { color: "#111827" },
            textColor: "#e5e7eb",
        },
        grid: {
            vertLines: { color: "#1f2937" },
            horzLines: { color: "#1f2937" },
        },
        timeScale: {
            timeVisible: true,
            secondsVisible: false,
            borderColor: "#374151",
        },
        rightPriceScale: {
            borderColor: "#374151",
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
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

    const entry = chart.addLineSeries({ color: "#eab308", lineWidth: 1 });
    const stop = chart.addLineSeries({ color: "#ef4444", lineWidth: 1 });
    const tp = chart.addLineSeries({ color: "#22c55e", lineWidth: 1 });

    charts[sym] = chart;
    candleSeries[sym] = candles;
    entryLines[sym] = entry;
    stopLines[sym] = stop;
    tpLines[sym] = tp;

    // 리사이즈 시 width만 맞춰주기
    window.addEventListener("resize", () => {
        const r = el.getBoundingClientRect();
        chart.applyOptions({ width: r.width || el.clientWidth || 600, height: 220 });
    });
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
        let raw = ohlcv[sym];
        if (!raw) continue;

        const candlesArr = Array.isArray(raw) ? raw : Object.values(raw);
        if (!candlesArr || candlesArr.length === 0) continue;

        if (!charts[sym]) {
            initChartForSymbol(sym);
            if (!charts[sym]) continue;  // 생성 실패 시 스킵
        }

        // time 은 초 단위 숫자여야 함
        const mapped = candlesArr.map(c => ({
            time: Number(c.time),
            open: Number(c.open),
            high: Number(c.high),
            low: Number(c.low),
            close: Number(c.close),
        }));

        candleSeries[sym].setData(mapped);
        charts[sym].timeScale().fitContent();

        // 포지션 라인
        const p = posState[sym] || {};
        const hasPosition = p.side && p.size > 0 && p.entry_price != null;

        const firstTime = mapped[0].time;
        const lastTime = mapped[mapped.length - 1].time;

        const makeLineData = (price) => ([
            { time: firstTime, value: price },
            { time: lastTime, value: price },
        ]);

        if (hasPosition) {
            const entryPrice = p.entry_price;
            const stopPrice = p.stop_price;
            const tpPrice = p.tp_price;

            entryLines[sym].setData(
                entryPrice != null ? makeLineData(entryPrice) : []
            );
            stopLines[sym].setData(
                stopPrice != null ? makeLineData(stopPrice) : []
            );
            tpLines[sym].setData(
                tpPrice != null ? makeLineData(tpPrice) : []
            );
        } else {
            entryLines[sym].setData([]);
            stopLines[sym].setData([]);
            tpLines[sym].setData([]);
        }
    }
}

// WebSocket 연결 & 자동 재접속
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
