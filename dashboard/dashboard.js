// =====================
// 기본 설정
// =====================

const SYMBOLS = ["BTC/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT"];

const CHART_IDS = {
    "BTC/USDT:USDT": "chart-btc",
    "XRP/USDT:USDT": "chart-xrp",
    "DOGE/USDT:USDT": "chart-doge",
};

const CCI_CHART_IDS = {
    "BTC/USDT:USDT": "cci-btc",
    "XRP/USDT:USDT": "cci-xrp",
    "DOGE/USDT:USDT": "cci-doge",
};

const charts = {};
const cciCharts = {};

const equityEl = document.getElementById("equity");
const entryRestrictEl = document.getElementById("entry_restrict");
const posEl = document.getElementById("position");
const logsEl = document.getElementById("logs");

// =====================
// 유틸
// =====================

function fmtUSDT(value) {
    if (value === null || value === undefined || isNaN(value)) return "-";
    return `${Number(value).toFixed(3)} USDT`;
}

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
            tp_price: p.tp_price || null,
            stop_order_id: p.stop_order_id || null,
            entry_time: p.entry_time || null,
            entry_candle_ts: p.entry_candle_ts || null,
        };
    }
    posEl.textContent = JSON.stringify(formatted, null, 2);
}

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
// 캔들 매핑
// =====================

function mapCandleForChart(raw) {
    if (!raw) return null;
    const t = raw.time ? raw.time * 1000 : null;
    if (!t) return null;

    const o = Number(raw.open);
    const h = Number(raw.high);
    const l = Number(raw.low);
    const c = Number(raw.close);
    if ([o, h, l, c].some(v => isNaN(v))) return null;

    return { x: new Date(t), o, h, l, c };
}

// =====================
// 가격 + 볼밴 + 진입/TP/SL + 진입 마커 차트
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
                    // 0: 캔들
                    label: symbol,
                    type: "candlestick",
                    data: [],
                    barThickness: 4,
                    barPercentage: 0.6,
                },
                {
                    // 1: Entry (주황색)
                    label: "Entry",
                    type: "line",
                    data: [],
                    borderColor: "#fb923c",
                    borderWidth: 1,
                    borderDash: [4, 2],
                    pointRadius: 0,
                },
                {
                    // 2: TP (노랑)
                    label: "TP",
                    type: "line",
                    data: [],
                    borderColor: "#facc15",
                    borderWidth: 1,
                    borderDash: [2, 2],
                    pointRadius: 0,
                },
                {
                    // 3: SL (빨강)
                    label: "SL",
                    type: "line",
                    data: [],
                    borderColor: "#ef4444",
                    borderWidth: 1,
                    borderDash: [2, 4],
                    pointRadius: 0,
                },
                {
                    // 4: BB Upper
                    label: "BB Upper",
                    type: "line",
                    data: [],
                    borderWidth: 1,
                    pointRadius: 0,
                    borderColor: "rgba(75,192,192,0.4)",   // #4BC0C0 + 투명
                },
                {
                    // 5: BB Lower
                    label: "BB Lower",
                    type: "line",
                    data: [],
                    borderWidth: 1,
                    pointRadius: 0,
                    borderColor: "rgba(153,102,255,0.4)",  // #9966FF + 투명
                },
                {
                    // 6: BB Mid
                    label: "BB Mid",
                    type: "line",
                    data: [],
                    borderWidth: 1,
                    pointRadius: 0,
                    borderColor: "rgba(228,229,231,0.4)",  // #E4E5E7 + 투명
                },
                {
                    // 7: 롱 진입 마커 (캔들 하단 삼각형)
                    label: "Long Entry Marker",
                    type: "scatter",
                    data: [],
                    showLine: false,
                    pointRadius: 5,
                    pointStyle: "triangle",
                    borderColor: "#38bdf8",
                    backgroundColor: "#38bdf8",
                },
                {
                    // 8: 숏 진입 마커 (캔들 상단 역삼각형)
                    label: "Short Entry Marker",
                    type: "scatter",
                    data: [],
                    showLine: false,
                    pointRadius: 5,
                    pointStyle: "triangle",
                    // rotation은 시각적으로는 중요치 않지만, 지원되는 버전에서는 아래 옵션:
                    // pointRotation: 180,
                    borderColor: "#fb7185",
                    backgroundColor: "#fb7185",
                },
            ],
        },
        options: {
            parsing: false,
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: "time",
                    time: { tooltipFormat: "yyyy-MM-dd HH:mm" },
                    ticks: { source: "auto" },
                },
                y: { position: "right" },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const v = ctx.raw;
                            if (!v) return "";
                            if (v.o !== undefined) {
                                return `O:${v.o} H:${v.h} L:${v.l} C:${v.c}`;
                            }
                            if (v.y !== undefined) {
                                return `Price: ${v.y}`;
                            }
                            return "";
                        },
                    },
                },
            },
        },
    });

    charts[symbol] = chart;
    return chart;
}

function updateChart(symbol, rawCandles, posStateForSymbol) {
    if (!Array.isArray(rawCandles)) return;

    let chart = charts[symbol];
    if (!chart) {
        chart = initChart(symbol);
        if (!chart) return;
    }

    const mapped = rawCandles.map(mapCandleForChart).filter(c => c !== null);
    chart.data.datasets[0].data = mapped;

    const hasCandles = mapped.length > 0;
    const firstX = hasCandles ? mapped[0].x : null;
    const lastX = hasCandles ? mapped[mapped.length - 1].x : null;

    // Entry / TP / SL
    const hasPos = posStateForSymbol && posStateForSymbol.side && posStateForSymbol.size > 0;
    let entryLineData = [];
    let tpLineData = [];
    let slLineData = [];

    if (hasPos && hasCandles) {
        const entryPrice = Number(posStateForSymbol.entry_price);
        const tpPrice = Number(posStateForSymbol.tp_price);
        const slPrice = Number(posStateForSymbol.stop_price);

        if (!isNaN(entryPrice) && entryPrice > 0) {
            entryLineData = [
                { x: firstX, y: entryPrice },
                { x: lastX, y: entryPrice },
            ];
        }
        if (!isNaN(tpPrice) && tpPrice > 0) {
            tpLineData = [
                { x: firstX, y: tpPrice },
                { x: lastX, y: tpPrice },
            ];
        }
        if (!isNaN(slPrice) && slPrice > 0) {
            slLineData = [
                { x: firstX, y: slPrice },
                { x: lastX, y: slPrice },
            ];
        }
    }

    if (chart.data.datasets[1]) chart.data.datasets[1].data = entryLineData;
    if (chart.data.datasets[2]) chart.data.datasets[2].data = tpLineData;
    if (chart.data.datasets[3]) chart.data.datasets[3].data = slLineData;

    // Bollinger
    const bbUpperData = [];
    const bbLowerData = [];
    const bbMidData = [];

    for (const raw of rawCandles) {
        if (!raw || !raw.time) continue;
        const t = new Date(raw.time * 1000);

        const bu = raw.bb_upper !== undefined && raw.bb_upper !== null ? Number(raw.bb_upper) : NaN;
        const bl = raw.bb_lower !== undefined && raw.bb_lower !== null ? Number(raw.bb_lower) : NaN;
        const bm = raw.bb_mid   !== undefined && raw.bb_mid   !== null ? Number(raw.bb_mid)   : NaN;

        if (!isNaN(bu)) bbUpperData.push({ x: t, y: bu });
        if (!isNaN(bl)) bbLowerData.push({ x: t, y: bl });
        if (!isNaN(bm)) bbMidData.push({ x: t, y: bm });
    }

    if (chart.data.datasets[4]) chart.data.datasets[4].data = bbUpperData;
    if (chart.data.datasets[5]) chart.data.datasets[5].data = bbLowerData;
    if (chart.data.datasets[6]) chart.data.datasets[6].data = bbMidData;

    // ===== 진입 마커 (롱/숏 삼각형) =====
    let longMarkers = [];
    let shortMarkers = [];

    // posStateForSymbol.side 가 있으면 그 포지션 기준으로 표시
    if (posStateForSymbol && posStateForSymbol.side && hasCandles) {
        let entryTsSec = null;

        // 1) entry_candle_ts (ms 단위) 있으면 우선 사용
        if (posStateForSymbol.entry_candle_ts) {
            const v = Number(posStateForSymbol.entry_candle_ts);
            if (!isNaN(v) && v > 0) {
                entryTsSec = Math.floor(v / 1000);
            }
        }

        // 2) 없으면 entry_time(ISO) 기반으로 계산
        if (entryTsSec === null && posStateForSymbol.entry_time) {
            const d = new Date(posStateForSymbol.entry_time);
            if (!isNaN(d.getTime())) {
                entryTsSec = Math.floor(d.getTime() / 1000);
            }
        }

        // 3) entryTsSec 기준으로 해당(또는 가장 가까운) 캔들 찾기
        let targetRaw = null;
        if (entryTsSec !== null) {
            // 정확히 time이 일치하는 캔들 먼저 시도
            for (const raw of rawCandles) {
                if (Number(raw.time) === entryTsSec) {
                    targetRaw = raw;
                    break;
                }
            }

            // 정확히 일치하는 게 없으면 "가장 가까운 time"을 가진 캔들 선택
            if (!targetRaw) {
                let minDiff = Infinity;
                for (const raw of rawCandles) {
                    const t = Number(raw.time);
                    if (isNaN(t)) continue;
                    const diff = Math.abs(t - entryTsSec);
                    if (diff < minDiff) {
                        minDiff = diff;
                        targetRaw = raw;
                    }
                }
            }
        }

        if (targetRaw) {
            const markerX = new Date(Number(targetRaw.time) * 1000);
            const high = Number(targetRaw.high);
            const low = Number(targetRaw.low);

            if (posStateForSymbol.side === "long" && !isNaN(low)) {
                longMarkers = [
                    { x: markerX, y: low * 0.992 },
                ];
            } else if (posStateForSymbol.side === "short" && !isNaN(high)) {
                shortMarkers = [
                    { x: markerX, y: high * 1.008 },
                ];
            }
        }
    }

    if (chart.data.datasets[7]) chart.data.datasets[7].data = longMarkers;
    if (chart.data.datasets[8]) chart.data.datasets[8].data = shortMarkers;

    chart.update();
}

// =====================
// CCI 차트 (라벨 + 값 방식)
// =====================

function initCciChart(symbol) {
    const canvasId = CCI_CHART_IDS[symbol];
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.warn("CCI canvas element not found for", symbol, canvasId);
        return null;
    }

    const ctx = canvas.getContext("2d");

    const chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    // 0: 실제 CCI 값
                    label: "CCI",
                    data: [],
                    borderWidth: 1,
                    pointRadius: 0,
                    borderColor: "#facc15",   // 밝은 노란색
                    tension: 0.1,
                },
                {
                    // 1: 0 라인 (회색, 반투명 실선)
                    label: "Zero",
                    data: [],
                    borderWidth: 1,
                    pointRadius: 0,
                    borderColor: "rgba(148,163,184,0.8)",
                    borderDash: [],
                },
                {
                    // 2: +100 라인 (회색 점선)
                    label: "+100",
                    data: [],
                    borderWidth: 1,
                    pointRadius: 0,
                    borderColor: "rgba(148,163,184,0.7)",
                    borderDash: [4, 4],
                },
                {
                    // 3: -100 라인 (회색 점선)
                    label: "-100",
                    data: [],
                    borderWidth: 1,
                    pointRadius: 0,
                    borderColor: "rgba(148,163,184,0.7)",
                    borderDash: [4, 4],
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: "category",
                    ticks: {
                        maxTicksLimit: 6,
                    },
                },
                y: {
                    position: "right",
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const v = ctx.raw;
                            if (v === undefined || v === null) return "";
                            return `CCI: ${v}`;
                        },
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

    const labels = [];
    const cciValues = [];

    for (const raw of rawCandles) {
        if (!raw || !raw.time) continue;

        const v = raw.cci;
        if (v === null || v === undefined) continue;
        const cci = typeof v === "number" ? v : Number(v);
        if (isNaN(cci)) continue;

        const d = new Date(raw.time * 1000);
        const label =
            `${d.getMonth() + 1}/${d.getDate()} ` +
            `${String(d.getHours()).padStart(2, "0")}h`;

        labels.push(label);
        cciValues.push(cci);
    }

    chart.data.labels = labels;
    chart.data.datasets[0].data = cciValues;

    const zeroLine = labels.map(() => 0);
    const plus100Line = labels.map(() => 100);
    const minus100Line = labels.map(() => -100);

    chart.data.datasets[1].data = zeroLine;
    chart.data.datasets[2].data = plus100Line;
    chart.data.datasets[3].data = minus100Line;

    chart.update();
}

// =====================
// WebSocket
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
// 상태 업데이트
// =====================

function handleStateUpdate(state) {
    if (!state) return;

    if (equityEl) {
        equityEl.textContent = fmtUSDT(state.equity);
    }

    renderEntryRestriction(state.entry_restrict);
    const posState = state.pos_state || {};
    renderPosition(posState);
    renderLogs(state);

    const ohlcv = state.ohlcv || {};
    for (const sym of SYMBOLS) {
        const candles = ohlcv[sym];
        if (!Array.isArray(candles) || candles.length === 0) continue;
        const p = posState[sym] || null;
        updateChart(sym, candles, p);
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
