// =====================
// 기본 설정
// =====================

const SYMBOLS = ["DOGE/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"];

const CHART_IDS = {
    "DOGE/USDT:USDT": "chart-btc",
    "ETH/USDT:USDT": "chart-eth",
    "SOL/USDT:USDT": "chat-sol",
};

const CCI_CHART_IDS = {
    "DOGE/USDT:USDT": "cci-btc",
    "ETH/USDT:USDT": "cci-eth",
    "SOL/USDT:USDT": "cci-sol",
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

function fmtNumber(value, digits = 4) {
    if (value === null || value === undefined || isNaN(value)) return "-";
    return Number(value).toFixed(digits);
}

function fmtDateTime(value) {
    if (!value) return "-";

    // entry_time 은 ISO(UTC)로 들어오므로, Date로 파싱한 뒤 KST(+9h)로 변환
    const d = new Date(value);
    if (isNaN(d.getTime())) return "-";

    const kst = new Date(d.getTime());

    const yyyy = kst.getFullYear();
    const MM = String(kst.getMonth() + 1).padStart(2, "0");
    const DD = String(kst.getDate()).padStart(2, "0");
    const hh = String(kst.getHours()).padStart(2, "0");
    const mm = String(kst.getMinutes()).padStart(2, "0");
    const ss = String(kst.getSeconds()).padStart(2, "0");

    // 👉 2025/11/25 01:02:03 형태
    return `${yyyy}/${MM}/${DD} ${hh}:${mm}:${ss}`;
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

// =====================
// 현재 포지션: 표 렌더
// =====================

function renderPosition(posState) {
    if (!posEl) return;

    if (!posState) {
        posEl.innerHTML = `<div class="text-gray-400 text-sm">포지션 정보가 없습니다.</div>`;
        return;
    }

    // pos_state 에서 실제로 포지션이 있는 심볼만 추리기
    const rows = [];
    for (const sym of SYMBOLS) {
        const p = posState[sym];
        if (!p || !p.side || !p.size || p.size === 0) continue;

        rows.push({
            symbol: sym,
            side: p.side,
            size: p.size,
            entry_price: p.entry_price,
            tp_price: p.tp_price,
            stop_price: p.stop_price,
            stop_order_id: p.stop_order_id,
            entry_time: p.entry_time,
        });
    }

    if (rows.length === 0) {
        posEl.innerHTML = `<div class="text-gray-400 text-sm">열려있는 포지션이 없습니다.</div>`;
        return;
    }

    let html = `
      <div class="overflow-x-auto">
        <table class="min-w-full text-xs sm:text-sm text-left border-collapse">
          <thead>
            <tr class="border-b border-gray-700 bg-gray-900/40">
              <th class="px-2 py-1 sm:px-3 sm:py-2">진입시간</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2">심볼</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2">방향</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2 text-right">사이즈</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2 text-right">진입가</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2 text-right">익절가</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2 text-right">손절가</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2">stop_order_id</th>
            </tr>
          </thead>
          <tbody>
    `;

    for (const r of rows) {
        const sideLabel = r.side === "long" ? "롱" : (r.side === "short" ? "숏" : "-");
        const sideColor =
            r.side === "long"
                ? "text-green-400"
                : r.side === "short"
                ? "text-red-400"
                : "text-gray-300";

        html += `
          <tr class="border-b border-gray-800 hover:bg-gray-900/40">
            <td class="px-2 py-1 sm:px-3 sm:py-2 whitespace-nowrap text-gray-300">${fmtDateTime(r.entry_time)}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2 whitespace-nowrap text-gray-200">${r.symbol}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2 whitespace-nowrap ${sideColor} font-semibold">${sideLabel}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2 text-right text-gray-100">${fmtNumber(r.size, 0)}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2 text-right text-gray-100">${fmtNumber(r.entry_price)}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2 text-right text-teal-300">${r.tp_price != null ? fmtNumber(r.tp_price) : "-"}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2 text-right text-red-300">${r.stop_price != null ? fmtNumber(r.stop_price) : "-"}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2 text-gray-400 break-all">${r.stop_order_id || "-"}</td>
          </tr>
        `;
    }

    html += `
          </tbody>
        </table>
      </div>
    `;

    posEl.innerHTML = html;
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
                    // 7: 롱 진입 마커
                    label: "Long Entry Marker",
                    type: "scatter",
                    data: [],
                    showLine: false,
                    pointRadius: 5,
                    pointStyle: "triangle",
                    borderColor: "rgba(56, 189, 248, 0.5)",
                    backgroundColor: "rgba(56, 189, 248, 0.5)",
                },
                {
                    // 8: 숏 진입 마커
                    label: "Short Entry Marker",
                    type: "scatter",
                    data: [],
                    showLine: false,
                    pointRadius: 5,
                    pointStyle: "triangle",
                    pointRotation: 180,
                    borderColor: "rgba(251, 113, 133, 0.5)",
                    backgroundColor: "rgba(251, 113, 133, 0.5)",
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

    if (posStateForSymbol && posStateForSymbol.side && hasCandles) {
        let entryTsSec = null;

        if (posStateForSymbol.entry_candle_ts) {
            const v = Number(posStateForSymbol.entry_candle_ts);
            if (!isNaN(v) && v > 0) {
                entryTsSec = Math.floor(v / 1000);
            }
        }

        if (entryTsSec === null && posStateForSymbol.entry_time) {
            const d = new Date(posStateForSymbol.entry_time);
            if (!isNaN(d.getTime())) {
                entryTsSec = Math.floor(d.getTime() / 1000);
            }
        }

        let targetRaw = null;
        if (entryTsSec !== null) {
            for (const raw of rawCandles) {
                if (Number(raw.time) === entryTsSec) {
                    targetRaw = raw;
                    break;
                }
            }

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
                longMarkers = [{ x: markerX, y: low * 0.99 }];
            } else if (posStateForSymbol.side === "short" && !isNaN(high)) {
                shortMarkers = [{ x: markerX, y: high * 1.01 }];
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
                    borderColor: "rgba(250,204,21,0.7)",   // 밝은 노란색
                    tension: 0.1,
                },
                {
                    // 1: 0 라인
                    label: "Zero",
                    data: [],
                    borderWidth: 1,
                    pointRadius: 0,
                    borderColor: "rgba(148,163,184,0.8)",
                    borderDash: [],
                },
                {
                    // 2: +100 라인
                    label: "+100",
                    data: [],
                    borderWidth: 1,
                    pointRadius: 0,
                    borderColor: "rgba(148,163,184,0.7)",
                    borderDash: [4, 4],
                },
                {
                    // 3: -100 라인
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
                    ticks: { maxTicksLimit: 6 },
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
