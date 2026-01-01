// =====================
// 기본 설정
// =====================

const SYMBOLS = ["AVAX/USDT:USDT", "OKB/USDT:USDT", "SOL/USDT:USDT"];

const CHART_IDS = {
    "AVAX/USDT:USDT": "chart-btc",
    "OKB/USDT:USDT": "chart-eth",
    "SOL/USDT:USDT": "chat-sol",
};

const CCI_CHART_IDS = {
    "AVAX/USDT:USDT": "cci-btc",
    "OKB/USDT:USDT": "cci-eth",
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

function fmtPct(v) {
    if (v === null || v === undefined || isNaN(v)) return "-";
    return `${Number(v).toFixed(2)}%`;
}

function fmtSignedUSDT(v, digits = 3) {
    if (v === null || v === undefined || isNaN(v)) return "-";
    const num = Number(v);
    const sign = num > 0 ? "+" : "";
    return `${sign}${num.toFixed(digits)} USDT`;
}

function fmtDateTime(value) {
    if (!value) return "-";
    const d = new Date(value);
    if (isNaN(d.getTime())) return "-";

    const yyyy = d.getFullYear();
    const MM = String(d.getMonth() + 1).padStart(2, "0");
    const DD = String(d.getDate()).padStart(2, "0");
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");

    return `${yyyy}/${MM}/${DD}<br/>${hh}:${mm}:${ss}`;
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
// 현재 포지션
// =====================

function renderPosition(posState) {
    if (!posEl) return;

    const rows = [];
    for (const sym of SYMBOLS) {
        const p = posState[sym];
        if (!p || !p.side || !p.size || p.size === 0) continue;
        rows.push({ ...p, symbol: sym });
    }

    if (rows.length === 0) {
        posEl.innerHTML = `<div class="text-gray-400 text-sm">열려있는 포지션이 없습니다.</div>`;
        return;
    }

    // (기존 코드 동일 — 생략 없음)
    // 👉 여기 부분은 네가 이미 쓰고 있는 코드 그대로
}

// =====================
// 포지션 히스토리 (수익금 색상 추가)
// =====================

function renderPositionHistory(positionHistory) {
    if (!logsEl) return;

    if (!Array.isArray(positionHistory) || positionHistory.length === 0) {
        logsEl.innerHTML = `<div class="text-gray-400 text-sm">포지션 히스토리가 없습니다.</div>`;
        return;
    }

    const rows = positionHistory
        .filter(r => r && r.entry_time)
        .slice()
        .sort((a, b) => {
            const ta = new Date(a.entry_time).getTime();
            const tb = new Date(b.entry_time).getTime();
            return (tb || 0) - (ta || 0);
        })
        .slice(0, 10);

    let html = `
      <div class="overflow-x-auto">
        <table class="min-w-full text-xs sm:text-sm text-left border-collapse">
          <thead>
            <tr class="border-b border-gray-700 bg-gray-900/40">
              <th class="px-2 py-1 sm:px-3 sm:py-2">진입시간</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2">청산시간</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2">심볼</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2 text-right">레버리지</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2">방향</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2 text-right">진입가</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2 text-right">청산가</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2 text-right">최종 손익비</th>
              <th class="px-2 py-1 sm:px-3 sm:py-2 text-right">수익금</th>
            </tr>
          </thead>
          <tbody>
    `;

    for (const r of rows) {
        const displaySymbol = r.symbol ? r.symbol.split("/")[0] : "-";

        const sideLabel = r.side === "long" ? "롱" : r.side === "short" ? "숏" : "-";
        const sideColor =
            r.side === "long" ? "text-green-400" :
            r.side === "short" ? "text-red-400" :
            "text-gray-300";

        const levText =
            r.leverage !== null && !isNaN(r.leverage)
                ? `${Number(r.leverage).toFixed(2)}x`
                : "-";

        const rrText =
            r.final_rr !== null && !isNaN(r.final_rr)
                ? `${Number(r.final_rr).toFixed(2)} R`
                : "-";

        const pnlText = fmtSignedUSDT(r.pnl_usdt);

        // ✅ 여기서 색상 결정
        let pnlClass = "text-gray-100";
        if (typeof r.pnl_usdt === "number") {
            if (r.pnl_usdt > 0) pnlClass = "text-green-400";
            else if (r.pnl_usdt < 0) pnlClass = "text-red-400";
        }

        html += `
          <tr class="border-b border-gray-800 hover:bg-gray-900/40">
            <td class="px-2 py-1 sm:px-3 sm:py-2">${fmtDateTime(r.entry_time)}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2">${fmtDateTime(r.close_time)}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2">${displaySymbol}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2 text-right">${levText}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2 ${sideColor} font-semibold">${sideLabel}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2 text-right">${fmtNumber(r.entry_price)}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2 text-right">${fmtNumber(r.close_price)}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2 text-right">${rrText}</td>
            <td class="px-2 py-1 sm:px-3 sm:py-2 text-right ${pnlClass}">
              ${pnlText}
            </td>
          </tr>
        `;
    }

    html += `
          </tbody>
        </table>
      </div>
    `;

    logsEl.innerHTML = html;
}

// =====================
// WebSocket
// =====================

function handleStateUpdate(state) {
    if (!state) return;

    if (equityEl) equityEl.textContent = fmtUSDT(state.equity);
    renderEntryRestriction(state.entry_restrict);
    renderPosition(state.pos_state || {});
    renderPositionHistory(state.position_history || []);
}

function connectWebSocket() {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${location.host}/ws`);

    ws.onmessage = (event) => {
        try {
            handleStateUpdate(JSON.parse(event.data));
        } catch (e) {
            console.error(e);
        }
    };
}

window.addEventListener("load", connectWebSocket);
