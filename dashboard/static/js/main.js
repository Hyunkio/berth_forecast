'use strict';

// ── 상태 ──────────────────────────────────────────────────────────────────────
const state = {
  model:       'ensemble',
  forecastPort:'부산',
  historyPort: '부산',
  data:        { forecast: null, metrics: null },
};

const PORTS     = ['부산', '울산', '인천', '광양'];
const PORT_IDX  = { '부산': 0, '울산': 1, '인천': 2, '광양': 3 };
const COLORS    = { line: '#1d6fa4', actual: '#e05555', grid: '#f0f2f5' };
const RISK_LABEL = { high: '위험', medium: '주의', low: '정상', unknown: '-' };

// ── 차트 인스턴스 ──────────────────────────────────────────────────────────────
let forecastChart = null;
let historyChart  = null;
let portRmseChart = null;

// ── 공통 차트 옵션 ─────────────────────────────────────────────────────────────
function baseOptions(yLabel = '체선율') {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true, position: 'top',
        labels: { font: { family: 'Pretendard', size: 12 }, boxWidth: 12, padding: 16 } },
      tooltip: {
        callbacks: {
          label: ctx => `${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(1)}%`
        }
      }
    },
    scales: {
      x: { grid: { color: COLORS.grid }, ticks: { font: { size: 12 } } },
      y: {
        grid: { color: COLORS.grid },
        ticks: { font: { size: 12 }, callback: v => (v * 100).toFixed(0) + '%' },
        title: { display: true, text: yLabel, font: { size: 12 }, color: '#8b96a5' },
        min: 0, suggestedMax: 1,
      }
    }
  };
}

// ── API 헬퍼 ──────────────────────────────────────────────────────────────────
async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

// ── 데이터 현황 ───────────────────────────────────────────────────────────────
async function loadDataStatus() {
  try {
    const d = await fetchJSON('/api/data_status');
    const badge = document.getElementById('data-status-badge');
    const lagText = d.lag_days > 0 ? ` (${d.lag_days}일 전)` : ' (오늘)';
    badge.textContent = `데이터 기준: ${d.last_data_date}${lagText}`;
    badge.className = 'data-badge' + (d.is_live ? ' data-badge-live' : '');
    badge.title = `예측 구간: ${d.pred_start} ~ ${d.pred_end} | 입력: ${d.source}`;
  } catch (_) {}
}

// ── 추천 & AI 브리핑 ──────────────────────────────────────────────────────────
async function loadRecommendation() {
  const data = await fetchJSON(`/api/recommend/${state.model}`);
  renderRecommendation(data);
}

async function loadSummary() {
  document.getElementById('summary-text').innerHTML = '<span class="summary-loading">분석 중...</span>';
  try {
    const data = await fetchJSON(`/api/summary/${state.model}`);
    const clean = (data.summary || '브리핑 생성 실패')
      .replace(/^#{1,6}\s*/gm, '')
      .replace(/\*{1,2}([^*]+)\*{1,2}/g, '$1')
      .trim();
    document.getElementById('summary-text').textContent = clean;
  } catch (e) {
    document.getElementById('summary-text').textContent = '브리핑 생성 실패';
  }
}

const REC_STATUS_COLOR = { '원활': '#1b7d3b', '보통': '#b07d00', '혼잡': '#c0392b' };

function _heatColor(rate, minR, maxR) {
  // 포트 내 상대 위치로 색상 계산 (초록→노랑→빨강)
  const RISK_HIGH = 0.12, RISK_MED = 0.06;
  if (rate >= RISK_HIGH) return { bg: '#fde8e8', border: '#e57373', text: '#b71c1c' };
  if (rate >= RISK_MED)  return { bg: '#fff8e1', border: '#ffd54f', text: '#795548' };
  return { bg: '#e8f5e9', border: '#81c784', text: '#1b5e20' };
}

function renderRecommendation(data) {
  const grid = document.getElementById('rec-grid');
  const bannerWrap = document.getElementById('best-banner-wrap');
  const days = data.days || [];

  grid.innerHTML = Object.entries(data.ports).map(([port, info]) => {
    const statusColor = REC_STATUS_COLOR[info.week_risk] || '#1b7d3b';
    const rates = info.daily_rates || [];
    const bestIdx = info.best_day - 1;

    const heatCells = rates.map((r, i) => {
      const c = _heatColor(r);
      const isBest = (i === bestIdx);
      return `
        <div class="heat-cell ${isBest ? 'heat-best' : ''}"
             style="background:${c.bg};border-color:${c.border}">
          <div class="heat-date">${(days[i] || '').split('/')[1]?.split('(')[0] || ''}</div>
          <div class="heat-dow">${(days[i] || '').replace(/.*\(/, '').replace(')', '')}</div>
          <div class="heat-val" style="color:${c.text}">${(r * 100).toFixed(1)}%</div>
          ${isBest ? '<div class="heat-star">★</div>' : ''}
        </div>`;
    }).join('');

    return `
      <div class="rec-card2">
        <div class="rec-card2-header">
          <span class="rec-port2">${port}항</span>
          <span class="rec-week-badge" style="color:${statusColor}">${info.week_risk}</span>
        </div>
        <div class="rec-action2">${info.recommendation}</div>
        <div class="heat-row">${heatCells}</div>
        <div class="rec-rate2">주간 평균 ${(info.avg_rate * 100).toFixed(1)}% · 최저 ${(info.best_rate * 100).toFixed(1)}%</div>
      </div>`;
  }).join('');

  if (data.best) {
    bannerWrap.innerHTML = `
      <div class="best-banner">
        이번 주 최적 입항: ${data.best.message}
      </div>`;
  }
}

// ── 예측 로드 & 렌더 ──────────────────────────────────────────────────────────
async function loadForecast() {
  const data = await fetchJSON(`/api/predict/${state.model}`);
  state.data.forecast = data;
  renderCards(data);
  renderForecastChart(data, state.forecastPort);
  renderForecastTable(data);
  loadRecommendation();
  loadSummary();
}

function renderCards(data) {
  PORTS.forEach(port => {
    const p = data.ports[port];
    document.getElementById(`max-${port}`).textContent = (p.max * 100).toFixed(1) + '%';
    const peakLabel = data.days[p.peak_day - 1] || `D+${p.peak_day}`;
    document.getElementById(`peak-${port}`).textContent = peakLabel;
    const badge = document.getElementById(`badge-${port}`);
    badge.textContent = RISK_LABEL[p.risk] || '-';
    badge.className = `risk-badge ${p.risk}`;
  });
}

function renderForecastChart(data, port) {
  const p = data.ports[port];
  const ctx = document.getElementById('forecast-chart').getContext('2d');
  if (forecastChart) forecastChart.destroy();
  forecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.days,
      datasets: [{
        label: `${port}항 예측 체선율`,
        data:  p.values,
        borderColor: COLORS.line,
        backgroundColor: 'rgba(29,111,164,0.08)',
        borderWidth: 2.5,
        pointRadius: 5,
        pointBackgroundColor: COLORS.line,
        tension: 0.35,
        fill: true,
      }]
    },
    options: { ...baseOptions(), animation: false }
  });
}

function renderForecastTable(data) {
  const tbody = document.getElementById('forecast-tbody');
  tbody.innerHTML = '';
  data.days.forEach((day, d) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${day}</td>` +
      PORTS.map(p => `<td>${(data.ports[p].values[d] * 100).toFixed(1)}%</td>`).join('');
    tbody.appendChild(tr);
  });
}

// ── 과거 데이터 로드 & 렌더 ───────────────────────────────────────────────────
async function loadHistory(port) {
  const data = await fetchJSON(`/api/daily/${port}`);
  renderHistoryChart(data, port);
  renderHistoryStats(data, port);
}

function renderHistoryChart(data, port) {
  const ctx = document.getElementById('history-chart').getContext('2d');
  if (historyChart) historyChart.destroy();
  historyChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.dates,
      datasets: [{
        label: `${port}항 체선율`,
        data: data.rate,
        borderColor: COLORS.line,
        backgroundColor: 'rgba(29,111,164,0.06)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.2,
        fill: true,
      }]
    },
    options: {
      ...baseOptions(),
      plugins: {
        ...baseOptions().plugins,
        tooltip: {
          callbacks: {
            label: ctx => `체선율: ${(ctx.parsed.y * 100).toFixed(1)}%`,
            title: ctx => ctx[0].label,
          }
        }
      },
      scales: { ...baseOptions().scales, x: { ...baseOptions().scales.x, ticks: { maxTicksLimit: 8, font: { size: 11 } } } }
    }
  });
}

function renderHistoryStats(data, port) {
  const rates = data.rate;
  const avg   = rates.reduce((a, b) => a + b, 0) / rates.length;
  const max   = Math.max(...rates);
  const cnt   = data.count.reduce((a, b) => a + b, 0) / data.count.length;

  document.getElementById('history-stats').innerHTML = `
    <div class="stat-item"><div class="stat-label">평균 체선율</div><div class="stat-val">${(avg*100).toFixed(1)}%</div></div>
    <div class="stat-item"><div class="stat-label">최고 체선율</div><div class="stat-val">${(max*100).toFixed(1)}%</div></div>
    <div class="stat-item"><div class="stat-label">평균 일입항 선박</div><div class="stat-val">${cnt.toFixed(0)}척</div></div>
    <div class="stat-item"><div class="stat-label">데이터 기간</div><div class="stat-val">${data.dates.length}일</div></div>
  `;
}

// ── 연쇄 혼잡 분석 로드 & 렌더 ───────────────────────────────────────────────
let ccfChart = null;
let attentionChart = null;
let backtestChart = null;
let eventRmseChart = null;

async function loadAnalysis() {
  await Promise.all([loadCCF(), loadAttention(), loadBacktest(), loadEventAnalysis(), loadEventSamples()]);
  setupClassifyDemo();
}

async function loadCCF() {
  const data = await fetchJSON('/api/ccf');
  const grid = document.getElementById('ccf-grid');

  const pairDesc = {
    '부산→울산': '동해안 인근 항만 — 빠른 전이',
    '부산→광양': '남해안 동일 항로',
    '부산→인천': '서해안 반대편 — 느린 전이',
    '울산→광양': '동·남해안 연계',
  };

  grid.innerHTML = Object.entries(data).map(([pair, d]) => `
    <div class="ccf-card">
      <div class="ccf-pair">${pair}</div>
      <div class="ccf-lag">최대 상관 lag: D+${d.lag}일 후 전이</div>
      <div class="ccf-corr">상관계수: ${d.corr >= 0 ? '+' : ''}${d.corr} | ${pairDesc[pair] || ''}</div>
    </div>`).join('');

  // CCF 막대 차트 (부산→광양 예시)
  const target = data['부산→광양'] || Object.values(data)[0];
  if (!target) return;
  const ctx = document.getElementById('ccf-chart').getContext('2d');
  if (ccfChart) ccfChart.destroy();
  ccfChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: (target.lags || []).map(l => `Lag ${l}`),
      datasets: Object.entries(data).map(([ pair, d], idx) => ({
        label: pair,
        data:  d.corrs,
        backgroundColor: ['rgba(29,111,164,0.7)', 'rgba(56,168,89,0.7)',
                          'rgba(220,100,50,0.7)',  'rgba(140,80,200,0.7)'][idx % 4],
        borderRadius: 3,
      }))
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: 'top',
          labels: { font: { family: 'Pretendard', size: 12 }, boxWidth: 12 } },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: r=${ctx.parsed.y.toFixed(3)}` } }
      },
      scales: {
        x: { grid: { color: '#f0f2f5' }, ticks: { font: { size: 11 } } },
        y: { grid: { color: '#f0f2f5' }, ticks: { font: { size: 11 } },
             title: { display: true, text: '상관계수 r', font: { size: 12 }, color: '#8b96a5' } }
      }
    }
  });
}

async function loadAttention() {
  const data = await fetchJSON('/api/attention');
  const ctx = document.getElementById('attention-chart').getContext('2d');
  if (attentionChart) attentionChart.destroy();

  const maxW = Math.max(...data.weights);
  const colors = data.weights.map(w =>
    `rgba(29,111,164,${0.15 + 0.85 * (w / maxW)})`
  );

  attentionChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{
        label: '평균 어텐션 가중치',
        data:  data.weights,
        backgroundColor: colors,
        borderRadius: 2,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `가중치: ${ctx.parsed.y.toFixed(4)}` } }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 10 }, maxTicksLimit: 10 } },
        y: { grid: { color: '#f0f2f5' }, ticks: { font: { size: 11 } },
             title: { display: true, text: 'Attention weight', font: { size: 12 }, color: '#8b96a5' } }
      }
    }
  });
}

async function loadBacktest() {
  const data = await fetchJSON('/api/backtest');
  const months = Object.keys(data[Object.keys(data)[0]] || {});
  const ctx = document.getElementById('backtest-chart').getContext('2d');
  if (backtestChart) backtestChart.destroy();
  const colors = {
    transformer:       '#1d6fa4',
    transformer_event: '#38a859',
    ensemble:          '#e06530',
  };
  const labels_map = {
    transformer:       'Transformer',
    transformer_event: 'Transformer+Event',
    ensemble:          '앙상블',
  };
  backtestChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: months,
      datasets: Object.entries(data).map(([key, monthly]) => ({
        label:           labels_map[key] || key,
        data:            months.map(m => monthly[m] || null),
        borderColor:     colors[key] || '#999',
        backgroundColor: (colors[key] || '#999') + '22',
        borderWidth:     2,
        pointRadius:     5,
        tension:         0.3,
        fill:            false,
      }))
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: 'top',
          labels: { font: { family: 'Pretendard', size: 12 }, boxWidth: 12 } },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: RMSE ${ctx.parsed.y.toFixed(4)}` } }
      },
      scales: {
        x: { grid: { color: '#f0f2f5' }, ticks: { font: { size: 12 } } },
        y: { grid: { color: '#f0f2f5' }, ticks: { font: { size: 11 }, callback: v => v.toFixed(3) },
             title: { display: true, text: 'RMSE (체선율)', font: { size: 12 }, color: '#8b96a5' },
             min: 0 }
      }
    }
  });
}

async function loadEventAnalysis() {
  const data = await fetchJSON('/api/event_period_analysis');

  // 윈도우 수 업데이트
  const nEvEl = document.getElementById('n-event-win');
  const nNmEl = document.getElementById('n-normal-win');
  if (nEvEl) nEvEl.textContent = data.n_event;
  if (nNmEl) nNmEl.textContent = data.n_normal;

  // 조건부 RMSE 그룹 막대 차트
  const cr = data.conditional_rmse;
  const modelKeys = ['transformer', 'transformer_event', 'ensemble'];
  const modelLabels = { transformer: 'Transformer', transformer_event: 'Transformer+Event', ensemble: '앙상블' };
  const ctx = document.getElementById('event-rmse-chart').getContext('2d');
  if (eventRmseChart) eventRmseChart.destroy();

  eventRmseChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: modelKeys.map(k => modelLabels[k]),
      datasets: [
        {
          label: '이벤트 기간 RMSE',
          data: modelKeys.map(k => cr[k]?.event ?? 0),
          backgroundColor: 'rgba(192,57,43,0.75)',
          borderRadius: 4,
        },
        {
          label: '평시 RMSE',
          data: modelKeys.map(k => cr[k]?.normal ?? 0),
          backgroundColor: 'rgba(29,111,164,0.65)',
          borderRadius: 4,
        },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: 'top',
          labels: { font: { family: 'Pretendard', size: 12 }, boxWidth: 12 } },
        tooltip: {
          callbacks: {
            label: ctx => {
              const k = modelKeys[ctx.dataIndex];
              const d = cr[k];
              const suffix = ctx.datasetIndex === 0
                ? ` (+${d?.diff_pct ?? 0}%p vs 평시)` : '';
              return `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(5)}${suffix}`;
            }
          }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 13 } } },
        y: { grid: { color: '#f0f2f5' }, ticks: { font: { size: 11 }, callback: v => v.toFixed(3) },
             title: { display: true, text: 'RMSE (체선율)', font: { size: 12 }, color: '#8b96a5' },
             min: 0 }
      }
    }
  });

  // 케이스 스터디 렌더
  renderCaseStudy(data.case_study, data.event_markers);
}

function renderCaseStudy(caseStudy, eventMarkers) {
  const wrap = document.getElementById('case-study-wrap');
  if (!wrap) return;
  const strikeSet = new Set((eventMarkers || []).filter(m => m.type === 'strike').map(m => m.date));
  const surgeSet  = new Set((eventMarkers || []).filter(m => m.type === 'surge').map(m => m.date));

  const rows = (caseStudy || []).map(c => {
    const isStrike = strikeSet.has(c.pred_date);
    const isSurge  = surgeSet.has(c.pred_date);
    const isGood   = c.error <= 0.008 && !isStrike && !isSurge;
    const actualPct = (c.actual_부산  * 100).toFixed(2);
    const predPct   = (c.pred_ensemble * 100).toFixed(2);
    const errorPct  = (c.error         * 100).toFixed(2);
    const sign = c.pred_ensemble < c.actual_부산 ? '↑' : (c.pred_ensemble > c.actual_부산 ? '↓' : '');
    let badge = '';
    if (isStrike) badge = '<span class="cs-badge miss">파업 당일</span>';
    else if (isSurge)  badge = '<span class="cs-badge surge">물동량급증</span>';
    else if (isGood)   badge = '<span class="cs-badge hit">정확 예측</span>';

    return `
      <div class="cs-row${isStrike ? ' cs-strike' : (isSurge ? ' cs-surge' : (isGood ? ' cs-good' : ''))}">
        <div class="cs-date">${c.pred_date}${badge}</div>
        <div class="cs-metrics">
          <span class="cs-chip"><span class="cs-chip-lbl">실제</span>${actualPct}%</span>
          <span class="cs-chip"><span class="cs-chip-lbl">예측</span>${predPct}%</span>
          <span class="cs-chip cs-chip-err${isStrike ? ' cs-chip-err-big' : ''}">
            <span class="cs-chip-lbl">오차</span>${sign}${errorPct}%p
          </span>
        </div>
      </div>`;
  }).join('');

  wrap.innerHTML = rows || '<p style="padding:16px;color:#8b96a5">데이터 없음</p>';
}

const EVENT_TYPE_KR = { strike: '파업', weather: '기상', surge: '물동량급증', normal: '정상' };
const EVENT_COLOR   = { strike: '#c0392b', weather: '#2980b9', surge: '#e67e22', normal: '#95a5a6' };

async function loadEventSamples() {
  const data = await fetchJSON('/api/event_samples');
  const grid = document.getElementById('event-samples-grid');

  // 통계 바
  const total = data.total || 0;
  const statsHtml = Object.entries(data.stats || {}).map(([type, cnt]) => `
    <div class="event-stat" style="border-left:3px solid ${EVENT_COLOR[type] || '#ccc'}">
      <div class="event-stat-type">${EVENT_TYPE_KR[type] || type}</div>
      <div class="event-stat-cnt">${cnt}건 (${(cnt/total*100).toFixed(1)}%)</div>
    </div>`).join('');

  const samplesHtml = (data.samples || []).map(s => `
    <div class="event-sample-card" style="border-left:3px solid ${EVENT_COLOR[s.event_type] || '#ccc'}">
      <div class="event-sample-meta">
        <span class="event-type-badge" style="background:${EVENT_COLOR[s.event_type] || '#ccc'}">${EVENT_TYPE_KR[s.event_type] || s.event_type}</span>
        <span class="event-date">${s.date}</span>
      </div>
      <div class="event-headline">${s.headline}</div>
      <div class="event-reason">${s.reason || ''}</div>
    </div>`).join('');

  grid.innerHTML = `<div class="event-stats-row">${statsHtml}</div>
    <div class="event-samples-list">${samplesHtml}</div>`;
}

function setupClassifyDemo() {
  const btn   = document.getElementById('classify-btn');
  const input = document.getElementById('classify-input');
  const result= document.getElementById('classify-result');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    const q = input.value.trim();
    if (!q) return;
    btn.disabled = true;
    result.innerHTML = '<span style="color:#8b96a5">분석 중...</span>';
    try {
      const d = await fetchJSON(`/api/classify_headline?q=${encodeURIComponent(q)}`);
      const color = EVENT_COLOR[d.event_type] || '#95a5a6';
      result.innerHTML = `
        <span class="event-type-badge" style="background:${color};font-size:14px">${EVENT_TYPE_KR[d.event_type] || d.event_type}</span>
        <span class="classify-reason">${d.reason || ''}</span>`;
    } catch(e) {
      result.innerHTML = `<span style="color:#c0392b">분류 실패: ${e.message}</span>`;
    }
    btn.disabled = false;
  });
}

// ── SHAP 피처 중요도 ──────────────────────────────────────────────────────────
let shapBarChart = null;

const GROUP_COLOR = {
  '래그':   '#1d6fa4',
  '이동평균': '#2e9e5b',
  '선박유형': '#e67e22',
  '기상':   '#8e44ad',
  '입항통계': '#16a085',
  '체선율':  '#c0392b',
  '시간':   '#7f8c8d',
  '기타':   '#bdc3c7',
};

let _shapData = null;

function _renderShap(top, groupSummary) {
  setTimeout(() => {
    const ctx = document.getElementById('shap-bar-chart').getContext('2d');
    if (shapBarChart) shapBarChart.destroy();
    shapBarChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: top.map(f => f.feature),
        datasets: [{
          label: 'SHAP 중요도',
          data:  top.map(f => f.importance),
          backgroundColor: top.map(f => GROUP_COLOR[f.group] || '#bdc3c7'),
          borderRadius: 3,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => {
                const f = top[ctx.dataIndex];
                return `${f.group} | SHAP: ${ctx.parsed.x.toFixed(6)}`;
              }
            }
          }
        },
        scales: {
          x: { grid: { color: '#f0f2f5' }, ticks: { font: { size: 11 } },
               title: { display: true, text: '평균 |SHAP|', font: { size: 11 }, color: '#8b96a5' } },
          y: { grid: { display: false }, ticks: { font: { size: 12 } } }
        }
      }
    });
  }, 0);

  const groupEl = document.getElementById('shap-group-list');
  if (groupEl && groupSummary) {
    groupEl.innerHTML = groupSummary.map(g => `
      <div class="shap-group-row">
        <span class="shap-group-dot" style="background:${GROUP_COLOR[g.group] || '#bdc3c7'}"></span>
        <span class="shap-group-name">${g.group}</span>
        <div class="shap-group-bar-wrap">
          <div class="shap-group-bar" style="width:${g.pct}%;background:${GROUP_COLOR[g.group] || '#bdc3c7'}"></div>
        </div>
        <span class="shap-group-pct">${g.pct}%</span>
      </div>`).join('');
  }
}

function switchShapPort(port) {
  if (!_shapData) return;
  let top, group;
  if (port === '전체') {
    top   = (_shapData.top_features || []).slice(0, 15);
    group = _shapData.group_summary;
  } else {
    const ps = (_shapData.port_shap || {})[port];
    if (!ps) return;
    top   = ps.top_features.slice(0, 15);
    group = ps.group_summary;
  }
  _renderShap(top, group);
}

async function loadShap() {
  let data;
  try {
    data = await fetchJSON('/api/shap');
  } catch (_) { return; }
  if (data.error) return;
  _shapData = data;

  _renderShap((data.top_features || []).slice(0, 15), data.group_summary);

  // 항만 탭 이벤트
  document.querySelectorAll('#shap-port-tabs .port-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#shap-port-tabs .port-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      switchShapPort(btn.dataset.shapPort);
    });
  });
}

// ── 모델 비교 로드 & 렌더 ─────────────────────────────────────────────────────
async function loadMetrics() {
  const data = await fetchJSON('/api/metrics');
  state.data.metrics = data;
  renderMetricTable(data);
  renderPortRmseChart(data);
  loadShap();
}

const MODEL_LABEL = {
  'transformer':       'Transformer',
  'transformer_event': 'Transformer + Event',
  'ensemble':          '앙상블 (LSTM + Transformer)',
};

function renderMetricTable(data) {
  const tbody = document.getElementById('metric-tbody');
  tbody.innerHTML = '';
  const bestRmse = Math.min(...Object.values(data).map(m => m.RMSE));
  Object.entries(data).forEach(([key, m]) => {
    const tr = document.createElement('tr');
    if (m.RMSE === bestRmse) tr.classList.add('best-row');
    tr.innerHTML = `
      <td>${MODEL_LABEL[key] || key}</td>
      <td>${m.MAE.toFixed(4)}</td>
      <td>${m.RMSE.toFixed(4)}${m.RMSE === bestRmse ? ' &nbsp;<small style="color:#1b7d3b;font-size:11px">최저</small>' : ''}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderPortRmseChart(data) {
  if (portRmseChart) { portRmseChart.destroy(); portRmseChart = null; }

  const datasets = Object.entries(data).map(([key, m], i) => ({
    label: MODEL_LABEL[key] || key,
    data: PORTS.map(p => m.port_rmse[p] || 0),
    backgroundColor: i === 0 ? 'rgba(29,111,164,0.75)' : 'rgba(52,168,83,0.75)',
    borderRadius: 4,
  }));

  // setTimeout(0) lets the browser repaint (display:none→block) before chart creation
  setTimeout(() => {
    const ctx = document.getElementById('port-rmse-chart').getContext('2d');
    portRmseChart = new Chart(ctx, {
      type: 'bar',
      data: { labels: PORTS.map(p => p + '항'), datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { position: 'top', labels: { font: { family: 'Pretendard', size: 12 }, boxWidth: 12 } },
          tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}` } }
        },
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 13, weight: '500' } } },
          y: { grid: { color: COLORS.grid }, ticks: { font: { size: 12 } },
               title: { display: true, text: 'RMSE', font: { size: 12 }, color: '#8b96a5' } }
        }
      }
    });
  }, 0);
}

// ── 이벤트 리스너 ──────────────────────────────────────────────────────────────
document.querySelectorAll('.nav-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
    if (btn.dataset.tab === 'history')  loadHistory(state.historyPort);
    if (btn.dataset.tab === 'comparison') loadMetrics();
    if (btn.dataset.tab === 'analysis') loadAnalysis();
  });
});

document.getElementById('model-select').addEventListener('change', e => {
  state.model = e.target.value;
  loadForecast();
});

// 예측 탭 항만 탭
document.querySelectorAll('#tab-forecast .port-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#tab-forecast .port-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.forecastPort = btn.dataset.port;
    if (state.data.forecast) renderForecastChart(state.data.forecast, state.forecastPort);
  });
});

// 과거 탭 항만 탭
document.querySelectorAll('#history-port-tabs .port-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#history-port-tabs .port-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.historyPort = btn.dataset.port;
    loadHistory(state.historyPort);
  });
});

// 항만 카드 클릭 → 해당 항만 차트로 이동
document.querySelectorAll('.port-card').forEach(card => {
  card.addEventListener('click', () => {
    const port = card.dataset.port;
    document.querySelectorAll('.port-card').forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    state.forecastPort = port;
    document.querySelectorAll('#tab-forecast .port-tab').forEach(b => {
      b.classList.toggle('active', b.dataset.port === port);
    });
    if (state.data.forecast) renderForecastChart(state.data.forecast, port);
  });
});

// ── 초기 로드 ─────────────────────────────────────────────────────────────────
loadDataStatus();
loadForecast();
