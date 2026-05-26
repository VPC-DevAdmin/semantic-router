'use strict';
// ===================================================================
// Semantic-routing cost demo - data-driven replay.
// Reads demo/data/demo_data.json (built by tools/build_demo_data.py).
// Nothing about tiers, models, pricing, or throughput is hardcoded.
// ===================================================================

const svgNS = 'http://www.w3.org/2000/svg';
const TIER_COLORS = {
  1: 'var(--t1)', 2: 'var(--t2)', 3: 'var(--t3)', 4: 'var(--t4)', 5: 'var(--t5)',
};
const TIER_HEX = { 1:'#1FB67A', 2:'#11A8C4', 3:'#F2A93B', 4:'#E8743B', 5:'#E64B5C' };

// Honor the OS "reduce motion" setting: skip trails, pulses, float-ups,
// and the cross-fade. The replay still advances - it just doesn't animate.
const REDUCED_MOTION = window.matchMedia
  && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Minimum on-screen dwell for the detail panel, per router speed. The
// stage runs at the real throughput, but a human can't read a full
// routed-vs-frontier comparison in a couple seconds, so the readable
// panel is gated to a comfortable reading pace (~20s at 1x) that only
// *shrinks* as you speed the router up.
const DWELL_MS = { 1: 20000, 2: 14000, 4: 10000 };
const dwellMs = () => DWELL_MS[speed] || 2500;

// Default model selection per tier + frontier (cost-optimal cloud picks).
// Applied if present in the dataset; otherwise the tier's first model.
const DEFAULT_PICKS = {
  1: 'gemini-3.1-flash-lite',
  2: 'gpt-5.4-nano',
  3: 'gpt-5.4-mini',
  4: 'gemini-3.1-pro-preview',
};
const DEFAULT_FRONTIER = 'Opus 4.7';

let DATA = null;           // full demo_data.json
let TIERS = [];            // [1,2,3,4,5] derived
let pickedModel = {};      // tier -> model name (1-4)
let pickedFrontier = null; // frontier model name
let referenceFrontier = null; // model whose measured output length anchors the baseline
let avgPromptTok = 0;      // mean prompt tokens across the set (frontier baseline)
let avgRefOutTok = 0;      // mean reference-frontier completion tokens across the set

// ---------- formatting ----------
const fmtInt = n => n.toLocaleString('en-US');
function fmtMoney(n) {
  const a = Math.abs(n);
  if (a >= 1e9) return '$' + (n/1e9).toFixed(2) + 'B';
  if (a >= 1e6) return '$' + (n/1e6).toFixed(2) + 'M';
  if (a >= 1e3) return '$' + (n/1e3).toFixed(0) + 'K';
  return '$' + n.toFixed(2);
}
// Dollar amounts show 2 decimals; sub-cent per-query costs keep 4 so they
// don't collapse to $0.00.
const fmtCost = n => '$' + n.toFixed(Math.abs(n) < 0.01 ? 4 : 2);
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const truncate = (s,n) => s.length>n ? s.slice(0,n-1)+'…' : s;

// Icon per verdict so color isn't the sole signal (accessibility).
const VERDICT_ICON = { adequate: '✓', marginal: '≈', failure: '✕' };
const SCORE_MAX = 4;  // scoring dimensions are on a 1–4 scale
// A labeled segmented bar: N of SCORE_MAX segments filled.
function scoreBar(label, score) {
  const n = Math.max(0, Math.min(SCORE_MAX, score || 0));
  let segs = '';
  for (let i = 1; i <= SCORE_MAX; i++) {
    segs += `<span class="seg${i <= n ? ' on' : ''}"></span>`;
  }
  return `<span class="score-bar"><span class="score-name">${label}</span>`
       + `<span class="segs">${segs}</span>`
       + `<span class="score-num">${n}/${SCORE_MAX}</span></span>`;
}

// ===================================================================
// BOOT
// ===================================================================
fetch('data/demo_data.json')
  .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
  .then(init)
  .catch(err => {
    document.getElementById('loadStatus').innerHTML =
      `<div class="errbox">Couldn't load <code>data/demo_data.json</code>: ${esc(err.message)}.<br>` +
      `Run <code>make demo-data</code> to build it, then serve via <code>make demo</code>.</div>`;
  });

function init(data) {
  DATA = data;
  TIERS = data.meta.tiers.map(t => t.level);          // routed tiers (1-4)
  const frontierLevel = 5;
  if (!TIERS.includes(frontierLevel)) TIERS.push(frontierLevel);  // frontier shown as tier 5

  // Presentation defaults (cost-optimal cloud picks; see analysis). If a
  // named model isn't in the loaded dataset, fall back to that tier's
  // first model so the demo still works with any dataset.
  data.meta.tiers.forEach(t => {
    const want = DEFAULT_PICKS[t.level];
    pickedModel[t.level] = t.models.some(m => m.model === want) ? want : t.models[0]?.model;
  });
  pickedFrontier = data.meta.frontier_models.some(m => m.model === DEFAULT_FRONTIER)
    ? DEFAULT_FRONTIER : data.meta.frontier_models[0]?.model;
  // The reference frontier anchors the baseline OUTPUT LENGTH. Switching
  // the frontier model must change only the per-token RATE, never the
  // assumed answer length - otherwise a tersely-answering pricier model
  // can paradoxically lower the baseline. We anchor to the default
  // frontier (frontier_models[0]) as the representative answer length;
  // all frontier candidates are then priced against this same length.
  referenceFrontier = data.meta.frontier_models[0]?.model || null;

  precomputeAnchors();
  buildPickers();
  buildStage();
  buildTierDiagram();
  initCalculator();
  initQueryPickers();

  document.getElementById('loadStatus').style.display = 'none';
  document.getElementById('demoRoot').style.display = 'block';

  // throughput stat from the data (measured)
  startEngine();
}

// Precompute the frontier-baseline length anchors (mean prompt tokens +
// mean reference-frontier output tokens). frontierCostFor falls back to
// these when a query lacks a per-query reference answer.
function precomputeAnchors() {
  let pSum = 0, oSum = 0, n = 0;
  DATA.queries.forEach(q => {
    const ref = q.frontier_answers.find(a => a.model === referenceFrontier)
              || q.frontier_answers[0];
    if (ref) {
      pSum += q.prompt_tokens || ref.prompt_tokens || 0;
      oSum += ref.completion_tokens || 0;
      n++;
    }
  });
  avgPromptTok = n ? pSum / n : 0;
  avgRefOutTok = n ? oSum / n : 0;
}
const rateOf = model => DATA.pricing[model] || {};

// Rate-driven frontier baseline for one query: the assumed answer LENGTH
// is fixed (the reference frontier's measured output for this query),
// and only the per-token RATE varies with the picked frontier model. So
// a pricier frontier always costs more - terseness can't invert it.
function frontierCostFor(q, model) {
  const ref = q.frontier_answers.find(a => a.model === referenceFrontier)
            || q.frontier_answers[0];
  const outTok = ref ? (ref.completion_tokens || 0) : avgRefOutTok;
  const pTok = q.prompt_tokens || (ref && ref.prompt_tokens) || avgPromptTok;
  const r = rateOf(model);
  return pTok * (r.in_per_1m || 0) / 1e6 + outTok * (r.out_per_1m || 0) / 1e6;
}
// Per-query routed + frontier cost under the current picks. This is the
// SINGLE source of truth for cost: the live replay sums it per query, and
// the hero calculator averages it over the whole set — so the two always
// agree on the savings percentage.
function queryCosts(q) {
  const tier = q.routed_tier;
  const frontierCost = frontierCostFor(q, pickedFrontier);
  let routedCost;
  if (tier === 5 || !q.routed_answers['tier' + tier]) {
    // Routed to the frontier tier — routed answer IS a frontier model.
    routedCost = frontierCost;
  } else {
    const list = q.routed_answers['tier' + tier];
    const a = list.find(x => x.model === pickedModel[tier]) || list[0];
    routedCost = a ? a.cost_usd : 0;
  }
  return { routedCost, frontierCost };
}

// Mean routed + frontier cost per query across the whole benchmark set,
// under the current picks. The replay is a uniform random sample of this
// same population, so its cumulative ratio converges to these means.
function populationAverages() {
  let routed = 0, frontier = 0, n = 0;
  DATA.queries.forEach(q => {
    const c = queryCosts(q);
    routed += c.routedCost; frontier += c.frontierCost; n++;
  });
  return n ? { routed: routed / n, frontier: frontier / n } : { routed: 0, frontier: 0 };
}

// ===================================================================
// PICKERS
// ===================================================================
function buildPickers() {
  const bar = document.getElementById('pickerBar');
  bar.innerHTML = '';
  DATA.meta.tiers.forEach(t => {
    bar.appendChild(makePicker(`Tier ${t.level}`, TIER_HEX[t.level], t.models, m => {
      pickedModel[t.level] = m;
      onPickerChange();
    }, false, pickedModel[t.level]));
  });
  bar.appendChild(makePicker('Frontier', TIER_HEX[5], DATA.meta.frontier_models, m => {
    pickedFrontier = m;
    onPickerChange();
  }, true, pickedFrontier));
}
function makePicker(label, hex, models, onChange, isFrontier, selected) {
  const div = document.createElement('div');
  div.className = 'picker' + (isFrontier ? ' frontier' : '');
  const opts = models.map(m =>
    `<option value="${esc(m.model)}"${m.model===selected?' selected':''}>${esc(m.model)} · ${esc(m.provider||'')}</option>`
  ).join('');
  div.innerHTML = `
    <div class="ptier"><span class="pdot" style="background:${hex}"></span>${esc(label)}</div>
    <select>${opts}</select>`;
  div.querySelector('select').addEventListener('change', e => onChange(e.target.value));
  return div;
}
function onPickerChange() {
  buildTierDiagram();
  updateCalc();
  // re-render the currently displayed query against the new selections
  if (lastShown) showDetail(lastShown);
}

// ===================================================================
// STAGE (paths + tier boxes generated for N tiers)
// ===================================================================
const pathLayer = () => document.getElementById('pathLayer');
const tierBoxLayer = () => document.getElementById('tierBoxLayer');
const ballLayer = () => document.getElementById('ballLayer');
const pulseLayer = () => document.getElementById('pulseLayer');
const overlayLayer = () => document.getElementById('overlayLayer');
const queueLayer = () => document.getElementById('queueLayer');

const ROUTER = { x: 714, y: 280 };  // right edge of router box, vertical center
const BOX = { x: 960, w: 220, h: 70 };

function tierBoxY(level, n) {
  const top = 60, bottom = 500;
  const gap = (bottom - top - BOX.h) / Math.max(1, n - 1);
  return top + (level - 1) * gap;
}

function buildStage() {
  const n = TIERS.length;
  pathLayer().innerHTML = '';
  tierBoxLayer().innerHTML = '';
  TIERS.forEach(level => {
    const y = tierBoxY(level, n);
    const cy = y + BOX.h/2;
    // path router -> box
    const path = document.createElementNS(svgNS, 'path');
    path.setAttribute('id', 'pathT' + level);
    path.setAttribute('class', 'path-tier');
    path.setAttribute('stroke', TIER_HEX[level]);
    path.setAttribute('d', `M ${ROUTER.x} ${ROUTER.y} C ${(ROUTER.x+BOX.x)/2} ${ROUTER.y}, ${(ROUTER.x+BOX.x)/2} ${cy}, ${BOX.x} ${cy}`);
    pathLayer().appendChild(path);

    // box
    const g = document.createElementNS(svgNS, 'g');
    const isFrontier = level === 5;
    const label = isFrontier ? 'FRONTIER' : 'TIER ' + level;
    const modelName = isFrontier ? pickedFrontier : pickedModel[level];
    g.innerHTML = `
      <rect class="tier-bg" id="tierBoxT${level}" x="${BOX.x}" y="${y}" width="${BOX.w}" height="${BOX.h}" rx="10"/>
      <circle cx="${BOX.x+26}" cy="${y+30}" r="7" fill="${TIER_HEX[level]}"/>
      <text x="${BOX.x+44}" y="${y+26}" class="label-md">${label}</text>
      <text x="${BOX.x+44}" y="${y+44}" class="small" id="boxModelT${level}">${esc(truncate(modelName||'',26))}</text>`;
    tierBoxLayer().appendChild(g);
  });
}

function refreshBoxLabels() {
  TIERS.forEach(level => {
    const el = document.getElementById('boxModelT' + level);
    if (el) el.textContent = truncate((level===5 ? pickedFrontier : pickedModel[level]) || '', 26);
  });
}

// ===================================================================
// BALL ANIMATION
// ===================================================================
let inFlight = [];
class Ball {
  constructor(level) {
    this.path = document.getElementById('pathT' + level);
    this.len = this.path.getTotalLength();
    this.t0 = gameTime;
    this.dur = 1.2;
    this.level = level;
    this.trail = [];
    for (let i = 0; i < (REDUCED_MOTION ? 0 : 5); i++) {
      const c = document.createElementNS(svgNS, 'circle');
      c.setAttribute('r', Math.max(1.5, 5 - i*0.7));
      c.setAttribute('fill', TIER_HEX[level]);
      c.style.opacity = (1 - i/5) * 0.5;
      ballLayer().appendChild(c); this.trail.push(c);
    }
    this.ball = document.createElementNS(svgNS, 'circle');
    this.ball.setAttribute('r', 6);
    this.ball.setAttribute('fill', TIER_HEX[level]);
    ballLayer().appendChild(this.ball);
    this.hist = [];
    this.path.classList.add('glow');
  }
  update() {
    const t = Math.min((gameTime - this.t0)/this.dur, 1);
    const e = 1 - Math.pow(1-t, 2);
    const p = this.path.getPointAtLength(this.len * e);
    this.ball.setAttribute('cx', p.x); this.ball.setAttribute('cy', p.y);
    this.hist.unshift({x:p.x,y:p.y}); this.hist = this.hist.slice(0,5);
    this.trail.forEach((c,i) => { const h=this.hist[i]||p; c.setAttribute('cx',h.x); c.setAttribute('cy',h.y); });
    if (t >= 1) { this.destroy(); this.path.classList.remove('glow'); bumpBox(this.level); return false; }
    return true;
  }
  destroy() { this.ball.remove(); this.trail.forEach(c=>c.remove()); }
}
function bumpBox(level) {
  const box = document.getElementById('tierBoxT'+level);
  if (!box) return;
  box.setAttribute('stroke', TIER_HEX[level]);
  box.style.filter = `drop-shadow(0 0 10px ${TIER_HEX[level]})`;
  clearTimeout(box._t);
  box._t = setTimeout(() => { box.removeAttribute('stroke'); box.style.filter=''; }, 350);
}

// Brief accent glow on the chassis when a query is dispatched - replaces
// the old expanding ring that engulfed and obscured the box.
function pulseRouter() {
  const c = document.getElementById('routerChassis');
  if (!c) return;
  c.style.filter = 'drop-shadow(0 0 7px rgba(0,155,215,0.85))';
  clearTimeout(c._pulse);
  c._pulse = setTimeout(() => { c.style.filter = ''; }, 220);
}
// Cap how many "+$ saved" callouts float at once and stagger their
// vertical offset so labels don't stack on top of each other.
let activeTags = 0;
const MAX_TAGS = 4;
function savedTag(level, savings) {
  if (savings <= 0 || REDUCED_MOTION || activeTags >= MAX_TAGS) return;
  const cy = tierBoxY(level, TIERS.length) + BOX.h/2;
  const jitter = (activeTags - (MAX_TAGS-1)/2) * 16;  // spread vertically
  const tag = document.createElement('div');
  tag.className = 'float-tag';
  tag.style.left = (820/1200*100)+'%';
  tag.style.top  = ((cy + jitter)/560*100)+'%';
  tag.textContent = '+' + fmtCost(savings) + ' saved';
  overlayLayer().appendChild(tag);
  activeTags++;
  setTimeout(() => { tag.remove(); activeTags--; }, 1500);
}

// Incoming-queries column: spawn a faint token on the left that drifts
// toward the router, so the empty "INCOMING QUERIES" lane reads as live.
function queueToken(q) {
  if (REDUCED_MOTION) return;
  const layer = queueLayer();
  if (!layer || layer.childElementCount > 12) return;
  const t = document.createElementNS(svgNS, 'text');
  const y = 70 + Math.random() * 380;
  t.setAttribute('x', 120);
  t.setAttribute('y', y);
  t.setAttribute('class', 'queue-token');
  t.textContent = truncate((q.prompt || '').replace(/\s+/g, ' '), 22);
  layer.appendChild(t);
  const t0 = performance.now();
  (function step(now) {
    const k = Math.min((now - t0) / 800, 1);
    t.setAttribute('x', 120 + k * 250);
    t.style.opacity = k < 0.3 ? k / 0.3 : (1 - (k - 0.3) / 0.7);
    if (k < 1) requestAnimationFrame(step); else t.remove();
  })(t0);
}

// ===================================================================
// COST + VERDICT for a query, given current picks
// ===================================================================
function resolveQuery(q) {
  const tier = q.routed_tier;
  const frontierAns = q.frontier_answers.find(a => a.model === pickedFrontier) || q.frontier_answers[0];
  const { routedCost, frontierCost } = queryCosts(q);

  let routedModel, routedAns;
  if (tier === 5 || !q.routed_answers['tier'+tier]) {
    // Routed to the frontier tier: the "routed answer" IS a frontier model.
    routedModel = pickedFrontier;
    routedAns = frontierAns;
  } else {
    const list = q.routed_answers['tier'+tier];
    routedAns = list.find(a => a.model === pickedModel[tier]) || list[0];
    routedModel = routedAns ? routedAns.model : pickedModel[tier];
  }

  // verdicts: one per evaluator, keyed routed|frontier|evaluator
  const verdicts = DATA.meta.evaluators.map(ev => {
    const key = `${routedModel}|${pickedFrontier}|${ev}`;
    return { evaluator: ev, v: q.evaluations[key] || null };
  });

  return { tier, routedModel, routedAns, routedCost, frontierAns, frontierCost, verdicts };
}

// ===================================================================
// LIVE TALLY + DETAIL PANEL
// ===================================================================
let cumulative = { count: 0, routed: 0, baseline: 0 };
let recentTimes = [];
let lastShown = null;

// Rolling routing-latency window. The stage moves at the real (fast)
// throughput, so a per-query "classified in Xms" readout just flickers -
// instead we keep the last N samples and show their average, updated at a
// human-readable cadence (see frame()).
const LAT_WINDOW = 20;
let latWindow = [];
function recordLatency(ms) {
  if (ms == null) return;
  latWindow.push(ms);
  if (latWindow.length > LAT_WINDOW) latWindow.shift();
}
const avgLatency = () =>
  latWindow.length ? latWindow.reduce((a, b) => a + b, 0) / latWindow.length : 0;

const el = id => document.getElementById(id);

function spawnQuery(q) {
  const r = resolveQuery(q);
  queueToken(q);
  if (!REDUCED_MOTION) pulseRouter();
  recordLatency(q.routing_latency_ms);
  el('ledAct').style.opacity = Math.random() > 0.5 ? 1 : 0.4;

  const ball = new Ball(r.tier);
  inFlight.push(ball);

  cumulative.count++;
  cumulative.routed += r.routedCost;
  cumulative.baseline += r.frontierCost;
  recentTimes.push(performance.now());
  updateLiveStats();
  pushChart();
  savedTag(r.tier, r.frontierCost - r.routedCost);
}

function updateLiveStats() {
  el('liveCount').textContent = fmtInt(cumulative.count);
  el('liveRouted').textContent = fmtCost(cumulative.routed);
  el('liveBaseline').textContent = fmtCost(cumulative.baseline);
  const saved = cumulative.baseline - cumulative.routed;
  el('liveSavings').textContent = fmtCost(saved);
  const pct = cumulative.baseline>0 ? Math.round(saved/cumulative.baseline*100) : 0;
  el('livePct').textContent = pct + '% lower';
  const sv = el('liveSavings'); sv.classList.add('flash'); setTimeout(()=>sv.classList.remove('flash'),400);
}

function showDetail(q) {
  lastShown = q;
  // 200ms cross-fade so the panel doesn't hard-cut between queries.
  const card = document.getElementById('recentCard');
  if (card && !REDUCED_MOTION) {
    card.classList.add('swapping');
    clearTimeout(card._swap);
    card._swap = setTimeout(() => card.classList.remove('swapping'), 180);
  }
  const r = resolveQuery(q);
  const rq = el('recentQ'), exBtn = el('expandQ');
  rq.textContent = q.prompt || '';
  // Reset to the clamped 2-line view on each new query, then reveal the
  // expander only if the prompt actually overflows.
  rq.classList.remove('expanded');
  if (exBtn) {
    exBtn.textContent = '⌄ Show full query';
    requestAnimationFrame(() => {
      exBtn.hidden = rq.scrollHeight <= rq.clientHeight + 1;
    });
  }
  const pill = el('recentPill');
  pill.style.background = TIER_HEX[r.tier] + '22';
  pill.style.color = TIER_HEX[r.tier];
  pill.textContent = (r.tier===5 ? 'Frontier · needed top-tier' : 'Tier ' + r.tier);

  // routed column
  el('ansModelR').textContent = r.routedModel + (r.tier===5 ? ' (routed to the right tier)' : '');
  el('ansCostR').textContent = 'cost: ' + fmtCost(r.routedCost);
  el('ansBodyR').textContent = (r.routedAns && r.routedAns.answer) || '-';
  // frontier column
  el('ansModelB').textContent = (r.frontierAns ? r.frontierAns.model : pickedFrontier) + ' · frontier';
  el('ansCostB').textContent = 'cost: ' + fmtCost(r.frontierCost);
  el('ansBodyB').textContent = (r.frontierAns && r.frontierAns.answer) || '-';

  // verdicts
  const rows = el('verdictRows');
  if (r.tier === 5) {
    rows.innerHTML = `<div class="verdict-row"><div class="verdict-detail" style="grid-column:1/-1;color:var(--text-muted);">
      This query was routed to the frontier tier: the router judged it needed top-tier quality and delivered it. Routed answer and baseline are the same model, so there's no routed-vs-frontier comparison to score.</div></div>`;
    return;
  }
  rows.innerHTML = r.verdicts.map((row, i) => {
    if (!row.v) {
      return `<div class="verdict-row">
        <div class="verdict-evaluator">Evaluator ${i+1}<span class="sub">${esc(row.evaluator)}</span></div>
        <div class="verdict-detail" style="grid-column:2/-1;color:var(--text-muted);">no verdict for this model pairing</div></div>`;
    }
    const v = row.v; const cls = v.verdict.toLowerCase();
    const sc = v.scores;
    return `<div class="verdict-row">
      <div class="verdict-evaluator">Evaluator ${i+1}<span class="sub">${esc(row.evaluator)}</span></div>
      <div><span class="verdict-badge ${cls}"><span class="vicon" aria-hidden="true">${VERDICT_ICON[cls]||''}</span>${esc(v.verdict)}</span></div>
      <div class="verdict-detail">
        <div class="verdict-rationale">${esc(v.rationale)}</div>
        <div class="verdict-scores">
          ${scoreBar('correctness', sc.correctness)}
          ${scoreBar('completeness', sc.completeness)}
          ${scoreBar('fitness', sc.fitness_for_purpose)}
          ${scoreBar('soundness', sc.soundness)}
        </div>
      </div></div>`;
  }).join('');
}

// ===================================================================
// CHART
// ===================================================================
const chartHistory = [];
const CHART_WINDOW = 60000;
function pushChart() { chartHistory.push({t:performance.now(), routed:cumulative.routed, baseline:cumulative.baseline}); }
function renderChart() {
  const now = performance.now();
  while (chartHistory.length && now - chartHistory[0].t > CHART_WINDOW) chartHistory.shift();
  const rA=el('routedArea'), fL=el('frontierLine'), sA=el('savingsArea'), sl=el('savingsLabel');
  if (chartHistory.length < 2) {
    rA.setAttribute('d',''); fL.setAttribute('d',''); sA.setAttribute('d','');
    if (sl) sl.style.display = 'none';
    return;
  }
  const ws = now - CHART_WINDOW;
  const minR = chartHistory[0].routed, minB = chartHistory[0].baseline;
  const maxB = chartHistory[chartHistory.length-1].baseline;
  const yMax = Math.max(maxB - minB, 1e-6);
  el('chartYMax').textContent = fmtCost(yMax);
  const xOf = t => ((t-ws)/CHART_WINDOW)*1200;
  const yOf = (v,b) => 122 - ((v-b)/yMax)*108;
  const rp = chartHistory.map(p=>({x:xOf(p.t),y:yOf(p.routed,minR)}));
  const fp = chartHistory.map(p=>({x:xOf(p.t),y:yOf(p.baseline,minB)}));
  rA.setAttribute('d', `M ${rp[0].x} 122 ${rp.map(p=>`L ${p.x} ${p.y}`).join(' ')} L ${rp[rp.length-1].x} 122 Z`);
  fL.setAttribute('d', `M ${fp[0].x} ${fp[0].y} ${fp.slice(1).map(p=>`L ${p.x} ${p.y}`).join(' ')}`);
  sA.setAttribute('d', `M ${fp[0].x} ${fp[0].y} ${fp.slice(1).map(p=>`L ${p.x} ${p.y}`).join(' ')} ${[...rp].reverse().map(p=>`L ${p.x} ${p.y}`).join(' ')} Z`);

  // Label the savings band with the running total saved, sat in the band
  // at the right edge where the gap is widest.
  if (sl) {
    const saved = cumulative.baseline - cumulative.routed;
    const lastF = fp[fp.length-1].y, lastR = rp[rp.length-1].y;
    if (saved > 0 && (lastR - lastF) > 6) {
      sl.textContent = fmtCost(saved) + ' saved';
      sl.setAttribute('y', Math.max(16, (lastF + lastR) / 2 + 4));
      sl.style.display = '';
    } else {
      sl.style.display = 'none';
    }
  }
}

// ===================================================================
// MAIN LOOP + SPAWN
// ===================================================================
let gameTime = 0, lastFrame = performance.now();
let speed = 1, paused = false;
let lastLatPaint = 0;   // throttle for the rolling-latency readout
// Detail-panel pacing (decoupled from the router's spawn rate).
let pinned = false;         // freeze the panel on the current query
let pendingQ = null;        // most-recent spawned query, queued for display
let nextDueAt = 0;          // wall-clock time the panel may next advance
let onScreen = true;        // set false by IntersectionObserver when scrolled away

function frame(now) {
  const dt = (now - lastFrame)/1000; lastFrame = now;
  if (!paused) gameTime += dt * speed;
  inFlight = inFlight.filter(b => b.update());
  recentTimes = recentTimes.filter(t => now - t < 3000);
  // Visible throughput (sampled) - but headline stat is the MEASURED rate.
  el('liveQps').textContent = DATA.meta.throughput_qps.toFixed(1);
  el('qpsInSvg').textContent = DATA.meta.throughput_qps.toFixed(1);
  // Rolling average routing latency - updated at a readable cadence (not
  // per-query) so the number is stable instead of strobing.
  if (now - lastLatPaint > 400) {
    lastLatPaint = now;
    const r = el('avgLatReadout');
    if (r) r.textContent = latWindow.length
      ? `avg routing ${Math.round(avgLatency())} ms`
      : 'awaiting queries…';
  }
  renderChart();

  // Advance the readable detail panel only after the min dwell elapses.
  if (!paused && !pinned && pendingQ && now >= nextDueAt) {
    showDetail(pendingQ);
    pendingQ = null;
    nextDueAt = now + dwellMs();
  }
  updateCountdown(now);

  requestAnimationFrame(frame);
}

function updateCountdown(now) {
  const cd = el('nextIn');
  if (!cd) return;
  if (pinned) { cd.textContent = 'pinned'; return; }
  const secs = Math.max(0, Math.ceil((nextDueAt - now) / 1000));
  cd.textContent = pendingQ || secs > 0 ? `next in ${secs}s` : '';
}

// Spawn cadence: balls fly at the real measured throughput so the stage
// conveys true speed; the readable detail panel advances on its own
// dwell timer (see frame()).
let spawnTimer = null;
function spawnTick() {
  // Don't spawn while hidden/off-screen: rAF (which advances + removes
  // balls) is throttled in background tabs, so setTimeout-driven spawns
  // would pile up. Also cap concurrent balls as a belt-and-suspenders guard.
  // Skip random spawn when the user has picked a specific query - the
  // detail panel is then pinned to it (selectQuery handles its own spawn).
  if (!paused && DATA && !document.hidden && onScreen && !selectedQueryId && inFlight.length < 80) {
    const pool = queriesInCategory(selectedCategory);
    const q = pool.length
      ? pool[Math.floor(Math.random()*pool.length)]
      : DATA.queries[Math.floor(Math.random()*DATA.queries.length)];
    spawnQuery(q);
    pendingQ = q;   // queue it for the detail panel's next dwell window
  }
  spawnTimer = setTimeout(spawnTick, spawnInterval());
}
function spawnInterval() {
  const qps = DATA.meta.throughput_qps * speed;
  return Math.max(12, 1000 / qps);   // cap so very high speeds don't lock the loop
}

// Manually advance to the latest query, resetting the dwell window.
function advanceDetail() {
  const q = pendingQ || lastShown
    || DATA.queries[Math.floor(Math.random()*DATA.queries.length)];
  if (!q) return;
  showDetail(q);
  pendingQ = null;
  nextDueAt = performance.now() + dwellMs();
}

function startEngine() {
  requestAnimationFrame(frame);
  spawnTick();
  // seed the detail panel quickly
  showDetail(DATA.queries[Math.floor(Math.random()*DATA.queries.length)]);
  nextDueAt = performance.now() + dwellMs();
}

// ===================================================================
// CONTROLS
// ===================================================================
document.getElementById('playBtn').addEventListener('click', e => {
  paused = !paused;
  e.target.textContent = paused ? '▶ Play' : '⏸ Pause';
  el('serverStatus').textContent = paused ? 'PAUSED' : 'CLASSIFYING';
});
document.getElementById('resetBtn').addEventListener('click', () => {
  cumulative = { count:0, routed:0, baseline:0 };
  inFlight.forEach(b=>b.destroy()); inFlight = [];
  chartHistory.length = 0; recentTimes = []; latWindow = [];
  ballLayer().innerHTML=''; pulseLayer().innerHTML=''; overlayLayer().innerHTML='';
  updateLiveStats();
});
// ===================================================================
// CATEGORY + QUERY PICKERS (replace the router-speed multiplier)
// ===================================================================
// The stage now always runs at real measured throughput. In its place we
// let the user explore by category, and optionally pin to one specific
// query. State:
//   selectedCategory: 'all' or a specializations label.
//   selectedQueryId : null (random within category) or a specific query id.
let selectedCategory = 'all';
let selectedQueryId = null;

const cap = s => s ? s[0].toUpperCase() + s.slice(1) : s;
const CAT_PALETTE = {
  general:'#9DB1C4', reasoning:'#E64B5C', math:'#F2A93B',
  code:'#11A8C4', creative:'#1FB67A',
};

function queriesInCategory(cat) {
  if (cat === 'all') return DATA.queries;
  return DATA.queries.filter(q => (q.specializations || []).includes(cat));
}

function categoryCounts() {
  const counts = { all: DATA.queries.length };
  DATA.queries.forEach(q => (q.specializations || []).forEach(s => {
    counts[s] = (counts[s] || 0) + 1;
  }));
  return counts;
}

// Sync the Pin button label/state when the picker drives pinning.
function setPinned(b) {
  pinned = !!b;
  const btn = document.getElementById('pinBtn');
  if (btn) {
    btn.classList.toggle('active', pinned);
    btn.setAttribute('aria-pressed', String(pinned));
    btn.textContent = pinned ? '📌 Pinned' : '📌 Pin';
  }
  if (!pinned) nextDueAt = performance.now() + dwellMs();
}

function closeAllPickers() {
  document.querySelectorAll('.qpicker.open').forEach(p => {
    p.classList.remove('open');
    const f = p.querySelector('.qpicker-field');
    if (f) f.setAttribute('aria-expanded', 'false');
  });
}

function selectCategory(cat) {
  selectedCategory = cat;
  selectedQueryId = null;
  document.getElementById('catValueText').textContent =
    cat === 'all' ? 'All categories' : cap(cat);
  document.getElementById('qValueText').textContent = 'Random in category';
  // mark selected in the category menu
  document.querySelectorAll('#catMenu [data-cat]').forEach(b => {
    b.classList.toggle('selected', b.dataset.cat === cat);
  });
  const search = document.getElementById('qSearch');
  if (search) search.value = '';
  renderQueryList('');
  setPinned(false);
}

function selectQuery(id) {
  if (!id) {
    selectedQueryId = null;
    document.getElementById('qValueText').textContent = 'Random in category';
    // mark "Random in category" as selected in the menu
    document.querySelectorAll('#qList [data-qid]').forEach(b => {
      b.classList.toggle('selected', b.dataset.qid === '__rand');
    });
    setPinned(false);
    return;
  }
  const q = DATA.queries.find(x => x.id === id);
  if (!q) return;
  selectedQueryId = id;
  document.getElementById('qValueText').textContent =
    truncate((q.prompt || '').replace(/\s+/g, ' '), 60);
  document.querySelectorAll('#qList [data-qid]').forEach(b => {
    b.classList.toggle('selected', b.dataset.qid === id);
  });
  // animate the picked query through the router and pin the detail panel
  spawnQuery(q);
  showDetail(q);
  setPinned(true);
}

function renderQueryList(filter) {
  const list = document.getElementById('qList');
  if (!list) return;
  const pool = queriesInCategory(selectedCategory);
  const f = (filter || '').trim().toLowerCase();
  const matches = f ? pool.filter(q => (q.prompt || '').toLowerCase().includes(f)) : pool;
  let html = `
    <button class="qpicker-option ${selectedQueryId ? '' : 'selected'}" type="button" role="option" data-qid="__rand">
      <span class="opt-tier-dot" style="background:#9DB1C4"></span>
      <span>Random in category</span>
      <span class="opt-meta">${pool.length}</span>
    </button>`;
  if (!matches.length) {
    html += `<div class="qpicker-empty">no matches</div>`;
  } else {
    const MAX = 200;  // cap rendered options so the DOM stays light
    matches.slice(0, MAX).forEach(q => {
      const t = q.routed_tier || 1;
      const short = truncate((q.prompt || '').replace(/\s+/g, ' '), 80);
      const isSel = q.id === selectedQueryId;
      html += `
        <button class="qpicker-option${isSel ? ' selected' : ''}" type="button" role="option" data-qid="${esc(q.id)}">
          <span class="opt-tier" style="background:${TIER_HEX[t]}22;color:${TIER_HEX[t]};">
            <span class="opt-tier-dot" style="background:${TIER_HEX[t]}"></span>T${t}
          </span>
          <span>${esc(short)}</span>
        </button>`;
    });
    if (matches.length > MAX) {
      html += `<div class="qpicker-empty">…${matches.length - MAX} more — type to filter</div>`;
    }
  }
  list.innerHTML = html;
}

function initQueryPickers() {
  // Build category menu
  const counts = categoryCounts();
  const order = ['general', 'reasoning', 'math', 'code', 'creative'];
  const cats = [{ id: 'all', label: 'All categories', dot: '#9DB1C4' }];
  order.forEach(k => { if (counts[k]) cats.push({ id:k, label:cap(k), dot:CAT_PALETTE[k] }); });
  // future-proof: any extra labels in the data show up too
  Object.keys(counts).forEach(k => {
    if (k !== 'all' && !cats.some(c => c.id === k)) {
      cats.push({ id:k, label:cap(k), dot:'#9DB1C4' });
    }
  });
  const catMenu = document.getElementById('catMenu');
  catMenu.innerHTML = cats.map(c => `
    <button class="qpicker-option${c.id===selectedCategory?' selected':''}" type="button" role="option" data-cat="${esc(c.id)}">
      <span class="opt-tier-dot" style="background:${c.dot}"></span>
      <span>${esc(c.label)}</span>
      <span class="opt-meta">${counts[c.id]||0}</span>
    </button>`).join('');

  catMenu.addEventListener('click', e => {
    const btn = e.target.closest('[data-cat]');
    if (!btn) return;
    selectCategory(btn.dataset.cat);
    closeAllPickers();
  });

  renderQueryList('');
  const search = document.getElementById('qSearch');
  search.addEventListener('input', e => renderQueryList(e.target.value));
  document.getElementById('qList').addEventListener('click', e => {
    const btn = e.target.closest('[data-qid]');
    if (!btn) return;
    if (btn.dataset.qid === '__rand') selectQuery(null);
    else selectQuery(btn.dataset.qid);
    closeAllPickers();
  });

  // Open / close the menus
  document.querySelectorAll('.qpicker-field').forEach(field => {
    field.addEventListener('click', e => {
      const picker = e.currentTarget.closest('.qpicker');
      const wasOpen = picker.classList.contains('open');
      closeAllPickers();
      if (!wasOpen) {
        picker.classList.add('open');
        e.currentTarget.setAttribute('aria-expanded', 'true');
        const search = picker.querySelector('.qpicker-search');
        if (search) setTimeout(() => search.focus(), 80);
      }
    });
  });
  // Outside click + Esc to close
  document.addEventListener('click', e => {
    if (!e.target.closest('.qpicker')) closeAllPickers();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeAllPickers();
  });
}
document.getElementById('expandQ').addEventListener('click', e => {
  const expanded = el('recentQ').classList.toggle('expanded');
  e.currentTarget.textContent = expanded ? '⌃ Show less' : '⌄ Show full query';
});
document.getElementById('nextBtn').addEventListener('click', advanceDetail);
document.getElementById('pinBtn').addEventListener('click', e => {
  pinned = !pinned;
  const btn = e.currentTarget;
  btn.classList.toggle('active', pinned);
  btn.setAttribute('aria-pressed', String(pinned));
  btn.textContent = pinned ? '📌 Pinned' : '📌 Pin';
  if (!pinned) nextDueAt = performance.now() + dwellMs();
});

// Pause the spawn loop while the stage is scrolled out of view (saves
// CPU and keeps the cumulative tally honest to what the viewer saw).
if ('IntersectionObserver' in window) {
  const stage = document.querySelector('.routing-section');
  if (stage) new IntersectionObserver(entries => {
    onScreen = entries[0].isIntersecting;
  }, { threshold: 0.05 }).observe(stage);
}

// ===================================================================
// TIER DIAGRAM
// ===================================================================
function buildTierDiagram() {
  refreshBoxLabels();
  const row = document.getElementById('tierRow');
  const counts = DATA.meta.tier_route_counts || {};
  const total = Object.values(counts).reduce((a,b)=>a+b,0) || 1;
  row.innerHTML = '';
  const cards = [...DATA.meta.tiers, {level:5, models:DATA.meta.frontier_models, frontier:true}];
  cards.forEach(t => {
    const model = t.frontier ? pickedFrontier : pickedModel[t.level];
    const price = DATA.pricing[model] || {};
    const share = Math.round((counts[t.level]||0)/total*100);
    const card = document.createElement('div');
    card.className = 'tier-card';
    card.style.borderTop = `4px solid ${TIER_HEX[t.level]}`;
    const priceStr = price.self_hosted
      ? 'self-hosted · $0/token'
      : `$${(price.in_per_1m??0).toFixed(2)} in · $${(price.out_per_1m??0).toFixed(2)} out / M`;
    card.innerHTML = `
      <div class="tlabel" style="color:${TIER_HEX[t.level]}">${t.frontier?'Frontier':'Tier '+t.level}</div>
      <div class="tmodel">${esc(model||'-')}</div>
      <div class="tprov">${esc((price.provider)||'')}</div>
      <div class="tcost">${priceStr}</div>
      <div class="tshare-bar"><span style="width:${share}%;background:${TIER_HEX[t.level]}"></span></div>
      <div class="tshare">${share}% of queries routed here</div>`;
    row.appendChild(card);
  });
}

// ===================================================================
// HERO CALCULATOR (driven by observed mix + picked models + pricing)
// ===================================================================
let slider, qvalue;
function initCalculator() {
  slider = document.getElementById('qslider');
  qvalue = document.getElementById('qvalue');
  slider.addEventListener('input', updateCalc);
  updateCalc();
}
function updateCalc() {
  if (!slider) return;
  const q = parseInt(slider.value, 10);
  qvalue.innerHTML = fmtInt(q) + ' <span>queries/day</span>';
  const perYear = q * 365;
  // Same per-query cost math the live replay uses, averaged over the set.
  const avg = populationAverages();
  const routed = avg.routed * perYear;
  const frontier = avg.frontier * perYear;
  const saved = frontier - routed;
  const pct = frontier > 0 ? Math.round(saved/frontier*100) : 0;
  document.getElementById('savings').textContent = fmtMoney(saved);
  document.getElementById('pct').innerHTML =
    `${pct}% lower than ${esc(pickedFrontier)}-only `
    + `<span class="delta">(${fmtMoney(saved)}/yr)</span>`;
  document.getElementById('vsline').textContent =
    `vs. ${fmtMoney(frontier)}/yr if every query went to ${pickedFrontier} `
    + `· routed avg ${fmtCost(avg.routed)}/query`;
  const bs = document.getElementById('baselineSub');
  if (bs) bs.textContent = 'if every query → ' + pickedFrontier;
}
