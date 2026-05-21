'use strict';
// ===================================================================
// Semantic-routing cost demo — data-driven replay.
// Reads demo/data/demo_data.json (built by tools/build_demo_data.py).
// Nothing about tiers, models, pricing, or throughput is hardcoded.
// ===================================================================

const svgNS = 'http://www.w3.org/2000/svg';
const TIER_COLORS = {
  1: 'var(--t1)', 2: 'var(--t2)', 3: 'var(--t3)', 4: 'var(--t4)', 5: 'var(--t5)',
};
const TIER_HEX = { 1:'#1FB67A', 2:'#11A8C4', 3:'#F2A93B', 4:'#E8743B', 5:'#E64B5C' };

let DATA = null;           // full demo_data.json
let TIERS = [];            // [1,2,3,4,5] derived
let pickedModel = {};      // tier -> model name (1-4)
let pickedFrontier = null; // frontier model name
let modelAvgCost = {};     // model -> {sum, n} mean cost/query

// ---------- formatting ----------
const fmtInt = n => n.toLocaleString('en-US');
function fmtMoney(n) {
  const a = Math.abs(n);
  if (a >= 1e9) return '$' + (n/1e9).toFixed(2) + 'B';
  if (a >= 1e6) return '$' + (n/1e6).toFixed(2) + 'M';
  if (a >= 1e3) return '$' + (n/1e3).toFixed(0) + 'K';
  return '$' + n.toFixed(2);
}
const fmtCost = n => '$' + n.toFixed(n < 0.01 ? 5 : 4);
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const truncate = (s,n) => s.length>n ? s.slice(0,n-1)+'…' : s;

// ===================================================================
// BOOT
// ===================================================================
fetch('data/demo_data.json')
  .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
  .then(init)
  .catch(err => {
    document.getElementById('loadStatus').innerHTML =
      `<div class="errbox">Couldn't load <code>data/demo_data.json</code> — ${esc(err.message)}.<br>` +
      `Run <code>make demo-data</code> to build it, then serve via <code>make demo</code>.</div>`;
  });

function init(data) {
  DATA = data;
  TIERS = data.meta.tiers.map(t => t.level);          // routed tiers (1-4)
  const frontierLevel = 5;
  if (!TIERS.includes(frontierLevel)) TIERS.push(frontierLevel);  // frontier shown as tier 5

  // default picks: first model per tier; first frontier model
  data.meta.tiers.forEach(t => { pickedModel[t.level] = t.models[0]?.model; });
  pickedFrontier = data.meta.frontier_models[0]?.model;

  precomputeModelCosts();
  buildPickers();
  buildStage();
  buildTierDiagram();
  initCalculator();

  document.getElementById('loadStatus').style.display = 'none';
  document.getElementById('demoRoot').style.display = 'block';

  // throughput stat from the data (measured)
  startEngine();
}

// average cost/query per model, for the calculator + tier cards
function precomputeModelCosts() {
  DATA.queries.forEach(q => {
    Object.values(q.routed_answers).forEach(list => list.forEach(a => {
      const m = modelAvgCost[a.model] || (modelAvgCost[a.model] = {sum:0, n:0});
      m.sum += a.cost_usd; m.n++;
    }));
    q.frontier_answers.forEach(a => {
      const m = modelAvgCost[a.model] || (modelAvgCost[a.model] = {sum:0, n:0});
      m.sum += a.cost_usd; m.n++;
    });
  });
}
const avgCost = model => {
  const m = modelAvgCost[model];
  return m && m.n ? m.sum / m.n : 0;
};

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
    }, false));
  });
  bar.appendChild(makePicker('Frontier', TIER_HEX[5], DATA.meta.frontier_models, m => {
    pickedFrontier = m;
    onPickerChange();
  }, true));
}
function makePicker(label, hex, models, onChange, isFrontier) {
  const div = document.createElement('div');
  div.className = 'picker' + (isFrontier ? ' frontier' : '');
  const opts = models.map(m => `<option value="${esc(m.model)}">${esc(m.model)} · ${esc(m.provider||'')}</option>`).join('');
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
    for (let i = 0; i < 5; i++) {
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

function pulseRouter() {
  const ring = document.createElementNS(svgNS,'circle');
  ring.setAttribute('cx', 600); ring.setAttribute('cy', ROUTER.y); ring.setAttribute('r', 50);
  ring.setAttribute('fill','none'); ring.setAttribute('stroke','var(--dell-accent)'); ring.setAttribute('stroke-width','2');
  pulseLayer().appendChild(ring);
  const t0 = performance.now();
  (function step(now){
    const t = Math.min((now-t0)/700,1);
    ring.setAttribute('r', 50 + t*40); ring.style.opacity = 0.6*(1-t);
    if (t<1) requestAnimationFrame(step); else ring.remove();
  })(t0);
}
function latencyBadge(ms) {
  const el = document.createElement('div');
  el.className = 'latency-badge';
  el.style.left = (600/1200*100)+'%';
  el.style.top  = (ROUTER.y/560*100)+'%';
  el.textContent = 'classified in ' + ms + 'ms';
  overlayLayer().appendChild(el);
  setTimeout(()=>el.remove(), 1300);
}
function savedTag(level, savings) {
  if (savings <= 0) return;
  const cy = tierBoxY(level, TIERS.length) + BOX.h/2;
  const el = document.createElement('div');
  el.className = 'float-tag';
  el.style.left = (820/1200*100)+'%';
  el.style.top  = (cy/560*100)+'%';
  el.textContent = '+' + fmtCost(savings) + ' saved';
  overlayLayer().appendChild(el);
  setTimeout(()=>el.remove(), 1500);
}

// ===================================================================
// COST + VERDICT for a query, given current picks
// ===================================================================
function resolveQuery(q) {
  const tier = q.routed_tier;
  const frontierAns = q.frontier_answers.find(a => a.model === pickedFrontier) || q.frontier_answers[0];
  const frontierCost = frontierAns ? frontierAns.cost_usd : 0;

  let routedModel, routedAns, routedCost;
  if (tier === 5 || !q.routed_answers['tier'+tier]) {
    // Routed to the frontier tier — the "routed answer" IS a frontier model.
    routedModel = pickedFrontier;
    routedAns = frontierAns;
    routedCost = frontierCost;
  } else {
    const list = q.routed_answers['tier'+tier];
    routedAns = list.find(a => a.model === pickedModel[tier]) || list[0];
    routedModel = routedAns ? routedAns.model : pickedModel[tier];
    routedCost = routedAns ? routedAns.cost_usd : 0;
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

const el = id => document.getElementById(id);

function spawnQuery(q) {
  const r = resolveQuery(q);
  pulseRouter();
  if (q.routing_latency_ms != null) latencyBadge(q.routing_latency_ms);
  el('serverStatus').textContent = (r.tier===5?'→ FRONTIER':'→ TIER '+r.tier);
  el('serverStatus').style.fill = TIER_HEX[r.tier];
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
  const r = resolveQuery(q);
  el('recentQ').textContent = truncate(q.prompt, 160);
  const pill = el('recentPill');
  pill.style.background = TIER_HEX[r.tier] + '22';
  pill.style.color = TIER_HEX[r.tier];
  pill.textContent = (r.tier===5 ? 'Frontier · needed top-tier' : 'Tier ' + r.tier);

  // routed column
  el('ansModelR').textContent = r.routedModel + (r.tier===5 ? ' (routed to the right tier)' : '');
  el('ansCostR').textContent = 'cost: ' + fmtCost(r.routedCost);
  el('ansBodyR').textContent = (r.routedAns && r.routedAns.answer) || '—';
  // frontier column
  el('ansModelB').textContent = (r.frontierAns ? r.frontierAns.model : pickedFrontier) + ' · frontier';
  el('ansCostB').textContent = 'cost: ' + fmtCost(r.frontierCost);
  el('ansBodyB').textContent = (r.frontierAns && r.frontierAns.answer) || '—';

  // verdicts
  const rows = el('verdictRows');
  if (r.tier === 5) {
    rows.innerHTML = `<div class="verdict-row"><div class="verdict-detail" style="grid-column:1/-1;color:var(--text-muted);">
      This query was routed to the frontier tier — the router judged it needed top-tier quality and delivered it. Routed answer and baseline are the same model, so there's no routed-vs-frontier comparison to score.</div></div>`;
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
      <div><span class="verdict-badge ${cls}">${esc(v.verdict)}</span></div>
      <div class="verdict-detail">
        <div class="verdict-rationale">${esc(v.rationale)}</div>
        <div class="verdict-scores">
          <span class="score-chip">correctness ${sc.correctness}</span>
          <span class="score-chip">completeness ${sc.completeness}</span>
          <span class="score-chip">fitness ${sc.fitness_for_purpose}</span>
          <span class="score-chip">soundness ${sc.soundness}</span>
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
  const rA=el('routedArea'), fL=el('frontierLine'), sA=el('savingsArea');
  if (chartHistory.length < 2) { rA.setAttribute('d',''); fL.setAttribute('d',''); sA.setAttribute('d',''); return; }
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
}

// ===================================================================
// MAIN LOOP + SPAWN
// ===================================================================
let gameTime = 0, lastFrame = performance.now();
let speed = 1, paused = false;
function frame(now) {
  const dt = (now - lastFrame)/1000; lastFrame = now;
  if (!paused) gameTime += dt * speed;
  inFlight = inFlight.filter(b => b.update());
  recentTimes = recentTimes.filter(t => now - t < 3000);
  // Visible throughput (sampled) — but headline stat is the MEASURED rate.
  el('liveQps').textContent = DATA.meta.throughput_qps.toFixed(1);
  el('qpsInSvg').textContent = DATA.meta.throughput_qps.toFixed(1);
  renderChart();
  requestAnimationFrame(frame);
}

// Spawn cadence: balls fly at the real measured throughput so the stage
// conveys true speed; the readable detail panel samples ~1 every 1.5s.
let spawnTimer = null, sampleAccumulator = 0;
const SAMPLE_EVERY_MS = 1500;
function spawnTick() {
  // Don't spawn while hidden: rAF (which advances + removes balls) is
  // throttled in background tabs, so setTimeout-driven spawns would pile
  // up. Also cap concurrent balls as a belt-and-suspenders guard.
  if (!paused && DATA && !document.hidden && inFlight.length < 80) {
    const q = DATA.queries[Math.floor(Math.random()*DATA.queries.length)];
    spawnQuery(q);
    sampleAccumulator += spawnInterval();
    if (sampleAccumulator >= SAMPLE_EVERY_MS) { sampleAccumulator = 0; showDetail(q); }
  }
  spawnTimer = setTimeout(spawnTick, spawnInterval());
}
function spawnInterval() {
  const qps = DATA.meta.throughput_qps * speed;
  return Math.max(12, 1000 / qps);   // cap so very high speeds don't lock the loop
}

function startEngine() {
  requestAnimationFrame(frame);
  spawnTick();
  // seed the detail panel quickly
  showDetail(DATA.queries[Math.floor(Math.random()*DATA.queries.length)]);
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
  chartHistory.length = 0; recentTimes = [];
  ballLayer().innerHTML=''; pulseLayer().innerHTML=''; overlayLayer().innerHTML='';
  updateLiveStats();
});
document.querySelectorAll('#speedPill button').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('#speedPill button').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); speed = parseInt(b.dataset.s,10);
  });
});

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
      <div class="tlabel">${t.frontier?'Frontier':'Tier '+t.level}</div>
      <div class="tmodel">${esc(model||'—')}</div>
      <div class="tprov">${esc((price.provider)||'')}</div>
      <div class="tcost">${priceStr}</div>
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
function routedCostPerQuery() {
  const counts = DATA.meta.tier_route_counts || {};
  const total = Object.values(counts).reduce((a,b)=>a+b,0) || 1;
  let cpq = 0;
  DATA.meta.tiers.forEach(t => {
    const share = (counts[t.level]||0)/total;
    cpq += share * avgCost(pickedModel[t.level]);
  });
  // tier-5-routed share uses the frontier model cost
  const f5 = (counts[5]||0)/total;
  cpq += f5 * avgCost(pickedFrontier);
  return cpq;
}
function updateCalc() {
  if (!slider) return;
  const q = parseInt(slider.value, 10);
  qvalue.innerHTML = fmtInt(q) + ' <span>queries/day</span>';
  const perYear = q * 365;
  const routed = routedCostPerQuery() * perYear;
  const frontier = avgCost(pickedFrontier) * perYear;
  const saved = frontier - routed;
  const pct = frontier > 0 ? Math.round(saved/frontier*100) : 0;
  document.getElementById('savings').textContent = fmtMoney(saved);
  document.getElementById('pct').textContent = pct + '% lower than frontier-only';
  document.getElementById('vsline').textContent =
    `vs. ${fmtMoney(frontier)} frontier-only (${esc(pickedFrontier)}) · routed avg ${fmtCost(routedCostPerQuery())}/query`;
  const bs = document.getElementById('baselineSub');
  if (bs) bs.textContent = 'if every query → ' + pickedFrontier;
}
