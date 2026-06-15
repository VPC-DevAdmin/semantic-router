'use strict';
// Interactive semantic-routing demo. Config (tiers/models/exemplars/keys) lives
// in localStorage; defaults come from config.default.json. The server scores the
// query against each tier's exemplars and (with a key) calls the chosen model.

const TIER_COLORS = ['#1FB67A', '#11A8C4', '#F2A93B', '#E8743B', '#E64B5C', '#8B5CF6', '#0EA5E9'];
const tierColor = i => TIER_COLORS[i % TIER_COLORS.length];
const LS_KEY = 'sr_interactive_cfg';

let CONFIG = null;
let routeMode = 'auto';
const $ = id => document.getElementById(id);
const esc = s => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// ── Config ───────────────────────────────────────────────────────────────────
async function loadConfig() {
  const saved = localStorage.getItem(LS_KEY);
  if (saved) { try { CONFIG = JSON.parse(saved); return; } catch (e) { /* fall through */ } }
  CONFIG = await (await fetch('config.default.json')).json();
}
const saveConfig = () => localStorage.setItem(LS_KEY, JSON.stringify(CONFIG));
const tierById = id => CONFIG.tiers.find(t => t.id === id);
const tierIndex = id => CONFIG.tiers.findIndex(t => t.id === id);

// ── Composer: route pills + key banner ─────────────────────────────────────────
function renderPills() {
  const wrap = $('routePills');
  wrap.querySelectorAll('.pill').forEach(p => p.remove());
  const mk = (mode, label, missingKey) => {
    const b = document.createElement('button');
    b.className = 'pill' + (routeMode === mode ? ' active' : '');
    b.dataset.mode = mode;
    b.innerHTML = esc(label) + (missingKey ? '<span class="nokey" title="no API key">●</span>' : '');
    b.onclick = () => { routeMode = mode; renderPills(); };
    wrap.appendChild(b);
  };
  mk('auto', 'auto', false);
  CONFIG.tiers.forEach(t => mk(t.id, t.name, !(t.api_key || '').trim()));
}

function renderKeyBanner() {
  const banner = $('keyBanner');
  const withKey = CONFIG.tiers.filter(t => (t.api_key || '').trim()).length;
  if (withKey === CONFIG.tiers.length && CONFIG.tiers.length) { banner.hidden = true; return; }
  banner.hidden = false;
  const missing = CONFIG.tiers.length - withKey;
  banner.innerHTML = `⚠ ${withKey === 0 ? 'No API keys set' : missing + ' tier(s) have no API key'} — `
    + `routing still works, but answers need a key. <a id="bannerSettings">Open Settings →</a>`;
  $('bannerSettings').onclick = openSettings;
}

// ── Chat ───────────────────────────────────────────────────────────────────────
function appendUser(text) {
  $('emptyState')?.remove();
  const el = document.createElement('div');
  el.className = 'msg-user';
  el.textContent = text;
  $('chat').appendChild(el);
  el.scrollIntoView({ block: 'end' });
}

function appendAssistant() {
  const el = document.createElement('div');
  el.className = 'msg-assistant';
  el.innerHTML = `<div class="answer"><span class="model-line">routing…</span></div>`;
  $('chat').appendChild(el);
  el.scrollIntoView({ block: 'end' });
  return el;
}

function routingHTML(routing, servedId, forced) {
  const maxScore = Math.max(0.0001, ...routing.tiers.map(t => t.score));
  const rows = routing.tiers.map(t => {
    const i = tierIndex(t.id);
    const w = Math.round(t.score / maxScore * 100);
    const chosen = t.id === servedId;
    return `<div class="tier-score${chosen ? ' chosen' : ''}">
      <span class="tname"><span class="tdot" style="background:${tierColor(i)}"></span>${esc(t.name)}</span>
      <span class="bar"><span style="width:${w}%;background:${tierColor(i)}"></span></span>
      <span class="pct">${t.score.toFixed(2)}</span></div>`;
  }).join('');
  const forcedNote = forced && routing.chosen_id && routing.chosen_id !== servedId
    ? `<div class="forced-note">You forced ${esc(tierById(servedId)?.name || servedId)}; auto would have routed to ${esc(tierById(routing.chosen_id)?.name || '')}.</div>`
    : '';
  return `<div class="routing">
    <div class="routing-head"><span>Routing decision</span><span class="scorer">${esc(routing.scorer)}</span></div>
    ${rows}
    <div class="reasoning">${esc(routing.reasoning)}</div>
    ${forcedNote}
  </div>`;
}

function answerHTML(res) {
  if (res.no_api_key) {
    const t = res.tier || {};
    return `<div class="answer notice">No API key set for <strong>${esc(t.name || 'this tier')}</strong>
      (${esc(t.model || '')}). Add it in Settings to get an answer — the routing above ran without one.</div>`;
  }
  if (res.error) return `<div class="answer err">Provider error: ${esc(res.error)}</div>`;
  const model = res.tier ? `${esc(res.tier.name)} · ${esc(res.tier.model)}` : '';
  return `<div class="answer"><span class="model-line">answered by ${model}</span>${esc(res.answer)}</div>`;
}

function deeperHTML(res) {
  if (!CONFIG.max_tier_id || !res.tier || res.tier.id === CONFIG.max_tier_id) return '';
  if (!(res.answer || res.no_api_key)) return '';
  const maxName = tierById(CONFIG.max_tier_id)?.name || 'max tier';
  return `<div class="deeper"><button class="deeper-btn">↑ Get a deeper answer (${esc(maxName)})</button></div>`;
}

function renderInto(card, query, res) {
  const servedId = res.tier ? res.tier.id : res.routing.chosen_id;
  card.innerHTML = routingHTML(res.routing, servedId, !!res.routing.forced) + answerHTML(res) + deeperHTML(res);
  const btn = card.querySelector('.deeper-btn');
  if (btn) btn.onclick = () => { btn.disabled = true; ask(query, CONFIG.max_tier_id); };
  card.scrollIntoView({ block: 'end' });
}

async function ask(query, mode) {
  const card = appendAssistant();
  try {
    const res = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, mode, tiers: CONFIG.tiers, max_tier_id: CONFIG.max_tier_id }),
    }).then(r => r.json());
    renderInto(card, query, res);
  } catch (e) {
    card.innerHTML = `<div class="answer err">Request failed: ${esc(e.message)}. Is the server running?</div>`;
  }
}

function send() {
  const q = $('input').value.trim();
  if (!q) return;
  $('input').value = ''; autoGrow();
  appendUser(q);
  ask(q, routeMode);
}

// ── Settings panel ──────────────────────────────────────────────────────────────
function makeExRow(text) {
  const row = document.createElement('div');
  row.className = 'te-ex-row';
  row.innerHTML = `<input class="te-ex" value="${esc(text)}"><button class="icon-btn te-ex-del" title="remove">✕</button>`;
  row.querySelector('.te-ex-del').onclick = () => row.remove();
  return row;
}

function renderTierEditors() {
  const host = $('tierEditors');
  host.innerHTML = '';
  CONFIG.tiers.forEach((t, i) => {
    const node = $('tierEditorTpl').content.cloneNode(true);
    const root = node.querySelector('.tier-editor');
    root.dataset.id = t.id;
    root.dataset.key = t.api_key || '';
    node.querySelector('.tier-dot').style.background = tierColor(i);
    node.querySelector('.te-name').value = t.name || '';
    node.querySelector('.te-provider').value =
      ['OpenAI', 'Anthropic', 'Google', 'OpenAI-compatible'].includes(t.provider) ? t.provider : 'OpenAI-compatible';
    node.querySelector('.te-model').value = t.model || '';
    node.querySelector('.te-base').value = t.base_url || '';
    node.querySelector('.te-threshold').value = t.threshold ?? 0.3;
    const ks = node.querySelector('.te-keystate');
    const hasKey = !!(t.api_key || '').trim();
    ks.textContent = hasKey ? 'key set' : 'no key';
    ks.className = 'te-keystate ' + (hasKey ? 'has' : 'no');
    node.querySelector('.te-key').oninput = e => {
      const has = !!e.target.value.trim() || hasKey;
      ks.textContent = has ? 'key set' : 'no key';
      ks.className = 'te-keystate ' + (has ? 'has' : 'no');
    };
    const exList = node.querySelector('.te-ex-list');
    (t.exemplars || []).forEach(ex => exList.appendChild(makeExRow(ex)));
    node.querySelector('.te-add-ex').onclick = () => exList.appendChild(makeExRow(''));
    node.querySelector('.te-del').onclick = () => root.remove();
    host.appendChild(node);
  });
}

function collectConfig() {
  const tiers = [];
  $('tierEditors').querySelectorAll('.tier-editor').forEach((root, i) => {
    const keyInput = root.querySelector('.te-key').value.trim();
    tiers.push({
      id: root.dataset.id || `tier${i + 1}`,
      name: root.querySelector('.te-name').value.trim() || `Tier ${i + 1}`,
      provider: root.querySelector('.te-provider').value,
      model: root.querySelector('.te-model').value.trim(),
      base_url: root.querySelector('.te-base').value.trim(),
      api_key: keyInput || root.dataset.key || '',   // preserve unless overwritten
      threshold: parseFloat(root.querySelector('.te-threshold').value) || 0,
      exemplars: [...root.querySelectorAll('.te-ex')].map(i => i.value.trim()).filter(Boolean),
    });
  });
  CONFIG.tiers = tiers;
  if (!tierById(CONFIG.max_tier_id)) CONFIG.max_tier_id = tiers.length ? tiers[tiers.length - 1].id : null;
}

function addTier() {
  collectConfig();
  const n = CONFIG.tiers.length + 1;
  CONFIG.tiers.push({ id: `tier${Date.now().toString(36)}`, name: `Tier ${n}`,
    provider: 'OpenAI-compatible', model: '', base_url: '', api_key: '',
    threshold: 0.3, exemplars: [] });
  renderTierEditors();
}

const openSettings = () => { renderTierEditors(); $('settingsModal').hidden = false; $('saveStatus').textContent = ''; };
const closeSettings = () => { $('settingsModal').hidden = true; };

function saveCfg() {
  collectConfig();
  saveConfig();
  renderPills(); renderKeyBanner();
  $('saveStatus').textContent = '✓ Saved';
}

async function resetCfg() {
  CONFIG = await (await fetch('config.default.json?v=' + Date.now())).json();
  saveConfig();
  renderTierEditors(); renderPills(); renderKeyBanner();
  $('saveStatus').textContent = '✓ Reset to defaults';
}

// ── Wire up ───────────────────────────────────────────────────────────────────
function autoGrow() {
  const el = $('input');
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

async function init() {
  await loadConfig();
  renderPills();
  renderKeyBanner();
  $('sendBtn').onclick = send;
  $('input').addEventListener('input', autoGrow);
  $('input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  $('settingsBtn').onclick = openSettings;
  $('closeSettings').onclick = closeSettings;
  $('addTier').onclick = addTier;
  $('saveCfg').onclick = saveCfg;
  $('resetCfg').onclick = resetCfg;
  $('settingsModal').addEventListener('click', e => { if (e.target.id === 'settingsModal') closeSettings(); });
}

init();
