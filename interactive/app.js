'use strict';
// Live interactive demo: every query is routed by a running vllm-sr. Config is a
// separate overlay on the server (config/live_demo.local.json); the canonical
// benchmark config is never touched. Settings → Apply rebuilds + reloads vllm-sr.

const TIER_COLORS = ['#1FB67A', '#11A8C4', '#F2A93B', '#E8743B', '#E64B5C', '#8B5CF6', '#0EA5E9'];
const tierColor = i => TIER_COLORS[i % TIER_COLORS.length];
const $ = id => document.getElementById(id);
const esc = s => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

let CONFIG = null;       // overlay from the server (keys masked; key_set flags)
let QUERIES = {};        // {category: [prompt,...]}
let routeMode = 'auto';

// chat sessions (persisted in localStorage)
let chats = [];          // [{id, title, msgs:[{role, text, query?, routing?, error?, pending?}], ts}]
let currentId = null;

const tierIndexById = id => CONFIG.tiers.findIndex(t => t.id === id);
const maxTierId = () => CONFIG.tiers.length ? CONFIG.tiers[CONFIG.tiers.length - 1].id : null;

// ── Boot ──────────────────────────────────────────────────────────────────────
async function boot() {
  CONFIG = await (await fetch('/api/config')).json();
  QUERIES = (await (await fetch('/api/queries')).json()).categories || {};
  $('routerTag').textContent = CONFIG.vllm_sr_url || 'vllm-sr';
  $('routeSummary').textContent = 'Routes every prompt to the right-sized model';
  loadChats();
  if (!chats.length) newChat(); else { currentId = chats[0].id; }
  renderPills(); renderKeyBanner(); renderExamples(); renderChatList(); renderMessages();
}

// ── Chat sessions ──────────────────────────────────────────────────────────────
function loadChats() {
  try { chats = JSON.parse(localStorage.getItem('sr_chats') || '[]'); } catch { chats = []; }
  if (!Array.isArray(chats)) chats = [];
}
function saveChats() {
  try { localStorage.setItem('sr_chats', JSON.stringify(chats.slice(0, 60))); } catch { /* quota */ }
}
const curChat = () => chats.find(c => c.id === currentId);

function newChat() {
  const c = { id: 'c' + Date.now().toString(36), title: 'New chat', msgs: [], ts: Date.now() };
  chats.unshift(c); currentId = c.id;
  saveChats(); renderChatList(); renderMessages();
  $('input')?.focus();
}
function selectChat(id) { currentId = id; renderChatList(); renderMessages(); closeSidebar(); }
function deleteChat(id) {
  chats = chats.filter(c => c.id !== id);
  if (currentId === id) currentId = chats[0]?.id || null;
  if (!chats.length) { newChat(); return; }
  saveChats(); renderChatList(); renderMessages();
}

function renderChatList() {
  const host = $('chatList'); host.innerHTML = '';
  if (!chats.length) { host.innerHTML = '<div class="chat-empty">No conversations yet</div>'; return; }
  chats.forEach(c => {
    const el = document.createElement('div');
    el.className = 'chat-item' + (c.id === currentId ? ' active' : '');
    el.innerHTML = `<span class="ci-dot"></span><span class="ci-title">${esc(c.title || 'New chat')}</span>
      <button class="ci-del" title="Delete">✕</button>`;
    el.onclick = () => selectChat(c.id);
    el.querySelector('.ci-del').onclick = e => { e.stopPropagation(); deleteChat(c.id); };
    host.appendChild(el);
  });
}

// ── Message rendering ───────────────────────────────────────────────────────────
function renderMessages() {
  const host = $('messages');
  const c = curChat();
  $('chatTitle').textContent = c ? (c.title || 'New chat') : 'New chat';
  host.innerHTML = '';
  if (!c || !c.msgs.length) { host.appendChild(welcomeNode()); return; }
  const thread = document.createElement('div'); thread.className = 'thread';
  c.msgs.forEach((m, i) => thread.appendChild(messageNode(m, c, i)));
  host.appendChild(thread);
  scrollBottom();
}
function scrollBottom() { const h = $('messages'); h.scrollTop = h.scrollHeight; }

function welcomeNode() {
  const node = $('welcomeTpl').content.cloneNode(true);
  const sg = node.querySelector('#suggestions');
  pickSuggestions().forEach(s => {
    const card = document.createElement('div');
    card.className = 'sugg-card';
    card.innerHTML = `<div class="sugg-cat">${esc(s.cat)}</div><div class="sugg-text">${esc(s.q)}</div>`;
    card.onclick = () => { $('input').value = s.q; autoGrow(); send(); };
    sg.appendChild(card);
  });
  return node;
}
function pickSuggestions() {
  const out = [];
  const cats = Object.keys(QUERIES);
  for (const cat of cats) {
    const list = QUERIES[cat] || [];
    if (list.length) out.push({ cat, q: list[Math.floor(list.length / 2)] });
    if (out.length >= 4) break;
  }
  return out;
}

function messageNode(m, chat, idx) {
  const el = document.createElement('div');
  el.className = 'msg ' + m.role;
  if (m.role === 'user') {
    el.innerHTML = `<div class="bubble">${esc(m.text)}</div>`;
    return el;
  }
  // assistant
  const card = document.createElement('div'); card.className = 'card';
  if (m.pending) {
    card.innerHTML = `<div class="thinking"><span class="spinner"></span> Routing via vLLM Semantic Router…</div>`;
  } else if (m.error) {
    card.innerHTML = (m.routing ? rationaleHTML(m.routing) : '') + `<div class="answer err">${esc(m.error)}</div>`;
  } else {
    card.innerHTML = (m.routing ? rationaleHTML(m.routing) : '') + answerHTML(m) + deeperHTML(m);
    const btn = card.querySelector('.deeper-btn');
    if (btn) btn.onclick = () => { btn.disabled = true; ask(m.query, maxTierId()); };
  }
  el.appendChild(card);
  return el;
}

function answerHTML(m) {
  const r = m.routing || {};
  const tier = CONFIG.tiers.find(t => t.id === r.selected_tier_id);
  if (tier && !tier.key_set) {
    return `<div class="answer err">Routed to <strong>${esc(r.selected_tier_name)}</strong>
      (${esc(r.served_model)}), but this tier has no API key — add one in Settings. The routing above is real.</div>`;
  }
  const label = `answered by ${esc(r.selected_tier_name || '')} · ${esc(r.served_model || '')}`;
  return `<div class="answer"><span class="model-line">${label}</span>${mdToHtml(m.text)}</div>`;
}
function deeperHTML(m) {
  const r = m.routing || {};
  const mx = maxTierId();
  if (m.error || !mx || r.selected_tier_id === mx || !m.query) return '';
  const name = CONFIG.tiers.find(t => t.id === mx)?.name || 'top tier';
  return `<div class="deeper"><button class="deeper-btn">↑ Get a deeper answer (${esc(name)})</button></div>`;
}

// routing rationale: selected tier + confidence + difficulty meter + signal scores
function rationaleHTML(r) {
  const selIdx = tierIndexById(r.selected_tier_id);
  const color = selIdx >= 0 ? tierColor(selIdx) : 'var(--accent)';
  const conf = r.confidence != null ? Math.round(r.confidence * 100) : null;
  const head = `<div class="rat-head">
    <span class="rat-badge" style="--c:${color}">routed → ${esc(r.selected_tier_name || '?')}</span>
    <span class="rat-model">${esc(r.served_model || '')}</span>
    ${conf != null ? `<span class="rat-conf">${conf}% confidence</span>` : ''}
    ${r.forced ? '<span class="rat-forced">forced</span>' : ''}
  </div>`;
  return `<div class="rationale">${head}${difficultyMeter(r)}${signalChips(r)}</div>`;
}

function difficultyMeter(r) {
  const tiers = CONFIG.tiers, cuts = (CONFIG.tier_cutoffs || []).slice();
  if (tiers.length < 2 || cuts.length !== tiers.length - 1) return '';
  const lo = 0, hi = Math.max((cuts[cuts.length - 1] || 0.4) * 1.4, 0.12);
  const edges = [lo, ...cuts, hi];
  const selIdx = tierIndexById(r.selected_tier_id);
  let segs = '';
  for (let i = 0; i < tiers.length; i++) {
    const w = ((edges[i + 1] - edges[i]) / (hi - lo)) * 100;
    segs += `<div class="seg-band${i === selIdx ? ' on' : ''}" style="width:${w}%;--c:${tierColor(i)}">
      <span class="seg-name">${esc(tiers[i].name)}</span></div>`;
  }
  const d = r.request_difficulty;
  let marker = '';
  if (d != null) {
    const pct = Math.max(0, Math.min(100, ((d - lo) / (hi - lo)) * 100));
    marker = `<div class="meter-marker" style="left:${pct}%">
      <span class="marker-val">difficulty ${Number(d).toFixed(3)}</span></div>`;
  }
  return `<div class="meter"><div class="meter-bands">${segs}</div>${marker}</div>`;
}

const SIGNAL_LABELS = {
  trivial_lookup: 'Trivial lookup', moderate_complexity: 'Moderate reasoning',
  frontier_synthesis: 'Frontier synthesis', argumentative_construction: 'Argument construction',
  short_prompt: 'Short prompt', long_prompt: 'Long prompt', medium_prompt: 'Medium prompt',
  'query_difficulty:hard': 'High difficulty', 'query_difficulty:medium': 'Medium difficulty',
  'query_difficulty:easy': 'Low difficulty',
};
function niceSignal(key) {
  const parts = String(key).split(':');
  if (parts[0] === 'projection') return null;     // shown in the meter
  const rest = parts.slice(1).join(':');
  return SIGNAL_LABELS[rest] || rest.replace(/_/g, ' ');
}
function signalChips(r) {
  let rows = [];
  const sc = r.signal_confidences || {};
  for (const [k, v] of Object.entries(sc)) {
    const label = niceSignal(k);
    if (label) rows.push({ label, score: typeof v === 'number' ? v : null });
  }
  if (!rows.length && Array.isArray(r.matched)) {
    rows = r.matched.filter(m => m.type !== 'projection')
      .map(m => ({ label: SIGNAL_LABELS[m.name] || m.name.replace(/_/g, ' '), score: null }));
  }
  if (!rows.length) return '';
  rows.sort((a, b) => (b.score || 0) - (a.score || 0));
  const chips = rows.slice(0, 6).map(s =>
    `<span class="sig-chip">${esc(s.label)}${s.score != null ? `<b>${s.score.toFixed(2)}</b>` : ''}</span>`).join('');
  return `<div class="rat-signals"><span class="rat-signals-label">Why</span>${chips}</div>`;
}

// minimal markdown: fenced code blocks, inline code, bold, paragraphs
function mdToHtml(src) {
  const parts = String(src == null ? '' : src).split('```');
  let html = '';
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      const body = part.replace(/^[a-zA-Z0-9_+-]*\n/, '').replace(/\n$/, '');
      html += `<pre class="code"><code>${esc(body)}</code></pre>`;
    } else {
      html += esc(part).split(/\n{2,}/).map(b => b.trim()).filter(Boolean).map(b =>
        '<p>' + b.replace(/`([^`]+)`/g, '<code class="ic">$1</code>')
          .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
          .replace(/\n/g, '<br>') + '</p>').join('');
    }
  });
  return html || '<p></p>';
}

// ── Ask flow ───────────────────────────────────────────────────────────────────
async function ask(query, mode) {
  let c = curChat();
  if (!c) { newChat(); c = curChat(); }
  c.msgs.push({ role: 'user', text: query });
  if (!c.title || c.title === 'New chat') c.title = query.slice(0, 52);
  const am = { role: 'assistant', query, pending: true };
  c.msgs.push(am);
  saveChats(); renderChatList(); renderMessages();
  try {
    const res = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, mode }),
    }).then(r => r.json());
    am.pending = false; am.routing = res.routing || null; am.text = res.answer || ''; am.error = res.error || null;
  } catch (e) { am.pending = false; am.error = 'Request failed: ' + e.message; }
  saveChats(); renderMessages();
}
function send() {
  const q = $('input').value.trim();
  if (!q) return;
  $('input').value = ''; autoGrow();
  ask(q, routeMode);
}

// ── Composer controls ───────────────────────────────────────────────────────────
function renderPills() {
  const wrap = $('routePills');
  wrap.querySelectorAll('.pill').forEach(p => p.remove());
  const mk = (mode, label, noKey) => {
    const b = document.createElement('button');
    b.className = 'pill' + (routeMode === mode ? ' active' : '');
    b.innerHTML = esc(label) + (noKey ? '<span class="nokey" title="no API key">●</span>' : '');
    b.onclick = () => { routeMode = mode; renderPills(); };
    wrap.appendChild(b);
  };
  mk('auto', 'auto', false);
  CONFIG.tiers.forEach(t => mk(t.id, t.name, !t.key_set));
}

function renderKeyBanner() {
  const banner = $('keyBanner');
  const withKey = CONFIG.tiers.filter(t => t.key_set).length;
  if (CONFIG.tiers.length && withKey === CONFIG.tiers.length) { banner.hidden = true; return; }
  banner.hidden = false;
  banner.innerHTML = `⚠ ${withKey === 0 ? 'No API keys set' : (CONFIG.tiers.length - withKey) + ' tier(s) have no API key'} — `
    + `routing runs, but a tier with no key can't answer. <a id="bannerSettings">Open Settings →</a>`;
  $('bannerSettings').onclick = openSettings;
}

function renderExamples() {
  const sel = $('catSelect');
  const cats = Object.keys(QUERIES);
  sel.innerHTML = cats.map(c => `<option>${esc(c)}</option>`).join('');
  sel.onchange = renderChips;
  renderChips();
}
function renderChips() {
  const cat = $('catSelect').value;
  const chips = $('exChips');
  chips.innerHTML = '';
  (QUERIES[cat] || []).slice(0, 20).forEach(q => {
    const b = document.createElement('button');
    b.className = 'ex-chip'; b.title = q; b.textContent = q;
    b.onclick = () => { $('input').value = q; autoGrow(); $('input').focus(); toggleExamples(false); };
    chips.appendChild(b);
  });
}
function toggleExamples(force) {
  const pop = $('examplesPop'), btn = $('exToggle');
  const open = force != null ? force : pop.hidden;
  pop.hidden = !open; btn.classList.toggle('on', open);
}

// ── Sidebar (mobile) ────────────────────────────────────────────────────────────
function openSidebar() {
  $('sidebar').classList.add('open');
  let scrim = $('scrim');
  if (!scrim) {
    scrim = document.createElement('div'); scrim.id = 'scrim'; scrim.className = 'scrim';
    scrim.onclick = closeSidebar; document.body.appendChild(scrim);
  }
  scrim.hidden = false;
}
function closeSidebar() { $('sidebar').classList.remove('open'); const s = $('scrim'); if (s) s.hidden = true; }

// ── Settings ──────────────────────────────────────────────────────────────────
function renderTierEditors() {
  const host = $('tierEditors'); host.innerHTML = '';
  CONFIG.tiers.forEach((t, i) => {
    const node = $('tierEditorTpl').content.cloneNode(true);
    const root = node.querySelector('.tier-editor');
    root.dataset.id = t.id; root.dataset.keyset = t.key_set ? '1' : '';
    node.querySelector('.tier-dot').style.background = tierColor(i);
    node.querySelector('.te-name').value = t.name || '';
    node.querySelector('.te-provider').value =
      ['OpenAI', 'Anthropic', 'Google', 'OpenAI-compatible'].includes(t.provider) ? t.provider : 'OpenAI-compatible';
    node.querySelector('.te-model').value = t.model || '';
    node.querySelector('.te-base').value = t.base_url || '';
    const ks = node.querySelector('.te-keystate');
    const setKs = has => { ks.textContent = has ? 'key set' : 'no key'; ks.className = 'te-keystate ' + (has ? 'has' : 'no'); };
    setKs(t.key_set);
    node.querySelector('.te-key').oninput = e => setKs(!!e.target.value.trim() || t.key_set);
    node.querySelector('.te-del').onclick = () => root.remove();
    const dl = node.querySelector('.te-models');
    dl.id = 'te-models-' + t.id;
    node.querySelector('.te-model').setAttribute('list', dl.id);
    node.querySelector('.te-load').onclick = ev => loadTierModels(root, t.id, ev.currentTarget);
    host.appendChild(node);
  });
}

async function loadTierModels(root, tierId, btn) {
  const provider = root.querySelector('.te-provider').value;
  const base_url = root.querySelector('.te-base').value.trim();
  const api_key = root.querySelector('.te-key').value.trim();
  const dl = root.querySelector('.te-models');
  const orig = btn.textContent;
  btn.disabled = true; btn.textContent = '…';
  try {
    const res = await fetch('/api/models', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tier_id: tierId, provider, base_url, api_key }),
    });
    const j = await res.json();
    if (j.error) { btn.textContent = '!'; btn.title = j.error; return; }
    const models = j.models || [];
    dl.innerHTML = models.map(m => `<option value="${esc(m)}"></option>`).join('');
    btn.textContent = models.length ? '✓' : '∅';
    btn.title = models.length ? `${models.length} models — click the model field to pick` : 'provider returned no models';
  } catch (e) {
    btn.textContent = '!'; btn.title = String(e);
  } finally {
    btn.disabled = false;
    setTimeout(() => { btn.textContent = orig; btn.title = 'Load models from provider (uses this tier\'s key)'; }, 2000);
  }
}

function renderCutoffs() {
  const host = $('cutoffEditors'); host.innerHTML = '';
  (CONFIG.tier_cutoffs || []).forEach((c, i) => {
    const l = document.createElement('label');
    l.innerHTML = `${esc(CONFIG.tiers[i]?.name || 'T' + (i + 1))} → ${esc(CONFIG.tiers[i + 1]?.name || 'T' + (i + 2))}
      <input class="cutoff" type="number" step="0.01" value="${c}">`;
    host.appendChild(l);
  });
}

const SIGNAL_INFO = {
  trivial_lookup: { label: 'Looks trivial', hint: 'Short factual or lookup-style prompts.' },
  moderate_complexity: { label: 'Moderate reasoning', hint: 'Single- or multi-step reasoning and summarization.' },
  frontier_synthesis: { label: 'Frontier synthesis', hint: 'Hard, novel, cross-source synthesis.' },
  argumentative_construction: { label: 'Argument construction', hint: 'Building structured arguments or proofs.' },
};
const sigInfo = s => SIGNAL_INFO[s.id] || { label: s.id, hint: s.description || '' };
const sigDir = w => w < 0 ? { txt: '↓ toward cheaper tiers', cls: 'down' } : { txt: '↑ toward stronger tiers', cls: 'up' };

function renderSignals() {
  const host = $('signalEditors'); host.innerHTML = '';
  (CONFIG.signals || []).forEach(s => {
    const node = $('signalEditorTpl').content.cloneNode(true);
    const root = node.querySelector('.signal-editor');
    root.dataset.id = s.id;
    const info = sigInfo(s), dir = sigDir(s.weight ?? 0);
    const cands = s.candidates || [];
    node.querySelector('.sig-label').textContent = info.label;
    const d = node.querySelector('.sig-dir'); d.textContent = dir.txt; d.className = 'sig-dir ' + dir.cls;
    node.querySelector('.sig-count').textContent = `${cands.length} examples`;
    node.querySelector('.sig-blurb').textContent = info.hint;
    node.querySelector('.sig-cands-ta').value = cands.join('\n');
    node.querySelector('.sig-weight').value = s.weight ?? 0;
    node.querySelector('.sig-threshold').value = s.threshold ?? 0.5;
    const toggle = node.querySelector('.sig-toggle'), body = node.querySelector('.sig-body');
    const ta = node.querySelector('.sig-cands-ta');
    const count = node.querySelector('.sig-count');
    toggle.onclick = () => {
      const opening = body.hidden;
      body.hidden = !opening;
      toggle.querySelector('.sig-chevron').textContent = opening ? '▾' : '▸';
    };
    ta.addEventListener('input', () => {
      count.textContent = `${ta.value.split('\n').filter(l => l.trim()).length} examples`;
    });
    host.appendChild(node);
  });
}

function collectConfig() {
  CONFIG.vllm_sr_url = $('vllmUrl').value.trim() || CONFIG.vllm_sr_url;
  const tiers = [];
  $('tierEditors').querySelectorAll('.tier-editor').forEach((root, i) => {
    const keyInput = root.querySelector('.te-key').value.trim();
    tiers.push({
      id: root.dataset.id || `tier${i + 1}`,
      name: root.querySelector('.te-name').value.trim() || `Tier ${i + 1}`,
      provider: root.querySelector('.te-provider').value,
      model: root.querySelector('.te-model').value.trim(),
      base_url: root.querySelector('.te-base').value.trim(),
      api_key: keyInput,
      key_set: !!keyInput || root.dataset.keyset === '1',
    });
  });
  CONFIG.tiers = tiers;
  CONFIG.tier_cutoffs = [...$('cutoffEditors').querySelectorAll('.cutoff')].map(i => parseFloat(i.value) || 0);
  CONFIG.signals = [...$('signalEditors').querySelectorAll('.signal-editor')].map(root => ({
    id: root.dataset.id,
    weight: parseFloat(root.querySelector('.sig-weight').value) || 0,
    threshold: parseFloat(root.querySelector('.sig-threshold').value) || 0,
    description: (CONFIG.signals.find(s => s.id === root.dataset.id) || {}).description || '',
    candidates: root.querySelector('.sig-cands-ta').value.split('\n').map(x => x.trim()).filter(Boolean),
  }));
}

function addTier() {
  collectConfig();
  CONFIG.tiers.push({ id: `tier${Date.now().toString(36)}`, name: `Tier ${CONFIG.tiers.length + 1}`,
    provider: 'OpenAI-compatible', model: '', base_url: '', api_key: '', key_set: false });
  renderTierEditors();
}

function openSettings() {
  $('vllmUrl').value = CONFIG.vllm_sr_url || '';
  renderTierEditors(); renderCutoffs(); renderSignals();
  $('saveStatus').textContent = ''; $('settingsModal').hidden = false;
}
const closeSettings = () => { $('settingsModal').hidden = true; };

async function persist() {
  collectConfig();
  await fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(CONFIG) });
  CONFIG = await (await fetch('/api/config')).json();
  renderPills(); renderKeyBanner();
}

async function saveCfg() {
  const st = $('saveStatus'); st.className = 'save-status busy'; st.textContent = 'Saving…';
  await persist();
  st.className = 'save-status ok'; st.textContent = '✓ Saved (config/live_demo.local.json)';
}

async function applyCfg() {
  const st = $('saveStatus'); st.className = 'save-status busy';
  st.textContent = 'Saving + rebuilding router config + reloading vllm-sr…';
  await persist();
  const res = await fetch('/api/apply', { method: 'POST' }).then(r => r.json());
  if (res.ok) { st.className = 'save-status ok'; st.textContent = '✓ ' + res.detail; }
  else { st.className = 'save-status err'; st.textContent = `✕ apply failed (${res.step}): ${res.detail}`; }
}

async function resetCfg() {
  const st = $('saveStatus'); st.className = 'save-status busy'; st.textContent = 'Resetting…';
  await fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...CONFIG, _reset: true }) });
  CONFIG = await (await fetch('/api/config')).json();
  openSettings(); renderPills(); renderKeyBanner();
  st.className = 'save-status ok'; st.textContent = '✓ Reset';
}

function autoGrow() {
  const el = $('input'); el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

(async function init() {
  await boot();
  $('newChat').onclick = newChat;
  $('sendBtn').onclick = send;
  $('input').addEventListener('input', autoGrow);
  $('input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });
  $('exToggle').onclick = () => toggleExamples();
  $('hamburger').onclick = openSidebar;
  $('settingsBtn').onclick = openSettings;
  $('closeSettings').onclick = closeSettings;
  $('addTier').onclick = addTier;
  $('saveCfg').onclick = saveCfg;
  $('applyCfg').onclick = applyCfg;
  $('resetCfg').onclick = resetCfg;
  $('settingsModal').addEventListener('click', e => { if (e.target.id === 'settingsModal') closeSettings(); });
})();
