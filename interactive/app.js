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

const tierIndex = id => CONFIG.tiers.findIndex(t => t.id === id);
const maxTierId = () => CONFIG.tiers.length ? CONFIG.tiers[CONFIG.tiers.length - 1].id : null;

// ── Boot ──────────────────────────────────────────────────────────────────────
async function boot() {
  CONFIG = await (await fetch('/api/config')).json();
  QUERIES = (await (await fetch('/api/queries')).json()).categories || {};
  $('routerTag').textContent = 'routes via vllm-sr at ' + (CONFIG.vllm_sr_url || '?');
  renderPills(); renderKeyBanner(); renderExamples();
}

// ── Composer ────────────────────────────────────────────────────────────────
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
    b.onclick = () => { $('input').value = q; autoGrow(); $('input').focus(); };
    chips.appendChild(b);
  });
}

// ── Chat ───────────────────────────────────────────────────────────────────────
function appendUser(text) {
  $('emptyState')?.remove();
  const el = document.createElement('div');
  el.className = 'msg-user'; el.textContent = text;
  $('chat').appendChild(el); el.scrollIntoView({ block: 'end' });
}
function appendAssistant() {
  const el = document.createElement('div');
  el.className = 'msg-assistant';
  el.innerHTML = `<div class="answer"><span class="model-line">routing via vllm-sr…</span></div>`;
  $('chat').appendChild(el); el.scrollIntoView({ block: 'end' });
  return el;
}

function routingHTML(r) {
  const i = tierIndex(r.selected_tier_id);
  const color = i >= 0 ? tierColor(i) : 'var(--blue)';
  const ladder = CONFIG.tiers.map((t, j) => {
    const on = t.id === r.selected_tier_id;
    return `<span class="tier-chip${on ? ' on' : ''}" ${on ? `style="background:${tierColor(j)};border-color:${tierColor(j)}"` : ''}>
      <span class="tdot" style="background:${tierColor(j)}"></span>${esc(t.name)}</span>`;
  }).join('');
  const meta = [];
  if (r.category) meta.push(`category <b>${esc(r.category)}</b>`);
  if (r.reasoning) meta.push(`reasoning <b>${esc(r.reasoning)}</b>`);
  if (r.cache_hit) meta.push('cache hit');
  const forced = r.forced
    ? `<div class="forced-note">You forced this tier; auto would let vllm-sr classify.</div>` : '';
  return `<div class="routing">
    <div class="routed-banner">
      <span class="routed-pill" style="background:${color}">routed → ${esc(r.selected_tier_name || r.selected_tier_id || '?')}</span>
      <span class="routed-model">${esc(r.served_model || '')}</span>
      <span class="routed-meta">${meta.join(' · ')}</span>
    </div>
    <div class="tier-ladder">${ladder}</div>${forced}
  </div>`;
}

function answerHTML(res) {
  if (res.error) return `<div class="answer err">${esc(res.error)}</div>`;
  const r = res.routing || {};
  const tier = CONFIG.tiers.find(t => t.id === r.selected_tier_id);
  if (tier && !tier.key_set) {
    return `<div class="answer err">Routed to <strong>${esc(r.selected_tier_name)}</strong>
      (${esc(r.served_model)}), but it has no API key — add one in Settings. The routing above is real.</div>`;
  }
  return `<div class="answer"><span class="model-line">answered by ${esc(r.selected_tier_name || '')} · ${esc(r.served_model || '')}</span>${esc(res.answer || '')}</div>`;
}

function deeperHTML(res) {
  const r = res.routing || {};
  const mx = maxTierId();
  if (res.error || !mx || r.selected_tier_id === mx) return '';
  const name = CONFIG.tiers.find(t => t.id === mx)?.name || 'top tier';
  return `<div class="deeper"><button class="deeper-btn">↑ Get a deeper answer (${esc(name)})</button></div>`;
}

function renderInto(card, query, res) {
  card.innerHTML = (res.routing ? routingHTML(res.routing) : '') + answerHTML(res) + deeperHTML(res);
  const btn = card.querySelector('.deeper-btn');
  if (btn) btn.onclick = () => { btn.disabled = true; ask(query, maxTierId()); };
  card.scrollIntoView({ block: 'end' });
}

async function ask(query, mode) {
  const card = appendAssistant();
  try {
    const res = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, mode }),
    }).then(r => r.json());
    renderInto(card, query, res);
  } catch (e) {
    card.innerHTML = `<div class="answer err">Request failed: ${esc(e.message)}.</div>`;
  }
}
function send() {
  const q = $('input').value.trim();
  if (!q) return;
  $('input').value = ''; autoGrow();
  appendUser(q); ask(q, routeMode);
}

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
    // Link the model datalist to its input and wire the "load models" button.
    const dl = node.querySelector('.te-models');
    dl.id = 'te-models-' + t.id;
    node.querySelector('.te-model').setAttribute('list', dl.id);
    node.querySelector('.te-load').onclick = ev => loadTierModels(root, t.id, ev.currentTarget);
    host.appendChild(node);
  });
}

async function loadTierModels(root, tierId, btn) {
  // Ask the server to fetch this provider's model list with the tier's key.
  // The key field is blank unless just typed; the server falls back to the
  // saved overlay key for this tier, so an already-saved tier works too.
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
    btn.title = models.length
      ? `${models.length} models — click the model field to pick`
      : 'provider returned no models';
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

// Plain-language labels for the built-in signals (fall back to the id + the
// signal's own description for any custom ones).
const SIGNAL_INFO = {
  trivial_lookup:             { label: 'Looks trivial',         hint: 'Short factual or lookup-style prompts.' },
  moderate_complexity:        { label: 'Moderate reasoning',    hint: 'Single- or multi-step reasoning and summarization.' },
  frontier_synthesis:         { label: 'Frontier synthesis',    hint: 'Hard, novel, cross-source synthesis.' },
  argumentative_construction: { label: 'Argument construction', hint: 'Building structured arguments or proofs.' },
};
const sigInfo = s => SIGNAL_INFO[s.id] || { label: s.id, hint: s.description || '' };
const sigDir = w => w < 0
  ? { txt: '↓ toward cheaper tiers', cls: 'down' }
  : { txt: '↑ toward stronger tiers', cls: 'up' };

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
      node.querySelector('.sig-chevron') && (toggle.querySelector('.sig-chevron').textContent = opening ? '▾' : '▸');
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
      api_key: keyInput,                       // blank = unchanged (server preserves)
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
  // reload masked view (keys never round-trip through the browser)
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
  // discard the local overlay by re-fetching the committed default isn't exposed;
  // simplest: clear keys client-side and reload from server default by deleting local.
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
  $('sendBtn').onclick = send;
  $('input').addEventListener('input', autoGrow);
  $('input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });
  $('settingsBtn').onclick = openSettings;
  $('closeSettings').onclick = closeSettings;
  $('addTier').onclick = addTier;
  $('saveCfg').onclick = saveCfg;
  $('applyCfg').onclick = applyCfg;
  $('resetCfg').onclick = resetCfg;
  $('settingsModal').addEventListener('click', e => { if (e.target.id === 'settingsModal') closeSettings(); });
})();
