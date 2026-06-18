'use strict';
// Live interactive demo: every query is routed by a running vllm-sr. Config is a
// separate overlay on the server (config/live_demo.local.json); the canonical
// benchmark config is never touched. Settings → Apply rebuilds + reloads vllm-sr.

const TIER_COLORS = ['#1FB67A', '#11A8C4', '#F2A93B', '#E8743B', '#E64B5C', '#8B5CF6', '#0EA5E9'];
const tierColor = i => TIER_COLORS[i % TIER_COLORS.length];
const AXIS_MAX = 0.7;   // top of the difficulty scale (shared by chat meter + cutoff editor)
const OPUS_RATE_PER_M = 25;   // Opus 4.8 output ~$25 / 1M tokens — the frontier cost yardstick
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
  // If the page was reloaded while an Apply was in flight, re-attach to it so
  // the sidebar status reflects the reload progress.
  fetch('/api/apply/status').then(r => r.json())
    .then(s => { if (s && s.running) pollApply(null); }).catch(() => {});
}

// ── Chat sessions ──────────────────────────────────────────────────────────────
function loadChats() {
  try { chats = JSON.parse(localStorage.getItem('sr_chats') || '[]'); } catch { chats = []; }
  if (!Array.isArray(chats)) chats = [];
}
function saveChats() {
  // Drop transient `_`-prefixed fields (e.g. _html render cache, _animated) so
  // they don't bloat or stale localStorage.
  try {
    localStorage.setItem('sr_chats',
      JSON.stringify(chats.slice(0, 60), (k, v) => (k.startsWith('_') ? undefined : v)));
  } catch { /* quota */ }
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
// Curated landing-page prompts that override the auto-pick for a category.
// Keyed by a lowercase substring of the category name. The 'general' override
// is a meatier-but-still-trivial question than the stock "plural of analysis"
// — recall + a one-line contrast, which the router should still keep in Tier 1.
const SUGGESTION_OVERRIDES = [
  { match: 'general', q: "What's the difference between weather and climate?" },
];
function pickSuggestions() {
  const out = [];
  const cats = Object.keys(QUERIES);
  for (const cat of cats) {
    const list = QUERIES[cat] || [];
    if (!list.length) continue;
    const ov = SUGGESTION_OVERRIDES.find(o => cat.toLowerCase().includes(o.match));
    out.push({ cat, q: ov ? ov.q : list[Math.floor(list.length / 2)] });
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
    // Freeze the resolved card: build its HTML once and reuse. Re-renders
    // (triggered by sending a NEW message, or a settings change) then can't
    // mutate a prior answer's text/rationale. `_html` is in-memory only —
    // saveChats() drops `_`-prefixed keys so it never bloats localStorage.
    if (m._html == null) {
      m._html = (m.routing ? rationaleHTML(m.routing) : '')
        + answerHTML(m) + costHTML(m) + deeperHTML(m);
    }
    card.innerHTML = m._html;
    if (!m._animated) { card.classList.add('card-enter'); m._animated = true; }
    const btn = card.querySelector('.deeper-btn');
    if (btn) btn.onclick = () => { btn.disabled = true; ask(m.query, maxTierId()); };
    renderMath(card.querySelector('.answer'));
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
// Frontier-cost yardstick: what THIS answer's token count would cost on Opus 4.8.
function costHTML(m) {
  const u = m.usage || {};
  const total = u.total_tokens || ((u.prompt_tokens || 0) + (u.completion_tokens || 0));
  if (!total) return '';
  const opus = total / 1e6 * OPUS_RATE_PER_M;
  const here = CONFIG.tiers.find(t => t.id === (m.routing || {}).selected_tier_id);
  const onTier = here ? ` on ${esc(here.name)}` : '';
  return `<div class="costline" title="Same token count priced at Opus 4.8's ~$${OPUS_RATE_PER_M}/M output rate">
    <span class="ct-tok">${total.toLocaleString()} tokens${onTier}</span>
    <span class="ct-sep">·</span>
    <span class="ct-opus">same tokens on <b>Opus 4.8</b> (~$${OPUS_RATE_PER_M}/M) ≈ <b>$${opus.toFixed(4)}</b></span>
  </div>`;
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
  // Confidence is an AUTO-routing signal; when a tier is forced the router
  // didn't choose, so showing a "% confidence" alongside "forced" is a
  // contradiction — suppress it. Forced is labelled "(manual)" to distinguish.
  const conf = (!r.forced && r.confidence != null) ? Math.round(r.confidence * 100) : null;
  const verb = r.forced ? 'forced →' : 'routed →';
  const head = `<div class="rat-head">
    <span class="rat-badge${r.forced ? ' forced' : ''}" style="--c:${color}">${verb} ${esc(r.selected_tier_name || '?')}${r.forced ? ' <span class="rat-manual">(manual)</span>' : ''}</span>
    <span class="rat-model">${esc(r.served_model || '')}</span>
    ${conf != null ? `<span class="rat-conf" title="The router's confidence in THIS tier choice (from vLLM Semantic Router) — how decisively the difficulty score landed in this tier's band, not a measure of answer quality.">${conf}% confidence</span>` : ''}
  </div>`;
  return `<div class="rationale">${head}${difficultyMeter(r)}${signalBars(r)}</div>`;
}

function difficultyMeter(r) {
  const tiers = CONFIG.tiers, cuts = (CONFIG.tier_cutoffs || []).slice();
  if (tiers.length < 2 || cuts.length !== tiers.length - 1) return '';
  const lo = 0, hi = AXIS_MAX;
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
    // The score can go slightly negative (cheap signals subtract); a negative
    // "difficulty" reads as a bug, so clamp the SHOWN value to 0. The marker
    // position is clamped 0–100 so it always stays on the scale, and the value
    // label is anchored (left/center/right) so it never clips the card edge.
    const shown = Math.max(0, d);
    const pct = Math.max(0, Math.min(100, ((shown - lo) / (hi - lo)) * 100));
    const anchor = pct < 12 ? 'left' : pct > 88 ? 'right' : 'center';
    marker = `<div class="meter-marker anc-${anchor}" style="left:${pct}%">
      <span class="marker-val">difficulty ${shown.toFixed(3)}</span></div>`;
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
// Per-signal bars: each tunable signal's score against its threshold, with the
// bar lit up when the score crosses (the signal "fired"). Pairs the live score
// (from the router log) with the threshold the user set (from CONFIG).
function signalBars(r) {
  const sc = r.signal_confidences || {};
  const rows = (CONFIG.signals || []).map(s => {
    const score = sc['embedding:' + s.id];
    return {
      label: (SIGNAL_INFO[s.id] || {}).label || SIGNAL_LABELS[s.id] || s.id.replace(/_/g, ' '),
      score: typeof score === 'number' ? score : null,
      threshold: s.threshold ?? null,
      down: (s.weight ?? 0) < 0,
    };
  }).filter(x => x.score != null);
  if (!rows.length) {
    // No live scores (log read failed) — fall back to matched-signal names.
    const names = (r.matched || []).filter(m => m.type !== 'projection')
      .map(m => SIGNAL_LABELS[m.name] || m.name.replace(/_/g, ' '));
    if (!names.length) return '';
    return `<div class="rat-signals"><span class="rat-signals-label">Signals that fired</span>${
      names.map(n => `<span class="sigbar-name" style="margin-right:10px">${esc(n)}</span>`).join('')}</div>`;
  }
  rows.sort((a, b) => (b.score || 0) - (a.score || 0));
  const bars = rows.map(x => {
    const crossed = x.threshold != null && x.score >= x.threshold;
    const fill = Math.max(0, Math.min(100, x.score * 100));
    const thr = x.threshold != null ? Math.max(0, Math.min(100, x.threshold * 100)) : null;
    return `<div class="sigbar${crossed ? ' crossed' : ''}${x.down ? ' down' : ''}">
      <div class="sigbar-head">
        <span class="sigbar-name">${esc(x.label)}</span>
        <span class="sigbar-nums"><b>${x.score.toFixed(2)}</b>${x.threshold != null ? ` / thr ${x.threshold.toFixed(2)}` : ''}</span>
      </div>
      <div class="sigbar-track"><div class="sigbar-fill" style="width:${fill}%"></div>${
        thr != null ? `<div class="sigbar-thresh" style="left:${thr}%" title="threshold ${x.threshold.toFixed(2)}"></div>` : ''}</div>
    </div>`;
  }).join('');
  const foot = r.request_difficulty != null
    ? `<div class="rat-foot">Combined difficulty score <b>${Math.max(0, r.request_difficulty).toFixed(3)}</b> → ${esc(r.selected_tier_name || '')}</div>`
    : '';
  return `<div class="rat-signals"><span class="rat-signals-label">Signals — score vs. threshold</span>${bars}${foot}</div>`;
}

// minimal markdown: fenced code blocks, inline code, bold, paragraphs.
// Math spans ($$…$$, $…$, \[…\], \(…\)) are pulled out BEFORE markdown so the
// `**`/`<br>` passes can't mangle them, then restored verbatim for KaTeX to
// render (auto-render runs after insert; see renderMath). Math inside code
// fences is left alone — extraction happens only in the prose segments.
const MATH_RE = /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$/g;

// Inline spans: code, ***bold-italic***, **bold**, *italic*, _italic_, links.
function mdInline(t) {
  return t
    .replace(/`([^`]+)`/g, '<code class="ic">$1</code>')
    .replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
    .replace(/(^|[\s(])_([^_\n]+)_/g, '$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}
const _isBlockStart = l => /^#{1,6}\s/.test(l) || /^\s*[-*+]\s/.test(l)
  || /^\s*\d+[.)]\s/.test(l) || /^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(l);
// Block-level markdown over an already-esc'd string (math placeholders survive
// intact). Handles headings, ordered/unordered lists, horizontal rules, and
// paragraphs; inline formatting is applied per line.
function renderBlocks(s) {
  const lines = s.split('\n');
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const ln = lines[i];
    if (!ln.trim()) { i++; continue; }
    const h = ln.match(/^(#{1,6})\s+(.*)$/);
    if (h) { const lvl = Math.min(6, h[1].length + 2); out.push(`<h${lvl} class="md-h">${mdInline(h[2].trim())}</h${lvl}>`); i++; continue; }
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(ln)) { out.push('<hr class="md-hr">'); i++; continue; }
    if (/^\s*[-*+]\s+/.test(ln)) {
      const items = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) { items.push(mdInline(lines[i].replace(/^\s*[-*+]\s+/, '').trim())); i++; }
      out.push('<ul class="md-ul">' + items.map(x => `<li>${x}</li>`).join('') + '</ul>'); continue;
    }
    if (/^\s*\d+[.)]\s+/.test(ln)) {
      const items = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) { items.push(mdInline(lines[i].replace(/^\s*\d+[.)]\s+/, '').trim())); i++; }
      out.push('<ol class="md-ol">' + items.map(x => `<li>${x}</li>`).join('') + '</ol>'); continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !_isBlockStart(lines[i])) { para.push(mdInline(lines[i].trim())); i++; }
    out.push('<p>' + para.join('<br>') + '</p>');
  }
  return out.join('');
}

function mdToHtml(src) {
  const parts = String(src == null ? '' : src).split('```');
  let html = '';
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      const body = part.replace(/^[a-zA-Z0-9_+-]*\n/, '').replace(/\n$/, '');
      html += `<pre class="code"><code>${esc(body)}</code></pre>`;
      return;
    }
    const math = [];
    const held = part.replace(MATH_RE, m => `@@M${math.push(m) - 1}@@`);
    let block = renderBlocks(esc(held));
    // Restore math source (esc'd — entities decode back to literal chars in the
    // DOM text node, which is exactly what KaTeX reads).
    block = block.replace(/@@M(\d+)@@/g, (_, k) => esc(math[+k]));
    html += block;
  });
  return html || '<p></p>';
}

// KaTeX auto-render over a freshly-inserted node (no-op until the CDN lib loads).
function renderMath(root) {
  if (root && window.renderMathInElement) {
    window.renderMathInElement(root, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false },
        { left: '$', right: '$', display: false },
      ],
      throwOnError: false,
    });
  }
}
// If KaTeX finishes loading after the first paint, re-render once.
window.__katexReady = () => renderMessages();

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
    am.pending = false; am.routing = res.routing || null; am.text = res.answer || '';
    am.error = res.error || null; am.usage = res.usage || null;
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
    + `routing runs, but a tier with no key can't answer. <a id="bannerSettings">Open Model tiers →</a>`;
  $('bannerSettings').onclick = openTiers;
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

// ── Routing: difficulty cutoffs (band preview + sliders) ────────────────────────
function renderScoreAxis() {
  const host = $('scoreAxis'); if (!host) return;
  const cuts = CONFIG.tier_cutoffs || [];
  const edges = [0, ...cuts, AXIS_MAX];
  host.innerHTML = CONFIG.tiers.map((t, i) => {
    const w = Math.max(0, ((edges[i + 1] - edges[i]) / AXIS_MAX) * 100);
    return `<div class="ax-band" style="width:${w}%;background:${tierColor(i)}"><span class="ax-name">${esc(t.name)}</span></div>`;
  }).join('');
}
function renderCutoffs() {
  renderScoreAxis();
  const host = $('cutoffEditors'); host.innerHTML = '';
  (CONFIG.tier_cutoffs || []).forEach((c, i) => {
    const row = document.createElement('div'); row.className = 'cutoff-row';
    row.innerHTML = `<span class="co-label">${esc(CONFIG.tiers[i]?.name || 'T' + (i + 1))} → ${esc(CONFIG.tiers[i + 1]?.name || 'T' + (i + 2))}</span>
      <input type="range" min="0" max="${AXIS_MAX}" step="0.01" value="${c}">
      <span class="co-val">${(+c).toFixed(2)}</span>`;
    const input = row.querySelector('input'), val = row.querySelector('.co-val');
    input.addEventListener('input', () => {
      const cuts = CONFIG.tier_cutoffs;
      const lo = i > 0 ? cuts[i - 1] + 0.01 : 0.005;
      const hi = i < cuts.length - 1 ? cuts[i + 1] - 0.01 : AXIS_MAX - 0.005;
      const v = Math.min(Math.max(parseFloat(input.value), lo), hi);
      input.value = v.toFixed(2); cuts[i] = v; val.textContent = v.toFixed(2);
      renderScoreAxis();
    });
    host.appendChild(row);
  });
}

const SIGNAL_INFO = {
  trivial_lookup: { label: 'Looks trivial', hint: 'Short factual or lookup-style prompts — should route cheap.' },
  moderate_complexity: { label: 'Moderate reasoning', hint: 'Single- or multi-step reasoning and summarization.' },
  frontier_synthesis: { label: 'Frontier synthesis', hint: 'Hard, novel, cross-source synthesis — should route to the frontier model.' },
  argumentative_construction: { label: 'Argument construction', hint: 'Building structured arguments or proofs.' },
};
const sigInfo = s => SIGNAL_INFO[s.id] || { label: s.id, hint: s.description || '' };
const sigDir = w => w < 0 ? { txt: '↓ cheaper', cls: 'down' } : { txt: '↑ stronger', cls: 'up' };

function renderSignals() {
  const host = $('signalEditors'); host.innerHTML = '';
  (CONFIG.signals || []).forEach(s => {
    const node = $('signalEditorTpl').content.cloneNode(true);
    const root = node.querySelector('.signal-editor');
    root.dataset.id = s.id;
    node.querySelector('.sig-label').textContent = sigInfo(s).label;
    node.querySelector('.sig-blurb').textContent = sigInfo(s).hint;
    const w = node.querySelector('.sig-weight'), t = node.querySelector('.sig-threshold');
    const wv = node.querySelector('.sig-weight-val'), tv = node.querySelector('.sig-threshold-val');
    const dir = node.querySelector('.sig-dir');
    w.value = s.weight ?? 0; t.value = s.threshold ?? 0.6;
    const syncW = () => {
      const v = parseFloat(w.value);
      wv.textContent = (v > 0 ? '+' : '') + v.toFixed(2);
      const d = sigDir(v); dir.textContent = d.txt; dir.className = 'sig-dir ' + d.cls;
    };
    const syncT = () => { tv.textContent = parseFloat(t.value).toFixed(2); };
    syncW(); syncT();
    w.addEventListener('input', syncW);
    t.addEventListener('input', syncT);
    const ta = node.querySelector('.sig-cands-ta'), count = node.querySelector('.sig-count');
    const cands = s.candidates || [];
    ta.value = cands.join('\n'); count.textContent = cands.length;
    ta.addEventListener('input', () => { count.textContent = ta.value.split('\n').filter(l => l.trim()).length; });
    host.appendChild(node);
  });
}

// ── Collect per modal (only the open modal's editors are in the DOM) ────────────
function collectTiers() {
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
}
function collectRouting() {
  CONFIG.tier_cutoffs = [...$('cutoffEditors').querySelectorAll('input[type=range]')].map(i => parseFloat(i.value) || 0);
  CONFIG.signals = [...$('signalEditors').querySelectorAll('.signal-editor')].map(root => ({
    id: root.dataset.id,
    weight: parseFloat(root.querySelector('.sig-weight').value) || 0,
    threshold: parseFloat(root.querySelector('.sig-threshold').value) || 0,
    description: (CONFIG.signals.find(s => s.id === root.dataset.id) || {}).description || '',
    candidates: root.querySelector('.sig-cands-ta').value.split('\n').map(x => x.trim()).filter(Boolean),
  }));
}
function collectFor(which) { if (which === 'tiers') collectTiers(); else if (which === 'routing') collectRouting(); }

function addTier() {
  collectTiers();
  CONFIG.tiers.push({ id: `tier${Date.now().toString(36)}`, name: `Tier ${CONFIG.tiers.length + 1}`,
    provider: 'OpenAI-compatible', model: '', base_url: '', api_key: '', key_set: false });
  renderTierEditors();
}

function openTiers() {
  $('vllmUrl').value = CONFIG.vllm_sr_url || '';
  renderTierEditors();
  $('tiersStatus').textContent = ''; $('tiersStatus').className = 'save-status';
  $('tiersModal').hidden = false; closeSidebar();
}
function openRouting() {
  renderSignals(); renderCutoffs();
  $('routingStatus').textContent = ''; $('routingStatus').className = 'save-status';
  $('routingModal').hidden = false; closeSidebar();
}

async function persist(which) {
  collectFor(which);
  await fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(CONFIG) });
  CONFIG = await (await fetch('/api/config')).json();
  renderPills(); renderKeyBanner();
}

async function doSave(which) {
  const st = which === 'tiers' ? $('tiersStatus') : $('routingStatus');
  st.className = 'save-status busy'; st.textContent = 'Saving…';
  await persist(which);
  st.className = 'save-status ok'; st.textContent = '✓ Saved (config/live_demo.local.json)';
}

async function doApply(which) {
  const st = which === 'tiers' ? $('tiersStatus') : $('routingStatus');
  st.className = 'save-status';
  st.innerHTML = applyProgressHTML({ steps: [], step: 0, detail: 'Saving settings…' });
  setRouterStatus('warming');
  await persist(which);
  await fetch('/api/apply', { method: 'POST' });
  await pollApply(st);
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

function applyProgressHTML(s) {
  const steps = s.steps || [];
  const total = steps.length || 6;
  const idx = Math.min(s.step || 0, total);
  const waiting = !s.done && idx >= total - 1;
  const pct = s.done ? 100 : (waiting ? 92 : Math.round((idx / total) * 100));
  const label = s.done
    ? (s.ok ? '✓ Router is live and serving' : `✕ Apply failed${s.failed_step ? ` · ${esc(s.failed_step)}` : ''}`)
    : `Step ${Math.min(idx + 1, total)} of ${total} — ${esc(steps[idx] || s.phase || 'working…')}`;
  const cls = s.done ? (s.ok ? 'ok' : 'err') : '';
  return `<div class="apply-prog ${cls}">
    <div class="ap-bar"><div class="ap-fill ${waiting ? 'indet' : ''}" style="width:${pct}%"></div></div>
    <div class="ap-label">${label}</div>
    ${s.detail ? `<div class="ap-detail">${esc(s.detail)}</div>` : ''}
  </div>`;
}

function setRouterStatus(state) {
  const dot = document.querySelector('#routerStatus .status-dot');
  if (dot) dot.className = 'status-dot' + (state === 'live' ? '' : state === 'warming' ? ' warm' : ' down');
  $('routerTag').textContent = state === 'warming' ? 'reloading vllm-sr…' : (CONFIG.vllm_sr_url || 'vllm-sr');
}

// Poll the background apply until done, driving both the modal progress bar
// (if `st` is given) and the always-visible sidebar status dot.
async function pollApply(st) {
  for (;;) {
    let s;
    try { s = await (await fetch('/api/apply/status')).json(); } catch { await sleep(1200); continue; }
    if (st) st.innerHTML = applyProgressHTML(s);
    if (s.running) setRouterStatus('warming');
    if (s.done) {
      if (st) st.className = 'save-status ' + (s.ok ? 'ok' : 'err');
      setRouterStatus(s.ok ? 'live' : 'down');
      return s;
    }
    await sleep(1200);
  }
}

async function resetCfg() {
  const st = $('tiersStatus'); st.className = 'save-status busy'; st.textContent = 'Resetting…';
  await fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...CONFIG, _reset: true }) });
  CONFIG = await (await fetch('/api/config')).json();
  openTiers(); renderPills(); renderKeyBanner();
  st.className = 'save-status ok'; st.textContent = '✓ Reset';
}

// ── Diagnostics ──────────────────────────────────────────────
let DIAG_TIMER = null;

async function openDiag() {
  $('diagModal').hidden = false; closeSidebar();
  $('diagUpstreams').innerHTML = '<div class="diag-empty">loading…</div>';
  $('diagContainers').innerHTML = '';
  $('diagLog').innerHTML = '<div class="diag-empty">loading…</div>';
  $('diagEnvoyLog').innerHTML = '<div class="diag-empty">loading…</div>';
  await refreshDiag(true);
}

async function refreshDiag(full) {
  let d;
  try { d = await (await fetch(full ? '/api/diag' : '/api/diag/log')).json(); }
  catch { $('diagLog').innerHTML = '<div class="diag-empty">router not reachable</div>'; return; }
  if (full) { renderUpstreams(d.upstreams || []); renderContainers(d.containers || []); }
  renderDiagLog($('diagLog'), d.log || []);
  renderDiagLog($('diagEnvoyLog'), d.envoy_log || []);
}

function renderUpstreams(rows) {
  const el = $('diagUpstreams');
  if (!rows.length) { el.innerHTML = '<div class="diag-empty">no router-config yet — apply settings first</div>'; return; }
  el.innerHTML = `<table class="diag-tbl"><thead><tr>
      <th>Tier</th><th>Model forwarded</th><th>API</th><th>Chat URL</th></tr></thead><tbody>${
    rows.map(r => `<tr>
      <td class="mono">${esc(r.name || '—')}</td>
      <td class="mono">${esc(r.served_model || '—')}</td>
      <td>${esc(r.api_format || '—')}</td>
      <td class="mono url">${esc(r.chat_url || '—')}</td></tr>`).join('')}</tbody></table>`;
}

function renderContainers(rows) {
  const el = $('diagContainers');
  if (!rows.length) { el.innerHTML = '<div class="diag-empty">no vllm-sr containers running</div>'; return; }
  el.innerHTML = rows.map(c => {
    const up = /^Up\b/.test(c.status || '');
    return `<span class="diag-cont ${up ? 'up' : 'down'}"><span class="cdot"></span>
      <span class="cname mono">${esc(c.name)}</span><span class="cstatus">${esc(c.status)}</span></span>`;
  }).join('');
}

function renderDiagLog(el, rows) {
  if (!rows.length) { el.innerHTML = '<div class="diag-empty">no log lines yet — send a prompt</div>'; return; }
  const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
  el.innerHTML = rows.map(r => {
    const lvl = (r.level || 'info').toLowerCase();
    const fields = Object.entries(r.fields || {}).map(([k, v]) =>
      `<span class="lf"><span class="lk">${esc(k)}</span><span class="lv">${esc(String(v))}</span></span>`).join('');
    return `<div class="logrow lvl-${esc(lvl)}">
      <span class="lts">${esc(r.ts || '')}</span>
      <span class="lmsg">${esc(r.msg || '')}</span>${fields}</div>`;
  }).join('');
  if (stick) el.scrollTop = el.scrollHeight;
}

function toggleDiagAuto() {
  if ($('diagAuto').checked) {
    DIAG_TIMER = setInterval(() => refreshDiag(false), 2000);
  } else if (DIAG_TIMER) { clearInterval(DIAG_TIMER); DIAG_TIMER = null; }
}

function closeDiag() {
  $('diagModal').hidden = true;
  if (DIAG_TIMER) { clearInterval(DIAG_TIMER); DIAG_TIMER = null; }
  if ($('diagAuto')) $('diagAuto').checked = false;
}

function autoGrow() {
  const el = $('input'); el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

(async function init() {
  await boot();
  $('newChat').onclick = newChat;
  $('sendBtn').onclick = send;
  $('input').addEventListener('input', autoGrow);
  // Enter submits; Shift+Enter newlines. Guard IME composition (e.isComposing /
  // keyCode 229) so committing a candidate doesn't fire a send. keydown (not
  // keypress) so it fires reliably across browsers.
  $('input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
      e.preventDefault(); toggleExamples(false); send();
    }
  });
  // Focusing/typing in the composer closes the Examples popover so it can't sit
  // over the send button and swallow the click/submit.
  $('input').addEventListener('focus', () => toggleExamples(false));
  $('exToggle').onclick = e => { e.stopPropagation(); toggleExamples(); };
  // Click-away closes the Examples popover.
  document.addEventListener('click', e => {
    const pop = $('examplesPop');
    if (!pop || pop.hidden) return;
    if (!pop.contains(e.target) && e.target !== $('exToggle')) toggleExamples(false);
  });
  $('hamburger').onclick = openSidebar;
  $('tiersBtn').onclick = openTiers;
  $('routingBtn').onclick = openRouting;
  $('diagBtn').onclick = openDiag;
  $('diagRefresh').onclick = () => refreshDiag(true);
  $('diagAuto').onchange = toggleDiagAuto;
  $('addTier').onclick = addTier;
  $('resetCfg').onclick = resetCfg;
  document.querySelectorAll('[data-save]').forEach(b => b.onclick = () => doSave(b.dataset.save));
  document.querySelectorAll('[data-apply]').forEach(b => b.onclick = () => doApply(b.dataset.apply));
  document.querySelectorAll('[data-close]').forEach(b => b.onclick = () =>
    (b.dataset.close === 'diagModal' ? closeDiag() : ($(b.dataset.close).hidden = true)));
  ['tiersModal', 'routingModal'].forEach(id => $(id).addEventListener('click',
    e => { if (e.target.id === id) $(id).hidden = true; }));
  $('diagModal').addEventListener('click', e => { if (e.target.id === 'diagModal') closeDiag(); });
})();
