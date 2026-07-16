// Mnemonic — Frontend Application
const API = '';
let TOKEN = localStorage.getItem('mnemonic_token') || '';
let currentChallengeId = null;
let currentChallengeLang = 'Python';

window.addEventListener('DOMContentLoaded', () => {
  if (TOKEN) document.getElementById('token-input').value = TOKEN;
  showView('dashboard');
  loadTopicsForSelect();
});

function setToken(val) {
  TOKEN = val.trim();
  localStorage.setItem('mnemonic_token', TOKEN);
  setStatus('Token saved', 'green');
  loadDashboard();
  loadTopicsForSelect();
}

function headers() {
  return { 'Content-Type': 'application/json', 'Authorization': `Bearer ${TOKEN}` };
}

function setStatus(msg, color = '') {
  const el = document.getElementById('status-text');
  el.textContent = msg;
  el.style.color = color === 'green' ? 'var(--green)'
                 : color === 'red'   ? 'var(--red)'
                 : color === 'yellow'? 'var(--yellow)'
                 : 'var(--text-muted)';
  if (msg) setTimeout(() => { if (el.textContent === msg) el.textContent = ''; }, 4000);
}

async function api(method, path, body) {
  const opts = { method, headers: headers() };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

function showView(name) {
  document.querySelectorAll('.view').forEach(v => { v.classList.remove('active'); v.style.display = 'none'; });
  document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
  const view = document.getElementById('view-' + name);
  if (view) { view.style.display = name === 'chat' ? 'flex' : 'block'; view.classList.add('active'); }
  const navEl = document.getElementById('nav-' + name);
  if (navEl) navEl.classList.add('active');
  const titles = { dashboard: 'Dashboard', challenge: 'Challenge', chat: 'Knowledge Chat', topics: 'Topics', logs: 'Study Log' };
  document.getElementById('view-title').textContent = titles[name] || name;
  if (name === 'dashboard') loadDashboard();
  if (name === 'topics') loadTopics();
  if (name === 'logs') loadLogs();
}

//  Dashboard 

async function loadDashboard() {
  if (!TOKEN) return;
  try {
    const [pending, topics, logs] = await Promise.all([
      api('GET', '/challenge/pending'),
      api('GET', '/topics'),
      api('GET', '/logs/recent?limit=5').catch(() => ({ entries: [] })),
    ]);
    document.getElementById('stat-pending').textContent = pending.count;
    document.getElementById('stat-topics').textContent = topics.count;

    const dc = document.getElementById('dash-challenges');
    if (!pending.challenges.length) {
      dc.innerHTML = '<div class="text-muted text-sm">No pending challenges. Generate one from the Challenge view.</div>';
    } else {
      dc.innerHTML = pending.challenges.map(c => `
        <div class="topic-row" style="cursor:pointer" onclick="loadChallenge(${c.id})">
          <div><div class="topic-name">${c.title}</div><div class="topic-lang">${c.programming_language} · ${c.topic_name}</div></div>
          <span class="badge badge-${c.difficulty === 'easy' ? 'easy' : c.difficulty === 'intermediate' ? 'inter' : 'advanced'}">${c.difficulty}</span>
        </div>`).join('');
    }

    const dl = document.getElementById('dash-logs');
    if (!logs.entries.length) {
      dl.innerHTML = '<div class="text-muted text-sm">No submissions yet.</div>';
    } else {
      dl.innerHTML = logs.entries.map(e => `
        <div class="topic-row">
          <div><div class="topic-name">${e.title}</div><div class="topic-lang">${e.programming_language} · ${e.topic_name}</div></div>
          <div class="flex gap-8" style="align-items:center">
            <span style="font-size:18px;font-weight:700;font-family:var(--font-mono)" class="text-${e.grade >= 7 ? 'green' : e.grade >= 4 ? 'yellow' : 'red'}">${e.grade}/10</span>
            <span class="text-muted text-sm">${e.timestamp ? e.timestamp.slice(0,10) : ''}</span>
          </div>
        </div>`).join('');
    }
    loadProgressCharts();
  } catch (e) { setStatus(e.message, 'red'); }
}

//  Progress Charts 

async function loadProgressCharts() {
  if (!TOKEN) return;
  const el = document.getElementById('dash-progress');
  if (!el) return;
  try {
    const data = await api('GET', '/submit/topic-history').catch(() => ({ topics: {} }));
    const topics = data.topics || {};
    if (!Object.keys(topics).length) {
      el.innerHTML = '<div class="text-muted text-sm">Submit solutions to see your progress here.</div>';
      return;
    }
    el.innerHTML = Object.entries(topics).map(([key, entries]) => {
      const [topicName, lang] = key.split('|');
      const grades = [...entries].reverse().map(e => e.grade);
      const latest = grades[grades.length - 1] || 0;
      const avg = grades.length ? (grades.reduce((a,b)=>a+b,0)/grades.length).toFixed(1) : '-';
      return `
        <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 0;border-bottom:1px solid var(--border)">
          <div style="min-width:160px">
            <div class="topic-name">${topicName}</div>
            <div class="topic-lang">${lang} · ${entries.length} submission${entries.length !== 1 ? 's' : ''} · avg ${avg}/10</div>
          </div>
          <div style="display:flex;align-items:center;gap:16px">
            ${_sparkline(grades)}
            <span style="font-size:22px;font-weight:700;font-family:var(--font-mono);min-width:48px;text-align:right" class="text-${gradeClass(latest)}">${latest}/10</span>
          </div>
        </div>`;
    }).join('');
  } catch (e) { el.innerHTML = '<div class="text-muted text-sm">Progress data unavailable.</div>'; }
}

function _sparkline(grades) {
  if (!grades.length) return '<svg width="80" height="30"></svg>';
  const w = 80, h = 30, pad = 4;
  const pts = grades.map((g, i) => {
    const x = pad + (grades.length < 2 ? (w-pad*2)/2 : (i/(grades.length-1))*(w-pad*2));
    const y = h - pad - ((g-1)/9)*(h-pad*2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const last = grades[grades.length-1];
  const color = last >= 7 ? '#3fb950' : last >= 4 ? '#d29922' : '#f85149';
  const dots = grades.map((g, i) => {
    const x = pad + (grades.length < 2 ? (w-pad*2)/2 : (i/(grades.length-1))*(w-pad*2));
    const y = h - pad - ((g-1)/9)*(h-pad*2);
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.5" fill="${color}"/>`;
  }).join('');
  return `<svg width="${w}" height="${h}" style="overflow:visible;flex-shrink:0">
    ${grades.length > 1 ? `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>` : ''}
    ${dots}</svg>`;
}

// Topics 

async function loadTopics() {
  if (!TOKEN) return;
  try {
    const data = await api('GET', '/topics');
    const el = document.getElementById('topics-list');
    if (!data.topics.length) { el.innerHTML = '<div class="text-muted text-sm">No topics yet. Add one above.</div>'; return; }
    el.innerHTML = data.topics.map(t => `
      <div class="topic-row" id="topic-row-${t.id}">
        <div><div class="topic-name">${t.name}</div><div class="topic-lang">${t.programming_language}</div></div>
        <div class="flex gap-8" style="align-items:center">
          <div class="text-right">
            <div class="text-sm ${t.next_review_date ? '' : 'text-muted'}">${t.next_review_date ? 'Review: ' + t.next_review_date : 'Not reviewed yet'}</div>
            <div class="text-muted text-sm">R: ${(t.fsrs_retrievability*100).toFixed(0)}% · ${t.total_reviews} reviews</div>
          </div>
          <button class="btn btn-ghost" style="padding:4px 8px;font-size:18px" onclick="toggleTopicHistory('${t.name}','${t.programming_language}',${t.id})">📋</button>
        </div>
      </div>
      <div id="topic-history-${t.id}" class="hidden" style="background:var(--surface2);padding:12px 16px;border-bottom:1px solid var(--border)">
        <div class="text-muted text-sm">Loading...</div>
      </div>`).join('');
  } catch (e) { setStatus(e.message, 'red'); }
}

async function loadTopicsForSelect() {
  if (!TOKEN) return;
  try {
    const data = await api('GET', '/topics');
    const sel = document.getElementById('topic-select');
    if (!data.topics.length) { sel.innerHTML = '<option value="">No topics - add one first</option>'; return; }
    sel.innerHTML = data.topics.map(t =>
      `<option value="${t.name}|${t.programming_language}">${t.name} (${t.programming_language})</option>`
    ).join('');
  } catch (e) {}
}

async function addTopic() {
  const name = document.getElementById('new-topic-name').value.trim();
  const lang = document.getElementById('new-topic-lang').value;
  if (!name) { setStatus('Enter a topic name', 'yellow'); return; }
  try {
    await api('POST', '/topics', { name, programming_language: lang });
    document.getElementById('new-topic-name').value = '';
    setStatus(`Topic "${name}" added`, 'green');
    loadTopics(); loadTopicsForSelect();
  } catch (e) { setStatus(e.message, 'red'); }
}

async function toggleTopicHistory(topicName, progLang, topicId) {
  const panel = document.getElementById(`topic-history-${topicId}`);
  if (!panel.classList.contains('hidden')) { panel.classList.add('hidden'); return; }
  panel.classList.remove('hidden');
  try {
    const data = await api('GET', '/submit/topic-history').catch(() => ({ topics: {} }));
    const key = `${topicName}|${progLang}`;
    const entries = (data.topics || {})[key] || [];
    if (!entries.length) { panel.innerHTML = '<div class="text-muted text-sm">No submissions yet for this topic.</div>'; return; }
    const grades = entries.map(e => e.grade).reverse();
    panel.innerHTML = `
      <div style="margin-bottom:8px">${_sparkline(grades)}</div>
      ${entries.map(e => `
        <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid var(--border)">
          <span class="text-muted text-sm">${e.date}</span>
          <span style="font-size:15px;font-weight:700;font-family:var(--font-mono)" class="text-${gradeClass(e.grade)}">${e.grade}/10</span>
        </div>
        ${e.notes ? `<div class="text-muted text-sm" style="padding:3px 0 6px;font-style:italic">${e.notes}</div>` : ''}
      `).join('')}`;
  } catch (err) { panel.innerHTML = `<div class="text-red text-sm">${err.message}</div>`; }
}

async function quickAddTopic(name, lang, btn) {
  try {
    await api('POST', '/topics', { name, programming_language: lang });
    btn.textContent = `✓ ${name}`;
    btn.disabled = true;
    btn.classList.add('text-green');
    loadTopicsForSelect();
    setStatus(`Topic "${name}" added`, 'green');
  } catch (e) { setStatus(e.message, 'red'); }
}

//  Challenge 

async function generateChallenge() {
  const val = document.getElementById('topic-select').value;
  if (!val) { setStatus('Select a topic first', 'yellow'); return; }
  const [topicName, progLang] = val.split('|');
  currentChallengeLang = progLang;
  const difficulty = document.getElementById('difficulty-select').value;
  const btn = document.getElementById('btn-generate');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Generating...';
  setStatus('Generating challenge...', '');
  document.getElementById('suggestion-panel').classList.add('hidden');
  try {
    const c = await api('POST', '/challenge/generate', { topic_name: topicName, programming_language: progLang, difficulty });
    renderChallenge(c);
    setStatus('Challenge ready', 'green');
  } catch (e) { setStatus(e.message, 'red'); }
  finally { btn.disabled = false; btn.innerHTML = '⚡ Generate'; }
}

async function loadChallenge(id) {
  showView('challenge');
  try { const c = await api('GET', `/challenge/${id}`); renderChallenge(c); }
  catch (e) { setStatus(e.message, 'red'); }
}

function renderChallenge(c) {
  currentChallengeId = c.id;
  currentChallengeLang = c.programming_language;
  document.getElementById('challenge-content').classList.remove('hidden');
  document.getElementById('result-panel').classList.add('hidden');
  document.getElementById('suggestion-panel').classList.add('hidden');
  const diffMap = { easy: 'badge-easy', intermediate: 'badge-inter', advanced: 'badge-advanced' };
  document.getElementById('challenge-difficulty-badge').className = `badge ${diffMap[c.difficulty] || 'badge-inter'}`;
  document.getElementById('challenge-difficulty-badge').textContent = c.difficulty;
  document.getElementById('challenge-id-label').textContent = `#${c.id}`;
  document.getElementById('challenge-title').textContent = c.title;
  document.getElementById('challenge-desc').textContent = c.description;
  document.getElementById('challenge-expected').textContent = c.expected_behavior;
  document.getElementById('challenge-starter').textContent = c.starter_code;
  document.getElementById('challenge-requirements').innerHTML = (c.requirements || []).map(r => `<li>${r}</li>`).join('');
  document.getElementById('challenge-refs').innerHTML = (c.knowledge_refs || []).map(r =>
    `<div class="text-muted text-sm" style="padding:4px 0;border-bottom:1px solid var(--border)">${r}</div>`).join('');
  document.getElementById('code-editor').value = '';
}

function clearEditor() {
  document.getElementById('code-editor').value = '';
  document.getElementById('result-panel').classList.add('hidden');
  document.getElementById('suggestion-panel').classList.add('hidden');
}

async function submitSolution() {
  if (!currentChallengeId) { setStatus('Generate a challenge first', 'yellow'); return; }
  const code = document.getElementById('code-editor').value.trim();
  if (!code) { setStatus('Paste your code first', 'yellow'); return; }
  const btn = document.getElementById('btn-submit');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Evaluating...';
  setStatus('Running code and grading...', '');
  try {
    const result = await api('POST', `/submit/${currentChallengeId}`, { code });
    renderResult(result);
    const title = document.getElementById('challenge-title').textContent;
    const topicVal = document.getElementById('topic-select').value;
    const [topicName, progLang] = topicVal ? topicVal.split('|') : ['', currentChallengeLang];
    await api('POST', '/logs/submission', {
      challenge_id: currentChallengeId,
      topic_name: topicName || title,
      programming_language: progLang || currentChallengeLang,
      difficulty: document.getElementById('difficulty-select').value,
      title, code_submitted: code, grade: result.grade,
      grade_rationale: result.grade_rationale, feedback: result.feedback,
      citations: result.citations, execution: result.execution, fsrs: result.fsrs,
    }).catch(() => {});
    _suggestTopics(result.feedback, result.citations, progLang || currentChallengeLang);
    setStatus(`Grade: ${result.grade}/10`, result.grade >= 7 ? 'green' : result.grade >= 4 ? 'yellow' : 'red');
  } catch (e) { setStatus(e.message, 'red'); }
  finally { btn.disabled = false; btn.innerHTML = '▶ Submit & Evaluate'; }
}

function gradeClass(g) {
  if (g <= 3) return 'poor'; if (g <= 6) return 'average'; if (g <= 9) return 'good'; return 'perfect';
}

function gradeLabel(g) {
  if (g <= 3) return 'Poor - review tomorrow'; if (g <= 6) return 'Average - keep practicing';
  if (g <= 9) return 'Good - solid understanding'; return 'Perfect - excellent work!';
}

function renderResult(r) {
  document.getElementById('result-panel').classList.remove('hidden');
  const badge = document.getElementById('result-grade-badge');
  badge.textContent = `${r.grade}/10`;
  badge.className = `grade-badge ${gradeClass(r.grade)}`;
  document.getElementById('result-grade-label').textContent = gradeLabel(r.grade);
  document.getElementById('result-rationale').textContent = r.grade_rationale;
  document.getElementById('result-feedback').textContent = r.feedback;
  const exec = r.execution;
  const execEl = document.getElementById('result-execution');
  if (exec.compile_success && exec.exit_code === 0) {
    execEl.innerHTML = `<div class="text-green text-sm">✓ Executed in ${exec.execution_ms}ms</div>${exec.stdout ? `<div class="starter-code mt-12">${exec.stdout}</div>` : ''}`;
  } else {
    execEl.innerHTML = `<div class="text-red text-sm">✗ ${exec.timed_out ? 'Timed out' : exec.compile_success ? 'Runtime error' : 'Compile error'}</div>${exec.compile_stderr ? `<div class="starter-code mt-12">${exec.compile_stderr}</div>` : ''}${exec.stderr ? `<div class="starter-code mt-12">${exec.stderr}</div>` : ''}`;
  }
  const citEl = document.getElementById('result-citations');
  citEl.innerHTML = !r.citations.length ? '<div class="text-muted text-sm">No citations.</div>'
    : r.citations.map(c => `<div class="citation"><div class="citation-source">${c.source_title} · page ${c.source_page}</div><div class="citation-quote">"${c.exact_source_quote}"</div></div>`).join('');
  document.getElementById('result-fsrs').innerHTML =
    `<span class="text-muted">Next review:</span> <span class="text-accent mono">${r.fsrs.next_review_date}</span> <span class="text-muted">(${r.fsrs.interval_days} days) · stability: ${r.fsrs.new_stability}</span>`;
  document.getElementById('result-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

//  Topic Suggestions 

const KNOWN_TOPICS = [
  'Attention Mechanism','Backpropagation','Gradient Descent','Neural Network',
  'Convolutional Neural Network','Recurrent Neural Network','LSTM','Transformer',
  'Transfer Learning','Regularization','Dropout','Batch Normalization',
  'Decision Tree','Random Forest','K-Means Clustering','Support Vector Machine',
  'Logistic Regression','Linear Regression','Principal Component Analysis',
  'Dynamic Programming','Graph Algorithms','Binary Search','Sorting',
  'Hash Table','Binary Tree','Recursion','Greedy Algorithm',
  'Memory Management','Pointer','Smart Pointer','Move Semantics',
  'Design Patterns','Object-Oriented Programming','SOLID Principles',
  'SQL','Graph Database','Indexing','Query Optimization',
  'Natural Language Processing','Tokenization','Language Model',
  'Object Detection','Image Segmentation','Feature Extraction',
  'Linear Algebra','Matrix Factorization','Optimization','Probability',
  'Concurrency','Threading','Async Await','Decorator','Generator',
  'LINQ','Generics','Lambda','Iterator','Template',
  'Topological Sort', 'Graph Traversal', 'Cycle Detection', 'Shortest Path',
  'Tree Traversal', 'Stack', 'Queue', 'Depth-First Search', 'Breadth-First Search',
  'Topological Sorting', 'DAG', 'Directed Graph',
];

async function _suggestTopics(feedback, citations, lang) {
  if (!TOKEN) return;
  const panel = document.getElementById('suggestion-panel');
  if (!panel) return;
  try {
    const topicsData = await api('GET', '/topics');
    const existing = new Set(topicsData.topics.map(t => t.name.toLowerCase()));
    const challengeTitle = document.getElementById('challenge-title') ? document.getElementById('challenge-title').textContent : '';
    const text = (challengeTitle + ' ' + feedback + ' ' + citations.map(c => `${c.source_title} ${c.exact_source_quote}`).join(' ')).toLowerCase();
    const suggestions = KNOWN_TOPICS.filter(t => !existing.has(t.toLowerCase()) && text.includes(t.toLowerCase())).slice(0, 4);
    if (!suggestions.length) { panel.classList.add('hidden'); return; }
    panel.classList.remove('hidden');
    panel.innerHTML = `
      <div class="card" style="border-color:var(--accent-dim);margin-top:0">
        <div class="card-title">💡 Topics detected in this submission</div>
        <div class="text-muted text-sm" style="margin-bottom:10px">Add these to your study list?</div>
        <div class="flex gap-8" style="flex-wrap:wrap">
          ${suggestions.map(t => `<button class="btn btn-ghost text-sm" onclick="quickAddTopic('${t}','${lang}',this)">+ ${t}</button>`).join('')}
        </div>
      </div>`;
  } catch (e) {}
}

// Chat

function chatKeydown(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); } }

async function sendChat() {
  const input = document.getElementById('chat-input');
  const query = input.value.trim();
  if (!query) return;
  input.value = '';
  appendMessage('user', query);
  const btn = document.getElementById('btn-chat-send');
  btn.disabled = true; btn.textContent = '...';
  try {
    const result = await api('POST', '/chat', { query, user_language: 'EN' });
    let content = result.response;
    if (result.citations && result.citations.length)
      content += '\n\n' + result.citations.map(c => `[${c.source_title}, p.${c.source_page}]`).join(' · ');
    appendMessage('assistant', content, result.reviewer_verdict);
  } catch (e) { appendMessage('assistant', `Error: ${e.message}`); }
  finally { btn.disabled = false; btn.textContent = 'Send'; }
}

function appendMessage(role, text, verdict) {
  const el = document.getElementById('chat-messages');
  const msg = document.createElement('div');
  msg.className = `message ${role}`;
  const vb = verdict && role === 'assistant' ? `<div style="font-size:11px;color:var(--text-muted);margin-top:4px">${verdict}</div>` : '';
  msg.innerHTML = `<div class="message-bubble">${text.replace(/\n/g,'<br>')}${vb}</div>`;
  el.appendChild(msg);
  el.scrollTop = el.scrollHeight;
}

// Logs 

async function loadLogs() {
  if (!TOKEN) return;
  try {
    const data = await api('GET', '/logs/recent?limit=20');
    const el = document.getElementById('log-entries');
    if (!data.entries.length) { el.innerHTML = '<div class="text-muted text-sm">No submissions logged yet.</div>'; return; }
    el.innerHTML = data.entries.map((e, idx) => {
      const logId = e.log_id || e.challenge_id || idx;
      return `
        <div class="card" style="margin-bottom:8px">
          <div class="flex" style="justify-content:space-between;align-items:center;margin-bottom:8px">
            <div><span class="topic-name">${e.title || 'Submission'}</span><span class="topic-lang" style="margin-left:8px">${e.programming_language || ''}</span></div>
            <div class="flex gap-8" style="align-items:center">
              <span style="font-size:20px;font-weight:700;font-family:var(--font-mono)" class="text-${e.grade >= 7 ? 'green' : e.grade >= 4 ? 'yellow' : 'red'}">${e.grade}/10</span>
              <span class="text-muted text-sm">${e.timestamp ? e.timestamp.slice(0,16).replace('T',' ') : ''}</span>
            </div>
          </div>
          <div class="text-muted text-sm" style="margin-bottom:8px">${e.grade_rationale || ''}</div>
          ${e.citations && e.citations.length ? e.citations.map(c => `<div class="citation"><div class="citation-source">${c.source_title} · p.${c.source_page}</div><div class="citation-quote">"${c.exact_source_quote}"</div></div>`).join('') : ''}
          <div class="text-muted text-sm" style="margin-top:8px">Next review: <span class="text-accent mono">${e.fsrs ? e.fsrs.next_review_date : '-'}</span>${e.fsrs ? ` (${e.fsrs.interval_days} days)` : ''}</div>
          <div style="margin-top:10px">
            <textarea id="note-${logId}" placeholder="Add a personal note about this submission..." style="width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:7px 10px;color:var(--text);font-size:12px;font-family:var(--font-ui);resize:vertical;min-height:48px;outline:none">${e.notes || ''}</textarea>
            <button class="btn btn-ghost text-sm" style="margin-top:5px" onclick="saveNote(${logId},'note-${logId}')">Save Note</button>
          </div>
        </div>`;
    }).join('');
  } catch (e) { setStatus(e.message, 'red'); }
}

async function saveNote(logId, textareaId) {
  const ta = document.getElementById(textareaId);
  if (!ta) return;
  try {
    await api('PATCH', `/logs/note/${logId}`, { notes: ta.value });
    setStatus('Note saved', 'green');
  } catch (e) { setStatus(`Note save failed: ${e.message}`, 'red'); }
}

async function syncLogs() {
  setStatus('Checking log status...', '');
  try {
    const data = await api('POST', '/logs/sync');
    const panel = document.getElementById('sync-result');
    if (panel) {
      panel.classList.remove('hidden');
      panel.innerHTML = `
        <div class="card" style="border-color:var(--accent-dim);margin-top:12px">
          <div class="card-title">Sync Instructions</div>
          <div class="text-sm mt-12">${data.message}</div>
          <div class="flex gap-8 mt-12" style="align-items:center">
            <code class="mono text-accent" style="flex:1;background:var(--surface2);padding:8px 10px;border-radius:var(--radius);font-size:12px;overflow-x:auto">${data.sync_command}</code>
            <button class="btn btn-ghost text-sm" onclick="navigator.clipboard.writeText('${data.sync_command}').then(()=>setStatus('Copied!','green'))">Copy</button>
          </div>
        </div>`;
    }
    setStatus(`${data.entries} log entries ready to sync`, 'green');
  } catch (e) { setStatus(e.message, 'red'); }
}