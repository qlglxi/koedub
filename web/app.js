/* KoeDub 前端逻辑：零依赖，原生 fetch + XHR 上传进度 */
'use strict';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};

const STAGES = [
  ['probe', '① 媒体探测'],
  ['channel', '② 左右耳声源判定'],
  ['segment', '③ VAD 分段'],
  ['asr', '④ 语音识别'],
  ['translate', '⑤ 自适应译中'],
  ['tts', '⑥ AI 配音'],
  ['align', '⑦ 排轴对齐'],
  ['mix', '⑧ 声道摆位与合成'],
  ['subtitle', '⑨ 字幕导出'],
  ['export', '⑩ 成品导出'],
];

const S = {
  cfg: null,
  registry: null,
  file: null,       // {alias, filename, size, media, url}
  jobId: null,
  since: 0,
  polling: false,
  result: null,
};

/* ============================ 初始化 ============================ */
async function boot() {
  buildSteps();
  bindUI();
  await loadProviders();
  await loadConfig();
  await checkHealth();
  await loadVoices();
  await loadDeps();
  await loadDiag();
  refreshOutputs();
}

function buildSteps() {
  const ul = $('stepList');
  ul.innerHTML = '';
  STAGES.forEach(([id, name]) => {
    const li = el('li', '', `<span class="dot"></span><span>${name}</span>`);
    li.dataset.stage = id;
    ul.appendChild(li);
  });
}

async function checkHealth() {
  try {
    const h = await (await fetch('/api/health')).json();
    if (!h.ffmpeg) {
      log('【错误】未找到 ffmpeg/ffprobe，无法处理音频。请安装 FFmpeg 后重启。', 'e');
    } else {
      log('环境就绪：Python ' + h.python + ' · numpy ' + h.numpy);
      log('ffmpeg：' + h.ffmpegPath, 'g');
    }
  } catch (e) { log('无法连接后端：' + e, 'e'); }
}

/* ============================ API 清单 ============================ */
async function loadProviders() {
  const r = await (await fetch('/api/providers')).json();
  S.registry = r.registry;
  const cap = r.registry.capabilityTotal;
  $('apiCount').textContent = cap;
  $('apiBadge').onclick = showApiModal;
  renderApiModal();

  fillSelect($('asrProvider'), r.registry, 'asr', true);
  fillSelect($('trProvider'), r.registry, 'translate', true);
  fillSelect($('ttsProvider'), r.registry, 'tts', false);
}

function capOf(id) { return (S.registry.capabilities || []).find(c => c.id === id); }

function fillSelect(sel, registry, capId, withAuto) {
  const cap = (registry.capabilities || []).find(c => c.id === capId);
  sel.innerHTML = '';
  if (withAuto) {
    const o = el('option'); o.value = 'auto'; o.textContent = '自动选择（按可用密钥/依赖）';
    sel.appendChild(o);
  }
  (cap.providers || []).forEach(p => {
    const o = el('option');
    o.value = p.id;
    o.textContent = p.name + (p.offline ? ' · 离线' : '') + (p.needs_key ? ' · 需密钥' : '');
    if (p.recommended) o.textContent += ' ★';
    sel.appendChild(o);
  });
}

function renderApiModal() {
  const s = S.registry;
  const box = $('apiModalBody');
  box.innerHTML = '';
  const stat = el('div', 'statline', `
    <div><b>${s.capabilityTotal}</b>种 API 能力类别</div>
    <div><b>${s.capabilityRequired}</b>必选</div>
    <div><b>${s.capabilityOptional}</b>可选</div>
    <div><b>${s.providerTotal}</b>个厂商实现</div>
    <div><b>${s.cloudEndpoints}</b>个云端点</div>`);
  box.appendChild(stat);
  box.appendChild(el('p', 'hint',
    '「一种 API」= 一个能力类别。同一类别下换厂商（例如 ASR 从本地模型换成 Groq）'
    + '不算新增种类。核心链路 6 种必选，全部有离线或免费实现，因此不填任何密钥也能出成品。'));

  (s.capabilities || []).forEach((c, i) => {
    const card = el('div', 'cap');
    const kind = c.kind === 'required' ? '<span class="kind req">必选</span>'
      : '<span class="kind opt">可选</span>';
    card.appendChild(el('div', 'h',
      `<b>${i + 1}. ${c.name} · ${c.name_en}</b>${kind}
       <span class="flags" style="color:var(--muted);font-size:11.5px">${c.providerCount} 个实现</span>`));
    card.appendChild(el('p', '', c.purpose));
    (c.providers || []).forEach(p => {
      const row = el('div', 'prov');
      const flags = [];
      if (p.needs_key) flags.push('需密钥');
      if (p.offline) flags.push('离线');
      if (p.recommended) flags.push('推荐');
      row.appendChild(el('span', 'nm', (p.recommended ? '★ ' : '') + p.name));
      row.appendChild(el('span', 'flags', '引擎：' + p.engine + (flags.length ? ' · ' + flags.join('/') : '')));
      if (p.endpoint) row.appendChild(el('code', '', p.endpoint));
      card.appendChild(row);
    });
    box.appendChild(card);
  });
}

function showApiModal() { $('apiModal').classList.remove('hidden'); }

/* ============================ 配置 ============================ */
async function loadConfig() {
  const r = await (await fetch('/api/config')).json();
  S.cfg = r.config;
  const c = r.config;
  setVal('forceSide', (c.channel && c.channel.forceSide) || 'auto');
  setVal('originalMode', (c.mix && c.mix.originalMode) || 'keep');
  setVal('asrProvider', (c.asr && c.asr.provider) || 'auto');
  setVal('asrModel', (c.asr && c.asr.model) || 'small');
  setVal('trProvider', (c.translate && c.translate.provider) || 'auto');
  setVal('trStyle', (c.translate && c.translate.style) || 'adaptive');
  setVal('ttsProvider', (c.tts && c.tts.provider) || 'edge');
  setVal('ttsRefMode', (c.tts && c.tts.sovitsRefMode) || 'auto');
  setVal('ttsRate', (c.tts && c.tts.rate) || '+0%');
  setVal('ttsPitch', (c.tts && c.tts.pitch) || '+0Hz');
  setVal('maxTempo', (c.align && c.align.maxTempo) || 1.45);
  setVal('maxCps', (c.translate && c.translate.maxCharsPerSecond) || 7.5);
  $('burnVideo').checked = !!(c.subtitle && c.subtitle.burnVideo);
  renderKeys(c);
}

function renderKeys(c) {
  const grid = $('keyGrid');
  grid.innerHTML = '';
  const keys = c.keys || {};
  Object.keys(keys).forEach(k => {
    if (k.endsWith('_BASE_URL') || k.endsWith('_ENDPOINT') || k.endsWith('_REGION')) return;
    const row = el('div', 'keyrow');
    row.appendChild(el('label', '', k.replace('_API_KEY', '')));
    const inp = el('input');
    inp.type = 'password';
    inp.placeholder = '未设置';
    inp.value = keys[k] || '';
    inp.dataset.key = k;
    row.appendChild(inp);
    grid.appendChild(row);
  });
}

function setVal(id, v) { const n = $(id); if (n) n.value = v; }

function collectConfig() {
  return {
    asr: { provider: $('asrProvider').value, model: $('asrModel').value },
    translate: { provider: $('trProvider').value, style: $('trStyle').value,
                 maxCharsPerSecond: parseFloat($('maxCps').value) || 7.5 },
    tts: { provider: $('ttsProvider').value, voice: $('ttsVoice').value,
           rate: $('ttsRate').value, pitch: $('ttsPitch').value,
           sovitsRefMode: $('ttsRefMode').value },
    align: { maxTempo: parseFloat($('maxTempo').value) || 1.45 },
    channel: { forceSide: $('forceSide').value, perSegment: true },
    mix: { originalMode: $('originalMode').value },
    subtitle: { enabled: true, bilingual: true, burnVideo: $('burnVideo').checked },
  };
}

/* ============================ 可选依赖 ============================ */
async function loadDeps() {
  let d;
  try { d = await (await fetch('/api/deps')).json(); } catch (e) { return; }
  if (!d.ok) return;
  S.deps = d;

  const whisper = d.deps.faster_whisper || {};
  const cloudKeys = ['GROQ_API_KEY', 'DEEPSEEK_API_KEY', 'SILICONFLOW_API_KEY', 'OPENAI_API_KEY'];
  const hasCloud = cloudKeys.some(k => (S.cfg && S.cfg.keys && S.cfg.keys[k]) ||
    (S.cfg && S.cfg._keyStatus && S.cfg._keyStatus[k]));
  const n = $('asrNotice');

  if (whisper.installed) {
    n.className = 'notice ok';
    n.innerHTML = '<b>本地识别模型已就绪</b>：可离线识别日语，音频不会离开本机。'
      + (d.models && d.models.length
        ? '<br>已缓存模型：' + d.models.map(m => m.model + '(' + Math.round(m.bytes / 1048576) + 'MB)').join('、')
        : '');
  } else if (hasCloud) {
    n.className = 'notice ok';
    n.innerHTML = '<b>已检测到云端密钥</b>：翻译/识别会走云端 API。若想完全离线，也可安装本地识别模型。';
  } else {
    n.className = 'notice warn';
    n.innerHTML = '<b>还没有真实识别后端</b>——现在只能产出「示范占位文本」的占位成品。'
      + '三选一：<br>① 安装本地识别模型（离线免费，推荐）'
      + '<br>② 在下面填入云端密钥（Groq / 硅基流动 最快）'
      + '<br>③ 用「手工导入」贴上带时间戳的日文'
      + '<div class="act"><button id="btnInstallWhisper">安装本地识别模型</button></div>';
  }
  renderDepsPanel(d);

  const btn = $('btnInstallWhisper');
  if (btn) btn.onclick = () => installDep('faster_whisper');
}

function renderDepsPanel(d) {
  const box = $('depsPanel');
  box.innerHTML = '';
  Object.keys(d.deps).filter(k => !k.startsWith('_')).forEach(k => {
    const dep = d.deps[k];
    const card = el('div', 'dep');
    card.appendChild(el('div', 'top',
      `<span class="nm">${dep.title}</span>
       <span class="st ${dep.installed ? 'on' : 'off'}">${dep.installed ? '已安装' : '未安装'}</span>`));
    card.appendChild(el('div', 'ds', dep.desc + '<br>体积：' + dep.size));
    const btn = el('button', '', dep.installed ? '已安装' : '安装到工程内 vendor/');
    btn.disabled = !!dep.installed;
    btn.onclick = () => installDep(k);
    card.appendChild(btn);
    if (k === 'faster_whisper' && d.models && d.models.length) {
      card.appendChild(el('div', 'models', '已缓存模型：'
        + d.models.map(m => m.model + ' (' + Math.round(m.bytes / 1048576) + 'MB)').join('、')));
    }
    box.appendChild(card);
  });
  if (d.hfEndpoint) {
    box.appendChild(el('p', 'hint', '模型下载端点：' + d.hfEndpoint
      + '（国内镜像；HF 官方直连不通时用这个）'));
  }
}

async function installDep(key) {
  const r = await postJSON('/api/deps/install', { dep: key });
  if (!r.ok) { log('无法启动安装：' + r.error, 'e'); return; }
  log('开始安装 ' + key + '，下载可能需要几分钟，请勿关闭窗口…');
  const jid = r.jobId;
  let since = 0;
  const tick = async () => {
    const ev = await (await fetch(`/api/job/events?id=${jid}&since=${since}`)).json();
    since = ev.nextSince;
    (ev.events || []).forEach(e => {
      if (e.stage === 'log' && /pip|镜像|安装|Downloading|Collecting|Successfully/.test(e.message)) {
        log('  ' + e.message);
      } else if (e.message) log(e.message);
    });
    if (ev.finished) {
      const j = await (await fetch('/api/job?id=' + jid)).json();
      const ok = j.job.result && j.job.result.ok;
      log(ok ? '安装成功，可立即使用本地识别。' : '安装失败，可改用云端密钥或手工导入。',
        ok ? 'g' : 'e');
      await loadDeps();
      return;
    }
    setTimeout(tick, 1500);
  };
  tick();
}

/* ============================ 音色 ============================ */
async function loadVoices() {
  const sel = $('ttsVoice');
  sel.innerHTML = '<option>加载中…</option>';
  try {
    const r = await (await fetch('/api/tts/voices?zh=1')).json();
    if (!r.ok) throw new Error(r.error);
    sel.innerHTML = '';
    (r.voices || []).forEach(v => {
      const o = el('option');
      o.value = v.name;
      o.textContent = `${v.name.replace('zh-CN-', '').replace('Neural', '')} · ${v.gender === 'Female' ? '女声' : '男声'}`;
      if (v.name === 'zh-CN-XiaoxiaoNeural') o.selected = true;
      sel.appendChild(o);
    });
    if (!sel.options.length) {
      sel.innerHTML = '<option value="zh-CN-XiaoxiaoNeural">zh-CN-XiaoxiaoNeural</option>';
    }
  } catch (e) {
    sel.innerHTML = '<option value="zh-CN-XiaoxiaoNeural">zh-CN-XiaoxiaoNeural（音色列表拉取失败）</option>';
  }
}

/* ============================ 上传 ============================ */
function bindUI() {
  if ($('btnDiag')) $('btnDiag').onclick = () => loadDiag();
  if ($('btnSovitsStart')) $('btnSovitsStart').onclick = () => sovitsStart();
  if ($('btnSovitsStop')) $('btnSovitsStop').onclick = () => sovitsStop();
  if ($('btnSovitsTest')) $('btnSovitsTest').onclick = () => sovitsSelfTest();
  if ($('btnSovitsDLCancel')) $('btnSovitsDLCancel').onclick = () => sovitsDLCancel();
  const dz = $('dropzone'), fi = $('fileInput');
  $('pickLink').onclick = (e) => { e.preventDefault(); fi.click(); };
  dz.onclick = () => fi.click();
  fi.onchange = () => { if (fi.files[0]) upload(fi.files[0]); };
  ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => {
    e.preventDefault(); dz.classList.add('drag');
  }));
  ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => {
    e.preventDefault(); dz.classList.remove('drag');
  }));
  dz.addEventListener('drop', e => {
    const f = e.dataTransfer.files[0];
    if (f) upload(f);
  });

  $('btnAnalyze').onclick = analyze;
  $('btnRun').onclick = runPipeline;
  $('btnPreviewVoice').onclick = previewVoice;
  $('btnSaveKeys').onclick = saveKeys;
  $('modalClose').onclick = () => $('apiModal').classList.add('hidden');
  $('apiModal').onclick = (e) => { if (e.target.id === 'apiModal') $('apiModal').classList.add('hidden'); };
  $('ttsProvider').onchange = () => {
    const p = $('ttsProvider').value;
    $('ttsVoice').disabled = !(p === 'edge' || p === 'auto' || p === 'azure');
  };
  $('ttsVoice').disabled = false;
}

function upload(file) {
  S.file = null;
  $('btnRun').disabled = true;
  const fd = new FormData();
  fd.append('file', file);
  const xhr = new XMLHttpRequest();
  $('dzProgress').classList.add('on');
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = e.loaded / e.total * 100;
      $('dzProgress').firstElementChild.style.width = pct.toFixed(1) + '%';
    }
  };
  xhr.onload = () => {
    $('dzProgress').classList.remove('on');
    $('dzProgress').firstElementChild.style.width = '0';
    let r;
    try { r = JSON.parse(xhr.responseText); } catch (e) { r = { ok: false, error: xhr.responseText }; }
    if (!r.ok) { log('上传失败：' + r.error, 'e'); return; }
    S.file = r.file;
    showFile(r.file);
    log(`已注入：${r.file.filename}（${fmtSize(r.file.size)}，${fmtTime(r.file.media.duration)}，`
      + `${r.file.media.sampleRate}Hz ${r.file.media.channels}声道）`, 'g');
    $('btnRun').disabled = false;
    syncRunBtn();
    $('srcAudio').src = r.file.url;
    analyze();
  };
  xhr.onerror = () => {
    $('dzProgress').classList.remove('on');
    log('上传出错', 'e');
  };
  xhr.open('POST', '/api/upload');
  xhr.send(fd);
}

function showFile(f) {
  $('fileCard').classList.remove('hidden');
  $('fcName').textContent = f.filename;
  $('fcSize').textContent = fmtSize(f.size) + ' · ' + fmtTime(f.media.duration)
    + ' · ' + f.media.sampleRate + 'Hz · ' + f.media.channels + 'ch · ' + (f.media.codec || '');
  const chips = $('fcChips');
  chips.innerHTML = '';
  ['format', 'bitRate'].forEach(k => {
    if (f.media[k]) chips.appendChild(el('span', 'chip', k + ': ' + f.media[k]));
  });
}

/* ============================ 声道分析 ============================ */
async function analyze() {
  if (!S.file) return;
  const r = await postJSON('/api/analyze', { alias: S.file.alias, config: { channel: { forceSide: $('forceSide').value } } });
  if (!r.ok) { log('分析失败：' + r.error, 'e'); return; }
  S.jobId = r.jobId; S.since = 0;
  log('开始左右耳声源判定…');
  pollJob(r.jobId, (job) => {
    const ch = job.result && job.result.channel;
    if (ch) renderEars(ch, 'auto');
    if (job.result && job.result.regionCount) {
      log(`VAD 分段：${job.result.regionCount} 段`, 'g');
    }
  });
}

function renderEars(ch, mode) {
  const L = $('earL'), R = $('earR');
  L.className = 'ear'; R.className = 'ear';
  const sideName = { left: '左耳', right: '右耳', center: '居中/双耳' }[ch.side] || ch.side;
  const conf = Math.round((ch.confidence || 0) * 100);
  if (ch.side === 'left') L.classList.add('active-left');
  if (ch.side === 'right') R.classList.add('active-right');
  const dubSide = ch.opposite === 'left' ? L : R;
  dubSide.classList.add('active-dub');
  $('earLMark').textContent = ch.side === 'left' ? '原声' : (ch.opposite === 'left' ? '中文' : '—');
  $('earRMark').textContent = ch.side === 'right' ? '原声' : (ch.opposite === 'right' ? '中文' : '—');
  $('earLNote').innerHTML = ch.side === 'left'
    ? `日语原声<br>置信度 ${conf}%` : (ch.opposite === 'left' ? '中文配音<br>完全对侧' : '环境音/音效');
  $('earRNote').innerHTML = ch.side === 'right'
    ? `日语原声<br>置信度 ${conf}%` : (ch.opposite === 'right' ? '中文配音<br>完全对侧' : '环境音/音效');
  log(`判定结果：原声在${sideName}（置信度 ${conf}%，电平差 ${fmtNum(ch.median_delta_db)} dB，ITD ${fmtNum(ch.itd_ms)} ms）→ 中文配音放 ${ch.opposite === 'left' ? '左' : '右'}耳`, 'g');
  if (ch.verdict_note) log('  ' + ch.verdict_note, 'w');
}

/* ============================ 主流程 ============================ */
async function runPipeline() {
  if (!S.file) return;
  clearSteps(); clearLog();
  // 体检没就绪也让他跑（有 Demo/占位兜底），但先把话说明白，
  // 免得他等十几分钟拿到一个「（示范文本）」翻译出来的成品还以为成了。
  const blockers = ((S.diag && S.diag.blockers) || []);
  const warns = ((S.diag && S.diag.warnings) || []);
  if (blockers.length) {
    log('⚠ 环境还缺 ' + blockers.length + ' 项：' + blockers.join('、'), 'w');
    log('  照现在跑，缺失的部分会用占位内容顶上，成品质量会很差。', 'w');
    log('  建议先在左上「环境体检」里把它们修掉（就在同一页面，不用重启）。', 'w');
    const box = $('diagBox');
    if (box) box.open = true;
  }
  if (warns.length) log('提醒：' + warns.join('、'), 'w');
  const body = { alias: S.file.alias, config: collectConfig() };
  const r = await postJSON('/api/run', body);
  if (!r.ok) { log('启动失败：' + r.error, 'e'); return; }
  S.jobId = r.jobId; S.since = 0;
  $('btnRun').disabled = true;
  $('resultBody').classList.add('hidden');
  $('resultEmpty').classList.remove('hidden');
  log('任务已启动：' + r.jobId);
  pollJob(r.jobId, handleJobDone);
}

function handleJobDone(job) {
  $('btnRun').disabled = false;
  if (job.status !== 'done') return;
  const res = job.result || {};
  S.result = res;
  log('全部完成。输出目录：' + res.outputDir, 'g');
  renderResult(res);
  refreshOutputs();
}

function renderResult(res) {
  $('resultEmpty').classList.add('hidden');
  $('resultBody').classList.remove('hidden');
  const prev = res.previews || {};
  if (prev.originalSide) {
    $('abOrig').src = dl(prev.originalSide);
    $('abDub').src = dl(prev.dubSide);
    $('abOrigLabel').textContent = '原声耳（' + (prev.origSide === 'left' ? '左' : '右') + '）';
    $('abDubLabel').textContent = '中文耳（' + (prev.origSide === 'left' ? '右' : '左') + '）';
  }
  const outs = res.outputs || {};
  if (outs.mp3) $('abMix').src = dl(outs.mp3);

  const list = $('fileList');
  list.innerHTML = '';
  const groups = [
    ['成品', [outs.mp3, outs.m4a, outs.mp4].filter(Boolean)],
    ['试听', [prev.originalSide, prev.dubSide].filter(Boolean)],
    ['字幕', Object.values((res.stages && res.stages.subtitle && res.stages.subtitle.files) || {})],
  ];
  groups.forEach(([label, arr]) => {
    if (!arr.length) return;
    list.appendChild(el('div', 'hint', label));
    arr.forEach(p => list.appendChild(fileRow(p)));
  });
  renderSegments(res);
}

function dl(path) { return '/api/download?path=' + encodeURIComponent(path); }

function fileRow(path) {
  const name = path.split(/[\\/]/).pop();
  const ext = (name.split('.').pop() || '').toLowerCase();
  const row = el('div', 'file');
  row.appendChild(el('span', 'ext', ext));
  row.appendChild(el('span', 'nm', name));
  const a = el('a', '', '下载');
  a.href = dl(path);
  a.setAttribute('download', name);
  row.appendChild(a);
  return row;
}

function renderSegments(res) {
  const tb = $('segTable').querySelector('tbody');
  tb.innerHTML = '';
  const plans = {};
  (res.plans || []).forEach(p => plans[p.id] = p);
  (res.segments || []).forEach(s => {
    const p = plans[s.id] || {};
    const tr = el('tr');
    tr.appendChild(el('td', 't', String(s.id)));
    tr.appendChild(el('td', 't', (p.start !== undefined ? p.start.toFixed(2) : s.start.toFixed(2)) + 's'));
    tr.appendChild(el('td', 'ja', s.text || ''));
    tr.appendChild(el('td', 'zh', s.zh || '<i class="muted">（空）</i>'));
    const sideCls = s.side === 'left' ? 'side-left' : (s.side === 'right' ? 'side-right' : 'side-center');
    tr.appendChild(el('td', '',
      `<span class="sidepill ${sideCls}">原${s.side === 'left' ? '左' : s.side === 'right' ? '右' : '中'}→中${s.opposite === 'left' ? '左' : '右'}</span>`));
    tr.appendChild(el('td', 't' + (p.overrun ? ' over' : ''),
      p.tempo ? p.tempo.toFixed(2) + 'x' + (p.overrun ? '⚠' : '') : '—'));
    tb.appendChild(tr);
  });
}

/* ============================ 任务轮询 ============================ */
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  try { return await r.json(); } catch (e) { return { ok: false, error: '响应解析失败' }; }
}

function pollJob(jobId, onDone) {
  S.polling = true;
  let since = 0;
  const tick = async () => {
    if (!S.polling) return;
    try {
      const r = await (await fetch(`/api/job/events?id=${jobId}&since=${since}`)).json();
      if (!r.ok) { S.polling = false; return; }
      since = r.nextSince;
      (r.events || []).forEach(ev => handleEvent(ev));
      setProgress(r.progress, r.events && r.events.length ? r.events[r.events.length - 1].message : '');
      if (r.finished || r.status === 'done' || r.status === 'error') {
        S.polling = false;
        const j = await (await fetch('/api/job?id=' + jobId)).json();
        if (j.ok && onDone) onDone(j.job);
        setProgress(j.job.status === 'error' ? 0 : 100,
          j.job.status === 'error' ? ('失败：' + j.job.error) : '完成');
        markAllDone();
        return;
      }
    } catch (e) { /* 忽略抖动 */ }
    setTimeout(tick, 700);
  };
  tick();
}

function handleEvent(ev) {
  if (ev.stage === 'log') {
    const cls = /错误|失败|Error|异常/.test(ev.message) ? 'e'
      : /警告|未覆盖|失败/.test(ev.message) ? 'w'
        : /完成|就绪|成功/.test(ev.message) ? 'g' : '';
    log(ev.message, cls);
    return;
  }
  if (ev.stage === 'error') { log('错误：' + ev.message, 'e'); return; }
  if (STAGES.some(s => s[0] === ev.stage)) {
    markStep(ev.stage);
    if (ev.message) log('▸ ' + ev.message);
  }
  if (ev.stage === 'done' && ev.data) { /* 结果由 handleJobDone 处理 */ }
  if (ev.data && ev.data.side && ev.data.confidence !== undefined) renderEars(ev.data);
}

function markStep(stage) {
  const items = [...document.querySelectorAll('#stepList li')];
  const idx = STAGES.findIndex(s => s[0] === stage);
  items.forEach((li, i) => {
    if (i < idx) { li.className = 'ok'; li.querySelector('.dot').textContent = '✓'; }
    else if (i === idx) { li.className = 'run'; li.querySelector('.dot').textContent = '●'; }
    else li.className = '';
  });
}
function markAllDone() {
  document.querySelectorAll('#stepList li').forEach(li => {
    li.className = 'ok'; li.querySelector('.dot').textContent = '✓';
  });
}
function clearSteps() {
  document.querySelectorAll('#stepList li').forEach(li => {
    li.className = ''; li.querySelector('.dot').textContent = '';
  });
}
function setProgress(pct, text) {
  $('progBar').style.width = Math.max(0, Math.min(100, pct || 0)) + '%';
  if (text) $('progText').textContent = text.slice(0, 90);
}
function clearLog() { $('logBox').innerHTML = ''; }
function log(msg, cls) {
  const box = $('logBox');
  const line = el('div', cls || '', escapeHTML(String(msg)));
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
  while (box.childNodes.length > 1200) box.removeChild(box.firstChild);
}
function escapeHTML(s) {
  return s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ============================ 其它 ============================ */
async function previewVoice() {
  const btn = $('btnPreviewVoice');
  btn.disabled = true; btn.textContent = '合成中…';
  const r = await postJSON('/api/tts/preview', {
    text: '呐，听得见吗？中文配音会放在日语原声的另外一只耳朵里。',
    config: { tts: { provider: $('ttsProvider').value, voice: $('ttsVoice').value,
                     rate: $('ttsRate').value, pitch: $('ttsPitch').value } },
  });
  btn.disabled = false; btn.textContent = '试听这个音色';
  if (!r.ok) { log('试听失败：' + r.error, 'e'); return; }
  const a = $('previewAudio');
  a.classList.remove('hidden');
  a.src = r.url;
  a.play().catch(() => {});
}

async function saveKeys() {
  const keys = {};
  document.querySelectorAll('#keyGrid input[data-key]').forEach(i => {
    if (i.value && !i.value.includes('***')) keys[i.dataset.key] = i.value;
  });
  const r = await postJSON('/api/config', { config: { keys } });
  if (r.ok) { log('密钥已保存到 config.json', 'g'); } else log('保存失败', 'e');
}

async function refreshOutputs() {
  try {
    const r = await (await fetch('/api/outputs')).json();
    if (r.ok && r.outputs && r.outputs.length) {
      // 侧栏不展示，保留接口供扩展
    }
  } catch (e) { /* 忽略 */ }
}

function fmtSize(n) {
  if (!n && n !== 0) return '-';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return n.toFixed(i ? 1 : 0) + ' ' + u[i];
}
function fmtTime(s) {
  if (!s && s !== 0) return '-';
  const m = Math.floor(s / 60), ss = (s % 60).toFixed(1);
  return m + ':' + (ss < 10 ? '0' : '') + ss;
}
function fmtNum(v) { return (v === undefined || v === null) ? '-' : Number(v).toFixed(2); }

/* ============================ 环境体检 ============================ */
const DG_MARK = {
  ok:              ['✓', 'dg-i-ok'],
  running:         ['●', 'dg-i-ok'],
  partial:         ['!', 'dg-i-partial'],
  running_partial: ['◐', 'dg-i-partial'],
  missing:         ['×', 'dg-i-missing'],
  optional:        ['○', 'dg-i-optional'],
};

async function loadDiag() {
  if (!$('diagBox')) return;
  $('diagSummary').textContent = '环境检查中…';
  $('dgDot').className = 'dg-dot';
  let r;
  try {
    r = await (await fetch('/api/diag')).json();
  } catch (e) {
    $('diagSummary').textContent = '检查失败：' + e.message;
    $('dgDot').className = 'dg-dot missing';
    return;
  }
  if (!r.ok) {
    $('diagSummary').textContent = '检查失败：' + (r.error || '未知错误');
    $('dgDot').className = 'dg-dot missing';
    return;
  }
  S.diag = r.diag;
  renderDiag(r.diag);
}

// 开始按钮上直接把「还差几项」写出来。新用户不会先读体检面板，
// 但一定会看这个按钮——不写在这儿，他就直接跑出机器人音了。
function syncRunBtn() {
  const b = $('btnRun');
  if (!b) return;
  const blockers = ((S.diag && S.diag.blockers) || []);
  if (blockers.length) {
    b.textContent = '开始（还有 ' + blockers.length + ' 项未就绪）';
    b.classList.add('warn');
    b.title = '缺：' + blockers.join('、') + '（点上面「环境体检」里的按钮可以就地修）';
  } else {
    b.textContent = '开始注入 · 翻译 · 配音 · 摆位';
    b.classList.remove('warn');
    b.title = '';
  }
}

function renderDiag(d) {
  $('dgDot').className = 'dg-dot ' + (d.ready ? 'ok' : 'missing');
  $('diagSummary').textContent = d.summary || (d.ready ? '可以出成品' : '还差东西');
  $('diagSummary').title = '阻断项：' + ((d.blockers || []).join('、') || '无')
                        + '｜提醒：' + ((d.warnings || []).join('、') || '无');
  syncRunBtn();

  const ul = $('diagList');
  ul.innerHTML = '';
  (d.items || []).forEach((it) => {
    const li = el('li');
    const pair = DG_MARK[it.status] || ['?', 'dg-i-optional'];
    const ic = el('span', 'dg-ic ' + pair[1]);
    ic.textContent = pair[0];

    const body = el('div', 'dg-body');
    const t = el('div', 'dl-t');
    const nm = el('span', 'dl-name');
    nm.textContent = it.title + (it.required ? '' : ' · 可选');
    t.appendChild(nm);
    body.appendChild(t);

    const dd = el('div', 'dl-d');
    dd.textContent = it.detail || '';
    body.appendChild(dd);

    if (it.fix) {
      const fx = el('div', 'dl-fix');
      const b = el('button', 'btn ghost micro');
      b.textContent = it.fix.label || '处理';
      b.onclick = () => handleFix(it.fix);
      fx.appendChild(b);
      (it.fix.alts || []).forEach((alt) => {
        const b2 = el('button', 'btn ghost micro');
        b2.textContent = alt.label || '其他';
        b2.onclick = () => handleFix(alt);
        fx.appendChild(b2);
      });
      if (it.fix.hint) {
        const hint = el('span', 'dl-hint');
        hint.textContent = it.fix.hint;
        fx.appendChild(hint);
      }
      body.appendChild(fx);
    }

    li.appendChild(ic);
    li.appendChild(body);
    ul.appendChild(li);
  });
}

function diagItem(key) {
  return ((S.diag && S.diag.items) || []).find((i) => i.key === key) || {};
}

async function handleFix(fix) {
  const k = fix.kind;
  if (k === 'install_dep') {
    await installDep(fix.target);
    setTimeout(loadDiag, 4000);
    return;
  }
  if (k === 'set_key') {
    log('把 ' + (fix.target || 'API 密钥') + ' 填到左下「API 密钥」区，保存后点「重新检查」', 'w');
    const box = $('keysGrid');
    if (box) box.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return;
  }
  if (k === 'install_ffmpeg') {
    log('缺 ffmpeg。在新 PowerShell 窗口执行：winget install Gyan.FFmpeg —— 装完点「重新检查」', 'w');
    return;
  }
  if (k === 'start_sovits') {
    await sovitsStart();
    return;
  }
  if (k === 'repair_sovits') {
    const sv = diagItem('sovits');
    log('GPT-SoVITS 目录不完整，缺：' + ((sv.missing || []).join('、') || '未知'), 'e');
    log('这些文件来自官方整合包，重新解压或单独补齐后再点「重新检查」', 'w');
    return;
  }
  if (k === 'locate_sovits') {
    log('没找到 GPT-SoVITS。在 config.json 里填 tts.sovitsInstallDir（指向含 api_v2.py 的目录），或在环境变量 KOEDUB_SOVITS_DIR 指定', 'w');
    return;
  }
  if (k === 'download_sovits') {
    await sovitsDownload();
    return;
  }
  if (k === 'note') {
    log(fix.hint || fix.label || '', '');
    return;
  }
  log('这个检查项需要手动处理：' + (fix.label || k), 'w');
}

/* ---------- GPT-SoVITS 一键启动 / 停止 / 自检 ---------- */
function dgProbe(show, src, info) {
  const box = $('dgProbe');
  if (!box) return;
  if (!show) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  if (src) {
    const a = $('dgProbeAudio');
    a.src = src;
    a.play().catch(() => {});
  }
  if (info !== undefined && $('dgProbeInfo')) $('dgProbeInfo').textContent = info;
}

function busy(btn, text) {
  if (!btn) return () => {};
  const old = btn.textContent;
  btn.disabled = true;
  btn.textContent = text;
  return () => { btn.disabled = false; btn.textContent = old; };
}

async function sovitsStart() {
  const done = busy($('btnSovitsStart'), '启动中…');
  log('正在启动 GPT-SoVITS（要加载 4 个模型，约 15 秒）…');
  try {
    const r = await postJSON('/api/sovits/start', {});
    (r.logs || []).forEach((m) => log('  ' + m));
    if (r.ok) {
      log(r.already ? 'GPT-SoVITS 本来就在跑：' + (r.endpoint || '')
                    : '已就绪：' + r.seconds + ' 秒，' + r.endpoint, 'g');
    } else {
      log('启动失败：' + (r.error || '未知原因'), 'e');
      String(r.logTail || '').split('\n').slice(-8).forEach((l) => { if (l.trim()) log('  ' + l, 'e'); });
    }
  } catch (e) {
    log('启动请求失败：' + e.message, 'e');
  }
  done();
  await loadDiag();
}

async function sovitsStop() {
  const done = busy($('btnSovitsStop'), '停止中…');
  try {
    const r = await postJSON('/api/sovits/stop', {});
    if (r.ok) log(r.message || '已停止', 'g');
    else log('停止失败：' + (r.error || r.message || ''), 'e');
  } catch (e) {
    log('停止请求失败：' + e.message, 'e');
  }
  done();
  await loadDiag();
}

async function sovitsSelfTest() {
  const done = busy($('btnSovitsTest'), '试听合成中…');
  log('自检：用一句中文走完整条合成链路…');
  try {
    const r = await postJSON('/api/sovits/selftest', {});
    (r.logs || []).forEach((m) => log('  ' + m));
    if (r.ok && r.stage === 'done') {
      log('自检通过：' + r.seconds + ' 秒合成出 ' + r.duration + ' 秒音频（参考：' + (r.promptText || '无台词') + '）', 'g');
      dgProbe(true, '/api/media?path=' + encodeURIComponent(r.path),
        '自检音频 · ' + r.duration + ' 秒 · ' + r.text);
    } else if (r.ok && r.partial) {
      log('自检未完成合成：' + (r.message || ''), 'w');
      dgProbe(false);
    } else {
      log('自检失败：' + (r.error || '未知原因'), 'e');
      dgProbe(false);
    }
  } catch (e) {
    log('自检请求失败：' + e.message, 'e');
  }
  done();
  await loadDiag();
}

/* ---------- GPT-SoVITS 整合包下载 ---------- */
let _dlJob = '';
let _dlPct = -1;

function dlShow(on, pct, text) {
  const box = $('dgProg');
  if (!box) return;
  if (!on) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  if (pct >= 0 && $('dgBar')) $('dgBar').style.width = Math.max(0, Math.min(100, pct)) + '%';
  if (text !== undefined && $('dgProgText')) $('dgProgText').textContent = text;
}

async function sovitsDownload() {
  if (_dlJob) { log('已经在下载了，别急', 'w'); return; }
  const done = busy($('btnSovitsDL'), '准备中…');
  let r;
  try {
    r = await postJSON('/api/sovits/download', {});
  } catch (e) {
    log('下载请求失败：' + e.message, 'e');
    done();
    return;
  }
  if (!r.ok) { log('没法开始下载：' + (r.error || '未知原因'), 'e'); done(); return; }
  _dlJob = r.jobId;
  _dlPct = -1;
  dlShow(true, 0, '开始下载（约 8GB，可以取消，下次接着下）…');
  log('开始下载 GPT-SoVITS 整合包：' + r.source + '，' + (r.size ? (r.size / 1073741824).toFixed(1) + 'GB' : '大小未知'));
  done();

  let since = 0;
  const tick = async () => {
    if (!_dlJob) return;
    let ev;
    try {
      ev = await (await fetch(`/api/job/events?id=${_dlJob}&since=${since}`)).json();
    } catch (e) {
      setTimeout(tick, 2000);
      return;
    }
    since = ev.nextSince || since;
    (ev.events || []).forEach((e) => {
      if (e.stage === 'download') {
        const p = Math.round(e.progress || 0);
        dlShow(true, p, e.message || '');
        if (p >= _dlPct + 10) { _dlPct = p; log('  下载 ' + p + '%'); }
      } else if (e.message) {
        log('  ' + e.message, e.stage === 'error' ? 'e' : '');
        const m = String(e.message).match(/(\d+)%/);
        if (e.stage === 'download' && m) dlShow(true, Number(m[1]), e.message);
      }
    });
    if (ev.finished) {
      const j = await (await fetch('/api/job?id=' + _dlJob)).json();
      const res = (j.job && j.job.result) || {};
      if (res.ok && res.dir) {
        dlShow(true, 100, '装好了：' + res.dir);
        log('整合包已就位：' + res.dir, 'g');
        log('配置里的安装路径也自动填好了，接着点「启动 GPT-SoVITS」就行', 'g');
      } else if (res.partial) {
        dlShow(false);
        log('下载到测试上限就停了（这是测试模式）：' + (res.error || ''), 'w');
      } else if (res.canceled) {
        dlShow(false);
        log('已取消。已下载的部分留着，再点一次会从断点继续。', 'w');
      } else if (res.need === '7z') {
        dlShow(true, 100, '下载完了，但解压需要 7-Zip');
        log('下载完成，解压需要 7-Zip：在新 PowerShell 里跑 winget install 7zip.7zip，装完点「重新检查」不用重下', 'w');
      } else {
        dlShow(false);
        log('下载/解压失败：' + (res.error || (j.job && j.job.error) || '未知原因'), 'e');
      }
      _dlJob = '';
      await loadDiag();
      return;
    }
    setTimeout(tick, 1500);
  };
  tick();
}

async function sovitsDLCancel() {
  if (!_dlJob) { log('当前没有在下载', 'w'); return; }
  try {
    const r = await postJSON('/api/job/cancel', { id: _dlJob });
    if (r.ok) log('已发取消，正在停…', 'w');
  } catch (e) {
    log('取消失败：' + e.message, 'e');
  }
}

boot();
