'use strict';
const $ = id => document.getElementById(id), pretty = x => JSON.stringify(x, null, 2);
let state = {}, running = false, busy = false, selected = null, catalogItems = [], receipt = null, signalHistory = [];

async function api(path, data) {
  const r = await fetch('/api/' + path, data === undefined ? {} : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  const result = await r.json();
  if (!r.ok) throw Error(result.error || r.statusText);
  return result;
}

function error(e) {
  $('error').hidden = false;
  $('error').textContent = e.message || e;
  running = false;
  $('runstatus').textContent = 'Paused · request failed';
}

async function action(fn) {
  $('error').hidden = true;
  try { return await fn(); } catch(e) { error(e); }
}

function show(id, obj) {
  $(id).textContent = pretty(obj);
}

// Robust, DPI-aware canvas setup that fills solid dark background to prevent whiteout
function canvas(id) {
  const c = $(id);
  if (!c) return [null, 0, 0];
  const ratio = window.devicePixelRatio || 1;
  const parentWidth = c.parentElement ? c.parentElement.clientWidth : 0;
  const w = Math.max(c.clientWidth || parentWidth || 400, 100);
  const h = Number(c.getAttribute('height')) || 340;
  const targetW = Math.floor(w * ratio);
  const targetH = Math.floor(h * ratio);

  if (c.width !== targetW || c.height !== targetH) {
    c.width = targetW;
    c.height = targetH;
  }
  c.style.width = '100%';
  c.style.height = h + 'px';

  const x = c.getContext('2d');
  x.setTransform(ratio, 0, 0, ratio, 0, 0);
  // Guarantee sleek dark background so canvas never renders white
  x.fillStyle = '#08111a';
  x.fillRect(0, 0, w, h);
  return [x, w, h];
}

function graph() {
  try {
    const [x, w, h] = canvas('graph');
    if (!x) return;
    const g = state.graph || {}, nodes = g.nodes || [], positions = new Map();
    if (!nodes.length) {
      x.fillStyle = '#8b9daf';
      x.font = '11px monospace';
      x.fillText('Initializing connectome substrate…', 20, 35);
      return;
    }

    const neuropilColors = ['#5de0c1', '#edbc75', '#c084fc'];
    const neuropilLabels = ['flywire', 'manc', 'optic-lobe'];

    nodes.forEach((n, i) => {
      const origin = n.origin || 0;
      const angle = i * 2.39996;
      const rad = 25 + Math.sqrt(i / nodes.length) * Math.min(w * 0.32, h * 0.42);
      const cx = w / 2 + Math.cos(origin * 2.1) * w * 0.12;
      positions.set(n.id, [cx + Math.cos(angle) * rad, h / 2 + Math.sin(angle) * rad * 0.82]);
    });

    // Draw synaptic and graft edges
    x.lineWidth = 0.6;
    (g.edges || []).forEach(e => {
      const a = positions.get(e[0]), b = positions.get(e[1]);
      if (!a || !b) return;
      const weight = e[2] || 0;
      x.strokeStyle = weight < 0 ? 'rgba(236,124,141,0.22)' : 'rgba(93,224,193,0.18)';
      x.beginPath();
      x.moveTo(a[0], a[1]);
      x.lineTo(b[0], b[1]);
      x.stroke();
    });

    // Draw active neurons with firing rate glow
    nodes.forEach(n => {
      const p = positions.get(n.id);
      if (!p) return;
      const origin = (n.origin || 0) % neuropilColors.length;
      const v = Math.max(0, Math.min(1, Math.abs(n.activity || 0)));
      const baseColor = neuropilColors[origin];

      x.fillStyle = baseColor;
      x.globalAlpha = 0.3 + 0.7 * v;
      x.beginPath();
      x.arc(p[0], p[1], 2.0 + v * 4.5, 0, Math.PI * 2);
      x.fill();

      if (v > 0.6) {
        x.fillStyle = '#ffffff';
        x.globalAlpha = v * 0.6;
        x.beginPath();
        x.arc(p[0], p[1], 1.2, 0, Math.PI * 2);
        x.fill();
      }
    });

    x.globalAlpha = 1.0;
    x.fillStyle = '#8499ac';
    x.font = '10px monospace';
    x.fillText(`${g.neurons ?? nodes.length} neurons · ${g.graft_edges ?? 0} graft edges across 3 neuropils`, 16, 22);
  } catch (err) {
    console.error('graph render error:', err);
  }
}

function parseColor(c, defaultColor = '#5de0c1') {
  if (typeof c === 'string') return c;
  if (Array.isArray(c) && c.length >= 3) {
    const r = Math.round(Math.max(0, Math.min(1, c[0])) * 255);
    const g = Math.round(Math.max(0, Math.min(1, c[1])) * 255);
    const b = Math.round(Math.max(0, Math.min(1, c[2])) * 255);
    return `rgb(${r},${g},${b})`;
  }
  return defaultColor;
}

function world() {
  try {
    const [x, w, h] = canvas('world');
    if (!x) return;
    const s = state.world || {}, a = s.agent;

    const project = (px, py, z = 0) => [
      w * 0.5 + (px - py) * w * 0.042,
      h * 0.22 + (px + py) * h * 0.035 - z * 18
    ];

    // Floor grid
    x.strokeStyle = '#182b3a';
    x.lineWidth = 1;
    for (let i = -5; i <= 5; i++) {
      const p1 = project(i, -5), p2 = project(i, 5);
      x.beginPath(); x.moveTo(p1[0], p1[1]); x.lineTo(p2[0], p2[1]); x.stroke();
      const q1 = project(-5, i), q2 = project(5, i);
      x.beginPath(); x.moveTo(q1[0], q1[1]); x.lineTo(q2[0], q2[1]); x.stroke();
    }

    if (!a) {
      x.fillStyle = '#8b9daf';
      x.font = '11px monospace';
      x.fillText('Awaiting embodied world state…', 20, 35);
      return;
    }

    // Walls
    for (const wall of s.walls || []) {
      if (!Array.isArray(wall) || wall.length < 4) continue;
      x.strokeStyle = '#41596c';
      x.lineWidth = 3;
      const p1 = project(wall[0], wall[1]);
      const p2 = project(wall[2], wall[3]);
      x.beginPath(); x.moveTo(p1[0], p1[1]); x.lineTo(p2[0], p2[1]); x.stroke();
    }

    // World targets / food / objects
    (s.objects || []).filter(o => o.visible !== false).sort((o1, o2) => (o1.x + o1.y) - (o2.x + o2.y)).forEach(o => {
      const p = project(o.x, o.y, o.z || 0);
      x.fillStyle = parseColor(o.color, ({ food: '#edbc75', hazard: '#ec7c8d', target: '#7faaff' }[o.kind] || '#5de0c1'));
      x.beginPath();
      x.ellipse(p[0], p[1], Math.max(4, (o.r || 0.25) * 22), Math.max(3, (o.r || 0.25) * 13), 0, 0, Math.PI * 2);
      x.fill();
      x.fillStyle = '#8b9daf';
      x.font = '9px monospace';
      x.fillText(o.kind || 'obj', p[0] + 8, p[1] - 8);
    });

    // Ray sensors
    const ap = project(a.x, a.y);
    if (Array.isArray(s.rays)) {
      x.strokeStyle = 'rgba(93,224,193,0.12)';
      x.lineWidth = 1;
      s.rays.forEach(ray => {
        if (!Array.isArray(ray) || ray.length < 2) return;
        const rp = project(ray[0], ray[1]);
        x.beginPath(); x.moveTo(ap[0], ap[1]); x.lineTo(rp[0], rp[1]); x.stroke();
      });
    }

    // Agent body and heading pointer
    x.fillStyle = '#5de0c1';
    x.beginPath();
    x.arc(ap[0], ap[1], 7.5, 0, Math.PI * 2);
    x.fill();

    const headingAngle = Number(a.th || 0);
    const tip = project(a.x + Math.cos(headingAngle) * 0.8, a.y + Math.sin(headingAngle) * 0.8);
    x.strokeStyle = '#ffffff';
    x.lineWidth = 2.2;
    x.beginPath(); x.moveTo(ap[0], ap[1]); x.lineTo(tip[0], tip[1]); x.stroke();

    x.fillStyle = '#8b9daf';
    x.font = '10px monospace';
    x.fillText(`Episode ${s.episode ?? 0} · t=${s.t ?? 0} · return=${Number(s.ep_return ?? 0).toFixed(3)}`, 16, 22);
    $('worldtask').textContent = s.task || '';
  } catch (err) {
    console.error('world render error:', err);
  }
}

function signals() {
  try {
    const [x, w, h] = canvas('signals');
    if (!x) return;

    // Grid lines
    x.strokeStyle = '#182b3a';
    x.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      x.beginPath(); x.moveTo(0, h * i / 4); x.lineTo(w, h * i / 4); x.stroke();
    }

    const keys = ['graph_mean', 'graph_delta', 'curiosity'];
    const colors = ['#5de0c1', '#edbc75', '#80aaff'];

    keys.forEach((key, j) => {
      const values = signalHistory.map(t => Number(t[key] ?? 0));
      const max = Math.max(0.001, ...values.map(Math.abs));
      x.strokeStyle = colors[j];
      x.lineWidth = 1.5;
      x.beginPath();
      values.forEach((v, i) => {
        const px = i / Math.max(1, values.length - 1) * w;
        const py = h - 25 - (v / max) * (h - 45);
        if (i === 0) x.moveTo(px, py);
        else x.lineTo(px, py);
      });
      x.stroke();

      x.fillStyle = colors[j];
      x.font = '9px monospace';
      x.fillText(`${key}: ${(values.at(-1) || 0).toFixed(4)}`, 12 + j * 140, h - 8);
    });
  } catch (err) {
    console.error('signals render error:', err);
  }
}

function gauges() {
  try {
    const [x, w, h] = canvas('gauges');
    if (!x) return;

    const telem = state.telemetry || {};
    const nm = telem.neuromodulators || {};
    const astro = telem.astrocytes || {};

    const da = Number(nm.dopamine ?? 0);
    const sht = Number(nm.serotonin ?? 0.2);
    const oa = Number(nm.octopamine ?? 0.2);
    const ach = Number(nm.acetylcholine ?? 0.5);
    const ca = Number(astro.mean_calcium ?? 0.1);
    const bcm = Number(astro.mean_bcm_threshold ?? 0.5);

    const items = [
      { label: 'Dopamine (DA)', val: da, max: 2.0, col: '#5de0c1', role: 'Plasticity / RPE' },
      { label: 'Serotonin (5-HT)', val: sht, max: 1.0, col: '#80aaff', role: 'Patience / Threat' },
      { label: 'Octopamine (OA)', val: oa, max: 1.0, col: '#edbc75', role: 'Vigor / Arousal' },
      { label: 'Acetylcholine (ACh)', val: ach, max: 1.0, col: '#c084fc', role: 'Attention / Precision' },
      { label: 'Astrocyte [Ca2+]', val: ca, max: 2.0, col: '#34d399', role: 'Glial Spillover' },
      { label: 'BCM Threshold', val: bcm, max: 1.5, col: '#fbbf24', role: 'Metaplasticity' },
    ];

    const barW = (w - 40) / 3;
    const barH = 14;

    items.forEach((item, idx) => {
      const col = idx % 3;
      const row = Math.floor(idx / 3);
      const bx = 16 + col * (barW + 10);
      const by = 28 + row * 65;

      // Label
      x.fillStyle = '#a9c4d9';
      x.font = '10px monospace';
      x.fillText(item.label, bx, by);

      // Value
      x.fillStyle = item.col;
      x.font = '10px monospace';
      x.fillText(item.val.toFixed(3), bx + barW - 35, by);

      // Track
      x.fillStyle = '#142535';
      x.fillRect(bx, by + 6, barW, barH);

      // Level
      const pct = Math.max(0, Math.min(1, item.val / item.max));
      x.fillStyle = item.col;
      x.fillRect(bx, by + 6, barW * pct, barH);

      // Role subtitle
      x.fillStyle = '#647c90';
      x.font = '8px monospace';
      x.fillText(item.role, bx, by + barH + 18);
    });
  } catch (err) {
    console.error('gauges render error:', err);
  }
}

function predictiveView() {
  try {
    const [x, w, h] = canvas('predictive');
    if (!x) return;

    const telem = state.telemetry || {};
    const pc = telem.predictive_coding || {};

    const fe = Number(pc.free_energy ?? 0);
    const surprisal = Number(pc.sensory_surprisal ?? 0);
    const precision = Number(pc.mean_precision ?? 1.0);
    const epistemic = Number(pc.epistemic_value ?? 0);

    // Summary tiles
    const tiles = [
      { label: 'Variational Free Energy', val: fe.toFixed(4), col: '#5de0c1' },
      { label: 'Sensory Surprisal', val: surprisal.toFixed(4), col: '#edbc75' },
      { label: 'Cholinergic Precision', val: precision.toFixed(3), col: '#c084fc' },
      { label: 'Epistemic Value', val: epistemic.toFixed(4), col: '#80aaff' },
    ];

    const tileW = (w - 30) / 2;
    const tileH = 50;

    tiles.forEach((t, i) => {
      const col = i % 2;
      const row = Math.floor(i / 2);
      const tx = 10 + col * (tileW + 10);
      const ty = 20 + row * (tileH + 15);

      x.fillStyle = '#101e2c';
      x.strokeStyle = '#20364c';
      x.lineWidth = 1;
      x.beginPath();
      x.roundRect(tx, ty, tileW, tileH, 5);
      x.fill();
      x.stroke();

      x.fillStyle = '#8b9daf';
      x.font = '9px monospace';
      x.fillText(t.label, tx + 10, ty + 18);

      x.fillStyle = t.col;
      x.font = '16px monospace';
      x.fillText(t.val, tx + 10, ty + 40);
    });

    x.fillStyle = '#647c90';
    x.font = '9px monospace';
    x.fillText('Karl Friston Free Energy Principle (FEP) Active Inference reflex arc', 12, h - 8);
  } catch (err) {
    console.error('predictive view error:', err);
  }
}

function flatten(obj, prefix = '', out = []) {
  Object.entries(obj || {}).forEach(([k, v]) => {
    const name = prefix + k;
    if (v && typeof v === 'object' && !Array.isArray(v)) flatten(v, name + '.', out);
    else out.push([name, Array.isArray(v) ? v.map(n => typeof n === 'number' ? n.toFixed(3) : String(n)).join(' · ') : typeof v === 'number' ? v.toFixed(5) : String(v)]);
  });
  return out;
}

function render(s) {
  state = s;
  show('snapshot', s);
  const m = [
    ['Simulation steps', s.steps ?? 0],
    ['Episode return', Number(s.world?.ep_return ?? 0).toFixed(3)],
    ['Neurons', s.graph?.neurons ?? '—'],
    ['Bridge', s.telemetry?.bridge_training ?? s.status?.bridge_training ?? '—']
  ];
  $('metrics').replaceChildren(...m.map(([label, value]) => {
    const d = document.createElement('div');
    d.className = 'metric';
    const small = document.createElement('small'), strong = document.createElement('strong');
    small.textContent = label;
    strong.textContent = value;
    d.append(small, strong);
    return d;
  }));

  $('telemetry').replaceChildren(...flatten(s.telemetry).map(([k, v]) => {
    const d = document.createElement('div');
    d.className = 'kv';
    const name = document.createElement('span'), value = document.createElement('b');
    name.textContent = k;
    value.textContent = v;
    d.append(name, value);
    return d;
  }));

  if (s.telemetry && signalHistory.at(-1)?.ticks !== s.telemetry.ticks) signalHistory.push(s.telemetry);
  signalHistory = signalHistory.slice(-100);

  graph();
  world();
  signals();
  gauges();
  predictiveView();

  $('connection').textContent = '● CONNECTED / LOOPBACK';
}

async function tick(steps = 1, learn = $('learn').checked) {
  if (busy) return;
  busy = true;
  try {
    render(await api('step', { steps, learn }));
  } finally {
    busy = false;
  }
}

document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tab,nav button').forEach(n => n.classList.remove('active'));
  $(b.dataset.tab).classList.add('active');
  b.classList.add('active');
  $('title').textContent = b.querySelector('span').textContent;
  if (b.dataset.tab === 'overview') render(state);
});

$('run').onclick = () => { running = true; $('runstatus').textContent = 'Running'; };
$('pause').onclick = () => { running = false; $('runstatus').textContent = 'Paused'; };
$('step').onclick = () => action(() => tick());
$('train').onclick = () => action(() => tick(32, true));

$('config').onsubmit = e => {
  e.preventDefault();
  running = false;
  const config = Object.fromEntries(new FormData(e.target));
  for (const key of ['seed', 'nodes_per_source', 'microsteps', 'cognition_interval']) config[key] = Number(config[key]);
  action(async () => {
    render(await api('reset', { config }));
    signalHistory = [];
  });
};

function items(c) {
  if (Array.isArray(c)) return c;
  if (Array.isArray(c.capabilities)) return c.capabilities;
  if (c.capabilities && typeof c.capabilities === 'object') return Object.entries(c.capabilities).map(([id, v]) => ({ id, ...v }));
  return Object.entries(c).filter(([, v]) => v && typeof v === 'object' && !Array.isArray(v)).map(([id, v]) => ({ id, ...v }));
}

function renderCatalog() {
  const q = $('search').value.toLowerCase();
  $('catalog').replaceChildren(...catalogItems.filter(c => pretty(c).toLowerCase().includes(q)).map(c => {
    const b = document.createElement('button');
    b.className = 'cap-item';
    b.textContent = c.id || c.name || c.capability_id;
    const s = document.createElement('small');
    s.textContent = c.status || c.availability || c.family || '';
    b.append(s);
    b.onclick = () => {
      selected = c;
      $('capname').textContent = c.id || c.name;
      show('capschema', c);
      $('cappayload').value = pretty(c.example || c.example_payload || {});
    };
    return b;
  }));
}

$('search').oninput = renderCatalog;
$('invoke').onclick = () => action(async () => {
  if (!selected) throw Error('Select a capability first');
  show('capresult', await api('invoke', { id: selected.id || selected.capability_id, payload: JSON.parse($('cappayload').value) }));
});

$('tensors').onclick = () => action(async () => show('surfaces', await api('tensors')));
$('makepatch').onclick = () => action(async () => {
  $('packet').value = pretty(await api('patch/make', { edits: JSON.parse($('edits').value), description: $('description').value }));
});
$('previewpatch').onclick = () => action(async () => show('patchresult', await api('patch/preview', { packet: JSON.parse($('packet').value) })));
$('applypatch').onclick = () => action(async () => {
  const r = await api('patch/apply', { packet: JSON.parse($('packet').value) });
  receipt = r.receipt_id;
  show('patchresult', r);
  render(await api('state'));
});
$('rollback').onclick = () => action(async () => {
  if (!receipt) throw Error('No applied patch in this browser session');
  show('patchresult', await api('patch/rollback', { receipt_id: receipt }));
  receipt = null;
  render(await api('state'));
});
$('importpatch').onchange = () => action(async () => {
  const f = $('importpatch').files[0];
  if (f.size > 1048576) throw Error('Patch file exceeds 1 MiB');
  $('packet').value = pretty(JSON.parse(await f.text()));
});
$('exportpatch').onclick = () => action(() => {
  const blob = new Blob([pretty(JSON.parse($('packet').value))], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'synapse-patch.json';
  a.click();
  URL.revokeObjectURL(url);
});

$('autoprogram').onclick = () => action(async () => {
  show('autorepairresult', { status: 'running diagnosis & repair...' });
  const res = await api('autoprogram', { diagnostic_episodes: 2, eval_episodes: 2 });
  show('autorepairresult', res);
  render(await api('state'));
});

$('transplant').onclick = () => action(async () => {
  show('transplantresult', { status: 'running factorial transplantation assay...' });
  const res = await api('transplant', { episodes: 2 });
  show('transplantresult', res);
  render(await api('state'));
});

$('metamorphic').onclick = () => action(async () => {
  show('metamorphicresult', { status: 'running metamorphic invariance oracle...' });
  const res = await api('metamorphic', {});
  show('metamorphicresult', res);
});

$('spectral').onclick = () => action(async () => {
  show('spectralresult', { status: 'computing connectome harmonics...' });
  const res = await api('spectral', {});
  show('spectralresult', res);
});

$('manifold').onclick = () => action(async () => {
  show('manifoldresult', { status: 'computing Gromov-Wasserstein manifold alignment...' });
  const res = await api('invoke', { id: 'manifold.gromov_wasserstein', payload: {} });
  show('manifoldresult', res);
});

$('astrocyte').onclick = () => action(async () => {
  show('astrocyteresult', { status: 'auditing tripartite astrocyte metaplasticity...' });
  const res = await api('invoke', { id: 'astrocyte.gliotransmission', payload: {} });
  show('astrocyteresult', res);
});

async function jobs() {
  const result = await api('jobs');
  $('jobs').replaceChildren();
  if (!result.jobs.length) {
    $('jobs').textContent = 'No studies recorded.';
    return;
  }
  result.jobs.slice().reverse().forEach(j => {
    const d = document.createElement('div');
    d.className = 'job';
    const b = document.createElement('b');
    b.textContent = j.status.toUpperCase();
    d.append(b, document.createTextNode(' · ' + j.id.slice(0, 12)));
    if (['running', 'queued'].includes(j.status)) {
      const cancel = document.createElement('button');
      cancel.textContent = j.cancel_requested ? 'Cancellation requested' : 'Cancel';
      cancel.onclick = () => action(async () => {
        await api('jobs/cancel', { id: j.id });
        await jobs();
      });
      d.append(cancel);
    }
    const p = document.createElement('pre');
    p.textContent = pretty(j.error || j.progress);
    d.append(p);
    d.onclick = e => {
      if (e.target.tagName === 'BUTTON') return;
      show('studyresult', j.result || j);
      lineage(j.result?.lineage || []);
    };
    $('jobs').append(d);
  });
}

function lineage(rows) {
  $('lineage').replaceChildren();
  if (!rows.length) {
    $('lineage').textContent = 'No lineage receipts in this job.';
    return;
  }
  rows.forEach(row => {
    const d = document.createElement('div');
    d.className = 'lineage-node';
    d.textContent = `${row.arm} · L${row.lineage} → G${row.generation}`;
    const s = document.createElement('span');
    s.textContent = `${row.generator?.slice(0, 10) || 'root'} → ${row.policy?.identity?.slice(0, 10) || row.policy?.id || 'policy'}`;
    d.append(s);
    $('lineage').append(d);
  });
}

$('startstudy').onclick = () => action(async () => {
  await api('studies', { config: JSON.parse($('studyconfig').value) });
  await jobs();
});

async function checkpoints() {
  const r = await api('checkpoints');
  $('checkpointlist').replaceChildren(...r.checkpoints.map(name => {
    const b = document.createElement('button');
    b.textContent = name;
    b.onclick = () => $('checkpointname').value = name;
    return b;
  }));
}

for (const key of ['save', 'load']) {
  $(key).onclick = () => action(async () => {
    running = false;
    show('checkpointresult', await api(key, { name: $('checkpointname').value }));
    await checkpoints();
    render(await api('state'));
  });
}
$('refreshcheckpoints').onclick = () => action(checkpoints);

window.addEventListener('resize', () => {
  if ($('overview').classList.contains('active')) render(state);
});

action(async () => {
  render(await api('state'));
  catalogItems = items(await api('capabilities'));
  renderCatalog();
  await checkpoints();
  await jobs();
});

setInterval(() => { if (running) action(() => tick(4)); }, 250);
setInterval(() => action(jobs), 3000);
