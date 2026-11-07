'use strict';

const $ = id => document.getElementById(id);
const pretty = x => JSON.stringify(x, null, 2);

let state = {};
let running = false;
let busy = false;
let selected = null;
let catalogItems = [];
let receipt = null;
let signalHistory = [];
let tickInterval = 125;
let tickTimer = null;

// View state for zero-warping renderers
let worldMode = 'iso';      // 'iso' | 'top'
let graphMode = '2d';       // '2d' | '3d'
let graphAutoRotate = false;
let graphRotX = 0.35;
let graphRotY = 0.55;
let graphDragging = false;
let graphLastMouse = { x: 0, y: 0 };
let hoveredNode = null;
let graphSource = 'all';
let graphRole = 'all';
let graphZoom = 1;
let selectedNodeId = null;
let graphDidDrag = false;
let worldTrail = [];
let trailIdentity = '';
let lastTrailStep = null;
let trainingStatus = {};
let trackedJobId = null;
let latestJobs = [];
let trainingRequest = false;
let omnibusLastPoll = 0;
const sourcePalette = ['#5de0c1', '#edbc75', '#c084fc', '#80aaff', '#ec7c8d', '#85d38e', '#f2a6cf', '#62cee3'];
const sourceIds = ['malecns', 'banc', 'optic-lobe', 'fanc', 'flywire', 'manc', 'larval', 'hemibrain'];
let manualGripState = 0.0;
let manualReachState = 0.0;

// API transport
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
  $('error').textContent = e.message || String(e);
  running = false;
  $('runstatus').textContent = 'Paused · request failed';
}

async function action(fn) {
  $('error').hidden = true;
  try {
    return await fn();
  } catch (e) {
    error(e);
  }
}

function show(id, obj) {
  const el = $(id);
  if (el) el.textContent = pretty(obj);
}

// -------------------------------------------------------------------------
// Zero-Warping, DPI-Aware, Responsive Canvas Initialization
// -------------------------------------------------------------------------
function canvas(id) {
  const c = $(id);
  if (!c) return [null, 0, 0];
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  const rect = c.getBoundingClientRect();

  // Use actual CSS layout geometry to completely eliminate buffer/style mismatch warping
  let w = rect.width;
  let h = rect.height;

  if (!w || w < 20) {
    w = c.parentElement ? c.parentElement.clientWidth : 400;
  }
  if (!h || h < 20) {
    h = Number(c.getAttribute('height')) || (id === 'graph' || id === 'world' ? 350 : 185);
  }
  w = Math.max(w, 80);
  h = Math.max(h, 60);

  const targetW = Math.round(w * ratio);
  const targetH = Math.round(h * ratio);

  if (c.width !== targetW || c.height !== targetH) {
    c.width = targetW;
    c.height = targetH;
  }

  const x = c.getContext('2d');
  if (x.resetTransform) {
    x.resetTransform();
  } else {
    x.setTransform(1, 0, 0, 1, 0, 0);
  }
  x.scale(ratio, ratio);

  // Guarantee solid dark canvas background
  x.fillStyle = '#08111a';
  x.fillRect(0, 0, w, h);
  return [x, w, h];
}

// -------------------------------------------------------------------------
// Integrated Neural Substrate (2D Phyllotaxis & 3D Anatomical Observatory)
// -------------------------------------------------------------------------
const nodeScreenPositions = new Map();

function graphSources() {
  return (state.config?.connectome_sources || []).map((id, origin) => {
    const evidence = state.graph?.sources?.[origin] || {};
    const paletteIndex = sourceIds.indexOf(id);
    return { id, origin, evidence, color: sourcePalette[paletteIndex < 0 ? origin % sourcePalette.length : paletteIndex],
      colorClass: `source-color-${paletteIndex < 0 ? origin % sourcePalette.length : paletteIndex}` };
  });
}

function syncGraphControls() {
  const sources = graphSources();
  const sourceKey = sources.map(source => source.id).join('|');
  if ($('graph-source').dataset.sources !== sourceKey) {
    $('graph-source').replaceChildren(new Option('All active sources', 'all'), ...sources.map(source => new Option(source.id, String(source.origin))));
    if (!sources.some(source => String(source.origin) === graphSource)) graphSource = 'all';
    $('graph-source').value = graphSource;
    $('graph-source').dataset.sources = sourceKey;
    const resetBtn = document.createElement('button');
    resetBtn.id = 'filter-reset';
    resetBtn.textContent = '↺ All';
    resetBtn.title = 'Reset source and role filters to show all active neuropils';
    resetBtn.onclick = () => {
      graphSource = 'all';
      graphRole = 'all';
      $('graph-source').value = 'all';
      $('graph-role').value = 'all';
      graph();
    };
    $('graph-legend').replaceChildren(...sources.map(source => {
      const button = document.createElement('button');
      button.className = source.colorClass;
      button.textContent = `● ${source.id}`;
      button.title = `${source.id} · ${source.evidence.representation || 'provenance unavailable'}`;
      button.onclick = () => { graphSource = graphSource === String(source.origin) ? 'all' : String(source.origin); $('graph-source').value = graphSource; graph(); };
      return button;
    }), resetBtn);
  }
  const roles = [...new Set((state.graph?.nodes || []).map(node => node.role || 'central'))].sort();
  const roleKey = roles.join('|');
  if ($('graph-role').dataset.roles !== roleKey) {
    $('graph-role').replaceChildren(new Option('All roles', 'all'), ...roles.map(role => new Option(role, role)));
    if (!roles.includes(graphRole)) graphRole = 'all';
    $('graph-role').value = graphRole;
    $('graph-role').dataset.roles = roleKey;
  }
}

function graph() {
  try {
    const [x, w, h] = canvas('graph');
    if (!x) return;
    const g = state.graph || {};
    const sources = graphSources();
    const nodes = g.nodes || [];
    nodeScreenPositions.clear();
    syncGraphControls();
    if (!nodes.length) {
      x.fillStyle = '#8b9daf'; x.font = '11px Consolas, monospace';
      x.fillText('Initializing connectome substrate…', 20, 35); return;
    }
    const sourceCount = Math.max(1, sources.length);
    const nodeBudget = Number($('graph-nodes').value);
    const edgeBudget = Number($('graph-edges').value);
    const activityScale = Number($('graph-activity').value);
    const matching = nodes.filter(node => (graphSource === 'all' || String(node.origin) === graphSource) && (graphRole === 'all' || node.role === graphRole));
    // Sample across the entire filtered population without changing cluster positions.
    const stride = matching.length / Math.min(nodeBudget, Math.max(1, matching.length));
    const visible = matching.filter((node, index) => Math.floor(index / stride) !== Math.floor((index - 1) / stride));
    if (selectedNodeId !== null && !visible.some(node => node.id === selectedNodeId)) {
      const pinned = matching.find(node => node.id === selectedNodeId);
      if (pinned) { if (visible.length >= nodeBudget) visible.pop(); visible.push(pinned); }
    }
    const roleOffsets = { sensory: -.25, visual: -.17, ascending: -.08, central: 0, descending: .14, motor: .25 };
    const cosY = Math.cos(graphRotY), sinY = Math.sin(graphRotY), cosX = Math.cos(graphRotX), sinX = Math.sin(graphRotX);
    const scale = Math.min(w * .43, h * .41) * graphZoom;

    // Group indices per origin for 2D Golden-Ratio Phyllotaxis spiral layout
    const originCounts = new Map();
    const originTotals = new Map();
    visible.forEach(n => {
      const o = Number(n.origin || 0);
      originTotals.set(o, (originTotals.get(o) || 0) + 1);
    });

    // Draw 2D Neuropil Cluster Boundary Halos & Source Titles
    if (graphMode === '2d') {
      sources.forEach(source => {
        if (graphSource !== 'all' && String(source.origin) !== graphSource) return;
        const cAngle = sourceCount === 1 ? 0 : source.origin / sourceCount * Math.PI * 2 - Math.PI / 2;
        const cDist = sourceCount === 1 ? 0 : .65;
        const cx = w / 2 + Math.cos(cAngle) * cDist * scale;
        const cy = h / 2 + Math.sin(cAngle) * cDist * scale;
        const haloR = (sourceCount === 1 ? .52 : .27) * scale;
        x.strokeStyle = source.color;
        x.globalAlpha = 0.24;
        x.lineWidth = 1.2;
        x.setLineDash([4, 4]);
        x.beginPath(); x.arc(cx, cy, haloR, 0, Math.PI * 2); x.stroke();
        x.setLineDash([]);
        x.globalAlpha = 0.85;
        x.fillStyle = source.color;
        x.font = '10px Consolas, monospace';
        x.fillText(`${source.id.toUpperCase()} (${originTotals.get(source.origin) || 0})`, cx - 32, cy - haloR - 6);
        x.globalAlpha = 1.0;
      });
    }

    // Optional MaleCNS 128-node Programmed Prior & Neural Graft Outer Cortical Ring
    const graftSummary = state.training?.graft || {};
    if (graftSummary.enabled) {
      const ringR = Math.min(w, h) * 0.46 * graphZoom;
      x.strokeStyle = 'rgba(93, 224, 193, 0.16)';
      x.lineWidth = 1.0;
      x.setLineDash([2, 3]);
      x.beginPath(); x.arc(w / 2, h / 2, ringR, 0, Math.PI * 2); x.stroke();
      x.setLineDash([]);
      const priorNodes = Math.min(64, Number(graftSummary.prior_nodes || 64));
      for (let i = 0; i < priorNodes; i++) {
        const a = (i / priorNodes) * Math.PI * 2 + (Number(state.steps || 0) * 0.015);
        const gx = w / 2 + Math.cos(a) * ringR;
        const gy = h / 2 + Math.sin(a) * ringR;
        x.fillStyle = i % 2 === 0 ? 'rgba(93,224,193,0.45)' : 'rgba(192,132,252,0.45)';
        x.beginPath(); x.arc(gx, gy, 1.5, 0, Math.PI * 2); x.fill();
      }
    }

    const projected = visible.map(node => {
      const origin = Number(node.origin || 0);
      const angle = sourceCount === 1 ? 0 : origin / sourceCount * Math.PI * 2 - Math.PI / 2;
      const cluster = sourceCount === 1 ? 0 : .65;
      let px, py, pz;
      if (graphMode === '2d') {
        // Golden-Ratio Phyllotaxis Spiral per neuropil cluster
        const idx = originCounts.get(origin) || 0;
        originCounts.set(origin, idx + 1);
        const totalInCluster = Math.max(1, originTotals.get(origin) || 1);
        const goldenAngle = idx * 2.399963229728653;
        const maxR = sourceCount === 1 ? .48 : .24;
        const radius = 0.02 + Math.sqrt((idx + 0.5) / totalInCluster) * maxR;
        px = Math.cos(angle) * cluster + Math.cos(goldenAngle) * radius;
        py = Math.sin(angle) * cluster + Math.sin(goldenAngle) * radius;
        pz = 0;
      } else {
        const hash = (Math.imul(node.id + 1, 1664525) + 1013904223) >>> 0;
        const theta = hash / 4294967296 * Math.PI * 2;
        const radius = .05 + ((hash >>> 8) % 997) / 997 * .19;
        px = Math.cos(angle) * cluster + Math.cos(theta) * radius;
        py = Math.sin(angle) * cluster + Math.sin(theta) * radius;
        pz = (roleOffsets[node.role] || 0) + (((hash >>> 16) % 101) / 101 - .5) * .10;
      }
      const rx = graphMode === '3d' ? px * cosY - pz * sinY : px;
      const rz = graphMode === '3d' ? px * sinY + pz * cosY : 0;
      const ry = graphMode === '3d' ? py * cosX - rz * sinX : py;
      const depth = graphMode === '3d' ? py * sinX + rz * cosX : 0;
      const perspective = 3 / (3 + depth);
      const point = { x: w / 2 + rx * scale * perspective, y: h / 2 + ry * scale * perspective, z: depth, node };
      nodeScreenPositions.set(node.id, point); return point;
    }).sort((a, b) => b.z - a.z);
    let edgesDrawn = 0, graftsDrawn = 0;
    const pulsePhase = ((Number(state.steps || 0) % 20) / 20);
    const drawEdge = (edge, graft) => {
      if (edgesDrawn >= edgeBudget) return;
      const from = nodeScreenPositions.get(Array.isArray(edge) ? edge[0] : edge.source);
      const to = nodeScreenPositions.get(Array.isArray(edge) ? edge[1] : edge.target);
      if (!from || !to) return;
      const weight = Number(Array.isArray(edge) ? edge[2] : edge.weight) || 0;
      const active = [hoveredNode?.id, selectedNodeId].some(id => id !== null && id !== undefined && (id === from.node.id || id === to.node.id));
      x.lineWidth = active ? 1.6 : graft ? .95 : .5;
      x.strokeStyle = active ? '#ffffff' : graft ? 'rgba(237,188,117,.52)' : weight < 0 ? 'rgba(236,124,141,.14)' : 'rgba(128,170,255,.13)';
      x.setLineDash(graft ? [3, 2] : []);
      x.beginPath(); x.moveTo(from.x, from.y); x.lineTo(to.x, to.y); x.stroke();
      if (graft && graftsDrawn < 45) {
        const px = from.x + (to.x - from.x) * pulsePhase;
        const py = from.y + (to.y - from.y) * pulsePhase;
        x.fillStyle = '#edbc75';
        x.beginPath(); x.arc(px, py, 1.5, 0, Math.PI * 2); x.fill();
      }
      edgesDrawn++; if (graft) graftsDrawn++;
    };
    if ($('graph-grafts').checked) (g.graft_links || []).forEach(edge => drawEdge(edge, true));
    (g.edges || []).forEach(edge => drawEdge(edge, false)); x.setLineDash([]);
    projected.forEach(point => {
      const node = point.node;
      const color = sources[node.origin]?.color || sourcePalette[0];
      const act = Math.min(1, Math.abs(Number(node.activity) || 0) * activityScale);
      const active = hoveredNode?.id === node.id || selectedNodeId === node.id;
      const radius = (active ? 4.6 : 2.1) + act * 2.8;
      if (act > .12 || active) {
        const glow = x.createRadialGradient(point.x, point.y, 0, point.x, point.y, radius * 2.6);
        glow.addColorStop(0, color); glow.addColorStop(1, 'transparent');
        x.fillStyle = glow; x.beginPath(); x.arc(point.x, point.y, radius * 2.6, 0, Math.PI * 2); x.fill();
      }
      x.fillStyle = active ? '#ffffff' : color;
      x.beginPath(); x.arc(point.x, point.y, radius, 0, Math.PI * 2); x.fill();
      if (selectedNodeId === node.id) { x.strokeStyle = color; x.lineWidth = 1.4; x.beginPath(); x.arc(point.x, point.y, radius + 4, 0, Math.PI * 2); x.stroke(); }
    });
    x.font = '10px Consolas, monospace'; x.fillStyle = '#8b9daf';
    x.fillText(`${graphMode.toUpperCase()} ${graphMode === '2d' ? 'phyllotaxis spiral' : 'anatomical'} layout · schematic`, 12, 20);
    const osc = state.telemetry?.oscillatory_coherence;
    if (osc?.enabled) {
      x.fillStyle = '#5de0c1';
      x.fillText(`Kuramoto R=${Number(osc.kuramoto_order_r || 0).toFixed(2)} · CTC×${Number(osc.ctc_routing_gain || 1).toFixed(2)} [${osc.dominant_band || 'THETA'}]`, Math.max(12, w - 265), 20);
    }
    $('graph-summary').textContent = `${visible.length.toLocaleString()} / ${matching.length.toLocaleString()} sampled neurons shown (${Number(g.neurons || nodes.length).toLocaleString()} total) · ${edgesDrawn} edges, ${graftsDrawn} executing graft links. Coordinates are illustrative, not measured anatomy.`;
    const pinned = nodes.find(node => node.id === selectedNodeId);
    if (pinned) $('graph-selection').textContent = `Pinned #${pinned.id} · ${sources[pinned.origin]?.id || 'unknown source'} · ${pinned.role} · activity ${Number(pinned.activity || 0).toFixed(5)} · ${sources[pinned.origin]?.evidence.representation || 'provenance unavailable'}`;
  } catch (err) { console.error('graph render error:', err); }
}

// -------------------------------------------------------------------------
// Embodied World (True Isotropic 3D & Orthographic Top-Down)
// -------------------------------------------------------------------------
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

    const s = state.world || {};
    const a = s.agent;

    // -----------------------------------------------------------------------
    // Exact, Non-Warping Coordinate Projection Math
    // -----------------------------------------------------------------------
    const cx = w * 0.5;
    const cy = h * 0.52;

    let project;
    let worldScale;

    if (worldMode === 'top') {
      // Top-Down Orthographic Radar View (Zero perspective distortion)
      const pad = 36;
      worldScale = Math.min((w - pad) / 10.5, (h - pad) / 10.5);
      project = (px, py, z = 0) => [
        cx + px * worldScale,
        cy - py * worldScale
      ];
    } else {
      // True Isotropic 3D Isometric Projection
      // Both X and Y axes use the exact same scale factor S.
      // Arena spans [-5, 5] x [-5, 5]
      const isoCos = Math.cos(Math.PI / 6); // 0.866025
      const isoSin = Math.sin(Math.PI / 6); // 0.500000
      const maxSpanX = 20 * isoCos;         // ~17.32
      const maxSpanY = 20 * isoSin + 2.5;    // ~12.50
      const padX = 36;
      const padY = 46;

      worldScale = Math.min((w - padX) / maxSpanX, (h - padY) / maxSpanY);

      project = (px, py, z = 0) => [
        cx + (px - py) * isoCos * worldScale,
        cy + (px + py) * isoSin * worldScale - z * worldScale * 1.15
      ];
    }

    // -----------------------------------------------------------------------
    // Floor Grid & Arena Perimeter
    // -----------------------------------------------------------------------
    x.strokeStyle = '#142535';
    x.lineWidth = 0.8;
    for (let i = -5; i <= 5; i++) {
      const p1 = project(i, -5, 0), p2 = project(i, 5, 0);
      x.beginPath(); x.moveTo(p1[0], p1[1]); x.lineTo(p2[0], p2[1]); x.stroke();
      const q1 = project(-5, i, 0), q2 = project(5, i, 0);
      x.beginPath(); x.moveTo(q1[0], q1[1]); x.lineTo(q2[0], q2[1]); x.stroke();
    }

    // Arena Boundary Border
    x.strokeStyle = '#22384d';
    x.lineWidth = 1.6;
    const bCorners = [
      project(-5, -5, 0),
      project(5, -5, 0),
      project(5, 5, 0),
      project(-5, 5, 0)
    ];
    x.beginPath();
    x.moveTo(bCorners[0][0], bCorners[0][1]);
    for (let i = 1; i < 4; i++) x.lineTo(bCorners[i][0], bCorners[i][1]);
    x.closePath();
    x.stroke();

    if (!a) {
      x.fillStyle = '#8b9daf';
      x.font = '11px Consolas, monospace';
      x.fillText('Awaiting embodied world state…', 20, 35);
      return;
    }

    worldLegend(s);
    if ($('world-trails').checked && worldTrail.length > 1) {
      x.strokeStyle = 'rgba(93,224,193,.55)'; x.lineWidth = 1.8;
      x.beginPath();
      worldTrail.forEach((point, index) => { const p = project(point.x, point.y, .015); if (index === 0) x.moveTo(p[0], p[1]); else x.lineTo(p[0], p[1]); });
      x.stroke();
      const start = project(worldTrail[0].x, worldTrail[0].y, .015);
      x.strokeStyle = '#5de0c1'; x.beginPath(); x.arc(start[0], start[1], 3, 0, Math.PI * 2); x.stroke();
    }

    // -----------------------------------------------------------------------
    // Walls with 3D Height Extrusion
    // -----------------------------------------------------------------------
    for (const wall of s.walls || []) {
      if (!Array.isArray(wall) || wall.length < 4) continue;
      const [wx1, wy1, wx2, wy2] = wall;

      if (worldMode === 'top') {
        x.strokeStyle = '#4a677f';
        x.lineWidth = 3.5;
        const p1 = project(wx1, wy1);
        const p2 = project(wx2, wy2);
        x.beginPath(); x.moveTo(p1[0], p1[1]); x.lineTo(p2[0], p2[1]); x.stroke();
      } else {
        // 3D Extrusion
        const wallHeight = 0.75;
        const b1 = project(wx1, wy1, 0);
        const b2 = project(wx2, wy2, 0);
        const t1 = project(wx1, wy1, wallHeight);
        const t2 = project(wx2, wy2, wallHeight);

        // Wall vertical face
        x.fillStyle = 'rgba(56, 82, 102, 0.45)';
        x.beginPath();
        x.moveTo(b1[0], b1[1]);
        x.lineTo(b2[0], b2[1]);
        x.lineTo(t2[0], t2[1]);
        x.lineTo(t1[0], t1[1]);
        x.closePath();
        x.fill();

        // Top edge
        x.strokeStyle = '#5de0c1';
        x.lineWidth = 1.4;
        x.beginPath(); x.moveTo(t1[0], t1[1]); x.lineTo(t2[0], t2[1]); x.stroke();

        // Pillar struts
        x.strokeStyle = '#395368';
        x.lineWidth = 1.0;
        x.beginPath(); x.moveTo(b1[0], b1[1]); x.lineTo(t1[0], t1[1]); x.stroke();
        x.beginPath(); x.moveTo(b2[0], b2[1]); x.lineTo(t2[0], t2[1]); x.stroke();
      }
    }

    // -----------------------------------------------------------------------
    // Variable Terrain Zones & Friction Fields (Ice, Sand, Mud)
    // -----------------------------------------------------------------------
    (s.terrain_zones || []).forEach(tz => {
      const p = project(tz.x, tz.y, 0);
      const r = (tz.r || 1.5) * worldScale;
      const colMap = {
        ice: { fill: 'rgba(0, 220, 255, 0.22)', stroke: '#00e5ff', label: 'ICE [μ=0.25]' },
        sand: { fill: 'rgba(237, 188, 117, 0.25)', stroke: '#edbc75', label: 'SAND [μ=2.4]' },
        mud: { fill: 'rgba(100, 75, 55, 0.35)', stroke: '#8c6b4f', label: 'MUD [μ=1.8]' },
        river: { fill: 'rgba(30, 144, 255, 0.30)', stroke: '#1e90ff', label: 'RIVER CURRENT' },
        conveyor: { fill: 'rgba(255, 140, 0, 0.30)', stroke: '#ff8c00', label: 'CONVEYOR BELT' },
        shadow: { fill: 'rgba(80, 50, 120, 0.35)', stroke: '#b388ff', label: 'SHADOW CLOAK' }
      };
      const info = colMap[tz.kind] || { fill: 'rgba(120, 150, 180, 0.2)', stroke: '#8fa5b8', label: tz.kind };

      x.fillStyle = info.fill;
      x.strokeStyle = info.stroke;
      x.lineWidth = 1.4;
      x.beginPath();
      if (worldMode === 'iso') {
        x.ellipse(p[0], p[1], r, r * 0.58, 0, 0, Math.PI * 2);
      } else {
        x.arc(p[0], p[1], r, 0, Math.PI * 2);
      }
      x.fill();
      x.stroke();

      x.fillStyle = info.stroke;
      x.font = '9px Consolas, monospace';
      x.fillText(info.label, p[0] - 24, p[1] + 3);

      // Flow direction arrow for river / conveyor
      if (tz.vx || tz.vy) {
        const arrowLen = 18;
        const tip = project(tz.x + (tz.vx || 0), tz.y + (tz.vy || 0), 0);
        const ang = Math.atan2(tip[1] - p[1], tip[0] - p[0]);
        x.strokeStyle = info.stroke;
        x.lineWidth = 1.8;
        x.beginPath();
        x.moveTo(p[0], p[1]);
        x.lineTo(p[0] + Math.cos(ang) * arrowLen, p[1] + Math.sin(ang) * arrowLen);
        x.stroke();
      }
    });

    // -----------------------------------------------------------------------
    // Topographic Elevation Hill Contours
    // -----------------------------------------------------------------------
    (s.elevation_hills || []).forEach(hill => {
      const p = project(hill.x, hill.y, 0);
      const isPlateau = hill.height > 0;
      const strokeColor = isPlateau ? 'rgba(255, 215, 0, 0.5)' : 'rgba(0, 180, 255, 0.5)';
      const rings = [0.6, 1.0, 1.4];
      rings.forEach(mult => {
        const r = (hill.sigma || 1.3) * mult * worldScale;
        x.strokeStyle = strokeColor;
        x.lineWidth = 1.0;
        x.setLineDash([3, 3]);
        x.beginPath();
        if (worldMode === 'iso') {
          x.ellipse(p[0], p[1], r, r * 0.58, 0, 0, Math.PI * 2);
        } else {
          x.arc(p[0], p[1], r, 0, Math.PI * 2);
        }
        x.stroke();
      });
      x.setLineDash([]);
      x.fillStyle = strokeColor;
      x.font = '8px Consolas, monospace';
      x.fillText(isPlateau ? `PLATEAU [+${hill.height}m]` : `BASIN [${hill.height}m]`, p[0] - 28, p[1] + 3);
    });

    // -----------------------------------------------------------------------
    // Rotating Sentinel Searchlights & Stealth Cones
    // -----------------------------------------------------------------------
    (s.sentinels || []).forEach(sent => {
      const p = project(sent.x, sent.y, 0);
      const beamLen = (sent.range || 2.8) * worldScale;
      const halfAngle = sent.cone_half_angle || 0.42;
      const ang = sent.angle || 0;

      // Draw searchlight beam cone
      x.fillStyle = sent.alerted ? 'rgba(255, 50, 50, 0.28)' : 'rgba(255, 170, 0, 0.18)';
      x.strokeStyle = sent.alerted ? '#ff3333' : '#ffaa00';
      x.lineWidth = 1.2;
      x.beginPath();
      x.moveTo(p[0], p[1]);
      for (let part = 0; part <= 20; part++) {
        const angle = ang - halfAngle + 2 * halfAngle * part / 20;
        const tip = project(sent.x + Math.cos(angle) * (sent.range || 2.8), sent.y + Math.sin(angle) * (sent.range || 2.8), 0);
        x.lineTo(tip[0], tip[1]);
      }
      x.closePath();
      x.fill();
      x.stroke();

      // Sentinel drone body
      x.fillStyle = sent.alerted ? '#ff2222' : '#ff9900';
      x.beginPath();
      x.arc(p[0], p[1], 4, 0, Math.PI * 2);
      x.fill();
    });

    // -----------------------------------------------------------------------
    // Acoustic Sound Tripwires
    // -----------------------------------------------------------------------
    (s.sound_tripwires || []).forEach(trip => {
      const p1 = project(trip.x1, trip.y1, 0);
      const p2 = project(trip.x2, trip.y2, 0);
      x.strokeStyle = '#ff00aa';
      x.lineWidth = 1.6;
      x.setLineDash([2, 4]);
      x.beginPath();
      x.moveTo(p1[0], p1[1]);
      x.lineTo(p2[0], p2[1]);
      x.stroke();
      x.setLineDash([]);
      x.fillStyle = '#ff00aa';
      x.font = '7px Consolas, monospace';
      x.fillText('TRIPWIRE [v<0.28]', (p1[0] + p2[0]) / 2 - 25, (p1[1] + p2[1]) / 2 - 3);
    });

    // -----------------------------------------------------------------------
    // Dynamic Sliding & Locked Barrier Gates
    // -----------------------------------------------------------------------
    (s.dynamic_gates || []).forEach(gate => {
      const p1 = project(gate.x1, gate.y1, 0);
      const p2 = project(gate.x2, gate.y2, 0);
      const isOpen = Boolean(gate.open);

      if (isOpen) {
        x.strokeStyle = 'rgba(93, 224, 193, 0.35)';
        x.lineWidth = 2.0;
        x.setLineDash([4, 4]);
        x.beginPath(); x.moveTo(p1[0], p1[1]); x.lineTo(p2[0], p2[1]); x.stroke();
        x.setLineDash([]);
        x.fillStyle = '#5de0c1';
        x.font = '8px Consolas, monospace';
        x.fillText('GATE OPEN', (p1[0] + p2[0]) / 2 - 20, (p1[1] + p2[1]) / 2 - 4);
      } else {
        x.strokeStyle = gate.locked ? '#ec7c8d' : '#ff9900';
        x.lineWidth = 4.0;
        x.beginPath(); x.moveTo(p1[0], p1[1]); x.lineTo(p2[0], p2[1]); x.stroke();
        x.fillStyle = gate.locked ? '#ec7c8d' : '#ff9900';
        x.font = '8px Consolas, monospace';
        x.fillText(gate.locked ? 'LOCKED GATE' : 'GATE CLOSED', (p1[0] + p2[0]) / 2 - 24, (p1[1] + p2[1]) / 2 - 4);
      }
    });

    // -----------------------------------------------------------------------
    // Objects / Targets / Food / Buttons (Depth sorted)
    // -----------------------------------------------------------------------
    const objects = (s.objects || []).filter(o => o.visible !== false);
    if (worldMode === 'iso') {
      objects.sort((o1, o2) => (o1.x + o1.y) - (o2.x + o2.y));
    }

    objects.forEach(o => {
      const p = project(o.x, o.y, o.z || 0);
      const rad = Math.max(3, (o.r || 0.25) * worldScale);
      const color = parseColor(o.color, ({ food: '#edbc75', hazard: '#ec7c8d', target: '#7faaff' }[o.kind] || '#5de0c1'));

      if (worldMode === 'iso') {
        // Floor shadow
        const sp = project(o.x, o.y, 0);
        x.fillStyle = 'rgba(0,0,0,0.35)';
        x.beginPath();
        x.ellipse(sp[0], sp[1], rad, rad * 0.58, 0, 0, Math.PI * 2);
        x.fill();

        // 3D Spherical body
        const grad = x.createRadialGradient(p[0] - rad * 0.3, p[1] - rad * 0.3, 0, p[0], p[1], rad);
        grad.addColorStop(0, '#ffffff');
        grad.addColorStop(0.3, color);
        grad.addColorStop(1, '#050a10');
        x.fillStyle = grad;
        x.beginPath();
        x.arc(p[0], p[1], rad, 0, Math.PI * 2);
        x.fill();
      } else {
        x.fillStyle = color;
        x.beginPath();
        x.arc(p[0], p[1], rad, 0, Math.PI * 2);
        x.fill();
      }

      if ($('world-goals').checked && ['food', 'target', 'goal', 'button', 'socket'].includes(o.kind)) {
        x.strokeStyle = color; x.lineWidth = 1.5; x.setLineDash([3, 3]);
        x.beginPath(); x.arc(p[0], p[1], rad + 5, 0, Math.PI * 2); x.stroke(); x.setLineDash([]);
        x.fillStyle = color; x.font = '9px Consolas, monospace';
        x.fillText(o.pressed ? 'PRESSED' : o.held ? 'HELD' : o.kind === 'food' ? 'FOOD' : 'TASK OBJECT', p[0] - rad, p[1] - rad - 9);
      }
      x.fillStyle = '#8b9daf';
      x.font = '9px Consolas, monospace';
      x.fillText(o.kind || 'obj', p[0] + rad + 3, p[1] - 3);
    });

    // -----------------------------------------------------------------------
    // Autonomous Moving Predators & Hazard Drones
    // -----------------------------------------------------------------------
    (s.predators || []).forEach(pred => {
      const pp = project(pred.x, pred.y, 0.1);
      const prad = Math.max(4, (pred.r || 0.35) * worldScale);
      const col = pred.color === 'purple' ? '#b066ff' : '#ec7c8d';

      // Pulsing threat aura
      const aura = x.createRadialGradient(pp[0], pp[1], 0, pp[0], pp[1], prad * 2.2);
      aura.addColorStop(0, pred.color === 'purple' ? 'rgba(176, 102, 255, 0.35)' : 'rgba(236, 124, 141, 0.35)');
      aura.addColorStop(1, 'transparent');
      x.fillStyle = aura;
      x.beginPath();
      x.arc(pp[0], pp[1], prad * 2.2, 0, Math.PI * 2);
      x.fill();

      // Drone Chassis
      x.fillStyle = col;
      x.beginPath();
      x.arc(pp[0], pp[1], prad, 0, Math.PI * 2);
      x.fill();

      // Heading Vector
      const pth = Number(pred.th || 0);
      const pTipX = pred.x + Math.cos(pth) * 0.6;
      const pTipY = pred.y + Math.sin(pth) * 0.6;
      const pTipP = project(pTipX, pTipY, 0.1);

      x.strokeStyle = '#ffffff';
      x.lineWidth = 2.0;
      x.beginPath(); x.moveTo(pp[0], pp[1]); x.lineTo(pTipP[0], pTipP[1]); x.stroke();

      x.fillStyle = col;
      x.font = '9px Consolas, monospace';
      x.fillText(`PREDATOR · ${pred.kind || 'stalker'}`, pp[0] + prad + 4, pp[1] - 2);
    });

    // -----------------------------------------------------------------------
    // Cooperative Helper Drones & Protective Shield Aura
    // -----------------------------------------------------------------------
    (s.helper_drones || []).forEach(helper => {
      const hp = project(helper.x, helper.y, 0.14);
      const hrad = Math.max(4, (helper.r || 0.32) * worldScale);
      const shieldR = Math.max(12, (helper.shield_radius || 1.65) * worldScale);
      const aura = x.createRadialGradient(hp[0], hp[1], 0, hp[0], hp[1], shieldR);
      aura.addColorStop(0, 'rgba(93, 224, 193, 0.25)');
      aura.addColorStop(0.75, 'rgba(0, 229, 255, 0.10)');
      aura.addColorStop(1, 'transparent');
      x.fillStyle = aura;
      x.beginPath();
      if (worldMode === 'iso') x.ellipse(hp[0], hp[1], shieldR, shieldR * 0.58, 0, 0, Math.PI * 2);
      else x.arc(hp[0], hp[1], shieldR, 0, Math.PI * 2);
      x.fill();

      x.strokeStyle = 'rgba(0, 229, 255, 0.55)';
      x.lineWidth = 1.2;
      x.setLineDash([4, 3]);
      x.beginPath();
      if (worldMode === 'iso') x.ellipse(hp[0], hp[1], shieldR, shieldR * 0.58, 0, 0, Math.PI * 2);
      else x.arc(hp[0], hp[1], shieldR, 0, Math.PI * 2);
      x.stroke();
      x.setLineDash([]);

      x.fillStyle = '#00e5ff';
      x.beginPath();
      x.arc(hp[0], hp[1], hrad, 0, Math.PI * 2);
      x.fill();
      x.fillStyle = '#5de0c1';
      x.font = '9px Consolas, monospace';
      x.fillText(`ALLY · ${helper.kind || 'guardian_drone'} [SHIELD]`, hp[0] + hrad + 4, hp[1] - 2);
    });

    // -----------------------------------------------------------------------
    // Planner Counterfactual Ghost Trajectory
    // -----------------------------------------------------------------------
    const planAct = state.telemetry?.experience_planner?.proposed_action || state.action;
    if (Array.isArray(planAct) && planAct.length >= 2 && (state.config?.experience_planning || state.telemetry?.recurrent_planner?.used)) {
      const horizon = Math.max(2, Number(state.config?.experience_horizon || 3));
      let gx = a.x, gy = a.y, gth = Number(a.th || 0);
      const drive = Number(planAct[0] || 0), turn = Number(planAct[1] || 0);
      x.strokeStyle = 'rgba(192, 132, 252, 0.65)';
      x.lineWidth = 1.6;
      x.setLineDash([3, 3]);
      x.beginPath();
      const p0 = project(gx, gy, 0.05);
      x.moveTo(p0[0], p0[1]);
      const ghostPts = [];
      for (let hStep = 1; hStep <= horizon; hStep++) {
        gth += turn * 0.25;
        gx = Math.max(-4.8, Math.min(4.8, gx + Math.cos(gth) * drive * 0.45));
        gy = Math.max(-4.8, Math.min(4.8, gy + Math.sin(gth) * drive * 0.45));
        const gp = project(gx, gy, 0.05);
        x.lineTo(gp[0], gp[1]);
        ghostPts.push(gp);
      }
      x.stroke();
      x.setLineDash([]);
      ghostPts.forEach((gp, idx) => {
        x.fillStyle = `rgba(192, 132, 252, ${0.8 - idx * 0.12})`;
        x.beginPath();
        x.arc(gp[0], gp[1], 3.2 - idx * 0.3, 0, Math.PI * 2);
        x.fill();
      });
    }

    // -----------------------------------------------------------------------
    // Retinal Sensing Rays
    // -----------------------------------------------------------------------
    const ap = project(a.x, a.y, a.z || 0);
    if (Array.isArray(s.rays)) {
      x.strokeStyle = 'rgba(93,224,193,0.18)';
      x.lineWidth = 0.9;
      s.rays.forEach(ray => {
        if (!Array.isArray(ray) || ray.length < 2) return;
        const rp = project(ray[0], ray[1], 0);
        x.beginPath();
        x.moveTo(ap[0], ap[1]);
        x.lineTo(rp[0], rp[1]);
        x.stroke();
      });
    }

    // -----------------------------------------------------------------------
    // Embodied Agent Body, Articulated Arm & Heading
    // -----------------------------------------------------------------------
    const agentRadius = Math.max(5, 0.35 * worldScale);
    const th = Number(a.th || 0);

    if (worldMode === 'iso') {
      // Agent Floor Shadow
      x.fillStyle = 'rgba(0,0,0,0.45)';
      x.beginPath();
      x.ellipse(ap[0], ap[1], agentRadius * 1.2, agentRadius * 0.7, 0, 0, Math.PI * 2);
      x.fill();

      // Agent Metallic Chassis Puck
      const grad = x.createRadialGradient(ap[0] - 2, ap[1] - 4, 0, ap[0], ap[1], agentRadius);
      grad.addColorStop(0, '#a0f5df');
      grad.addColorStop(0.6, '#5de0c1');
      grad.addColorStop(1, '#1b4a3f');
      x.fillStyle = grad;
      x.beginPath();
      x.ellipse(ap[0], ap[1] - 3, agentRadius, agentRadius * 0.85, 0, 0, Math.PI * 2);
      x.fill();

      // Heading Vector (True isotropic projection, uniform rotation speed)
      const tipWorldX = a.x + Math.cos(th) * 0.75;
      const tipWorldY = a.y + Math.sin(th) * 0.75;
      const tipP = project(tipWorldX, tipWorldY, 0.1);

      x.strokeStyle = '#ffffff';
      x.lineWidth = 2.4;
      x.beginPath();
      x.moveTo(ap[0], ap[1] - 3);
      x.lineTo(tipP[0], tipP[1]);
      x.stroke();

      // Articulated Arm & Gripper
      if (Array.isArray(a.tip) && a.tip.length >= 3) {
        const armTipP = project(a.tip[0], a.tip[1], a.tip[2]);
        x.strokeStyle = '#edbc75';
        x.lineWidth = 2.0;
        x.beginPath();
        x.moveTo(ap[0], ap[1] - 5);
        x.lineTo(armTipP[0], armTipP[1]);
        x.stroke();

        // Gripper jaws
        const isGripping = (a.grip ?? 0) > 0.5;
        x.fillStyle = isGripping ? '#ec7c8d' : '#edbc75';
        x.beginPath();
        x.arc(armTipP[0], armTipP[1], 3.5, 0, Math.PI * 2);
        x.fill();
      }

    } else {
      // Top-Down Radar Agent
      x.fillStyle = '#5de0c1';
      x.beginPath();
      x.arc(ap[0], ap[1], agentRadius, 0, Math.PI * 2);
      x.fill();

      // Directional arrow
      const hx = Math.cos(th) * agentRadius * 1.6;
      const hy = -Math.sin(th) * agentRadius * 1.6;
      x.strokeStyle = '#ffffff';
      x.lineWidth = 2.2;
      x.beginPath();
      x.moveTo(ap[0], ap[1]);
      x.lineTo(ap[0] + hx, ap[1] + hy);
      x.stroke();
    }

    // Ambient Illumination & Day/Night Cycle
    if (typeof s.ambient_light === 'number') {
      const amb = s.ambient_light;
      if (amb < 0.85) {
        const darkAlpha = Math.max(0, Math.min(0.65, (0.85 - amb) * 0.75));
        x.fillStyle = `rgba(5, 12, 24, ${darkAlpha.toFixed(3)})`;
        x.fillRect(0, 0, w, h);
      }
      x.fillStyle = amb < 0.75 ? '#8fa5b8' : '#edbc75';
      x.font = '10px Consolas, monospace';
      const icon = amb < 0.75 ? '🌙 NIGHT' : '☀️ DAY';
      x.fillText(`${icon} (Light ${Math.round(amb * 100)}%)`, w - 140, 22);
    }

    // Status overlay text
    x.fillStyle = '#8b9daf';
    x.font = '10px Consolas, monospace';
    x.fillText(`Episode ${s.episode ?? 0} · t=${s.t ?? 0} · return=${Number(s.ep_return ?? 0).toFixed(3)} · View: ${worldMode.toUpperCase()}`, 14, 22);
    $('worldtask').textContent = s.task ? `TASK: ${s.task.toUpperCase()}` : '';

  } catch (err) {
    console.error('world render error:', err);
  }
}

// -------------------------------------------------------------------------
// Routing & Learning Signals Timeseries
// -------------------------------------------------------------------------
function signals() {
  try {
    const [x, w, h] = canvas('signals');
    if (!x) return;

    // Grid baseline
    x.strokeStyle = '#142535';
    x.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      x.beginPath(); x.moveTo(0, h * i / 4); x.lineTo(w, h * i / 4); x.stroke();
    }

    const keys = ['graph_mean', 'graph_delta', 'curiosity'];
    const colors = ['#5de0c1', '#edbc75', '#80aaff'];

    keys.forEach((key, j) => {
      const values = signalHistory.map(t => Number(t[key] ?? 0));
      const max = Math.max(0.01, ...values.map(Math.abs));

      // Gradient area fill
      const grad = x.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, colors[j] + '22');
      grad.addColorStop(1, 'transparent');

      x.fillStyle = grad;
      x.strokeStyle = colors[j];
      x.lineWidth = 1.6;

      x.beginPath();
      values.forEach((v, i) => {
        const px = (i / Math.max(1, values.length - 1)) * w;
        const py = h - 25 - (v / max) * (h - 45);
        if (i === 0) x.moveTo(px, py);
        else x.lineTo(px, py);
      });
      x.stroke();

      x.fillStyle = colors[j];
      x.font = '9px Consolas, monospace';
      x.fillText(`${key}: ${(values.at(-1) || 0).toFixed(4)}`, 14 + j * 140, h - 8);
    });
  } catch (err) {
    console.error('signals render error:', err);
  }
}

// -------------------------------------------------------------------------
// Neuromodulatory & Astrocyte Field Gauges
// -------------------------------------------------------------------------
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

    const cols = w > 600 ? 3 : (w > 360 ? 2 : 1);
    const barW = (w - 30 - (cols - 1) * 12) / cols;
    const barH = 13;

    items.forEach((item, idx) => {
      const colIdx = idx % cols;
      const rowIdx = Math.floor(idx / cols);
      const bx = 15 + colIdx * (barW + 12);
      const by = 24 + rowIdx * 50;

      // Label & Value
      x.fillStyle = '#a9c4d9';
      x.font = '10px Consolas, monospace';
      x.fillText(item.label, bx, by);

      x.fillStyle = item.col;
      x.fillText(item.val.toFixed(3), bx + barW - 38, by);

      // Track
      x.fillStyle = '#142535';
      x.fillRect(bx, by + 5, barW, barH);

      // Level with gradient
      const pct = Math.max(0, Math.min(1, item.val / item.max));
      x.fillStyle = item.col;
      x.fillRect(bx, by + 5, barW * pct, barH);

      // Role subtitle
      x.fillStyle = '#647c90';
      x.font = '8px Consolas, monospace';
      x.fillText(item.role, bx, by + barH + 16);
    });
  } catch (err) {
    console.error('gauges render error:', err);
  }
}

// -------------------------------------------------------------------------
// Predictive Coding & Free Energy
// -------------------------------------------------------------------------
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
      if (x.roundRect) x.roundRect(tx, ty, tileW, tileH, 5);
      else x.rect(tx, ty, tileW, tileH);
      x.fill();
      x.stroke();

      x.fillStyle = '#8b9daf';
      x.font = '9px Consolas, monospace';
      x.fillText(t.label, tx + 10, ty + 18);

      x.fillStyle = t.col;
      x.font = '16px Consolas, monospace';
      x.fillText(t.val, tx + 10, ty + 40);
    });

    x.fillStyle = '#647c90';
    x.font = '9px Consolas, monospace';
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

// Training controls and evidence are populated from the executing runtime.
function setTone(element, tone) {
  if (!element) return;
  for (const name of ['good', 'warn', 'bad', 'muted', 'ponder', 'purple']) element.classList.toggle(`tone-${name}`, name === tone);
}

function keyValues(element, rows) {
  element.replaceChildren(...rows.map(([label, value]) => {
    const row = document.createElement('div'); row.className = 'kv';
    const name = document.createElement('span'), result = document.createElement('b');
    name.textContent = label; result.textContent = value; row.append(name, result); return row;
  }));
}

function renderTraining(status) {
  trainingStatus = { ...trainingStatus, ...Object.fromEntries(Object.entries(status || {}).filter(([, value]) => value !== undefined)) };
  const compute = trainingStatus.compute || {}, graft = trainingStatus.graft || {}, prior = trainingStatus.programmed_prior || {};
  const active = !!trainingStatus.active_training;
  $('training-badge').textContent = graft.enabled ? active ? 'ONLINE TRAINING' : 'WEIGHTS FROZEN' : 'GRAFT DISABLED';
  setTone($('training-badge'), graft.enabled ? active ? 'good' : 'warn' : 'muted');
  if (!$('training-active').dataset.dirty) $('training-active').checked = active;
  if (!$('training-threads').dataset.dirty) $('training-threads').value = String(Math.min(8, compute.threads || state.config?.compute_threads || 1));
  const logical = compute.logical_cpus || compute.logical_cpu_count;
  [...$('training-threads').options].forEach(option => { option.disabled = !!logical && Number(option.value) > logical; });
  $('training-compute').textContent = `${compute.threads || 1} requested CPU threads · ${compute.torch_threads || compute.intraop_threads || compute.threads || 1} PyTorch threads${logical ? ` · ${logical} logical CPUs available` : ''}. Online fitting also requires “Learn from executed steps”.`;
  keyValues($('training-metrics'), [
    ['Learned parameters', Number(graft.parameters || 0).toLocaleString()],
    ['Executed replay', `${graft.replay_size || 0} / ${graft.replay_capacity || '—'}`],
    ['Optimizer updates', graft.optimizer_updates || 0],
    ['Actor updates', graft.actor_updates || 0],
    ['Programmed source prior', graft.connectome?.bound ? `${graft.connectome.nodes} nodes · ${graft.connectome.edges} edges bound` : 'No source prior bound']
  ]);
  $('training-prior').textContent = prior.enabled ? prior.available ? `Source prior bound from ${prior.nodes || graft.connectome?.nodes || 0} source nodes. This initialization is separate from learned weights and measured task performance.` : `Requested source prior unavailable: ${prior.reason || 'inspect provenance below'}` : 'Programmed source prior disabled. Source graph provenance is reported separately from graft training.';
  show('source-details', { active_sources: state.graph?.sources || [], programmed_prior: prior, graft_connectome: graft.connectome || {} });
  if (!$('learn').dataset.initialized) { $('learn').checked = active; $('learn').dataset.initialized = 'true'; }
}

async function refreshTraining() { renderTraining(await api('training/status')); }

async function refreshInventory() {
  const inventory = await api('connectomes/inventory');
  const activeIds = new Set((state.config?.connectome_sources || []));
  $('source-inventory').replaceChildren(...(inventory.sources || []).map(source => {
    const row = document.createElement('div'); row.className = 'source-record';
    const label = document.createElement('b'), detail = document.createElement('span');
    label.textContent = `${source.id}${activeIds.has(source.id) ? ' · active' : ''}`;
    detail.textContent = `${source.availability || 'unknown'} · ${source.representation || 'unspecified'} · ${source.neuron_count == null ? 'count unavailable' : Number(source.neuron_count).toLocaleString() + ' source nodes'} · ${source.verified ? 'verified cache' : 'verification not established'}`;
    row.append(label, detail);
    if (source.source_scope) { const scope = document.createElement('small'); scope.textContent = source.source_scope; row.append(scope); }
    return row;
  }));
}

const observedTerminalJobs = new Set();
function renderTrainingJob() {
  const relevant = latestJobs.filter(job => ['pretrain', 'rsi'].includes(job.kind));
  const job = latestJobs.find(item => item.id === trackedJobId) || relevant.at(-1);
  if (!job) return;
  trackedJobId = job.id;
  const progress = job.progress || {};
  const live = ['queued', 'running'].includes(job.status);
  const applied = job.result?.applied;
  const outcome = applied === true ? ' · weights applied to session' : applied === false ? ' · not applied' : '';
  $('training-job-status').textContent = `${job.kind === 'rsi' ? 'Live RSI' : 'Pretrain'} · ${job.status}${progress.phase ? ' · ' + progress.phase : ''}${progress.total ? ` · ${progress.completed || 0}/${progress.total}` : ''}${outcome}`;
  $('training-job-progress').max = progress.total || 1;
  if (live && !progress.total) $('training-job-progress').removeAttribute('value');
  else $('training-job-progress').value = live ? progress.completed || 0 : job.status === 'completed' ? progress.total || 1 : 0;
  $('training-cancel').hidden = !live;
  $('training-cancel').disabled = !!job.cancel_requested;
  $('training-pretrain').disabled = live || trainingRequest;
  $('rsi-live').disabled = live || trainingRequest;
  show('training-receipt', job.error || job.result || progress);
  if (job.kind === 'rsi') { $('rsi-status').textContent = $('training-job-status').textContent; show('rsi-receipt', job.error || job.result || progress); }
  if (!live && !observedTerminalJobs.has(job.id)) {
    observedTerminalJobs.add(job.id);
    action(async () => { render(await api('state')); await refreshTraining(); });
  }
}

function updateWorldTrail(snapshot) {
  const world = snapshot.world || {}, agent = world.agent;
  const identity = JSON.stringify([snapshot.config?.seed, world.task, world.episode]);
  if (identity !== trailIdentity || Number(snapshot.steps) < Number(lastTrailStep)) { worldTrail = []; trailIdentity = identity; lastTrailStep = null; }
  if (agent && snapshot.steps !== lastTrailStep && Number.isFinite(agent.x) && Number.isFinite(agent.y)) {
    worldTrail.push({ x: agent.x, y: agent.y }); worldTrail = worldTrail.slice(-160); lastTrailStep = snapshot.steps;
  }
}

function worldLegend(world) {
  const kinds = new Set((world.objects || []).filter(object => object.visible !== false).map(object => object.kind));
  const entries = [['agent', 'Agent / heading']];
  if ($('world-trails').checked) entries.push(['trail', `Trajectory · ${worldTrail.length} samples`]);
  for (const kind of kinds) entries.push([['target', 'goal', 'food', 'hazard', 'button'].includes(kind) ? kind : 'object', kind]);
  if (world.predators?.length) entries.push(['hazard', `Predators · ${world.predators.length}`]);
  if (world.helper_drones?.length) entries.push(['agent', `Helper Drones · ${world.helper_drones.length}`]);
  if (world.sentinels?.length) entries.push(['sentinel', `Sentinels · ${world.sentinels.length}`]);
  if (world.dynamic_gates?.length) entries.push(['gate', `Gates · ${world.dynamic_gates.filter(gate => gate.open).length}/${world.dynamic_gates.length} open`]);
  if (world.sound_tripwires?.length) entries.push(['tripwire', 'Sound tripwires']);
  for (const kind of new Set((world.terrain_zones || []).map(zone => zone.kind))) entries.push(['terrain', kind]);
  $('world-legend').replaceChildren(...entries.map(([kind, label]) => { const item = document.createElement('span'); item.className = `world-key key-${kind}`; item.textContent = label; return item; }));
  const targets = (world.objects || []).filter(object => object.visible !== false && ['food', 'target', 'goal', 'button', 'socket'].includes(object.kind));
  const nearest = world.agent && targets.reduce((best, object) => {
    const distance = Math.hypot(object.x - world.agent.x, object.y - world.agent.y);
    return !best || distance < best.distance ? { kind: object.kind, distance } : best;
  }, null);
  $('world-summary').textContent = `${world.task || 'World'} · ${world.alarm_tripped ? 'ALARM TRIPPED · ' : ''}${nearest ? `Nearest visible ${nearest.kind}: ${nearest.distance.toFixed(2)} world units · ` : ''}Trail resets each episode. Goals and hazards reflect the current snapshot.`;
}

// -------------------------------------------------------------------------
// Master Render Loop
// -------------------------------------------------------------------------
function render(s) {
  if (!s) return;
  state = s;
  updateWorldTrail(s);
  renderTraining({ compute: s.compute, graft: s.training?.graft, programmed_prior: s.training?.programmed_prior, active_training: s.config?.active_training });
  show('snapshot', s);
  const neuronInput = $('config').elements.namedItem('total_neurons');
  const sourceCount = s.config?.connectome_sources?.length || 3;
  const sizeKey = JSON.stringify([s.config?.total_neurons, s.config?.nodes_per_source, sourceCount]);
  if (neuronInput && neuronInput.dataset.sizeKey !== sizeKey) {
    neuronInput.value = String(s.config?.total_neurons ?? s.graph?.neurons ?? 2000);
    neuronInput.min = String(16 * sourceCount);
    neuronInput.max = String(10000 * sourceCount);
    neuronInput.dataset.sizeKey = sizeKey;
  }

  const experience = s.experience || {};
  $('experience-mode').value = experience.mode || 'off';
  $('experience-horizon').value = String(s.config?.experience_horizon || 1);
  if ($('experience-algorithm')) {
    $('experience-algorithm').value = s.config?.planner_algorithm || 'experience_mpc';
  }
  if ($('experience-value-guided')) {
    $('experience-value-guided').checked = !!s.config?.value_guided_planning;
  }
  const configForm = $('config');
  if (configForm) {
    for (const boolField of [
      'continual_backprop', 'fast_weight_memory', 'dormant_lifecycle', 'latent_reasoning',
      'value_guided_planning', 'self_model', 'experimental_cognition',
      'dendritic_ltc', 'oscillatory_coherence', 'verifiable_dpo', 'programmed_connectome'
    ]) {
      const el = configForm.elements.namedItem(boolField);
      if (el && !el.dataset.userTouched) {
        if (boolField === 'dormant_lifecycle' || boolField === 'self_model' || boolField === 'programmed_connectome') {
          el.checked = s.config?.[boolField] !== false;
        } else {
          el.checked = !!s.config?.[boolField];
        }
      }
    }
    for (const selectField of ['task', 'readout', 'planner_algorithm']) {
      const el = configForm.elements.namedItem(selectField);
      if (el && s.config?.[selectField] && !el.dataset.userTouched) {
        el.value = String(s.config[selectField]);
      }
    }
  }
  const model = experience.model || {}, memory = experience.memory || {}, planner = experience.planner || {};
  $('experience-summary').textContent = `${model.samples || 0} observed transitions · ${memory.size || 0}/${memory.capacity || 128} memory slots · reward prediction RMSE ${Number(model.prequential_reward_rmse || 0).toFixed(5)} · ${planner.reason || 'collect support to begin'}`;
  show('experience-details', experience);
  const gates = [
    ['Reward model', model.prediction_gate_passed ? 'support gate passed' : 'collect more support', !!model.prediction_gate_passed],
    ['Dynamics model', model.dynamics_gate_passed ? 'support gate passed' : 'multi-step not ready', !!model.dynamics_gate_passed],
    ['Learning', s.telemetry?.manual_teleop ? 'manual executed action' : ($('learn').checked ? 'enabled' : 'frozen'), false],
    ['Action order', 'drive · turn · reach · height · grip', false]
  ];
  $('learning-contract').replaceChildren(...gates.map(([label, value, ready]) => {
    const item = document.createElement('div');
    item.className = `contract-item${ready ? ' ready' : ''}`;
    const name = document.createElement('small'), text = document.createElement('span');
    name.textContent = label; text.textContent = value; item.append(name, text); return item;
  }));
  if (!$('plan-preview').hidden && Number($('plan-step').dataset.step) !== s.steps) {
    $('plan-step').textContent = `Step ${$('plan-step').dataset.step} · stale preview`;
  }

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

  // Homeostatic Vitals HUD
  const homeo = s.telemetry?.homeostasis || {};
  const vitals = homeo.vitals || {};
  if ($('thrust-governor')) {
    const gov = Math.round((homeo.max_allowed_thrust ?? 1.0) * 100);
    $('thrust-governor').textContent = `Thrust Limit: ${gov}%`;
    setTone($('thrust-governor'), gov < 50 ? 'bad' : gov < 90 ? 'warn' : 'good');
  }
  if ($('vitals-hud')) {
    const vitalSpecs = [
      { name: 'Energy (Metabolic reserve)', val: vitals.energy ?? 0.85, col: '#5de0c1' },
      { name: 'Stamina (Dynamic effort)', val: vitals.stamina ?? 1.00, col: '#edbc75' },
      { name: 'Health (Integrity)', val: vitals.health ?? 1.00, col: '#4ae582' },
      { name: 'Threat Proximity', val: vitals.threat ?? 0.00, col: '#ec7c8d' },
      { name: 'Hunger Drive', val: vitals.hunger ?? 0.15, col: '#bc85e6' }
    ];
    $('vitals-hud').replaceChildren(...vitalSpecs.map(v => {
      const row = document.createElement('div');
      row.className = 'vital-row';
      const label = document.createElement('div');
      label.className = 'vital-label';
      const pct = Math.max(0, Math.min(100, Math.round(v.val * 100)));
      label.innerHTML = `<span>${v.name}</span><span>${pct}%</span>`;
      const track = document.createElement('progress');
      track.className = `vital-track vital-${['energy', 'stamina', 'health', 'threat', 'hunger'][vitalSpecs.indexOf(v)]}`;
      track.max = 100; track.value = pct; track.setAttribute('aria-label', v.name);
      row.append(label, track);
      return row;
    }));
  }

  // MiMoMix Cognition & Neurogenesis HUD
  const mimomix = s.telemetry?.mimomix || {};
  const neuro = s.telemetry?.neurogenesis || {};
  if ($('mimomix-badge')) {
    const engaged = mimomix.system_engaged || (s.config?.experimental_cognition ? 'Awaiting action' : 'Experimental cognition disabled');
    $('mimomix-badge').textContent = engaged;
    if (engaged.includes('System 2')) {
      setTone($('mimomix-badge'), 'ponder');
    } else {
      setTone($('mimomix-badge'), 'good');
    }
  }
  if ($('cognition-hud')) {
    const cogMetrics = [
      ['Deliberation Mode', mimomix.system_engaged || 'System 1 (Reflex)'],
      ['Ponder Steps', `${mimomix.ponder_steps || 0} / 4 ACT steps`],
      ['Ponder Cost', Number(mimomix.ponder_cost || 0).toFixed(4)],
      ['Heuristic confidence', s.config?.experimental_cognition ? `${Math.round((mimomix.confidence || 0) * 100)}%` : 'inactive'],
      ['Synapses Born', neuro.synapses_born ?? 0],
      ['Synapses Pruned', neuro.synapses_pruned ?? 0],
      ['Active Synapses', neuro.active_synapses ?? s.graph?.edges?.length ?? '—']
    ];
    $('cognition-hud').replaceChildren(...cogMetrics.map(([k, val]) => {
      const d = document.createElement('div');
      d.className = 'kv';
      const name = document.createElement('span'), v = document.createElement('b');
      name.textContent = k;
      v.textContent = val;
      d.append(name, v);
      return d;
    }));
  }

  // Dormant Expert Lifecycle HUD
  const dormant = s.telemetry?.dormant_lifecycle || {};
  const dormantSlots = dormant.slots || [];
  if ($('dormant-badge')) {
    const isEnabled = s.config?.dormant_lifecycle !== false;
    const activeSlots = dormantSlots.filter(sl => sl.state === 'ACTIVE' || sl.state === 'ACTIVATING').length;
    $('dormant-badge').textContent = isEnabled ? `${activeSlots} active / ${dormantSlots.length} slots` : 'Disabled';
    setTone($('dormant-badge'), isEnabled ? 'good' : 'bad');
  }
  if ($('dormant-hud')) {
    const slot0 = dormantSlots[0] || {};
    const slot1 = dormantSlots[1] || {};
    const dormantMetrics = [
      ['Slot 0 State', `${slot0.state || 'DORMANT'} (gain: ${Number(slot0.activation_gain || 0).toFixed(2)})`],
      ['Slot 1 State', `${slot1.state || 'DORMANT'} (gain: ${Number(slot1.activation_gain || 0).toFixed(2)})`],
      ['Slot 0 Usage', `${slot0.usage_count || 0} calls · score ${Number(slot0.contribution_score || 0).toFixed(3)}`],
      ['Slot 1 Usage', `${slot1.usage_count || 0} calls · score ${Number(slot1.contribution_score || 0).toFixed(3)}`],
      ['Consolidations', `${dormant.consolidations_completed || 0} native distillations`],
      ['Recruited Expert', s.telemetry?.recruited_expert != null ? `Slot ${s.telemetry.recruited_expert}` : 'None (idle)']
    ];
    $('dormant-hud').replaceChildren(...dormantMetrics.map(([k, val]) => {
      const d = document.createElement('div');
      d.className = 'kv';
      const name = document.createElement('span'), v = document.createElement('b');
      name.textContent = k;
      v.textContent = val;
      d.append(name, v);
      return d;
    }));
  }

  // Fast-Weight Plasticity & Continual Backprop HUD
  const cbp = s.telemetry?.continual_backprop || {};
  const fwm = s.telemetry?.fast_weight_memory || {};
  const lat = s.telemetry?.latent_reasoner || {};
  const evoked = s.telemetry?.evoked_channels || {};
  if ($('plasticity-badge')) {
    const plastActive = (cbp.enabled || fwm.enabled || lat.engaged);
    $('plasticity-badge').textContent = plastActive ? 'Plasticity engaged' : 'Standard Adam';
    setTone($('plasticity-badge'), plastActive ? 'good' : 'muted');
  }
  if ($('plasticity-hud')) {
    const plastMetrics = [
      ['Continual Backprop', cbp.enabled ? `Active · ${cbp.reinit_count || 0} reinits · step ${cbp.grad_step_count || 0}` : 'Disabled'],
      ['Fast-Weight (Titans)', fwm.enabled ? `Active · ${fwm.inner_step_count || 0} updates · η=${fwm.eta || 0.1}` : 'Disabled'],
      ['Coconut Deliberation', lat.engaged ? `Engaged · ${lat.recurrent_depth || 0} steps · cost ${Number(lat.ponder_cost || 0).toFixed(3)}` : (s.config?.latent_reasoning ? 'Monitoring surprise' : 'Disabled')],
      ['Evoked Readout', `m: ${Number(evoked.channel_m_norm || 0).toFixed(3)} · d: ${Number(evoked.channel_d_norm || 0).toFixed(3)} · α: ${Number(evoked.gate_alpha || 1).toFixed(2)}`],
      ['MPC Value Guidance', s.config?.value_guided_planning ? 'Value critic enabled' : 'Reward rollout only']
    ];
    $('plasticity-hud').replaceChildren(...plastMetrics.map(([k, val]) => {
      const d = document.createElement('div');
      d.className = 'kv';
      const name = document.createElement('span'), v = document.createElement('b');
      name.textContent = k;
      v.textContent = val;
      d.append(name, v);
      return d;
    }));
  }

  const selfMod = s.telemetry?.self_model || {};
  if ($('self-model-badge')) {
    const authorActive = (selfMod.mean_self_authorship || 0) > 0.4;
    $('self-model-badge').textContent = authorActive ? 'High authorship' : 'Low / Exogenous';
    setTone($('self-model-badge'), authorActive ? 'good' : 'warn');
  }
  if ($('self-model-hud')) {
    const selfMetrics = [
      ['Authorship Index A_t', `${Number(selfMod.mean_self_authorship || 0).toFixed(3)}`],
      ['Disturbance E_t', `${Number(selfMod.mean_external_disturbance || 0).toFixed(3)}`],
      ['Actuator Slip F_t', `${Number(selfMod.mean_actuator_fault || 0).toFixed(3)}`],
      ['Self-Caused Ratio', `${Number((selfMod.self_caused_ratio || 0) * 100).toFixed(1)}%`],
      ['Comparator Mode', selfMod.control_mode || 'active']
    ];
    $('self-model-hud').replaceChildren(...selfMetrics.map(([k, val]) => {
      const d = document.createElement('div');
      d.className = 'kv';
      const name = document.createElement('span'), v = document.createElement('b');
      name.textContent = k;
      v.textContent = val;
      d.append(name, v);
      return d;
    }));
  }
  if ($('uncertainty-hud')) {
    const unc = s.telemetry?.uncertainty || {};
    const plast = s.telemetry?.plasticity_monitor || {};
    const dpo = s.telemetry?.verifiable_dpo || {};
    const conf = unc.confidence !== undefined ? `${(Number(unc.confidence) * 100).toFixed(1)}%` : 'Calibrated';
    if ($('uncertainty-badge')) {
      $('uncertainty-badge').textContent = `Conf ${conf} · ${plast.recommended_action || 'HEALTHY'}`;
      setTone($('uncertainty-badge'), plast.is_collapsed ? 'bad' : 'good');
    }
    const uncertMetrics = [
      ['Epistemic Disagreement', unc.epistemic !== undefined ? `σ²_ep=${Number(unc.epistemic).toFixed(5)} · ${unc.ensemble_size || 3} members` : 'Ensemble active · 3 members'],
      ['Aleatoric Variance', unc.aleatoric !== undefined ? `σ²_al=${Number(unc.aleatoric).toFixed(5)} (Gaussian NLL)` : 'Heteroscedastic Gaussian NLL'],
      ['OOD Support Distance', unc.ood_score !== undefined ? `d_M=${Number(unc.ood_score).toFixed(4)} (${unc.ood_support_samples || 0} samples)` : 'Mahalanobis empirical buffer'],
      ['Representation Rank', plast.effective_rank != null ? `erank=${Number(plast.effective_rank).toFixed(2)} · dead=${((plast.dead_neuron_ratio || 0) * 100).toFixed(1)}%` : `Warmup (${plast.details?.samples || 0}/10 samples)`],
      ['Verifiable DPO Sidecar', dpo.enabled ? `Pairs=${dpo.preference_pairs_count || 0} · updates=${dpo.dpo_updates || 0}` : 'Standby (enable in config)']
    ];
    $('uncertainty-hud').replaceChildren(...uncertMetrics.map(([k, val]) => {
      const d = document.createElement('div');
      d.className = 'kv';
      const name = document.createElement('span'), v = document.createElement('b');
      name.textContent = k;
      v.textContent = val;
      d.append(name, v);
      return d;
    }));
  }

  // Dendritic Liquid Time-Constant (LTC) Sub-Graft HUD
  const dend = s.telemetry?.dendritic_ltc || {};
  if ($('dendritic-badge')) {
    $('dendritic-badge').textContent = dend.enabled ? `Active · τ=${Number(dend.mean_tau || 1).toFixed(2)}` : 'Standby';
    setTone($('dendritic-badge'), dend.enabled ? 'good' : 'muted');
  }
  if ($('dendritic-hud')) {
    const dendMetrics = [
      ['Compartment Status', dend.enabled ? `Active · ${dend.compartments || 32} two-compartment units` : 'Disabled (enable in Session Config)'],
      ['Mean Liquid Time-Constant τ(t)', dend.enabled ? `${Number(dend.mean_tau || 0).toFixed(4)} [${Number(dend.min_tau || 0).toFixed(2)}..${Number(dend.max_tau || 0).toFixed(2)}]` : '—'],
      ['NMDA Apical Plateau Gate', dend.enabled ? `${Number(dend.apical_gate_mean || 0).toFixed(4)} · coincidence=${Number(dend.coincidence_gain || 0).toFixed(4)}` : '—'],
      ['Plateau Active Fraction', dend.enabled ? `${(Number(dend.plateau_fraction || 0) * 100).toFixed(1)}% compartments` : '—'],
      ['Dendritic Eligibility Updates', dend.enabled ? `${dend.plastic_updates || 0} updates · ||h||=${Number(dend.state_norm || 0).toFixed(3)}` : '0']
    ];
    keyValues($('dendritic-hud'), dendMetrics);
  }

  // Oscillatory Phase-Coherence & Kuramoto Synchrony HUD
  const osc = s.telemetry?.oscillatory_coherence || {};
  if ($('oscillatory-badge')) {
    $('oscillatory-badge').textContent = osc.enabled ? `R=${Number(osc.kuramoto_order_r || 0).toFixed(2)} · ${osc.dominant_band || 'THETA'}` : 'Standby';
    setTone($('oscillatory-badge'), osc.enabled ? 'good' : 'muted');
  }
  if ($('oscillatory-hud')) {
    const oscMetrics = [
      ['Kuramoto Order Parameter R(t)', osc.enabled ? `${Number(osc.kuramoto_order_r || 0).toFixed(4)} (${osc.num_neuropils || 3} neuropils)` : 'Disabled (enable in Session Config)'],
      ['Theta (4–8 Hz) / Beta (13–30 Hz)', osc.enabled ? `θ=${Number(osc.theta_power || 0).toFixed(3)} · β=${Number(osc.beta_power || 0).toFixed(3)}` : '—'],
      ['Gamma Binding Power (40–80 Hz)', osc.enabled ? `γ=${Number(osc.gamma_power || 0).toFixed(3)} · dominant=${osc.dominant_band || 'THETA'}` : '—'],
      ['Cross-Neuropil Coherence', osc.enabled ? `${Number(osc.cross_neuropil_coherence || 0).toFixed(4)}` : '—'],
      ['CTC Routing Gain', osc.enabled ? `×${Number(osc.ctc_routing_gain || 1).toFixed(4)}` : '×1.0000']
    ];
    keyValues($('oscillatory-hud'), oscMetrics);
  }

  // -----------------------------------------------------------------------
  // Frontier Apex Omnibus HUD Status
  // -----------------------------------------------------------------------
  if ($('omnibus-hud') && Date.now() - omnibusLastPoll > 2500) {
    omnibusLastPoll = Date.now();
    fetch('/api/omnibus/status')
      .then(r => r.json())
      .then(omnibusStatus => {
        if ($('omnibus-badge')) {
          if (omnibusStatus.is_completed) {
            $('omnibus-badge').textContent = 'Curriculum Complete';
            setTone($('omnibus-badge'), 'good');
          } else if (omnibusStatus.current_stage_idx > 0 || omnibusStatus.total_steps > 0) {
            $('omnibus-badge').textContent = `Stage ${omnibusStatus.current_stage_idx + 1}/${omnibusStatus.total_stages}: ${omnibusStatus.current_task}`;
            setTone($('omnibus-badge'), 'warn');
          } else {
            $('omnibus-badge').textContent = 'Apex Standby (2,000 N)';
            setTone($('omnibus-badge'), 'purple');
          }
        }
        const stagesDone = (omnibusStatus.stages_completed || []).length;
        const lastReceipt = omnibusStatus.receipt;
        const items = [
          ['Connectome Scale', `${omnibusStatus.total_neurons || 2000} Neurons (667/667/666)`],
          ['Active Curriculum Stage', `${stagesDone}/${omnibusStatus.total_stages || 11} Completed`],
          ['Current Task', omnibusStatus.current_task || 'forage'],
          ['Retention Score', lastReceipt ? `${(lastReceipt.final_backward_retention * 100).toFixed(1)}%` : 'Pending final probe'],
          ['Catastrophic Forgetting', lastReceipt ? `${(lastReceipt.catastrophic_forgetting * 100).toFixed(1)}%` : 'Target < 5%'],
          ['SHA-256 Receipt', lastReceipt ? lastReceipt.cryptographic_sha256.slice(0, 16) + '...' : 'Signed upon completion']
        ];
        $('omnibus-hud').replaceChildren(...items.map(([k, val]) => {
          const d = document.createElement('div');
          d.className = 'kv';
          const name = document.createElement('span'), v = document.createElement('b');
          name.textContent = k;
          v.textContent = val;
          d.append(name, v);
          return d;
        }));
      })
      .catch(() => {});
  }

  const btnOmnibus = $('btn-omnibus-run');
  if (btnOmnibus && !btnOmnibus.dataset.bound) {
    btnOmnibus.dataset.bound = 'true';
    btnOmnibus.addEventListener('click', async () => {
      btnOmnibus.disabled = true;
      btnOmnibus.textContent = 'Running Omnibus Curriculum...';
      try {
        const res = await fetch('/api/omnibus/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ steps_per_task: 15, total_neurons: 2000 })
        });
        const data = await res.json();
        alert(`Omnibus Complete! Stages: ${data.receipt?.stages?.length}, Backward Retention: ${(data.receipt?.final_backward_retention * 100).toFixed(1)}%`);
      } catch (err) {
        alert('Omnibus run error: ' + err.message);
      } finally {
        btnOmnibus.disabled = false;
        btnOmnibus.textContent = '✦ Frontier Apex Omnibus';
        render(await api('state')); 
      }
    });
  }

  if (s.telemetry && signalHistory.at(-1)?.ticks !== s.telemetry.ticks) signalHistory.push(s.telemetry);
  signalHistory = signalHistory.slice(-100);

  graph();
  world();
  signals();
  gauges();
  predictiveView();

  $('connection').textContent = '● CONNECTED / LOOPBACK';
}

async function tick(steps = 1, learn = $('learn').checked, actionVec = null) {
  if (busy) return;
  busy = true;
  try {
    const payload = { steps, learn };
    if (actionVec) payload.action = actionVec;
    render(await api('step', payload));
  } finally {
    busy = false;
  }
}

function updateSimSpeed() {
  tickInterval = Number($('simspeed').value) || 125;
  if (tickTimer) clearInterval(tickTimer);
  tickTimer = setInterval(() => {
    if (running) action(() => tick(Number($('simsteps').value)));
  }, tickInterval);
}

// -------------------------------------------------------------------------
// Navigation & Core Controls
// -------------------------------------------------------------------------
document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tab,nav button').forEach(n => n.classList.remove('active'));
  $(b.dataset.tab).classList.add('active');
  b.classList.add('active');
  $('title').textContent = b.querySelector('span').textContent;
  if (b.dataset.tab === 'overview') {
    window.requestAnimationFrame(() => render(state));
  }
});

$('run').onclick = () => { running = true; $('runstatus').textContent = 'Running'; };
$('pause').onclick = () => { running = false; $('runstatus').textContent = 'Paused'; };
$('step').onclick = () => action(() => tick(1));
$('train').onclick = () => action(() => tick(32, true));
for (const id of ['training-active', 'training-threads']) $(id).onchange = () => { $(id).dataset.dirty = 'true'; };
$('training-apply').onclick = () => action(async () => {
  const snapshot = await api('training/config', { active: $('training-active').checked, threads: Number($('training-threads').value) });
  for (const id of ['training-active', 'training-threads']) delete $(id).dataset.dirty;
  if (snapshot.config.active_training) $('learn').checked = true;
  render(snapshot); await refreshTraining();
});
async function startTrainingJob(kind) {
  running = false; $('runstatus').textContent = 'Paused · preparing candidate'; trainingRequest = true;
  $('training-pretrain').disabled = true; $('rsi-live').disabled = true;
  try {
    if (kind === 'pretrain' && !trainingStatus.graft?.enabled) {
      render(await api('training/config', { active: $('training-active').checked, threads: Number($('training-threads').value) }));
    }
    const payload = kind === 'pretrain' ? { steps: 128, updates: 32, batch_size: Number($('training-batch').value), workers: Number($('training-workers').value) } : { steps: 24, candidates: 2, seed: state.config?.seed ?? 7 };
    const result = await api(kind === 'pretrain' ? 'training/pretrain' : 'rsi/live', payload);
    trackedJobId = result.id || result.job?.id;
    await jobs();
  } finally { trainingRequest = false; $('training-pretrain').disabled = false; $('rsi-live').disabled = false; renderTrainingJob(); }
}
$('training-pretrain').onclick = () => action(() => startTrainingJob('pretrain'));
$('rsi-live').onclick = () => action(() => startTrainingJob('rsi'));
$('training-cancel').onclick = () => action(async () => { if (trackedJobId) { await api('jobs/cancel', { id: trackedJobId }); await jobs(); } });
$('skills-refresh').onclick = () => action(async () => show('skills-result', await api('reusable_skills')));

$('experience-apply').onclick = () => action(async () => {
  running = false;
  render(await api('experience/mode', {
    mode: $('experience-mode').value,
    horizon: Number($('experience-horizon').value),
    value_guided: $('experience-value-guided') ? $('experience-value-guided').checked : false,
    planner_algorithm: $('experience-algorithm') ? $('experience-algorithm').value : 'experience_mpc'
  }));
});

function renderPlanPreview(preview) {
  if (!preview) return;
  $('plan-preview').hidden = false;
  $('plan-step').dataset.step = String(preview.source_step);
  $('plan-step').textContent = `Step ${preview.source_step} · horizon ${preview.horizon} · ${preview.planner_algorithm || 'experience_mpc'}`;
  $('plan-reason').textContent = `${preview.used ? 'Proposed local change' : 'Baseline retained'}: ${preview.reason}. ${preview.evaluations || 0} model evaluations.`;
  $('plan-actions').replaceChildren(...(preview.action_order || ['drive', 'turn', 'reach', 'height', 'grip']).map((name, index) => {
    const row = document.createElement('div'); row.className = 'plan-action';
    const label = document.createElement('span'), values = document.createElement('b');
    label.textContent = name;
    const baseline = Number(preview.baseline[index]), proposed = Number(preview.proposed_action[index]);
    values.textContent = `${baseline.toFixed(3)} → ${proposed.toFixed(3)}`;
    const meter = document.createElement('div'); meter.className = 'plan-meter';
    const mark = document.createElement('meter'); mark.min = -1; mark.max = 1; mark.value = proposed; mark.setAttribute('aria-label', `${name} proposed action`);
    meter.append(mark); row.append(label, meter, values); return row;
  }));
  const candidates = preview.candidates || [];
  const table = document.createElement('table'), head = document.createElement('tr');
  for (const text of ['Model forecast', 'Return', 'Penalized score', 'Uncertainty']) {
    const th = document.createElement('th'); th.textContent = text; head.append(th);
  }
  table.append(head);
  candidates.forEach((candidate, index) => {
    const row = document.createElement('tr');
    for (const value of [candidate.baseline ? 'Repeated baseline' : `Sequence ${index}`,
      candidate.predicted_return, candidate.score, candidate.uncertainty]) {
      const cell = document.createElement('td');
      cell.textContent = typeof value === 'number' ? value.toFixed(4) : value;
      row.append(cell);
    }
    table.append(row);
  });
  $('plan-candidates').replaceChildren(...(candidates.length ? [table] : []));
  show('plan-details', preview);
}

$('experience-preview').onclick = () => action(async () => {
  running = false;
  const preview = await api('experience/preview', {
    horizon: Number($('experience-horizon').value),
    memory: $('experience-mode').value === 'memory',
    value_guided: $('experience-value-guided') ? $('experience-value-guided').checked : false,
    planner_algorithm: $('experience-algorithm') ? $('experience-algorithm').value : 'experience_mpc'
  });
  renderPlanPreview(preview);
});

$('simspeed').onchange = updateSimSpeed;

$('quicktask').onchange = () => action(async () => {
  const task = $('quicktask').value;
  if (!task) return;
  running = false;
  render(await api('reset', { task }));
  signalHistory = [];
});

$('config').onsubmit = e => {
  e.preventDefault();
  running = false;
  const config = { ...state.config, ...Object.fromEntries(new FormData(e.target)) };
  for (const key of ['seed', 'total_neurons', 'microsteps', 'cognition_interval']) config[key] = Number(config[key]);
  config.continual_backprop = !!e.target.elements.continual_backprop?.checked;
  config.fast_weight_memory = !!e.target.elements.fast_weight_memory?.checked;
  config.dormant_lifecycle = !!e.target.elements.dormant_lifecycle?.checked;
  config.latent_reasoning = !!e.target.elements.latent_reasoning?.checked;
  config.value_guided_planning = !!e.target.elements.value_guided_planning?.checked;
  config.self_model = e.target.elements.self_model ? !!e.target.elements.self_model.checked : (state.config?.self_model ?? true);
  config.experimental_cognition = e.target.elements.experimental_cognition ? !!e.target.elements.experimental_cognition.checked : (state.config?.experimental_cognition ?? false);
  config.dendritic_ltc = !!e.target.elements.dendritic_ltc?.checked;
  config.oscillatory_coherence = !!e.target.elements.oscillatory_coherence?.checked;
  config.verifiable_dpo = !!e.target.elements.verifiable_dpo?.checked;
  config.programmed_connectome = e.target.elements.programmed_connectome ? !!e.target.elements.programmed_connectome.checked : (state.config?.programmed_connectome ?? true);
  config.planner_algorithm = e.target.elements.planner_algorithm?.value || state.config?.planner_algorithm || 'experience_mpc';
  config.experience_planning = state.config?.experience_planning ?? false;
  config.surprise_memory = state.config?.surprise_memory ?? true;
  config.experience_horizon = state.config?.experience_horizon ?? 1;
  config.connectome_sources = state.config?.connectome_sources ?? ['flywire', 'manc', 'optic-lobe'];
  action(async () => {
    render(await api('reset', { config }));
    signalHistory = [];
  });
};

// -------------------------------------------------------------------------
// Interactive View Toggles (Embodied World & Neural Substrate)
// -------------------------------------------------------------------------
$('world-view-iso').onclick = () => {
  worldMode = 'iso';
  $('world-view-iso').classList.add('active');
  $('world-view-top').classList.remove('active');
  world();
};
$('world-view-top').onclick = () => {
  worldMode = 'top';
  $('world-view-top').classList.add('active');
  $('world-view-iso').classList.remove('active');
  world();
};

$('graph-view-2d').onclick = () => {
  graphMode = '2d';
  $('graph-view-2d').classList.add('active');
  $('graph-view-3d').classList.remove('active');
  graph();
};
$('graph-view-3d').onclick = () => {
  graphMode = '3d';
  $('graph-view-3d').classList.add('active');
  $('graph-view-2d').classList.remove('active');
  graph();
};
$('graph-autorotate').onclick = () => {
  graphAutoRotate = !graphAutoRotate;
  $('graph-autorotate').classList.toggle('active', graphAutoRotate);
  if (graphAutoRotate) graph();
};

$('graph-source').onchange = () => { graphSource = $('graph-source').value; graph(); };
$('graph-role').onchange = () => { graphRole = $('graph-role').value; graph(); };
for (const id of ['graph-activity', 'graph-nodes', 'graph-edges', 'graph-grafts']) $(id).onchange = graph;
$('graph-camera-reset').onclick = () => { graphRotX = .35; graphRotY = .55; graphZoom = 1; graph(); };
for (const id of ['world-trails', 'world-goals']) $(id).onchange = world;
$('world-clear-trail').onclick = () => { worldTrail = []; world(); };

// -------------------------------------------------------------------------
// Direct Teleoperation & Keyboard Controls
// -------------------------------------------------------------------------
async function executeTeleopAction(turn = 0, thrust = 0, gripToggle = false, armReach = false) {
  if (gripToggle) manualGripState = manualGripState > 0.5 ? 0.0 : 1.0;
  if (armReach) manualReachState = manualReachState > 0.5 ? 0.0 : 0.8;
  const actionVec = [thrust, turn, manualReachState, 0.0, manualGripState];
  await tick(1, $('learn').checked, actionVec);
}

$('teleop-fwd').onclick = () => action(() => executeTeleopAction(0.0, 1.0));
$('teleop-left').onclick = () => action(() => executeTeleopAction(-1.0, 0.2));
$('teleop-right').onclick = () => action(() => executeTeleopAction(1.0, 0.2));
$('teleop-grip').onclick = () => action(() => executeTeleopAction(0.0, 0.0, true));
$('teleop-arm').onclick = () => action(() => executeTeleopAction(0.0, 0.0, false, true));

window.addEventListener('keydown', e => {
  if (['input', 'textarea', 'select'].includes(document.activeElement.tagName.toLowerCase())) return;
  if (e.key === 'w' || e.key === 'ArrowUp') {
    e.preventDefault();
    action(() => executeTeleopAction(0.0, 1.0));
  } else if (e.key === 's' || e.key === 'ArrowDown') {
    e.preventDefault();
    action(() => executeTeleopAction(0.0, -0.6));
  } else if (e.key === 'a' || e.key === 'ArrowLeft') {
    e.preventDefault();
    action(() => executeTeleopAction(-1.0, 0.2));
  } else if (e.key === 'd' || e.key === 'ArrowRight') {
    e.preventDefault();
    action(() => executeTeleopAction(1.0, 0.2));
  } else if (e.key.toLowerCase() === 'e') {
    e.preventDefault(); action(() => executeTeleopAction(0.0, 0.0, false, true));
  } else if (e.key === ' ') {
    e.preventDefault();
    action(() => executeTeleopAction(0.0, 0.0, true));
  }
});

// -------------------------------------------------------------------------
// Mouse Interaction for 3D Connectome Orbit & Tooltip Inspection
// -------------------------------------------------------------------------
const graphCanvas = $('graph');
graphCanvas.addEventListener('pointerdown', event => {
  graphDragging = true; graphDidDrag = false;
  graphLastMouse = { x: event.clientX, y: event.clientY };
  graphCanvas.setPointerCapture(event.pointerId);
});
graphCanvas.addEventListener('pointerup', () => {
  graphDragging = false;
  if (!graphDidDrag && hoveredNode) { selectedNodeId = selectedNodeId === hoveredNode.id ? null : hoveredNode.id; graph(); }
});
graphCanvas.addEventListener('pointercancel', () => { graphDragging = false; });
graphCanvas.addEventListener('pointermove', event => {
  const rect = graphCanvas.getBoundingClientRect(), mx = event.clientX - rect.left, my = event.clientY - rect.top;
  if (graphDragging) {
    const dx = event.clientX - graphLastMouse.x, dy = event.clientY - graphLastMouse.y;
    if (Math.abs(dx) + Math.abs(dy) > 1) graphDidDrag = true;
    if (graphMode === '3d') { graphRotY += dx * .01; graphRotX = Math.max(-1.4, Math.min(1.4, graphRotX + dy * .01)); }
    graphLastMouse = { x: event.clientX, y: event.clientY };
  }
  let closest = null, distance = 14;
  nodeScreenPositions.forEach(point => {
    const d = Math.hypot(point.x - mx, point.y - my);
    if (d < distance) { closest = point.node; distance = d; }
  });
  hoveredNode = closest;
  const tooltip = $('graph-tooltip');
  tooltip.classList.toggle('visible', !!closest && !graphDragging);
  if (closest) tooltip.textContent = `#${closest.id} · ${graphSources()[closest.origin]?.id || 'unknown'} · ${closest.role} · activity ${Number(closest.activity || 0).toFixed(4)}`;
  graph();
});
graphCanvas.addEventListener('pointerleave', () => { hoveredNode = null; $('graph-tooltip').classList.remove('visible'); graph(); });
graphCanvas.addEventListener('wheel', event => { event.preventDefault(); graphZoom = Math.max(.5, Math.min(2.5, graphZoom * (event.deltaY > 0 ? .92 : 1.08))); graph(); }, { passive: false });
let lastOrbitTime = 0;
function orbitFrame(time) {
  if (graphAutoRotate && graphMode === '3d' && !graphDragging && $('overview').classList.contains('active') && time - lastOrbitTime > 40) {
    graphRotY += .006; graph(); lastOrbitTime = time;
  }
  requestAnimationFrame(orbitFrame);
}
requestAnimationFrame(orbitFrame);

// -------------------------------------------------------------------------
// Capabilities Explorer
// -------------------------------------------------------------------------
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

// -------------------------------------------------------------------------
// Weight Programs & Numeric Patches
// -------------------------------------------------------------------------
$('tensors').onclick = () => action(async () => show('surfaces', await api('tensors')));
$('makepatch').onclick = () => action(async () => {
  $('packet').value = pretty(await api('patch/make', { edits: JSON.parse($('edits').value), description: $('description').value }));
});
$('previewpatch').onclick = () => action(async () => show('patchresult', await api('patch/preview', { packet: JSON.parse($('packet').value) })));
$('evaluatepatch').onclick = () => action(async () => {
  running = false; $('runstatus').textContent = 'Paused · screening candidate';
  $('evaluatepatch').disabled = true;
  try {
    const result = await api('patch/evaluate', { packet: JSON.parse($('packet').value), steps: 24, seeds: [19, 23], min_gain: 0 });
    show('patchresult', result); receipt = null; render(await api('state')); await refreshTraining();
  } finally { $('evaluatepatch').disabled = false; }
});
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

// -------------------------------------------------------------------------
// Advanced Assays & Invariance Oracles
// -------------------------------------------------------------------------
$('atlas').onclick = () => action(async () => {
  show('atlasresult', { status: 'executing systematic optogenetic silencing & behavioral mapping...' });
  const res = await api('atlas', {});
  show('atlasresult', res);
});

let lastSynthesizedHyperpacket = null;
$('hyperprogram').onclick = () => action(async () => {
  show('hyperresult', { status: 'synthesizing weight-space hypernetwork program from live telemetry...' });
  const res = await api('hyperprogram', {});
  lastSynthesizedHyperpacket = res;
  show('hyperresult', res);
});
$('hyperapply').onclick = () => action(async () => {
  if (!lastSynthesizedHyperpacket) throw Error('Synthesize a hyperprogram packet first');
  const r = await api('patch/apply', { packet: lastSynthesizedHyperpacket });
  receipt = r.receipt_id;
  show('hyperresult', { applied: true, receipt: r });
  render(await api('state'));
});

$('flagship').onclick = () => action(async () => {
  show('flagshipresult', { status: 'running 6-stage flagship developmental connectome assay...' });
  const res = await api('flagship', { training_steps: 15, generations: 3, eval_episodes: 3 });
  show('flagshipresult', res);
});

$('autoprogram').onclick = () => action(async () => {
  show('autorepairresult', { status: 'running autonomous failure diagnosis & repair...' });
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
  show('metamorphicresult', { status: 'running metamorphic invariance oracle suite...' });
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

if ($('diffusion-dpo')) {
  $('diffusion-dpo').onclick = () => action(async () => {
    show('diffusiondporesult', { status: 'running Masked Diffusion trajectory repair & Machine-Verifiable DPO assay...' });
    const res = await api('diffusion_dpo', { horizon: 5 });
    show('diffusiondporesult', res);
    render(await api('state'));
  });
}

if ($('controlled-study-btn')) {
  $('controlled-study-btn').onclick = () => action(async () => {
    show('controlledstudyresult', { status: 'running 6-Arm Continual Ablation Study across sequential stages...' });
    const res = await api('controlled_study', { steps_per_stage: 12, eval_episodes_per_task: 1 });
    show('controlledstudyresult', res);
  });
}

if ($('causal-9arm-btn')) {
  $('causal-9arm-btn').onclick = () => action(async () => {
    show('causal9armresult', { status: 'running 9-Arm Connectome Causal Controls & Unified Deployment Scorecard...' });
    const res = await api('causal_9arm', { steps_per_arm: 12, episodes_per_arm: 1 });
    show('causal9armresult', res);
  });
}

if ($('sweep-capabilities')) {
  $('sweep-capabilities').onclick = () => action(async () => {
    show('capsweepresult', { status: 'sweeping all capability families against live runtime...' });
    const familySweep = {};
    for (const c of catalogItems) {
      const fam = c.family || 'general';
      if (!familySweep[fam]) {
        const cid = c.id || c.capability_id;
        try {
          const r = await api('invoke', { id: cid, payload: c.example || c.example_payload || {} });
          familySweep[fam] = { capability_id: cid, evidence_kind: c.evidence_kind || 'measured', result: r };
        } catch (err) {
          familySweep[fam] = { capability_id: cid, error: err.message };
        }
      }
    }
    show('capsweepresult', { families_verified: Object.keys(familySweep).length, family_sweep: familySweep });
  });
}

if ($('training-intensive')) {
  $('training-intensive').onclick = () => action(async () => {
    running = false;
    $('training-intensive').disabled = true;
    $('training-intensive').textContent = '🔥 Boosting Compute...';
    try {
      const res = await api('training/intensive', {
        steps: 64,
        updates: 24,
        batch_size: Number($('training-batch')?.value || 256),
        online_steps: 12,
        workers: Number($('training-workers')?.value || 4)
      });
      show('training-receipt', res);
      $('learn').checked = true;
      render(await api('state'));
      await refreshTraining();
    } finally {
      $('training-intensive').disabled = false;
      $('training-intensive').textContent = '🔥 Max Compute Boost';
    }
  });
}

function setAllControlsToMaxUI() {
  $('learn').checked = true;
  if ($('simspeed')) { $('simspeed').value = '50'; updateSimSpeed(); }
  if ($('simsteps')) $('simsteps').value = '16';
  if ($('training-threads')) {
    const maxOpt = Math.max(...[...$('training-threads').options].filter(o => !o.disabled).map(o => Number(o.value) || 1));
    $('training-threads').value = String(maxOpt || 8);
  }
  if ($('training-active')) $('training-active').checked = true;
  if ($('training-workers')) $('training-workers').value = '4';
  if ($('training-batch')) $('training-batch').value = '256';
  if ($('graph-nodes')) $('graph-nodes').value = '2000';
  if ($('graph-edges')) $('graph-edges').value = '1600';
  if ($('graph-grafts')) $('graph-grafts').checked = true;
  if ($('experience-mode')) $('experience-mode').value = 'memory';
  if ($('experience-horizon')) $('experience-horizon').value = '5';
  if ($('experience-algorithm')) $('experience-algorithm').value = 'hybrid';
  if ($('experience-value-guided')) $('experience-value-guided').checked = true;

  const form = $('config');
  if (form) {
    if (form.elements.total_neurons && Number(form.elements.total_neurons.value) < 2000) {
      form.elements.total_neurons.value = '2000';
    }
    if (form.elements.microsteps) form.elements.microsteps.value = '4';
    if (form.elements.cognition_interval) form.elements.cognition_interval.value = '4';
    if (form.elements.readout) form.elements.readout.value = 'persistent_evoked';
    if (form.elements.planner_algorithm) form.elements.planner_algorithm.value = 'hybrid';
    for (const field of [
      'continual_backprop', 'fast_weight_memory', 'dormant_lifecycle', 'latent_reasoning',
      'value_guided_planning', 'self_model', 'experimental_cognition',
      'dendritic_ltc', 'oscillatory_coherence', 'verifiable_dpo', 'programmed_connectome'
    ]) {
      if (form.elements[field]) form.elements[field].checked = true;
    }
  }
}

if ($('config-max-preset')) {
  $('config-max-preset').onclick = () => {
    setAllControlsToMaxUI();
    $('config').requestSubmit();
  };
}

async function runTandemMaxMode() {
  running = false;
  setAllControlsToMaxUI();
  const banner = $('tandem-banner');
  if (banner) banner.hidden = false;
  const pill = $('tandem-status-pill');
  if (pill) {
    pill.textContent = 'Executing Sections 01–05 in Tandem...';
    setTone(pill, 'warn');
  }
  const btns = [$('btn-tandem-max'), $('btn-tandem-max-toolbar')].filter(Boolean);
  btns.forEach(b => { b.disabled = true; b.textContent = '⚡ Running Tandem Max (Sections 1–5)...'; });
  try {
    const neurons = Number($('config')?.elements?.total_neurons?.value || 2000);
    const task = $('config')?.elements?.task?.value || state.config?.task || 'forage';
    const res = await api('tandem/max', {
      total_neurons: Math.max(2000, neurons),
      task,
      pretrain_steps: 32,
      pretrain_updates: 16,
      batch_size: 256,
      live_steps: 12,
      omnibus_steps_per_task: 6
    });

    // Populate Section 01 panels
    if (res.section_01) {
      show('training-receipt', res.section_01.pretrain_receipt);
      renderPlanPreview(res.section_01.plan_preview);
    }
    // Populate Section 02 panels
    if (res.section_02) {
      show('capsweepresult', res.section_02);
      show('capresult', res.section_02.representative_invocation);
    }
    // Populate Section 03 panels
    if (res.section_03) {
      show('surfaces', res.section_03.tensors);
      lastSynthesizedHyperpacket = res.section_03.hyperprogram;
      if (res.section_03.hyperprogram?.packet) {
        $('packet').value = pretty(res.section_03.hyperprogram.packet);
      }
      show('hyperresult', res.section_03.hyperprogram);
      show('patchresult', { preview: res.section_03.patch_preview, screen: res.section_03.patch_screen });
      show('atlasresult', res.section_03.atlas);
      show('autorepairresult', res.section_03.autoprogram);
      show('transplantresult', res.section_03.transplant);
      show('flagshipresult', res.section_03.flagship);
      show('metamorphicresult', res.section_03.metamorphic);
      show('spectralresult', res.section_03.spectral);
      show('diffusiondporesult', res.section_03.diffusion_dpo);
    }
    // Populate Section 04 panels
    if (res.section_04) {
      show('rsi-receipt', res.section_04.live_rsi);
      show('skills-result', res.section_04.reusable_skills);
      show('controlledstudyresult', res.section_04.six_arm_study);
      show('causal9armresult', res.section_04.causal_9arm_and_scorecard);
      show('studyresult', res.section_04.study_receipt);
      lineage(res.section_04.study_receipt?.lineage || []);
    }
    // Populate Section 05 panels
    if (res.section_05) {
      $('checkpointname').value = 'tandem_max_session';
      show('checkpointresult', res.section_05.saved_checkpoint);
      await checkpoints();
    }

    show('tandem-receipt-out', res);
    if ($('tandem-summary')) {
      $('tandem-summary').textContent = `Completed all 5 sections in ${res.total_wallclock_seconds}s · ${res.total_neurons} neurons · ${res.compute_threads} CPU threads · ${res.parallel_workers} rollout workers · SHA-256 ${String(res.tandem_sha256 || '').slice(0, 16)}…`;
    }
    if (pill) {
      pill.textContent = `Completed · ${res.total_wallclock_seconds}s`;
      setTone(pill, 'good');
    }
    if ($('tandem-section-pills')) {
      [...$('tandem-section-pills').children].forEach(el => setTone(el, 'good'));
    }

    if (res.state) render(res.state);
    await refreshTraining();
    omnibusLastPoll = 0;
    running = true;
    $('runstatus').textContent = 'Running · Tandem Max Active';
  } finally {
    if ($('btn-tandem-max')) {
      $('btn-tandem-max').disabled = false;
      $('btn-tandem-max').textContent = '⚡ Run Complete System in Tandem (Max Mode · Sections 1–5)';
    }
    if ($('btn-tandem-max-toolbar')) {
      $('btn-tandem-max-toolbar').disabled = false;
      $('btn-tandem-max-toolbar').textContent = '⚡ Max Mode (Sections 1–5)';
    }
  }
}

if ($('btn-tandem-max')) $('btn-tandem-max').onclick = () => action(runTandemMaxMode);
if ($('btn-tandem-max-toolbar')) $('btn-tandem-max-toolbar').onclick = () => action(runTandemMaxMode);

// -------------------------------------------------------------------------
// Experiments, Jobs & Lineage
// -------------------------------------------------------------------------
async function jobs() {
  const result = await api('jobs');
  latestJobs = result.jobs || [];
  renderTrainingJob();
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
    d.append(b, document.createTextNode(' · ' + (j.kind || 'study') + ' · ' + j.id.slice(0, 12)));
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
      trackedJobId = j.id; renderTrainingJob();
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

// -------------------------------------------------------------------------
// Checkpoints
// -------------------------------------------------------------------------
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

// -------------------------------------------------------------------------
// Zero-Warping Resize Observer & Startup
// -------------------------------------------------------------------------
if (window.ResizeObserver) {
  const resizeObserver = new ResizeObserver(() => {
    if ($('overview') && $('overview').classList.contains('active')) {
      window.requestAnimationFrame(() => render(state));
    }
  });
  document.querySelectorAll('canvas').forEach(c => resizeObserver.observe(c));
} else {
  window.addEventListener('resize', () => {
    if ($('overview') && $('overview').classList.contains('active')) {
      render(state);
    }
  });
}

action(async () => {
  render(await api('state'));
  catalogItems = items(await api('capabilities'));
  renderCatalog();
  await checkpoints();
  await jobs();
  await refreshTraining();
  await refreshInventory();
  show('skills-result', await api('reusable_skills'));
});

updateSimSpeed();
setInterval(() => action(jobs), 3000);
