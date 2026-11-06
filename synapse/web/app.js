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
let filterNeuropil = null;  // null | 0 | 1 | 2
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

function graph() {
  try {
    const [x, w, h] = canvas('graph');
    if (!x) return;

    const g = state.graph || {};
    const nodes = g.nodes || [];
    nodeScreenPositions.clear();

    if (!nodes.length) {
      x.fillStyle = '#8b9daf';
      x.font = '11px Consolas, monospace';
      x.fillText('Initializing connectome substrate…', 20, 35);
      return;
    }

    const neuropilColors = ['#5de0c1', '#edbc75', '#c084fc'];
    const neuropilNames = ['FlyWire Central Brain', 'MANC Nerve Cord', 'Optic Lobe'];

    const cx = w * 0.5;
    const cy = h * 0.5;

    if (graphMode === '3d') {
      // -------------------------------------------------------------
      // 3D Anatomical Neuropil Observatory View
      // -------------------------------------------------------------
      if (graphAutoRotate) {
        graphRotY += 0.008;
      }

      const cosY = Math.cos(graphRotY), sinY = Math.sin(graphRotY);
      const cosX = Math.cos(graphRotX), sinX = Math.sin(graphRotX);
      const scale3d = Math.min(w, h) * 0.36;

      const projected = nodes.map(n => {
        const origin = (n.origin || 0) % 3;
        // Anatomical cluster base offsets matching NeuralObservatory
        const baseOffset = [
          [0.0, -0.45, 0.0],
          [0.0, 0.55, 0.0],
          [-0.75, -0.2, 0.25]
        ][origin] || [0, 0, 0];

        // Seeded spherical dispersion
        const seedVal = (n.id * 1013904223 + 1664525) >>> 0;
        const theta = (seedVal % 1000) / 1000 * Math.PI * 2;
        const phi = ((seedVal >>> 10) % 1000) / 1000 * Math.PI - Math.PI / 2;
        const rad = 0.08 + (((seedVal >>> 20) % 1000) / 1000) * 0.25;

        const lx = baseOffset[0] + rad * Math.cos(phi) * Math.cos(theta);
        const ly = baseOffset[1] + rad * Math.sin(phi);
        const lz = baseOffset[2] + rad * Math.cos(phi) * Math.sin(theta);

        // Rotate Y then X
        const x1 = lx * cosY - lz * sinY;
        const z1 = lx * sinY + lz * cosY;
        const y1 = ly * cosX - z1 * sinX;
        const z2 = ly * sinX + z1 * cosX;

        const persp = 3.0 / (3.0 + z2);
        const sx = cx + x1 * scale3d * persp;
        const sy = cy + y1 * scale3d * persp;

        nodeScreenPositions.set(n.id, { x: sx, y: sy, node: n, z: z2 });
        return { id: n.id, sx, sy, z: z2, node: n, origin };
      });

      // Depth sorting
      projected.sort((a, b) => b.z - a.z);

      // Render edges
      x.lineWidth = 0.5;
      (g.edges || []).forEach(e => {
        const p1 = nodeScreenPositions.get(e[0]);
        const p2 = nodeScreenPositions.get(e[1]);
        if (!p1 || !p2) return;
        if (filterNeuropil !== null && (p1.node.origin !== filterNeuropil || p2.node.origin !== filterNeuropil)) return;

        const isHovered = hoveredNode && (hoveredNode.id === e[0] || hoveredNode.id === e[1]);
        const weight = e[2] || 0;

        if (isHovered) {
          x.lineWidth = 1.6;
          x.strokeStyle = weight < 0 ? 'rgba(236,124,141,0.9)' : 'rgba(93,224,193,0.9)';
        } else {
          x.lineWidth = 0.5;
          x.strokeStyle = weight < 0 ? 'rgba(236,124,141,0.14)' : 'rgba(93,224,193,0.12)';
        }
        x.beginPath();
        x.moveTo(p1.x, p1.y);
        x.lineTo(p2.x, p2.y);
        x.stroke();
      });

      // Render nodes
      projected.forEach(p => {
        if (filterNeuropil !== null && p.origin !== filterNeuropil) return;
        const n = p.node;
        const act = Math.max(0, Math.min(1, Math.abs(n.activity || 0)));
        const col = neuropilColors[p.origin];
        const isHovered = hoveredNode && hoveredNode.id === n.id;
        const nodeRadius = (isHovered ? 5.5 : 2.5) + act * 3.5;

        // Firing glow
        if (act > 0.15 || isHovered) {
          const glow = x.createRadialGradient(p.sx, p.sy, 0, p.sx, p.sy, nodeRadius * 2.2);
          glow.addColorStop(0, col);
          glow.addColorStop(1, 'transparent');
          x.fillStyle = glow;
          x.beginPath();
          x.arc(p.sx, p.sy, nodeRadius * 2.2, 0, Math.PI * 2);
          x.fill();
        }

        x.fillStyle = isHovered ? '#ffffff' : col;
        x.beginPath();
        x.arc(p.sx, p.sy, nodeRadius, 0, Math.PI * 2);
        x.fill();
      });

      x.fillStyle = '#647c90';
      x.font = '9px Consolas, monospace';
      x.fillText('3D Anatomical Observatory · Drag to rotate connectome', 14, h - 10);

    } else {
      // -------------------------------------------------------------
      // 2D Isotropic Neuropil Substrate View
      // -------------------------------------------------------------
      const clusterRadius = Math.min(w, h) * 0.22;
      const clusterCenters = [
        [cx, cy - clusterRadius * 0.65],                 // FlyWire top
        [cx, cy + clusterRadius * 0.75],                 // MANC bottom
        [cx - clusterRadius * 1.05, cy - clusterRadius * 0.05] // Optic left
      ];

      // Group nodes by origin
      const clusters = [[], [], []];
      nodes.forEach(n => {
        const origin = (n.origin || 0) % 3;
        clusters[origin].push(n);
      });

      clusters.forEach((cNodes, oIdx) => {
        const [ccx, ccy] = clusterCenters[oIdx];
        const count = cNodes.length;
        cNodes.forEach((n, i) => {
          const angle = i * 2.399963; // Golden ratio spiral
          const dist = 12 + Math.sqrt((i + 1) / Math.max(1, count)) * clusterRadius * 0.82;
          const nx = ccx + Math.cos(angle) * dist;
          const ny = ccy + Math.sin(angle) * dist;
          nodeScreenPositions.set(n.id, { x: nx, y: ny, node: n });
        });
      });

      // Cluster boundary hulls / halos
      clusterCenters.forEach(([ccx, ccy], oIdx) => {
        if (filterNeuropil !== null && filterNeuropil !== oIdx) return;
        x.strokeStyle = 'rgba(255,255,255,0.04)';
        x.lineWidth = 1;
        x.setLineDash([3, 4]);
        x.beginPath();
        x.arc(ccx, ccy, clusterRadius * 0.9, 0, Math.PI * 2);
        x.stroke();
        x.setLineDash([]);

        x.fillStyle = 'rgba(255,255,255,0.2)';
        x.font = '9px Consolas, monospace';
        x.fillText(neuropilNames[oIdx], ccx - 45, ccy - clusterRadius * 0.95);
      });

      // Synaptic and graft edges
      (g.edges || []).forEach(e => {
        const p1 = nodeScreenPositions.get(e[0]);
        const p2 = nodeScreenPositions.get(e[1]);
        if (!p1 || !p2) return;
        if (filterNeuropil !== null && (p1.node.origin !== filterNeuropil || p2.node.origin !== filterNeuropil)) return;

        const isHovered = hoveredNode && (hoveredNode.id === e[0] || hoveredNode.id === e[1]);
        const weight = e[2] || 0;

        if (isHovered) {
          x.lineWidth = 1.8;
          x.strokeStyle = weight < 0 ? 'rgba(236,124,141,0.95)' : 'rgba(93,224,193,0.95)';
        } else {
          x.lineWidth = 0.55;
          x.strokeStyle = weight < 0 ? 'rgba(236,124,141,0.16)' : 'rgba(93,224,193,0.13)';
        }
        x.beginPath();
        x.moveTo(p1.x, p1.y);
        x.lineTo(p2.x, p2.y);
        x.stroke();
      });

      // Active neurons with firing rate glow
      nodes.forEach(n => {
        const p = nodeScreenPositions.get(n.id);
        if (!p) return;
        const origin = (n.origin || 0) % 3;
        if (filterNeuropil !== null && origin !== filterNeuropil) return;

        const act = Math.max(0, Math.min(1, Math.abs(n.activity || 0)));
        const col = neuropilColors[origin];
        const isHovered = hoveredNode && hoveredNode.id === n.id;
        const nodeRadius = (isHovered ? 5.5 : 2.5) + act * 3.5;

        if (act > 0.1 || isHovered) {
          const glow = x.createRadialGradient(p.x, p.y, 0, p.x, p.y, nodeRadius * 2.4);
          glow.addColorStop(0, col);
          glow.addColorStop(1, 'transparent');
          x.fillStyle = glow;
          x.beginPath();
          x.arc(p.x, p.y, nodeRadius * 2.4, 0, Math.PI * 2);
          x.fill();
        }

        x.fillStyle = isHovered ? '#ffffff' : col;
        x.beginPath();
        x.arc(p.x, p.y, nodeRadius, 0, Math.PI * 2);
        x.fill();
      });

      x.fillStyle = '#647c90';
      x.font = '9px Consolas, monospace';
      x.fillText('2D Isotropic Substrate · Hover node to inspect synaptic paths', 14, h - 10);
    }

    // Active hovered tooltip
    if (hoveredNode) {
      const p = nodeScreenPositions.get(hoveredNode.id);
      if (p) {
        x.strokeStyle = '#ffffff';
        x.lineWidth = 1.5;
        x.beginPath();
        x.arc(p.x, p.y, 8, 0, Math.PI * 2);
        x.stroke();
      }
    }

  } catch (err) {
    console.error('graph render error:', err);
  }
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
        const ang = Math.atan2(tz.vy || 0, tz.vx || 0);
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
      x.arc(p[0], p[1], beamLen, ang - halfAngle, ang + halfAngle);
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

// -------------------------------------------------------------------------
// Master Render Loop
// -------------------------------------------------------------------------
function render(s) {
  if (!s) return;
  state = s;
  show('snapshot', s);
  const neuronInput = $('config').elements.namedItem('total_neurons');
  const sourceCount = s.config?.connectome_sources?.length || 3;
  const sizeKey = JSON.stringify([s.config?.total_neurons, s.config?.nodes_per_source, sourceCount]);
  if (neuronInput && neuronInput.dataset.sizeKey !== sizeKey) {
    neuronInput.value = String(s.config?.total_neurons ?? s.graph?.neurons ?? 2000);
    neuronInput.min = String(16 * sourceCount);
    neuronInput.max = String(5000 * sourceCount);
    neuronInput.dataset.sizeKey = sizeKey;
  }

  const experience = s.experience || {};
  $('experience-mode').value = experience.mode || 'off';
  $('experience-horizon').value = String(s.config?.experience_horizon || 1);
  if ($('experience-value-guided')) {
    $('experience-value-guided').checked = !!s.config?.value_guided_planning;
  }
  const configForm = $('config');
  if (configForm) {
    for (const boolField of ['continual_backprop', 'fast_weight_memory', 'dormant_lifecycle', 'latent_reasoning', 'value_guided_planning']) {
      const el = configForm.elements.namedItem(boolField);
      if (el && !el.dataset.userTouched) {
        if (boolField === 'dormant_lifecycle') {
          el.checked = s.config?.[boolField] !== false;
        } else {
          el.checked = !!s.config?.[boolField];
        }
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
    $('thrust-governor').style.color = gov < 50 ? '#ec7c8d' : (gov < 90 ? '#edbc75' : '#00ffb4');
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
      const track = document.createElement('div');
      track.className = 'vital-track';
      const fill = document.createElement('div');
      fill.className = 'vital-fill';
      fill.style.width = `${pct}%`;
      fill.style.background = v.col;
      track.appendChild(fill);
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
      $('mimomix-badge').style.background = 'rgba(255,100,200,0.25)';
      $('mimomix-badge').style.color = '#ff64c8';
    } else {
      $('mimomix-badge').style.background = 'rgba(0,255,180,0.15)';
      $('mimomix-badge').style.color = '#00ffb4';
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
    $('dormant-badge').style.color = isEnabled ? '#00ffb4' : '#ec7c8d';
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
    $('plasticity-badge').style.color = plastActive ? '#00ffb4' : '#8892b0';
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
    $('self-model-badge').style.color = authorActive ? '#00ffb4' : '#ffaa00';
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
    const uncertMetrics = [
      ['Epistemic Disagreement', 'Ensemble active · 4 members'],
      ['Aleatoric Variance', 'Heteroscedastic Gaussian NLL'],
      ['OOD Support', 'Mahalanobis empirical buffer'],
      ['Supermix Fabric', 'v2026.11 · Dale-compliant'],
      ['Representation Rank', 'Plasticity monitor healthy']
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

  // -----------------------------------------------------------------------
  // Frontier Apex Omnibus HUD Status
  // -----------------------------------------------------------------------
  if ($('omnibus-hud')) {
    fetch('/api/omnibus/status')
      .then(r => r.json())
      .then(omnibusStatus => {
        if ($('omnibus-badge')) {
          if (omnibusStatus.is_completed) {
            $('omnibus-badge').textContent = 'Curriculum Complete';
            $('omnibus-badge').style.color = '#00ffb4';
          } else if (omnibusStatus.current_stage_idx > 0 || omnibusStatus.total_steps > 0) {
            $('omnibus-badge').textContent = `Stage ${omnibusStatus.current_stage_idx + 1}/${omnibusStatus.total_stages}: ${omnibusStatus.current_task}`;
            $('omnibus-badge').style.color = '#ffd166';
          } else {
            $('omnibus-badge').textContent = 'Apex Standby (2,000 N)';
            $('omnibus-badge').style.color = '#c77dff';
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
        poll();
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
    if (running) action(() => tick(4));
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
$('experience-apply').onclick = () => action(async () => {
  running = false;
  render(await api('experience/mode', {
    mode: $('experience-mode').value,
    horizon: Number($('experience-horizon').value),
    value_guided: $('experience-value-guided') ? $('experience-value-guided').checked : false
  }));
});
$('experience-preview').onclick = () => action(async () => {
  running = false;
  const preview = await api('experience/preview', {
    horizon: Number($('experience-horizon').value),
    memory: $('experience-mode').value === 'memory',
    value_guided: $('experience-value-guided') ? $('experience-value-guided').checked : false
  });
  $('plan-preview').hidden = false;
  $('plan-step').dataset.step = String(preview.source_step);
  $('plan-step').textContent = `Step ${preview.source_step} · horizon ${preview.horizon}`;
  $('plan-reason').textContent = `${preview.used ? 'Proposed local change' : 'Baseline retained'}: ${preview.reason}. ${preview.evaluations || 0} model evaluations.`;
  $('plan-actions').replaceChildren(...preview.action_order.map((name, index) => {
    const row = document.createElement('div'); row.className = 'plan-action';
    const label = document.createElement('span'), values = document.createElement('b');
    label.textContent = name;
    const baseline = Number(preview.baseline[index]), proposed = Number(preview.proposed_action[index]);
    values.textContent = `${baseline.toFixed(3)} → ${proposed.toFixed(3)}`;
    const meter = document.createElement('div'); meter.className = 'plan-meter';
    const mark = document.createElement('i'); mark.style.left = `${(proposed + 1) * 50}%`;
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
  const config = Object.fromEntries(new FormData(e.target));
  for (const key of ['seed', 'total_neurons', 'microsteps', 'cognition_interval']) config[key] = Number(config[key]);
  config.continual_backprop = !!e.target.elements.continual_backprop?.checked;
  config.fast_weight_memory = !!e.target.elements.fast_weight_memory?.checked;
  config.dormant_lifecycle = !!e.target.elements.dormant_lifecycle?.checked;
  config.latent_reasoning = !!e.target.elements.latent_reasoning?.checked;
  config.value_guided_planning = !!e.target.elements.value_guided_planning?.checked;
  config.experience_planning = state.config?.experience_planning ?? false;
  config.surprise_memory = state.config?.surprise_memory ?? true;
  config.experience_horizon = state.config?.experience_horizon ?? 1;
  config.experimental_cognition = state.config?.experimental_cognition ?? false;
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

// Neuropil legend filters
$('filter-flywire').onclick = () => { filterNeuropil = 0; graph(); };
$('filter-manc').onclick = () => { filterNeuropil = 1; graph(); };
$('filter-optic').onclick = () => { filterNeuropil = 2; graph(); };
$('filter-reset').onclick = () => { filterNeuropil = null; graph(); };

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
  } else if (e.key === ' ') {
    e.preventDefault();
    action(() => executeTeleopAction(0.0, 0.0, true));
  }
});

// -------------------------------------------------------------------------
// Mouse Interaction for 3D Connectome Orbit & Tooltip Inspection
// -------------------------------------------------------------------------
const graphCanvas = $('graph');
graphCanvas.addEventListener('mousedown', e => {
  graphDragging = true;
  graphLastMouse = { x: e.clientX, y: e.clientY };
});
window.addEventListener('mouseup', () => { graphDragging = false; });
window.addEventListener('mousemove', e => {
  if (!graphDragging) return;
  const dx = e.clientX - graphLastMouse.x;
  const dy = e.clientY - graphLastMouse.y;
  graphRotY += dx * 0.01;
  graphRotX += dy * 0.01;
  graphLastMouse = { x: e.clientX, y: e.clientY };
  if (graphMode === '3d') graph();
});

graphCanvas.addEventListener('mousemove', e => {
  const rect = graphCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  let closest = null;
  let minDist = 14;

  nodeScreenPositions.forEach((pos, id) => {
    const dist = Math.hypot(pos.x - mx, pos.y - my);
    if (dist < minDist) {
      minDist = dist;
      closest = pos.node;
    }
  });

  hoveredNode = closest;
  const tooltip = $('graph-tooltip');
  if (closest && tooltip) {
    const neuropils = ['FlyWire Central Brain', 'MANC Nerve Cord', 'Optic Lobe'];
    tooltip.style.display = 'block';
    tooltip.style.left = (mx + 12) + 'px';
    tooltip.style.top = (my + 12) + 'px';
    tooltip.textContent = `Neuron #${closest.id} · ${neuropils[closest.origin || 0]} · Rate: ${(closest.activity || 0).toFixed(4)} · Role: ${closest.role || 'relay'}`;
  } else if (tooltip) {
    tooltip.style.display = 'none';
  }
  graph();
});

graphCanvas.addEventListener('mouseleave', () => {
  hoveredNode = null;
  const tooltip = $('graph-tooltip');
  if (tooltip) tooltip.style.display = 'none';
  graph();
});

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

// -------------------------------------------------------------------------
// Experiments, Jobs & Lineage
// -------------------------------------------------------------------------
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
});

updateSimSpeed();
setInterval(() => action(jobs), 3000);
