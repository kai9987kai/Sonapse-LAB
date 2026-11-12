"""Portable, offline snapshot export and bounded replay capture.

The exported HTML is deliberately an inspector, not a browser-side simulator.
The exact native checkpoint is embedded unchanged; Python/PyTorch state cannot
be executed faithfully by a browser-only runtime in this project today.
"""
from __future__ import annotations

import base64
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

import numpy as np


PORTABLE_SCHEMA = "sonapse-portable-html-v1"
REPLAY_SCHEMA = 1
DEFAULT_REPLAY_FRAMES = 2048
DEFAULT_REPLAY_NEURONS = 128
_CHUNK_BYTES = 48 * 1024
_MAX_NATIVE_CHECKPOINT_BYTES = 1024 ** 3


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _json_for_script(value) -> str:
    text = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    # JSON is inert in this script type, but still escape HTML-sensitive
    # characters to prevent a checkpoint string from closing its script node.
    return (text.replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026").replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


class PortableReplayRecorder:
    """Keep a bounded, read-only activity trace for offline visual replay."""

    def __init__(self, runtime=None, max_frames: int = DEFAULT_REPLAY_FRAMES,
                 max_neurons: int = DEFAULT_REPLAY_NEURONS):
        self.max_frames = max(1, int(max_frames))
        self.max_neurons = max(1, int(max_neurons))
        self._lock = threading.RLock()
        self.runtime = None
        self.frames = deque(maxlen=self.max_frames)
        self.neuron_ids = np.empty(0, dtype=np.int64)
        self.history_start_step = 0
        self.started_at = time.time()
        if runtime is not None:
            self.bind(runtime)

    def bind(self, runtime, *, reset: bool = False) -> None:
        with self._lock:
            self.runtime = runtime
            brain = getattr(getattr(runtime, "agent", None), "brain", None)
            count = int(getattr(brain, "n", 0) or 0)
            self.neuron_ids = (np.linspace(0, count - 1,
                min(count, self.max_neurons)).astype(np.int64)
                if count > 0 else np.empty(0, dtype=np.int64))
            if reset:
                self.frames.clear()
                self.history_start_step = int(getattr(runtime, "total_steps", 0)) + 1
                self.started_at = time.time()
            elif not self.frames:
                self.history_start_step = int(getattr(runtime, "total_steps", 0)) + 1

    def __deepcopy__(self, memo):
        """Make candidate runtime clones independent and safely non-recording."""
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        cloned = type(self)(runtime=None, max_frames=self.max_frames,
                            max_neurons=self.max_neurons)
        memo[id(self)] = cloned
        with self._lock:
            cloned.neuron_ids = self.neuron_ids.copy()
            cloned.history_start_step = self.history_start_step
            cloned.started_at = self.started_at
            cloned.frames = deque(deepcopy(list(self.frames), memo), maxlen=self.max_frames)
        return cloned

    def __call__(self, previous_observation, action, reward, next_observation, done) -> None:
        # Runtime transition taps execute inside the existing step lock. This
        # observer never writes into the simulation and stores only 128 rates.
        runtime = self.runtime
        if runtime is None or self.neuron_ids.size == 0:
            return
        try:
            rates = runtime.agent.brain.r.detach().cpu().numpy()
            activity = np.asarray(rates[self.neuron_ids], dtype=np.float32)
            frame = {
                "step": int(runtime.total_steps) + 1,
                "action": [float(value) for value in np.asarray(action).reshape(-1)],
                "reward": float(reward),
                "done": bool(done),
                "activity": activity.tolist(),
            }
            with self._lock:
                self.frames.append(frame)
        except Exception:
            # A passive replay recorder must never interfere with the live loop.
            return

    def export_state(self) -> dict:
        with self._lock:
            frames = [dict(frame) for frame in self.frames]
            return {
                "schema_version": REPLAY_SCHEMA,
                "history_start_step": int(self.history_start_step),
                "history_end_step": int(frames[-1]["step"]) if frames else int(self.history_start_step) - 1,
                "sampled_neuron_ids": self.neuron_ids.astype(int).tolist(),
                "frames": frames,
                "max_frames": self.max_frames,
                "truncated": bool(frames and frames[0]["step"] > self.history_start_step),
                "scope": "sampled activity/action/reward trace recorded since this workbench attached",
            }


_HTML_START = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>Sonapse Portable Snapshot</title><style>
:root{color-scheme:dark;--bg:#07111d;--panel:#101d2b;--line:#263a4e;--text:#e7f0f7;--muted:#98adbd;--teal:#5de0c1;--amber:#edbc75;--red:#ff9b87}*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 15% 0,#15354a 0,transparent 48%),var(--bg);color:var(--text);font:14px/1.5 system-ui,Segoe UI,sans-serif}main{max-width:1440px;margin:auto;padding:24px}.head{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}.eyebrow{color:var(--teal);font-weight:700;letter-spacing:.14em;text-transform:uppercase;font-size:11px}h1{margin:4px 0;font-size:clamp(24px,4vw,38px)}p{margin:8px 0;color:var(--muted)}.notice{margin:18px 0;padding:14px 16px;border:1px solid #805e32;background:#251c11;border-radius:12px;color:#f5d7a7}.notice strong{color:#ffd895}.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:14px}.card{background:#0d1a27eF;border:1px solid var(--line);border-radius:14px;padding:16px;min-width:0}.card h2{font-size:16px;margin:0 0 8px}.wide{grid-column:1/-1}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:16px 0}.stat{padding:12px;background:#102233;border:1px solid #20384c;border-radius:10px}.stat b{display:block;font-size:19px;color:var(--teal)}.stat span{color:var(--muted);font-size:12px}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:8px 0 12px}button,select,input[type=range]{accent-color:var(--teal)}button,select{background:#142739;color:var(--text);border:1px solid #35516a;border-radius:8px;padding:8px 12px;cursor:pointer}button:hover:not(:disabled){border-color:var(--teal)}button:disabled{opacity:.55;cursor:not-allowed}canvas{display:block;width:100%;background:#08131e;border:1px solid #203548;border-radius:10px;touch-action:none}#network{height:560px}#timeline{height:160px}.small{font-size:12px;color:var(--muted)}.inspector{white-space:pre-wrap;overflow-wrap:anywhere;background:#091520;border-radius:8px;padding:12px;max-height:420px;overflow:auto}.badge{display:inline-block;border:1px solid #38546c;border-radius:99px;padding:3px 9px;color:#b8d4e5;font-size:11px}.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}footer{margin:20px 0;color:var(--muted);font-size:12px}@media(max-width:900px){main{padding:12px}.grid{grid-template-columns:1fr}.wide{grid-column:auto}#network{height:440px}.head{display:block}}
</style></head><body><main>
<div class="head"><div><div class="eyebrow">Offline artifact · verified checkpoint archive</div><h1>Sonapse Portable Snapshot</h1><p id="subtitle"></p></div><span class="badge">NO NETWORK REQUIRED</span></div>
<div class="notice"><strong>Live continuation is unavailable in this standalone HTML.</strong> The embedded checkpoint is the exact native state archive and can be restored by a compatible Sonapse-LAB Python/PyTorch runtime. This viewer does not simulate or approximate neural updates.</div>
<div id="stats" class="stats"></div>
<div class="grid">
<section class="card"><h2>Recorded neural activity · schematic projection</h2><p class="small">Canvas shows sampled nodes and sampled connections from the real checkpoint view. Layout is an inspection aid, not measured anatomy.</p><div class="controls"><button id="play">▶ Play replay</button><button id="back">← Step back</button><button id="forward">Step forward →</button><button id="reset">Return to last recorded frame</button><input id="timeline-range" type="range" min="0" max="0" value="0" style="flex:1;min-width:150px"><select id="role-filter"><option value="all">All displayed roles</option></select></div><canvas id="network"></canvas><p id="frame-label" class="small"></p></section>
<section class="card"><h2>Selected neuron / connection data</h2><p class="small">Select a displayed neuron. Connection rows and current weights are from the exported snapshot.</p><pre id="inspector" class="inspector">Select a node to inspect its stored properties.</pre><h2 style="margin-top:16px">Export and runtime status</h2><pre id="runtime" class="inspector"></pre><button id="download-checkpoint">Download embedded native checkpoint</button><p class="small">The checkpoint download uses the original content-addressed archive bytes embedded in this HTML.</p></section>
<section class="card wide"><h2>Recorded replay timeline</h2><p id="history-label" class="small"></p><canvas id="timeline"></canvas></section>
<section class="card wide"><h2>Sampled connection weight distribution</h2><p class="small">Distribution is calculated from the displayed connection sample. The checkpoint contains the complete connection state.</p><canvas id="weights"></canvas></section>
<section class="card wide"><h2>Integrity and compatibility</h2><div id="integrity" class="inspector"></div></section>
</div><footer>Sonapse-LAB portable snapshot format v1 · Open this HTML directly from disk. Refreshing returns to the same captured snapshot.</footer>
<script id="portable-meta" type="application/json">"""

_HTML_BETWEEN_JSON_AND_CHECKPOINT = r"""</script><script id="snapshot-data" type="application/json">"""
_HTML_CHECKPOINT_START = r"""</script><script id="replay-data" type="application/json">"""
_HTML_SCRIPT_START = r"""</script><script id="checkpoint-bytes" type="application/octet-stream">
"""
_HTML_SCRIPT_END = r"""
</script><script>
(()=>{'use strict';
const meta=JSON.parse(document.getElementById('portable-meta').textContent);
const snapshot=JSON.parse(document.getElementById('snapshot-data').textContent);
const replay=JSON.parse(document.getElementById('replay-data').textContent);
const encodedNode=document.getElementById('checkpoint-bytes');
const graph=snapshot.graph||{}, nodes=Array.isArray(graph.nodes)?graph.nodes:[], edges=Array.isArray(graph.edges)?graph.edges.slice():[];if(Array.isArray(graph.graft_links))edges.push(...graph.graft_links);
const frames=Array.isArray(replay.frames)?replay.frames:[], neuronIds=Array.isArray(replay.sampled_neuron_ids)?replay.sampled_neuron_ids:[];
const $=id=>document.getElementById(id);let index=Math.max(0,frames.length-1),playing=false,timer=null,zoom=1,panX=0,panY=0,drag=null,selected=null,fpsTime=performance.now(),fpsFrames=0,fps=0;
const roleSet=[...new Set(nodes.map(n=>String(n.role||'unknown')))];for(const role of roleSet){const o=document.createElement('option');o.value=role;o.textContent=role;$('role-filter').appendChild(o)}
const fmt=n=>Number.isFinite(Number(n))?Number(n).toLocaleString():String(n??'unknown');
$('subtitle').textContent=`Step ${fmt(meta.simulation_step)} · ${meta.export_timestamp} · ${meta.task||'task unavailable'}`;
const stats=[['Simulation step',fmt(meta.simulation_step)],['Neurons',fmt(graph.neurons)],['Displayed nodes',fmt(nodes.length)],['Displayed connections',fmt(edges.length)],['Checkpoint size',`${fmt(meta.checkpoint_bytes)} bytes`],['Replay samples',fmt(frames.length)]];
$('stats').innerHTML=stats.map(([k,v])=>`<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');
$('runtime').textContent=JSON.stringify({checkpoint_format:meta.checkpoint_format,checkpoint_sha256:meta.checkpoint_sha256,restore_support:meta.restore_support,portable_engine:meta.portable_engine,history_start_step:replay.history_start_step,history_end_step:replay.history_end_step,history_truncated:replay.truncated,recorded_scope:replay.scope},null,2);
$('integrity').textContent=`Portable format: ${meta.schema}\n\nEmbedded native checkpoint: ${meta.checkpoint_bytes} bytes\nSHA-256: ${meta.checkpoint_sha256}\n\nThe exporter verifies the archive checksum before embedding it and writes the HTML atomically. To verify again outside the viewer, extract the native checkpoint and compare its SHA-256 with the value above.\n\nCompatibility boundary: ${meta.compatibility_note}`;
$('history-label').textContent=frames.length?`Recorded ${frames.length} transitions from step ${replay.history_start_step} to ${replay.history_end_step}${replay.truncated?' (oldest samples were dropped to honor the bounded history limit).':''} Activity is stored for ${neuronIds.length} evenly sampled neurons.`:'No transition samples were available at export time. The checkpoint still contains the full current computational state.';
$('timeline-range').max=String(Math.max(0,frames.length-1));$('timeline-range').value=String(index);
const network=$('network'),ctx=network.getContext('2d');let coords=[],visibleNodes=[];
function resizeCanvas(c){const rect=c.getBoundingClientRect(),dpr=Math.min(2,window.devicePixelRatio||1);c.width=Math.max(1,Math.round(rect.width*dpr));c.height=Math.max(1,Math.round(rect.height*dpr));return dpr}
function activityMap(){const f=frames[index],m=new Map();if(f&&Array.isArray(f.activity))for(let i=0;i<Math.min(neuronIds.length,f.activity.length);i++)m.set(neuronIds[i],f.activity[i]);return m}
function positions(w,h){const cx=w/2+panX,cy=h/2+panY,r=Math.min(w,h)*.42*zoom;return nodes.map((n,i)=>{const a=(i/nodes.length)*Math.PI*2-Math.PI/2;return{x:cx+Math.cos(a)*r,y:cy+Math.sin(a)*r}})}
function paintNetwork(){const dpr=resizeCanvas(network),w=network.width/dpr,h=network.height/dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);coords=positions(w,h);visibleNodes=nodes.map((n,i)=>({n,p:coords[i],i})).filter(x=>$('role-filter').value==='all'||String(x.n.role||'unknown')===$('role-filter').value);const visibleIndexes=new Set(visibleNodes.map(x=>x.i)),lookup=new Map(nodes.map((n,i)=>[Number(n.id),i]));ctx.lineWidth=.65;
for(const e of edges.slice(0,12000)){const a=lookup.get(Number(e[0])),b=lookup.get(Number(e[1]));if(a===undefined||b===undefined||!visibleIndexes.has(a)||!visibleIndexes.has(b))continue;const p=coords[a],q=coords[b],weight=Number(e[2])||0;ctx.strokeStyle=weight>=0?'rgba(93,224,193,.13)':'rgba(255,155,135,.2)';ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke()}
const values=activityMap();for(const x of shown){const v=values.has(Number(x.n.id))?Number(values.get(Number(x.n.id))):Number(x.n.activity||0),r=selected===x.n.id?5.4:2.6+Math.min(2.3,Math.abs(v)*1.8);ctx.beginPath();ctx.arc(x.p.x,x.p.y,r,0,Math.PI*2);ctx.fillStyle=v>=0?'rgba(93,224,193,.45)':'rgba(237,188,117,.55)';ctx.fill();if(selected===x.n.id){ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke()}}
fpsFrames++;const now=performance.now();if(now-fpsTime>700){fps=fpsFrames*1000/(now-fpsTime);fpsFrames=0;fpsTime=now;$('frame-label').textContent=frames[index]?`Replay step ${fmt(frames[index].step)} · reward ${Number(frames[index].reward).toFixed(4)} · redraw rate ${fps.toFixed(1)} Hz`:`Snapshot view · redraw rate ${fps.toFixed(1)} Hz`}}
function paintTimeline(){const c=$('timeline'),cx=c.getContext('2d'),dpr=resizeCanvas(c),w=c.width/dpr,h=c.height/dpr;cx.setTransform(dpr,0,0,dpr,0,0);cx.clearRect(0,0,w,h);cx.strokeStyle='#263a4e';cx.beginPath();cx.moveTo(32,12);cx.lineTo(32,h-24);cx.lineTo(w-8,h-24);cx.stroke();if(!frames.length)return;const vals=frames.map(f=>Number(f.reward)||0),lo=Math.min(...vals),hi=Math.max(...vals),span=hi-lo||1;cx.strokeStyle='#5de0c1';cx.lineWidth=1.4;cx.beginPath();vals.forEach((v,i)=>{const x=32+i*Math.max(1,(w-44)/(vals.length-1||1)),y=12+(h-40)*(1-(v-lo)/span);if(i===0)cx.moveTo(x,y);else cx.lineTo(x,y)});cx.stroke();const fx=32+index*Math.max(1,(w-44)/(vals.length-1||1));cx.strokeStyle='#edbc75';cx.beginPath();cx.moveTo(fx,8);cx.lineTo(fx,h-24);cx.stroke();cx.fillStyle='#98adbd';cx.font='11px system-ui';cx.fillText(`reward ${lo.toFixed(3)} … ${hi.toFixed(3)} · redraw rate ${fps.toFixed(1)} Hz`,36,h-7)}
function paintWeights(){const c=$('weights'),cx=c.getContext('2d'),dpr=resizeCanvas(c),w=c.width/dpr,h=c.height/dpr;cx.setTransform(dpr,0,0,dpr,0,0);cx.clearRect(0,0,w,h);const vals=edges.map(e=>Number(e[2])).filter(Number.isFinite);cx.strokeStyle='#263a4e';cx.beginPath();cx.moveTo(36,12);cx.lineTo(36,h-26);cx.lineTo(w-8,h-26);cx.stroke();if(!vals.length){cx.fillStyle='#98adbd';cx.fillText('No sampled connection weights',44,32);return}const lo=Math.min(...vals),hi=Math.max(...vals),span=hi-lo||1,bins=new Array(24).fill(0);for(const v of vals)bins[Math.min(bins.length-1,Math.floor((v-lo)/span*bins.length))]++;const max=Math.max(...bins,1),plotW=w-52,barW=plotW/bins.length;bins.forEach((n,i)=>{const barH=(h-48)*n/max;cx.fillStyle=i<bins.length/2?'rgba(93,224,193,.8)':'rgba(237,188,117,.8)';cx.fillRect(38+i*barW,h-26-barH,Math.max(1,barW-2),barH)});cx.fillStyle='#98adbd';cx.font='11px system-ui';cx.fillText(`${vals.length} displayed edges · min ${lo.toPrecision(4)} · max ${hi.toPrecision(4)}`,40,h-7)}
function inspect(node){selected=node.id;const rows=edges.filter(e=>Number(e[0])===Number(node.id)||Number(e[1])===Number(node.id)).slice(0,80).map(e=>({source:e[0],target:e[1],weight:e[2]}));$('inspector').textContent=JSON.stringify({id:node.id,origin:node.origin,role:node.role,snapshot_activity:node.activity,recorded_activity:activityMap().get(Number(node.id))??null,incident_displayed_connections:rows,connection_note:'Only sampled graph rows are shown; the embedded checkpoint is the complete native state.'},null,2);paintNetwork()}
function setFrame(i){if(!frames.length)return;index=Math.max(0,Math.min(frames.length-1,i));$('timeline-range').value=String(index);paintNetwork();paintTimeline()}
network.addEventListener('pointerdown',e=>{network.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,moved:false,panX,panY}});network.addEventListener('pointermove',e=>{if(drag){const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>3)drag.moved=true;panX=drag.panX+dx;panY=drag.panY+dy;paintNetwork()}});network.addEventListener('pointerup',e=>{if(drag&&!drag.moved){const rect=network.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;let best=-1,dist=15;visibleNodes.forEach((node,i)=>{const d=Math.hypot(node.p.x-x,node.p.y-y);if(d<dist){dist=d;best=i}});if(best>=0)inspect(visibleNodes[best].n)}drag=null});network.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.35,Math.min(4,zoom*(e.deltaY<0?1.1:.9)));paintNetwork()},{passive:false});
const stop=()=>{playing=false;clearInterval(timer);timer=null;$('play').textContent='▶ Play replay'};$('play').onclick=()=>{if(!frames.length)return;if(playing){stop();return}playing=true;$('play').textContent='Ⅱ Pause replay';timer=setInterval(()=>{if(index>=frames.length-1){stop();return}setFrame(index+1)},120)};$('back').onclick=()=>setFrame(index-1);$('forward').onclick=()=>setFrame(index+1);$('reset').onclick=()=>setFrame(frames.length-1);$('timeline-range').oninput=()=>setFrame(Number($('timeline-range').value));$('role-filter').onchange=paintNetwork;
$('download-checkpoint').onclick=()=>{const b64=encodedNode.textContent.replace(/\s+/g,'');const a=document.createElement('a');a.href='data:application/octet-stream;base64,'+b64;a.download=`Sonapse_Native_Checkpoint_step_${meta.simulation_step}.pt`;a.click();a.remove()};
window.addEventListener('resize',()=>{paintNetwork();paintTimeline();paintWeights()});setFrame(index);paintNetwork();paintTimeline();paintWeights();
})();
</script></body></html>
"""


def write_portable_html(destination, *, checkpoint_path, checkpoint_sha256: str,
                        overview: dict, replay_history: dict, metadata: dict) -> dict:
    """Atomically write one self-contained HTML with an exact native archive."""
    destination = Path(destination)
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.stat().st_size > _MAX_NATIVE_CHECKPOINT_BYTES:
        raise ValueError("checkpoint exceeds the existing 1 GiB native loader limit")
    observed_sha256, checkpoint_bytes = _sha256_file(checkpoint_path)
    if observed_sha256 != checkpoint_sha256:
        raise ValueError("checkpoint checksum does not match embedded archive")
    portable_meta = {
        **metadata,
        "schema": PORTABLE_SCHEMA,
        "checkpoint_format": "synapse-state-v1",
        "checkpoint_sha256": observed_sha256,
        "checkpoint_bytes": checkpoint_bytes,
        "restore_support": "exact native restore through compatible Sonapse-LAB Python/PyTorch runtime",
        "portable_engine": "read-only offline snapshot inspector; no browser simulation engine",
        "compatibility_note": (
            "The current engine uses PyTorch modules, optimizers, and Python world implementations. "
            "This HTML preserves the native checkpoint but does not implement browser-side continuation."
        ),
    }
    prefix = (_HTML_START + _json_for_script(portable_meta) + _HTML_BETWEEN_JSON_AND_CHECKPOINT
              + _json_for_script(overview) + _HTML_CHECKPOINT_START
              + _json_for_script(replay_history) + _HTML_SCRIPT_START).encode("utf-8")
    suffix = _HTML_SCRIPT_END.encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    embedded_hash = hashlib.sha256()
    embedded_bytes = 0
    try:
        with temp_path.open("wb") as output:
            output.write(prefix)
            carry = b""
            with checkpoint_path.open("rb") as source:
                while True:
                    chunk = source.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    embedded_hash.update(chunk)
                    embedded_bytes += len(chunk)
                    chunk = carry + chunk
                    usable = len(chunk) - (len(chunk) % 3)
                    if usable:
                        output.write(base64.b64encode(chunk[:usable]))
                    carry = chunk[usable:]
            if carry:
                output.write(base64.b64encode(carry))
            output.write(suffix)
            output.flush()
            os.fsync(output.fileno())
        if embedded_bytes != checkpoint_bytes or embedded_hash.hexdigest() != observed_sha256:
            raise ValueError("checkpoint changed while the portable HTML was being generated")
        os.replace(temp_path, destination)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
    html_sha256, html_bytes = _sha256_file(destination)
    return {
        "filename": destination.name,
        "html_bytes": html_bytes,
        "html_sha256": html_sha256,
        "checkpoint_sha256": observed_sha256,
        "checkpoint_bytes": checkpoint_bytes,
        "schema": PORTABLE_SCHEMA,
    }


def portable_filename(exported_at: datetime | None = None, token: str | None = None) -> str:
    exported_at = exported_at or datetime.now(timezone.utc)
    stamp = exported_at.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = (token or uuid.uuid4().hex)[:8]
    return f"Sonapse_Portable_Simulation_{stamp}_{suffix}.html"
