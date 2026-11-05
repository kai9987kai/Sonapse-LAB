"""Closed-loop runtime: world <-> ANN <-> grafted connectome, with continuous state saving."""
from __future__ import annotations

import base64
import dataclasses
import json
import os
import queue
import threading
import time

import numpy as np
import torch

from . import connectomes as cx
from .brain import Agent
from .graft import GraftedGraph, build
from .plasticity import PlasticityConfig
from .senses import Webcam
from .world import TASKS, World

ACT_DIM = 5  # drive, turn, reach, height, grip


# ----------------------------------------------------------------------------- persistence
def save_checkpoint(path, payload, keep=3):
    """Atomic save with rolling backups: path, path.1, path.2 ..."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    for i in range(keep - 1, 0, -1):
        src = path if i == 1 else f"{path}.{i - 1}"
        if os.path.exists(src):
            os.replace(src, f"{path}.{i}")
    os.replace(tmp, path)


def load_checkpoint(path):
    for p in [path] + [f"{path}.{i}" for i in range(1, 4)]:
        if os.path.exists(p):
            try:
                return torch.load(p, map_location="cpu", weights_only=False), p
            except Exception as e:  # corrupted -> fall back to an older backup
                print(f"[persist] could not read {p}: {e}")
    return None, None


# ----------------------------------------------------------------------------- simulation
class Simulation:
    def __init__(self, specs, state_path="state/graftbrain.pt", max_neurons=4000, webcam=False,
                 autosave_s=60, task="auto", seed=0, microsteps=4, fresh=False, plasticity_config=None,
                 plasticity_overrides=None):
        self.state_path, self.autosave_s = state_path, autosave_s
        self.journal_path = os.path.splitext(state_path)[0] + "_journal.jsonl"
        self.cmds: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.paused = False
        self.target_hz = 30.0
        self.total_steps = 0
        self.started = time.time()

        ckpt, used = (None, None) if fresh else load_checkpoint(state_path)
        if ckpt is not None:
            saved = ckpt["config"]
            saved_plasticity = PlasticityConfig(**saved.get("plasticity", {}))
            conflict = (any(getattr(saved_plasticity, key) != value for key, value in plasticity_overrides.items())
                        if plasticity_overrides is not None else
                        plasticity_config is not None and plasticity_config != saved_plasticity)
            if conflict:
                raise ValueError("checkpoint uses different graft settings; resume without overrides or use --fresh")
            plasticity_config = saved_plasticity
            seed = saved.get("seed", seed)
            microsteps = saved.get("microsteps", microsteps)
            max_neurons = saved.get("max_neurons", max_neurons)
        plasticity_config = plasticity_config or PlasticityConfig()
        if ckpt is not None and ckpt["config"]["specs"] != list(specs):
            print(f"[persist] checkpoint was built from {ckpt['config']['specs']}, requested {list(specs)}; "
                  f"use --fresh or a different --state to start a new brain. Resuming the saved one.")
            specs = ckpt["config"]["specs"]
        if ckpt is not None:
            self.graph = GraftedGraph(**ckpt["graph"])
            print(f"[persist] resumed from {used} (step {ckpt['total_steps']}, episode {ckpt['world']['episode']})")
        else:
            print("[build] loading connectomes:", ", ".join(specs))
            cs = [cx.load(s, max_neurons=max_neurons, seed=seed) for s in specs]
            for c in cs:
                print(f"   {c.name:12s} {c.n:7d} neurons {c.e:9d} edges  ({c.provenance.get('source')})")
            self.graph = build(cs, seed=seed)
        g = self.graph
        print(f"[graft] {g.n} neurons, {len(g.base_pre)} connectome edges, {len(g.graft_pre)} graft edges "
              f"across {len(g.sites)} sites")

        self.config = {"specs": list(specs), "max_neurons": max_neurons, "seed": seed, "microsteps": microsteps,
                       "plasticity": dataclasses.asdict(plasticity_config)}
        self.world = World(seed=seed, curriculum=task)
        self.cam = Webcam(enabled=webcam)
        self.agent = Agent(g, World.obs_dim, (2, 48, 64), ACT_DIM, microsteps=microsteps, seed=seed,
                           plasticity_config=plasticity_config)
        if ckpt is not None:
            self.agent.load_state_dict(ckpt["agent"])
            self.total_steps = ckpt["total_steps"]
            self.world.episode = ckpt["world"]["episode"]; self.world.task_i = ckpt["world"]["task_i"]
            for t, h in ckpt["world"]["history"].items():
                if t in self.world.history:
                    self.world.history[t].extend(h)
            self.world.reset()
        self.obs = self.world.observe()
        self.last_save = time.time()
        self.recent = []
        self.snapshot = {}
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    # ------------------------------------------------------------------ persistence
    def save(self, reason="autosave"):
        payload = {
            "version": 2, "config": self.config, "graph": dataclasses.asdict(self.graph),
            "agent": self.agent.state_dict(), "total_steps": self.total_steps, "saved_at": time.time(),
            "world": {"episode": self.world.episode, "task_i": self.world.task_i,
                      "history": {t: list(h) for t, h in self.world.history.items()}},
        }
        save_checkpoint(self.state_path, payload)
        self.last_save = time.time()
        print(f"[persist] {reason}: saved step {self.total_steps} -> {self.state_path}")

    def _journal(self, info):
        os.makedirs(os.path.dirname(os.path.abspath(self.journal_path)), exist_ok=True)
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.time(), "step": self.total_steps, "episode": self.world.episode, **info}) + "\n")

    # ------------------------------------------------------------------ loop
    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=5)
        self.save("shutdown")
        self.cam.close()

    def _handle(self, cmd):
        c = cmd.get("cmd")
        if c == "pause": self.paused = not self.paused
        elif c == "save": self.save("manual")
        elif c == "speed": self.target_hz = float(cmd.get("hz", 30))
        elif c == "task":
            t = cmd.get("task", "auto")
            self.world.curriculum = t
            self.agent.buf.clear()
            self.obs = self.world.reset(t if t in TASKS else None)
        elif c == "reset_activity":
            self.agent.buf.clear()
            self.agent.reset_episode()

    def step_once(self):
        cam, centroid = self.cam.read()
        self.world.webcam_target = centroid
        action, value = self.agent.act(self.obs, cam)
        self.obs, r, done, info = self.world.step(action)
        self.agent.feedback(r, done, next_obs=self.obs, next_cam=cam)
        self.total_steps += 1
        if done:
            self.recent = (self.recent + [info])[-30:]
            self._journal(info)
            self.agent.reset_episode()
            self.obs = self.world.reset()
        return value

    def run_steps(self, n):
        for _ in range(n):
            self.step_once()

    def _loop(self):
        last_snap = 0.0
        while not self._stop.is_set():
            while not self.cmds.empty():
                self._handle(self.cmds.get())
            if self.paused:
                time.sleep(0.05)
                self._make_snapshot(0.0)
                continue
            t0 = time.time()
            value = self.step_once()
            if time.time() - last_snap > 1 / 15:
                self._make_snapshot(value); last_snap = time.time()
            if time.time() - self.last_save > self.autosave_s:
                self.save()
            if self.target_hz > 0:
                time.sleep(max(0.0, 1 / self.target_hz - (time.time() - t0)))

    # ------------------------------------------------------------------ telemetry
    def _make_snapshot(self, value):
        g, b = self.graph, self.agent.brain
        r = b.r.numpy()
        act = []
        for i, name in enumerate(g.names):
            seg = r[g.offsets[i]:g.offsets[i + 1]]
            k = 1024
            img = np.interp(np.linspace(0, len(seg) - 1, k), np.arange(len(seg)), seg) if len(seg) else np.zeros(k)
            act.append({"name": name, "mean": float(seg.mean()) if len(seg) else 0.0,
                        "img": base64.b64encode((np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes()).decode()})
        gw = self.agent.plasticity.effective_weights(b.graft_w).detach().numpy()
        flow = np.abs(gw) * r[g.graft_pre]
        site_stats = []
        plastic_sites = {s["id"]: s for s in self.agent.plasticity.site_stats()}
        for s in g.sites:
            m = g.graft_site == s["id"]
            site_stats.append({**s, **plastic_sites[s["id"]],
                               "w": float(np.abs(gw[m]).mean()), "flow": float(flow[m].mean())})
        snap = {
            "world": self.world.snapshot(), "brain": act, "sites": site_stats,
            "plasticity": self.agent.plasticity.summary(),
            "compass": {
                "bump": float(self.agent.compass.bump_angle),
                "wedges": self.agent.last_compass.tolist(),
            },
            "mb": {
                "kc_sparsity": float((self.agent.mb.kc_act > 0).float().mean()),
                "mbon": self.agent.last_mbon.tolist(),
            },
            "stats": {**self.agent.stats, "value": value, "steps": self.total_steps, "paused": self.paused,
                      "hz": self.target_hz, "neurons": g.n, "conn_edges": int(len(g.base_pre)),
                      "graft_edges": int(len(g.graft_pre)), "since_save": time.time() - self.last_save,
                      "webcam_live": self.cam.live},
            "recent": self.recent[-10:], "cam": self.cam.thumb_b64(),
        }
        with self.lock:
            self.snapshot = snap

    def get_snapshot(self):
        with self.lock:
            return self.snapshot
