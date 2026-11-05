"""Virtual 3D environment with sensory input, object manipulation, fine motor and cognitive tasks.

Physics is planar rigid-body (x, y) with height (z) for the arm, grasped objects and buttons,
which keeps it fast enough to run in lock-step with the connectome on a CPU. Rendering is
full 3D in the browser viewer.

Body: a mobile base (drive + turn) carrying a 2-DoF arm (reach, height) with a gripper.
Senses: a 33-ray colour+depth retina (120 deg FOV), proprioception, a task cue channel,
reward, and (separately) the live webcam stream.

Tasks
-----
navigation : forage, maze
manipulation: push
fine motor : grasp_place, peg_insert
cognitive  : delayed_match, go_nogo, button_sequence
live       : webcam_chase (an orb in the world is driven by motion seen by the webcam)
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

TASKS = ["forage", "maze", "push", "grasp_place", "peg_insert", "delayed_match", "go_nogo",
         "button_sequence", "webcam_chase"]
COLORS = {"red": (1.0, 0.2, 0.2), "green": (0.2, 1.0, 0.3), "blue": (0.25, 0.4, 1.0),
          "yellow": (1.0, 0.9, 0.2), "purple": (0.7, 0.3, 1.0), "white": (1.0, 1.0, 1.0),
          "gold": (1.0, 0.75, 0.1), "cyan": (0.2, 0.9, 1.0), "grey": (0.55, 0.55, 0.6)}
N_RAYS, FOV, RANGE = 33, math.radians(120), 9.0
HALF = 5.0
DT = 0.1


class World:
    obs_dim = N_RAYS * 4 + 8 + len(TASKS) + 4 + 2

    def __init__(self, seed=0, curriculum="auto"):
        self.rng = np.random.default_rng(seed)
        self.curriculum = curriculum          # "auto" cycles tasks, or a fixed task name
        self.task_i = 0
        self.episode = 0
        self.webcam_target = (0.5, 0.5, 0.0)  # normalised centroid x,y + motion energy from the webcam
        self.history = {t: deque(maxlen=50) for t in TASKS}
        self.reset()

    # ------------------------------------------------------------------ setup
    def _border(self):
        h = HALF
        return [(-h, -h, h, -h), (h, -h, h, h), (h, h, -h, h), (-h, h, -h, -h)]

    def _obj(self, kind, x, y, color, r=0.25, shape="sphere", movable=False, graspable=False, z=0.0, solid=True, **kw):
        o = dict(id=len(self.objects), kind=kind, x=float(x), y=float(y), z=float(z), r=r, color=COLORS[color],
                 cname=color, shape=shape, movable=movable, graspable=graspable, solid=solid, visible=True, **kw)
        self.objects.append(o)
        return o

    def _free_spot(self, margin=0.8, avoid=()):
        for _ in range(200):
            p = self.rng.uniform(-HALF + margin, HALF - margin, 2)
            if all(np.hypot(*(p - np.array(a))) > 1.2 for a in avoid):
                return p
        return p

    def reset(self, task=None):
        if task is None:
            task = self.curriculum if self.curriculum in TASKS else TASKS[self.task_i % len(TASKS)]
        self.task = task
        self.walls = self._border()
        self.objects = []
        self.t = 0
        self.cue = np.zeros(4)
        self.last_reward = 0.0
        self.ep_return = 0.0
        self.st = {}
        self.a = dict(x=0.0, y=-3.5, th=math.pi / 2, v=0.0, w=0.0, reach=0.35, height=0.3, grip=0.0, held=None)
        getattr(self, f"_setup_{task}")()
        self.max_t = {"maze": 900, "webcam_chase": 400}.get(task, 450)
        self._prev = self._progress()
        return self.observe()

    def _setup_forage(self):
        self.a.update(x=self.rng.uniform(-3, 3), y=self.rng.uniform(-3, 3), th=self.rng.uniform(-math.pi, math.pi))
        p = self._free_spot(avoid=[(self.a["x"], self.a["y"])])
        self.st["goal"] = self._obj("target", *p, "green", r=0.3)

    def _setup_maze(self):
        n = 6; cs = 2 * HALF / n
        walls_h = np.ones((n + 1, n), bool); walls_v = np.ones((n, n + 1), bool)
        seen = np.zeros((n, n), bool); stack = [(0, 0)]; seen[0, 0] = True
        while stack:                                           # randomised DFS maze
            cx, cy = stack[-1]
            nb = [(cx + dx, cy + dy, dx, dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if 0 <= cx + dx < n and 0 <= cy + dy < n and not seen[cx + dx, cy + dy]]
            if not nb:
                stack.pop(); continue
            nx, ny, dx, dy = nb[self.rng.integers(len(nb))]
            if dx: walls_v[cy, max(cx, nx)] = False
            else: walls_h[max(cy, ny), cx] = False
            seen[nx, ny] = True; stack.append((nx, ny))
        self.walls = []
        for j in range(n + 1):
            for i in range(n):
                if walls_h[j, i]:
                    self.walls.append((-HALF + i * cs, -HALF + j * cs, -HALF + (i + 1) * cs, -HALF + j * cs))
        for j in range(n):
            for i in range(n + 1):
                if walls_v[j, i]:
                    self.walls.append((-HALF + i * cs, -HALF + j * cs, -HALF + i * cs, -HALF + (j + 1) * cs))
        self.st.update(n=n, cs=cs, wh=walls_h, wv=walls_v)
        self.a.update(x=-HALF + cs / 2, y=-HALF + cs / 2, th=0.0)
        self.st["goal"] = self._obj("target", HALF - cs / 2, HALF - cs / 2, "gold", r=0.3, shape="pad", solid=False)
        self.st["geo"] = self._maze_bfs(n - 1, n - 1)

    def _maze_bfs(self, gx, gy):
        n, wh, wv = self.st["n"], self.st["wh"], self.st["wv"]
        d = np.full((n, n), 99); d[gx, gy] = 0; q = deque([(gx, gy)])
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < n and 0 <= ny < n) or d[nx, ny] < 99:
                    continue
                blocked = wv[y, max(x, nx)] if dx else wh[max(y, ny), x]
                if not blocked:
                    d[nx, ny] = d[x, y] + 1; q.append((nx, ny))
        return d

    def _setup_push(self):
        b = self._free_spot(1.5, [(0, -3.5)])
        p = self._free_spot(1.2, [(0, -3.5), b])
        self.st["box"] = self._obj("box", *b, "red", r=0.3, shape="box", movable=True)
        self.st["pad"] = self._obj("pad", *p, "blue", r=0.45, shape="pad", solid=False)

    def _setup_grasp_place(self):
        c = self._free_spot(1.5, [(0, -3.5)])
        b = self._free_spot(1.5, [(0, -3.5), c])
        self.st["cube"] = self._obj("cube", *c, "yellow", r=0.12, shape="box", movable=True, graspable=True)
        self.st["bin"] = self._obj("bin", *b, "purple", r=0.4, shape="bin", solid=False)
        self.st["grasped_once"] = False

    def _setup_peg_insert(self):
        c = self._free_spot(1.5, [(0, -3.5)])
        b = self._free_spot(1.5, [(0, -3.5), c])
        self.st["peg"] = self._obj("peg", *c, "cyan", r=0.08, shape="cylinder", movable=True, graspable=True)
        self.st["hole"] = self._obj("hole", *b, "grey", r=0.14, shape="hole", solid=False)
        self.st["grasped_once"] = False

    def _setup_delayed_match(self):
        names = ["red", "green", "blue"]
        self.st["sample"] = names[self.rng.integers(3)]
        self.st["phase"] = "cue"
        spots = [(-2.5, 2.0), (0.0, 2.5), (2.5, 2.0)]
        order = self.rng.permutation(3)
        self.st["choices"] = [self._obj("choice", *spots[k], names[i], r=0.35) for k, i in zip(range(3), order)]
        for o in self.st["choices"]:
            o["visible"] = False

    def _setup_go_nogo(self):
        self.st["go"] = bool(self.rng.random() < 0.5)
        self.st["orb"] = self._obj("orb", 0.0, 1.5, "white", r=0.35)

    def _setup_button_sequence(self):
        names = ["red", "green", "blue"]
        xs = [-1.5, 0.0, 1.5]
        perm = self.rng.permutation(3)
        self.st["buttons"] = [self._obj("button", xs[k], 3.6, names[perm[k]], r=0.12, shape="button",
                                        z=float(self.rng.uniform(0.3, 0.6)), solid=False, pressed=False)
                              for k in range(3)]
        self.st["seq"] = [names[i] for i in self.rng.permutation(3)]
        self.st["next"] = 0
        self._obj("post", 0.0, 4.2, "grey", r=0.3, shape="box")

    def _setup_webcam_chase(self):
        self.st["orb"] = self._obj("orb", 0.0, 3.0, "gold", r=0.3, solid=False)

    # ------------------------------------------------------------------ physics
    @staticmethod
    def _seg_push(px, py, rad, walls):
        for x1, y1, x2, y2 in walls:
            dx, dy = x2 - x1, y2 - y1
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy + 1e-9)))
            cx, cy = x1 + t * dx, y1 + t * dy
            d = math.hypot(px - cx, py - cy)
            if d < rad:
                nx, ny = ((px - cx) / d, (py - cy) / d) if d > 1e-6 else (0.0, 1.0)
                px, py = cx + nx * rad, cy + ny * rad
        return px, py

    def tip(self):
        a = self.a
        return a["x"] + a["reach"] * math.cos(a["th"]), a["y"] + a["reach"] * math.sin(a["th"]), a["height"]

    def step(self, action):
        a = self.a
        act = np.clip(np.asarray(action, dtype=float), -1, 1)
        a["v"], a["w"] = 1.4 * act[0], 2.5 * act[1]
        a["reach"] = float(np.clip(a["reach"] + 0.07 * act[2], 0.25, 0.9))
        a["height"] = float(np.clip(a["height"] + 0.07 * act[3], 0.0, 0.75))
        a["grip"] = float(np.clip(a["grip"] + (0.3 if act[4] > 0 else -0.3), 0.0, 1.0))
        a["th"] = (a["th"] + a["w"] * DT + math.pi) % (2 * math.pi) - math.pi
        nx, ny = a["x"] + a["v"] * DT * math.cos(a["th"]), a["y"] + a["v"] * DT * math.sin(a["th"])
        nx, ny = self._seg_push(nx, ny, 0.25, self.walls)
        for o in self.objects:
            if not o["solid"] or not o["visible"] or o.get("held"):
                continue
            d = math.hypot(nx - o["x"], ny - o["y"]); m = 0.25 + o["r"]
            if d < m and d > 1e-6:
                ux, uy = (o["x"] - nx) / d, (o["y"] - ny) / d
                if o["movable"]:                                  # push the object
                    o["x"], o["y"] = nx + ux * m, ny + uy * m
                    o["x"], o["y"] = self._seg_push(o["x"], o["y"], o["r"], self.walls)
                    d2 = math.hypot(nx - o["x"], ny - o["y"])
                    if d2 < m:                                    # object pinned against a wall
                        nx, ny = o["x"] - ux * m, o["y"] - uy * m
                else:
                    nx, ny = o["x"] - ux * m, o["y"] - uy * m
        a["x"], a["y"] = nx, ny

        events = []
        tx, ty, tz = self.tip()
        if a["held"] is None and a["grip"] > 0.7:
            for o in self.objects:
                if o["graspable"] and math.hypot(tx - o["x"], ty - o["y"]) < o["r"] + 0.08 and abs(o["z"] + o["r"] - tz) < 0.2:
                    a["held"] = o["id"]; o["held"] = True; events.append(("grasp", o)); break
        if a["held"] is not None:
            o = self.objects[a["held"]]
            if a["grip"] < 0.3:
                o["held"] = False; o["z"] = 0.0; a["held"] = None; events.append(("release", o))
            else:
                o["x"], o["y"], o["z"] = tx, ty, max(0.0, tz - o["r"])
        for o in self.objects:
            if o["kind"] == "button" and not o["pressed"]:
                if math.hypot(tx - o["x"], ty - o["y"]) < 0.16 and abs(tz - o["z"]) < 0.14:
                    o["pressed"] = True; events.append(("press", o))
        self.t += 1
        r, done, info = self._task_reward(events)
        self.last_reward = r
        self.ep_return += r
        if self.t >= self.max_t:
            done = True; info.setdefault("outcome", "timeout")
        if done:
            self.history[self.task].append(1.0 if info.get("outcome") == "success" else 0.0)
            info.update(task=self.task, ret=self.ep_return, steps=self.t)
            self.episode += 1; self.task_i += 1
        return self.observe(), r, done, info

    # ------------------------------------------------------------------ rewards
    def _dist(self, o, p=None):
        p = p or (self.a["x"], self.a["y"])
        return math.hypot(o["x"] - p[0], o["y"] - p[1])

    def _progress(self):
        """Task potential (higher = closer to solving); reward includes its increase."""
        t, s = self.task, self.st
        if t == "forage":
            return -self._dist(s["goal"])
        if t == "maze":
            n, cs = s["n"], s["cs"]
            cx = int(np.clip((self.a["x"] + HALF) // cs, 0, n - 1)); cy = int(np.clip((self.a["y"] + HALF) // cs, 0, n - 1))
            return -s["geo"][cx, cy] * cs - 0.3 * self._dist(s["goal"])
        if t == "push":
            return -2 * self._dist(s["box"], (s["pad"]["x"], s["pad"]["y"])) - 0.3 * self._dist(s["box"])
        if t in ("grasp_place", "peg_insert"):
            obj = s["cube"] if t == "grasp_place" else s["peg"]
            dst = s["bin"] if t == "grasp_place" else s["hole"]
            tx, ty, tz = self.tip()
            if obj.get("held"):
                return 2.0 - 1.5 * self._dist(obj, (dst["x"], dst["y"]))
            return -math.sqrt((tx - obj["x"]) ** 2 + (ty - obj["y"]) ** 2 + (tz - obj["z"] - obj["r"]) ** 2) - 0.5 * self._dist(obj)
        if t == "webcam_chase":
            o = s["orb"]; ang = math.atan2(o["y"] - self.a["y"], o["x"] - self.a["x"]) - self.a["th"]
            return 0.5 * math.cos(ang) - 0.2 * self._dist(o)
        if t == "button_sequence" and s["next"] < 3:
            b = next(b for b in s["buttons"] if b["cname"] == s["seq"][s["next"]])
            tx, ty, tz = self.tip()
            return -math.sqrt((tx - b["x"]) ** 2 + (ty - b["y"]) ** 2 + (tz - b["z"]) ** 2) if self.t > 30 else 0.0
        return 0.0

    def _task_reward(self, events):
        t, s = self.task, self.st
        done, info = False, {}
        if t == "delayed_match":                      # cue 12 steps -> delay 20 steps -> choice
            if self.t == 12: s["phase"] = "delay"
            if self.t == 32:
                s["phase"] = "choice"
                for o in s["choices"]: o["visible"] = True
            self.cue = np.r_[COLORS[s["sample"]], 1.0] if s["phase"] == "cue" else np.zeros(4)
            if s["phase"] == "choice":
                for o in s["choices"]:
                    if self._dist(o) < 0.7:
                        ok = o["cname"] == s["sample"]
                        return (5.0 if ok else -2.0), True, {"outcome": "success" if ok else "fail"}
            return -0.005, False, info
        if t == "go_nogo":
            self.cue = np.r_[COLORS["green" if s["go"] else "red"], 1.0] if self.t <= 15 else np.zeros(4)
            near = self._dist(s["orb"]) < 0.75
            if s["go"] and near:
                return 3.0, True, {"outcome": "success"}
            if not s["go"] and near:
                return -3.0, True, {"outcome": "fail"}
            if not s["go"] and self.t >= 80:
                return 3.0, True, {"outcome": "success"}
            if s["go"] and self.t >= 150:
                return -1.0, True, {"outcome": "fail"}
            return -0.005, False, info
        if t == "button_sequence":                    # show order: 3 colours x 10 steps
            k = self.t // 10
            self.cue = np.r_[COLORS[s["seq"][k]], 1.0] if k < 3 else np.zeros(4)
            r = 0.0
            for ev, o in events:
                if ev == "press":
                    if self.t <= 30 or o["cname"] != s["seq"][s["next"]]:
                        return -1.0, True, {"outcome": "fail"}
                    s["next"] += 1; r += 2.0
                    if s["next"] == 3:
                        return r + 5.0, True, {"outcome": "success"}
            pot = self._progress(); r += 0.5 * (pot - self._prev); self._prev = pot
            return r - 0.005, False, info
        if t == "webcam_chase":
            cx, cy, energy = self.webcam_target
            o = s["orb"]
            o["x"] += 0.15 * ((cx * 2 - 1) * (HALF - 0.6) - o["x"])
            o["y"] += 0.15 * ((1 - cy) * 2 * (HALF - 0.6) - (HALF - 0.6) - o["y"])
            self.cue = np.array([cx, cy, energy, 1.0])

        pot = self._progress()
        r = (pot - self._prev) - 0.01
        self._prev = pot
        for ev, o in events:
            if ev == "grasp" and not s.get("grasped_once", True):
                s["grasped_once"] = True; r += 1.0
            if ev == "release" and t in ("grasp_place", "peg_insert"):
                dst = s["bin"] if t == "grasp_place" else s["hole"]
                tol = dst["r"] - 0.1 if t == "grasp_place" else 0.07
                if self._dist(o, (dst["x"], dst["y"])) < tol:
                    return r + 6.0, True, {"outcome": "success"}
        if t == "forage" and self._dist(s["goal"]) < 0.55:
            return r + 5.0, True, {"outcome": "success"}
        if t == "maze" and self._dist(s["goal"]) < 0.6:
            return r + 8.0, True, {"outcome": "success"}
        if t == "push" and self._dist(s["box"], (s["pad"]["x"], s["pad"]["y"])) < 0.35:
            return r + 6.0, True, {"outcome": "success"}
        if t == "webcam_chase":
            r += 0.02 if self._dist(s["orb"]) < 1.0 else 0.0
        return r, done, info

    # ------------------------------------------------------------------ senses
    def retina(self):
        a = self.a
        ang = a["th"] + np.linspace(-FOV / 2, FOV / 2, N_RAYS)
        dx, dy = np.cos(ang), np.sin(ang)
        best = np.full(N_RAYS, RANGE); col = np.zeros((N_RAYS, 3))
        if self.walls:
            W = np.array(self.walls)
            ex, ey = W[:, 2] - W[:, 0], W[:, 3] - W[:, 1]
            qx, qy = W[:, 0] - a["x"], W[:, 1] - a["y"]
            den = dx[:, None] * ey[None] - dy[:, None] * ex[None]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (qx[None] * ey[None] - qy[None] * ex[None]) / den
                u = (qx[None] * dy[:, None] - qy[None] * dx[:, None]) / den
            t = np.where((t > 0) & (u >= 0) & (u <= 1) & np.isfinite(t), t, RANGE)
            tw = t.min(1)
            m = tw < best; best[m] = tw[m]; col[m] = (0.45, 0.45, 0.5)
        for o in self.objects:
            if not o["visible"] or o.get("held"):
                continue
            ox, oy = o["x"] - a["x"], o["y"] - a["y"]
            b = ox * dx + oy * dy
            c = ox * ox + oy * oy - o["r"] ** 2
            disc = b * b - c
            hit = (disc >= 0) & (b > 0)
            tt = np.where(hit, b - np.sqrt(np.maximum(disc, 0)), RANGE)
            m = tt < best; best[m] = tt[m]; col[m] = o["color"]
        depth = 1.0 - best / RANGE
        self._rays = best
        return np.concatenate([depth, col[:, 0], col[:, 1], col[:, 2]])

    def observe(self):
        a = self.a
        prop = [(a["reach"] - 0.25) / 0.65, a["height"] / 0.75, a["grip"], float(a["held"] is not None),
                a["v"] / 1.4, a["w"] / 2.5, math.sin(a["th"]), math.cos(a["th"])]
        onehot = np.eye(len(TASKS))[TASKS.index(self.task)]
        return np.concatenate([self.retina(), prop, onehot, self.cue, [self.last_reward, self.t / self.max_t]]).astype(np.float32)

    def success_rates(self):
        return {t: (float(np.mean(h)) if h else None) for t, h in self.history.items()}

    def snapshot(self):
        a = self.a
        ang = a["th"] + np.linspace(-FOV / 2, FOV / 2, N_RAYS)
        rays = getattr(self, "_rays", np.full(N_RAYS, RANGE))
        return {
            "task": self.task, "t": self.t, "episode": self.episode, "cue": self.cue.tolist(),
            "walls": self.walls,
            "objects": [{k: o[k] for k in ("id", "kind", "x", "y", "z", "r", "color", "shape", "visible")}
                        | {"pressed": o.get("pressed", False), "held": bool(o.get("held"))} for o in self.objects],
            "agent": {k: a[k] for k in ("x", "y", "th", "reach", "height", "grip")} | {"tip": self.tip()},
            "rays": np.stack([a["x"] + rays * np.cos(ang), a["y"] + rays * np.sin(ang)], 1)[::2].round(3).tolist(),
            "success": self.success_rates(), "ep_return": self.ep_return,
        }
