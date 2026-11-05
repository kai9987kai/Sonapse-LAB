"""Sanity checks: every task is solvable by a scripted oracle (so failures are learning, not bugs)."""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from graftbrain.world import TASKS, World  # noqa: E402


def steer(w, gx, gy, stop=0.0):
    a = w.a
    ang = (math.atan2(gy - a["y"], gx - a["x"]) - a["th"] + math.pi) % (2 * math.pi) - math.pi
    d = math.hypot(gx - a["x"], gy - a["y"])
    fwd = 0.0 if abs(ang) > 0.6 or d < stop else min(1.0, d)
    return fwd, float(np.clip(ang * 2, -1, 1))


def oracle(w):
    s, a, t = w.st, w.a, w.task
    act = np.zeros(5); act[4] = -1
    if t == "forage":
        act[:2] = steer(w, s["goal"]["x"], s["goal"]["y"])
    elif t == "maze":
        n, cs, geo = s["n"], s["cs"], s["geo"]
        cx = int(np.clip((a["x"] + 5) // cs, 0, n - 1)); cy = int(np.clip((a["y"] + 5) // cs, 0, n - 1))
        best = (cx, cy)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < n and 0 <= ny < n and geo[nx, ny] == geo[cx, cy] - 1:
                blocked = s["wv"][cy, max(cx, nx)] if dx else s["wh"][max(cy, ny), cx]
                if not blocked:
                    best = (nx, ny)
        act[:2] = steer(w, -5 + (best[0] + .5) * cs, -5 + (best[1] + .5) * cs)
    elif t == "push":
        b, p = s["box"], s["pad"]
        ux, uy = p["x"] - b["x"], p["y"] - b["y"]; d = math.hypot(ux, uy) + 1e-9
        bx, by = b["x"] - ux / d * 0.65, b["y"] - uy / d * 0.65
        if math.hypot(a["x"] - bx, a["y"] - by) > 0.2:
            act[:2] = steer(w, bx, by)
        else:
            act[:2] = steer(w, p["x"], p["y"])
    elif t in ("grasp_place", "peg_insert"):
        o = s["cube"] if t == "grasp_place" else s["peg"]
        dst = s["bin"] if t == "grasp_place" else s["hole"]
        tx, ty, tz = w.tip()
        if not o.get("held"):
            act[:2] = steer(w, o["x"], o["y"], stop=0.55)
            act[2] = np.clip((math.hypot(o["x"] - a["x"], o["y"] - a["y"]) - a["reach"]) * 10, -1, 1)
            act[3] = np.clip((o["r"] - a["height"]) * 10, -1, 1)
            act[4] = 1 if math.hypot(tx - o["x"], ty - o["y"]) < 0.1 else -1
        else:
            act[:2] = steer(w, dst["x"], dst["y"], stop=a["reach"])
            act[2] = np.clip((math.hypot(dst["x"] - a["x"], dst["y"] - a["y"]) - a["reach"]) * 10, -1, 1)
            act[3] = 1
            act[4] = -1 if math.hypot(tx - dst["x"], ty - dst["y"]) < 0.04 else 1
            if abs(act[0]) < 1e-6 and abs(act[2]) < 0.2 and math.hypot(tx - dst["x"], ty - dst["y"]) >= 0.04:
                act[1] = steer(w, dst["x"], dst["y"])[1]
    elif t == "delayed_match":
        if s["phase"] == "choice":
            o = next(o for o in s["choices"] if o["cname"] == s["sample"])
            act[:2] = steer(w, o["x"], o["y"])
    elif t == "go_nogo":
        if s["go"]:
            act[:2] = steer(w, s["orb"]["x"], s["orb"]["y"])
    elif t == "button_sequence":
        if w.t > 30 and s["next"] < 3:
            b = next(b for b in s["buttons"] if b["cname"] == s["seq"][s["next"]])
            act[:2] = steer(w, b["x"], b["y"] - 0.5, stop=0.05)
            if math.hypot(a["x"] - b["x"], a["y"] - (b["y"] - 0.5)) < 0.15:
                act[:2] = (0, steer(w, b["x"], b["y"])[1])
            act[2] = np.clip((math.hypot(b["x"] - a["x"], b["y"] - a["y"]) - a["reach"]) * 10, -1, 1)
            act[3] = np.clip((b["z"] - a["height"]) * 10, -1, 1)
    elif t == "webcam_chase":
        act[:2] = steer(w, s["orb"]["x"], s["orb"]["y"], stop=0.8)
    return act


if __name__ == "__main__":
    for task in TASKS:
        wins = 0; trials = 10
        for k in range(trials):
            w = World(seed=k, curriculum=task); w.reset(task)
            done, info = False, {}
            while not done:
                _, r, done, info = w.step(oracle(w))
            wins += info.get("outcome") == "success" or (task == "webcam_chase" and info["ret"] > 0)
        print(f"{task:16s} oracle success {wins}/{trials}")
