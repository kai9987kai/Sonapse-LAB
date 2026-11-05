"""GraftBrain CLI.

    python -m graftbrain run                     # build/resume, open the live 3D viewer
    python -m graftbrain run --webcam            # ... with the live webcam feed
    python -m graftbrain train --steps 20000     # headless, as fast as possible, autosaving
    python -m graftbrain info                    # describe the grafted topology
    python -m graftbrain fetch-flywire           # download FlyWire v783 connections (~1 GB, CC-BY-NC)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

DEFAULT_SPECS = ["proxy:flywire", "proxy:hemibrain", "proxy:malecns", "proxy:banc",
                 "proxy:optic-lobe", "proxy:manc", "proxy:fanc", "proxy:larval"]


def _common(p):
    p.add_argument("--connectome", "-c", action="append", dest="specs",
                   help="connectome spec (repeatable). proxy:<name> | flywire:<file> | neuprint:<dataset> | csv:<file>")
    p.add_argument("--max-neurons", type=int, default=1500, help="per-connectome neuron cap (real-time budget)")
    p.add_argument("--state", default="state/graftbrain.pt", help="checkpoint path (auto-resumed)")
    p.add_argument("--fresh", action="store_true", help="ignore any saved state and build a new brain")
    p.add_argument("--task", default="auto", help="auto (curriculum) or a fixed task name")
    p.add_argument("--webcam", action="store_true", help="use the live webcam (frames stay local, never saved)")
    p.add_argument("--autosave", type=float, default=60, help="seconds between autosaves")
    p.add_argument("--microsteps", type=int, default=4, help="neural microsteps per world step")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--graft-mode", choices=("legacy", "static", "routing", "structural", "adaptive"), default=None,
                   help="graft plasticity mode (fresh default: legacy; resume uses saved settings)")
    p.add_argument("--graft-density", type=float, default=None, help="active fraction per directional site bucket, (0, 1]")
    p.add_argument("--graft-interval", "--graft-rewire-interval", dest="graft_interval", type=int, default=None,
                   help="complete learner updates between rewiring events")
    p.add_argument("--graft-warmup", type=int, default=None, help="learner updates before structural adaptation")
    p.add_argument("--graft-turnover", type=float, default=None, help="fraction of each active bucket to replace, [0, 1]")
    p.add_argument("--graft-router-lr", type=float, default=None, help="router Adam learning rate")
    p.add_argument("--graft-growth", choices=("coactivity", "gradient", "random"), default=None,
                   help="dormant candidate recruitment score (fresh default: coactivity)")
    p.add_argument("--graft-growth-init", choices=("original", "zero"), default=None,
                   help="initial weight of regrown candidate records")
    p.add_argument("--graft-min-age", type=int, default=None,
                   help="minimum completed learning updates before an active record can be pruned")


def _sim(a):
    from .runtime import Simulation
    return Simulation(a.specs or DEFAULT_SPECS, a.state, a.max_neurons, a.webcam, a.autosave, a.task, a.seed,
                      a.microsteps, a.fresh, plasticity_config=a.plasticity_config,
                      plasticity_overrides=a.plasticity_overrides)


def parse_args(argv=None):
    """Parse and validate controls before opening cameras or loading saved state."""
    ap = argparse.ArgumentParser(prog="graftbrain", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="live closed loop + 3D viewer"); _common(r); r.add_argument("--port", type=int, default=8770)
    t = sub.add_parser("train", help="headless training"); _common(t); t.add_argument("--steps", type=int, default=10000)
    i = sub.add_parser("info", help="describe topology"); _common(i)
    f = sub.add_parser("fetch-flywire", help="download FlyWire v783 connections")
    f.add_argument("--out", default="data/proofread_connections_783.feather")
    b = sub.add_parser("benchmark-plasticity", help="bounded paired comparison of five graft modes")
    b.add_argument("--out", default=None, help="new output directory (existing paths are refused)")
    b.add_argument("--connectome", "-c", action="append", dest="specs", default=None)
    b.add_argument("--max-neurons", type=int, default=400, help="per-connectome cap; graph must contain at least 600 neurons")
    b.add_argument("--train-steps", type=int, default=512, help="environment steps per mode/task/training seed")
    b.add_argument("--tasks", nargs="+", default=["forage", "go_nogo"])
    b.add_argument("--seeds", nargs="+", type=int, default=[0], help="paired graph and training seeds")
    b.add_argument("--eval-seeds", nargs="+", type=int, default=[1000, 1001], help="held-out world seeds, disjoint from training")
    b.add_argument("--microsteps", type=int, default=2)
    c = sub.add_parser("benchmark-graft-credit", help="balanced pre/post comparison of growth rules and maturation")
    c.add_argument("--out", default=None, help="new output directory")
    c.add_argument("--connectome", "-c", action="append", dest="specs", default=None)
    c.add_argument("--max-neurons", type=int, default=400)
    c.add_argument("--train-steps", type=int, default=384, help="steps per growth-rule/age/task/training-seed arm")
    c.add_argument("--tasks", nargs="+", default=["forage", "go_nogo"])
    c.add_argument("--seeds", nargs="+", type=int, default=[0])
    c.add_argument("--eval-seeds", nargs="+", type=int, default=[1000])
    c.add_argument("--microsteps", type=int, default=1)
    p = sub.add_parser("probe-graft-credit", help="small supervised support-disruption and recovery probe")
    p.add_argument("--out", default=None, help="new output directory")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--train-steps", type=int, default=80)
    p.add_argument("--recovery-steps", type=int, default=80)
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("run", "train", "info", "fetch-flywire", "benchmark-plasticity",
                                    "benchmark-graft-credit", "probe-graft-credit", "-h", "--help"):
        args.insert(0, "run")
    a = ap.parse_args(args)
    for field in ("max_neurons", "microsteps", "steps", "train_steps", "recovery_steps"):
        if hasattr(a, field) and getattr(a, field) <= 0:
            ap.error(f"--{field.replace('_', '-')} must be positive")
    if hasattr(a, "seed") and a.seed < 0:
        ap.error("--seed must be nonnegative")
    if hasattr(a, "autosave") and (not math.isfinite(a.autosave) or a.autosave < 0):
        ap.error("--autosave must be finite and nonnegative")
    if hasattr(a, "port") and not 1 <= a.port <= 65535:
        ap.error("--port must be between 1 and 65535")
    a.plasticity_config = None
    fields = {"mode": "graft_mode", "density": "graft_density", "interval": "graft_interval",
              "warmup": "graft_warmup", "turnover": "graft_turnover", "router_lr": "graft_router_lr",
              "growth_rule": "graft_growth", "growth_init": "graft_growth_init", "min_age": "graft_min_age"}
    overrides = {field: getattr(a, flag) for field, flag in fields.items() if getattr(a, flag, None) is not None}
    a.plasticity_overrides = overrides
    if overrides:
        from .plasticity import PlasticityConfig
        try:
            a.plasticity_config = PlasticityConfig(**overrides)
        except (ValueError, TypeError) as exc:
            ap.error(str(exc))
    if a.cmd == "benchmark-plasticity":
        from .experiments import validate_benchmark
        try:
            validate_benchmark(a.specs, a.max_neurons, a.train_steps, a.tasks, a.seeds, a.eval_seeds, a.microsteps)
        except ValueError as exc:
            ap.error(str(exc))
    elif a.cmd == "benchmark-graft-credit":
        from .credit_experiments import validate_credit_benchmark
        try:
            validate_credit_benchmark(a.specs, a.max_neurons, a.train_steps, a.tasks, a.seeds, a.eval_seeds, a.microsteps)
        except ValueError as exc:
            ap.error(str(exc))
    elif a.cmd == "probe-graft-credit":
        if not a.seeds or len(set(a.seeds)) != len(a.seeds) or any(not 0 <= s < 2**64 for s in a.seeds):
            ap.error("--seeds must be unique nonnegative integers below 2**64")
    elif hasattr(a, "task"):
        from .world import TASKS
        if a.task not in ["auto", *TASKS]:
            ap.error(f"--task must be auto or one of {', '.join(TASKS)}")
    return a


def main(argv=None):
    a = parse_args(argv)

    if a.cmd in ("benchmark-graft-credit", "probe-graft-credit"):
        out = a.out or os.path.join("runs", a.cmd + "-" + time.strftime("%Y%m%d-%H%M%S"))
        try:
            if a.cmd == "benchmark-graft-credit":
                from .credit_experiments import benchmark_graft_credit
                result = benchmark_graft_credit(out, a.specs, a.max_neurons, a.train_steps, a.tasks,
                                                a.seeds, a.eval_seeds, a.microsteps)
            else:
                from .graft_probe import run_graft_probe
                result = run_graft_probe(out, seeds=a.seeds, train_steps=a.train_steps,
                                         recovery_steps=a.recovery_steps)
        except (ValueError, FileExistsError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"graft credit artifacts saved to {result}")
        return

    if a.cmd == "benchmark-plasticity":
        from .experiments import benchmark_plasticity
        out = a.out or os.path.join("runs", "plasticity-" + time.strftime("%Y%m%d-%H%M%S"))
        try:
            result = benchmark_plasticity(out, a.specs, a.max_neurons, a.train_steps, a.tasks,
                                          a.seeds, a.eval_seeds, a.microsteps)
        except (ValueError, FileExistsError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"plasticity benchmark saved to {result}")
        return

    if a.cmd == "fetch-flywire":
        import urllib.request
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        url = "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1"
        print(f"downloading {url}\n -> {a.out}\nFlyWire data is CC-BY-NC-4.0: cite Dorkenwald et al. 2024 & Schlegel et al. 2024.")
        urllib.request.urlretrieve(url, a.out, lambda b, bs, tot: print(f"\r{b * bs / 1e6:8.1f} / {tot / 1e6:.1f} MB", end=""))
        print("\ndone. use:  python -m graftbrain run -c flywire:" + a.out + " -c proxy:manc ...")
        return

    if a.cmd == "info":
        from . import connectomes as cx
        from .graft import build
        g = build([cx.load(s, a.max_neurons, a.seed) for s in (a.specs or DEFAULT_SPECS)], seed=a.seed)
        d = g.describe()
        print(json.dumps({k: v for k, v in d.items() if k != "sites"}, indent=2, default=str))
        kinds = {}
        for s in d["sites"]:
            kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
        print(f"\n{len(d['sites'])} graft sites by kind: {kinds}")
        for s in d["sites"][:25]:
            print(f"  #{s['id']:3d} {s['kind']:9s} {s['from']:>10s} -> {s['to']:<10s} {s['edges']:5d} edges  {s['detail']}")
        if len(d["sites"]) > 25:
            print(f"  ... {len(d['sites']) - 25} more")
        return

    if a.cmd == "train":
        sim = _sim(a)
        t0 = time.time(); last = t0
        try:
            for k in range(a.steps):
                sim.step_once()
                if time.time() - sim.last_save > a.autosave:
                    sim.save()
                if time.time() - last > 10:
                    last = time.time()
                    sr = {t: round(v, 2) for t, v in sim.world.success_rates().items() if v is not None}
                    print(f"step {sim.total_steps:7d} | {k / (last - t0):5.1f} steps/s | ep {sim.world.episode} | "
                          f"loss {sim.agent.stats['loss']:.3f} | success {sr}")
        except KeyboardInterrupt:
            print("\ninterrupted")
        sim.agent.flush(next_obs=sim.obs, next_cam=sim.cam.read()[0])
        sim.save("end of training"); sim.cam.close()
        return

    if a.cmd == "run":
        from .server import serve
        sim = _sim(a)
        sim.start()
        try:
            serve(sim, a.port)
        finally:
            sim.stop()


if __name__ == "__main__":
    main()
