"""Bounded parameter search in copied workspaces; all candidates remain inactive.

Repeated fixed-seed results are selection evidence, not an unbiased estimate of
improvement. Only the final selected candidate is compared with the original
baseline on disjoint audit seeds, and audit results never feed proposal selection.
No command restores, stages, commits, or promotes files in the shared checkout.
An explicit --oracle artifact is required; fallback-only searches stop inactive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "auto_improvements"
SELECTION_SEEDS = [90500000, 90500001, 90500002, 90500003]
AUDIT_SEEDS = [90600000, 90600001, 90600002, 90600003]
EVAL_MAX_STEPS = 32
EVAL_SCARCITY = 2.8
TEST_FILES = [
    "tests/test_cognitive_agent.py", "tests/test_active_inference.py",
    "tests/test_horizon_x.py", "tests/test_guardrails.py",
    "tests/test_metacognition.py",
]
SOURCE_DIRS = ("poseidon", "tests", "configs")
SOURCE_FILES = ("pyproject.toml", "pytest.ini", "setup.cfg", "requirements.txt")
CORE_RELATIVE_PATH = "runs/search_core.pt"
ORACLE_RELATIVE_PATH = "runs/search_oracle.pt"


class InactiveSearchError(RuntimeError):
    """The controller cannot exercise the searched predictive parameters."""

# Search Space Definitions: (filepath, regex_pattern, min_val, max_val, format_str)
SEARCH_SPACE = [
    # Active Inference
    {
        "file": "poseidon/active_inference.py",
        "name": "base_epistemic_weight",
        "pattern": r"(base_epistemic_weight:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.10, "max": 0.65, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "min_epistemic_weight",
        "pattern": r"(min_epistemic_weight:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.01, "max": 0.12, "type": "float", "fmt": "{:.3f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "w_health",
        "pattern": r"(w_health:\s*float\s*=\s*)([0-9\.]+)",
        "min": 3.0, "max": 8.0, "type": "float", "fmt": "{:.1f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "w_energy",
        "pattern": r"(w_energy:\s*float\s*=\s*)([0-9\.]+)",
        "min": 1.5, "max": 5.0, "type": "float", "fmt": "{:.1f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "w_hydration",
        "pattern": r"(w_hydration:\s*float\s*=\s*)([0-9\.]+)",
        "min": 2.0, "max": 5.5, "type": "float", "fmt": "{:.1f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "w_stamina",
        "pattern": r"(w_stamina:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.8, "max": 2.5, "type": "float", "fmt": "{:.1f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "w_threat",
        "pattern": r"(w_threat:\s*float\s*=\s*)([0-9\.]+)",
        "min": 2.0, "max": 6.0, "type": "float", "fmt": "{:.1f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "stamina_depletion_cost",
        "pattern": r"(stamina_depletion_cost:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.01, "max": 0.10, "type": "float", "fmt": "{:.3f}",
    },
    {
        "file": "poseidon/active_inference.py",
        "name": "active_inference_blend",
        "pattern": r"(active_inference_blend:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.01, "max": 0.25, "type": "float", "fmt": "{:.3f}",
    },
    # Horizon-X Planner
    {
        "file": "poseidon/horizon_x.py",
        "name": "base_lambda",
        "pattern": r"(base_lambda:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.05, "max": 0.35, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/horizon_x.py",
        "name": "base_beta",
        "pattern": r"(base_beta:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.60, "max": 2.50, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/horizon_x.py",
        "name": "override_threshold",
        "pattern": r"(override_threshold:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.015, "max": 0.045, "type": "float", "fmt": "{:.3f}",
    },
    {
        "file": "poseidon/horizon_x.py",
        "name": "horizon_decay",
        "pattern": r"(horizon_decay:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.65, "max": 0.95, "type": "float", "fmt": "{:.2f}",
    },
    # Continual Adapter
    {
        "file": "poseidon/adapt.py",
        "name": "anchor_weight",
        "pattern": r"(anchor_weight:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.20, "max": 3.00, "type": "float", "fmt": "{:.2f}",
    },
    # Formal Safety Guardrails
    {
        "file": "poseidon/guardrails.py",
        "name": "critical_hydration_threshold",
        "pattern": r"(critical_hydration_threshold:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.10, "max": 0.22, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/guardrails.py",
        "name": "critical_energy_threshold",
        "pattern": r"(critical_energy_threshold:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.10, "max": 0.22, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/guardrails.py",
        "name": "critical_stamina_threshold",
        "pattern": r"(critical_stamina_threshold:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.04, "max": 0.14, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/guardrails.py",
        "name": "severe_storm_threshold",
        "pattern": r"(severe_storm_threshold:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.60, "max": 0.85, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/guardrails.py",
        "name": "safe_shelter_threshold",
        "pattern": r"(safe_shelter_threshold:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.30, "max": 0.55, "type": "float", "fmt": "{:.2f}",
    },
    # Metacognitive Monitor
    {
        "file": "poseidon/metacognition.py",
        "name": "volatility_threshold_elevated",
        "pattern": r"(volatility_threshold_elevated:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.10, "max": 0.24, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/metacognition.py",
        "name": "volatility_threshold_critical",
        "pattern": r"(volatility_threshold_critical:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.25, "max": 0.45, "type": "float", "fmt": "{:.2f}",
    },
    {
        "file": "poseidon/metacognition.py",
        "name": "energy_deep_horizon_threshold",
        "pattern": r"(energy_deep_horizon_threshold:\s*float\s*=\s*)([0-9\.]+)",
        "min": 0.35, "max": 0.65, "type": "float", "fmt": "{:.2f}",
    },
    # Spatial Cognitive Map
    {
        "file": "poseidon/mnemosyne.py",
        "name": "lcb_z",
        "pattern": r"(\n    lcb_z:\s*float\s*=\s*)([0-9\.]+)",
        "min": 1.0, "max": 2.5, "type": "float", "fmt": "{:.2f}",
    },
]


def remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("search deadline reached")
    return remaining


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(root: Path) -> list[Path]:
    paths = []
    for name in SOURCE_DIRS:
        base = root / name
        if not base.exists():
            continue
        for path in [base, *sorted(base.rglob("*"))]:
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise ValueError(f"source snapshots cannot contain links: {path}")
            if path.is_file() and not any(part in {"__pycache__", ".pytest_cache"}
                                          for part in path.relative_to(base).parts):
                paths.append(path)
    for name in SOURCE_FILES:
        path = root / name
        if path.is_symlink():
            raise ValueError(f"source snapshots cannot contain links: {path}")
        if path.is_file():
            paths.append(path)
    return sorted(paths)


def source_binding(workspace: Path) -> dict:
    manifest = {path.relative_to(workspace).as_posix(): file_hash(path)
                for path in source_files(workspace)}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return {"source_sha256": digest, "source_files": manifest,
            "core_sha256": file_hash(workspace / CORE_RELATIVE_PATH),
            "oracle_sha256": (file_hash(workspace / ORACLE_RELATIVE_PATH)
                              if (workspace / ORACLE_RELATIVE_PATH).is_file() else None)}


def resolve_core(source_root: Path, core_path: Path | None) -> Path:
    if core_path is not None:
        path = Path(core_path)
        path = path if path.is_absolute() else source_root / path
    else:
        pointer = source_root / "runs/active_core.json"
        relative = "runs/tidal/core.pt"
        if pointer.exists():
            relative = json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError("active core pointer must stay within the source tree")
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"trained core checkpoint does not exist: {path}")
    return path


def copy_workspace(source: Path, destination: Path, core: Path, deadline: float,
                   *, oracle: Path | None = None) -> dict:
    """Copy bytes, never links; refuse a moving source rather than mix revisions."""
    remaining_seconds(deadline)
    files = source_files(source)
    before = {path.relative_to(source).as_posix(): file_hash(path) for path in files}
    core_before = file_hash(core)
    oracle_before = file_hash(oracle) if oracle is not None else None
    destination.mkdir(parents=True)
    for path in files:
        remaining_seconds(deadline)
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    target_core = destination / CORE_RELATIVE_PATH
    target_core.parent.mkdir(parents=True, exist_ok=True)
    remaining_seconds(deadline)
    shutil.copyfile(core, target_core)
    if oracle is not None:
        remaining_seconds(deadline)
        shutil.copyfile(oracle, destination / ORACLE_RELATIVE_PATH)
    (destination / "runs/active_core.json").write_text(
        json.dumps({"checkpoint": CORE_RELATIVE_PATH, "sha256": core_before,
                    "reason": "isolated search snapshot; never active in shared checkout"}, indent=2),
        encoding="utf-8")
    binding = source_binding(destination)
    after = {path.relative_to(source).as_posix(): file_hash(path) for path in source_files(source)}
    if before != after or binding["source_files"] != before or file_hash(core) != core_before:
        raise RuntimeError("source or core changed while copying; snapshot is unusable")
    if binding["core_sha256"] != core_before:
        raise RuntimeError("copied core does not match its source")
    if oracle is not None and (file_hash(oracle) != oracle_before or binding["oracle_sha256"] != oracle_before):
        raise RuntimeError("ORACLE changed while copying or copied artifact does not match its source")
    return binding


def run_command(command: list[str], *, workspace: Path, deadline: float,
                log_name: str = "command") -> subprocess.CompletedProcess:
    """Run one owned child with a hard remaining-time limit and local temp files."""
    remaining_seconds(deadline)
    temporary = workspace / ".tmp"
    temporary.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TORCH_NUM_THREADS": "2",
                "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0",
                "PYTHONPATH": str(workspace), "PYTHONDONTWRITEBYTECODE": "1",
                "TEMP": str(temporary), "TMP": str(temporary), "TMPDIR": str(temporary)})
    process = subprocess.Popen(command, cwd=workspace, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    stdout, stderr = "", ""
    try:
        stdout, stderr = process.communicate(timeout=remaining_seconds(deadline))
    except (TimeoutError, subprocess.TimeoutExpired, KeyboardInterrupt):
        # Only the child created above is stopped; unrelated user jobs are never touched.
        process.kill()
        stdout, stderr = process.communicate()
        raise
    finally:
        logs = workspace / "logs"
        logs.mkdir(exist_ok=True)
        (logs / f"{log_name}.stdout.log").write_text(stdout, encoding="utf-8")
        (logs / f"{log_name}.stderr.log").write_text(stderr, encoding="utf-8")
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def run_tests(workspace: Path, *, deadline: float) -> bool:
    result = run_command([sys.executable, "-m", "pytest", *TEST_FILES, "-q",
                          "--basetemp", str(workspace / ".test-temp"),
                          "-o", "cache_dir=.pytest_cache"],
                         workspace=workspace, deadline=deadline, log_name="tests")
    return result.returncode == 0


def run_benchmark(workspace: Path, *, seeds: list[int], deadline: float,
                  label: str = "selection", oracle_path: Path | None = None) -> dict:
    """Fresh agent/core per episode; retain raw outcomes for every specified seed."""
    if oracle_path is not None:
        oracle_path = Path(oracle_path).resolve()
        if not oracle_path.is_relative_to(workspace.resolve()) or not oracle_path.is_file():
            raise ValueError("benchmark ORACLE must be an existing workspace-local copied artifact")
    checkpoint = str(oracle_path) if oracle_path is not None else None
    script = f"""
import json, random, torch
from collections import Counter
from poseidon.cognitive_agent import CognitiveAgent
from poseidon.core import CoreRuntime
from poseidon.worldlab import POMDPTidePool, POMDPConfig

torch.set_num_threads(2)
rows = []
for seed in {seeds!r}:
    random.seed(seed)
    torch.manual_seed(seed)
    agent = CognitiveAgent(core=CoreRuntime({CORE_RELATIVE_PATH!r}), oracle_checkpoint={checkpoint!r})
    env = POMDPTidePool(POMDPConfig(seed=seed, scarcity={EVAL_SCARCITY}, sensor_noise_std=0.02,
                                  fog_threshold=0.65), max_steps={EVAL_MAX_STEPS})
    outcome = agent.rollout(env, max_steps={EVAL_MAX_STEPS})
    decisions = outcome['decisions']
    if len(decisions) != outcome['steps'] or any(type(d['planning_enabled']) is not bool for d in decisions):
        raise ValueError('benchmark requires complete planning telemetry')
    enabled = sum(d['planning_enabled'] for d in decisions)
    fallback = dict(Counter(d.get('fallback_reason') or 'unspecified_fallback'
                            for d in decisions if not d['planning_enabled']))
    rows.append({{'seed': seed, 'total_reward': outcome['total_reward'], 'survived': outcome['survived'],
                 'steps': outcome['steps'], 'overrides': outcome['overrides'],
                 'planning_enabled_steps': enabled, 'fallback_reasons': fallback, 'trajectory': outcome}})
print('__RESULT__' + json.dumps({{'per_seed': rows}}, allow_nan=False))
"""
    result = run_command([sys.executable, "-c", script], workspace=workspace,
                         deadline=deadline, log_name=label)
    if result.returncode:
        raise RuntimeError(f"benchmark failed in {workspace}; see logs/{label}.stderr.log")
    outputs = [line[len("__RESULT__"):] for line in result.stdout.splitlines()
               if line.startswith("__RESULT__")]
    if len(outputs) != 1:
        raise ValueError("benchmark did not produce exactly one result")
    rows = json.loads(outputs[0])["per_seed"]
    if [row["seed"] for row in rows] != seeds or not seeds:
        raise ValueError("benchmark results do not match requested seeds")
    for row in rows:
        if (not math.isfinite(row["total_reward"]) or type(row["survived"]) is not bool
                or type(row["steps"]) is not int or type(row["overrides"]) is not int
                or not 0 <= row["overrides"] <= row["steps"] <= EVAL_MAX_STEPS
                or type(row["planning_enabled_steps"]) is not int
                or not 0 <= row["planning_enabled_steps"] <= row["steps"]
                or not isinstance(row["fallback_reasons"], dict)
                or any(not isinstance(reason, str) or type(count) is not int or count < 1
                       for reason, count in row["fallback_reasons"].items())
                or sum(row["fallback_reasons"].values()) + row["planning_enabled_steps"] != row["steps"]):
            raise ValueError("benchmark contains invalid or nonfinite outcomes")
    rewards = [row["total_reward"] for row in rows]
    fallback_reasons = {}
    for row in rows:
        for reason, count in row["fallback_reasons"].items():
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + count
    return {"per_seed": rows, "rewards": rewards, "mean_reward": sum(rewards) / len(rows),
            "survival_rate": sum(row["survived"] for row in rows) / len(rows),
            "overrides": sum(row["overrides"] for row in rows),
            "planning_enabled_steps": sum(row["planning_enabled_steps"] for row in rows),
            "fallback_reasons": fallback_reasons,
            "total_steps": sum(row["steps"] for row in rows)}


def read_param_value(spec: dict, workspace: Path) -> float:
    content = (workspace / spec["file"]).read_text(encoding="utf-8")
    matches = list(re.finditer(spec["pattern"], content))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one parameter {spec['name']} in {spec['file']}")
    return float(matches[0].group(2))


def write_param_value(spec: dict, val: float, workspace: Path) -> None:
    read_param_value(spec, workspace)
    path = workspace / spec["file"]
    content = path.read_text(encoding="utf-8")
    path.write_text(re.sub(spec["pattern"], lambda m: m.group(1) + spec["fmt"].format(val),
                           content, count=1), encoding="utf-8")


def propose_mutation(workspace: Path, rng: random.Random) -> dict:
    spec = rng.choice(SEARCH_SPACE)
    current = read_param_value(spec, workspace)
    if rng.random() < 0.70:
        scale = rng.uniform(0.10, 0.25)
        direction = 1.0 if rng.random() < 0.5 else -1.0
        value = current + direction * scale * (spec["max"] - spec["min"])
    else:
        value = rng.uniform(spec["min"], spec["max"])
    value = float(spec["fmt"].format(max(spec["min"], min(spec["max"], value))))
    return {"name": spec["name"], "file": spec["file"], "old": current, "new": value}


def comparison(baseline: dict, candidate: dict) -> dict:
    left, right = baseline["per_seed"], candidate["per_seed"]
    if [row["seed"] for row in left] != [row["seed"] for row in right]:
        raise ValueError("paired comparison requires identical ordered seeds")
    deltas = [{"seed": a["seed"], "reward": b["total_reward"] - a["total_reward"],
               "survival": int(b["survived"]) - int(a["survived"])} for a, b in zip(left, right)]
    reward = sum(row["reward"] for row in deltas) / len(deltas)
    survival = sum(row["survival"] for row in deltas) / len(deltas)
    improved = (survival > 0 and reward >= -0.05) or (survival >= 0 and reward > 0.005)
    return {"delta_reward": reward, "delta_survival": survival, "per_seed_deltas": deltas,
            "improvement_supported": improved}


def parse_target_end_time(until_str: str | None, hours: float | None) -> datetime:
    now = datetime.now()
    if hours is not None:
        if not math.isfinite(hours) or hours <= 0:
            raise ValueError("duration-hours must be finite and positive")
        try:
            return now + timedelta(hours=hours)
        except OverflowError as error:
            raise ValueError("duration-hours is too large") from error
    parts = (until_str or "06:00").split(":")
    if len(parts) > 2:
        raise ValueError("until must use HH:MM")
    target = now.replace(hour=int(parts[0]), minute=int(parts[1]) if len(parts) == 2 else 0,
                         second=0, microsecond=0)
    return target if target > now else target + timedelta(days=1)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_search(*, source_root: Path = ROOT, output_root: Path = OUT_DIR,
               end_time: datetime, seed: int = 1729, max_iterations: int | None = None,
               core_path: Path | None = None, oracle_path: Path | None = None) -> dict:
    """Search frozen copies with 20% of the time reserved for the final audit.

    The source tree may already be dirty. It is never an execution workspace or
    an incumbent destination. Selection can overfit its four reused seeds; audit
    is one descriptive four-seed check, not a generalization or significance claim.
    Explicit ORACLE selection is required to exercise predictive parameters;
    missing artifacts or an entirely disabled planner stop before proposals.
    """
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if max_iterations is not None and (type(max_iterations) is not int or max_iterations < 1):
        raise ValueError("max_iterations must be a positive integer")
    if set(SELECTION_SEEDS) & set(AUDIT_SEEDS):
        raise ValueError("selection and audit seeds must be disjoint")
    for name in (*SOURCE_DIRS, "runs", ".git"):
        if output_root.is_relative_to(source_root / name):
            raise ValueError("output directory must not be inside source, tests, runs, or git metadata")
    now = time.monotonic()
    budget = max(0.0, (end_time - datetime.now()).total_seconds())
    deadline, selection_deadline = now + budget, now + budget * 0.8
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="search-" + datetime.now().strftime("%Y%m%d-%H%M%S-"),
                                    dir=output_root))
    baseline = run_dir / "baseline"
    result = {"schema": "poseidon-isolated-parameter-search-v1", "status": "running", "seed": seed,
              "started_at": datetime.now().isoformat(), "deadline": end_time.isoformat(),
              "selection_budget_fraction": 0.8, "source_root": str(source_root),
              "baseline_workspace": str(baseline), "selection_seeds": list(SELECTION_SEEDS),
              "audit_seeds": list(AUDIT_SEEDS), "selected_candidate": None, "candidates": [],
              "protocol": {"max_steps": EVAL_MAX_STEPS, "scarcity": EVAL_SCARCITY,
                           "sensor_noise_std": 0.02, "fog_threshold": 0.65, "torch_threads": 2,
                           "episode_reset": "fresh_core_and_agent",
                           "search_source_sha256": file_hash(Path(__file__))},
              "audit": {"status": "not_run", "reason": "no selected candidate"},
              "report_path": str(run_dir / "report.json"), "active": False,
              "interpretation": "Selection reuses fixed seeds and may overfit. Disjoint audit seeds are used "
                                "once after selection; they are a small descriptive check, not statistical "
                                "proof of general improvement. All candidates remain inactive."}
    current = None
    rng = random.Random(seed)
    try:
        remaining_seconds(selection_deadline)
        if oracle_path is None:
            raise InactiveSearchError("No trained ORACLE selected; use --oracle to enable predictive parameter search.")
        source_oracle = Path(oracle_path)
        source_oracle = (source_oracle if source_oracle.is_absolute() else source_root / source_oracle).resolve()
        if not source_oracle.is_file():
            raise ValueError(f"selected ORACLE artifact does not exist: {source_oracle}")
        source_core = resolve_core(source_root, core_path)
        binding = copy_workspace(source_root, baseline, source_core, selection_deadline, oracle=source_oracle)
        result["baseline_binding"] = binding
        result["core_origin"] = str(source_core)
        result["oracle_origin"] = str(source_oracle)
        incumbent, incumbent_binding = baseline, binding
        base = run_benchmark(baseline, seeds=SELECTION_SEEDS, deadline=selection_deadline,
                             oracle_path=baseline / ORACLE_RELATIVE_PATH)
        if source_binding(baseline) != binding:
            raise RuntimeError("baseline source, frozen core or ORACLE changed during evaluation")
        result["baseline_selection"] = base
        if base["planning_enabled_steps"] == 0:
            raise InactiveSearchError("Planning was disabled for every baseline step; predictive parameter search is inert. "
                                      f"Fallback reasons: {base['fallback_reasons']}")
        incumbent_result = base
        iteration = 0
        while time.monotonic() < selection_deadline and (max_iterations is None or iteration < max_iterations):
            iteration += 1
            proposal = propose_mutation(incumbent, rng)
            current = {"iteration": iteration, "mutation": proposal, "status": "PREPARING", "active": False,
                       "parent_workspace": str(incumbent), "baseline_binding": incumbent_binding}
            result["candidates"].append(current)
            candidate = run_dir / f"candidate_{iteration:04d}"
            current["workspace"] = str(candidate)
            if proposal["old"] == proposal["new"]:
                current["status"] = "SKIPPED_UNCHANGED"
                current.pop("workspace")
                write_json(run_dir / f"receipt_{iteration:04d}.json", current)
                current = None
                continue
            copied_binding = copy_workspace(incumbent, candidate, incumbent / CORE_RELATIVE_PATH, selection_deadline,
                                             oracle=incumbent / ORACLE_RELATIVE_PATH)
            if copied_binding != incumbent_binding:
                raise RuntimeError("incumbent source, frozen core or ORACLE changed since selection")
            spec = next(item for item in SEARCH_SPACE if item["name"] == proposal["name"])
            write_param_value(spec, proposal["new"], candidate)
            candidate_binding = source_binding(candidate)
            current["candidate_binding"] = candidate_binding
            if not run_tests(candidate, deadline=selection_deadline):
                current["status"] = "REJECTED_TESTS_FAILED"
            else:
                candidate_result = run_benchmark(candidate, seeds=SELECTION_SEEDS, deadline=selection_deadline,
                                                oracle_path=candidate / ORACLE_RELATIVE_PATH)
                if source_binding(candidate) != candidate_binding:
                    raise RuntimeError("candidate source, frozen core or ORACLE changed during evaluation")
                if candidate_result["planning_enabled_steps"] == 0:
                    current["selection"] = {"incumbent": incumbent_result, "candidate": candidate_result}
                    current["status"] = "REJECTED_PLANNING_DISABLED"
                    raise InactiveSearchError("Planning was disabled for every candidate step; search stopped. "
                                              f"Fallback reasons: {candidate_result['fallback_reasons']}")
                paired = comparison(incumbent_result, candidate_result)
                current["selection"] = {"incumbent": incumbent_result, "candidate": candidate_result, **paired}
                current["status"] = "SELECTED_INACTIVE" if paired["improvement_supported"] else "REJECTED_NO_IMPROVEMENT"
                if paired["improvement_supported"]:
                    incumbent, incumbent_binding, incumbent_result = candidate, candidate_binding, candidate_result
                    result["selected_candidate"] = {"workspace": str(candidate), "binding": candidate_binding,
                                                    "iteration": iteration, "active": False}
            write_json(run_dir / f"receipt_{iteration:04d}.json", current)
            write_json(Path(result["report_path"]), result)
            current = None
        result["status"] = "completed"
    except InactiveSearchError as error:
        result["status"], result["reason"] = "inactive", str(error)
    except KeyboardInterrupt:
        result["status"] = "cancelled"
        if current is not None:
            current["status"] = "CANCELLED"
    except (TimeoutError, subprocess.TimeoutExpired):
        result["status"] = "timed_out"
        if current is not None:
            current["status"] = "TIMED_OUT"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        if current is not None:
            current["status"] = "FAILED"
            current["error"] = result["error"]
    if current is not None:
        write_json(run_dir / f"receipt_{current['iteration']:04d}.json", current)

    # Audit never resumes search, changes the selected candidate, or triggers promotion.
    if result["selected_candidate"] is not None and result["status"] != "cancelled":
        audit = result["audit"] = {"status": "running", "seeds": list(AUDIT_SEEDS),
                                    "baseline_binding": result["baseline_binding"],
                                    "candidate_binding": result["selected_candidate"]["binding"]}
        try:
            remaining_seconds(deadline)
            audit["baseline"] = run_benchmark(baseline, seeds=AUDIT_SEEDS, deadline=deadline, label="audit",
                                              oracle_path=baseline / ORACLE_RELATIVE_PATH)
            selected = Path(result["selected_candidate"]["workspace"])
            audit["candidate"] = run_benchmark(selected, seeds=AUDIT_SEEDS, deadline=deadline, label="audit",
                                               oracle_path=selected / ORACLE_RELATIVE_PATH)
            if source_binding(baseline) != result["baseline_binding"] or source_binding(selected) != audit["candidate_binding"]:
                raise RuntimeError("source, core or ORACLE changed during audit")
            if audit["baseline"]["planning_enabled_steps"] == 0 or audit["candidate"]["planning_enabled_steps"] == 0:
                raise InactiveSearchError("Audit planning was entirely disabled; improvement cannot be supported.")
            audit.update(comparison(audit["baseline"], audit["candidate"]))
            audit["status"] = "completed"
        except InactiveSearchError as error:
            audit["status"], audit["reason"] = "inactive", str(error)
        except KeyboardInterrupt:
            audit["status"], result["status"] = "cancelled", "cancelled"
        except (TimeoutError, subprocess.TimeoutExpired):
            audit["status"] = "timed_out"
        except Exception as error:
            audit["status"], audit["error"] = "failed", f"{type(error).__name__}: {error}"
    elif result["selected_candidate"] is not None:
        result["audit"] = {"status": "not_run", "reason": "search cancelled; selected artifact remains inactive"}
    result["finished_at"] = datetime.now().isoformat()
    write_json(Path(result["report_path"]), result)
    report = (f"# Poseidon isolated parameter search\n\nStatus: {result['status']}\n\n"
              f"{result.get('reason', '')}\n\n"
              f"Proposals: {len(result['candidates'])}. Audit: {result['audit']['status']}.\n\n"
              f"{result['interpretation']}\n\n"
              "The source checkout, Git HEAD and active core pointer were not promoted or restored.\n\n"
              "Full source/core/ORACLE bindings, raw per-seed results and candidate paths are in report.json.\n")
    (run_dir / "OVERNIGHT_REPORT.md").write_text(report, encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded isolated Poseidon parameter search; no automatic promotion")
    parser.add_argument("--until", default="06:00", help="Local stop time HH:MM (default 06:00)")
    parser.add_argument("--duration-hours", type=float, help="Optional duration in hours")
    parser.add_argument("--seed", type=int, default=1729, help="Deterministic proposal RNG seed")
    parser.add_argument("--max-iterations", type=int, help="Additional finite proposal limit")
    parser.add_argument("--core", type=Path, help="Explicit trained core; defaults to the active checkpoint")
    parser.add_argument("--oracle", type=Path, help="Explicit trained ORACLE; search stops inactive if omitted or planning is disabled")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    try:
        end_time = parse_target_end_time(args.until, args.duration_hours)
        if args.max_iterations is not None and args.max_iterations < 1:
            raise ValueError("max-iterations must be positive")
    except ValueError as error:
        parser.error(str(error))
    print(f"Isolated search ending at {end_time.isoformat()}; selected candidates stay inactive.", flush=True)
    result = run_search(source_root=ROOT, output_root=args.output_dir, end_time=end_time,
                        seed=args.seed, max_iterations=args.max_iterations, core_path=args.core,
                        oracle_path=args.oracle)
    print(f"Search {result['status']}; report: {result['report_path']}", flush=True)
    if result.get("reason"):
        print(result["reason"], flush=True)
    return {"failed": 1, "inactive": 2, "cancelled": 130}.get(result["status"], 0)


if __name__ == "__main__":
    raise SystemExit(main())
