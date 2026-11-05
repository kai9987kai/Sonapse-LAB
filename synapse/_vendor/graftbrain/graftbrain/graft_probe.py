"""Deterministic supervised graft-current probe, not an embodied-policy benchmark.

The teacher, target switch, finite candidate graph and random streams are
declared before fitting. Every requested seed and every recovery point is kept,
including cases where structural adaptation is worse than its fixed control.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np
import torch

from .plasticity import GraftPlasticity, PlasticityConfig


BRANCHES = ("sham", "disrupted_rewire", "disrupted_no_rewire")
OUTPUTS = (6, 7, 14, 15)
BATCH_SIZE = 64
EVAL_SIZE = 128
LEARNING_RATE = 0.04
DISRUPTION_FRACTION = 0.5


@dataclass
class _ProbeState:
    engine: GraftPlasticity
    weights: torch.nn.Parameter
    optimizer: torch.optim.Optimizer
    elig: torch.Tensor
    heb: torch.Tensor


def _graph():
    # Each direction has 6 source features x 2 destination currents. No base
    # edges or anatomical data enter this explicitly synthetic problem.
    return SimpleNamespace(
        n=16, names=["synthetic_A", "synthetic_B"],
        sites=[{"id": 0, "kind": "synthetic_bridge"}],
        origin=np.repeat([0, 1], 8),
        graft_pre=np.tile(np.arange(6), 4) + np.repeat([0, 0, 8, 8], 6),
        graft_post=np.repeat([14, 15, 6, 7], 6),
        graft_site=np.zeros(24, dtype=np.int64),
        graft_w0=np.where(np.arange(24) % 2, -0.1, 0.1).astype(np.float32),
    )


def _targets():
    before, after = torch.zeros(24), torch.zeros(24)
    before[[0, 2, 7, 9, 12, 14, 19, 21]] = torch.tensor([0.8, -0.6, 1.0, -0.7] * 2)
    after[[1, 4, 6, 10, 13, 16, 18, 22]] = torch.tensor([-0.9, 0.7, -0.8, 1.0] * 2)
    return before, after


def _config():
    return PlasticityConfig(mode="structural", density=0.5, interval=10,
                            warmup=0, turnover=0.34, growth_rule="gradient",
                            growth_init="zero", min_age=20)


def _make_state(graph, seed):
    engine = GraftPlasticity(graph, obs_dim=1, config=_config(), seed=int(seed))
    weights = torch.nn.Parameter(torch.tensor(graph.graft_w0.copy()))
    return _ProbeState(engine, weights, torch.optim.Adam([weights], lr=LEARNING_RATE),
                       torch.zeros_like(weights), torch.zeros_like(weights))


def _clone_state(source):
    # Copy the optimizer and its parameter in the same memoized operation so its
    # state points at the cloned coefficient rather than the trained source.
    clone = deepcopy(source)
    if clone.optimizer.param_groups[0]["params"][0] is not clone.weights:
        raise RuntimeError("probe optimizer did not clone with its parameter")
    return clone


def _digest(value):
    digest = hashlib.sha256()

    def add(item):
        if torch.is_tensor(item):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(f"tensor:{array.dtype}:{array.shape}:".encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item, key=str):
                add(key)
                add(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(b"[")
            for part in item:
                add(part)
            digest.update(b"]")
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
            digest.update(b";")

    add(value)
    return digest.hexdigest()


def _state_digest(state):
    return _digest({"weights": state.weights, "gradient": state.weights.grad,
                    "engine": state.engine.state_dict(),
                    "optimizer": state.optimizer.state_dict(),
                    "elig": state.elig, "heb": state.heb})


def _current(graph, rates, effective):
    current = rates.new_zeros((len(rates), graph.n))
    current.index_add_(1, torch.as_tensor(graph.graft_post),
                       rates[:, graph.graft_pre] * effective)
    return current[:, OUTPUTS]


def _prediction(state, graph, rates, capture=False):
    # There is no context router in this focused edge-learning fixture. Gains
    # are still passed through the real mask/gate path used by the learner.
    gates = state.weights.new_ones(state.engine.n_sites)
    effective = state.engine.effective_weights(state.weights, gates)
    if capture:
        state.engine.capture_gradient(effective, gates)
    return _current(graph, rates, effective)


@torch.no_grad()
def _error(state, graph, rates, target):
    return float((_prediction(state, graph, rates) - _current(graph, rates, target)).square().mean())


def _bucket_counts(engine):
    counts = [int(engine.active_mask[engine.bucket_ids == bucket].sum())
              for bucket in range(len(engine.bucket_budgets))]
    if counts != engine.bucket_budgets.tolist():
        raise RuntimeError("probe changed a directional support budget")
    return counts


def _active(engine):
    return torch.nonzero(engine.active_mask, as_tuple=True)[0].tolist()


def _curve_row(state, graph, evaluation, target, seed, phase, branch, step, old_target=None):
    row = {"seed": seed, "phase": phase, "branch": branch, "step": step,
           "mse": _error(state, graph, evaluation, target),
           "active": int(state.engine.active_mask.sum()),
           "bucket_active": _bucket_counts(state.engine),
           "support_sha256": _digest(state.engine.active_mask)}
    if old_target is not None:
        row["old_target_mse"] = _error(state, graph, evaluation, old_target)
    return row


def _fit(state, graph, batches, target, evaluation, *, seed, phase, branch, allow_rewire,
         old_target=None):
    rows = [_curve_row(state, graph, evaluation, target, seed, phase, branch, 0, old_target)]
    events, max_dormant_gradient = [], 0.0
    for step, rates in enumerate(batches, 1):
        state.optimizer.zero_grad(set_to_none=True)
        state.engine.begin_gradient_update()
        prediction = _prediction(state, graph, rates, capture=True)
        loss = (prediction - _current(graph, rates, target)).square().mean()
        loss.backward()
        state.engine.finish_gradient_update()
        dormant_gradient = state.weights.grad[~state.engine.active_mask]
        if dormant_gradient.numel():
            max_dormant_gradient = max(max_dormant_gradient, float(dormant_gradient.abs().max()))
        state.optimizer.step()
        with torch.no_grad():
            state.elig.mul_(0.9).add_(state.weights.grad.abs())
            # Coactivity uses observable feature and current magnitudes, never
            # teacher coefficients. Traces are carried so interventions exercise
            # the real reset path; this probe applies no Hebbian weight update.
            observed = rates.abs().mean(dim=0)
            observed[list(OUTPUTS)] = prediction.detach().abs().mean(dim=0)
            state.engine.observe(observed)
        # Bracket the structural edit after ordinary coefficient learning.
        # Adjacent curve points would conflate that update with rewiring.
        pre_rewire_mse = _error(state, graph, evaluation, target)
        event = state.engine.after_update(state.weights, state.optimizer, state.elig,
                                         state.heb, allow_rewire=allow_rewire)
        row = _curve_row(state, graph, evaluation, target, seed, phase, branch, step, old_target)
        if event is not None:
            events.append({"step": step, **event,
                           "pre_rewire_mse": pre_rewire_mse,
                           "post_rewire_mse": row["mse"],
                           "rewire_mse_delta": row["mse"] - pre_rewire_mse})
        rows.append(row)
    return rows, events, max_dormant_gradient


def _trajectory_metrics(rows):
    """Summarize every recovery point, retaining both trapezoid endpoints."""
    duration = rows[-1]["step"] - rows[0]["step"]
    area = sum((left["mse"] + right["mse"]) * (right["step"] - left["step"]) / 2.0
               for left, right in zip(rows, rows[1:]))
    tail_count = min(10, len(rows) - 1)
    return {"recovery_auc_mean": float(area / duration),
            "tail_mean_error": float(np.mean([row["mse"] for row in rows[-tail_count:]])),
            "tail_update_count": tail_count, "recovery_steps": duration,
            "final_error": rows[-1]["mse"]}


def _paired_metric(results, metric):
    differences = [row["branches"]["disrupted_no_rewire"][metric] -
                   row["branches"]["disrupted_rewire"][metric] for row in results]
    return {"metric": metric,
            "definition": f"fixed-support {metric} minus rewiring {metric} (positive favors rewiring)",
            "per_seed": [{"seed": row["seed"], "difference": difference}
                         for row, difference in zip(results, differences)],
            "mean_difference": float(np.mean(differences)),
            "rewiring_lower_error_seeds": sum(value > 0 for value in differences),
            "rewiring_higher_error_seeds": sum(value < 0 for value in differences),
            "equal_error_seeds": sum(value == 0 for value in differences)}


def _data(seed, steps, stream):
    # SeedSequence entropy namespaces isolate data and intervention randomness
    # from engine growth, global NumPy/PyTorch state and one another.
    generator = np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))
    values = generator.standard_normal((steps, BATCH_SIZE, 12)).astype(np.float32)
    rates = torch.zeros((steps, BATCH_SIZE, 16))
    rates[:, :, :6], rates[:, :, 8:14] = torch.from_numpy(values[:, :, :6]), torch.from_numpy(values[:, :, 6:])
    return rates


def _run_seed(seed, train_steps, recovery_steps, graph):
    before, after = _targets()
    training = _data(seed, train_steps, 101)
    recovery = _data(seed, recovery_steps, 102)
    evaluation = _data(seed, EVAL_SIZE // BATCH_SIZE, 103).reshape(EVAL_SIZE, graph.n)
    intervention_seed = int(np.random.SeedSequence([seed, 104]).generate_state(1, dtype=np.uint32)[0])
    source = _make_state(graph, seed)
    curves, source_events, dormant_gradient = _fit(
        source, graph, training, before, evaluation, seed=seed, phase="training",
        branch="source", allow_rewire=True)
    source_hash = _state_digest(source)
    result = {
        "seed": seed, "intervention_seed": intervention_seed,
        "training_data_sha256": _digest(training), "recovery_data_sha256": _digest(recovery),
        "evaluation_data_sha256": _digest(evaluation),
        "source_state_before_branches": source_hash,
        "source": {"error_before_training": curves[0]["mse"],
                   "error_before_switch": curves[-1]["mse"],
                   "error_after_target_switch": _error(source, graph, evaluation, after),
                   "weights": source.weights.detach().tolist(),
                   "active_indices": _active(source.engine), "support_events": source_events,
                   "gradient_updates": int(source.engine.gradient_update_count),
                   "gradient_positive_candidates": int((source.engine.gradient_ema > 0).sum()),
                   "max_dormant_raw_gradient": dormant_gradient},
        "branches": {},
    }
    for branch in BRANCHES:
        state = _clone_state(source)
        intervention = None
        if branch != "sham":
            intervention = state.engine.perturb_support(
                state.weights, state.optimizer, state.elig, state.heb,
                fraction=DISRUPTION_FRACTION, seed=intervention_seed)
        initial_active = _active(state.engine)
        initial_age = state.engine.edge_age.tolist()
        initial_buckets = _bucket_counts(state.engine)
        newborn_max = (float(state.weights[intervention["grown"]].detach().abs().max())
                       if intervention and intervention["grown"] else 0.0)
        rows, events, max_raw = _fit(
            state, graph, recovery, after, evaluation, seed=seed, phase="recovery",
            branch=branch, allow_rewire=branch != "disrupted_no_rewire", old_target=before)
        curves.extend(rows)
        result["branches"][branch] = {
            "allow_rewire": branch != "disrupted_no_rewire", "intervention": intervention,
            "initial_active_indices": initial_active, "final_active_indices": _active(state.engine),
            "initial_edge_age": initial_age, "final_edge_age": state.engine.edge_age.tolist(),
            "initial_bucket_active": initial_buckets, "final_bucket_active": _bucket_counts(state.engine),
            "newborn_max_abs_weight": newborn_max, "immediate_error": rows[0]["mse"],
            **_trajectory_metrics(rows),
            "error_reduction": rows[0]["mse"] - rows[-1]["mse"],
            "old_target_tradeoff": {
                "initial_error": rows[0]["old_target_mse"],
                "final_error": rows[-1]["old_target_mse"],
                "source_end_error": result["source"]["error_before_switch"],
                "change_from_recovery_start": rows[-1]["old_target_mse"] - rows[0]["old_target_mse"],
                "change_from_source_end": rows[-1]["old_target_mse"] - result["source"]["error_before_switch"],
            },
            "learned_swaps": sum(event["swaps"] for event in events), "support_events": events,
            "gradient_updates": int(state.engine.gradient_update_count),
            "max_dormant_raw_gradient": max_raw, "weights": state.weights.detach().tolist(),
            "final_state_sha256": _state_digest(state),
        }
    result["source_state_after_branches"] = _state_digest(source)
    if result["source_state_after_branches"] != source_hash:
        raise RuntimeError("probe recovery mutated the trained source")
    result["source_unchanged"] = True
    return result, curves


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("graftbrain/graft_probe.py", "graftbrain/plasticity.py")}


def run_graft_probe(out, seeds=(0, 1, 2), train_steps=80, recovery_steps=80):
    """Write a new JSON evidence directory and return its resolved ``Path``.

    No checkpoint is loaded or written. The sham branch receives the same
    declared target switch and recovery data as both disrupted branches.
    Recovery-disabled means structural swaps are disabled, while gradient
    collection, ordinary coefficient learning and connection aging continue.
    """
    output = Path(out).resolve()
    if output.exists():
        raise FileExistsError(f"probe output already exists: {output}")
    for name, value in (("train_steps", train_steps), ("recovery_steps", recovery_steps)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    try:
        seeds = list(seeds)
    except TypeError as error:
        raise ValueError("seeds must be unique nonnegative integers") from error
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, Integral)
                        or not 0 <= seed < 2**64 for seed in seeds):
        raise ValueError("seeds must be unique nonnegative integers below 2**64")
    seeds = [int(seed) for seed in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    graph = _graph()
    before, after = _targets()
    manifest = {
        "schema_version": 2, "status": "running", "probe": "supervised_graft_current_recovery",
        "compatibility": "Additive measurements; schema-1 result fields and endpoint comparison are retained.",
        "scope": "Deterministic finite synthetic supervised mechanism probe; not RL performance or biological evidence.",
        "parameters": {"seeds": seeds, "train_steps": int(train_steps), "recovery_steps": int(recovery_steps),
                       "batch_size": BATCH_SIZE, "evaluation_samples": EVAL_SIZE,
                       "optimizer": "Adam", "learning_rate": LEARNING_RATE,
                       "disruption_fraction": DISRUPTION_FRACTION, "plasticity": asdict(_config())},
        "graph": {"n": graph.n, "names": graph.names, "origin": graph.origin.tolist(),
                  "pre": graph.graft_pre.tolist(), "post": graph.graft_post.tolist(),
                  "site": graph.graft_site.tolist(), "initial_weights": graph.graft_w0.tolist(),
                  "candidates": 24, "active": 12, "directional_budgets": [6, 6], "outputs": list(OUTPUTS)},
        "targets": {"before": before.tolist(), "after": after.tolist(),
                    "definition": "output current = sum(input_feature[pre] * target_edge_coefficient) at each post",
                    "switch": "after source training, before intervention and every recovery update",
                    "loss": "mean squared current error over the four outputs"},
        "randomness": {"generator": "NumPy default_rng with SeedSequence([seed, stream])",
                       "streams": {"training": 101, "recovery": 102, "evaluation": 103, "intervention": 104},
                       "inputs": "independent standard normal, float32; 12 observable source features",
                       "pairing": "all branches clone one trained source and share identical recovery inputs and evaluation set"},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "source_sha256": _source_hashes(),
        "metrics": {
            "primary": "recovery_auc_mean",
            "secondary": ["tail_mean_error", "final_error"],
            "recovery_auc_mean": "Trapezoidal new-target MSE area over all recovery points 0..T divided by T updates.",
            "tail_mean_error": "Mean new-target MSE over the last min(10,T) recovery updates, excluding step 0; tail_update_count records the window size.",
            "final_error": "New-target MSE after recovery update T, including any structural edit at that update.",
            "paired_difference": "Fixed-support minus adaptive for each declared seed; positive favors adaptive. All seeds retained.",
            "rewire_mse_delta": "Post-rewire minus pre-rewire MSE for the phase's optimized target: source training uses the before-switch target; recovery uses the after-switch target. Both measurements follow optimizer.step and bracket only after_update. Positive means the edit increases that phase's target error.",
            "old_target_mse": "Evaluation-only MSE at every recovery point; never part of the optimized objective or selection scores.",
            "old_target_tradeoff": "Terminal old-target error and changes from recovery start and trained source. Identical inputs receive incompatible old/new labels, creating a necessary fitting tradeoff; this is not general forgetting or contextual retention evidence.",
        },
        "limitations": ["Finite fixed candidate pool and one hand-declared target switch.",
                        "No router, recurrent brain, policy, environment, reward or biological data is tested.",
                        "Zero-initialized newborns contribute no immediate current; dropped records can change current.",
                        "Old/new labels conflict on the same inputs without a context cue; their error tradeoff does not measure general forgetting.",
                        "Sensitivity scores and support movement do not guarantee improved recovery."],
    }
    # Claim the output atomically before computation; never overwrite artifacts.
    output.mkdir(parents=True, exist_ok=False)
    results, curves = [], []

    def write(name, data):
        (output / f"{name}.json").write_text(
            json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    write("manifest", manifest)
    try:
        for seed in seeds:
            result, rows = _run_seed(seed, int(train_steps), int(recovery_steps), graph)
            results.append(result)
            curves.extend(rows)
        endpoint = _paired_metric(results, "final_error")
        comparison = {key: value for key, value in endpoint.items() if key != "metric"}
        comparison["definition"] = "fixed-support final MSE minus rewiring final MSE (positive favors rewiring)"
        comparison["primary"] = _paired_metric(results, "recovery_auc_mean")
        comparison["secondary"] = {"tail_mean_error": _paired_metric(results, "tail_mean_error"),
                                   "final_error": endpoint}
        summary = {"schema_version": 2, "results": results, "comparison": comparison,
                   "claim": "Observed current-error recovery in this synthetic fixture only; all requested seeds retained."}
        write("curves", {"schema_version": 2, "rows": curves})
        write("summary", summary)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["source_sha256_end"] = _source_hashes()
        manifest["source_changed_during_run"] = manifest["source_sha256_end"] != manifest["source_sha256"]
        write("manifest", manifest)
    return output
