"""Ten-controller experiments with cognitive spatial mapping, multi-horizon advantage audits, and portable replay.

Evaluator can fork simulator snapshots; controllers receive observations only.
Compares Odyssey against Horizon Atlas, Contrast Atlas, Atlas v2, Neural MPC, Policy, Heuristic, and Random.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean
import time

import torch

from .atlas import _base, _model_digest
from .experiments import ExperimentSpec, digest, source_identity, write_receipt
from .world import ACTIONS, TidePool, WORLD_VERSION, event_uniform, teacher_action

SCHEMA = "poseidon-odyssey-experiment-v1"
ARMS = (
    "policy", "neural_mpc", "atlas_v2", "contrast", "horizon",
    "odyssey", "odyssey_no_memory", "odyssey_ungated",
    "heuristic", "random"
)
MAX_TRANSITIONS = 16000
MAX_RECEIPT_BYTES = 128 * 1024 * 1024
LIMITS = [
    "Synthetic TidePool development evidence; no real-world or robotics inference.",
    "Odyssey uses episodic cognitive mapping and replenishment memory purely from 16-d observation streams.",
    "Episode-max advantage error radii are descriptive, not conformal coverage guarantees.",
    "Multi-horizon rollouts assume the frozen core policy acts for subsequent steps.",
    "Controller timings exclude counterfactual auditing and rollouts; controllers have unequal compute budgets.",
    "Replay verifies simulator execution and evidence arithmetic without model weights, not model authorship.",
    "No weights, incumbent pointer or default planner are changed or promoted by an experiment.",
]


def _spec(spec):
    spec = spec or ExperimentSpec(seed=108000001)
    if not isinstance(spec, ExperimentSpec):
        raise ValueError("spec must be ExperimentSpec")
    if spec.episodes * spec.max_steps * len(ARMS) > MAX_TRANSITIONS:
        raise ValueError("Odyssey experiment exceeds the 16000-transition budget")
    return spec


def _vector(value, size=16, *, unit=True):
    if not isinstance(value, list) or len(value) != size or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        or (unit and not 0 <= x <= 1) for x in value
    ):
        raise ValueError("Invalid recorded vector")
    return value


def _utility(observation, probability):
    from .contrast import reserve_utility
    return reserve_utility(observation, probability, policy_weight=0.15, margin_penalty=2.0)


def _audit(env, probabilities, decision):
    snapshot = env.snapshot()
    successors = []
    for action in range(6):
        branch = TidePool.from_snapshot(snapshot)
        successors.append(branch.step(action)[0])
    utilities = [_utility(nxt, probabilities[a]) for a, nxt in enumerate(successors)]
    policy = decision["policy_action"]
    action = decision["action"]
    best = max(range(6), key=lambda a: (utilities[a], -a))
    proposed = decision.get("proposed_action", action)
    actual_adv = utilities[action] - utilities[policy]
    actual_prop_adv = utilities[proposed] - utilities[policy]
    margin_exceeded = False
    if "advantage_error_radius" in decision and "point_advantage" in decision:
        margin_exceeded = bool(decision["point_advantage"] - actual_prop_adv > decision["advantage_error_radius"])
    return {
        "next_observations": successors,
        "utilities": utilities,
        "best_action": best,
        "actual_advantage": actual_adv,
        "actual_proposed_advantage": actual_prop_adv,
        "one_step_regret": utilities[best] - utilities[action],
        "override": action != policy,
        "margin_exceeded": margin_exceeded,
    }


def _prediction(decision, arm, core, observation):
    if arm in ("atlas_v2", "contrast"):
        return decision["candidates"][decision["action"]]["predicted_observation"]
    if arm == "neural_mpc":
        return decision["predicted_futures"][ACTIONS[decision["action"]]]["step_1_observation"]
    predicted, _ = _base(core, [observation], [decision["action"]])
    return predicted[0].tolist()


def _row(episode):
    trace = episode["trajectory"]
    count = len(trace)
    overrides = [event for event in trace if event["audit"]["override"]]
    channel = [fmean(event["prediction_squared_errors"][i] for event in trace) for i in range(16)]

    max_patches = max((event["decision"].get("cognitive_map", {}).get("discovered_patches", 0) for event in trace if isinstance(event.get("decision"), dict)), default=0)
    pretransit_rests = sum(bool(event["decision"].get("pretransit_rest")) for event in trace if isinstance(event.get("decision"), dict))
    waypoints_nav = sum(bool(event["decision"].get("best_waypoint")) for event in trace if isinstance(event.get("decision"), dict))

    return {
        "arm": episode["controller"],
        "seed": episode["seed"],
        "steps": count,
        "survived": episode["survived"],
        "reward": episode["reward"],
        "death_reason": episode["death_reason"],
        "final_health": episode["final_health"],
        "decision_ms": sum(event["decision_ms"] for event in trace),
        "audit_ms": sum(event["audit_ms"] for event in trace),
        "fallback_steps": sum(bool(event["decision"].get("fallback_reason") in (
            "outside_fitted_support", "insufficient_margin"
        )) for event in trace),
        "overrides": len(overrides),
        "discovered_patches": max_patches,
        "pretransit_rests": pretransit_rests,
        "waypoints_evaluated": waypoints_nav,
        "per_channel_mse": channel,
        "prediction_mse": fmean(channel),
        "counterfactual_audit": {
            "anchors": count,
            "overrides_audited": len(overrides),
            "mean_actual_advantage_at_overrides": (
                fmean(event["audit"]["actual_advantage"] for event in overrides) if overrides else None
            ),
            "adverse_overrides": sum(event["audit"]["actual_advantage"] < 0 for event in overrides),
            "mean_one_step_regret": fmean(event["audit"]["one_step_regret"] for event in trace),
            "ranking_accuracy": fmean(event["action"] == event["audit"]["best_action"] for event in trace),
            "paired_margin_exceedances": sum(event["audit"]["margin_exceeded"] for event in trace),
        },
    }


def paired_summary(rows):
    summaries, paired = {}, {}
    baseline = {row["seed"]: row for row in rows if row["arm"] == "policy"}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        summaries[arm] = {
            "episodes": len(subset),
            "survival_rate": fmean(row["survived"] for row in subset),
            "mean_steps": fmean(row["steps"] for row in subset),
            "mean_reward": fmean(row["reward"] for row in subset),
            "mean_decision_ms": fmean(row["decision_ms"] for row in subset),
            "mean_audit_ms": fmean(row["audit_ms"] for row in subset),
            "mean_overrides": fmean(row["overrides"] for row in subset),
            "mean_prediction_mse": fmean(row["prediction_mse"] for row in subset),
            "mean_one_step_regret": fmean(row["counterfactual_audit"]["mean_one_step_regret"] for row in subset),
            "ranking_accuracy": fmean(row["counterfactual_audit"]["ranking_accuracy"] for row in subset),
            "total_adverse_overrides": sum(row["counterfactual_audit"]["adverse_overrides"] for row in subset),
            "total_paired_margin_exceedances": sum(row["counterfactual_audit"]["paired_margin_exceedances"] for row in subset),
            "mean_discovered_patches": fmean(row["discovered_patches"] for row in subset),
            "total_pretransit_rests": sum(row["pretransit_rests"] for row in subset),
            "total_waypoints_evaluated": sum(row["waypoints_evaluated"] for row in subset),
        }
        if arm != "policy":
            paired[f"{arm}_vs_policy"] = {
                "survival_rate_diff": summaries[arm]["survival_rate"] - summaries["policy"]["survival_rate"],
                "mean_steps_diff": summaries[arm]["mean_steps"] - summaries["policy"]["mean_steps"],
                "mean_reward_diff": summaries[arm]["mean_reward"] - summaries["policy"]["mean_reward"],
                "mean_regret_diff": summaries[arm]["mean_one_step_regret"] - summaries["policy"]["mean_one_step_regret"],
                "ranking_accuracy_diff": summaries[arm]["ranking_accuracy"] - summaries["policy"]["ranking_accuracy"],
                "per_seed": [
                    {
                        "seed": row["seed"],
                        "survival_diff": int(row["survived"]) - int(baseline[row["seed"]]["survived"]),
                        "steps_diff": row["steps"] - baseline[row["seed"]]["steps"],
                        "reward_diff": row["reward"] - baseline[row["seed"]]["reward"],
                    }
                    for row in subset
                ],
            }
    return summaries, paired


def _episode(core, atlas_v2, contrast, horizon, odyssey, mpc, arm, seed, spec):
    env = TidePool(seed=seed, scarcity=spec.scarcity, max_steps=spec.max_steps)
    initial = env.snapshot()
    trace = []
    counts = {name: 0 for name in ACTIONS}

    if arm.startswith("odyssey"):
        odyssey.reset(spec.scarcity)

    while not env.done:
        obs = env.observe()
        start = time.perf_counter()
        if arm == "odyssey":
            decision = odyssey.plan(obs, use_memory=True, gate="calibrated")
        elif arm == "odyssey_no_memory":
            decision = odyssey.plan(obs, use_memory=False, gate="calibrated")
        elif arm == "odyssey_ungated":
            decision = odyssey.plan(obs, use_memory=True, gate="uncalibrated")
        elif arm == "horizon":
            decision = horizon.plan(obs, use_memory=True, gate="calibrated")
        elif arm == "contrast":
            decision = contrast.plan(obs)
        elif arm == "atlas_v2":
            decision = atlas_v2.plan(obs)
        elif arm == "neural_mpc":
            decision = mpc.plan(obs)
        else:
            action = (
                core.act(obs) if arm == "policy" else teacher_action(obs) if arm == "heuristic"
                else min(5, int(event_uniform(seed, env.tick, "random-control", "action") * 6))
            )
            decision = {"action": action, "action_name": ACTIONS[action]}
        elapsed = (time.perf_counter() - start) * 1000
        action = decision["action"]
        if type(action) is not int or not 0 <= action < 6:
            raise ValueError("Controller returned an invalid action")

        audit_start = time.perf_counter()
        _, logits = _base(core, [obs], [action])
        probabilities = logits[0].softmax(-1).tolist()
        policy = int(logits[0].argmax())
        decision["policy_action"] = policy
        decision["policy_probabilities"] = probabilities

        audit = _audit(env, probabilities, decision)
        prediction = _prediction(decision, arm, core, obs)
        audit_ms = (time.perf_counter() - audit_start) * 1000

        nxt, reward, _, info = env.step(action)
        errors = [(a - b)**2 for a, b in zip(prediction, nxt)]
        event = {
            "observation": obs, "action": action, "action_name": ACTIONS[action],
            "decision": decision, "decision_ms": elapsed, "audit_ms": audit_ms,
            "audit": audit, "predicted_observation": prediction,
            "prediction_squared_errors": errors, "reward": reward,
            "next_observation": nxt, "info": info, "state": env.state(),
        }
        trace.append(event)
        counts[ACTIONS[action]] += 1

    episode = {
        "schema": "poseidon-episode-v1", "controller": arm, "world_version": WORLD_VERSION,
        "seed": seed, "scarcity": spec.scarcity, "max_steps": spec.max_steps, "steps": env.tick,
        "survived": env.alive, "alive": env.alive, "death": not env.alive, "death_reason": env.death_reason,
        "truncated": info["truncated"], "terminal_reason": info["terminal_reason"], "reward": env.total_reward,
        "final_health": env.health, "action_counts": counts, "initial_snapshot": initial,
        "final_snapshot": env.snapshot(), "trajectory": trace,
    }
    return _row(episode), episode


def run_experiment(core, atlas_v2, contrast, horizon, odyssey, spec=None):
    from .planning import ModelPredictivePlanner, PlanningConfig
    spec = _spec(spec)
    sources = source_identity()
    frozen_odyssey = odyssey.artifact
    frozen_horizon = horizon.artifact
    frozen_contrast = contrast.artifact
    legacy_atlas = atlas_v2.artifact
    checkpoint_hash = hashlib.sha256(core.path.read_bytes()).hexdigest()
    model_hash = _model_digest(core)

    for artifact in (frozen_odyssey, frozen_horizon, frozen_contrast, legacy_atlas):
        if artifact["checkpoint_sha256"] != checkpoint_hash or artifact["model_sha256"] != model_hash:
            raise ValueError("Experiment artifact does not match core checkpoint and model")

    seeds = set(range(spec.seed, spec.seed + spec.episodes))
    for artifact in (frozen_odyssey, frozen_horizon, frozen_contrast, legacy_atlas):
        used = set().union(*(set(values) for name, values in artifact["partition"].items() if name.endswith("_seeds")))
        if seeds & used:
            raise ValueError("Experiment seeds overlap a fitting or calibration partition")

    planning = PlanningConfig()
    protocol = {
        "schema": SCHEMA, "spec": asdict(spec), "arms": list(ARMS), "world_version": WORLD_VERSION,
        "source_sha256": sources, "checkpoint_sha256": checkpoint_hash, "model_sha256": model_hash,
        "atlas_v2_sha256": legacy_atlas["sha256"], "contrast_sha256": frozen_contrast["sha256"],
        "horizon_sha256": frozen_horizon["sha256"], "odyssey_sha256": frozen_odyssey["sha256"],
        "odyssey_config": frozen_odyssey["config"], "odyssey_partition": frozen_odyssey["partition"],
        "odyssey_calibration": frozen_odyssey["calibration"],
        "neural_mpc_config": asdict(planning), "randomization": "blake2b-named-events-v1",
        "horizon": frozen_odyssey["config"]["horizon"],
        "advantage_audit": "all six action interventions unrolled for H steps under frozen policy",
        "runtime": {"device": "cpu", "torch_threads": torch.get_num_threads()},
    }
    mpc = ModelPredictivePlanner(core, planning)
    rows, episodes = [], []
    for index, seed in enumerate(sorted(seeds)):
        for arm in ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)]:
            row, episode = _episode(core, atlas_v2, contrast, horizon, odyssey, mpc, arm, seed, spec)
            rows.append(row)
            episodes.append(episode)

    if (
        source_identity() != sources
        or hashlib.sha256(core.path.read_bytes()).hexdigest() != checkpoint_hash
        or _model_digest(core) != model_hash
        or odyssey.artifact != frozen_odyssey
        or horizon.artifact != frozen_horizon
        or contrast.artifact != frozen_contrast
        or atlas_v2.artifact != legacy_atlas
    ):
        raise RuntimeError("Source, checkpoint or fitted artifact changed during experiment; no receipt published")

    summary, paired = paired_summary(rows)
    result = {
        "schema": SCHEMA, "experiment_id": digest(protocol), "protocol": protocol,
        "summary": summary, "paired": paired, "rows": rows, "episodes": episodes,
        "odyssey": {
            "ready": True, "artifact_sha256": frozen_odyssey["sha256"],
            "fit_receipt": {
                "training_samples": len(frozen_odyssey["records"]),
                "partition": frozen_odyssey["partition"],
                "calibration": frozen_odyssey["calibration"],
            },
        },
        "limits": LIMITS, "status": "completed", "promoted": False,
    }
    result["receipt_sha256"] = digest(result)
    if len(json.dumps(result, allow_nan=False).encode()) > MAX_RECEIPT_BYTES:
        raise ValueError("Receipt exceeds the 128 MiB limit; use a smaller experiment")
    return result


def _validate_decision(event, arm, protocol):
    decision = event["decision"]
    action = event["action"]
    if not isinstance(decision, dict) or type(decision.get("action")) is not int or decision["action"] != action:
        raise ValueError("Invalid controller decision")
    if type(decision.get("policy_action")) is not int or not 0 <= decision["policy_action"] < 6:
        raise ValueError("Invalid policy action in decision")
    probabilities = _vector(decision.get("policy_probabilities"), 6)
    if abs(sum(probabilities) - 1) > 1e-5:
        raise ValueError("Invalid policy probabilities")
    if arm.startswith("odyssey") or arm.startswith("horizon"):
        if decision.get("override_accepted") != (action != decision["policy_action"]):
            raise ValueError("Override acceptance mismatch")
        if "margin" in decision and "point_advantage" in decision and "advantage_error_radius" in decision:
            if abs(decision["margin"] - (decision["point_advantage"] - decision["advantage_error_radius"])) > 1e-5:
                raise ValueError("Advantage margin arithmetic mismatch")
    return probabilities


def verify_receipt(receipt):
    try:
        return _verify(receipt)
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("Malformed odyssey receipt evidence") from error


def _verify(receipt):
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise ValueError("Unsupported odyssey receipt")
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if digest(unsigned) != receipt.get("receipt_sha256"):
        raise ValueError("Receipt checksum mismatch")
    if receipt.get("status") != "completed" or receipt.get("promoted") is not False:
        raise ValueError("Only completed, unpromoted odyssey experiments can be verified")
    protocol = receipt["protocol"]
    if protocol.get("schema") != SCHEMA or protocol.get("arms") != list(ARMS) or digest(protocol) != receipt["experiment_id"]:
        raise ValueError("Experiment protocol identity mismatch")

    current = source_identity()
    if any(protocol["source_sha256"].get(name) != current.get(name) for name in ("world.py", "odyssey.py", "odyssey_experiments.py")):
        raise ValueError("Replay requires recorded simulator, odyssey and experiment implementation")

    spec = _spec(ExperimentSpec(**protocol["spec"]))
    seeds = set(range(spec.seed, spec.seed + spec.episodes))
    episodes, rows = receipt["episodes"], receipt["rows"]
    if not isinstance(episodes, list) or not isinstance(rows, list):
        raise ValueError("Missing paired episodes or rows")

    seen, transitions = set(), 0
    verified_rows = []
    indexed = {(r["arm"], r["seed"]): r for r in rows}

    for episode in episodes:
        key = (episode["controller"], episode["seed"])
        seen.add(key)
        env = TidePool(episode["seed"], scarcity=spec.scarcity, max_steps=spec.max_steps)
        if env.snapshot() != episode["initial_snapshot"]:
            raise ValueError("Initial snapshot mismatch")
        trace = episode["trajectory"]
        counts = {name: 0 for name in ACTIONS}
        for event in trace:
            if env.done or event["observation"] != env.observe():
                raise ValueError("Observation or termination mismatch")
            action = event["action"]
            probabilities = _validate_decision(event, key[0], protocol)
            audit = _audit(env, probabilities, event["decision"])
            if audit != event["audit"]:
                raise ValueError("Matched counterfactual audit mismatch")
            nxt, reward, _, info = env.step(action)
            if nxt != event["next_observation"] or reward != event["reward"] or info != event["info"]:
                raise ValueError("Replay transition mismatch")
            counts[ACTIONS[action]] += 1
            transitions += 1
        values = {
            "steps": env.tick, "survived": env.alive, "alive": env.alive,
            "reward": env.total_reward, "death_reason": env.death_reason, "final_health": env.health,
            "action_counts": counts,
        }
        if any(episode[name] != value for name, value in values.items()):
            raise ValueError("Terminal episode mismatch")
        row = _row(episode)
        if row != indexed[key]:
            raise ValueError("Episode summary arithmetic mismatch")
        verified_rows.append(row)

    summary, paired = paired_summary(verified_rows)
    if summary != receipt["summary"] or paired != receipt["paired"]:
        raise ValueError("Paired summary mismatch")
    return {
        "verified": True, "experiment_id": receipt["experiment_id"],
        "episodes_replayed": len(seen), "transitions_replayed": transitions,
        "counterfactual_branches_replayed": transitions * 6,
        "scope": "same-version simulation, multi-horizon interventions and evidence arithmetic",
    }


def load_and_verify(path):
    path = Path(path)
    if path.stat().st_size > MAX_RECEIPT_BYTES:
        raise ValueError("Receipt exceeds the 128 MiB limit")
    return verify_receipt(json.loads(path.read_text(encoding="utf-8")))
