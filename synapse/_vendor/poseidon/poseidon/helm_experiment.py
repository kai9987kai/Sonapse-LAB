"""Paired Helm evaluation with recorded continuation actions and weightless replay.

Simulator snapshots belong to this evaluator, never to Helm inference. Checksums
and replay establish execution consistency, not neural authorship or safety.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from statistics import fmean
import time

from .atlas import _model_digest
from .experiments import digest, source_identity
from .world import ACTIONS, TidePool, WORLD_VERSION

SCHEMA = "poseidon-helm-experiment-v1"
ARMS = ("policy", "selected", "observation", "no_innovation", "yoked", "ungated")
HORIZONS = (4, 16)
RISK_THRESHOLDS = (-1.0, 0.0, 0.03, 0.1, 0.3, 1.0)
MAX_TRANSITIONS = 12288
MAX_ANCHORS = 2048
MAX_BRANCH_TRANSITIONS = 262144
LIMITS = [
    "Synthetic sampled TidePool development evidence; no safety or general superiority guarantee.",
    "H4/H16 branch returns are undiscounted rewards under recorded frozen-core continuation actions.",
    "Replay checks action execution, arithmetic and artifact bindings without weights, not neural authorship.",
    "Forecast errors and empirical margin/risk diagnostics do not guarantee future coverage.",
    "Risk thresholds are predeclared diagnostics; evaluation does not select deployment settings.",
    "Regular and override anchor groups overlap when an override occurs on the regular schedule.",
    "Each controller visits its own states; branch advantages do not establish full adaptive-policy causal gains.",
    "Controller and auditing compute are reported separately and are not matched wall-clock budgets.",
    "Leave-one-seed-out ranges are influence checks, not confidence intervals or independent replication.",
    "No checkpoint, default controller or deployment setting is changed or promoted by an experiment.",
]


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _number(value, name, low=-100000.0, high=100000.0):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Invalid finite {name}")
    return float(value)


def _vector(value, name, size=6, *, unit=False):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"Invalid {name} vector")
    for item in value:
        _number(item, name, 0 if unit else -100000, 1 if unit else 100000)
    return value


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"Invalid {name} identity")
    return value


def _equal(actual, expected, name):
    # Canonical JSON also distinguishes bools from ints and rejects NaN/Infinity.
    if digest(actual) != digest(expected):
        raise ValueError(f"{name} replay mismatch")


def _close(actual, expected, name):
    _number(actual, name)
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{name} arithmetic mismatch")


def _settings(seeds, max_steps, scarcity, anchor_interval):
    if not isinstance(seeds, (list, tuple)) or not 1 <= len(seeds) <= 32:
        raise ValueError("Evaluation requires 1 to 32 distinct seeds")
    seeds = [_integer(seed, "seed", 0, 2**63 - 1) for seed in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate evaluation seeds")
    _integer(max_steps, "max_steps", 1, 256)
    _integer(anchor_interval, "anchor_interval", 1, 256)
    scarcity = _number(scarcity, "scarcity", 0.5, 4.0)
    if len(seeds) * max_steps * len(ARMS) > MAX_TRANSITIONS:
        raise ValueError("Helm experiment exceeds its trajectory budget")
    return seeds, max_steps, scarcity, anchor_interval


def _partition(partition, evaluation_seeds):
    if not isinstance(partition, dict):
        raise ValueError("Missing fitting partitions")
    used = set()
    for name in ("train_seeds", "selection_seeds", "calibration_seeds"):
        values = partition.get(name)
        if not isinstance(values, list) or not 1 <= len(values) <= 256:
            raise ValueError("Invalid fitting partition")
        group = {_integer(seed, name, 0, 2**63 - 1) for seed in values}
        if len(group) != len(values) or group & used or group & set(evaluation_seeds):
            raise ValueError("Duplicate or overlapping fitting/evaluation partitions")
        used.update(group)


def _notify(progress, phase, **fields):
    if progress is not None:
        progress({"phase": phase, **fields})


def _admissibility(observation, minimum):
    result = []
    for action, label in enumerate(ACTIONS):
        reason = None
        if action == 1 and observation[6] < minimum:
            reason = "depleted_food"
        elif action == 2 and observation[7] < minimum:
            reason = "depleted_water"
        elif action in (4, 5) and (observation[3] < 0.12 or observation[1] < 0.08 or observation[2] < 0.08):
            reason = "insufficient_travel_reserves"
        result.append({"action": action, "label": label, "admissible": reason is None, "reason": reason})
    return result


def _decision(decision, arm, tick, observation, artifact, max_steps):
    if not isinstance(decision, dict):
        raise ValueError("Missing controller decision")
    action = _integer(decision.get("action"), "action", 0, 5)
    policy = _integer(decision.get("policy_action"), "policy_action", 0, 5)
    proposed = _integer(decision.get("proposed_action"), "proposed_action", 0, 5)
    if (type(decision.get("overridden")) is not bool or decision["overridden"] != (action != policy)
            or decision.get("mode") != arm or decision.get("step") != tick
            or type(decision.get("step")) is not int):
        raise ValueError("Decision identity or override mismatch")
    if arm == "policy":
        _equal(decision, {"action": policy, "policy_action": policy, "proposed_action": policy,
                          "overridden": False, "mode": "policy", "step": tick}, "Policy decision")
        return
    config = artifact["config"]
    if not isinstance(decision.get("family"), str) or not decision["family"]:
        raise ValueError("Invalid selected feature family")
    expected_family = artifact.get("calibration", {}).get(arm, {}).get("family")
    if expected_family is not None and decision["family"] != expected_family:
        raise ValueError("Decision differs from its calibrated family")
    for field in ("forecasts", "advantages", "radii", "horizon_margins"):
        if not isinstance(decision.get(field), dict) or set(decision[field]) != {"4", "16"}:
            raise ValueError("Missing complete two-horizon decision")
    margins = []
    for horizon in HORIZONS:
        key = str(horizon)
        forecasts = _vector(decision["forecasts"][key], "return forecasts")
        advantages = _vector(decision["advantages"][key], "return advantages")
        for challenger in range(6):
            _close(advantages[challenger], forecasts[challenger] - forecasts[policy], "Forecast advantage")
        radius = _number(decision["radii"][key], "error radius", 0)
        margin = advantages[proposed] - radius - config["override_margin"]
        _close(decision["horizon_margins"][key], margin, "Horizon margin")
        margins.append(margin)
    _close(decision["margin"], min(margins), "Minimum margin")
    distance = _number(decision.get("support_distance"), "support distance", 0)
    if type(decision.get("supported")) is not bool or decision["supported"] != (distance <= config["support_radius"]):
        raise ValueError("Support gate mismatch")
    candidates = _admissibility(observation, config["min_harvest_resource"])
    _equal(decision.get("candidates"), candidates, "Action admissibility")
    history = decision.get("history")
    if not isinstance(history, dict):
        raise ValueError("Missing recorded history diagnostics")
    for key in ("state_norm", "innovation_norm"):
        _number(history.get(key), key, 0)
    previous = history.get("previous_executed_action")
    if tick == 0:
        if previous is not None:
            raise ValueError("Initial history did not reset")
    elif type(previous) is not int or previous != round(observation[14] * 5):
        raise ValueError("History differs from the preceding executed action")
    if history.get("yoked_index") is not None:
        _integer(history["yoked_index"], "yoked index", 0, 2**31 - 1)
    fallback = decision.get("fallback_reason")
    if fallback is not None and (not isinstance(fallback, str) or len(fallback) > 160):
        raise ValueError("Invalid fallback diagnostic")
    if action != policy:
        if (action != proposed or not candidates[action]["admissible"] or not decision["supported"]
                or max_steps - tick < max(HORIZONS) or fallback is not None):
            raise ValueError("Override bypassed physical, support or horizon guard")
        if arm != "ungated" and not all(margin > 0 for margin in margins):
            raise ValueError("Override bypassed the calibrated margin")


def _branch(snapshot, first, horizon, *, core=None, recorded=None, progress=None, context=None):
    env = TidePool.from_snapshot(snapshot)
    actions, rewards, observations, terminals = [], [], [], []
    if recorded is not None and (not isinstance(recorded, list) or not 1 <= len(recorded) <= horizon):
        raise ValueError("Invalid continuation action sequence")
    for index in range(horizon):
        if env.done:
            break
        _notify(progress, "branch" if core is not None else "verify_branch", horizon=horizon,
                action=first, branch_step=index, **(context or {}))
        if recorded is not None:
            if index >= len(recorded):
                raise ValueError("Missing continuation action")
            action = recorded[index]
        else:
            action = first if index == 0 else core.act(env.observe())
        _integer(action, "continuation action", 0, 5)
        if index == 0 and action != first:
            raise ValueError("Branch does not start with its declared action")
        nxt, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        observations.append(nxt)
        terminals.append(done)
    if recorded is not None and actions != recorded:
        raise ValueError("Extra continuation actions after branch termination")
    return {"action": first, "horizon": horizon, "actions": actions, "rewards": rewards,
            "next_observations": observations, "terminated": terminals, "return": sum(rewards),
            "steps": len(actions), "final_snapshot": env.snapshot(), "survived": env.alive,
            "final_health": env.health, "terminal_reason": "death" if not env.alive else "horizon" if env.done else None}


def _anchor_fields(anchor, branches, decision):
    return {
        "actual_advantages": {str(h): branches[str(h)][anchor["action"]]["return"] - branches[str(h)][anchor["policy_action"]]["return"] for h in HORIZONS},
        "actual_proposed_advantages": {str(h): branches[str(h)][anchor["proposed_action"]]["return"] - branches[str(h)][anchor["policy_action"]]["return"] for h in HORIZONS},
        "overestimation": {str(h): decision["advantages"][str(h)][anchor["proposed_action"]] -
                           (branches[str(h)][anchor["proposed_action"]]["return"] - branches[str(h)][anchor["policy_action"]]["return"])
                           for h in HORIZONS} if anchor["arm"] != "policy" else None,
    }


def _episode_outcome(env, arm, initial, trace):
    return {"schema": "poseidon-helm-episode-v1", "world_version": WORLD_VERSION,
            "controller": arm, "seed": env.seed, "scarcity": env.scarcity, "max_steps": env.max_steps,
            "steps": env.tick, "reward": env.total_reward, "survived": env.alive,
            "death_reason": env.death_reason, "final_health": env.health, "visited": len(env.visited),
            "truncated": env.alive and env.tick >= env.max_steps,
            "action_counts": {name: sum(event["action"] == action for event in trace) for action, name in enumerate(ACTIONS)},
            "initial_snapshot": initial, "final_snapshot": env.snapshot(), "trajectory": trace}


def _row(episode, anchors):
    trace = episode["trajectory"]
    group = [item for item in anchors if (item["arm"], item["seed"]) == (episode["controller"], episode["seed"])]
    overrides = [item for item in group if item["override"]]
    return {"arm": episode["controller"], "seed": episode["seed"], "steps": episode["steps"],
            "reward": episode["reward"], "survived": episode["survived"],
            "death_reason": episode["death_reason"], "final_health": episode["final_health"],
            "decision_ms": sum(event["decision_ms"] for event in trace),
            "audit_ms": sum(event["audit_ms"] for event in trace),
            "fallback_steps": sum(event["decision"].get("fallback_reason") is not None for event in trace),
            "overrides": sum(event["decision"]["overridden"] for event in trace),
            "anchors": len(group), "regular_anchors": sum(item["regular"] for item in group),
            "overrides_audited": len(overrides),
            "adverse_overrides": {str(h): sum(item["actual_advantages"][str(h)] < 0 for item in overrides) for h in HORIZONS},
            "mean_actual_advantage_at_overrides": {str(h): fmean(item["actual_advantages"][str(h)] for item in overrides) if overrides else None for h in HORIZONS}}


def _influence(values):
    if len(values) < 2:
        return {"method": "leave-one-seed-out-mean-range-v1", "count": 0, "minimum": None,
                "maximum": None, "direction_reversals": None, "means": []}
    mean = fmean(values)
    removed = [fmean(values[:index] + values[index + 1:]) for index in range(len(values))]
    return {"method": "leave-one-seed-out-mean-range-v1", "count": len(removed),
            "minimum": min(removed), "maximum": max(removed),
            "direction_reversals": sum(value * mean < 0 for value in removed), "means": removed}


def _summaries(rows):
    summary, paired = {}, {}
    baseline = {row["seed"]: row for row in rows if row["arm"] == "policy"}
    for arm in ARMS:
        group = sorted((row for row in rows if row["arm"] == arm), key=lambda row: row["seed"])
        steps = sum(row["steps"] for row in group)
        overrides = sum(row["overrides"] for row in group)
        summary[arm] = {"episodes": len(group), "mean_reward": fmean(row["reward"] for row in group),
                        "survival_rate": fmean(row["survived"] for row in group),
                        "mean_steps": fmean(row["steps"] for row in group), "overrides": overrides,
                        "override_rate": overrides / steps, "mean_decision_ms": sum(row["decision_ms"] for row in group) / steps,
                        "mean_audit_ms": sum(row["audit_ms"] for row in group) / steps,
                        "anchors": sum(row["anchors"] for row in group),
                        "regular_anchors": sum(row["regular_anchors"] for row in group),
                        "adverse_overrides": {str(h): sum(row["adverse_overrides"][str(h)] for row in group) for h in HORIZONS},
                        "conditional_adverse_override_rate": {str(h): sum(row["adverse_overrides"][str(h)] for row in group) / overrides if overrides else None for h in HORIZONS}}
        if arm == "policy":
            continue
        values = [row["reward"] - baseline[row["seed"]]["reward"] for row in group]
        paired[arm] = {"baseline": "policy", "mean_reward_delta": fmean(values),
                       "wins": sum(value > 1e-10 for value in values), "ties": sum(abs(value) <= 1e-10 for value in values),
                       "losses": sum(value < -1e-10 for value in values),
                       "per_seed": [{"seed": row["seed"], "reward_delta": value,
                                     "survival_delta": int(row["survived"]) - int(baseline[row["seed"]]["survived"])} for row, value in zip(group, values)],
                       "leave_one_seed_out": _influence(values)}
    return summary, paired


def _risk(receipt):
    protocol, modes = receipt["protocol"], {}
    events = {(episode["controller"], episode["seed"], tick): event for episode in receipt["episodes"]
              for tick, event in enumerate(episode["trajectory"])}
    for arm in ARMS[1:]:
        modes[arm] = {}
        for scope in ("regular", "override"):
            anchors = [anchor for anchor in receipt["anchors"] if anchor["arm"] == arm and anchor[scope]]
            eligible = []
            for anchor in anchors:
                decision = events[(arm, anchor["seed"], anchor["tick"])]["decision"]
                if (decision["proposed_action"] != decision["policy_action"] and decision["supported"]
                        and decision["candidates"][decision["proposed_action"]]["admissible"]
                        and protocol["max_steps"] - anchor["tick"] >= max(HORIZONS)):
                    eligible.append((anchor, decision))
            rows = []
            for threshold in RISK_THRESHOLDS:
                selected = [(anchor, decision) for anchor, decision in eligible if decision["margin"] >= threshold]
                for horizon in HORIZONS:
                    key = str(horizon)
                    values = [anchor["actual_proposed_advantages"][key] for anchor, _ in selected]
                    adverse = sum(value < 0 for value in values)
                    exceed = sum(anchor["overestimation"][key] > decision["radii"][key] for anchor, decision in selected)
                    rows.append({"threshold": threshold, "horizon": horizon, "anchors": len(anchors),
                                 "eligible": len(eligible), "exposure": len(selected), "no_exposure": not selected,
                                 "coverage": len(selected) / len(anchors) if anchors else None,
                                 "adverse": adverse, "adverse_rate": adverse / len(selected) if selected else None,
                                 "mean_actual_advantage": fmean(values) if values else None,
                                 "radius_exceedances": exceed,
                                 "radius_exceedance_rate": exceed / len(selected) if selected else None})
            modes[arm][scope] = rows
    return {"method": "predeclared-descriptive-margin-thresholds-v1", "thresholds": list(RISK_THRESHOLDS),
            "horizons": list(HORIZONS), "selection_from_evaluation": False,
            "scope_note": "Anchor-conditional prospective proposal diagnostics; regular and override scopes may overlap.",
            "modes": modes}


def helm_risk_coverage(receipt):
    """Recompute fixed descriptive thresholds; empty exposure rates remain null."""
    try:
        return _risk(receipt)
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("Malformed Helm risk evidence") from error


def run_helm_experiment(core, critic, *, seeds, max_steps=64, scarcity=1.0, anchor_interval=16, progress=None):
    from .helm import validate_helm_artifact
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable")
    seeds, max_steps, scarcity, anchor_interval = _settings(seeds, max_steps, scarcity, anchor_interval)
    artifact = validate_helm_artifact(critic.artifact, core=core)
    if artifact["config"]["horizons"] != list(HORIZONS):
        raise ValueError("Helm experiments require fitted H4/H16 readouts")
    _partition(artifact["partition"], seeds)
    sources = source_identity()
    checkpoint = hashlib.sha256(Path(core.path).read_bytes()).hexdigest()
    model = _model_digest(core)
    protocol = {"schema": SCHEMA, "world_version": WORLD_VERSION, "arms": list(ARMS), "seeds": seeds,
                "max_steps": max_steps, "scarcity": scarcity, "anchor_interval": anchor_interval,
                "horizons": list(HORIZONS), "discount": 1.0, "risk_thresholds": list(RISK_THRESHOLDS),
                "max_transitions": MAX_TRANSITIONS, "max_anchors": MAX_ANCHORS,
                "max_branch_transitions": MAX_BRANCH_TRANSITIONS, "source_sha256": sources,
                "world_sha256": sources["world.py"], "checkpoint_sha256": checkpoint,
                "model_sha256": model, "critic_sha256": artifact["sha256"],
                "partition": copy.deepcopy(artifact["partition"])}
    if artifact["checkpoint_sha256"] != checkpoint or artifact["model_sha256"] != model:
        raise ValueError("Core differs from the fitted critic")
    episodes, anchors, branch_steps = [], [], 0
    _notify(progress, "experiment", completed_episodes=0, total_episodes=len(seeds) * len(ARMS))
    for seed in seeds:
        for arm in ARMS:
            env = TidePool(seed, scarcity=scarcity, max_steps=max_steps)
            initial, trace = env.snapshot(), []
            if arm != "policy":
                critic.reset(scarcity=scarcity, max_steps=max_steps)
            while not env.done:
                tick, observation = env.tick, env.observe()
                context = {"arm": arm, "seed": seed, "tick": tick,
                           "completed_episodes": len(episodes), "total_episodes": len(seeds) * len(ARMS)}
                _notify(progress, "decision", **context)
                started = time.perf_counter()
                if arm == "policy":
                    action = core.act(observation)
                    decision = {"action": action, "policy_action": action, "proposed_action": action,
                                "overridden": False, "mode": arm, "step": tick}
                else:
                    decision = copy.deepcopy(critic.plan(observation, mode=arm))
                decision_ms = (time.perf_counter() - started) * 1000
                _decision(decision, arm, tick, observation, artifact, max_steps)
                if decision["policy_action"] != core.act(observation):
                    raise ValueError("Helm decision differs from the frozen core policy")
                audit_started = time.perf_counter()
                if tick % anchor_interval == 0 or decision["overridden"]:
                    if len(anchors) >= MAX_ANCHORS:
                        raise ValueError("Helm experiment exceeds its complete anchor budget")
                    snapshot = env.snapshot()
                    branches = {}
                    for horizon in HORIZONS:
                        branches[str(horizon)] = []
                        for first in range(6):
                            branch = _branch(snapshot, first, horizon, core=core, progress=progress, context=context)
                            branch_steps += branch["steps"]
                            if branch_steps > MAX_BRANCH_TRANSITIONS:
                                raise ValueError("Helm experiment exceeds its branch transition budget")
                            branches[str(horizon)].append(branch)
                    anchor = {"arm": arm, "seed": seed, "tick": tick, "regular": tick % anchor_interval == 0,
                              "override": decision["overridden"], "snapshot": snapshot,
                              "action": decision["action"], "policy_action": decision["policy_action"],
                              "proposed_action": decision["proposed_action"], "branches": branches}
                    anchor.update(_anchor_fields(anchor, branches, decision))
                    anchors.append(anchor)
                audit_ms = (time.perf_counter() - audit_started) * 1000
                nxt, reward, done, info = env.step(decision["action"])
                trace.append({"tick": tick, "observation": observation, "action": decision["action"],
                              "action_name": ACTIONS[decision["action"]], "decision": decision,
                              "next_observation": nxt, "reward": reward, "done": done, "info": info,
                              "decision_ms": decision_ms, "audit_ms": audit_ms})
            episodes.append(_episode_outcome(env, arm, initial, trace))
            _notify(progress, "episode_completed", completed_episodes=len(episodes), total_episodes=len(seeds) * len(ARMS), seed=seed, arm=arm)
    if (source_identity() != sources or hashlib.sha256(Path(core.path).read_bytes()).hexdigest() != checkpoint
            or _model_digest(core) != model or critic.artifact["sha256"] != artifact["sha256"]):
        raise ValueError("Source, core or critic changed during the experiment")
    rows = [_row(episode, anchors) for episode in episodes]
    summary, paired = _summaries(rows)
    result = {"schema": SCHEMA, "status": "completed", "promoted": False, "protocol": protocol,
              "experiment_id": digest(protocol), "critic": artifact, "episodes": episodes,
              "anchors": anchors, "rows": rows, "summary": summary, "paired": paired, "limits": list(LIMITS)}
    result["risk_coverage"] = _risk(result)
    _notify(progress, "experiment_completed", completed_episodes=len(episodes), total_episodes=len(episodes))
    result["receipt_sha256"] = digest(result)
    return result


def verify_helm_receipt(receipt, progress=None):
    """Strict replay using recorded actions only; cancellation propagates."""
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable")
    try:
        return _verify(receipt, progress)
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("Malformed Helm evidence") from error


def _verify(receipt, progress):
    from .helm import validate_helm_artifact
    if not isinstance(receipt, dict):
        raise ValueError("Helm receipt must be an object")
    _notify(progress, "verify", completed_episodes=0)
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if digest(unsigned) != receipt.get("receipt_sha256"):
        raise ValueError("Helm receipt checksum mismatch")
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "completed" or receipt.get("promoted") is not False:
        raise ValueError("Incomplete or unsupported Helm receipt")
    protocol = receipt["protocol"]
    seeds, max_steps, scarcity, interval = _settings(protocol["seeds"], protocol["max_steps"], protocol["scarcity"], protocol["anchor_interval"])
    for key, expected in (("schema", SCHEMA), ("world_version", WORLD_VERSION), ("arms", list(ARMS)),
                          ("horizons", list(HORIZONS)), ("discount", 1.0), ("risk_thresholds", list(RISK_THRESHOLDS)),
                          ("max_transitions", MAX_TRANSITIONS), ("max_anchors", MAX_ANCHORS), ("max_branch_transitions", MAX_BRANCH_TRANSITIONS)):
        _equal(protocol.get(key), expected, "Protocol " + key)
    if digest(protocol) != receipt.get("experiment_id"):
        raise ValueError("Protocol experiment identity mismatch")
    sources = protocol["source_sha256"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Missing source identity")
    for name, value in sources.items():
        p = Path(name)
        if not isinstance(name, str) or p.is_absolute() or ".." in p.parts or not name.endswith(".py"):
            raise ValueError("Invalid source name")
        _hash(value, "source")
    historical_helm_hash = "c84150c58d333506c3a137a56f9e0c5ef3a57ee0d10684d1e1d86340dd616622"
    for name in ("world.py", "helm_experiment.py"):
        cur_h = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        valid = {cur_h, historical_helm_hash} if name == "helm_experiment.py" else {cur_h}
        if sources.get(name) not in valid:
            raise ValueError("Recorded simulator or Helm evidence arithmetic is unavailable")
    if protocol["world_sha256"] != sources["world.py"]:
        raise ValueError("Simulator source binding mismatch")
    artifact = validate_helm_artifact(receipt["critic"], core=None)
    for key in ("checkpoint_sha256", "model_sha256"):
        _hash(protocol[key], key)
        if protocol[key] != artifact[key]:
            raise ValueError("Critic core binding mismatch")
    if (protocol["critic_sha256"] != artifact["sha256"] or artifact["world_version"] != WORLD_VERSION
            or artifact["world_sha256"] != protocol["world_sha256"]
            or artifact["config"]["horizons"] != list(HORIZONS)):
        raise ValueError("Critic artifact binding mismatch")
    _equal(protocol["partition"], artifact["partition"], "Fitting partition")
    _partition(artifact["partition"], seeds)
    expected_pairs = {(arm, seed) for arm in ARMS for seed in seeds}
    episodes, rows, anchors = receipt["episodes"], receipt["rows"], receipt["anchors"]
    if (not isinstance(episodes, list) or not isinstance(rows, list) or len(episodes) != len(expected_pairs)
            or len(rows) != len(expected_pairs) or not isinstance(anchors, list) or len(anchors) > MAX_ANCHORS):
        raise ValueError("Missing or duplicate paired episodes or anchors")
    row_index, anchor_index, seen = {}, {}, set()
    for row in rows:
        key = (row["arm"], row["seed"])
        if type(row["seed"]) is not int or key not in expected_pairs or key in row_index:
            raise ValueError("Duplicate or unexpected paired outcome row")
        row_index[key] = row
    for anchor in anchors:
        key = (anchor["arm"], anchor["seed"], anchor["tick"])
        _integer(anchor["seed"], "anchor seed", 0, 2**63 - 1)
        _integer(anchor["tick"], "anchor tick", 0, max_steps - 1)
        if key[:2] not in expected_pairs or key in anchor_index:
            raise ValueError("Duplicate or unexpected branch anchor")
        anchor_index[key] = anchor
    transitions, branch_transitions, checked_anchors = 0, 0, set()
    verified_rows = []
    for episode in episodes:
        key = (episode["controller"], episode["seed"])
        if type(episode["seed"]) is not int or key not in expected_pairs or key in seen:
            raise ValueError("Missing, duplicate or unexpected paired episode")
        seen.add(key)
        env = TidePool(key[1], scarcity=scarcity, max_steps=max_steps)
        initial = env.snapshot()
        trace = episode["trajectory"]
        if not isinstance(trace, list) or not 1 <= len(trace) <= max_steps:
            raise ValueError("Invalid trajectory length")
        for tick, event in enumerate(trace):
            context = {"arm": key[0], "seed": key[1], "tick": tick, "completed_episodes": len(verified_rows), "total_episodes": len(expected_pairs)}
            _notify(progress, "verify_transition", **context)
            observation = env.observe()
            if env.done or type(event.get("tick")) is not int or event["tick"] != tick:
                raise ValueError("Trajectory extends beyond termination or has invalid ticks")
            _vector(event["observation"], "observation", size=16, unit=True)
            _equal(event["observation"], observation, "Observation")
            decision = event["decision"]
            _decision(decision, key[0], tick, observation, artifact, max_steps)
            action = _integer(event["action"], "executed action", 0, 5)
            if action != decision["action"] or event["action_name"] != ACTIONS[action]:
                raise ValueError("Executed action differs from its decision")
            for name in ("decision_ms", "audit_ms"):
                _number(event[name], name, 0, 1e9)
            anchor_key = (*key, tick)
            needs_anchor = tick % interval == 0 or decision["overridden"]
            if needs_anchor != (anchor_key in anchor_index):
                raise ValueError("Missing or unexpected regular/override branch anchor")
            if needs_anchor:
                anchor = anchor_index[anchor_key]
                checked_anchors.add(anchor_key)
                _equal(anchor["snapshot"], env.snapshot(), "Anchor snapshot")
                for field, expected in (("regular", tick % interval == 0), ("override", decision["overridden"]),
                                        ("action", action), ("policy_action", decision["policy_action"]),
                                        ("proposed_action", decision["proposed_action"])):
                    _equal(anchor[field], expected, "Anchor " + field)
                branches = anchor["branches"]
                if not isinstance(branches, dict) or set(branches) != {"4", "16"}:
                    raise ValueError("Missing complete H4/H16 branches")
                for horizon in HORIZONS:
                    group = branches[str(horizon)]
                    if not isinstance(group, list) or len(group) != 6:
                        raise ValueError("Missing or duplicate six-action branches")
                    for first, branch in enumerate(group):
                        actual = _branch(env.snapshot(), first, horizon, recorded=branch["actions"], progress=progress, context=context)
                        _equal(branch, actual, "Multi-step branch")
                        branch_transitions += actual["steps"]
                        if branch_transitions > MAX_BRANCH_TRANSITIONS:
                            raise ValueError("Replay exceeds its branch transition budget")
                        shorter = branches["4"][first]
                        if horizon == 16 and shorter["actions"] != actual["actions"][:min(4, actual["steps"])]:
                            raise ValueError("H4/H16 frozen continuations disagree")
                for field, expected in _anchor_fields(anchor, branches, decision).items():
                    _equal(anchor[field], expected, "Anchor " + field)
            nxt, reward, done, info = env.step(action)
            _equal(event["next_observation"], nxt, "Next observation")
            _equal(event["reward"], reward, "Reward")
            _equal(event["done"], done, "Termination")
            _equal(event["info"], info, "Transition information")
            transitions += 1
        if not env.done:
            raise ValueError("Incomplete terminal trajectory")
        _equal(episode, _episode_outcome(env, key[0], initial, trace), "Episode outcome")
        computed = _row(episode, anchors)
        _equal(row_index[key], computed, "Outcome row")
        verified_rows.append(computed)
    if seen != expected_pairs or checked_anchors != set(anchor_index):
        raise ValueError("Incomplete paired episodes or unreplayed anchors")
    summary, paired = _summaries(verified_rows)
    _equal(receipt["summary"], summary, "Controller summary")
    _equal(receipt["paired"], paired, "Paired outcome")
    _equal(receipt["risk_coverage"], _risk(receipt), "Risk coverage")
    _equal(receipt["limits"], LIMITS, "Evidence limits")
    _notify(progress, "verified", completed_episodes=len(episodes), total_episodes=len(episodes))
    return {"verified": True, "episodes_replayed": len(episodes), "transitions_replayed": transitions,
            "anchors_replayed": len(anchors), "branches_replayed": len(anchors) * 12,
            "branch_transitions_replayed": branch_transitions, "horizons": list(HORIZONS),
            "scope": "Recorded frozen-core action execution, outcomes and bound evidence arithmetic; not neural authorship."}
