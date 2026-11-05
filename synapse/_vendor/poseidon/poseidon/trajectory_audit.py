"""Strict historical evidence checks and real multi-step intervention receipts.

Controllers and published experiment code remain separate. Continuation actions
are recorded so an auditor can replay returns without loading neural weights.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

from .atlas import _model_digest
from .experiments import ExperimentSpec, digest, source_identity, write_receipt
from .world import ACTIONS, TidePool, WORLD_VERSION

SCHEMA = "poseidon-trajectory-audit-v1"
MAX_BYTES = 128 * 1024 * 1024
LIMITS = [
    "Synthetic sampled development evidence; no safety or general superiority claim.",
    "Branch return is the undiscounted H-step reward used by the fitted Horizon and Odyssey artifacts.",
    "Continuation actions come from the frozen policy during collection; replay checks execution, not neural authorship.",
    "The legacy experiment's reserve-utility audit is one step and does not validate a multi-step return margin.",
]


def load_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Evidence exceeds the 128 MiB limit")
    def invalid(value):
        raise ValueError("Evidence contains a non-finite number")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def load_artifacts(root):
    return {name: load_json(Path(root) / "outputs" / directory / "atlas.json")
            for name, directory in (("atlas_v2", "atlas"), ("contrast", "contrast"),
                                    ("horizon", "horizon"), ("odyssey", "odyssey"))}


def _module(receipt):
    if receipt.get("schema") == "poseidon-odyssey-experiment-v1":
        from . import odyssey_experiments as module
    elif receipt.get("schema") == "poseidon-horizon-experiment-v1":
        from . import horizon_experiments as module
    else:
        raise ValueError("This audit requires a Horizon or Odyssey experiment receipt")
    return module


def _signed(value, field):
    if digest({key: item for key, item in value.items() if key != field}) != value.get(field):
        raise ValueError("Evidence checksum mismatch")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_parent(receipt, artifacts):
    """Replay legacy evidence with stricter structural and observation checks."""
    try:
        return _verify_parent(receipt, artifacts)
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("Malformed parent evidence") from error


def _verify_parent(receipt, artifacts):
    module = _module(receipt)
    _signed(receipt, "receipt_sha256")
    protocol = receipt["protocol"]
    if (receipt.get("status") != "completed" or receipt.get("promoted") is not False
            or protocol.get("schema") != module.SCHEMA or protocol.get("arms") != list(module.ARMS)
            or protocol.get("world_version") != WORLD_VERSION or digest(protocol) != receipt["experiment_id"]):
        raise ValueError("Invalid parent experiment protocol or status")
    current = source_identity()
    # These pure functions define the historical evidence arithmetic. Controllers
    # are not rerun; their Python source may change without changing recorded actions.
    for name in ("world.py", "contrast.py", module.__name__.split(".")[-1] + ".py"):
        if protocol["source_sha256"].get(name) != current.get(name):
            raise ValueError("Recorded simulator or evidence arithmetic is unavailable")
    spec = module._spec(ExperimentSpec(**protocol["spec"]))
    seeds = set(range(spec.seed, spec.seed + spec.episodes))
    target = "odyssey" if "odyssey" in module.ARMS else "horizon"
    required_artifacts = ("atlas_v2", "contrast", "horizon") + (("odyssey",) if target == "odyssey" else ())
    if any(name + "_sha256" not in protocol for name in required_artifacts):
        raise ValueError("Missing required artifact binding")
    checked = {}
    for name in required_artifacts:
        artifact = artifacts[name]
        _signed(artifact, "sha256")
        if (artifact["sha256"] != protocol[name + "_sha256"]
                or any(artifact[key] != protocol[key] for key in ("checkpoint_sha256", "model_sha256"))
                or artifact["world_version"] != WORLD_VERSION):
            raise ValueError("Parent artifact identity mismatch")
        groups = {key: value for key, value in artifact["partition"].items() if key.endswith("_seeds")}
        required = {"train_seeds", "calibration_seeds"} | ({"selection_seeds"} if name == "contrast" else set())
        used = set()
        if set(groups) != required:
            raise ValueError("Invalid fitting partitions")
        for values in groups.values():
            if (not isinstance(values, list) or not 1 <= len(values) <= 256
                    or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in values)
                    or len(set(values)) != len(values) or used & set(values) or seeds & set(values)):
                raise ValueError("Duplicate or overlapping fitting/audit partitions")
            used.update(values)
        checked[name] = copy.deepcopy(artifact["partition"])
    artifact = artifacts[target]
    if type(protocol.get("horizon")) is not int or protocol["horizon"] != artifact["config"]["horizon"]:
        raise ValueError("Parent horizon differs from its bound artifact")
    metadata = receipt[target]
    fit = metadata["fit_receipt"]
    if (metadata.get("ready") is not True or metadata.get("artifact_sha256") != artifact["sha256"]
            or type(fit.get("training_samples")) is not int or fit["training_samples"] != len(artifact["records"])
            or any(fit.get(key) != artifact[key] for key in ("partition", "calibration"))
            or any(protocol.get(target + "_" + key) != artifact[key] for key in ("partition", "calibration", "config"))):
        raise ValueError("Embedded fitting evidence differs from its bound artifact")
    expected = {(arm, seed) for arm in module.ARMS for seed in seeds}
    episodes, rows = receipt["episodes"], receipt["rows"]
    if not isinstance(episodes, list) or not isinstance(rows, list) or len(episodes) != len(expected) or len(rows) != len(expected):
        raise ValueError("Missing or duplicate paired episodes")
    indexed = {}
    for row in rows:
        key = (row["arm"], row["seed"])
        if key not in expected or key in indexed:
            raise ValueError("Duplicate or unexpected outcome row")
        indexed[key] = row
    seen, verified_rows, transitions, maps = set(), [], 0, 0
    mismatches = set()
    for episode in episodes:
        key = (episode["controller"], episode["seed"])
        if (key not in expected or key in seen or episode["schema"] != "poseidon-episode-v1"
                or episode["world_version"] != WORLD_VERSION or episode["scarcity"] != spec.scarcity
                or episode["max_steps"] != spec.max_steps):
            raise ValueError("Duplicate episode or settings mismatch")
        seen.add(key)
        env = TidePool(key[1], scarcity=spec.scarcity, max_steps=spec.max_steps)
        if env.snapshot() != episode["initial_snapshot"]:
            raise ValueError("Initial snapshot mismatch")
        trace = episode["trajectory"]
        if not isinstance(trace, list) or not 1 <= len(trace) <= spec.max_steps:
            raise ValueError("Invalid trajectory length")
        cognitive = None
        if key[0].startswith("odyssey"):
            from .odyssey import CognitiveMap
            # v0.5.0 reset used config scarcity at tick zero. Reconstruct that
            # recorded map faithfully, and disclose a world-setting mismatch.
            map_scarcity = trace[0]["decision"]["cognitive_map"]["scarcity"]
            if map_scarcity not in (spec.scarcity, artifacts["odyssey"]["config"]["scarcity"]):
                raise ValueError("Invalid recorded cognitive-map scarcity")
            cognitive = CognitiveMap(map_scarcity)
            if map_scarcity != spec.scarcity:
                mismatches.add(key)
        counts = {name: 0 for name in ACTIONS}
        info = None
        for tick, event in enumerate(trace):
            action = event["action"]
            if (env.done or event["observation"] != env.observe() or type(action) is not int or not 0 <= action < 6
                    or event["action_name"] != ACTIONS[action]
                    or any(not _finite(event[name]) or event[name] < 0 for name in ("decision_ms", "audit_ms"))):
                raise ValueError("Invalid transition evidence")
            probabilities = module._validate_decision(event, key[0], protocol)
            if max(range(6), key=lambda a: (probabilities[a], -a)) != event["decision"]["policy_action"]:
                raise ValueError("Incumbent action differs from recorded policy probabilities")
            if cognitive:
                cognitive.update(event["observation"], tick=tick + 1)
                waypoints = cognitive.evaluate_waypoints(event["observation"])
                threshold = artifacts["odyssey"]["config"]["waypoint_threshold"]
                waypoint = waypoints[0] if waypoints and waypoints[0]["score"] > threshold else None
                cognitive.record_action(action)
                if event["decision"]["cognitive_map"] != cognitive.state_summary() or event["decision"].get("best_waypoint") != waypoint:
                    raise ValueError("Cognitive map or waypoint does not follow recorded observations")
                maps += 1
            if event["audit"] != module._audit(env, probabilities, event["decision"]):
                raise ValueError("One-step branch audit mismatch")
            nxt, reward, _, info = env.step(action)
            prediction = module._vector(event["predicted_observation"])
            if (nxt != event["next_observation"] or reward != event["reward"] or info != event["info"]
                    or event["state"] != env.state()
                    or event["prediction_squared_errors"] != [(p - n)**2 for p, n in zip(prediction, nxt)]):
                raise ValueError("Transition or forecast-error arithmetic mismatch")
            counts[ACTIONS[action]] += 1
            transitions += 1
        values = {"steps": env.tick, "survived": env.alive, "alive": env.alive, "death": not env.alive,
                  "death_reason": env.death_reason, "reward": env.total_reward, "final_health": env.health,
                  "action_counts": counts, "final_snapshot": env.snapshot(),
                  "truncated": info["truncated"], "terminal_reason": info["terminal_reason"]}
        if not env.done or any(episode[name] != value for name, value in values.items()):
            raise ValueError("Incomplete or mismatched terminal episode")
        row = module._row(episode)
        if row != indexed[key]:
            raise ValueError("Outcome arithmetic mismatch")
        verified_rows.append(row)
    summary, paired = module.paired_summary(verified_rows)
    if summary != receipt["summary"] or paired != receipt["paired"]:
        raise ValueError("Paired outcome arithmetic mismatch")
    return {"verified": True, "experiment_id": receipt["experiment_id"], "episodes_replayed": len(seen),
            "transitions_replayed": transitions, "cognitive_map_transitions_replayed": maps,
            "partitions": checked, "legacy_branch_audit_horizon": 1,
            "map_scarcity_mismatch_episodes": len(mismatches),
            "scope": "simulator, paired structure, fitting partitions, forecast arithmetic and observation-derived map; not neural authorship"}


def _keys(parent, stride):
    return [(episode["controller"], episode["seed"], tick)
            for episode in parent["episodes"]
            if episode["controller"] == "policy" or episode["controller"].startswith(("horizon", "odyssey"))
            for tick, event in enumerate(episode["trajectory"])
            if tick % stride == 0 or event["action"] != event["decision"]["policy_action"]]


def _branch(snapshot, first, horizon, core=None, recorded=None):
    env = TidePool.from_snapshot(snapshot)
    actions, total = [], 0.0
    for index in range(horizon):
        if env.done:
            break
        action = first if index == 0 else core.act(env.observe()) if core is not None else recorded[index]
        if type(action) is not int or not 0 <= action < 6:
            raise ValueError("Invalid continuation action")
        _, reward, _, info = env.step(action)
        actions.append(action)
        total += reward
    if recorded is not None and actions != recorded:
        raise ValueError("Missing or extra continuation actions")
    return {"action": first, "actions": actions, "return": total, "steps": len(actions),
            "final_health": env.health, "survived": env.alive,
            "terminal_reason": info["terminal_reason"]}


def _return_summary(anchors):
    result = {}
    for arm in sorted({item["arm"] for item in anchors}):
        overrides = [item for item in anchors if item["arm"] == arm and item["action"] != item["policy_action"]]
        selected = [item for item in anchors if item["arm"] == arm]
        result[arm] = {"anchors": len(selected), "overrides_audited": len(overrides),
                       "mean_return_advantage_at_overrides": fmean(item["actual_return_advantage"] for item in overrides) if overrides else None,
                       "adverse_return_overrides": sum(item["actual_return_advantage"] < 0 for item in overrides)}
    return result


def _audit_id(bundle):
    return digest({key: bundle[key] for key in ("schema", "parent_receipt_sha256", "parent_experiment_id",
                                               "source_sha256", "horizon", "stride", "max_anchors", "discount")})


def collect(core, parent, artifacts, *, stride=32, horizon=16, max_anchors=256):
    if type(stride) is not int or not 1 <= stride <= 1024 or type(horizon) is not int or not 2 <= horizon <= 64 or type(max_anchors) is not int or not 1 <= max_anchors <= 256:
        raise ValueError("Invalid bounded trajectory audit settings")
    checked = verify_parent(parent, artifacts)
    sources = source_identity()
    checkpoint = hashlib.sha256(core.path.read_bytes()).hexdigest()
    model = _model_digest(core)
    if checkpoint != parent["protocol"]["checkpoint_sha256"] or model != parent["protocol"]["model_sha256"]:
        raise ValueError("Audit core differs from parent checkpoint")
    keys = _keys(parent, stride)
    if len(keys) > max_anchors:
        raise ValueError("Audit anchor budget exceeded; increase stride or use a smaller parent experiment")
    targets = set(keys)
    anchors = []
    for episode in parent["episodes"]:
        env = TidePool.from_snapshot(episode["initial_snapshot"])
        for tick, event in enumerate(episode["trajectory"]):
            key = (episode["controller"], episode["seed"], tick)
            if key in targets:
                branches = [_branch(env.snapshot(), action, horizon, core=core) for action in range(6)]
                policy = event["decision"]["policy_action"]
                action = event["action"]
                anchors.append({"arm": key[0], "seed": key[1], "tick": tick,
                                "snapshot": env.snapshot(), "policy_action": policy, "action": action,
                                "branches": branches,
                                "actual_return_advantage": branches[action]["return"] - branches[policy]["return"]})
            env.step(event["action"])
    if source_identity() != sources or hashlib.sha256(core.path.read_bytes()).hexdigest() != checkpoint or _model_digest(core) != model:
        raise ValueError("Audit source or core changed during collection")
    result = {"schema": SCHEMA, "parent_receipt_sha256": parent["receipt_sha256"],
              "parent_experiment_id": parent["experiment_id"], "source_sha256": sources,
              "horizon": horizon, "stride": stride, "max_anchors": max_anchors, "discount": 1.0,
              "parent_verification": checked, "anchors": anchors,
              "summary": _return_summary(anchors), "limits": LIMITS, "promoted": False}
    result["experiment_id"] = _audit_id(result)
    result["receipt_sha256"] = digest(result)
    return result


def verify_audit(bundle, parent, artifacts):
    try:
        _signed(bundle, "receipt_sha256")
        checked = verify_parent(parent, artifacts)
        if (bundle["schema"] != SCHEMA or bundle["promoted"] is not False or bundle["discount"] != 1.0
                or bundle["experiment_id"] != _audit_id(bundle)
                or bundle["parent_receipt_sha256"] != parent["receipt_sha256"]
                or bundle["parent_experiment_id"] != parent["experiment_id"] or bundle["parent_verification"] != checked
                or bundle["source_sha256"].get("trajectory_audit.py") != source_identity().get("trajectory_audit.py")
                or type(bundle["horizon"]) is not int or not 2 <= bundle["horizon"] <= 64
                or type(bundle["stride"]) is not int or not 1 <= bundle["stride"] <= 1024
                or type(bundle["max_anchors"]) is not int or not 1 <= bundle["max_anchors"] <= 256):
            raise ValueError("Invalid trajectory audit protocol")
        keys = _keys(parent, bundle["stride"])
        anchors = bundle["anchors"]
        if not isinstance(anchors, list) or len(anchors) != len(keys) or len(anchors) > bundle["max_anchors"]:
            raise ValueError("Missing or duplicate sampled anchors")
        indexed = {}
        for item in anchors:
            key = (item["arm"], item["seed"], item["tick"])
            if key not in keys or key in indexed:
                raise ValueError("Duplicate or unexpected audit anchor")
            indexed[key] = item
        transitions = 0
        for episode in parent["episodes"]:
            env = TidePool.from_snapshot(episode["initial_snapshot"])
            for tick, event in enumerate(episode["trajectory"]):
                key = (episode["controller"], episode["seed"], tick)
                if key in indexed:
                    item = indexed[key]
                    if item["snapshot"] != env.snapshot() or item["action"] != event["action"] or item["policy_action"] != event["decision"]["policy_action"] or len(item["branches"]) != 6:
                        raise ValueError("Audit anchor differs from parent execution")
                    branches = []
                    for action, branch in enumerate(item["branches"]):
                        if (type(branch["steps"]) is not int or not _finite(branch["return"])
                                or not _finite(branch["final_health"]) or type(branch["survived"]) is not bool):
                            raise ValueError("Invalid multi-step branch values")
                        actual = _branch(env.snapshot(), action, bundle["horizon"], recorded=branch["actions"])
                        if actual != branch:
                            raise ValueError("Multi-step branch replay mismatch")
                        branches.append(actual)
                        transitions += actual["steps"]
                    advantage = branches[item["action"]]["return"] - branches[item["policy_action"]]["return"]
                    if advantage != item["actual_return_advantage"]:
                        raise ValueError("Multi-step advantage arithmetic mismatch")
                env.step(event["action"])
        if bundle["summary"] != _return_summary(anchors):
            raise ValueError("Multi-step audit summary mismatch")
        return {"verified": True, "anchors_replayed": len(anchors), "branches_replayed": len(anchors) * 6,
                "branch_transitions_replayed": transitions, "horizon": bundle["horizon"],
                "scope": "actual H-step reward execution under recorded continuation actions; not neural authorship"}
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("Malformed trajectory audit") from error


def save_audit(bundle, directory):
    return write_receipt(bundle, directory)
