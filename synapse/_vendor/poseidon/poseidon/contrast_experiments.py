"""Nine-controller experiments with matched action audits and portable replay.

The evaluator can fork simulator snapshots; controllers receive observations only.
An action's one-step reserve utility is distinct from episode reward or survival.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean
import time

from .atlas import _base, _model_digest
from .experiments import ExperimentSpec, digest, source_identity, write_receipt
from .world import ACTIONS, TidePool, WORLD_VERSION, event_uniform, teacher_action

SCHEMA = "poseidon-contrast-experiment-v1"
ARMS = ("policy", "neural_mpc", "atlas_v2", "contrast", "contrast_absolute",
        "contrast_no_memory", "contrast_unfiltered", "heuristic", "random")
MODES = ("base", "unfiltered", "memory")
MAX_TRANSITIONS = 12000
MAX_RECEIPT_BYTES = 128 * 1024 * 1024
LIMITS = [
    "Synthetic TidePool development evidence; no real-world or biological inference.",
    "Episode-max advantage error radii are descriptive, not safety or conformal guarantees.",
    "A true one-step reserve-utility advantage does not establish improved episode return or survival.",
    "On-policy prediction errors use different visited states; the policy-arm prediction probe compares modes at identical six-action anchors.",
    "Controller timings exclude counterfactual auditing and probes; controllers have unequal compute budgets.",
    "Replay verifies simulator execution and evidence arithmetic without model weights, not model authorship.",
    "No weights, incumbent pointer or default planner are changed or promoted by an experiment.",
]


def _spec(spec):
    spec = spec or ExperimentSpec(seed=104000001)
    if not isinstance(spec, ExperimentSpec):
        raise ValueError("spec must be ExperimentSpec")
    if spec.episodes * spec.max_steps * len(ARMS) > MAX_TRANSITIONS:
        raise ValueError("Contrast experiment exceeds the 12000-transition budget")
    return spec


def _vector(value, size=16, *, unit=True):
    if not isinstance(value, list) or len(value) != size or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        or (unit and not 0 <= x <= 1) for x in value
    ):
        raise ValueError("Invalid recorded vector")
    return value


def _utility(observation, probability, config):
    from .contrast import reserve_utility
    return reserve_utility(observation, probability, policy_weight=config["policy_weight"],
                           margin_penalty=config["margin_penalty"])


def _audit(env, probabilities, config, decision):
    snapshot = env.snapshot()
    successors = []
    for action in range(6):
        branch = TidePool.from_snapshot(snapshot)
        successors.append(branch.step(action)[0])
    utilities = [_utility(nxt, probabilities[a], config) for a, nxt in enumerate(successors)]
    policy = decision["policy_action"]
    action = decision["action"]
    best = max(range(6), key=lambda a: (utilities[a], -a))
    proposed = decision.get("proposed_action", action)
    return {"next_observations": successors, "utilities": utilities, "best_action": best,
            "actual_advantage": utilities[action] - utilities[policy],
            "actual_proposed_advantage": utilities[proposed] - utilities[policy],
            "one_step_regret": utilities[best] - utilities[action],
            "override": action != policy,
            "margin_exceeded": bool("advantage_error_radius" in decision and
                decision["point_advantage"] - (utilities[proposed] - utilities[policy]) > decision["advantage_error_radius"])}


def _probe(contrast, observation):
    return {mode: [row["predicted_observation"] for row in contrast.plan(
        observation, use_memory=mode != "base", use_selection=mode != "unfiltered")["candidates"]]
        for mode in MODES}


def _prediction(decision, arm, probabilities, core, observation):
    if arm == "atlas_v2" or arm.startswith("contrast"):
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
    probes = [event for event in trace if "prediction_probe" in event]
    probe_summary = None
    if probes:
        probe_summary = {mode: {
            "anchors": len(probes), "branches": len(probes) * 6,
            "per_channel_mse": [fmean(
                (event["prediction_probe"][mode][a][i] - event["audit"]["next_observations"][a][i])**2
                for event in probes for a in range(6)) for i in range(16)]
        } for mode in MODES}
        for values in probe_summary.values():
            values["mse"] = fmean(values["per_channel_mse"])
    return {"arm": episode["controller"], "seed": episode["seed"], "steps": count,
            "survived": episode["survived"], "reward": episode["reward"],
            "death_reason": episode["death_reason"], "final_health": episode["final_health"],
            "decision_ms": sum(event["decision_ms"] for event in trace),
            "audit_ms": sum(event["audit_ms"] for event in trace),
            "fallback_steps": sum(bool(event["decision"].get("fallback_reason")) for event in trace),
            "overrides": len(overrides), "per_channel_mse": channel, "prediction_mse": fmean(channel),
            "counterfactual_audit": {"anchors": count, "overrides_audited": len(overrides),
                "mean_actual_advantage_at_overrides": fmean(event["audit"]["actual_advantage"] for event in overrides) if overrides else None,
                "adverse_overrides": sum(event["audit"]["actual_advantage"] < 0 for event in overrides),
                "mean_one_step_regret": fmean(event["audit"]["one_step_regret"] for event in trace),
                "ranking_accuracy": fmean(event["action"] == event["audit"]["best_action"] for event in trace),
                "paired_margin_exceedances": sum(event["audit"]["margin_exceeded"] for event in trace)},
            "prediction_probe": probe_summary}


def paired_summary(rows):
    summaries, paired = {}, {}
    baseline = {row["seed"]: row for row in rows if row["arm"] == "policy"}
    for arm in ARMS:
        group = sorted((row for row in rows if row["arm"] == arm), key=lambda row: row["seed"])
        steps = sum(row["steps"] for row in group)
        overrides = sum(row["overrides"] for row in group)
        audit = {
            "anchors": steps, "overrides_audited": overrides,
            "mean_actual_advantage_at_overrides": sum((row["counterfactual_audit"]["mean_actual_advantage_at_overrides"] or 0) * row["overrides"] for row in group) / overrides if overrides else None,
            "adverse_overrides": sum(row["counterfactual_audit"]["adverse_overrides"] for row in group),
            "mean_one_step_regret": sum(row["counterfactual_audit"]["mean_one_step_regret"] * row["steps"] for row in group) / steps,
            "ranking_accuracy": sum(row["counterfactual_audit"]["ranking_accuracy"] * row["steps"] for row in group) / steps,
            "paired_margin_exceedances": sum(row["counterfactual_audit"]["paired_margin_exceedances"] for row in group),
        }
        summaries[arm] = {"episodes": len(group), "survival_rate": fmean(row["survived"] for row in group),
            "mean_steps": fmean(row["steps"] for row in group), "mean_reward": fmean(row["reward"] for row in group),
            "mean_decision_ms": sum(row["decision_ms"] for row in group) / steps,
            "mean_latency_ms": sum(row["decision_ms"] for row in group) / steps,
            "mean_audit_ms": sum(row["audit_ms"] for row in group) / steps,
            "fallback_rate": sum(row["fallback_steps"] for row in group) / steps,
            "overrides": overrides, "override_rate": overrides / steps,
            "mean_prediction_mse": sum(row["prediction_mse"] * row["steps"] for row in group) / steps,
            "per_channel_mse": [sum(row["per_channel_mse"][i] * row["steps"] for row in group) / steps for i in range(16)],
            "counterfactual_audit": audit}
        if arm != "policy":
            deltas = [row["reward"] - baseline[row["seed"]]["reward"] for row in group]
            rng = random.Random(782341)
            boot = sorted(fmean(rng.choices(deltas, k=len(deltas))) for _ in range(1000))
            paired[arm] = {"baseline": "policy", "mean_reward_delta": fmean(deltas), "ci95": [boot[24], boot[974]],
                "wins": sum(x > 1e-10 for x in deltas), "ties": sum(abs(x) <= 1e-10 for x in deltas), "losses": sum(x < -1e-10 for x in deltas),
                "per_seed": [{"seed": row["seed"], "reward_delta": delta, "survival_delta": int(row["survived"]) - int(baseline[row["seed"]]["survived"])} for row, delta in zip(group, deltas)]}
    probes = [row for row in rows if row["prediction_probe"]]
    comparison = {mode: {"anchors": sum(row["prediction_probe"][mode]["anchors"] for row in probes),
        "per_channel_mse": [sum(row["prediction_probe"][mode]["per_channel_mse"][i] * row["prediction_probe"][mode]["anchors"] for row in probes) / sum(row["prediction_probe"][mode]["anchors"] for row in probes) for i in range(16)]}
        for mode in MODES}
    for values in comparison.values():
        values["mse"] = fmean(values["per_channel_mse"])
        values["branches"] = values["anchors"] * 6
    return summaries, paired, comparison


def _episode(core, atlas_v2, contrast, mpc, arm, seed, spec, config):
    env = TidePool(seed, scarcity=spec.scarcity, max_steps=spec.max_steps)
    initial = env.snapshot()
    trace, counts = [], {name: 0 for name in ACTIONS}
    while not env.done:
        obs = env.observe()
        start = time.perf_counter()
        if arm.startswith("contrast"):
            decision = contrast.plan(obs, use_memory=arm != "contrast_no_memory",
                use_selection=arm != "contrast_unfiltered", gate="absolute" if arm == "contrast_absolute" else "paired")
        elif arm == "atlas_v2":
            decision = atlas_v2.plan(obs)
        elif arm == "neural_mpc":
            decision = mpc.plan(obs)
        else:
            action = core.act(obs) if arm == "policy" else teacher_action(obs) if arm == "heuristic" else min(5, int(event_uniform(seed, env.tick, "random-control", "action") * 6))
            decision = {"action": action, "action_name": ACTIONS[action]}
        elapsed = (time.perf_counter() - start) * 1000
        action = decision["action"]
        if type(action) is not int or not 0 <= action < 6:
            raise ValueError("Controller returned an invalid action")
        audit_start = time.perf_counter()
        _, logits = _base(core, [obs], [action])
        probabilities = logits[0].softmax(-1).tolist()
        policy = int(logits[0].argmax())
        if decision.get("policy_action", policy) != policy:
            raise ValueError("Recorded incumbent differs from core policy")
        decision["policy_action"] = policy
        if "policy_probabilities" in decision:
            recorded = _vector(decision["policy_probabilities"], 6)
            # CPU matrix kernels differ slightly between one-row and six-row passes.
            # Preserve the actual controller prior after checking this numerical bound.
            if any(abs(a - b) > 1e-5 for a, b in zip(recorded, probabilities)):
                raise ValueError("Controller policy probabilities differ from core")
            probabilities = recorded
        else:
            decision["policy_probabilities"] = probabilities
        audit = _audit(env, probabilities, config, decision)
        prediction = _prediction(decision, arm, probabilities, core, obs)
        probe = _probe(contrast, obs) if arm == "policy" else None
        audit_ms = (time.perf_counter() - audit_start) * 1000
        nxt, reward, _, info = env.step(action)
        errors = [(a - b)**2 for a, b in zip(prediction, nxt)]
        event = {"observation": obs, "action": action, "action_name": ACTIONS[action], "decision": decision,
                 "decision_ms": elapsed, "audit_ms": audit_ms, "audit": audit,
                 "predicted_observation": prediction, "prediction_squared_errors": errors,
                 "reward": reward, "next_observation": nxt, "info": info, "state": env.state()}
        if probe is not None:
            event["prediction_probe"] = probe
        trace.append(event)
        counts[ACTIONS[action]] += 1
    episode = {"schema": "poseidon-episode-v1", "controller": arm, "world_version": WORLD_VERSION,
        "seed": seed, "scarcity": spec.scarcity, "max_steps": spec.max_steps, "steps": env.tick,
        "survived": env.alive, "alive": env.alive, "death": not env.alive, "death_reason": env.death_reason,
        "truncated": info["truncated"], "terminal_reason": info["terminal_reason"], "reward": env.total_reward,
        "final_health": env.health, "action_counts": counts, "initial_snapshot": initial,
        "final_snapshot": env.snapshot(), "trajectory": trace}
    return _row(episode), episode


def run_experiment(core, atlas_v2, contrast, spec=None):
    from .planning import ModelPredictivePlanner, PlanningConfig
    spec = _spec(spec)
    sources = source_identity()
    frozen = contrast.artifact
    legacy = atlas_v2.artifact
    checkpoint_hash = hashlib.sha256(core.path.read_bytes()).hexdigest()
    model_hash = _model_digest(core)
    for artifact in (frozen, legacy):
        if artifact["checkpoint_sha256"] != checkpoint_hash or artifact["model_sha256"] != model_hash:
            raise ValueError("Experiment artifact does not match core checkpoint and model")
    seeds = set(range(spec.seed, spec.seed + spec.episodes))
    for artifact in (frozen, legacy):
        used = set().union(*(set(values) for name, values in artifact["partition"].items() if name.endswith("_seeds")))
        if seeds & used:
            raise ValueError("Experiment seeds overlap a fitting, selection or calibration partition")
    planning = PlanningConfig()
    protocol = {"schema": SCHEMA, "spec": asdict(spec), "arms": list(ARMS), "world_version": WORLD_VERSION,
        "source_sha256": sources, "checkpoint_sha256": checkpoint_hash, "model_sha256": model_hash,
        "atlas_v2_sha256": legacy["sha256"], "contrast_sha256": frozen["sha256"],
        "atlas_v2_partition": legacy["partition"], "contrast_partition": frozen["partition"],
        "contrast_training_samples": len(frozen["records"]),
        "contrast_config": frozen["config"], "contrast_selection": frozen["selection"], "contrast_calibration": frozen["calibration"],
        "neural_mpc_config": asdict(planning), "randomization": "blake2b-named-events-v1",
        "prediction_probe": "all six actions on identical incumbent-policy anchors; base/unfiltered/selected residual modes",
        "advantage_audit": "all six one-step branches, fixed reserve utility with recorded policy prior; evaluator-only snapshots",
        "runtime": {"device": "cpu", "torch_threads": __import__("torch").get_num_threads()}}
    mpc = ModelPredictivePlanner(core, planning)
    rows, episodes = [], []
    for index, seed in enumerate(sorted(seeds)):
        for arm in ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)]:
            row, episode = _episode(core, atlas_v2, contrast, mpc, arm, seed, spec, frozen["config"])
            rows.append(row)
            episodes.append(episode)
    if source_identity() != sources or hashlib.sha256(core.path.read_bytes()).hexdigest() != checkpoint_hash or _model_digest(core) != model_hash or contrast.artifact != frozen or atlas_v2.artifact != legacy:
        raise RuntimeError("Source, checkpoint or fitted artifact changed during experiment; no receipt published")
    summary, paired, comparison = paired_summary(rows)
    result = {"schema": SCHEMA, "experiment_id": digest(protocol), "protocol": protocol,
        "summary": summary, "paired": paired, "prediction_comparison": comparison, "rows": rows, "episodes": episodes,
        "contrast": {"ready": True, "artifact_sha256": frozen["sha256"], "fit_receipt": {
            "training_samples": len(frozen["records"]), "partition": frozen["partition"],
            "selection": frozen["selection"], "calibration": frozen["calibration"]}},
        "limits": LIMITS, "status": "completed", "promoted": False}
    result["receipt_sha256"] = digest(result)
    if len(json.dumps(result, allow_nan=False).encode()) > MAX_RECEIPT_BYTES:
        raise ValueError("Receipt exceeds the 128 MiB limit; use a smaller experiment")
    return result


def _validate_decision(event, arm, protocol):
    decision = event["decision"]
    action = event["action"]
    if not isinstance(decision, dict) or type(decision.get("action")) is not int or decision["action"] != action or type(decision.get("policy_action")) is not int or not 0 <= decision["policy_action"] < 6:
        raise ValueError("Invalid controller decision")
    probabilities = _vector(decision.get("policy_probabilities"), 6)
    if abs(sum(probabilities) - 1) > 1e-5:
        raise ValueError("Invalid policy probabilities")
    if max(range(6), key=lambda a: (probabilities[a], -a)) != decision["policy_action"]:
        raise ValueError("Incumbent policy identity differs from recorded probabilities")
    prediction = _vector(event["predicted_observation"], unit=False)
    if arm.startswith("contrast") or arm == "atlas_v2":
        candidates = decision.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 6:
            raise ValueError("Missing six candidate futures")
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict) or type(candidate.get("action")) is not int or candidate["action"] != index or candidate.get("action_name") != ACTIONS[index]:
                raise ValueError("Candidate action identity mismatch")
            _vector(candidate.get("predicted_observation"))
            _vector(candidate.get("base_observation"))
            if not isinstance(candidate.get("source_ids"), list) or any(not isinstance(item, str) or len(item) != 66 or not item.endswith(":" + str(index)) or any(c not in "0123456789abcdef" for c in item[:64]) for item in candidate["source_ids"]):
                raise ValueError("Invalid candidate source identities")
            for name in ("support_distance", "error_radius"):
                value = candidate.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid candidate diagnostic")
        if prediction != candidates[action]["predicted_observation"]:
            raise ValueError("Executed prediction differs from candidate")
        if arm.startswith("contrast"):
            config = protocol["contrast_config"]
            _vector(decision.get("feature_alphas"))
            mode = "base" if arm == "contrast_no_memory" else "unfiltered" if arm == "contrast_unfiltered" else "memory"
            alphas = [0.0] * 16 if mode == "base" else [1.0] * 16 if mode == "unfiltered" else protocol["contrast_selection"]["alphas"]
            if decision["feature_alphas"] != alphas or decision.get("gate") != ("absolute" if arm == "contrast_absolute" else "paired"):
                raise ValueError("Contrast mode evidence mismatch")
            for index, candidate in enumerate(candidates):
                value = _utility(candidate["predicted_observation"], probabilities[index], config)
                if value != candidate.get("point_value"):
                    raise ValueError("Candidate utility mismatch")
                calibration = protocol["contrast_calibration"][mode][index]
                radius = calibration["error_radius"]
                supported = candidate["support_distance"] <= config["support_radius"]
                trusted = supported and calibration["supported_count"] > 0 and radius <= config["max_vital_error"]
                lower = candidate["predicted_observation"][:]
                upper = lower[:]
                for channel in range(4):
                    lower[channel] = max(0.0, lower[channel] - radius)
                    upper[channel] = min(1.0, upper[channel] + radius)
                lower[4] = min(1.0, lower[4] + radius)
                upper[4] = max(0.0, upper[4] - radius)
                if (candidate["error_radius"] != radius or candidate.get("supported") is not supported
                    or candidate.get("trusted") is not trusted or candidate.get("calibration_count") != calibration["count"]
                    or candidate.get("supported_calibration_count") != calibration["supported_count"]
                    or candidate.get("feature_alphas") != alphas or candidate.get("policy_probability") != probabilities[index]
                    or candidate.get("value") != _utility(lower, probabilities[index], config)
                    or candidate.get("upper_value") != _utility(upper, probabilities[index], config)):
                    raise ValueError("Candidate bound or trust evidence mismatch")
            proposed = decision.get("proposed_action")
            if type(proposed) is not int or not 0 <= proposed < 6:
                raise ValueError("Invalid proposed action")
            advantage = candidates[proposed]["point_value"] - candidates[decision["policy_action"]]["point_value"]
            radius = protocol["contrast_calibration"]["paired"][mode]["error_radius"]
            if decision.get("point_advantage") != advantage or decision.get("advantage_error_radius") != radius:
                raise ValueError("Paired advantage evidence mismatch")
            if decision.get("override_accepted") != (action != decision["policy_action"]):
                raise ValueError("Override evidence mismatch")
            incumbent = decision["policy_action"]
            gate = decision["gate"]
            margin = advantage - radius if gate == "paired" else candidates[proposed]["value"] - candidates[incumbent]["upper_value"]
            if decision.get("empirical_advantage_margin") != margin:
                raise ValueError("Advantage margin arithmetic mismatch")
            valid = [candidate for candidate in candidates if candidate["trusted"]]
            rank = "point_value" if gate == "paired" else "value"
            expected_proposal = max(valid, key=lambda row: (row[rank], -row["action"]))["action"] if valid else incumbent
            reason, chosen = None, expected_proposal
            if not valid:
                reason = "outside_fitted_support" if not any(row["supported"] for row in candidates) else (
                    "no_supported_calibration" if not any(row["supported"] and row["supported_calibration_count"] > 0 for row in candidates)
                    else "prediction_error_exceeds_budget")
            elif expected_proposal != incumbent:
                if not candidates[incumbent]["trusted"]:
                    reason, chosen = "incumbent_prediction_not_supported", incumbent
                elif margin <= config["override_margin"]:
                    reason, chosen = "counterfactual_advantage_unresolved", incumbent
            if proposed != expected_proposal or action != chosen or decision.get("fallback_reason") != reason or decision.get("trusted") is not (bool(valid) and reason is None):
                raise ValueError("Executed action does not satisfy the frozen gate")
    elif arm == "neural_mpc":
        if prediction != decision["predicted_futures"][ACTIONS[action]]["step_1_observation"]:
            raise ValueError("Neural MPC prediction mismatch")
    if arm == "policy" and action != decision["policy_action"]:
        raise ValueError("Policy arm action mismatch")
    if (arm == "policy") != ("prediction_probe" in event):
        raise ValueError("Matched prediction probe appears in wrong arm")
    if arm == "policy":
        probe = event["prediction_probe"]
        if not isinstance(probe, dict) or set(probe) != set(MODES):
            raise ValueError("Incomplete matched prediction probe")
        for values in probe.values():
            if not isinstance(values, list) or len(values) != 6:
                raise ValueError("Invalid probe branches")
            for vector in values:
                _vector(vector)
    return probabilities


def verify_receipt(receipt):
    try:
        return _verify(receipt)
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("Malformed contrast receipt evidence") from error


def _verify(receipt):
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise ValueError("Unsupported contrast receipt")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if digest(unsigned) != receipt.get("receipt_sha256"):
        raise ValueError("Receipt checksum mismatch")
    if receipt.get("status") != "completed" or receipt.get("promoted") is not False:
        raise ValueError("Only completed, unpromoted contrast experiments can be verified")
    protocol = receipt["protocol"]
    if protocol.get("schema") != SCHEMA or protocol.get("arms") != list(ARMS) or digest(protocol) != receipt["experiment_id"] or protocol.get("world_version") != WORLD_VERSION:
        raise ValueError("Experiment protocol identity mismatch")
    contrast = receipt["contrast"]
    fit = contrast["fit_receipt"]
    count = protocol["contrast_training_samples"]
    partition = protocol["contrast_partition"]
    anchors = partition["anchors_per_episode"]
    if (type(anchors) is not int or not 1 <= anchors <= 256 or type(count) is not int
            or count % 6 or not 6 * len(partition["train_seeds"]) <= count
            <= min(protocol["contrast_config"]["max_samples"], 6 * len(partition["train_seeds"]) * anchors)):
        raise ValueError("Invalid contrast training sample count")
    if (contrast.get("ready") is not True or contrast.get("artifact_sha256") != protocol["contrast_sha256"]
            or fit.get("training_samples") != count
            or any(fit.get(name) != protocol["contrast_" + name] for name in ("partition", "selection", "calibration"))):
        raise ValueError("Embedded contrast fit metadata differs from protocol")
    current = source_identity()
    if any(protocol["source_sha256"].get(name) != current.get(name) for name in ("world.py", "contrast.py", "contrast_experiments.py")):
        raise ValueError("Replay requires the recorded simulator, utility and audit implementation")
    spec = _spec(ExperimentSpec(**protocol["spec"]))
    seeds = set(range(spec.seed, spec.seed + spec.episodes))
    for key, partitions in (("atlas_v2_partition", ("train_seeds", "calibration_seeds")), ("contrast_partition", ("train_seeds", "selection_seeds", "calibration_seeds"))):
        used = set()
        for name in partitions:
            values = protocol[key][name]
            if not isinstance(values, list) or not 1 <= len(values) <= 256 or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in values) or len(set(values)) != len(values) or used & set(values) or seeds & set(values):
                raise ValueError("Invalid or overlapping experiment partitions")
            used.update(values)
    expected = {(arm, seed) for arm in ARMS for seed in seeds}
    episodes, rows = receipt["episodes"], receipt["rows"]
    if not isinstance(episodes, list) or not isinstance(rows, list) or len(episodes) != len(expected) or len(rows) != len(expected):
        raise ValueError("Missing paired episodes")
    indexed = {}
    for row in rows:
        key = (row["arm"], row["seed"])
        if key not in expected or key in indexed:
            raise ValueError("Duplicate or unexpected outcome row")
        indexed[key] = row
    seen, transitions = set(), 0
    verified_rows = []
    for episode in episodes:
        key = (episode["controller"], episode["seed"])
        if key not in expected or key in seen:
            raise ValueError("Duplicate or unexpected episode")
        seen.add(key)
        if episode["schema"] != "poseidon-episode-v1" or episode["world_version"] != WORLD_VERSION or episode["max_steps"] != spec.max_steps or episode["scarcity"] != spec.scarcity:
            raise ValueError("Episode settings mismatch")
        env = TidePool(episode["seed"], scarcity=spec.scarcity, max_steps=spec.max_steps)
        if env.snapshot() != episode["initial_snapshot"]:
            raise ValueError("Initial snapshot mismatch")
        trace = episode["trajectory"]
        if not isinstance(trace, list) or not 1 <= len(trace) <= spec.max_steps:
            raise ValueError("Invalid trajectory length")
        counts = {name: 0 for name in ACTIONS}
        for event in trace:
            if env.done or event["observation"] != env.observe():
                raise ValueError("Observation or termination mismatch")
            action = event["action"]
            if type(action) is not int or not 0 <= action < 6 or event["action_name"] != ACTIONS[action]:
                raise ValueError("Invalid executed action")
            probabilities = _validate_decision(event, key[0], protocol)
            audit = _audit(env, probabilities, protocol["contrast_config"], event["decision"])
            if audit != event["audit"]:
                raise ValueError("Matched counterfactual audit mismatch")
            nxt, reward, _, info = env.step(action)
            if nxt != event["next_observation"] or reward != event["reward"] or info != event["info"] or env.state() != event["state"]:
                raise ValueError("Replay transition mismatch")
            errors = [(a - b)**2 for a, b in zip(event["predicted_observation"], nxt)]
            if errors != event["prediction_squared_errors"]:
                raise ValueError("Prediction error arithmetic mismatch")
            for name in ("decision_ms", "audit_ms"):
                value = event[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid recorded timing")
            counts[ACTIONS[action]] += 1
            transitions += 1
        values = {"steps": env.tick, "survived": env.alive, "alive": env.alive, "death": not env.alive,
                  "reward": env.total_reward, "death_reason": env.death_reason, "final_health": env.health,
                  "truncated": info["truncated"], "terminal_reason": info["terminal_reason"], "action_counts": counts}
        if not env.done or env.snapshot() != episode["final_snapshot"] or any(episode[name] != value for name, value in values.items()):
            raise ValueError("Terminal episode mismatch")
        row = _row(episode)
        if row != indexed[key]:
            raise ValueError("Episode summary arithmetic mismatch")
        verified_rows.append(row)
    summary, paired, comparison = paired_summary(verified_rows)
    if summary != receipt["summary"] or paired != receipt["paired"] or comparison != receipt["prediction_comparison"]:
        raise ValueError("Paired summary mismatch")
    return {"verified": True, "experiment_id": receipt["experiment_id"], "episodes_replayed": len(seen),
            "transitions_replayed": transitions, "counterfactual_branches_replayed": transitions * 6,
            "scope": "same-version simulation, all six action interventions and evidence arithmetic; not model authorship"}


def load_and_verify(path):
    path = Path(path)
    if path.stat().st_size > MAX_RECEIPT_BYTES:
        raise ValueError("Receipt exceeds the 128 MiB limit")
    return verify_receipt(json.loads(path.read_text(encoding="utf-8")))
