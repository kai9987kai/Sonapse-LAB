import pytest
import copy
import hashlib
import json

from poseidon.cognitive_experiment import (
    ARMS,
    COGNITIVE_EXPERIMENT_SCHEMA,
    run_cognitive_benchmark,
    verify_cognitive_receipt,
)
from poseidon.core import active_core_path


def test_micro_cognitive_benchmark():
    result = run_cognitive_benchmark(
        core_path=active_core_path(),
        seed_base=90000000,
        episodes=2,
        max_steps=4,
        scarcity=2.5,
        use_pomdp=False,
    )

    assert result["schema"] == COGNITIVE_EXPERIMENT_SCHEMA
    assert "sha256" in result
    for arm in ARMS:
        assert arm in result["arms"]
        assert result["arms"][arm]["episodes"] == 2

    assert "intact_vs_frozen_tidal" in result["paired"]
    assert "intact_vs_memory_free" in result["paired"]

    # Verify receipt hash
    assert verify_cognitive_receipt(result) is True


def _reseal(receipt):
    receipt.pop("sha256", None)
    receipt["sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    return receipt


def test_empty_resealed_receipt_rejected():
    with pytest.raises(ValueError):
        verify_cognitive_receipt(_reseal({"schema": COGNITIVE_EXPERIMENT_SCHEMA,
                                        "episodes_data": {}}))


@pytest.fixture(scope="module", params=[False, True], ids=["normal", "noisy"])
def replay_receipt(request):
    return run_cognitive_benchmark(episodes=1, max_steps=2, seed_base=91234567,
                                   use_pomdp=request.param)


def test_cognitive_full_replay(replay_receipt):
    # Exercise the persisted JSON representation, including sensor RNG tuples.
    assert verify_cognitive_receipt(json.loads(json.dumps(replay_receipt))) is True


@pytest.mark.parametrize("mutation", ["schema", "action", "reward", "death", "missing_pair",
                                       "duplicate_pair", "summary", "model", "source", "extra_field",
                                       "observation", "nan_observation", "missing_decision", "info",
                                       "truncated", "empty_trajectory", "incomplete_trajectory", "extra_step",
                                       "decision_telemetry", "nested_extra_field", "snapshot_seed"])
def test_cognitive_resealed_mutants_fail(replay_receipt, mutation):
    receipt = copy.deepcopy(replay_receipt)
    episode = receipt["episodes_data"]["intact"][0]
    if mutation == "schema":
        receipt["schema"] = "unsupported"
    elif mutation == "action":
        episode["trajectory"][0]["action"] = (episode["trajectory"][0]["action"] + 1) % 6
    elif mutation == "reward":
        episode["trajectory"][0]["reward"] += 1.0
    elif mutation == "death":
        episode["survived"] = not episode["survived"]
    elif mutation == "missing_pair":
        receipt["episodes_data"]["intact"] = []
    elif mutation == "duplicate_pair":
        receipt["episodes_data"]["intact"].append(copy.deepcopy(episode))
    elif mutation == "summary":
        receipt["arms"]["intact"]["mean_reward"] += 1.0
    elif mutation == "model":
        receipt["model"]["checkpoint_sha256"] = "0" * 64
    elif mutation == "source":
        receipt["source_sha256"]["world.py"] = "0" * 64
    elif mutation == "observation":
        episode["trajectory"][0]["observation"][0] = 0.17
    elif mutation == "nan_observation":
        episode["trajectory"][0]["next_observation"][0] = float("nan")
    elif mutation == "missing_decision":
        del episode["trajectory"][0]["decision"]
    elif mutation == "info":
        episode["trajectory"][-1]["info"]["death"] = True
    elif mutation == "truncated":
        episode["truncated"] = False
    elif mutation == "empty_trajectory":
        episode["trajectory"] = []
        episode["steps"] = 0
    elif mutation == "incomplete_trajectory":
        episode["trajectory"].pop()
        episode["steps"] -= 1
    elif mutation == "extra_step":
        episode["trajectory"].append(copy.deepcopy(episode["trajectory"][-1]))
        episode["steps"] += 1
    elif mutation == "decision_telemetry":
        episode["trajectory"][0]["decision"]["surprise"] += 1.0
    elif mutation == "nested_extra_field":
        episode["trajectory"][0]["fabricated_claim"] = 42
    elif mutation == "snapshot_seed":
        snapshot = episode["initial_snapshot"]
        snapshot.get("underlying", snapshot)["seed"] += 1
    else:
        receipt["fabricated_claim"] = 42
    with pytest.raises(ValueError):
        verify_cognitive_receipt(_reseal(receipt))


def test_legacy_v2_is_explicitly_unsupported(replay_receipt):
    receipt = copy.deepcopy(replay_receipt)
    receipt["schema"] = "poseidon-cognitive-experiment-v2"
    with pytest.raises(ValueError, match="v2"):
        verify_cognitive_receipt(_reseal(receipt))


def test_wrong_checkpoint_cannot_replay_receipt(replay_receipt, tmp_path):
    import torch
    from poseidon.core import CoreRuntime

    payload = CoreRuntime(active_core_path()).payload
    first_weight = next(iter(payload["state_dict"].values()))
    first_weight.add_(0.01)
    path = tmp_path / "wrong-core.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="model identity"):
        verify_cognitive_receipt(replay_receipt, core_path=path)


def test_oracle_artifact_bound_and_replayed(tmp_path, monkeypatch):
    import torch
    from poseidon.cognitive_agent import CognitiveAgent
    from poseidon.core import CoreRuntime
    from poseidon.research.oracle_training import fit_oracle

    directory = tmp_path / "cognitive-fit"
    fit_oracle(directory, base_seed=633000000, train_episodes=12, selection_episodes=2,
               calibration_episodes=4, max_steps=24, training_updates=2, batch_size=8,
               hidden_size=16, ensemble_size=3, coverage=0.8)
    path = directory / "oracle.pt"
    agent = CognitiveAgent(core=CoreRuntime(active_core_path()), oracle_checkpoint=path, mode="memory_free")
    obs = [0.5] * 16
    decision = agent.act(obs)
    assert decision.planning_enabled
    with torch.no_grad():
        out = agent.oracle(None, torch.tensor([obs]), torch.tensor([decision.action]))
        predicted = (torch.tensor([obs]) + out["mean_delta"]).clamp(0, 1)[0].tolist()

    def cannot_predict_after_execution(*args, **kwargs):
        pytest.fail("The executed-action forecast must have been cached before the successor arrived")

    monkeypatch.setattr(agent.oracle, "forward", cannot_predict_after_execution)
    agent.observe_transition(obs, decision.action, 0.0, predicted)
    assert agent.metacognition.surprise_history[-1] == 0.0
    receipt = run_cognitive_benchmark(episodes=1, max_steps=2, seed_base=633100000,
                                      oracle_checkpoint=path, use_pomdp=True)
    assert receipt["oracle_artifact"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for arm in ARMS[:-1]:
        recorded = receipt["episodes_data"][arm][0]["trajectory"][0]["decision"]
        assert recorded["planning_enabled"]
        assert recorded["oracle"]["member_forward_calls"] > 0
    assert verify_cognitive_receipt(receipt)
    mutant = copy.deepcopy(receipt)
    mutant["oracle_artifact"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ORACLE"):
        verify_cognitive_receipt(_reseal(mutant))
