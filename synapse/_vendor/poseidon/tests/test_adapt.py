import json
import pytest
import torch

from poseidon.adapt import (
    AdaptConfig,
    AnchorDataset,
    ContinualAdapter,
    ReplayBuffer,
    ResidualPolicyAdapter,
)
from poseidon.core import CoreRuntime, active_core_path
from poseidon.world import N_ACTIONS, OBS_SIZE


@pytest.fixture(scope="module")
def core():
    return CoreRuntime(active_core_path())


def test_residual_policy_adapter_zero_init():
    adapter = ResidualPolicyAdapter(obs_size=OBS_SIZE, action_count=N_ACTIONS, hidden_size=32)
    obs = torch.randn(2, OBS_SIZE)
    base_logits = torch.randn(2, N_ACTIONS)

    # At init, output must be identical to base_logits
    adapted = adapter(obs, base_logits)
    assert torch.allclose(adapted, base_logits, atol=1e-6)


def test_replay_buffer_and_anchor_dataset(core):
    buffer = ReplayBuffer(capacity=10)
    for i in range(12):
        buffer.add(
            obs=[0.1] * OBS_SIZE,
            action=0,
            reward=1.0,
            next_obs=[0.2] * OBS_SIZE,
        )
    assert len(buffer) == 10
    batch = buffer.sample(4)
    assert len(batch) == 4

    anchors = AnchorDataset(capacity=20)
    anchors.populate_from_core(core, count=10)
    assert len(anchors.items) == 10
    adapter = ResidualPolicyAdapter(obs_size=OBS_SIZE, action_count=N_ACTIONS)
    retention = anchors.evaluate_retention(adapter, core)
    # Identical at initialization -> 100% retention
    assert retention == 1.0


def test_continual_adapter_training_and_audit(core, tmp_path):
    cfg = AdaptConfig(
        max_training_steps=5,
        batch_size=4,
        eval_seeds=(101, 103),
        promotion_min_delta=0.01,
        max_anchor_regression=0.05,
    )
    cadapter = ContinualAdapter(core, config=cfg)

    # Populate dummy replay
    for i in range(10):
        cadapter.replay.add(
            obs=[0.5] * OBS_SIZE,
            action=i % N_ACTIONS,
            reward=float(i % 2),
            next_obs=[0.5] * OBS_SIZE,
        )

    train_receipt = cadapter.train_candidate()
    assert train_receipt["steps"] == 5

    ledger_file = tmp_path / "ledger.json"
    audit = cadapter.evaluate_and_audit_promotion(ledger_path=ledger_file)

    assert audit["status"] in ("PROMOTED", "REJECTED")
    assert "anchor_retention" in audit
    assert "return_delta" in audit
    assert ledger_file.is_file()

    ledger_content = json.loads(ledger_file.read_text(encoding="utf-8"))
    assert len(ledger_content) == 1
    assert ledger_content[0]["status"] == audit["status"]
