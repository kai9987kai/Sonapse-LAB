import hashlib
import json
from pathlib import Path

import pytest
import torch

from poseidon.core import CoreConfig, CoreRuntime, TidalCore, hash_features, save_checkpoint


def test_prompt_hashing_is_deterministic_and_retains_distinct_words():
    a = hash_features(["A red cube", "A blue sphere"], buckets=64)
    torch.manual_seed(947)
    assert torch.equal(a, hash_features(["A red cube", "A blue sphere"], buckets=64))
    assert not torch.equal(a[0], a[1])
    assert torch.isfinite(hash_features(["", "é ocean 🌊"], buckets=64)).all()


def test_inflected_scene_vocabulary_has_the_same_features():
    assert torch.equal(hash_features(["cubes bouncing spheres orbiting cylinders spinning pyramids stationary"]),
                       hash_features(["cube bounce sphere orbit cylinder spin pyramid still"]))


def test_joint_outputs_are_differentiable_and_action_conditioned():
    model = TidalCore(CoreConfig(hash_buckets=64, hidden_size=24, experts=4, recurrent_steps=2))
    obs = torch.rand(3, 16)
    output = model(["red cube", "blue sphere", "green pyramid"], obs, torch.tensor([0, 1, 2]))
    assert output["scene"]["shape"].shape == (3, 4)
    assert output["action"].shape == (3, 6)
    assert output["delta"].shape == (3, 16)
    assert torch.allclose(output["routing"].sum(-1), torch.ones(3), atol=1e-6)
    loss = output["scene"]["shape"].square().mean() + output["action"].square().mean() + output["delta"].square().mean()
    loss.backward()
    assert model.text_projection.weight.grad.abs().sum() > 0
    changed = model(["red cube"] * 3, obs, torch.tensor([5, 5, 5]))
    original = model(["red cube"] * 3, obs, torch.tensor([0, 0, 0]))
    assert not torch.allclose(changed["delta"], original["delta"])


def test_checkpoint_validates_manifest_and_runtime(tmp_path):
    model = TidalCore(CoreConfig(hash_buckets=64, hidden_size=24))
    path = tmp_path / "core.pt"
    save_checkpoint(path, model, receipt={"examples_seen": 0})
    runtime = CoreRuntime(path)
    scene = runtime.scene("A red cube")
    assert scene["shape"] in {"cube", "sphere", "pyramid", "cylinder"}
    assert scene["count"] in {1, 2, 3}
    assert 0 <= runtime.act([0.0] * 16) < 6
    assert runtime.diagnostics()["examples_seen"] == 0
    payload = torch.load(path, weights_only=True)
    payload["config"]["hidden_size"] = 25
    torch.save(payload, path)
    with pytest.raises(ValueError, match="config.*hash"):
        CoreRuntime(path)


def test_invalid_observations_fail_before_inference():
    model = TidalCore(CoreConfig(hash_buckets=64, hidden_size=24))
    with pytest.raises(ValueError, match="finite"):
        model(["test"], torch.full((1, 16), float("nan")))
    with pytest.raises(ValueError, match="16"):
        model(["test"], torch.zeros(1, 15))


def test_exact_next_training_update_after_resume(tmp_path):
    from poseidon.train_core import restore_training_state, training_state

    torch.manual_seed(22)
    model = TidalCore(CoreConfig(hash_buckets=64, hidden_size=24))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    sampler = torch.Generator().manual_seed(19)

    def update(m, o, g):
        obs = torch.rand(4, 16, generator=g)
        result = m(["red cube"] * 4, obs, torch.tensor([0, 1, 2, 3]))
        loss = result["action"].square().mean() + result["delta"].square().mean()
        o.zero_grad(); loss.backward(); o.step()

    update(model, optimizer, sampler)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model, receipt={}, training=training_state(optimizer, sampler, epoch=0, batch=1, examples_seen=4))
    update(model, optimizer, sampler)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    restored = CoreRuntime(path).model
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
    restored_sampler = torch.Generator()
    payload = torch.load(path, weights_only=True)
    restore_training_state(payload["training"], restored_optimizer, restored_sampler)
    update(restored, restored_optimizer, restored_sampler)
    for name, parameter in restored.state_dict().items():
        assert torch.equal(parameter, expected[name]), name
