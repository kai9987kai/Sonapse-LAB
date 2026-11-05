import json
import pytest

from poseidon.world import ACTIONS, N_ACTIONS, OBS_SIZE
from poseidon.worldlab import (
    POMDPConfig,
    POMDPTidePool,
    StepResult,
    WorldBridgeProtocol,
)


def test_step_result_serialization():
    res = StepResult(
        observation=[0.5] * OBS_SIZE,
        reward=1.234567,
        done=False,
        truncated=False,
        info={"test": True},
    )
    d = res.to_dict()
    assert len(d["observation"]) == OBS_SIZE
    assert d["reward"] == 1.234567
    assert d["done"] is False
    assert d["info"]["test"] is True


def test_pomdp_tidepool_reset_and_step():
    cfg = POMDPConfig(sensor_noise_std=0.01, fog_threshold=0.5, tidal_amplitude=0.2)
    env = POMDPTidePool(config=cfg, max_steps=16)

    obs = env.reset(seed=42)
    assert len(obs) == OBS_SIZE
    assert all(0.0 <= x <= 1.0 for x in obs)

    step_res = env.step(action=0)
    assert isinstance(step_res, StepResult)
    assert len(step_res.observation) == OBS_SIZE
    assert isinstance(step_res.reward, float)
    assert isinstance(step_res.done, bool)
    assert isinstance(step_res.truncated, bool)


def test_pomdp_fog_and_sensor_noise():
    # Test zero-noise vs noisy observations
    cfg_clean = POMDPConfig(sensor_noise_std=0.0)
    cfg_noisy = POMDPConfig(sensor_noise_std=0.05)

    env_clean = POMDPTidePool(config=cfg_clean, max_steps=10)
    env_noisy = POMDPTidePool(config=cfg_noisy, max_steps=10)

    obs_clean = env_clean.reset(seed=99)
    obs_noisy = env_noisy.reset(seed=99)

    # They should differ due to noise injection
    assert obs_clean != obs_noisy


def test_world_bridge_protocol():
    # Action encode/decode
    encoded_act = WorldBridgeProtocol.encode_action(3)
    assert b"poseidon-worldbridge-v1" in encoded_act
    assert b"ACTION_DISPATCH" in encoded_act
    decoded_act = WorldBridgeProtocol.decode_action(encoded_act)
    assert decoded_act == 3

    # StepResult encode/decode
    res = StepResult(
        observation=[0.123456] * OBS_SIZE,
        reward=0.75,
        done=True,
        truncated=False,
        info={"alive": False},
    )
    encoded_step = WorldBridgeProtocol.encode_step_result(res)
    assert b"STEP_RESULT" in encoded_step
    decoded_step = WorldBridgeProtocol.decode_step_result(encoded_step)
    assert len(decoded_step.observation) == OBS_SIZE
    assert decoded_step.reward == 0.75
    assert decoded_step.done is True
    assert decoded_step.info["alive"] is False


def test_pomdp_snapshot_replays_sensor_rng_and_reset():
    env = POMDPTidePool(POMDPConfig(seed=19, sensor_noise_std=0.2), max_steps=8)
    first = env.reset()
    env.step(4)
    env.observe()  # Snapshot must preserve arbitrary sensor draws, not just tick.
    env.rng.gauss(0.0, 0.2)  # Also preserve Random's cached second Gaussian draw.
    assert env.rng.getstate()[2] is not None
    snapshot = json.loads(json.dumps(env.snapshot()))
    restored = POMDPTidePool.from_snapshot(snapshot)
    assert restored.observe() == env.observe()
    assert restored.step(2).to_dict() == env.step(2).to_dict()
    assert restored.snapshot() == env.snapshot()
    assert env.reset() == first


def test_pomdp_legacy_snapshot_cannot_claim_exact_sensor_replay():
    env = POMDPTidePool()
    with pytest.raises(ValueError, match="sensor"):
        POMDPTidePool.from_snapshot(env.underlying.snapshot())
