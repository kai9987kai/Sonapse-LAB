import pytest

from poseidon.genesis_3d import ContinuousTidePool3D, Genesis3DConfig
from poseidon.world import OBS_SIZE


def test_genesis_3d_simulation_step():
    env = ContinuousTidePool3D(seed=42)
    obs, r, done, info = env.step(action=4)  # explore

    assert len(obs) == OBS_SIZE
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert info["alive"] is True
    assert 0.0 <= obs[0] <= 1.0  # health
    assert 0.0 <= obs[1] <= 1.0  # energy
    assert 0.0 <= obs[2] <= 1.0  # hydration


def test_genesis_3d_multi_step_trajectory():
    env = ContinuousTidePool3D(seed=101)
    for step_i in range(16):
        obs, r, done, info = env.step(action=step_i % 6)
        if done:
            break
    assert env.tick > 0
    assert len(env.patches) == 12
