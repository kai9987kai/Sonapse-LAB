import numpy as np
import pytest

from poseidon.neuropil import SparseNeuropilArbiter


def test_kenyon_cell_apl_sparsity_enforcement():
    arbiter = SparseNeuropilArbiter(obs_dim=16, seed=123)
    dummy_obs = np.random.uniform(0.0, 1.0, size=16)
    kc_activity = arbiter.compute_kenyon_activity(dummy_obs)
    
    # Exactly k=8 Kenyon cells should be active
    active_count = np.count_nonzero(kc_activity)
    assert active_count == 8
    # Sparsity ratio should be 8/128 = 0.0625
    assert abs(active_count / 128.0 - 0.0625) < 1e-4


def test_homeostatic_drives_calculation():
    arbiter = SparseNeuropilArbiter(obs_dim=16)
    # Obs: low energy (0.1), low stamina (0.2), high depth (0.9), close predator (0.1)
    obs = [0.1, 0.2, 0.1, 0.9, 0.0, 0.0, 0.5, 0.0, 0.8, 0.0, 0.1, 0.0, 1, 0.5, 0.1, 10]
    drives = arbiter.compute_homeostatic_drives(obs)
    
    assert drives["metabolic"] > 0.8
    assert drives["fatigue"] > 0.7
    assert drives["threat"] > 0.5


def test_neuropil_descending_arbitration_channels():
    arbiter = SparseNeuropilArbiter(obs_dim=16)
    # Hungry state
    hungry_obs = [0.05, 0.9, 0.1, 0.2, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 1.0, 0.0, 1, 0.1, 0.8, 5]
    decision = arbiter.arbitrate(hungry_obs)
    assert decision["winning_channel"] == "harvester"
    assert decision["confidence"] > 0.3
    
    # Endangered state (high predator proximity)
    endangered_obs = [0.8, 0.8, 0.8, 0.9, 0.0, 0.0, 0.8, 0.0, 0.05, 0.0, 0.05, 0.0, 1, 0.9, 0.1, 5]
    decision_escaper = arbiter.arbitrate(endangered_obs)
    assert decision_escaper["winning_channel"] == "escaper"
