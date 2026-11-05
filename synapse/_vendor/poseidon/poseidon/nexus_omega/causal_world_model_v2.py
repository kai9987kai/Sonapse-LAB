"""Factored NEXUS-Ω prior aligned to `nexus-phaseworld-v2`."""
from __future__ import annotations

from typing import Optional

import numpy as np

from poseidon.nexus_omega.causal_world_model import FactoredCausalWorldModel, FactoredMember


class ViablePhaseWorldFactoredMember(FactoredMember):
    """Structured prior corrected for v2 storm shelter and Rest dynamics."""

    def forward(
        self,
        s_endo: np.ndarray,
        s_exo: np.ndarray,
        z_regime: np.ndarray,
        mem_working: np.ndarray,
        action: int,
    ):
        delta_endo, delta_exo, reward = super().forward(
            s_endo, s_exo, z_regime, mem_working, action
        )
        is_storm = (
            (len(s_endo) > 3 and float(s_endo[3]) > 0.8)
            or (len(mem_working) > 0 and float(mem_working[0]) > 0.1)
            or (len(z_regime) > 0 and float(z_regime[0]) > 0.1)
        )
        if action == 1 and is_storm and len(delta_endo) > 0:
            # The v1 prior invented health recovery from sheltering in a storm.
            delta_endo = delta_endo.copy()
            delta_endo[0] -= 0.05
        if action == 3 and len(delta_endo) > 2:
            delta_endo = delta_endo.copy()
            delta_endo[2] += 0.12
        return delta_endo, delta_exo, reward


class ViablePhaseWorldFactoredModel(FactoredCausalWorldModel):
    """Eight-channel/four-action world model for the versioned viable world."""

    PROFILE = "nexus-phaseworld-v2"

    def __init__(
        self,
        d_obs: int = 8,
        d_mem: int = 18,
        n_actions: int = 4,
        n_members: int = 3,
        base_seed: int = 300,
    ):
        if (d_obs, n_actions, n_members) != (8, 4, 3):
            raise ValueError("phaseworld-v2 is defined for 8 observations, 4 actions, and 3 members")
        super().__init__(d_obs=d_obs, d_mem=d_mem, n_actions=n_actions, n_members=n_members, base_seed=base_seed)
        self.members = [
            ViablePhaseWorldFactoredMember(
                d_endo=self.d_endo,
                d_exo=self.d_exo,
                d_regime=self.d_regime,
                d_mem=d_mem,
                n_actions=n_actions,
                seed=base_seed + i * 50,
            )
            for i in range(n_members)
        ]
        self.dynamics_profile = self.PROFILE
