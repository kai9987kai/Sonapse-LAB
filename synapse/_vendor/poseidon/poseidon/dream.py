"""POSEIDON DREAM: Generative Sleep Replay & Counterfactual Memory Consolidation.

Inspired by mammalian sleep consolidation (Diekelmann & Born; Buzsáki hippocampal sharp-wave ripples)
and the Morpheus architecture:
1. Sleep-Triggered Activation:
   When the agent rests (action=0, "rest") or enters offline consolidation, the dream engine
   activates offline generative rollouts.

2. Counterfactual Dream Synthesis:
   Samples high-surprise (e_t >> \\bar{e}) or near-failure episodes from Mnemosyne.
   Recursively projects counterfactual branch trajectories through the Oracle world model:
     - "What if I had sheltered instead of exploring during the storm?"
     - "What if I had conserved stamina instead of sprinting?"

3. Schema Consolidation & Inoculation:
   Distills episodic traces into invariant semantic schemas and injects prioritized synthetic
   recovery trajectories into the ContinualAdapter replay buffer, inoculating the policy
   against edge-case catastrophes.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import math
import random
from typing import Any, Sequence

import torch

from .adapt import ReplayBuffer
from .core import CoreRuntime
from .mnemosyne import MnemosyneMemory
from .oracle import OracleEnsemble
from .world import ACTIONS, N_ACTIONS, OBS_SIZE


@dataclass(frozen=True)
class DreamConfig:
    dream_cycles_per_rest: int = 3
    dream_horizon: int = 4
    min_surprise_threshold: float = 0.12
    catastrophic_health_threshold: float = 0.25
    inoculation_priority: float = 2.5


@dataclass
class DreamSummary:
    dream_cycles: int
    episodes_inspected: int
    synthetic_rollouts_generated: int
    counterfactual_recovery_samples: int
    mean_dream_reward: float


class DreamConsolidator:
    """Simulates generative dream replay during resting intervals."""

    def __init__(
        self,
        core: CoreRuntime,
        oracle: OracleEnsemble,
        mnemosyne: MnemosyneMemory,
        replay_buffer: ReplayBuffer,
        config: DreamConfig | None = None,
    ):
        self.core = core
        self.oracle = oracle
        self.mnemosyne = mnemosyne
        self.replay = replay_buffer
        self.config = config or DreamConfig()
        self.consolidated_count: int = 0
        self.dream_history: list[dict[str, Any]] = []

    def consolidate_during_rest(self) -> DreamSummary:
        """Runs offline generative replay on high-surprise memory traces."""
        records = self.mnemosyne.episodic.records
        if not records:
            return DreamSummary(0, 0, 0, 0, 0.0)

        # Prioritize high surprise or critical health states
        salient_records = [
            r for r in records
            if r.surprise >= self.config.min_surprise_threshold or r.observation[0] <= self.config.catastrophic_health_threshold
        ]

        # Fallback to most recent if no extreme states found
        candidates = salient_records if salient_records else records[-min(5, len(records)):]

        cycles = min(self.config.dream_cycles_per_rest, len(candidates))
        selected_memories = random.sample(candidates, cycles)

        total_reward = 0.0
        synthetic_samples = 0
        recovery_samples = 0

        for mem in selected_memories:
            start_obs = torch.tensor(mem.observation, dtype=torch.float32)
            core_out = self.core.model(["Survive"], start_obs.reshape(1, -1))
            latent_state = core_out["memory"].squeeze(0)

            # Test alternative counterfactual actions distinct from what was actually done
            actual_action = mem.action
            counterfactual_actions = [a for a in range(N_ACTIONS) if a != actual_action]
            alt_action = random.choice(counterfactual_actions)

            # Dream sequence: alternative action followed by baseline policy
            seq = [alt_action] + [0] * (self.config.dream_horizon - 1)
            dream_res = self.oracle.imagine_trajectory(latent_state, start_obs, seq)

            tot_r = dream_res["total_predicted_reward"]
            total_reward += tot_r

            # Inoculate: If dream trajectory found safe survival, inject into replay buffer
            if (dream_res.get("failure_probability_calibrated", False)
                    and dream_res.get("survival_probability") is not None
                    and dream_res["survival_probability"] > 0.8):
                recovery_samples += 1
                traj = dream_res["trajectory"]
                curr_o = mem.observation
                for step_data in traj:
                    next_o = step_data["predicted_obs"]
                    self.replay.add(
                        obs=curr_o,
                        action=step_data["action"],
                        reward=step_data["step_reward"],
                        next_obs=next_o,
                        priority=self.config.inoculation_priority,
                    )
                    synthetic_samples += 1
                    curr_o = next_o

        self.consolidated_count += synthetic_samples
        summary = DreamSummary(
            dream_cycles=cycles,
            episodes_inspected=len(records),
            synthetic_rollouts_generated=synthetic_samples,
            counterfactual_recovery_samples=recovery_samples,
            mean_dream_reward=float(total_reward / max(1, cycles)),
        )
        self.dream_history.append(asdict(summary))
        return summary
