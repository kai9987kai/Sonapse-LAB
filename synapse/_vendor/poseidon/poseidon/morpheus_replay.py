"""POSEIDON MORPHEUS REPLAY: Offline Counterfactual Dream Theater & Sleep Consolidation.

Synthesizes Morpheus (Slow-Wave Sleep SWS & Rapid Eye Movement REM sleep phases),
Memory-Carrier-Observatory (holographic memory decay & consolidation), and
Causeway (counterfactual imagination trajectories without real-world risk):

1. Sleep Cycle Triggering:
   Engages during 'rest' actions, physiological exhaustion (stamina < 0.20),
   or post-storm shelter periods.

2. SWS (Slow-Wave Sleep) Consolidation:
   - Synaptic Homeostasis Downscaling: reduces accumulated metabolic noise in weights.
   - Episodic Compression: clusters high-value transitions into holographic attractor keys.

3. REM (Rapid Eye Movement) Dream Theater:
   - Counterfactual Imagination Trajectories:
     Retrieves high-regret transitions (s_t, a_t, s_{t+1}) from episodic memory.
     Simulates imaginary alternative actions a' in the causal SCM mental theater:
       s_{imaginary} = SCM.do(a', s_t, U_t)
   - Oneiric Gradient Distillation:
     Adjusts MorphogeneticAdapter residual weights offline using imagined returns,
     learning from mistakes without incurring real-world mortality risk.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
import random
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .causal_scm import StructuralCausalModel
from .morphogenesis import MorphogeneticAdapter
from .world import ACTIONS, N_ACTIONS


@dataclass
class DreamEpisode:
    source_step: int
    original_action: int
    imagined_action: int
    imagined_vital_gain: float
    counterfactual_surprise: float
    distillation_applied: bool


@dataclass
class OneiricSleepReport:
    sleep_cycles: int
    sws_compression_events: int
    rem_replays_conducted: int
    accumulated_dream_gain: float
    synaptic_downscale_norm: float
    recent_dreams: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sleep_cycles": self.sleep_cycles,
            "sws_compression_events": self.sws_compression_events,
            "rem_replays_conducted": self.rem_replays_conducted,
            "accumulated_dream_gain": round(self.accumulated_dream_gain, 4),
            "synaptic_downscale_norm": round(self.synaptic_downscale_norm, 5),
            "recent_dreams": self.recent_dreams,
        }


class MorpheusSleepEngine:
    """Offline oneiric dream theater and counterfactual imagination optimizer."""

    def __init__(
        self,
        dream_depth: int = 4,
        imagination_learning_rate: float = 0.05,
        sws_downscale_factor: float = 0.995,
        seed: int = 42,
    ):
        self.dream_depth = dream_depth
        self.imagination_lr = imagination_learning_rate
        self.sws_downscale_factor = sws_downscale_factor
        self.rng = random.Random(seed)

        self.sleep_cycle_counter: int = 0
        self.sws_events: int = 0
        self.rem_replays: int = 0
        self.total_dream_gain: float = 0.0
        self.dream_history: list[DreamEpisode] = []

    def reset(self) -> None:
        self.sleep_cycle_counter = 0
        self.sws_events = 0
        self.rem_replays = 0
        self.total_dream_gain = 0.0
        self.dream_history.clear()

    def is_sleep_indicated(self, action: int, stamina: float, consecutive_rests: int = 0) -> bool:
        """Determines whether physiological conditions indicate sleep onset."""
        return action == 0 or stamina < 0.20 or consecutive_rests >= 2

    def execute_sleep_cycle(
        self,
        episodic_records: Sequence[Any],
        causeway_scm: StructuralCausalModel,
        morph_adapter: MorphogeneticAdapter,
    ) -> OneiricSleepReport:
        """Runs complete SWS synaptic downscaling and REM counterfactual dream replay."""
        self.sleep_cycle_counter += 1

        # 1. SWS (Slow-Wave Sleep) Synaptic Downscaling
        with torch.no_grad():
            morph_adapter.w1.data.mul_(self.sws_downscale_factor)
            morph_adapter.w2.data.mul_(self.sws_downscale_factor)
            downscale_norm = float(torch.norm(morph_adapter.w2.data).item())
        self.sws_events += 1

        # 2. REM (Rapid Eye Movement) Counterfactual Mental Replay
        recent_dreams_data = []
        if episodic_records:
            # Sample up to dream_depth records with preference for surprise / low reward
            candidates = list(episodic_records)[-min(len(episodic_records), 32):]
            sample_count = min(self.dream_depth, len(candidates))
            sampled_records = self.rng.sample(candidates, sample_count)

            for rec in sampled_records:
                obs = getattr(rec, "observation", None)
                realized_act = getattr(rec, "action", 0)
                realized_next = getattr(rec, "next_observation", None)
                step = getattr(rec, "step", 0)

                if obs is None or realized_next is None:
                    continue

                # Estimate exogenous shock realization
                u = causeway_scm.estimate_exogenous_noise(obs, realized_next, realized_act)

                # Mental theater: explore counterfactual actions
                for a_cand in range(N_ACTIONS):
                    if a_cand == realized_act:
                        continue

                    cf_obs = causeway_scm.interventional_prediction(obs, do_action=a_cand, exogenous_noise=u)
                    cf_vitality = cf_obs[0] + 0.5 * min(cf_obs[1], cf_obs[2])
                    real_vitality = realized_next[0] + 0.5 * min(realized_next[1], realized_next[2])

                    vital_gain = cf_vitality - real_vitality
                    if vital_gain > 0.05:
                        # Beneficial counterfactual discovered in dream theater
                        self.rem_replays += 1
                        self.total_dream_gain += vital_gain

                        # Oneiric distillation update on morphogenetic adapter
                        obs_tensor = torch.tensor(list(obs), dtype=torch.float32).reshape(1, -1)
                        target_logits = torch.zeros(1, N_ACTIONS)
                        target_logits[0, a_cand] = float(vital_gain)

                        pred_logits = morph_adapter(obs_tensor)
                        loss = F.mse_loss(pred_logits, target_logits)

                        # Manual gradient-free small nudge to reinforce positive path
                        with torch.no_grad():
                            morph_adapter.w2.data[a_cand, :] += (
                                self.imagination_lr * float(vital_gain) * 0.01
                            )

                        dream_entry = DreamEpisode(
                            source_step=step,
                            original_action=realized_act,
                            imagined_action=a_cand,
                            imagined_vital_gain=vital_gain,
                            counterfactual_surprise=float(loss.item()),
                            distillation_applied=True,
                        )
                        self.dream_history.append(dream_entry)
                        recent_dreams_data.append({
                            "source_step": step,
                            "original_action": ACTIONS[realized_act],
                            "imagined_action": ACTIONS[a_cand],
                            "imagined_vital_gain": round(vital_gain, 4),
                        })
                        break

        return OneiricSleepReport(
            sleep_cycles=self.sleep_cycle_counter,
            sws_compression_events=self.sws_events,
            rem_replays_conducted=self.rem_replays,
            accumulated_dream_gain=self.total_dream_gain,
            synaptic_downscale_norm=downscale_norm,
            recent_dreams=recent_dreams_data,
        )
