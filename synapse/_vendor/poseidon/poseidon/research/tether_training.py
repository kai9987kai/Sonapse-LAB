"""TETHER 4-View Causal Training & 7-Arm Evaluation Suite for Poseidon ORACLE.

Implements:
- 4-View transition batch generation (Factual, Relevant, Irrelevant, Damaged)
- Differentiable 7-arm objective matching Supermix-Tether:
  ('passive', 'dropout', 'iit', 'effect_only', 'invariance_only', 'tether', 'unpaired')
- Zero-shot 1-of-5 and 2-of-5 corrupted specialist evaluation
- Rejection and calibration metrics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .tether_gate import TetherReliabilityGate
from ..world import N_ACTIONS, OBS_SIZE


TETHER_ARMS = ("passive", "dropout", "iit", "effect_only", "invariance_only", "tether", "unpaired")


class TetherTransitionBatch:
    """Container for 4-view paired causal training views."""

    def __init__(
        self,
        factual_obs: torch.Tensor,
        factual_action: torch.Tensor,
        factual_delta: torch.Tensor,
        factual_reward: torch.Tensor,
        relevant_obs: torch.Tensor,
        relevant_action: torch.Tensor,
        relevant_delta: torch.Tensor,
        relevant_reward: torch.Tensor,
        irrelevant_obs: torch.Tensor,
        corrupted_deltas: torch.Tensor,
        corrupted_indices: Sequence[int],
    ):
        self.factual_obs = factual_obs
        self.factual_action = factual_action
        self.factual_delta = factual_delta
        self.factual_reward = factual_reward
        self.relevant_obs = relevant_obs
        self.relevant_action = relevant_action
        self.relevant_delta = relevant_delta
        self.relevant_reward = relevant_reward
        self.irrelevant_obs = irrelevant_obs
        self.corrupted_deltas = corrupted_deltas
        self.corrupted_indices = list(corrupted_indices)


def create_synthetic_tether_batch(
    batch_size: int = 32,
    ensemble_size: int = 5,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[TetherTransitionBatch, torch.Tensor]:
    """Generates synthetic 4-view batch and ensemble predictions for tests and training."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    factual_obs = torch.rand(batch_size, OBS_SIZE, generator=gen, device=device)
    factual_action = torch.randint(0, N_ACTIONS, (batch_size,), generator=gen, device=device)
    factual_delta = (torch.rand(batch_size, OBS_SIZE, generator=gen, device=device) - 0.5) * 0.2
    factual_reward = torch.rand(batch_size, generator=gen, device=device)

    # Relevant intervention: different action or vital deficit -> changed transition
    relevant_action = (factual_action + 1) % N_ACTIONS
    relevant_obs = factual_obs.clone()
    relevant_obs[:, 0] = (relevant_obs[:, 0] - 0.3).clamp(0.0, 1.0)
    relevant_delta = factual_delta * 1.5 + 0.05
    relevant_reward = factual_reward * 0.8 - 0.1

    # Irrelevant intervention: small sensor noise on background features (indices 8-15)
    irrelevant_obs = factual_obs.clone()
    noise = (torch.rand(batch_size, 8, generator=gen, device=device) - 0.5) * 0.02
    irrelevant_obs[:, 8:16] = (irrelevant_obs[:, 8:16] + noise).clamp(0.0, 1.0)

    # Base ensemble predictions around factual_delta
    # [K, batch, 16]
    base_deltas = []
    for k in range(ensemble_size):
        noise_k = (torch.rand(batch_size, OBS_SIZE, generator=gen, device=device) - 0.5) * 0.04
        base_deltas.append(factual_delta.unsqueeze(0) + noise_k.unsqueeze(0))
    clean_deltas = torch.cat(base_deltas, dim=0)

    # Corrupted deltas: inject wild errors into member 0 (and member 1)
    corrupted_deltas = clean_deltas.clone()
    corrupted_deltas[0] = corrupted_deltas[0] + 2.5
    corrupted_indices = [0]

    batch = TetherTransitionBatch(
        factual_obs=factual_obs,
        factual_action=factual_action,
        factual_delta=factual_delta,
        factual_reward=factual_reward,
        relevant_obs=relevant_obs,
        relevant_action=relevant_action,
        relevant_delta=relevant_delta,
        relevant_reward=relevant_reward,
        irrelevant_obs=irrelevant_obs,
        corrupted_deltas=corrupted_deltas,
        corrupted_indices=corrupted_indices,
    )
    return batch, clean_deltas


def tether_training_step(
    gate: TetherReliabilityGate,
    batch: TetherTransitionBatch,
    member_deltas: torch.Tensor,
    arm: str = "tether",
    effect_weight: float = 1.0,
    invariance_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute differentiable loss for any of the 7 Tether arms."""
    if arm not in TETHER_ARMS:
        raise ValueError(f"Unknown arm '{arm}'. Must be one of {TETHER_ARMS}")

    # 1. View 0: Factual
    res_fact = gate(batch.factual_obs, batch.factual_action, member_deltas)
    loss_factual = F.mse_loss(res_fact["gated_delta"], batch.factual_delta)

    # 2. View 1: Relevant (Co-Variation Effect)
    # Simulate ensemble tracking the relevant intervention
    relevant_member_deltas = member_deltas + (batch.relevant_delta - batch.factual_delta).unsqueeze(0)
    res_rel = gate(batch.relevant_obs, batch.relevant_action, relevant_member_deltas)
    loss_rel_mse = F.mse_loss(res_rel["gated_delta"], batch.relevant_delta)
    # Stale prediction repel loss
    stale_dist = (res_rel["gated_delta"] - batch.factual_delta).square().mean()
    loss_repel = (-stale_dist).exp()
    loss_effect = 0.5 * (loss_rel_mse + loss_repel)

    # Unpaired effect control: row-rolled target
    loss_unpaired_effect = 0.5 * (F.mse_loss(res_rel["gated_delta"], batch.relevant_delta.roll(1, dims=0)) + loss_repel)

    # 3. View 2: Irrelevant Invariance
    res_irrel = gate(batch.irrelevant_obs, batch.factual_action, member_deltas)
    loss_irrel_inv = F.mse_loss(res_irrel["gated_delta"], res_fact["gated_delta"])

    # 4. View 3: Damaged Invariance & Corrupted Weight Suppression
    res_dmg = gate(batch.factual_obs, batch.factual_action, batch.corrupted_deltas)
    loss_dmg_inv = F.mse_loss(res_dmg["gated_delta"], res_fact["gated_delta"])
    corrupted_w = res_dmg["weights"][:, batch.corrupted_indices]
    loss_dmg_suppress = corrupted_w.square().mean()
    loss_invariance = 0.5 * (loss_irrel_inv + loss_dmg_inv) + loss_dmg_suppress

    # Arm routing
    if arm == "passive":
        loss = loss_factual
    elif arm == "dropout":
        # Randomly mask 1 member posterior to uniform
        masked_deltas = member_deltas.clone()
        masked_deltas[0] = masked_deltas.mean(dim=0, keepdim=True)
        res_drop = gate(batch.factual_obs, batch.factual_action, masked_deltas)
        loss = F.mse_loss(res_drop["gated_delta"], batch.factual_delta)
    elif arm == "iit":
        loss = 0.25 * (loss_factual + loss_rel_mse + loss_irrel_inv + loss_dmg_inv)
    elif arm == "effect_only":
        loss = loss_factual + effect_weight * loss_effect
    elif arm == "invariance_only":
        loss = loss_factual + invariance_weight * loss_invariance
    elif arm == "tether":
        loss = loss_factual + effect_weight * loss_effect + invariance_weight * loss_invariance
    elif arm == "unpaired":
        loss = loss_factual + effect_weight * loss_unpaired_effect + invariance_weight * loss_invariance

    diagnostics = {
        "loss": float(loss.item()),
        "factual_mse": float(loss_factual.item()),
        "effect_loss": float(loss_effect.item()),
        "invariance_loss": float(loss_invariance.item()),
        "corrupted_suppression": float(loss_dmg_suppress.item()),
        "mean_trust": float(res_fact["epistemic_trust"].mean().item()),
    }
    return loss, diagnostics


def evaluate_gate_resilience(
    gate: TetherReliabilityGate,
    batch: TetherTransitionBatch,
    clean_deltas: torch.Tensor,
) -> Dict[str, Any]:
    """Evaluates zero-shot accuracy, corruption isolation, and epistemic trust."""
    gate.eval()
    with torch.no_grad():
        # 1. Clean evaluation
        clean_out = gate(batch.factual_obs, batch.factual_action, clean_deltas)
        clean_mse = float(F.mse_loss(clean_out["gated_delta"], batch.factual_delta).item())
        clean_trust = float(clean_out["epistemic_trust"].mean().item())

        # 2. Corrupted evaluation (1/5 corrupted)
        dmg_out = gate(batch.factual_obs, batch.factual_action, batch.corrupted_deltas)
        dmg_mse = float(F.mse_loss(dmg_out["gated_delta"], batch.factual_delta).item())
        corrupted_weights = float(dmg_out["weights"][:, batch.corrupted_indices].mean().item())

        # 3. Two-corrupted evaluation (2/5 corrupted)
        two_corrupted_deltas = clean_deltas.clone()
        two_corrupted_deltas[0] = two_corrupted_deltas[0] + 3.0
        two_corrupted_deltas[1] = two_corrupted_deltas[1] - 4.0
        two_dmg_out = gate(batch.factual_obs, batch.factual_action, two_corrupted_deltas)
        two_dmg_mse = float(F.mse_loss(two_dmg_out["gated_delta"], batch.factual_delta).item())
        two_corrupted_weights = float(two_dmg_out["weights"][:, [0, 1]].mean().item())

        # 4. Total spoofing (all corrupted) -> Must abstain!
        total_spoofed_deltas = clean_deltas + 10.0
        spoofed_out = gate(batch.factual_obs, batch.factual_action, total_spoofed_deltas)
        abstain_rate = float(spoofed_out["abstain"].float().mean().item())

        return {
            "clean_mse": clean_mse,
            "clean_trust": clean_trust,
            "one_corrupted_mse": dmg_mse,
            "one_corrupted_member_weight": corrupted_weights,
            "two_corrupted_mse": two_dmg_mse,
            "two_corrupted_member_weight": two_corrupted_weights,
            "total_spoof_abstain_rate": abstain_rate,
            "resilience_ratio": clean_mse / max(1e-8, dmg_mse),
            "zero_shot_resilience_passed": corrupted_weights < 0.15,
        }
