"""Multi-Emotion Continuous Affective State and Dynamic Interaction Engine.

Implements the 32-dimensional continuous affective representation:
e_t in [0, 1]^32 and 6-dimensional interpretable core a_t = [v_t, alpha_t, delta_t, nu_t, sigma_t, eta_t].

Multi-affective synergy and competition:
e_{t+1} = clamp(f_theta(e_t, h_t^A, x_t) + Gamma e_t, 0, 1)
Dual timescale: e_t = e_t^fast + e_t^slow
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple


AFFECTIVE_DIMENSIONS: List[str] = [
    "fear",
    "anxiety",
    "dread",
    "calmness",
    "happiness",
    "joy",
    "positive_excitement",
    "laughter_social",
    "love_attachment",
    "affiliation",
    "trust",
    "pride",
    "envy",
    "jealousy",
    "greed",
    "indulgence",
    "pleasure_seeking",
    "frustration",
    "anger",
    "hate_aversive",
    "sadness",
    "low_mood",
    "surprise",
    "curiosity",
    "contentment",
    "fatigue",
    "restfulness",
    "social_rejection",
    "relief",
    "anticipation",
    "boredom",
    "stress_arousal",
]


class AffectiveStepOutput(dict):
    """Output of MixedAffectiveState.step().
    Acts as a dictionary with keys ['e_total', 'emotions', 'core', 'diagnostics'],
    and also supports tuple unpacking: (e_total, core, diagnostics)."""
    def __init__(
        self,
        e_total: torch.Tensor,
        core: torch.Tensor,
        diagnostics: Dict[str, Any],
        emotions: Dict[str, float],
    ):
        super().__init__({
            "e_total": e_total,
            "core": core,
            "diagnostics": diagnostics,
            "emotions": emotions,
        })
        self.e_total = e_total
        self.core = core
        self.diagnostics = diagnostics
        self.emotions = emotions

    def __iter__(self):
        yield self.e_total
        yield self.core
        yield self.diagnostics


class MixedAffectiveState(nn.Module):
    """Computes, maintains, and evolves the multi-dimensional affective state."""

    def __init__(self, hidden_dim: int = 256, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.num_emotions = len(AFFECTIVE_DIMENSIONS)  # 32
        self.hidden_dim = hidden_dim

        # Projection from affective connectome hidden state h_t^A to 32 emotions
        self.proj_emotions = nn.Linear(hidden_dim, self.num_emotions)
        nn.init.orthogonal_(self.proj_emotions.weight, gain=0.5)
        nn.init.zeros_(self.proj_emotions.bias)

        # Core 6-dim appraisal projection: [valence, arousal, dominance, novelty, social, deficit]
        self.proj_core = nn.Linear(hidden_dim, 6)
        nn.init.orthogonal_(self.proj_core.weight, gain=0.5)
        nn.init.zeros_(self.proj_core.bias)

        # Bounded interaction matrix Gamma for synergies/antagonisms (32 x 32)
        # e.g., fear excites anxiety, calm dampens fear, curiosity opposes boredom
        gamma_init = torch.zeros(self.num_emotions, self.num_emotions)
        # Design priors for interaction matrix
        idx = {name: i for i, name in enumerate(AFFECTIVE_DIMENSIONS)}
        # Synergies
        gamma_init[idx["anxiety"], idx["fear"]] = 0.25
        gamma_init[idx["dread"], idx["anxiety"]] = 0.20
        gamma_init[idx["joy"], idx["happiness"]] = 0.30
        gamma_init[idx["affiliation"], idx["love_attachment"]] = 0.25
        gamma_init[idx["frustration"], idx["anger"]] = 0.25
        # Oppositions (negative feedback)
        gamma_init[idx["calmness"], idx["stress_arousal"]] = -0.35
        gamma_init[idx["stress_arousal"], idx["calmness"]] = -0.35
        gamma_init[idx["relief"], idx["dread"]] = -0.30
        gamma_init[idx["contentment"], idx["greed"]] = -0.20
        gamma_init[idx["curiosity"], idx["boredom"]] = -0.40

        self.gamma_matrix = nn.Parameter(gamma_init)

        # Dual-timescale state buffers
        self.register_buffer("e_fast", torch.zeros(self.num_emotions))
        self.register_buffer("e_slow", torch.zeros(self.num_emotions))
        self.register_buffer("core_state", torch.zeros(6))  # [v, alpha, delta, nu, sigma, eta]

        # Fast decay ~ 0.3, Slow decay ~ 0.05
        self.decay_fast = 0.35
        self.decay_slow = 0.05

    def reset_state(self) -> None:
        """Reset internal affective state buffers."""
        self.e_fast.zero_()
        self.e_slow.zero_()
        self.core_state.zero_()

    def get_state_dict(self) -> Dict[str, Any]:
        """Return human-interpretable snapshot of current affective dynamics."""
        vals = torch.clamp(0.6 * self.e_fast + 0.4 * self.e_slow, 0.0, 1.0).detach().cpu().numpy()
        emotions = {AFFECTIVE_DIMENSIONS[i]: round(float(vals[i]), 4) for i in range(self.num_emotions)}
        core = [round(float(v), 4) for v in self.core_state.detach().cpu().numpy().tolist()]
        return {
            "emotions": emotions,
            "core": core,
            "e_fast": self.e_fast.detach().cpu().numpy().tolist(),
            "e_slow": self.e_slow.detach().cpu().numpy().tolist(),
        }

    def step(
        self,
        h_affective: Optional[torch.Tensor] = None,
        sensory_context: Optional[torch.Tensor] = None,
        dt: float = 1.0,
        wpac_hidden: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> AffectiveStepOutput:
        """
        Advance affective state given current WPAC activation h_affective or wpac_hidden.

        Args:
            h_affective: [hidden_dim] activation of affective connectome
            sensory_context: Optional contextual modulation tensor
            dt: Timestep scaling
            wpac_hidden: Optional alias for h_affective

        Returns:
            AffectiveStepOutput: Dict-like object with ['e_total', 'emotions', 'core', 'diagnostics']
                                 and supports unpacking as (e_total, core, diagnostics)
        """
        if h_affective is None and wpac_hidden is not None:
            h_affective = wpac_hidden
        elif h_affective is None:
            h_affective = torch.zeros(self.hidden_dim)

        if isinstance(h_affective, (tuple, list)):
            h_affective = h_affective[0]

        h = h_affective.flatten()
        if len(h) < self.hidden_dim:
            h = F.pad(h, (0, self.hidden_dim - len(h)))
        elif len(h) > self.hidden_dim:
            h = h[:self.hidden_dim]

        # Instantaneous drive from WPAC
        raw_e = torch.sigmoid(self.proj_emotions(h))
        raw_core = self.proj_core(h)

        # Bounded interaction matrix: bounded spectral radius
        bounded_gamma = torch.clamp(self.gamma_matrix, -0.5, 0.5)

        # Fast and slow timescale dynamics
        curr_total = torch.clamp(0.6 * self.e_fast + 0.4 * self.e_slow, 0.0, 1.0)
        interaction_term = torch.matmul(bounded_gamma, curr_total)

        # e_fast responds rapidly to instantaneous input + interactions
        alpha_fast = np.clip(self.decay_fast * dt, 0.0, 1.0)
        self.e_fast.data.copy_(
            (1.0 - alpha_fast) * self.e_fast + alpha_fast * torch.clamp(raw_e + interaction_term, 0.0, 1.0)
        )

        # e_slow accumulates sustained emotional trends
        alpha_slow = np.clip(self.decay_slow * dt, 0.0, 1.0)
        self.e_slow.data.copy_(
            (1.0 - alpha_slow) * self.e_slow + alpha_slow * self.e_fast
        )

        e_total = torch.clamp(0.6 * self.e_fast + 0.4 * self.e_slow, 0.0, 1.0)

        # Core calculation:
        # Valence [-1, 1], Arousal [0, 1], Dominance [0, 1], Novelty [0, 1], Social [0, 1], Deficit [0, 1]
        v = torch.tanh(raw_core[0])
        alpha = torch.sigmoid(raw_core[1])
        delta = torch.sigmoid(raw_core[2])
        nu = torch.sigmoid(raw_core[3])
        sigma = torch.sigmoid(raw_core[4])
        eta = torch.sigmoid(raw_core[5])
        core = torch.stack([v, alpha, delta, nu, sigma, eta])
        self.core_state.data.copy_(core)

        # Extract active emotions for telemetry
        vals = e_total.detach().cpu().numpy()
        top_indices = np.argsort(-vals)[:5]
        top_emotions = {AFFECTIVE_DIMENSIONS[i]: round(float(vals[i]), 4) for i in top_indices}
        all_emotions = {AFFECTIVE_DIMENSIONS[i]: round(float(vals[i]), 4) for i in range(self.num_emotions)}

        diagnostics = {
            "top_emotions": top_emotions,
            "valence": round(float(v.item()), 4),
            "arousal": round(float(alpha.item()), 4),
            "dominance": round(float(delta.item()), 4),
            "novelty": round(float(nu.item()), 4),
            "social_salience": round(float(sigma.item()), 4),
            "interoceptive_deficit": round(float(eta.item()), 4),
            "mean_affective_intensity": round(float(e_total.mean().item()), 4),
        }

        return AffectiveStepOutput(e_total, core, diagnostics, all_emotions)

    def export_state(self) -> Dict[str, Any]:
        return {
            "gamma_matrix": self.gamma_matrix.detach().cpu().numpy().tolist(),
            "proj_emotions_weight": self.proj_emotions.weight.detach().cpu().numpy().tolist(),
            "proj_emotions_bias": self.proj_emotions.bias.detach().cpu().numpy().tolist(),
            "proj_core_weight": self.proj_core.weight.detach().cpu().numpy().tolist(),
            "proj_core_bias": self.proj_core.bias.detach().cpu().numpy().tolist(),
            "e_fast": self.e_fast.detach().cpu().numpy().tolist(),
            "e_slow": self.e_slow.detach().cpu().numpy().tolist(),
            "core_state": self.core_state.detach().cpu().numpy().tolist(),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if "gamma_matrix" in state:
            t = torch.tensor(state["gamma_matrix"], dtype=torch.float32)
            if t.shape == self.gamma_matrix.shape:
                self.gamma_matrix.data.copy_(t)
        if "proj_emotions_weight" in state:
            t = torch.tensor(state["proj_emotions_weight"], dtype=torch.float32)
            if t.shape == self.proj_emotions.weight.shape:
                self.proj_emotions.weight.data.copy_(t)
        if "proj_emotions_bias" in state:
            t = torch.tensor(state["proj_emotions_bias"], dtype=torch.float32)
            if t.shape == self.proj_emotions.bias.shape:
                self.proj_emotions.bias.data.copy_(t)
        if "proj_core_weight" in state:
            t = torch.tensor(state["proj_core_weight"], dtype=torch.float32)
            if t.shape == self.proj_core.weight.shape:
                self.proj_core.weight.data.copy_(t)
        if "proj_core_bias" in state:
            t = torch.tensor(state["proj_core_bias"], dtype=torch.float32)
            if t.shape == self.proj_core.bias.shape:
                self.proj_core.bias.data.copy_(t)
        if "e_fast" in state:
            t = torch.tensor(state["e_fast"], dtype=torch.float32)
            if t.shape == self.e_fast.shape:
                self.e_fast.data.copy_(t)
        if "e_slow" in state:
            t = torch.tensor(state["e_slow"], dtype=torch.float32)
            if t.shape == self.e_slow.shape:
                self.e_slow.data.copy_(t)
        if "core_state" in state:
            t = torch.tensor(state["core_state"], dtype=torch.float32)
            if t.shape == self.core_state.shape:
                self.core_state.data.copy_(t)
