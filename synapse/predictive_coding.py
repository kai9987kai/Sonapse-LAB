"""Hierarchical Active Inference & Predictive Coding Connectome.

Implements Karl Friston's Free Energy Principle (FEP) across the developmental
connectome:
1. Top-Down Generative Priors: Derived from slow Tidal memory representations.
2. Precision-Weighted Prediction Errors: Dynamic sensory discrepancies weighted
   by cholinergic attention (ACh).
3. Variational Free Energy (VFE) & Surprisal minimization.
4. Expected Free Energy (EFE) Action Biasing: Epistemic (information gain) vs
   pragmatic (resource target fulfillment) value.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class PredictiveCodingState:
    free_energy: float = 0.0
    sensory_surprisal: float = 0.0
    epistemic_value: float = 0.0
    pragmatic_value: float = 0.0
    mean_precision: float = 1.0

    def to_dict(self) -> dict:
        return {
            'free_energy': round(float(self.free_energy), 5),
            'sensory_surprisal': round(float(self.sensory_surprisal), 5),
            'epistemic_value': round(float(self.epistemic_value), 5),
            'pragmatic_value': round(float(self.pragmatic_value), 5),
            'mean_precision': round(float(self.mean_precision), 5),
        }

class PredictiveCodingEngine(nn.Module):
    """Hierarchical predictive coding and active inference engine.

    Args:
        obs_dim: Dimension of sensory observation vector (default 155).
        latent_dim: Dimension of Poseidon Tidal latent space (default 128).
        pred_dim: Dimension of primary sensory prediction subspace (default 33, depth rays).
    """
    def __init__(self, obs_dim: int = 155, latent_dim: int = 128, pred_dim: int = 33):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.pred_dim = pred_dim

        # Generative top-down prediction model: latent memory -> predicted sensory rays
        self.generative_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, pred_dim, bias=True)
        )
        # Log-precision projection: modulates sensory error confidence by acetylcholine
        self.precision_head = nn.Linear(latent_dim, pred_dim, bias=False)
        nn.init.zeros_(self.precision_head.weight)

        self.state = PredictiveCodingState()

    def reset(self):
        self.state = PredictiveCodingState()

    def export_state(self) -> dict:
        return {
            'free_energy': float(self.state.free_energy),
            'sensory_surprisal': float(self.state.sensory_surprisal),
            'epistemic_value': float(self.state.epistemic_value),
            'pragmatic_value': float(self.state.pragmatic_value),
            'mean_precision': float(self.state.mean_precision),
        }

    def import_state(self, state: dict):
        if not state:
            return
        self.state.free_energy = float(state.get('free_energy', 0.0))
        self.state.sensory_surprisal = float(state.get('sensory_surprisal', 0.0))
        self.state.epistemic_value = float(state.get('epistemic_value', 0.0))
        self.state.pragmatic_value = float(state.get('pragmatic_value', 0.0))
        self.state.mean_precision = float(state.get('mean_precision', 1.0))

    def step(
        self,
        obs: torch.Tensor,
        memory: torch.Tensor,
        acetylcholine: float = 0.5,
        target_resource: float = 1.0
    ) -> dict:
        """Compute top-down sensory prediction, prediction error, and Variational Free Energy.

        Args:
            obs: Current observation vector [155].
            memory: Tidal latent memory [1, 128] or [128].
            acetylcholine: Dynamic cholinergic attention level [0..1].
            target_resource: Target food/resource distance expectation.

        Returns:
            Dictionary containing prediction errors, reflex action bias, and free energy.
        """
        mem = memory.flatten()
        if mem.shape[0] != self.latent_dim:
            mem = F.pad(mem, (0, max(0, self.latent_dim - mem.shape[0])))[:self.latent_dim]

        # 1. Top-down sensory prediction g(mu)
        sensory_actual = obs[:self.pred_dim]
        pred_rays = torch.sigmoid(self.generative_predictor(mem))

        # 2. Prediction error: eps = actual - predicted
        eps = sensory_actual - pred_rays

        # 3. Precision weighting: Pi = exp(log_prec + ACh gain)
        log_prec = self.precision_head(mem) + (acetylcholine - 0.5) * 2.0
        precision = torch.exp(torch.clamp(log_prec, -2.0, 3.0))

        # 4. Variational Free Energy (VFE): F = 0.5 * eps^T * Pi * eps - 0.5 * sum(ln(Pi))
        weighted_sq_err = (precision * (eps ** 2)).sum()
        complexity = -0.5 * torch.log(precision + 1e-6).sum()
        vfe = 0.5 * weighted_sq_err + complexity
        sensory_surprisal = 0.5 * (eps ** 2).mean()

        # 5. Expected Free Energy (EFE) decomposition:
        # Epistemic value: expected information gain (higher variance in prediction errors)
        epistemic = float(eps.var().item()) if len(eps) > 1 else 0.0
        # Pragmatic value: negative discrepancy from desired target state
        pragmatic = -float(abs(float(sensory_actual.max().item()) - target_resource))

        # 6. Active inference motor bias (reflex arc): motor actions that cancel prediction errors
        # Directional gradient across left vs right FOV rays
        left_error = float(eps[:self.pred_dim // 2].mean().item())
        right_error = float(eps[self.pred_dim // 2:].mean().item())
        steer_bias = float(np.clip((right_error - left_error) * 0.5, -0.2, 0.2))

        self.state.free_energy = float(vfe.item())
        self.state.sensory_surprisal = float(sensory_surprisal.item())
        self.state.epistemic_value = epistemic
        self.state.pragmatic_value = pragmatic
        self.state.mean_precision = float(precision.mean().item())

        return {
            'free_energy': self.state.free_energy,
            'sensory_surprisal': self.state.sensory_surprisal,
            'prediction_error_norm': float(eps.norm().item()),
            'mean_precision': self.state.mean_precision,
            'steer_bias': steer_bias,
            'epistemic_value': epistemic,
            'pragmatic_value': pragmatic,
            'scope': 'hierarchical predictive coding and variational free energy'
        }
