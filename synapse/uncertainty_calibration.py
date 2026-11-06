"""Tripartite Uncertainty Decomposition and Calibration.

Grounding & Review Requirements:
Separates three distinct uncertainty components in model-based dynamics:
1. Epistemic Uncertainty: Model disagreement across an ensemble of latent transition/reward predictors.
2. Aleatoric Uncertainty: Irreducible data/environmental noise estimated via heteroscedastic Gaussian NLL.
3. Out-of-Distribution (OOD) Distance: Latent distance from empirical training support (Mahalanobis / distance metric).

Also provides:
- Expected Calibration Error (ECE) on held-out transitions.
- Risk-vs-Coverage evaluation (selective prediction monotonicity).
- Temperature / conformal scaling calibration.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TripartiteUncertaintyEstimate:
    """Explicitly decomposed uncertainty estimate."""
    epistemic: float          # Model disagreement / variance across ensemble
    aleatoric: float          # Irreducible environmental noise (heteroscedastic variance)
    ood_score: float          # Out-of-distribution distance from empirical support [0, inf)
    composite_penalty: float  # Calibrated weighted combination for planning cost
    confidence: float         # Normalised confidence in [0, 1]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'epistemic': self.epistemic,
            'aleatoric': self.aleatoric,
            'ood_score': self.ood_score,
            'composite_penalty': self.composite_penalty,
            'confidence': self.confidence,
            'details': self.details
        }


class AleatoricUncertaintyHead(nn.Module):
    """Predicts predictive mean and heteroscedastic log-variance for reward/transition targets.
    
    Trained via Gaussian Negative Log-Likelihood (NLL):
        NLL(y; mu, s) = 0.5 * exp(-s) * (y - mu)^2 + 0.5 * s + 0.5 * log(2*pi)
    where s = log(sigma^2).
    """

    def __init__(self, in_dim: int, out_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim * 2)
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, log_variance)."""
        out = self.net(x)
        mu = out[..., :self.out_dim]
        # Clamp log_variance for numerical stability
        log_var = torch.clamp(out[..., self.out_dim:], -6.0, 4.0)
        return mu, log_var

    @staticmethod
    def gaussian_nll_loss(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Heteroscedastic Gaussian NLL loss."""
        inv_var = torch.exp(-log_var)
        loss = 0.5 * inv_var * ((target - mu) ** 2) + 0.5 * log_var + 0.5 * math.log(2 * math.pi)
        return loss.mean()


class LatentDynamicsEnsemble(nn.Module):
    """Ensemble of recurrent latent dynamics heads for calibrated epistemic uncertainty."""

    def __init__(self, state_dim: int = 32, action_dim: int = 5, ensemble_size: int = 4, seed: int = 42):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        torch.manual_seed(seed)
        # Transition heads (one per member)
        self.transitions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, state_dim)
            ) for _ in range(ensemble_size)
        ])
        # Reward & heteroscedastic aleatoric heads
        self.reward_heads = nn.ModuleList([
            AleatoricUncertaintyHead(state_dim + action_dim, out_dim=1, hidden_dim=32)
            for _ in range(ensemble_size)
        ])

    def forward_all(
        self,
        z: torch.Tensor,
        a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward all ensemble members.
        
        Returns:
            all_z_next: [ensemble_size, batch_size, state_dim]
            all_r_mu: [ensemble_size, batch_size, 1]
            all_r_log_var: [ensemble_size, batch_size, 1]
        """
        inp = torch.cat([z, a], dim=-1)
        z_next_list = []
        r_mu_list = []
        r_log_var_list = []

        for trans, rew_head in zip(self.transitions, self.reward_heads):
            dz = trans(inp)
            z_next = z + dz  # Residual dynamics
            mu, log_var = rew_head(inp)
            z_next_list.append(z_next)
            r_mu_list.append(mu)
            r_log_var_list.append(log_var)

        return (
            torch.stack(z_next_list, dim=0),
            torch.stack(r_mu_list, dim=0),
            torch.stack(r_log_var_list, dim=0)
        )

    def estimate_uncertainty(
        self,
        z: torch.Tensor,
        a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute mean prediction, epistemic disagreement, and mean aleatoric variance.
        
        Returns:
            mean_reward: [batch_size]
            epistemic_var: [batch_size] (ensemble variance of predicted rewards)
            aleatoric_var: [batch_size] (mean predicted heteroscedastic variance)
        """
        with torch.no_grad():
            _, all_r_mu, all_r_log_var = self.forward_all(z, a)
            # all_r_mu: [M, B, 1]
            mean_r = all_r_mu.mean(dim=0).squeeze(-1)  # [B]
            # Epistemic variance across ensemble predictions: Var_m(r_m)
            epistemic_var = all_r_mu.var(dim=0, unbiased=False).squeeze(-1)  # [B]
            # Aleatoric variance: Mean_m(exp(log_var_m))
            aleatoric_var = torch.exp(all_r_log_var).mean(dim=0).squeeze(-1)  # [B]

        return mean_r, epistemic_var, aleatoric_var


class OutDistributionDetector:
    """Detects out-of-distribution (OOD) states via latent support modeling (running mean & covariance)."""

    def __init__(self, latent_dim: int = 32, max_stored: int = 2000, seed: int = 42):
        self.latent_dim = latent_dim
        self.max_stored = max_stored
        self.rng = np.random.default_rng(seed)
        self.storage: List[np.ndarray] = []
        self.mean: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self._fitted = False

    def update_support(self, z_batch: np.ndarray):
        """Append observed latent embeddings to empirical support."""
        if z_batch.ndim == 1:
            z_batch = z_batch[np.newaxis, :]
        for z in z_batch:
            if len(self.storage) < self.max_stored:
                self.storage.append(z.copy())
            else:
                idx = self.rng.integers(0, self.max_stored)
                self.storage[idx] = z.copy()
        if len(self.storage) >= max(10, self.latent_dim):
            self._fit()

    def _fit(self):
        data = np.stack(self.storage, axis=0)  # [N, D]
        self.mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        # Add small regularizer to ensure positive definiteness
        cov_reg = cov + np.eye(self.latent_dim) * 1e-4
        try:
            self.inv_cov = np.linalg.pinv(cov_reg)
            self._fitted = True
        except np.linalg.LinAlgError:
            self._fitted = False

    def compute_ood_score(self, z: np.ndarray) -> float:
        """Compute Mahalanobis distance from empirical latent support."""
        if not self._fitted or self.mean is None or self.inv_cov is None:
            return 0.0
        if z.ndim > 1:
            z = z.flatten()
        diff = z - self.mean
        dist_sq = float(diff @ self.inv_cov @ diff.T)
        # Return normalized distance
        return float(math.sqrt(max(0.0, dist_sq)))


class TripartiteUncertaintyDecomposer:
    """Integrated engine coordinating ensemble epistemic, heteroscedastic aleatoric, and OOD metrics."""

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 5,
        ensemble_size: int = 4,
        beta_epistemic: float = 1.0,
        beta_aleatoric: float = 0.5,
        beta_ood: float = 0.5,
        seed: int = 42
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.beta_epistemic = beta_epistemic
        self.beta_aleatoric = beta_aleatoric
        self.beta_ood = beta_ood

        self.ensemble = LatentDynamicsEnsemble(
            state_dim=state_dim,
            action_dim=action_dim,
            ensemble_size=ensemble_size,
            seed=seed
        )
        self.optimizer = torch.optim.Adam(self.ensemble.parameters(), lr=1e-3)
        self.ood_detector = OutDistributionDetector(latent_dim=state_dim, seed=seed)
        self.calibration_scale: float = 1.0

    def train_batch(
        self,
        z_batch: np.ndarray,
        act_batch: np.ndarray,
        rew_batch: np.ndarray,
        next_z_batch: np.ndarray
    ) -> Dict[str, float]:
        """Train ensemble with heteroscedastic NLL and transition loss."""
        z_t = torch.as_tensor(z_batch, dtype=torch.float32)
        act_t = torch.as_tensor(act_batch, dtype=torch.float32)
        rew_t = torch.as_tensor(rew_batch, dtype=torch.float32).unsqueeze(-1)
        next_z_t = torch.as_tensor(next_z_batch, dtype=torch.float32)

        self.ood_detector.update_support(z_batch)

        self.optimizer.zero_grad()
        all_z_next, all_r_mu, all_r_log_var = self.ensemble.forward_all(z_t, act_t)

        total_loss = torch.tensor(0.0)
        nll_losses = []
        trans_losses = []

        for m in range(self.ensemble_size):
            # Transition MSE
            loss_trans = F.mse_loss(all_z_next[m], next_z_t)
            # Gaussian NLL for reward
            loss_nll = AleatoricUncertaintyHead.gaussian_nll_loss(
                all_r_mu[m], all_r_log_var[m], rew_t
            )
            member_loss = loss_trans + 1.0 * loss_nll
            total_loss = total_loss + member_loss
            nll_losses.append(float(loss_nll.item()))
            trans_losses.append(float(loss_trans.item()))

        total_loss.backward()
        self.optimizer.step()

        return {
            'loss_total': float(total_loss.item()),
            'loss_nll_mean': float(np.mean(nll_losses)),
            'loss_trans_mean': float(np.mean(trans_losses))
        }

    def evaluate_uncertainty(
        self,
        z: np.ndarray,
        a: np.ndarray
    ) -> TripartiteUncertaintyEstimate:
        """Decompose uncertainty for a state-action pair."""
        z_t = torch.as_tensor(z, dtype=torch.float32).unsqueeze(0) if z.ndim == 1 else torch.as_tensor(z, dtype=torch.float32)
        act_t = torch.as_tensor(a, dtype=torch.float32).unsqueeze(0) if a.ndim == 1 else torch.as_tensor(a, dtype=torch.float32)

        mean_r, ep_var, al_var = self.ensemble.estimate_uncertainty(z_t, act_t)

        epistemic = float(ep_var[0].item())
        aleatoric = float(al_var[0].item())
        ood_score = self.ood_detector.compute_ood_score(z if z.ndim == 1 else z[0])

        # Composite penalty: calibrated combination
        composite = (
            self.beta_epistemic * math.sqrt(max(0.0, epistemic)) +
            self.beta_aleatoric * math.sqrt(max(0.0, aleatoric)) +
            self.beta_ood * ood_score
        ) * self.calibration_scale

        # Confidence bounded in [0, 1]
        confidence = float(1.0 / (1.0 + composite))

        return TripartiteUncertaintyEstimate(
            epistemic=epistemic,
            aleatoric=aleatoric,
            ood_score=ood_score,
            composite_penalty=composite,
            confidence=confidence,
            details={
                'predicted_mean_reward': float(mean_r[0].item()),
                'calibration_scale': self.calibration_scale
            }
        )

    def calibrate_on_held_out(
        self,
        z_val: np.ndarray,
        act_val: np.ndarray,
        rew_val: np.ndarray,
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """Evaluate calibration against held-out outcomes: ECE and risk-coverage curve.
        
        Returns:
            ece: Expected Calibration Error for 68% credible interval (1 sigma).
            risk_coverage_curve: List of (coverage, mse) points testing selective prediction.
            monotonic_risk_reduction: True if MSE decreases monotonically as uncertain predictions are discarded.
        """
        N = len(rew_val)
        if N < 5:
            return {'ece': 0.0, 'risk_coverage_curve': [], 'monotonic_risk_reduction': True}

        z_t = torch.as_tensor(z_val, dtype=torch.float32)
        act_t = torch.as_tensor(act_val, dtype=torch.float32)
        mean_r, ep_var, al_var = self.ensemble.estimate_uncertainty(z_t, act_t)

        preds = mean_r.cpu().numpy()
        total_std = np.sqrt(np.maximum(0.0, ep_var.cpu().numpy() + al_var.cpu().numpy()))
        actuals = rew_val.squeeze() if rew_val.ndim > 1 else rew_val

        errors = np.abs(preds - actuals)
        sq_errors = (preds - actuals) ** 2

        # 1. Expected Calibration Error (ECE) for 1-sigma intervals (expected coverage ~68.27%)
        expected_coverage = 0.6827
        inside_interval = errors <= total_std
        bin_size = max(1, N // num_bins)
        sorted_indices = np.argsort(total_std)

        bin_errors = []
        for b in range(0, N, bin_size):
            idx = sorted_indices[b:b + bin_size]
            if len(idx) == 0:
                continue
            empirical_coverage = np.mean(inside_interval[idx])
            bin_errors.append(abs(empirical_coverage - expected_coverage))
        ece = float(np.mean(bin_errors)) if bin_errors else 0.0

        # 2. Risk-Coverage Curve (Selective Prediction)
        # Sort samples by total uncertainty (ascending: most confident first)
        conf_order = np.argsort(total_std)
        coverages = [1.0, 0.8, 0.6, 0.4, 0.2]
        risk_coverage = []
        for cov in coverages:
            k = max(1, int(round(cov * N)))
            subset_idx = conf_order[:k]
            subset_mse = float(np.mean(sq_errors[subset_idx]))
            risk_coverage.append({'coverage': cov, 'mse': subset_mse})

        # Check monotonic risk reduction: MSE should decrease or remain flat as coverage decreases
        mses = [pt['mse'] for pt in risk_coverage]
        monotonic = all(mses[i] >= mses[i+1] - 1e-5 for i in range(len(mses) - 1))

        # Update calibration scale
        if ece > 0.0:
            empirical_std_ratio = np.mean(errors) / (np.mean(total_std) + 1e-6)
            self.calibration_scale = float(np.clip(empirical_std_ratio, 0.5, 2.5))

        return {
            'ece': ece,
            'expected_coverage': expected_coverage,
            'risk_coverage_curve': risk_coverage,
            'monotonic_risk_reduction': monotonic,
            'calibration_scale': self.calibration_scale
        }
