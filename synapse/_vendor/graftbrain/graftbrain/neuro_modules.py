"""Biological neuro-computational modules inspired by modern Drosophila connectomics.

Includes:
1. **CentralComplexCompass**: Ring attractor model of the Ellipsoid Body (E-PG / PB / Δ7 neurons)
   maintaining an internal heading direction bump updated via angular velocity and visual contrast.
2. **MushroomBody**: Sparse Kenyon Cell (KC) representation with compartmentalized
   dopaminergic neuron (DAN) plasticity modulating Mushroom Body Output Neurons (MBONs).
3. **CuriosityRND**: Intrinsic exploration bonus via Random Network Distillation to drive
   autonomous exploration in sparse-reward environments (mazes, manipulation).
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralComplexCompass(nn.Module):
    """Ring attractor representing the Ellipsoid Body (EB) / Protocerebral Bridge (PB) compass.

    Maintains a 16-wedge head direction representation on S^1.
    Integrates rotational proprioception (ω) and visually anchors to the highest-contrast retina azimuth.
    """

    def __init__(self, n_wedges: int = 16, sigma: float = 0.45):
        super().__init__()
        self.n_wedges = n_wedges
        self.sigma = sigma
        self.wedges = torch.linspace(-math.pi, math.pi, n_wedges + 1)[:-1]
        self.register_buffer("wedge_angles", self.wedges)
        self.bump_angle = 0.0

    def step(self, angular_velocity: float, retina_rays: np.ndarray | None = None, dt: float = 0.1) -> torch.Tensor:
        """Update bump position and return 16-dimensional compass rate vector."""
        # 1. Path integration from rotational velocity
        self.bump_angle = (self.bump_angle + float(angular_velocity) * dt + math.pi) % (2 * math.pi) - math.pi

        # 2. Visual landmark anchoring: pull bump toward prominent visual cue if available
        if retina_rays is not None and len(retina_rays) >= 33:
            # First 33 features are depth; features 33:66, 66:99, 99:132 are RGB
            depth = retina_rays[:33]
            # Saliency: nearest objects or high contrast
            saliency = depth
            if np.max(saliency) > 0.3:
                sal_idx = int(np.argmax(saliency))
                # Map ray index (0..32) across FOV [-60 deg, +60 deg]
                ray_angle = -math.pi / 3 + (sal_idx / 32.0) * (2 * math.pi / 3)
                # Compute visual angular discrepancy
                diff = (ray_angle - self.bump_angle + math.pi) % (2 * math.pi) - math.pi
                # Gentle visual anchoring pull (gain = 0.1)
                self.bump_angle = (self.bump_angle + 0.1 * diff + math.pi) % (2 * math.pi) - math.pi

        # 3. Compute continuous Gaussian bump across the 16 ring wedges
        diffs = (self.wedge_angles - self.bump_angle + math.pi) % (2 * math.pi) - math.pi
        bump = torch.exp(- (diffs ** 2) / (2 * self.sigma ** 2))
        return bump / (bump.sum() + 1e-6)

    def reset(self, initial_heading: float = 0.0):
        self.bump_angle = float(initial_heading)


class MushroomBody(nn.Module):
    """Mushroom Body (MB) associative learning module.

    - Kenyon Cells (KCs): high-dimensional, sparse top-k expansion of sensory cues.
    - Dopaminergic Neurons (DANs): encode positive reward vs negative penalty.
    - Mushroom Body Output Neurons (MBONs): read out approach vs avoidance valence.
    """

    def __init__(self, in_dim: int, n_kc: int = 64, k_sparse: int = 4, n_mbon: int = 2):
        super().__init__()
        self.n_kc = n_kc
        self.k_sparse = k_sparse
        self.n_mbon = n_mbon

        # Fixed, pseudo-random sensory claw expansion matrix (KC claws)
        self.register_buffer("w_claw", torch.randn(n_kc, in_dim) / math.sqrt(in_dim))
        # Plastic KC -> MBON synapses (initialized to small neutral weights)
        self.register_buffer("w_mbon", torch.zeros(n_mbon, n_kc))
        self.register_buffer("kc_act", torch.zeros(n_kc))
        self.register_buffer("elig", torch.zeros(n_mbon, n_kc))

    def forward(self, cue_vec: torch.Tensor, adapt=True) -> torch.Tensor:
        """Encode sensory cue, apply top-k Winner-Take-All sparse gating, and compute MBON output."""
        # 1. Linear claw projection
        raw = F.linear(cue_vec, self.w_claw)
        # 2. Top-k Winner-Take-All sparsity (~6% active KCs, matching fly biology)
        topk_vals, topk_idx = torch.topk(raw, self.k_sparse)
        kc = torch.zeros_like(raw)
        kc.scatter_(0, topk_idx, F.relu(topk_vals))
        if adapt:
            self.kc_act.copy_(kc.detach())

        # 3. Readout via plastic KC -> MBON weights: MBON[0] = approach, MBON[1] = avoid
        mbon_out = torch.mv(self.w_mbon, kc)

        # 4. Update synaptic eligibility trace for 3-factor neuromodulation
        # eligibility = post * pre
        if adapt:
            self.elig.mul_(0.85).add_(torch.outer(torch.tanh(mbon_out).detach(), kc.detach()))
        return mbon_out

    @torch.no_grad()
    def dopaminergic_update(self, reward: float, eta: float = 0.02):
        """Dopamine release modulates KC->MBON synapses based on reward/punishment."""
        # Reward DAN reinforces approach (MBON 0), Punishment DAN reinforces avoidance (MBON 1)
        # Weight update: ΔW = η * [DAN_reward, DAN_punish] ⊗ Eligibility
        dan_signal = torch.tensor([float(reward), -float(reward)], dtype=torch.float32)
        delta_w = eta * (dan_signal.unsqueeze(1) * self.elig)
        self.w_mbon.add_(delta_w).clamp_(-1.0, 1.0)


class CuriosityRND(nn.Module):
    """Lightweight Random Network Distillation (RND) for intrinsic curiosity."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64, feat_dim: int = 16):
        super().__init__()
        # Fixed random target network
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )
        for p in self.target.parameters():
            p.requires_grad = False

        # Predictor network trained to match target
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)

    def compute_bonus(self, obs: torch.Tensor, train: bool = True) -> float:
        """Compute intrinsic prediction error reward, and optionally train the predictor."""
        with torch.no_grad():
            target_feat = self.target(obs)
        pred_feat = self.predictor(obs)
        error = F.mse_loss(pred_feat, target_feat)

        if train:
            self.opt.zero_grad()
            error.backward()
            self.opt.step()

        # Scale bonus to healthy exploration range [0, 0.2]
        return float(torch.clamp(error.detach() * 0.5, 0.0, 0.2))
