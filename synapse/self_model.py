"""Action-Grounded Efference-Copy Self-Model (GhostInTheMachine Comparator).

Grounding & Review Requirements:
Biological nervous systems distinguish self-generated sensory consequences (reafference)
from externally imposed changes (exafference) by comparing efference copies of motor
commands against incoming sensory reafference (von Holst & Mittelstaedt 1950, Wolpert 1995).

This module provides:
1. EfferenceCopyComparator:
   Predicts sensory delta Delta s_hat = M_efference(s_t, a_t).
2. Authorship & Disturbance Decomposition:
   - Self-Authorship Index A_t in [0, 1]: degree to which sensory delta was self-caused.
   - External Disturbance Index E_t in [0, inf): exafference / exogenous shock.
   - Actuator Slip / Fault Index F_t in [0, inf): motor intent without kinematic effect.
3. Rigorous Matched Controls:
   - 'active': genuine online efference copy prediction.
   - 'scrambled_timing': time-delayed motor commands (temporal decoherence control).
   - 'shuffled_action': permuted motor coordinates (contingency control).
   - 'ablated': null prediction model (M = 0 baseline).
"""
from __future__ import annotations
import math
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlMode(str, Enum):
    ACTIVE = 'active'
    SCRAMBLED_TIMING = 'scrambled_timing'
    SHUFFLED_ACTION = 'shuffled_action'
    ABLATED = 'ablated'


@dataclass
class AuthorshipReceipt:
    """Quantitative decomposition of sensory transition authorship."""
    step: int
    self_authorship: float     # A_t in [0, 1]
    external_disturbance: float # E_t >= 0
    actuator_fault: float       # F_t >= 0
    discrepancy_norm: float     # ||delta_t||_2
    predicted_delta_norm: float # ||Delta s_hat||_2
    actual_delta_norm: float    # ||Delta s||_2
    control_mode: str
    is_self_caused: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            'step': self.step,
            'self_authorship': self.self_authorship,
            'external_disturbance': self.external_disturbance,
            'actuator_fault': self.actuator_fault,
            'discrepancy_norm': self.discrepancy_norm,
            'predicted_delta_norm': self.predicted_delta_norm,
            'actual_delta_norm': self.actual_delta_norm,
            'control_mode': self.control_mode,
            'is_self_caused': self.is_self_caused
        }


class EfferenceForwardModel(nn.Module):
    """Predicts sensory consequences Delta s_hat from state s_t and efference copy a_t."""

    def __init__(self, sensory_dim: int = 155, action_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.sensory_dim = sensory_dim
        self.action_dim = action_dim

        # Compact sensory encoder (focuses on ego-motion: pos, vel, ray deltas)
        self.net = nn.Sequential(
            nn.Linear(sensory_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sensory_dim)
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([s, a], dim=-1)
        return self.net(inp)


class EfferenceCopyComparator:
    """Action-grounded comparator measuring reafference vs exafference."""

    def __init__(
        self,
        sensory_dim: int = 155,
        action_dim: int = 5,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        authorship_threshold: float = 0.65,
        control_mode: ControlMode = ControlMode.ACTIVE,
        seed: int = 42
    ):
        self.sensory_dim = sensory_dim
        self.action_dim = action_dim
        self.authorship_threshold = authorship_threshold
        self.control_mode = control_mode
        self.step_counter = 0

        self.rng = np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = EfferenceForwardModel(sensory_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Buffers for matched controls
        self.action_history: List[np.ndarray] = []
        self.recent_receipts: List[AuthorshipReceipt] = []

    def set_control_mode(self, mode: str | ControlMode):
        self.control_mode = ControlMode(mode)

    def process_transition(
        self,
        s_t: np.ndarray,
        a_t: np.ndarray,
        s_next: np.ndarray,
        learn: bool = True
    ) -> AuthorshipReceipt:
        """Compares actual sensory change with predicted reafference."""
        self.step_counter += 1
        s_t_flat = np.asarray(s_t, dtype=np.float32).flatten()
        s_next_flat = np.asarray(s_next, dtype=np.float32).flatten()
        a_t_flat = np.asarray(a_t, dtype=np.float32).flatten()

        actual_delta = s_next_flat - s_t_flat
        actual_norm = float(np.linalg.norm(actual_delta))

        # Store action for scrambled timing control
        self.action_history.append(a_t_flat.copy())
        if len(self.action_history) > 20:
            self.action_history.pop(0)

        # Apply matched control transformations
        eff_action = self._apply_control_transformation(a_t_flat)

        # Forward prediction
        if self.control_mode == ControlMode.ABLATED:
            pred_delta = np.zeros_like(actual_delta)
        else:
            with torch.no_grad():
                s_tensor = torch.as_tensor(s_t_flat, dtype=torch.float32).unsqueeze(0)
                a_tensor = torch.as_tensor(eff_action, dtype=torch.float32).unsqueeze(0)
                pred_delta = self.model(s_tensor, a_tensor).squeeze(0).cpu().numpy()

        pred_norm = float(np.linalg.norm(pred_delta))

        # Discrepancy: delta_t = actual - predicted
        discrepancy = actual_delta - pred_delta
        discrepancy_norm = float(np.linalg.norm(discrepancy))

        # Metric 1: Self-Authorship Index A_t in [0, 1]
        # High when discrepancy is low relative to expected sensory motion
        sigma_author = max(0.1, pred_norm)
        coherence = math.exp(- (discrepancy_norm ** 2) / (2.0 * (sigma_author ** 2) + 1e-6))
        # Scaled by whether motion actually happened
        motion_ratio = min(1.0, actual_norm / (pred_norm + 1e-5)) if pred_norm > 1e-5 else 1.0
        self_authorship = float(np.clip(coherence * motion_ratio, 0.0, 1.0))

        # Metric 2: External Disturbance Index E_t in [0, inf)
        # Unexplained orthogonal or sudden sensory displacement
        external_disturbance = float(max(0.0, discrepancy_norm - 0.2 * pred_norm))

        # Metric 3: Actuator Slip / Fault Index F_t in [0, inf)
        # Strong motor command delivered, but actual motion near zero
        motor_effort = float(np.linalg.norm(a_t_flat))
        slip = max(0.0, motor_effort - 2.0 * actual_norm) if motor_effort > 0.2 else 0.0
        actuator_fault = float(slip)

        is_self_caused = bool(self_authorship >= self.authorship_threshold)

        receipt = AuthorshipReceipt(
            step=self.step_counter,
            self_authorship=self_authorship,
            external_disturbance=external_disturbance,
            actuator_fault=actuator_fault,
            discrepancy_norm=discrepancy_norm,
            predicted_delta_norm=pred_norm,
            actual_delta_norm=actual_norm,
            control_mode=self.control_mode.value,
            is_self_caused=is_self_caused
        )

        self.recent_receipts.append(receipt)
        if len(self.recent_receipts) > 100:
            self.recent_receipts.pop(0)

        # Online self-supervised learning (only on active control)
        if learn and self.control_mode == ControlMode.ACTIVE:
            self._update_model(s_t_flat, a_t_flat, actual_delta)

        return receipt

    def _apply_control_transformation(self, a_t: np.ndarray) -> np.ndarray:
        """Implements matched scientific control baselines."""
        if self.control_mode == ControlMode.ACTIVE:
            return a_t
        elif self.control_mode == ControlMode.SCRAMBLED_TIMING:
            # Use action from 3 steps ago (temporal decorrelation)
            idx = max(0, len(self.action_history) - 4)
            return self.action_history[idx]
        elif self.control_mode == ControlMode.SHUFFLED_ACTION:
            # Random permutation of action coordinates
            perm = self.rng.permutation(len(a_t))
            return a_t[perm]
        elif self.control_mode == ControlMode.ABLATED:
            return np.zeros_like(a_t)
        return a_t

    def _update_model(self, s_t: np.ndarray, a_t: np.ndarray, delta_target: np.ndarray):
        """Single online gradient update for the efference forward model."""
        self.optimizer.zero_grad()
        s_tensor = torch.as_tensor(s_t, dtype=torch.float32).unsqueeze(0)
        a_tensor = torch.as_tensor(a_t, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.as_tensor(delta_target, dtype=torch.float32).unsqueeze(0)

        pred = self.model(s_tensor, a_tensor)
        loss = F.mse_loss(pred, target_tensor)
        loss.backward()
        self.optimizer.step()

    def summary(self) -> Dict[str, Any]:
        """Aggregate statistical summary of recent self-model telemetry."""
        if not self.recent_receipts:
            return {
                'step_counter': self.step_counter,
                'control_mode': self.control_mode.value,
                'mean_self_authorship': 0.0,
                'mean_external_disturbance': 0.0,
                'mean_actuator_fault': 0.0,
                'self_caused_ratio': 0.0
            }
        authorships = [r.self_authorship for r in self.recent_receipts]
        disturbances = [r.external_disturbance for r in self.recent_receipts]
        faults = [r.actuator_fault for r in self.recent_receipts]
        self_caused = [1.0 if r.is_self_caused else 0.0 for r in self.recent_receipts]

        return {
            'step_counter': self.step_counter,
            'control_mode': self.control_mode.value,
            'mean_self_authorship': float(np.mean(authorships)),
            'mean_external_disturbance': float(np.mean(disturbances)),
            'mean_actuator_fault': float(np.mean(faults)),
            'self_caused_ratio': float(np.mean(self_caused)),
            'recent_samples': len(self.recent_receipts)
        }

    def export_state(self) -> Dict[str, Any]:
        return deepcopy({
            'schema': 2,
            'step_counter': self.step_counter,
            'control_mode': self.control_mode.value,
            'action_history': [a.tolist() for a in self.action_history],
            'model_state': {k: v.detach().cpu().numpy().tolist() for k, v in self.model.state_dict().items()},
            'optimizer': self.optimizer.state_dict(),
            'rng': self.rng.bit_generator.state,
            'recent_receipts': [r.to_dict() for r in self.recent_receipts],
            'authorship_threshold': self.authorship_threshold,
        })

    def import_state(self, state: Dict[str, Any]):
        self.step_counter = int(state.get('step_counter', 0))
        self.control_mode = ControlMode(state.get('control_mode', ControlMode.ACTIVE.value))
        if 'model_state' in state:
            state_dict = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in state['model_state'].items()}
            self.model.load_state_dict(state_dict)
        if 'action_history' in state:
            self.action_history = [np.asarray(a, dtype=np.float32) for a in state['action_history']]
        if 'optimizer' in state:
            self.optimizer.load_state_dict(deepcopy(state['optimizer']))
        else:
            self.optimizer.state.clear()
        if 'rng' in state:
            self.rng.bit_generator.state = deepcopy(state['rng'])
        self.recent_receipts = [AuthorshipReceipt(**r) for r in state.get('recent_receipts', [])]
        self.authorship_threshold = float(state.get('authorship_threshold', self.authorship_threshold))
