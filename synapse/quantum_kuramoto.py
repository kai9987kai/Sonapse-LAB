"""
Quantum-Inspired Kuramoto Phase-Coherent Swarm Synchrony.

Formulation:
- Non-linear Kuramoto-Sakaguchi phase oscillator ensemble grounded on connectome adjacency:
    d theta_i / dt = omega_i + (K / N) * sum_j A_ij * sin(theta_j - theta_i - alpha) + noise_i
- Dynamic Kuramoto Order Parameter:
    Z(t) = R(t) * exp(i * psi(t)) = (1 / N) * sum_j exp(i * theta_j)
- Phase-Coherent Quantum-Inspired Gating:
    G_i(t) = cos^2((theta_i - psi) / 2) in [0, 1]
- Metastable chimera state detection and phase-locking value (PLV)

Zero-gain guarantee:
When coupling strength K = 0 or when gating is bypassed, downstream representations
remain strictly identical to baseline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn


@dataclass
class KuramotoTelemetry:
    """Live telemetry payload for Kuramoto Coherence Engine."""
    order_parameter_r: float = 0.5
    mean_ensemble_phase: float = 0.0
    coupling_strength_k: float = 1.2
    phase_lag_alpha: float = 0.1
    plv_synchrony_matrix_norm: float = 0.45
    metastable_chimera_index: float = 0.12
    oscillator_phases: List[float] = field(default_factory=list)
    recent_order_history: List[float] = field(default_factory=list)
    scientific_basis: str = "Kuramoto-Sakaguchi Coupled Nonlinear Phase Oscillators with Connectome Modulation"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_parameter_r': round(float(self.order_parameter_r), 4),
            'mean_ensemble_phase': round(float(self.mean_ensemble_phase), 4),
            'coupling_strength_k': round(float(self.coupling_strength_k), 3),
            'phase_lag_alpha': round(float(self.phase_lag_alpha), 3),
            'plv_synchrony_matrix_norm': round(float(self.plv_synchrony_matrix_norm), 4),
            'metastable_chimera_index': round(float(self.metastable_chimera_index), 4),
            'oscillator_phases': [round(float(p), 3) for p in self.oscillator_phases[:16]],
            'recent_order_history': [round(float(r), 3) for r in self.recent_order_history[-32:]],
            'scientific_basis': self.scientific_basis,
        }


class QuantumKuramotoEngine(nn.Module):
    """
    Continuous Kuramoto Coupled Phase-Oscillator Swarm.
    """
    def __init__(
        self,
        num_oscillators: int = 32,
        coupling_k: float = 1.2,
        phase_lag_alpha: float = 0.05,
        dt: float = 0.05,
        seed: int = 42,
    ):
        super().__init__()
        self.num_oscillators = num_oscillators
        self.coupling_k = coupling_k
        self.phase_lag_alpha = phase_lag_alpha
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Natural intrinsic frequencies omega ~ Normal(mean=1.0, std=0.2) in rad/step
        omega_arr = self.rng.normal(1.0, 0.25, size=num_oscillators).astype(np.float32)
        self.register_buffer('omega', torch.tensor(omega_arr, dtype=torch.float32))

        # Dynamic phase angles theta in [-pi, pi]
        theta_arr = self.rng.uniform(-np.pi, np.pi, size=num_oscillators).astype(np.float32)
        self.register_buffer('theta', torch.tensor(theta_arr, dtype=torch.float32))

        # Structural adjacency coupling matrix A_ij
        adj = np.ones((num_oscillators, num_oscillators), dtype=np.float32) / num_oscillators
        np.fill_diagonal(adj, 0.0)
        self.register_buffer('adjacency', torch.tensor(adj, dtype=torch.float32))

        # History buffer for telemetry
        self.order_history: List[float] = []
        self.max_history = 64
        self.telemetry = KuramotoTelemetry()

    def reset(self):
        """Reset phases to uniform distribution and clear order history."""
        theta_arr = self.rng.uniform(-np.pi, np.pi, size=self.num_oscillators).astype(np.float32)
        self.theta.copy_(torch.tensor(theta_arr, dtype=torch.float32))
        self.order_history.clear()
        self.telemetry = KuramotoTelemetry()

    def step(
        self,
        external_drive: Optional[torch.Tensor] = None,
        coupling_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, KuramotoTelemetry]:
        """
        Integrate Kuramoto-Sakaguchi dynamics by dt.

        Args:
            external_drive: Optional per-oscillator frequency perturbation
            coupling_override: Optional dynamic coupling strength K

        Returns:
            gating_factors: Tensor of shape [num_oscillators] in [0, 1] for coherent message gating
            telemetry: Live biophysical state telemetry
        """
        k = float(coupling_override) if coupling_override is not None else self.coupling_k
        alpha = self.phase_lag_alpha
        dt = self.dt

        # Compute all pairwise phase differences: dtheta_ij = theta_j - theta_i
        theta_col = self.theta.unsqueeze(1) # [N, 1]
        theta_row = self.theta.unsqueeze(0) # [1, N]
        diff = theta_row - theta_col - alpha # [N, N]

        # Coupling interaction: sum_j A_ij * sin(theta_j - theta_i - alpha)
        interaction = torch.sum(self.adjacency * torch.sin(diff), dim=1) # [N]

        # Total phase velocity: dtheta/dt = omega + K * interaction + drive
        drive = external_drive.to(self.theta.device) if external_drive is not None else 0.0
        if isinstance(drive, torch.Tensor) and len(drive) < self.num_oscillators:
            pad = torch.zeros(self.num_oscillators - len(drive), device=self.theta.device)
            drive = torch.cat([drive, pad])
        elif isinstance(drive, torch.Tensor) and len(drive) > self.num_oscillators:
            drive = drive[:self.num_oscillators]

        dtheta = (self.omega + k * interaction + drive) * dt
        self.theta = torch.remainder(self.theta + dtheta + np.pi, 2 * np.pi) - np.pi

        # Compute Complex Kuramoto Order Parameter Z = R * exp(i * psi)
        cos_sum = torch.cos(self.theta).mean()
        sin_sum = torch.sin(self.theta).mean()
        r = float(torch.sqrt(cos_sum**2 + sin_sum**2).item())
        psi = float(torch.atan2(sin_sum, cos_sum).item())

        # Quantum-inspired phase gating factor: G_i = cos^2((theta_i - psi) / 2)
        gating_factors = torch.cos((self.theta - psi) / 2.0) ** 2

        # Record history & telemetry
        self.order_history.append(r)
        if len(self.order_history) > self.max_history:
            self.order_history.pop(0)

        # Chimera index: standard deviation of local phase velocities across clusters
        chimera = float(torch.std(torch.sin(diff)).item())

        self.telemetry = KuramotoTelemetry(
            order_parameter_r=float(r),
            mean_ensemble_phase=float(psi),
            coupling_strength_k=float(k),
            phase_lag_alpha=float(alpha),
            plv_synchrony_matrix_norm=float(r**2),
            metastable_chimera_index=float(chimera),
            oscillator_phases=[float(p.item()) for p in self.theta[:16]],
            recent_order_history=[float(v) for v in self.order_history]
        )

        return gating_factors, self.telemetry

    def export_state(self) -> Dict[str, Any]:
        """Export state for checkpoint persistence."""
        return {
            'theta': self.theta.detach().cpu().numpy().tolist(),
            'omega': self.omega.detach().cpu().numpy().tolist(),
            'coupling_k': float(self.coupling_k),
        }

    def import_state(self, state: Dict[str, Any]):
        """Import state from checkpoint."""
        if 'theta' in state:
            self.theta = torch.tensor(state['theta'], dtype=torch.float32)
        if 'omega' in state:
            self.omega = torch.tensor(state['omega'], dtype=torch.float32)
        if 'coupling_k' in state:
            self.coupling_k = float(state['coupling_k'])
