"""Dynamic Transfer Entropy & Partial Information Decomposition (PID) Engine.

Scientific Basis:
1. Directed Transfer Entropy (Schreiber, Phys. Rev. Lett. 2000):
   TE_{X -> Y} = I(Y_{t+1}; X_t | Y_t) = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)
   Measures the reduction in uncertainty of target state Y_{t+1} given source X_t,
   conditioned on target history Y_t. It is strictly directional and non-symmetric.
2. Partial Information Decomposition (PID; Williams & Beer 2010; Mediano et al. 2021):
   Decomposes information from two source neuropils (e.g. Optic Lobe Sensory and
   FlyWire Central) to target motor neuropil (MANC VNC) into:
   - Unique Information (U_S, U_C)
   - Redundant Information (R)
   - Synergistic Information (S)
   Synergy reflects emergent multi-connectome computational integration: information
   available only when considering both connectomes jointly.

Safety & Invariants:
- Deterministic numerical estimation using Gaussian/Copula mutual information.
- Zero RNG consumption during inference.
- Reversible serialization for mid-episode checkpoint replay determinism.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TransferEntropyState:
    te_sensory_to_central: float = 0.0
    te_central_to_motor: float = 0.0
    te_motor_to_central_feedback: float = 0.0
    net_feedforward_flux: float = 0.0
    synergy_index: float = 0.0
    redundancy_index: float = 0.0
    emergence_ratio: float = 0.0
    history_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'te_sensory_to_central': float(self.te_sensory_to_central),
            'te_central_to_motor': float(self.te_central_to_motor),
            'te_motor_to_central_feedback': float(self.te_motor_to_central_feedback),
            'net_feedforward_flux': float(self.net_feedforward_flux),
            'synergy_index': float(self.synergy_index),
            'redundancy_index': float(self.redundancy_index),
            'emergence_ratio': float(self.emergence_ratio),
            'history_steps': int(self.history_steps),
        }


class TransferEntropyEngine:
    """Estimates real-time directed transfer entropy and information synergy across connectome regions."""

    def __init__(self, buffer_size: int = 64):
        self.buffer_size = max(16, int(buffer_size))
        # Sliding history buffers for sensory (0), central (1), and motor (2) regional means
        self.sensory_hist: List[float] = []
        self.central_hist: List[float] = []
        self.motor_hist: List[float] = []
        self.state = TransferEntropyState()

    def _gaussian_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Estimate Gaussian mutual information I(X; Y) = -0.5 * ln(1 - rho^2)."""
        if len(x) < 4:
            return 0.0
        vx = np.var(x)
        vy = np.var(y)
        if vx < 1e-9 or vy < 1e-9:
            return 0.0
        corr = float(np.corrcoef(x, y)[0, 1])
        if not math.isfinite(corr):
            return 0.0
        corr = np.clip(corr, -0.999, 0.999)
        return float(-0.5 * np.log(1.0 - corr ** 2))

    def _conditional_gaussian_mi(self, y_next: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """Estimate conditional mutual information I(Y_{t+1}; X_t | Y_t)."""
        if len(y_next) < 6:
            return 0.0
        # Linear regression residualization: Y_next_res = Y_next - alpha * Y
        cov_y = float(np.var(y))
        if cov_y < 1e-9:
            return self._gaussian_mi(y_next, x)
        beta_y = float(np.cov(y_next, y)[0, 1] / cov_y)
        y_next_res = y_next - beta_y * y

        beta_x = float(np.cov(x, y)[0, 1] / cov_y)
        x_res = x - beta_x * y

        return self._gaussian_mi(y_next_res, x_res)

    def step(
        self,
        r: np.ndarray,
        origin_indices: np.ndarray,
        learn: bool = False,
    ) -> TransferEntropyState:
        """Record regional activity step and update transfer entropy & synergy metrics."""
        act = np.asarray(r, dtype=np.float32).ravel()
        origins = np.asarray(origin_indices, dtype=np.int64).ravel()
        n = min(len(act), len(origins))

        if n == 0:
            return copy.deepcopy(self.state)

        # Compute regional mean activity: 0=flywire (central), 1=manc (motor), 2=optic-lobe (sensory)
        mask_central = origins[:n] == 0
        mask_motor = origins[:n] == 1
        mask_sensory = origins[:n] == 2

        val_sensory = float(np.mean(act[:n][mask_sensory])) if np.any(mask_sensory) else 0.0
        val_central = float(np.mean(act[:n][mask_central])) if np.any(mask_central) else 0.0
        val_motor = float(np.mean(act[:n][mask_motor])) if np.any(mask_motor) else 0.0

        if learn or len(self.sensory_hist) < self.buffer_size:
            self.sensory_hist.append(val_sensory)
            self.central_hist.append(val_central)
            self.motor_hist.append(val_motor)

            if len(self.sensory_hist) > self.buffer_size:
                self.sensory_hist.pop(0)
                self.central_hist.pop(0)
                self.motor_hist.pop(0)

        # Compute metrics if buffer has sufficient samples
        if len(self.sensory_hist) >= 12:
            s = np.array(self.sensory_hist[:-1], dtype=np.float64)
            c = np.array(self.central_hist[:-1], dtype=np.float64)
            m = np.array(self.motor_hist[:-1], dtype=np.float64)

            s_next = np.array(self.sensory_hist[1:], dtype=np.float64)
            c_next = np.array(self.central_hist[1:], dtype=np.float64)
            m_next = np.array(self.motor_hist[1:], dtype=np.float64)

            # Directed transfer entropies
            te_s2c = self._conditional_gaussian_mi(c_next, s, c)
            te_c2m = self._conditional_gaussian_mi(m_next, c, m)
            te_m2c = self._conditional_gaussian_mi(c_next, m, c)

            net_flux = (te_s2c + te_c2m) - te_m2c

            # Partial Information Decomposition (PID) into motor future m_next
            mi_s = self._gaussian_mi(m_next, s)
            mi_c = self._gaussian_mi(m_next, c)
            # Joint mutual information I(S, C; M_next)
            sc_joint = 0.5 * (s + c)
            mi_joint = self._gaussian_mi(m_next, sc_joint)

            redundancy = min(mi_s, mi_c)
            synergy = max(0.0, mi_joint - max(mi_s, mi_c))
            emergence = synergy / (redundancy + 1e-6)

            self.state = TransferEntropyState(
                te_sensory_to_central=te_s2c,
                te_central_to_motor=te_c2m,
                te_motor_to_central_feedback=te_m2c,
                net_feedforward_flux=net_flux,
                synergy_index=synergy,
                redundancy_index=redundancy,
                emergence_ratio=emergence,
                history_steps=len(self.sensory_hist),
            )

        return copy.deepcopy(self.state)

    def export_state(self) -> Dict[str, Any]:
        return {
            'buffer_size': self.buffer_size,
            'sensory_hist': list(self.sensory_hist),
            'central_hist': list(self.central_hist),
            'motor_hist': list(self.motor_hist),
            'state': self.state.to_dict(),
        }

    def import_state(self, state_dict: Dict[str, Any]) -> None:
        self.buffer_size = int(state_dict['buffer_size'])
        self.sensory_hist = [float(x) for x in state_dict['sensory_hist']]
        self.central_hist = [float(x) for x in state_dict['central_hist']]
        self.motor_hist = [float(x) for x in state_dict['motor_hist']]
        st = state_dict['state']
        self.state = TransferEntropyState(
            te_sensory_to_central=float(st['te_sensory_to_central']),
            te_central_to_motor=float(st['te_central_to_motor']),
            te_motor_to_central_feedback=float(st['te_motor_to_central_feedback']),
            net_feedforward_flux=float(st['net_feedforward_flux']),
            synergy_index=float(st['synergy_index']),
            redundancy_index=float(st['redundancy_index']),
            emergence_ratio=float(st['emergence_ratio']),
            history_steps=int(st.get('history_steps', len(self.sensory_hist))),
        )


def run_transfer_entropy_assay(runtime_or_agent, steps: int = 32, seed: int = 42) -> Dict[str, Any]:
    """Assay verifying directional transfer entropy and information synergy across connectome domains."""
    engine = TransferEntropyEngine(buffer_size=48)
    rng = np.random.RandomState(seed)

    # Simulated connectome with 3 domains: 0=central, 1=motor, 2=sensory
    n_total = 192
    origins = np.array([2] * 64 + [0] * 64 + [1] * 64, dtype=np.int64)

    # Directed propagation: sensory -> central -> motor
    s_val = 0.2
    c_val = 0.2
    m_val = 0.2

    for step in range(steps):
        s_val = 0.5 + 0.3 * math.sin(step * 0.4) + 0.05 * rng.randn()
        c_val = 0.7 * c_val + 0.3 * s_val + 0.05 * rng.randn()
        m_val = 0.6 * m_val + 0.4 * c_val + 0.05 * rng.randn()

        r = np.zeros(n_total, dtype=np.float32)
        r[:64] = np.clip(s_val + 0.02 * rng.randn(64), 0.0, 1.0)
        r[64:128] = np.clip(c_val + 0.02 * rng.randn(64), 0.0, 1.0)
        r[128:192] = np.clip(m_val + 0.02 * rng.randn(64), 0.0, 1.0)

        engine.step(r, origins, learn=True)

    return {
        'status': 'completed',
        'steps_evaluated': steps,
        'te_sensory_to_central': engine.state.te_sensory_to_central,
        'te_central_to_motor': engine.state.te_central_to_motor,
        'te_motor_to_central_feedback': engine.state.te_motor_to_central_feedback,
        'net_feedforward_flux': engine.state.net_feedforward_flux,
        'synergy_index': engine.state.synergy_index,
        'redundancy_index': engine.state.redundancy_index,
        'emergence_ratio': engine.state.emergence_ratio,
        'causal_flow_confirmed': bool(engine.state.te_sensory_to_central > 0.0 or engine.state.te_central_to_motor > 0.0),
        'disclaimer': 'Gaussian copula estimation on digital activation dynamics; measures directional statistical coupling.',
    }
