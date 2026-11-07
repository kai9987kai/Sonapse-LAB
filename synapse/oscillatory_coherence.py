"""Multi-Band Oscillatory Phase-Locking & Cross-Neuropil Communication Through Coherence (CTC).

Grounding:
Inspired by Fries (2015) Communication Through Coherence (CTC) and Kuramoto coupled
oscillators across distinct anatomical neuropils (e.g., FlyWire central brain,
MANC nerve cord, Optic Lobe, MaleCNS prior):
    d(theta_m)/dt = omega_m + (K / M) * sum_n A_{mn} * sin(theta_n - theta_m) + drive_m(t)
    R(t) = | (1/M) * sum_{m=1}^M exp(i * theta_m(t)) |   in [0, 1]

Tracks three canonical frequency bands:
- Theta (4-8 Hz): Episodic memory & spatial navigation binding
- Beta (13-30 Hz): Top-down motor hold & postural stability
- Gamma (40-80 Hz): Bottom-up sensory prediction error & fast reflex routing
"""
from __future__ import annotations
from copy import deepcopy
import math
from typing import Any, Dict, List, Sequence
import numpy as np


class OscillatoryCoherenceEngine:
    """Coupled Kuramoto phase oscillators tracking cross-neuropil coherence and multi-band power."""

    def __init__(self, source_names: Sequence[str] = ("flywire", "manc", "optic-lobe"), coupling_k: float = 0.45):
        self.source_names: List[str] = [str(s) for s in source_names] or ["flywire", "manc", "optic-lobe"]
        self.num_oscillators = len(self.source_names)
        self.coupling_k = float(coupling_k)
        self.reset()

    def reset(self) -> None:
        # Deterministic initial phase spread across neuropils
        self.phases = np.linspace(0.0, 1.2, self.num_oscillators, dtype=np.float64)
        self.natural_freqs = np.linspace(0.32, 0.48, self.num_oscillators, dtype=np.float64)
        self.theta_power = 0.33
        self.beta_power = 0.34
        self.gamma_power = 0.33
        self.kuramoto_r = float(np.abs(np.mean(np.exp(1j * self.phases))))
        self.ctc_gate = 1.0
        self.ticks = 0

    def step(
        self,
        rates_np: np.ndarray,
        origin_np: np.ndarray,
        surprisal: float = 0.0,
        ach: float = 0.5,
        dopamine: float = 0.5,
        learn: bool = True,
    ) -> Dict[str, Any]:
        """Advance coupled neuropil oscillators one step from real population rates."""
        m = self.num_oscillators
        pop_drives = np.zeros(m, dtype=np.float64)
        for idx in range(m):
            mask = origin_np == idx
            if np.any(mask):
                pop_drives[idx] = float(np.mean(np.abs(rates_np[mask])))
            else:
                pop_drives[idx] = float(np.mean(np.abs(rates_np))) if len(rates_np) else 0.0

        # Kuramoto phase update
        new_phases = self.phases.copy()
        for i in range(m):
            diff_sum = np.sum(np.sin(self.phases - self.phases[i]))
            drive_term = 0.25 * math.tanh(pop_drives[i]) + 0.1 * float(ach - 0.5)
            dtheta = self.natural_freqs[i] + (self.coupling_k / max(1, m)) * diff_sum + drive_term
            new_phases[i] = (self.phases[i] + dtheta) % (2.0 * math.pi)

        # Global Kuramoto order parameter R(t) in [0, 1]
        z = np.mean(np.exp(1j * new_phases))
        kuramoto_r = float(np.clip(np.abs(z), 0.0, 1.0))

        # Spectral band allocation from activity, neuromodulation, and sensory surprisal
        raw_gamma = 0.30 + 0.50 * math.tanh(abs(surprisal) + 0.5 * float(ach)) + 0.20 * float(np.mean(pop_drives))
        raw_theta = 0.30 + 0.45 * math.tanh(float(dopamine) + 0.4 * kuramoto_r)
        raw_beta = 0.30 + 0.40 * math.tanh(max(0.0, 1.0 - abs(surprisal)))
        total_p = max(1e-6, raw_theta + raw_beta + raw_gamma)
        theta_power = float(raw_theta / total_p)
        beta_power = float(raw_beta / total_p)
        gamma_power = float(raw_gamma / total_p)
        ctc_gate = float(np.clip(0.92 + 0.18 * kuramoto_r, 0.90, 1.12))

        if learn:
            self.ticks += 1
            self.phases = new_phases
            self.kuramoto_r = kuramoto_r
            self.theta_power = theta_power
            self.beta_power = beta_power
            self.gamma_power = gamma_power
            self.ctc_gate = ctc_gate
            return self.summary()

        bands = {"theta_4_8hz": theta_power, "beta_13_30hz": beta_power, "gamma_40_80hz": gamma_power}
        dominant = max(bands, key=bands.get)
        return {
            "enabled": True,
            "sources": list(self.source_names),
            "kuramoto_order_r": round(float(kuramoto_r), 4),
            "ctc_routing_gate": round(float(ctc_gate), 4),
            "theta_power": round(float(theta_power), 4),
            "beta_power": round(float(beta_power), 4),
            "gamma_power": round(float(gamma_power), 4),
            "dominant_band": dominant,
            "phases_rad": [round(float(p), 4) for p in new_phases],
            "coherence_matrix": self.coherence_matrix(),
            "ticks": int(self.ticks),
        }

    def coherence_matrix(self) -> List[List[float]]:
        m = self.num_oscillators
        mat = []
        for i in range(m):
            row = []
            for j in range(m):
                row.append(round(float(0.5 * (1.0 + math.cos(float(self.phases[i] - self.phases[j])))), 4))
            mat.append(row)
        return mat

    def summary(self) -> Dict[str, Any]:
        bands = {"theta_4_8hz": self.theta_power, "beta_13_30hz": self.beta_power, "gamma_40_80hz": self.gamma_power}
        dominant = max(bands, key=bands.get)
        return {
            "enabled": True,
            "sources": list(self.source_names),
            "kuramoto_order_r": round(float(self.kuramoto_r), 4),
            "ctc_routing_gate": round(float(self.ctc_gate), 4),
            "theta_power": round(float(self.theta_power), 4),
            "beta_power": round(float(self.beta_power), 4),
            "gamma_power": round(float(self.gamma_power), 4),
            "dominant_band": dominant,
            "phases_rad": [round(float(p), 4) for p in self.phases],
            "coherence_matrix": self.coherence_matrix(),
            "ticks": int(self.ticks),
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "source_names": list(self.source_names),
            "phases": self.phases.tolist(),
            "natural_freqs": self.natural_freqs.tolist(),
            "theta_power": float(self.theta_power),
            "beta_power": float(self.beta_power),
            "gamma_power": float(self.gamma_power),
            "kuramoto_r": float(self.kuramoto_r),
            "ctc_gate": float(self.ctc_gate),
            "ticks": int(self.ticks),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if "source_names" in state and len(state["source_names"]) == len(state.get("phases", [])):
            self.source_names = [str(s) for s in state["source_names"]]
            self.num_oscillators = len(self.source_names)
        if "phases" in state:
            self.phases = np.asarray(state["phases"], dtype=np.float64)
        if "natural_freqs" in state:
            self.natural_freqs = np.asarray(state["natural_freqs"], dtype=np.float64)
        self.theta_power = float(state.get("theta_power", 0.33))
        self.beta_power = float(state.get("beta_power", 0.34))
        self.gamma_power = float(state.get("gamma_power", 0.33))
        self.kuramoto_r = float(state.get("kuramoto_r", 1.0))
        self.ctc_gate = float(state.get("ctc_gate", 1.0))
        self.ticks = int(state.get("ticks", 0))
