"""Relational, Holographic & Neuro-Symbolic Working Memory (HRR / VSA).

Supports compositional role-filler binding for:
Objects, Agents, Locations, Events, Actions, Goals, Causes, and Task Contexts
using exact Fourier-domain circular convolution and involution unbinding.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class HRRCore:
    """Fourier Holographic Reduced Representation Core."""

    def __init__(self, dim: int = 64, seed: int = 42):
        self.dim = int(dim)
        self.rng = np.random.default_rng(seed)

    def random_unitary(self) -> np.ndarray:
        half = self.dim // 2
        phases = self.rng.uniform(-math.pi, math.pi, size=half - 1)
        fft_vec = np.ones(self.dim, dtype=np.complex128)
        fft_vec[0] = 1.0
        fft_vec[half] = 1.0
        fft_vec[1:half] = np.exp(1j * phases)
        fft_vec[half + 1 :] = np.conj(fft_vec[1:half][::-1])
        v = np.fft.ifft(fft_vec).real.astype(np.float32)
        return v / (float(np.linalg.norm(v)) + 1e-8)

    @staticmethod
    def circular_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        fa = np.fft.rfft(np.asarray(a, dtype=np.float64))
        fb = np.fft.rfft(np.asarray(b, dtype=np.float64))
        return np.fft.irfft(fa * fb, n=len(a)).astype(np.float32)

    @staticmethod
    def involution_inverse(a: np.ndarray) -> np.ndarray:
        arr = np.asarray(a, dtype=np.float32)
        inv = np.empty_like(arr)
        inv[0] = arr[0]
        inv[1:] = arr[:0:-1]
        return inv

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.circular_conv(a, b)

    def unbind(self, bound_item: np.ndarray, key: np.ndarray) -> np.ndarray:
        return self.circular_conv(bound_item, self.involution_inverse(key))

    @staticmethod
    def superpose(vectors: List[np.ndarray]) -> np.ndarray:
        if not vectors:
            return np.zeros(1, dtype=np.float32)
        accum = np.zeros_like(vectors[0], dtype=np.float32)
        for v in vectors:
            accum += np.asarray(v, dtype=np.float32)
        return accum


class RelationalMemoryBank:
    """Manages compositional role-filler bindings in a fixed-width vector space."""

    def __init__(self, dim: int = 64, seed: int = 42, max_slots: Optional[int] = None, **kwargs):
        self.dim = int(dim)
        self.max_slots = int(max_slots) if max_slots is not None else 64
        self.hrr = HRRCore(dim=self.dim, seed=seed)
        self.rng = self.hrr.rng

        # Base role symbols (unitary vectors with unit spectral magnitude)
        self.roles = {
            name: self.hrr.random_unitary()
            for name in [
                "object", "agent", "location", "event", "action",
                "goal", "cause", "temporal_order", "task_context"
            ]
        }
        self.symbols: Dict[str, np.ndarray] = dict(self.roles)
        self.memory_trace = np.zeros(self.dim, dtype=np.float32)
        self.bound_items_count: int = 0

    def circular_conv(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.hrr.circular_conv(a, b)

    def involution_inverse(self, a: np.ndarray) -> np.ndarray:
        return self.hrr.involution_inverse(a)

    def bind(self, role: str, filler: np.ndarray) -> np.ndarray:
        r_vec = self.roles.get(role, self.roles["object"])
        return self.hrr.bind(r_vec, filler)

    def unbind(self, bound_item: np.ndarray, role: str) -> np.ndarray:
        r_vec = self.roles.get(role, self.roles["object"])
        return self.hrr.unbind(bound_item, r_vec)

    def store_relation(self, role: str, filler: np.ndarray) -> None:
        item = self.bind(role, filler)
        self.memory_trace += item
        self.bound_items_count += 1

    def query_relation(self, role: str) -> np.ndarray:
        return self.unbind(self.memory_trace, role)

    def bind_step(self, state: Any, action: Any, next_state: Any) -> Dict[str, Any]:
        """Bind transition (state, action, next_state) into relational memory trace."""
        s = np.asarray(state, dtype=np.float32).flatten()
        a = np.asarray(action, dtype=np.float32).flatten()
        ns = np.asarray(next_state, dtype=np.float32).flatten()

        def _to_dim(vec: np.ndarray) -> np.ndarray:
            if len(vec) < self.dim:
                return np.pad(vec, (0, self.dim - len(vec)))
            return vec[:self.dim]

        s_bound = self.bind("task_context", _to_dim(s))
        a_bound = self.bind("action", _to_dim(a))
        ns_bound = self.bind("event", _to_dim(ns))

        step_trace = s_bound + a_bound + ns_bound
        self.memory_trace = 0.95 * self.memory_trace + 0.05 * step_trace
        self.bound_items_count += 1
        return {
            "bound_items": int(self.bound_items_count),
            "memory_trace_norm": round(float(np.linalg.norm(self.memory_trace)), 4),
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "bound_items_count": int(self.bound_items_count),
            "memory_trace": self.memory_trace.copy(),
            "symbols": {k: v.copy() for k, v in self.symbols.items()},
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self.bound_items_count = int(state.get("bound_items_count", self.bound_items_count))
        if "memory_trace" in state:
            self.memory_trace = np.asarray(state["memory_trace"], dtype=np.float32).copy()
        if "symbols" in state:
            self.symbols = {k: np.asarray(v, dtype=np.float32).copy() for k, v in state["symbols"].items()}
