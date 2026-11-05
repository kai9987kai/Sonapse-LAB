"""Holographic Reduced Representation (HRR) Associative Memory for Poseidon.

Implements high-dimensional Vector Symbolic Architecture (VSA / HRR) with
circular convolution binding, circular correlation unbinding, superposition
trace bundling, clean-up vocabulary resonance, and capacity load auditing.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class HolographicAssociativeMemory:
    """Vector Symbolic Architecture using circular convolution Holographic Reduced Representations."""

    def __init__(self, dimension: int = 256, decay_rate: float = 0.98):
        self.dimension = dimension
        self.decay_rate = decay_rate
        # Memory trace vector M in R^D
        self.trace = np.zeros(dimension, dtype=np.float64)
        self.item_count = 0

        # Clean-up item memory vocabulary: name -> unit vector
        self.vocabulary: Dict[str, np.ndarray] = {}
        self._seed_default_vocabulary()

    def _generate_unit_vector(self, name: str) -> np.ndarray:
        """Deterministically generate an approximately orthogonal unit vector for a concept name."""
        import hashlib
        h = hashlib.sha256(name.encode("utf-8")).digest()
        # Seed local PRNG with hash bytes
        seed = int.from_bytes(h[:4], "little")
        rng = np.random.default_rng(seed)
        vec = rng.normal(0.0, 1.0 / math.sqrt(self.dimension), size=self.dimension)
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec /= norm
        return vec

    def _seed_default_vocabulary(self) -> None:
        """Seed foundational domain concepts for Poseidon TidePool world."""
        concepts = [
            "ROLE_FOOD", "ROLE_WATER", "ROLE_SHELTER", "ROLE_PREDATOR",
            "ROLE_STAMINA", "ROLE_WEATHER", "ROLE_HEALTH", "ROLE_COMPASS",
            "STATE_CRITICAL", "STATE_DEPLETED", "STATE_SCARCE", "STATE_STABLE",
            "STATE_ABUNDANT", "STATE_THREATENING", "STATE_STORM_IMPENDING", "STATE_RESTED",
        ]
        for c in concepts:
            self.vocabulary[c] = self._generate_unit_vector(c)

    def register_concept(self, name: str) -> np.ndarray:
        """Register a new concept atom in the clean-up vocabulary."""
        if name not in self.vocabulary:
            self.vocabulary[name] = self._generate_unit_vector(name)
        return self.vocabulary[name]

    @staticmethod
    def bind(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Circular convolution: z = x (*) y = ifft(fft(x) * fft(y))."""
        fx = np.fft.rfft(x)
        fy = np.fft.rfft(y)
        z = np.fft.irfft(fx * fy, n=len(x))
        norm = np.linalg.norm(z)
        if norm > 1e-12:
            z /= norm
        return z

    @staticmethod
    def unbind(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Circular correlation: y' = x (*)^dagger z = ifft(conj(fft(x)) * fft(z))."""
        fx = np.fft.rfft(x)
        fz = np.fft.rfft(z)
        y = np.fft.irfft(np.conj(fx) * fz, n=len(x))
        norm = np.linalg.norm(y)
        if norm > 1e-12:
            y /= norm
        return y

    def remember_association(self, role_name: str, state_name: str, weight: float = 1.0) -> None:
        """Bind role and state into a holographic chunk and bundle into trace M."""
        role_vec = self.register_concept(role_name)
        state_vec = self.register_concept(state_name)
        bound = self.bind(role_vec, state_vec)

        # Decay existing trace and accumulate
        self.trace = self.decay_rate * self.trace + (weight * bound)
        norm = np.linalg.norm(self.trace)
        if norm > 1e-12:
            self.trace /= norm
        self.item_count += 1

    def query(self, role_name: str) -> Tuple[Optional[str], float]:
        """Query memory trace with role cue, unbind and resonant match against vocabulary."""
        if role_name not in self.vocabulary or np.linalg.norm(self.trace) < 1e-12:
            return None, 0.0

        role_vec = self.vocabulary[role_name]
        retrieved_vec = self.unbind(role_vec, self.trace)

        # Clean-up match against vocabulary
        best_match: Optional[str] = None
        best_sim = -1.0

        for name, item_vec in self.vocabulary.items():
            if name == role_name:
                continue
            sim = float(np.dot(retrieved_vec, item_vec))
            if sim > best_sim:
                best_sim = sim
                best_match = name

        return best_match, max(-1.0, min(1.0, best_sim))

    def encode_observation(self, obs: List[float] | np.ndarray) -> Dict[str, Any]:
        """Convert 16-d observation vector into structured holographic memory associations."""
        if len(obs) < 16:
            return {"associations_added": 0}

        food_level = float(obs[0])
        water_level = float(obs[1])
        stamina_level = float(obs[2])
        health_level = float(obs[3])
        threat_level = float(obs[4])
        storm_level = float(obs[5])

        added = 0
        if food_level < 0.2:
            self.remember_association("ROLE_FOOD", "STATE_CRITICAL")
            added += 1
        elif food_level < 0.5:
            self.remember_association("ROLE_FOOD", "STATE_SCARCE")
            added += 1
        else:
            self.remember_association("ROLE_FOOD", "STATE_ABUNDANT")
            added += 1

        if water_level < 0.2:
            self.remember_association("ROLE_WATER", "STATE_CRITICAL")
            added += 1
        elif water_level < 0.5:
            self.remember_association("ROLE_WATER", "STATE_SCARCE")
            added += 1

        if stamina_level < 0.25:
            self.remember_association("ROLE_STAMINA", "STATE_DEPLETED")
            added += 1

        if threat_level > 0.4:
            self.remember_association("ROLE_PREDATOR", "STATE_THREATENING")
            added += 1

        if storm_level > 0.5:
            self.remember_association("ROLE_WEATHER", "STATE_STORM_IMPENDING")
            added += 1

        return {
            "associations_added": added,
            "total_items": self.item_count,
            "capacity_load": round(self.item_count / self.dimension, 4),
        }

    def telemetry(self) -> Dict[str, Any]:
        """Return diagnostic metrics of holographic memory state."""
        food_match, food_sim = self.query("ROLE_FOOD")
        water_match, water_sim = self.query("ROLE_WATER")
        threat_match, threat_sim = self.query("ROLE_PREDATOR")

        trace_norm = float(np.linalg.norm(self.trace))
        return {
            "dimension": self.dimension,
            "items_stored": self.item_count,
            "capacity_load": round(self.item_count / self.dimension, 4),
            "trace_norm": round(trace_norm, 4),
            "top_associations": {
                "ROLE_FOOD": {"retrieved": food_match, "resonance": round(food_sim, 4)},
                "ROLE_WATER": {"retrieved": water_match, "resonance": round(water_sim, 4)},
                "ROLE_PREDATOR": {"retrieved": threat_match, "resonance": round(threat_sim, 4)},
            },
        }
