"""PROJECT NEXUS-Ω: Causal Memory Fabric.

Implements multi-substrate memory with formal intervention operators
(ablation, shuffling, transplantation, reconstruction) and the four-fold
operational distinction for memory validation.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from poseidon.nexus_omega.contracts import (
    CarrierState,
    InterventionSpec,
    InterventionType,
    MemoryCarrierType,
)


class CausalMemoryFabric:
    """Multi-carrier memory substrate with verifiable intervention semantics."""

    def __init__(self, d_working: int = 16, d_key: int = 8, d_value: int = 8, max_episodes: int = 128):
        self.d_working = d_working
        self.d_key = d_key
        self.d_value = d_value
        self.max_episodes = max_episodes

        # Reconstruction auto-projector: maps structural graph summary to working memory
        np.random.seed(42)
        self.W_recon = np.random.randn(d_working, d_working) * 0.1
        self.W_recon[0, 0] = 2.0
        # Structural prior matrix (e.g., node connectivity)
        self.structural_prior = np.eye(d_working) * 0.5 + 0.05 * np.ones((d_working, d_working))

        self.reset()

    def reset(self) -> None:
        self.state = CarrierState(
            working_state=np.zeros(self.d_working, dtype=np.float32),
            episodic_keys=[],
            episodic_values=[],
            structural_graph=self.structural_prior.copy(),
            step_count=0,
            intervention_log=[],
        )

    def write_working(self, delta: np.ndarray, decay: float = 0.85) -> None:
        """Bounded recurrent working memory update."""
        if delta.shape[0] != self.d_working:
            raise ValueError(f"Delta shape {delta.shape} does not match d_working {self.d_working}")
        self.state.working_state = np.tanh(decay * self.state.working_state + delta)
        self.state.step_count += 1

    def store_episode(self, key: np.ndarray, value: np.ndarray) -> None:
        """Stores a key-value transition trace into episodic carrier."""
        if len(self.state.episodic_keys) >= self.max_episodes:
            self.state.episodic_keys.pop(0)
            self.state.episodic_values.pop(0)
        self.state.episodic_keys.append(key.copy().astype(np.float32))
        self.state.episodic_values.append(value.copy().astype(np.float32))

    def retrieve_episodic(self, query_key: np.ndarray, temperature: float = 0.5) -> np.ndarray:
        """Associative softmax retrieval over episodic carrier."""
        if not self.state.episodic_keys:
            return np.zeros(self.d_value, dtype=np.float32)

        keys = np.stack(self.state.episodic_keys, axis=0)  # (N, d_key)
        # Cosine similarity
        norm_q = np.linalg.norm(query_key) + 1e-8
        norm_k = np.linalg.norm(keys, axis=-1) + 1e-8
        sims = np.dot(keys, query_key) / (norm_k * norm_q)

        # Numerically stable softmax
        logits = sims / max(temperature, 1e-4)
        logits -= np.max(logits)
        weights = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-8)

        values = np.stack(self.state.episodic_values, axis=0)  # (N, d_value)
        retrieved = np.sum(weights[:, None] * values, axis=0)
        return retrieved.astype(np.float32)

    def apply_intervention(self, spec: InterventionSpec) -> None:
        """Applies explicit causal intervention operator do(X = x)."""
        log_entry = f"do({spec.intervention_type.value}, target={spec.target_carrier})"

        if spec.intervention_type == InterventionType.INTACT:
            return

        elif spec.intervention_type == InterventionType.ABLATE_CARRIER:
            if spec.target_carrier == MemoryCarrierType.WORKING or spec.target_carrier is None:
                self.state.working_state = np.zeros_like(self.state.working_state)
            if spec.target_carrier == MemoryCarrierType.EPISODIC or spec.target_carrier is None:
                self.state.episodic_keys.clear()
                self.state.episodic_values.clear()
            if spec.target_carrier == MemoryCarrierType.STRUCTURAL or spec.target_carrier is None:
                self.state.structural_graph = np.zeros_like(self.state.structural_graph)

        elif spec.intervention_type == InterventionType.SHUFFLE_CARRIER:
            if spec.target_carrier == MemoryCarrierType.WORKING or spec.target_carrier is None:
                perm = np.random.permutation(len(self.state.working_state))
                self.state.working_state = self.state.working_state[perm]
            if spec.target_carrier == MemoryCarrierType.EPISODIC and self.state.episodic_keys:
                perm = np.random.permutation(len(self.state.episodic_keys))
                self.state.episodic_keys = [self.state.episodic_keys[i] for i in perm]
                self.state.episodic_values = [self.state.episodic_values[i] for i in perm]

        elif spec.intervention_type == InterventionType.TRANSPLANT_CARRIER:
            if spec.donor_state is not None:
                self.state.working_state = spec.donor_state.copy()
            else:
                raise ValueError("Transplant intervention requires non-null donor_state")

        elif spec.intervention_type == InterventionType.RECONSTRUCT_CARRIER:
            # Reconstruct working memory from structural graph projection
            graph_diag = np.diag(self.state.structural_graph)
            if len(graph_diag) == self.d_working:
                reconstructed = np.tanh(np.dot(self.W_recon, graph_diag))
                self.state.working_state = reconstructed.astype(np.float32)

        self.state.intervention_log.append(log_entry)

    def get_combined_vector(self) -> np.ndarray:
        """Returns consolidated representation for world model conditioning."""
        # Working state: (d_working,), structural mean: (1,), episodic count: (1,)
        struct_summary = np.array([np.mean(self.state.structural_graph)], dtype=np.float32)
        ep_summary = np.array([float(len(self.state.episodic_keys)) / max(1, self.max_episodes)], dtype=np.float32)
        return np.concatenate([self.state.working_state, struct_summary, ep_summary], axis=0)
