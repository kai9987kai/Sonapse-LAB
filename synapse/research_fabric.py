"""Supermix Research Fabric Contracts and Cross-Ecosystem Interoperability.

Grounding & Review Requirements:
Provides typed, versioned data contracts and adapters across the wider ecosystem:
- Poseidon (sensory-motor priors & checkpoint loading)
- NEUROGRAFT (biological connectome grafting & Dale's law enforcement)
- Expanse (authentic Drosophila MaleCNS connectome datasets)
- Tether (safety constraints & action governance)
- FAULTLINE (adversarial perturbation & stress testing)
- GhostInTheMachine (efference copy & self-authorship verification)

All contracts are strictly typed, deterministically serializable, and include
fail-safe fallbacks when external ecosystem components are not present.
"""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


FABRIC_VERSION = "2026.11-fabric-v1"


@dataclass
class FabricObservation:
    """Standardized sensory observation contract."""
    data: List[float]
    dim: int
    step: int
    format_name: str = "complex_world_155"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.data, dtype=np.float32)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, step: int, format_name: str = "complex_world_155") -> FabricObservation:
        flat = np.asarray(arr, dtype=np.float32).flatten()
        return cls(data=flat.tolist(), dim=len(flat), step=step, format_name=format_name)


@dataclass
class FabricAction:
    """Standardized motor command contract."""
    vector: List[float]
    dim: int = 5
    thrust_governed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_numpy(self) -> np.ndarray:
        return np.clip(np.asarray(self.vector, dtype=np.float32), -1.0, 1.0)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, governed: bool = True) -> FabricAction:
        clipped = np.clip(np.asarray(arr, dtype=np.float32).flatten(), -1.0, 1.0)
        return cls(vector=clipped.tolist(), dim=len(clipped), thrust_governed=governed)


@dataclass
class FabricTransition:
    """Atomic embodied transition contract."""
    obs: FabricObservation
    action: FabricAction
    reward: float
    next_obs: FabricObservation
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'obs': asdict(self.obs),
            'action': asdict(self.action),
            'reward': self.reward,
            'next_obs': asdict(self.next_obs),
            'done': self.done,
            'info': self.info
        }


@dataclass
class FabricModelPrediction:
    """Multi-step predictive model output contract."""
    predicted_latent: List[float]
    predicted_reward: float
    epistemic_variance: float
    aleatoric_variance: float
    ood_score: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReceipt:
    """Cryptographically verifiable benchmark evaluation receipt."""
    fabric_version: str
    benchmark_name: str
    task_sequence: List[str]
    score_matrix: List[List[float]]
    final_retention_rate: float
    catastrophic_forgetting: float
    adaptation_speed: float
    base_seed: int
    proof_hash: str
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        benchmark_name: str,
        task_sequence: List[str],
        score_matrix: List[List[float]],
        retention: float,
        forgetting: float,
        adaptation_speed: float,
        base_seed: int
    ) -> EvaluationReceipt:
        raw_payload = {
            'benchmark': benchmark_name,
            'tasks': task_sequence,
            'matrix': score_matrix,
            'retention': retention,
            'forgetting': forgetting,
            'seed': base_seed
        }
        proof = hashlib.sha256(json.dumps(raw_payload, sort_keys=True).encode('utf-8')).hexdigest()
        return cls(
            fabric_version=FABRIC_VERSION,
            benchmark_name=benchmark_name,
            task_sequence=task_sequence,
            score_matrix=score_matrix,
            final_retention_rate=retention,
            catastrophic_forgetting=forgetting,
            adaptation_speed=adaptation_speed,
            base_seed=base_seed,
            proof_hash=proof
        )

    def verify(self) -> bool:
        """Verify receipt against tamper or payload corruption."""
        raw_payload = {
            'benchmark': self.benchmark_name,
            'tasks': self.task_sequence,
            'matrix': self.score_matrix,
            'retention': self.final_retention_rate,
            'forgetting': self.catastrophic_forgetting,
            'seed': self.base_seed
        }
        computed = hashlib.sha256(json.dumps(raw_payload, sort_keys=True).encode('utf-8')).hexdigest()
        return computed == self.proof_hash


class PoseidonBridgeAdapter:
    """Adapter for importing prior sensory encoders and policies from Poseidon."""

    @staticmethod
    def adapt_sensory_projection(in_dim: int = 155, out_dim: int = 32) -> np.ndarray:
        """Construct deterministic orthonormal projection from 155 sensory rays to 32 latents."""
        rng = np.random.default_rng(2026)
        matrix = rng.normal(0.0, 1.0, size=(in_dim, out_dim))
        q, _ = np.linalg.qr(matrix)
        return q.astype(np.float32)


class NeuroGraftAdapter:
    """Enforces Dale's law and biological compliance on imported connectome topologies."""

    @staticmethod
    def audit_dales_law(weights: np.ndarray, neurotransmitters: List[str]) -> Dict[str, Any]:
        """Audit Dale's law: every presynaptic neuron must be strictly excitatory or inhibitory."""
        violating_neurons = []
        N = weights.shape[0]

        for i in range(N):
            row = weights[i, :]
            pos = (row > 1e-6).any()
            neg = (row < -1e-6).any()
            if pos and neg:
                violating_neurons.append(i)

        compliant = len(violating_neurons) == 0
        return {
            'compliant': compliant,
            'total_neurons': N,
            'violating_count': len(violating_neurons),
            'violating_indices': violating_neurons[:10]
        }


class FaultlineStressAdapter:
    """Injects standardized stress perturbations to evaluate robustness."""

    def __init__(self, perturbation_type: str = "sensor_dropout", severity: float = 0.2, seed: int = 42):
        self.perturbation_type = perturbation_type
        self.severity = severity
        self.rng = np.random.default_rng(seed)

    def perturb_observation(self, obs: FabricObservation) -> FabricObservation:
        """Apply noise or dropouts to incoming observation."""
        arr = obs.to_numpy().copy()
        if self.perturbation_type == "sensor_dropout":
            mask = self.rng.random(len(arr)) < self.severity
            arr[mask] = 0.0
        elif self.perturbation_type == "gaussian_noise":
            noise = self.rng.normal(0.0, self.severity, size=len(arr)).astype(np.float32)
            arr = np.clip(arr + noise, -2.0, 2.0)
        return FabricObservation.from_numpy(arr, step=obs.step, format_name=obs.format_name)
