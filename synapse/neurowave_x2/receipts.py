"""Reproducible Integrity Receipts for Sonapse NeuroWave-X2.

Generates SHA-256 provenance receipts binding configuration, topology,
spectral basis, model parameters, seed, evaluation results, and resource profiles.

DOCUMENTATION NOTICE:
A SHA-256 receipt guarantees numerical integrity and change detection across runs;
it does not constitute an external cryptographic authenticity proof unless paired with a digital signature.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ResearchReceipt:
    """Immutable experiment audit receipt."""

    receipt_id: str
    timestamp: str
    git_commit: str
    config_hash: str
    topology_hash: str
    basis_hash: str
    seed: int
    environment_name: str
    evaluation_return: float
    evaluation_success: bool
    total_wire_bits: int
    total_analytical_flops: float
    status: str  # 'confirmatory', 'exploratory', 'replicated'
    metrics: Dict[str, Any] = field(default_factory=dict)
    sha256_hash: str = field(init=False)

    def __post_init__(self) -> None:
        hasher = hashlib.sha256()
        payload = {
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "topology_hash": self.topology_hash,
            "basis_hash": self.basis_hash,
            "seed": self.seed,
            "environment_name": self.environment_name,
            "evaluation_return": float(self.evaluation_return),
            "evaluation_success": bool(self.evaluation_success),
            "total_wire_bits": int(self.total_wire_bits),
            "total_analytical_flops": float(self.total_analytical_flops),
            "status": self.status,
            "metrics": self.metrics,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        hasher.update(serialized)
        object.__setattr__(self, "sha256_hash", hasher.hexdigest())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "topology_hash": self.topology_hash,
            "basis_hash": self.basis_hash,
            "seed": self.seed,
            "environment_name": self.environment_name,
            "evaluation_return": self.evaluation_return,
            "evaluation_success": self.evaluation_success,
            "total_wire_bits": self.total_wire_bits,
            "total_analytical_flops": self.total_analytical_flops,
            "status": self.status,
            "metrics": self.metrics,
            "sha256_hash": self.sha256_hash,
        }


def generate_experiment_receipt(
    config_dict: Dict[str, Any],
    topology_routes: List[Tuple[int, int]],
    basis_version: int,
    seed: int,
    env_name: str,
    eval_return: float,
    eval_success: bool,
    wire_bits: int,
    flops: float,
    metrics: Optional[Dict[str, Any]] = None,
    status: str = "exploratory",
    git_commit: str = "HEAD",
) -> ResearchReceipt:
    """Generates a verifiable SHA-256 research run receipt."""
    cfg_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:16]
    topo_hash = hashlib.sha256(str(sorted(topology_routes)).encode()).hexdigest()[:16]
    basis_hash = hashlib.sha256(f"basis_v{basis_version}".encode()).hexdigest()[:16]
    receipt_id = f"rcpt_{int(time.time())}_{cfg_hash[:8]}"

    return ResearchReceipt(
        receipt_id=receipt_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=git_commit,
        config_hash=cfg_hash,
        topology_hash=topo_hash,
        basis_hash=basis_hash,
        seed=int(seed),
        environment_name=str(env_name),
        evaluation_return=float(eval_return),
        evaluation_success=bool(eval_success),
        total_wire_bits=int(wire_bits),
        total_analytical_flops=float(flops),
        status=status,
        metrics=metrics or {},
    )
