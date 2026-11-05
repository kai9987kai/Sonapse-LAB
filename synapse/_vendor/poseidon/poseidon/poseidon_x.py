"""POSEIDON-X: Unified Apex Cognitive Operating System.

Synthesizes the complete portfolio breakthrough lineages:
1. HyperAgent (Causal SCM, Quantum superposition, Morphogenesis, Phase-plane, Ghost monologue, Morpheus dream replay)
2. Clifford Geometric Algebra Cl(3,0) Kinematics & Rotor dynamics (geometric_algebra.py)
3. Autopoietic Living System Homeostasis, PAD Affect & Intrinsic Curiosity (autopoiesis.py)
4. Neural-Symbolic Inductive Hypergraph Theorem Synthesizer (hypergraph_discovery.py)
5. Tessera Formal LTL Safety Contract Prover & Invariant Shield (tessera_prover.py)
6. Topological Persistent Homology Memory Carrier Observatory (memory_observatory.py)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Sequence

from .autopoiesis import AutopoieticAudit, AutopoieticEngine, PADAffectiveState
from .causal_scm import StructuralCausalModel
from .core import CoreRuntime
from .geometric_algebra import GeometricKinematicsEngine, Multivector3D, Rotor3D
from .hypergraph_discovery import InductiveHypergraphEngine, SymbolicTheorem
from .hyper_agent import HyperAgent, HyperDecision
from .memory_observatory import MemoryCarrierObservatory, TopologicalManifoldAudit
from .tessera_prover import SafetyProofReceipt, TesseraProver
from .world import ACTIONS, N_ACTIONS, OBS_SIZE


@dataclass
class PoseidonXDecision:
    """Unified apex decision containing full multi-plane telemetry."""
    action: int
    action_name: str
    base_action: int
    shield_intervened: bool
    quantum_coherence: float
    causal_effect: float
    lyapunov_drift: float
    separatrix_distance: float
    rotor_orientation: tuple[float, float, float, float]  # [s, b12, b23, b31]
    affect: PADAffectiveState
    negentropy_flux: float
    curiosity_drive: float
    ratified_theorems: int
    topology_betti_0: int
    topology_betti_1: int
    monologue: str
    certificate_sha256: str = ""

    def __int__(self) -> int:
        return self.action

    def __index__(self) -> int:
        return self.action

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "action_name": self.action_name,
            "base_action": self.base_action,
            "shield_intervened": self.shield_intervened,
            "quantum_coherence": round(self.quantum_coherence, 4),
            "causal_effect": round(self.causal_effect, 6),
            "lyapunov_drift": round(self.lyapunov_drift, 5),
            "separatrix_distance": round(self.separatrix_distance, 4),
            "rotor_orientation": [round(x, 4) for x in self.rotor_orientation],
            "affect": self.affect.to_dict(),
            "negentropy_flux": round(self.negentropy_flux, 4),
            "curiosity_drive": round(self.curiosity_drive, 4),
            "ratified_theorems": self.ratified_theorems,
            "topology_betti_0": self.topology_betti_0,
            "topology_betti_1": self.topology_betti_1,
            "monologue": self.monologue,
            "certificate_sha256": self.certificate_sha256,
        }


class PoseidonXKernel:
    """Unified Pinnacle Cognitive Operating System synthesizing all 22 portfolio lineages."""

    def __init__(self, core: CoreRuntime | None = None):
        # 1. HyperAgent Apex substrate
        self.hyper_agent = HyperAgent(core=core)

        # 2. Clifford Geometric Algebra Engine
        self.kinematics = GeometricKinematicsEngine()

        # 3. Autopoietic Affective & Curiosity Engine
        self.autopoiesis = AutopoieticEngine()

        # 4. Neural-Symbolic Inductive Hypergraph Engine
        self.hypergraph = InductiveHypergraphEngine()

        # 5. Tessera Formal LTL Safety Prover
        self.prover = TesseraProver()

        # 6. Topological Memory Observatory
        self.observatory = MemoryCarrierObservatory()

        self.step_history: list[dict[str, Any]] = []
        self.trajectory_vitals: list[list[float]] = []

    def reset(self) -> None:
        self.hyper_agent.reset()
        self.kinematics = GeometricKinematicsEngine()
        self.autopoiesis = AutopoieticEngine()
        self.step_history.clear()
        self.trajectory_vitals.clear()

    def decide(self, obs: list[float]) -> PoseidonXDecision:
        """Unified decision cycle balancing causal, quantum, autopoietic, and formal invariant planes."""
        # A. Query HyperAgent multi-plane consensus
        hyper_dec: HyperDecision = self.hyper_agent.act(obs)
        candidate_action = hyper_dec.action

        # B. Autopoietic Affective & Curiosity Update
        auto_audit: AutopoieticAudit = self.autopoiesis.update(
            current_obs=obs,
            separatrix_distance=hyper_dec.separatrix_distance,
        )

        # Modulate action with epistemic curiosity if safe
        if auto_audit.curiosity_drive > 0.6 and hyper_dec.separatrix_distance > 0.4:
            if candidate_action == 0:  # Rest -> Explore for curiosity
                candidate_action = 4

        # C. Pass candidate action through Tessera Formal Invariant Shield
        shielded_action, proof = self.prover.shield_action(obs, candidate_action)

        # D. Geometric Kinematics Update: apply rotor-based step
        thrust = 8.0 if shielded_action == 4 else (14.0 if shielded_action == 5 else 0.0)
        torque = 0.5 if shielded_action == 5 else 0.0
        self.kinematics.apply_thrust_and_torque(thrust_forward=thrust, torque_yaw_b12=torque)

        # E. Ingest transition for inductive theorem discovery
        self.trajectory_vitals.append(obs[:4])
        exp_entry = {
            "action": shielded_action,
            "stamina": obs[3] if len(obs) > 3 else 1.0,
            "next_stamina": obs[3] if len(obs) > 3 else 1.0,
            "energy": obs[1] if len(obs) > 1 else 1.0,
            "in_patch": bool(obs[13] > 0.5) if len(obs) > 13 else False,
            "is_submerged": bool(obs[4] > 0.5) if len(obs) > 4 else False,
            "drag": 0.8 if (len(obs) > 4 and obs[4] > 0.5) else 0.02,
        }
        self.step_history.append(exp_entry)
        if len(self.step_history) % 10 == 0:
            self.hypergraph.ingest_trajectory(self.step_history[-20:])

        ratified_count = sum(1 for t in self.hypergraph.discovered_theorems if t.ratified)

        # F. Persistent Homology Analysis on trajectory point cloud
        topo_audit: TopologicalManifoldAudit = self.observatory.analyze_trajectory_topology(self.trajectory_vitals)

        # Build rotor orientation representation [s, b12, b23, b31]
        r_mv = self.kinematics.rotor.mv
        rotor_tuple = (r_mv.s, r_mv.b12, r_mv.b23, r_mv.b31)

        action_name = ACTIONS[shielded_action] if shielded_action < len(ACTIONS) else "unknown"

        return PoseidonXDecision(
            action=shielded_action,
            action_name=action_name,
            base_action=hyper_dec.base_action,
            shield_intervened=proof.shield_triggered,
            quantum_coherence=hyper_dec.quantum_coherence,
            causal_effect=hyper_dec.causal_effect,
            lyapunov_drift=hyper_dec.lyapunov_drift,
            separatrix_distance=hyper_dec.separatrix_distance,
            rotor_orientation=rotor_tuple,
            affect=auto_audit.affect,
            negentropy_flux=auto_audit.negentropy_flux,
            curiosity_drive=auto_audit.curiosity_drive,
            ratified_theorems=ratified_count,
            topology_betti_0=topo_audit.betti_0,
            topology_betti_1=topo_audit.betti_1,
            monologue=hyper_dec.monologue,
            certificate_sha256=proof.certificate_sha256,
        )

    act = decide
