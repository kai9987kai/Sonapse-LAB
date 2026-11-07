"""Sonapse NeuroWave-X2: Advanced Neural Communication Research Platform.

Additive, default-OFF inter-region neural communication fabric combining:
- Topology-aware spectral neural communication (Graph Laplacian eigenmodes)
- Structured immutable neural packet protocol with wire-cost accounting
- Phase-aware Communication-Through-Coherence (CTC)
- Hybrid biological and learned orthogonal bases
- Learned context-dependent protocol selection
- Causally validated transactional structural plasticity
- Independent NumPy reference transport for scientific conformance
- Zero-gain identity guarantee (gamma=0.0 default)
"""
from __future__ import annotations

from .config import NeuroWaveX2Config
from .packets import NeuralPacket, PacketValidationError
from .reference_transport import ReferenceTransport
from .transport import PyTorchTransport
from .conformance import run_conformance_suite, ConformanceVerificationResult
from .spectral_basis import SpectralBasisManager, compute_eigenbasis, compute_subspace_drift
from .hybrid_basis import HybridBasis
from .phase_channel import PhaseCoherenceAdapter, WaveChannel
from .protocol_registry import PROTOCOLS, ProtocolSpec, get_protocol_spec
from .protocol_selector import ProtocolSelector
from .compute_profiler import ComputeProfiler
from .resource_budget import ResourceBudgetController
from .information_metrics import (
    compute_effective_rank,
    compute_message_entropy_estimate,
    compute_communication_sparsity,
    compute_spectral_energy_concentration,
)
from .stress_tasks import (
    StrictSeparationBenchmark,
    CompositionalInformationBenchmark,
    CommunicationStressHarness,
)
from .ablations import CausalAblationController, evaluate_route_causal_importance
from .topology_controller import CommunicationTopologyController, TopologyInvariantError
from .topology_transaction import TopologyTransaction, TopologyTransactionError
from .optimizer_migration import migrate_optimizer_state_for_edges
from .causal_rewiring import CausalRewiringEngine
from .communication_development import CommunicationDevelopmentEngine, DEVELOPMENTAL_LEVELS
from .multiscale_channels import MultiTimescaleChannelManager
from .receipts import ResearchReceipt, generate_experiment_receipt
from .replay import ReplayRecorder, verify_replay_consistency
from .promotion import FailClosedPromotionGate, PromotionDecision
from .architecture_search import CommunicationMutation, ParetoFrontierTracker, ParetoCandidate
from .experiment_runner import run_benchmark_arm, run_full_benchmark_matrix, BENCHMARK_ARMS
from .telemetry import NeuroWaveTelemetryAggregator
from .fabric import NeuroWaveX2Fabric, NeuroWaveStepOutput

__all__ = [
    "NeuroWaveX2Config",
    "NeuralPacket",
    "PacketValidationError",
    "ReferenceTransport",
    "PyTorchTransport",
    "run_conformance_suite",
    "ConformanceVerificationResult",
    "SpectralBasisManager",
    "compute_eigenbasis",
    "compute_subspace_drift",
    "HybridBasis",
    "PhaseCoherenceAdapter",
    "WaveChannel",
    "PROTOCOLS",
    "ProtocolSpec",
    "get_protocol_spec",
    "ProtocolSelector",
    "ComputeProfiler",
    "ResourceBudgetController",
    "compute_effective_rank",
    "compute_message_entropy_estimate",
    "compute_communication_sparsity",
    "compute_spectral_energy_concentration",
    "StrictSeparationBenchmark",
    "CompositionalInformationBenchmark",
    "CommunicationStressHarness",
    "CausalAblationController",
    "evaluate_route_causal_importance",
    "CommunicationTopologyController",
    "TopologyInvariantError",
    "TopologyTransaction",
    "TopologyTransactionError",
    "migrate_optimizer_state_for_edges",
    "CausalRewiringEngine",
    "CommunicationDevelopmentEngine",
    "DEVELOPMENTAL_LEVELS",
    "MultiTimescaleChannelManager",
    "ResearchReceipt",
    "generate_experiment_receipt",
    "ReplayRecorder",
    "verify_replay_consistency",
    "FailClosedPromotionGate",
    "PromotionDecision",
    "CommunicationMutation",
    "ParetoFrontierTracker",
    "ParetoCandidate",
    "run_benchmark_arm",
    "run_full_benchmark_matrix",
    "BENCHMARK_ARMS",
    "NeuroWaveTelemetryAggregator",
    "NeuroWaveX2Fabric",
    "NeuroWaveStepOutput",
]
