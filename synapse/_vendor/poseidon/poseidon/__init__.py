"""Supermix Poseidon: explicitly bounded, locally trained research system."""
__version__ = "3.14.0"

from .provenance import capture_sources as _capture_sources
from .cognitive_agent import CognitiveAgent, CognitiveDecision
from .oracle import OracleConfig, OracleEnsemble, OracleRuntime
from .mnemosyne import MNEMOSYNE_MODES, MnemosyneConfig, MnemosyneMemory
from .horizon_x import HorizonXConfig, HorizonXPlanner
from .adapt import AdaptConfig, ContinualAdapter, ResidualPolicyAdapter
from .worldlab import POMDPConfig, POMDPTidePool, StepResult, WorldBridgeProtocol
from .neuro import NeuroConfig, NeuromodulatedRouter
from .active_inference import ActiveInferenceConfig, ActiveInferenceEvaluator, HomeostaticTarget
from .dream import DreamConfig, DreamConsolidator, DreamSummary
from .metacognition import MetacognitionConfig, MetacognitiveMonitor, MetacognitiveState
from .guardrails import FormalSafetyGuard, GuardrailAudit, InvariantResult
from .information_seeking import EpistemicInformationProber, CandidateAdapterSession, RetentionAndAdaptationAuditor
from .external_adapter import (
    ExternalAdapterIdentity,
    AdapterHandshake,
    ExternalTickMessage,
    MemoryInfluenceTelemetry,
    ExternalActionResponse,
    ExternalAdapterBridge,
    Compact3DReplayVerifier,
)
from .nexus_omega import (
    NexusOmegaAgent,
    POMDPPhaseWorld,
    AutonomousDiscoveryEngine,
    verify_discovery_receipt,
    FactoredCausalWorldModel,
    CausalEFEPlanner,
    FactoredState,
    CausalEFEDecision,
    MacroIntent,
    ImaginedTrajectory,
    HierarchicalEFEDecision,
    TrajectoryDiffusionImaginer,
    HierarchicalEFEPlanner,
    QuantumCausalState,
    HarmonicSpectralProfile,
    SynthesizedOpcode,
    ThermodynamicState,
    DreamConsolidationReceipt,
    UltimaDecision,
    BioharmonicResonator,
    QuantumCausalKernel,
    MetamorphicBytecodeWeaver,
    LTLSafetyShield,
    AutopoiesisThermodynamicField,
    MorpheusDreamEngine,
    CognitiveMonologue,
    ApexDecision,
    NonstationaryPhaseFieldResonator,
    NonstationaryPhaseState,
    ContrastiveCausalMemoryFabric,
    ContrastiveMemoryDecision,
    ApexReasoningEngine,
    NexusApexEngine,
    POMDPPhaseWorldV2,
    POMDPPhaseWorldV3,
)
from .hypothesis_engine import (
    ScientificHypothesis,
    CrucialExperiment,
    HypothesisFalsificationReceipt,
    FalsificationOutcome,
    HypothesisEngine,
)
from .proof_engine import (
    ProofStep,
    ProofCertificate,
    ProofEngine,
    verify_proof_step,
    verify_algebraic_identity,
)
from .creativity_engine import (
    ConceptualMatrix,
    BisociativeSynthesis,
    TRIZResolution,
    InventiveInventionDossier,
    CreativityEngine,
    CANONICAL_MATRICES,
    TRIZ_PRINCIPLES,
)
from .athenaeum import (
    ENode,
    EGraph,
    RewriteRule,
    EqualitySaturationEngine,
    MCTSNode,
    ProcessRewardModel,
    TestTimeComputeEngine,
    SymbolicRegressionSINDy,
    AthenaeumEngine,
)
from .hephaestus import (
    LambdaAbstraction,
    WakeSleepLibraryLearner,
    GFlowNetSampler,
    KoopmanEDMDLinearizer,
    BuchbergerGroebnerProver,
    HephaestusEngine,
)
from .hyperborea import (
    SheafCohomologyEngine,
    PortHamiltonianDiracEngine,
    RenormalizationGroupEngine,
    TopologicalAnyonBraidingEngine,
    HyperboreaEngine,
)
from .neurograft import (
    HeterogeneousConnectomeGraph,
    LiquidTimeConstantEngine,
    TriMonoaminergicEngine,
    CrossSourceGraftRouter,
    VirtualConnectomeLesionAuditor,
    NeurograftEngine,
)
from .aether import (
    GaugeFieldHolonomyEngine,
    NonEquilibriumThermodynamicsEngine,
    InformationGeometryFisherEngine,
    SymplecticContactMechanicsEngine,
    CategoricalOpenGameLensEngine,
    AetherEngine,
)
from .chronos import (
    BeaconPulse,
    ChronosBeaconClock,
    GhostTraceAuditor,
    CausalDynamicalTriangulationsEngine,
    NoncommutativeSpectralGeometryEngine,
    TropicalAlgebraicGeometryEngine,
    StiefelGrassmannianOptimizationEngine,
    CoalgebraicBisimulationEngine,
    ChronosEngine,
)
from .noether import (
    NoetherSymmetryEngine,
    ModularTensorCategoryEngine,
    CliffordSpacetimeAlgebraEngine,
    BatalinVilkoviskyEngine,
    ActiveMatterEngine,
    NoetherEngine,
)
from .cartan import (
    CartanExteriorSystemEngine,
    NonHermitianPTEngine,
    ExceptionalJordanAlgebraEngine,
    HolographicEntanglementEngine,
    MorphogeneticTuringEngine,
    CartanEngine,
)
from .tether_engine import (
    TetherContinuousReliabilityEngine,
    CounterfactualCovariationInvarianceEngine,
    CausalAttributionSaliencyEngine,
    ZeroShotCorruptionDetector,
    EvidenceCalibratedFallbackGate,
    TetherEngine,
)
from .research.tether_gate import (
    TetherReliabilityGate,
    TetherOracleAdapter,
)
from .faultline_engine import (
    FaultlineDiscreteEngine,
    ContinuousJointErrorEngine,
    ParityBenchmark,
    FaultlineEngine,
)
from .research.faultline_adapter import (
    FaultlineOracleAdapter,
)
from .crosswire_engine import (
    CandidateMapping,
    CandidateSpace,
    VersionSpaceTracker,
    ActiveDiagnosticProber,
    InterfaceContractSpec,
    InterfaceContractVerifier,
    ClosedLoopRuntimeShield,
    CrosswireEngine,
)
from .research.crosswire_adapter import (
    CrosswireAdapterVerifier,
    discretize_tidepool_obs_16d,
)
from .neurograft_engine import (
    SuccessorRepresentationEngine,
    SharpWaveRippleReplay,
    GraphLaplacianTopologicalEngine,
    TrainableGraphCircuitRouter,
    NeurograftEngine as NeurograftEngineV2,
)
from .research.successor_memory_adapter import (
    SuccessorMemoryAdapter,
    discretize_obs_to_spatial_state,
)
from .poincare_engine import (
    PoincareReturnMapEngine,
    HyperbolicPoincareDiskEngine,
    PoincareDualityMorseEngine,
    RicciFlowPerelmanEngine,
    UnifiedQuartetCognitiveEngine,
    PoincareEngine,
)
from .research.unified_quartet_coordinator import (
    QuartetCognitiveDecision,
    UnifiedQuartetCoordinator,
)

_SOURCE_SNAPSHOT = _capture_sources()


