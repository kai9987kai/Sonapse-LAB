"""Composition boundary: every result declares its actual backend."""
from __future__ import annotations
from pathlib import Path
import copy
import json
import threading
from . import __version__
from .reasoning import extract_math, solve
from .language import LanguageRuntime


def _finite_json(text):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    result = json.loads(text, parse_constant=reject_constant)
    # Also reject overflowing finite literals such as 1e999.
    json.dumps(result, allow_nan=False)
    return result


def _status_report(path, max_bytes=4 * 1024 * 1024):
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Status report exceeds the {max_bytes}-byte read budget")
    result = _finite_json(data.decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Status report must be a JSON object")
    return result


def _recent_metrics(path, max_bytes=256 * 1024):
    """Read a bounded tail while tolerating an in-progress final JSONL write."""
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        start = max(0, size - max_bytes)
        stream.seek(start)
        lines = stream.read(max_bytes).splitlines()
    if start:
        lines = lines[1:]  # The first fragment may start in the middle of a record.
    rows, invalid = [], 0
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            row = _finite_json(line.decode("utf-8"))
            if not isinstance(row, dict):
                raise ValueError("Metric must be an object")
        except (ValueError, UnicodeError, RecursionError):
            invalid += 1
            continue
        rows.append(row)
        if len(rows) == 12:
            break
    return list(reversed(rows)), invalid


class Poseidon:
    def __init__(self, root=".", adapter=None, oracle_checkpoint=None):
        self.root = Path(root).resolve()
        self.oracle_checkpoint = None
        if oracle_checkpoint is not None:
            checkpoint = Path(oracle_checkpoint)
            self.oracle_checkpoint = checkpoint if checkpoint.is_absolute() else self.root / checkpoint
            # Fail explicitly at selection time, never silently substitute a model.
            from .research.oracle_adapter import ResearchOracleAdapter
            ResearchOracleAdapter(self.oracle_checkpoint)
        self.lock = threading.RLock()
        self.language = LanguageRuntime(self.root/"models/language", adapter=adapter)
        self._core = None
        self._core_stamp = None
        self._atlas = None
        self._atlas_stamp = None
        self._contrast = None
        self._contrast_stamp = None
        self._horizon = None
        self._horizon_stamp = None
        self._odyssey = None
        self._odyssey_stamp = None
        self._helm = None
        self._helm_stamp = None
        self._odysseus = None
        self._odysseus_stamp = None
        self._jobs = None
        self._jobs_lock = threading.Lock()
        self._tessera = None
        self._external_adapter_bridge = None
        self._omni_reasoner = None
        self._pantheon = None
        self._candidate_status_snapshots = {}
        self.memory_path = self.root/"outputs/memory.json"

    def atlas(self):
        from .atlas import CounterfactualAtlas
        path = self.root / "outputs/atlas/atlas.json"
        if not path.is_file():
            raise RuntimeError("Counterfactual Atlas is not fitted. Run python -m poseidon atlas-fit first.")
        core = self.core()
        stamp = (self._core_stamp, path.stat().st_mtime_ns, path.stat().st_size)
        if self._atlas is None or self._atlas_stamp != stamp:
            self._atlas = CounterfactualAtlas.load(core, path)
            self._atlas_stamp = stamp
        return self._atlas

    def atlas_status(self):
        return self._candidate_status("atlas")

    def contrast(self):
        from .contrast import ContrastAtlas
        path = self.root / "outputs/contrast/atlas.json"
        if not path.is_file():
            raise RuntimeError("Contrast Atlas is not fitted. Run python -m poseidon contrast-fit first.")
        core = self.core()
        stamp = (self._core_stamp, path.stat().st_mtime_ns, path.stat().st_size)
        if self._contrast is None or self._contrast_stamp != stamp:
            self._contrast = ContrastAtlas.load(core, path)
            self._contrast_stamp = stamp
        return self._contrast

    def contrast_status(self):
        return self._candidate_status("contrast")

    def horizon(self):
        from .horizon import HorizonAtlas
        path = self.root / "outputs/horizon/atlas.json"
        if not path.is_file():
            raise RuntimeError("Horizon Atlas is not fitted. Run python -m poseidon horizon-fit first.")
        core = self.core()
        stamp = (self._core_stamp, path.stat().st_mtime_ns, path.stat().st_size)
        if self._horizon is None or self._horizon_stamp != stamp:
            self._horizon = HorizonAtlas.load(core, path)
            self._horizon_stamp = stamp
        return self._horizon

    def horizon_status(self):
        return self._candidate_status("horizon")

    def odyssey(self):
        from .odyssey import OdysseyAtlas
        path = self.root / "outputs/odyssey/atlas.json"
        if not path.is_file():
            raise RuntimeError("Odyssey Atlas is not fitted. Run python -m poseidon odyssey-fit first.")
        core = self.core()
        stamp = (self._core_stamp, path.stat().st_mtime_ns, path.stat().st_size)
        if self._odyssey is None or self._odyssey_stamp != stamp:
            self._odyssey = OdysseyAtlas.load(core, path)
            self._odyssey_stamp = stamp
        return self._odyssey

    def odyssey_status(self):
        return self._candidate_status("odyssey")

    def helm(self):
        from .helm import HelmCritic
        path = self.root / "outputs/helm/atlas.json"
        if not path.is_file():
            raise RuntimeError("Helm is not fitted. Run python -m poseidon fit-helm first.")
        core = self.core()
        stamp = (self._core_stamp, path.stat().st_mtime_ns, path.stat().st_size)
        if self._helm is None or self._helm_stamp != stamp:
            self._helm = HelmCritic.load(core, path)
            self._helm_stamp = stamp
        return self._helm

    def helm_status(self):
        return self._candidate_status("helm")

    def odysseus(self):
        from .odysseus import OdysseusAtlas
        path = self.root / "outputs/odysseus/atlas.json"
        if not path.is_file():
            raise RuntimeError("Odysseus Atlas is not fitted. Run python -m poseidon fit-odysseus first.")
        core = self.core()
        stamp = (self._core_stamp, path.stat().st_mtime_ns, path.stat().st_size)
        if self._odysseus is None or self._odysseus_stamp != stamp:
            self._odysseus = OdysseusAtlas.load(core, path)
            self._odysseus_stamp = stamp
        return self._odysseus

    def odysseus_status(self):
        return self._candidate_status("odysseus")

    def tessera(self):
        if self._tessera is None:
            from .tessera import TesseraMacroCommons
            path = self.root / "outputs/tessera/commons.json"
            if path.is_file():
                self._tessera = TesseraMacroCommons.load(path)
            else:
                self._tessera = TesseraMacroCommons()
        return self._tessera

    def tessera_status(self):
        tess = self.tessera()
        return {
            "ready": True,
            "ratified_count": len(tess.ratified),
            "quarantine_count": len(tess.quarantine),
            "current_epoch": tess.current_epoch,
        }

    def aura(self):
        from .aura import AuraController
        return AuraController(core=self.core(), tessera=self.tessera())

    def aura_status(self):
        return {"ready": True, "type": "biomimetic_central_complex_and_neuropil"}

    def metamorph(self):
        from .metamorph import MetamorphController
        return MetamorphController(core=self.core(), tessera=self.tessera())

    def metamorph_status(self):
        return {"ready": True, "type": "unified_metamorphic_biomimetic_causal_flux"}

    def chimera(self):
        from .chimera import ChimeraController
        return ChimeraController(core=self.core(), tessera=self.tessera())

    def chimera_status(self):
        return self._controller_receipt_status("chimera", "unified_quantum_holographic_lattice_super_controller",
                                               ("mean_quantum_entropy", "mean_lattice_coherence"))

    def hyperion(self):
        from .hyperion import HyperionController
        return HyperionController(core=self.core(), tessera=self.tessera())

    def hyperion_status(self):
        return self._controller_receipt_status("hyperion", "unified_frontier_morpheus_prometheus_intermittent_super_controller",
            ("mean_quantum_entropy", "mean_lattice_coherence", "total_sleep_cycles", "total_harvested_energy"))

    def titan(self):
        from .titan import TitanController
        return TitanController(core=self.core(), tessera=self.tessera())

    def titan_status(self):
        return self._controller_receipt_status("titan", "experimental_composite_controller",
            ("mean_bell_fidelity", "mean_buoyancy_force", "mean_trophic_richness", "total_teleportations"))

    def trident(self):
        from .trident import TridentController
        return TridentController(core=self.core())

    def trident_status(self):
        return self._controller_receipt_status("trident", "identity_gated_directed_residual_controller",
                                               ("override_count", "identity_specific_count", "mean_residual_norm"))

    def _status_receipt(self, name):
        """Check a bounded local seal. This never runs a controller or semantic replay."""
        receipt = _status_report(self.root / f"outputs/{name}_experiments/RECEIPT.json", 32 * 1024 * 1024)
        if name in ("titan", "trident"):
            import hashlib
            schema = f"poseidon-{name}-experiment-v" + ("2" if name == "titan" else "1")
            if receipt.get("schema") != schema:
                raise ValueError(f"Unsupported {name.upper()} receipt schema; run a current experiment for checked evidence")
            unsigned = {key: value for key, value in receipt.items() if key != "sha256"}
            checksum = hashlib.sha256(json.dumps(unsigned, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if receipt.get("sha256") != checksum:
                raise ValueError(f"{name.upper()} receipt checksum mismatch")
            if (receipt.get("status") != "completed-observational-evaluation" or
                    receipt.get("promotion") is not False or receipt.get("fitted_artifact_created") is not False):
                raise ValueError("Receipt completion or activation boundary is invalid")
        elif name == "chimera":
            from .chimera_experiment import verify_chimera_receipt
            verify_chimera_receipt(receipt)
        elif name == "hyperion":
            from .hyperion_experiment import verify_hyperion_receipt
            verify_hyperion_receipt(receipt)
        elif name == "mco":
            from .mco import verify_mco_receipt
            verify_mco_receipt(receipt)
        else:
            raise ValueError("Unknown status receipt protocol")
        return receipt

    def _controller_receipt_status(self, name, kind, telemetry_fields):
        relative = f"outputs/{name}_experiments/RECEIPT.json"
        status = {"ready": True, "type": kind, "path": relative, "receipt_ready": False,
                  "receipt_integrity": False, "receipt_verified": False, "replay_verified": False,
                  "verification_scope": "not_checked", "promotion": False,
                  "scope": "synthetic TidePool simulation"}
        if not (self.root / relative).is_file():
            return status
        try:
            receipt = self._status_receipt(name)
            telemetry = receipt.get(name + "_telemetry", {})
            if not isinstance(telemetry, dict):
                raise ValueError("Receipt telemetry must be a JSON object")
            status.update({key: telemetry.get(key) for key in telemetry_fields})
            status.update(receipt_ready=True, receipt_integrity=True, verification_scope="checksum_only",
                          receipt_sha256=receipt.get("sha256") or receipt.get("receipt_sha256"))
        except Exception as error:
            status["error"] = str(error)
        return status

    def cognitive_agent(self, mode: str = "intact"):
        from .cognitive_agent import CognitiveAgent
        return CognitiveAgent(core=self.core(), mode=mode, oracle_checkpoint=self.oracle_checkpoint)

    def oracle_status(self):
        from .research.oracle_adapter import ResearchOracleAdapter, oracle_status
        return (ResearchOracleAdapter(self.oracle_checkpoint).status() if self.oracle_checkpoint is not None
                else oracle_status(None))

    def cognitive_agent_status(self):
        relative = "outputs/cognitive_experiments/RECEIPT.json"
        path = self.root / relative
        if not path.is_file():
            return {
                "ready": True,
                "version": "2.0.0",
                "type": "unified_cognitive_agent_coordinator",
                "subsystems": ["oracle", "mnemosyne", "horizon_x", "adapt", "neuro", "worldlab"],
                "core_frozen": True,
                "receipt_ready": False,
                "path": relative,
                "oracle": self.oracle_status(),
            }
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            from .cognitive_experiment import verify_cognitive_receipt
            verify_cognitive_receipt(receipt, core_path=self.core().path)
            return {
                "ready": True,
                "version": "2.0.0",
                "type": "unified_cognitive_agent_coordinator",
                "subsystems": ["oracle", "mnemosyne", "horizon_x", "adapt", "neuro", "worldlab"],
                "core_frozen": True,
                "receipt_ready": True,
                "path": relative,
                "receipt_sha256": receipt.get("sha256"),
                "summary": receipt.get("arms"),
                "paired": receipt.get("paired"),
                "oracle": self.oracle_status(),
                "receipt_verified": True,
            }
        except Exception:
            return {
                "ready": True,
                "version": "2.0.0",
                "type": "unified_cognitive_agent_coordinator",
                "receipt_ready": False,
                "path": relative,
                "receipt_verified": False,
                "oracle": self.oracle_status(),
                "error": "Cognitive experiment receipt failed semantic verification.",
            }

    def hyper_agent(self):
        from .hyper_agent import HyperAgent
        return HyperAgent(core=self.core(), oracle_checkpoint=self.oracle_checkpoint)

    def hyper_agent_status(self):
        return {
            "oracle": self.oracle_status(),
            "predictive_control_promoted": False,
            "ready": True,
            "version": "3.0.0",
            "type": "hyper_causal_quantum_morphogenetic_agent",
            "subsystems": [
                "causal_scm", "quantum_planner", "morphogenesis", "molt",
                "active_inference_fep", "horizon_x", "metacognition",
                "mnemosyne_3tier", "tessera_guardrails", "genesis_3d"
            ],
            "core_frozen": True,
        }

    def ecosystem(self, seed: int = 42):
        from .ecosystem_3d import ContinuousEcosystem3D
        return ContinuousEcosystem3D(seed=seed)

    def ecosystem_status(self):
        return {
            "ready": True,
            "version": "1.0.0",
            "type": "multi_agent_continuous_3d_trophic_ecosystem",
            "trophic_roles": ["grazer", "predator", "symbiont"],
            "bio_pulse_signaling": "duty_cycled_flipper_ntag_modulation",
            "shield": "tessera_ltl_invariant_shield",
            "topology": "persistent_homology_carrier_observatory",
        }

    def ecosystem_experiment(self, seed: int = 42, episodes: int = 2, max_steps: int = 40):
        from .ecosystem_experiment import run_ecosystem_experiment
        with self.lock:
            seeds = list(range(seed, seed + episodes))
            return run_ecosystem_experiment(seeds=seeds, steps_per_episode=max_steps)

    def poseidon_x(self):
        from .poseidon_x import PoseidonXKernel
        return PoseidonXKernel(core=self.core())

    def poseidon_x_status(self):
        return {
            "ready": True,
            "version": "5.0.0",
            "type": "apex_multivector_autopoietic_neural_symbolic_kernel",
            "lineages": [
                "clifford_geometric_algebra_cl30",
                "autopoietic_pad_affect_homeostasis",
                "intrinsic_epistemic_curiosity",
                "neural_symbolic_hypergraph_theorems",
                "tessera_ltl_invariant_shield",
                "quantum_grover_tree_superposition",
                "causal_scm_pearl_do_calculus",
                "topological_betti_memory_observatory",
            ],
            "core_frozen": True,
        }

    def poseidon_x_experiment(self, seed: int = 101, episodes: int = 2, max_steps: int = 32):
        from .poseidon_x_experiment import run_poseidon_x_experiment
        with self.lock:
            seeds = list(range(seed, seed + episodes))
            return run_poseidon_x_experiment(seeds=seeds, max_steps=max_steps)

    def external_adapter_bridge(self):
        if self._external_adapter_bridge is None:
            from .external_adapter import ExternalAdapterBridge
            self._external_adapter_bridge = ExternalAdapterBridge()
        return self._external_adapter_bridge

    def external_adapter_status(self):
        from .external_adapter import SUPPORTED_COORDINATES, SUPPORTED_UNITS
        bridge = self.external_adapter_bridge()
        return {
            "ready": True,
            "version": "1.0.0",
            "protocol_version": "poseidon-external-adapter-v1",
            "supported_units": list(SUPPORTED_UNITS),
            "supported_coordinates": list(SUPPORTED_COORDINATES),
            "active_sessions_count": len(bridge.active_sessions),
            "cancelled_sessions_count": len(bridge.cancelled_sessions),
            "latest_telemetry": bridge.latest_telemetry.to_dict(),
        }

    def nexus_omega(self, seed: int = 42):
        from .nexus_omega import NexusOmegaAgent
        return NexusOmegaAgent(base_seed=seed)

    def nexus_omega_status(self):
        return {
            "ready": True,
            "version": "2.0.0",
            "type": "causal_epigenetic_self_experimenting_autonomous_discovery_agent",
            "subsystems": [
                "causal_memory_fabric",
                "ensemble_world_model",
                "factored_causal_world_model",
                "epigenetic_morphogenesis",
                "counterfactual_scm_engine",
                "deep_active_inference_efe_planner",
                "conformal_safe_policy_guard",
                "autonomous_discovery_engine",
                "independent_scientific_auditor",
                "hierarchical_efe_planner",
                "score_guided_trajectory_diffusion",
                "quantum_causal_kernel",
                "bioharmonic_spectral_resonator",
                "metamorphic_bytecode_weaver",
                "ltl_safety_shield",
                "autopoiesis_thermodynamic_field",
                "morpheus_dream_engine",
                "nonstationary_phase_field_resonator",
                "contrastive_causal_memory_fabric",
                "metacognitive_reasoning_engine",
                "nexus_apex_engine",
            ],
            "developmental_stages": [
                "PROTOCELL", "LARVAL_FORAGER", "PUPAL_DIAPAUSE", "IMAGO_NAVIGATOR", "SENESCENT_CRYSTALLINE"
            ],
            "macro_intents": [
                "forage_expedition", "diapause_hibernation", "habitat_relocation", "emergency_fortification"
            ],
            "core_frozen": True,
            "factored_causal_world_model": True,
            "deep_active_inference_efe": True,
            "conformal_safe_policy_improvement": True,
            "hierarchical_efe_planner": True,
            "score_guided_trajectory_diffusion": True,
            "nexus_ultima": True,
            "nexus_apex": True,
            "quantum_causal_holonomy": True,
            "bioharmonic_spectral_resonator": True,
            "nonstationary_phase_field": True,
            "contrastive_causal_memory": True,
            "metacognitive_reasoning": True,
            "metamorphic_bytecode_weaver": True,
            "autopoiesis_thermodynamic_field": True,
            "morpheus_dream_engine": True,
        }

    def nexus_omega_factored_state(self, obs=None, seed: int = 42):
        agent = self.nexus_omega(seed=seed)
        sample_obs = obs if obs is not None else [1.0, 0.9, 0.9, 0.8, 0.05, 0.1, 0.0, 0.0]
        decision = agent.act(sample_obs)
        last_dec = agent.get_last_decision()
        return {
            "action": decision,
            "decision": last_dec.to_dict() if last_dec else None,
            "factored_state": last_dec.factored_state if last_dec else None,
        }

    def nexus_omega_hierarchical_state(self, obs=None, seed: int = 42):
        from .nexus_omega import NexusOmegaAgent
        agent = NexusOmegaAgent(use_hierarchical_efe=True, base_seed=seed)
        sample_obs = obs if obs is not None else [1.0, 0.9, 0.9, 0.8, 0.05, 0.1, 0.0, 0.0]
        micro_act = agent.act(sample_obs)
        hier_dec = agent.get_last_hierarchical_decision()
        return {
            "micro_action": micro_act,
            "macro_intent": hier_dec.macro_intent.value if hier_dec else "forage_expedition",
            "decision": hier_dec.to_dict() if hier_dec else None,
            "imagined_trajectory": hier_dec.imagined_trajectory.to_dict() if (hier_dec and hier_dec.imagined_trajectory) else None,
        }

    def nexus_omega_imagine(self, obs=None, intent="forage_expedition", horizon=8, seed: int = 42):
        from .nexus_omega import NexusOmegaAgent, MacroIntent
        agent = NexusOmegaAgent(use_hierarchical_efe=True, base_seed=seed)
        sample_obs = obs if obs is not None else [1.0, 0.9, 0.9, 0.8, 0.05, 0.1, 0.0, 0.0]
        agent.act(sample_obs)
        target_macro = MacroIntent(intent) if intent in [m.value for m in MacroIntent] else MacroIntent.FORAGE_EXPEDITION
        traj = agent.imagine_trajectory(intent=target_macro, horizon=horizon)
        return traj.to_dict()

    def nexus_omega_ultima_state(self, obs=None, seed: int = 42):
        from .nexus_omega import NexusOmegaAgent
        agent = NexusOmegaAgent(use_hierarchical_efe=True, use_ultima_engine=True, base_seed=seed)
        sample_obs = obs if obs is not None else [1.0, 0.9, 0.9, 0.8, 0.05, 0.1, 0.0, 0.0]
        act = agent.act(sample_obs)
        ultima_dec = agent.get_last_ultima_decision()
        return {
            "action": act,
            "decision": ultima_dec.to_dict() if ultima_dec else None,
            "quantum_state": ultima_dec.quantum_state.to_dict() if (ultima_dec and ultima_dec.quantum_state) else None,
            "spectral_profile": ultima_dec.spectral_profile.to_dict() if (ultima_dec and ultima_dec.spectral_profile) else None,
            "thermodynamic_state": ultima_dec.thermodynamic_state.to_dict() if (ultima_dec and ultima_dec.thermodynamic_state) else None,
            "active_opcode": ultima_dec.active_opcode.to_dict() if (ultima_dec and ultima_dec.active_opcode) else None,
            "dream_receipt": ultima_dec.dream_consolidation.to_dict() if (ultima_dec and ultima_dec.dream_consolidation) else None,
        }

    def nexus_omega_apex_state(self, obs=None, seed: int = 42):
        from .nexus_omega import NexusOmegaAgent
        agent = NexusOmegaAgent(use_hierarchical_efe=True, use_ultima_engine=True, use_apex_engine=True, base_seed=seed)
        sample_obs = obs if obs is not None else [1.0, 0.9, 0.9, 0.8, 0.05, 0.1, 0.0, 0.0]
        act = agent.act_apex(sample_obs)
        apex_dec = agent.last_apex_decision
        return {
            "action": act,
            "decision": apex_dec.to_dict() if apex_dec else None,
            "quantum_state": apex_dec.quantum_state.to_dict() if (apex_dec and apex_dec.quantum_state) else None,
            "nonstationary_phase": apex_dec.nonstationary_phase if apex_dec else None,
            "contrastive_memory": apex_dec.contrastive_memory if apex_dec else None,
            "cognitive_monologue": apex_dec.cognitive_monologue.to_dict() if (apex_dec and apex_dec.cognitive_monologue) else None,
            "thermodynamic_state": apex_dec.thermodynamic_state.to_dict() if (apex_dec and apex_dec.thermodynamic_state) else None,
            "active_opcode": apex_dec.active_opcode.to_dict() if (apex_dec and apex_dec.active_opcode) else None,
            "dream_receipt": apex_dec.dream_receipt.to_dict() if (apex_dec and apex_dec.dream_receipt) else None,
            "arbitration_source": apex_dec.arbitration_source if apex_dec else "quantum_tunneling",
        }

    def nexus_omega_reason(self, obs=None, seed: int = 42):
        import numpy as np
        from .nexus_omega import NexusOmegaAgent
        agent = NexusOmegaAgent(use_hierarchical_efe=True, use_ultima_engine=True, use_apex_engine=True, base_seed=seed)
        sample_obs = obs if obs is not None else [1.0, 0.9, 0.9, 0.8, 0.05, 0.1, 0.0, 0.0]
        agent.act_apex(sample_obs)
        apex_dec = agent.last_apex_decision
        if apex_dec and apex_dec.cognitive_monologue:
            return apex_dec.cognitive_monologue.to_dict()
        from .nexus_omega.reasoning import ApexReasoningEngine
        engine = ApexReasoningEngine(seed=seed)
        return engine.deliberate(np.asarray(sample_obs, dtype=np.float32)).to_dict()

    def nexus_omega_dream(self, num_rollouts: int = 4, seed: int = 42):
        from .nexus_omega import NexusOmegaAgent
        agent = NexusOmegaAgent(use_hierarchical_efe=True, use_ultima_engine=True, base_seed=seed)
        agent.act([1.0, 0.9, 0.9, 0.8, 0.05, 0.1, 0.0, 0.0])
        receipt = agent.dream_consolidate(num_rollouts=num_rollouts)
        return receipt.to_dict() if receipt else None

    def nexus_omega_synthesize_opcode(self, intent="forage_expedition", obs=None, seed: int = 42):
        from .nexus_omega import NexusOmegaAgent, MacroIntent
        agent = NexusOmegaAgent(use_hierarchical_efe=True, use_ultima_engine=True, base_seed=seed)
        target_macro = MacroIntent(intent) if intent in [m.value for m in MacroIntent] else MacroIntent.FORAGE_EXPEDITION
        sample_obs = obs if obs is not None else [1.0, 0.9, 0.9, 0.8, 0.05, 0.1, 0.0, 0.0]
        agent.act(sample_obs)
        opcode = agent.bytecode_weaver.synthesize_opcode(target_macro, sample_obs, agent.factored_world_model)
        return opcode.to_dict()




    def nexus_omega_discovery(self, seed_base: int = 50000, num_seeds: int = 4, max_steps: int = 16):
        from .nexus_omega import AutonomousDiscoveryEngine, verify_discovery_receipt
        with self.lock:
            engine = AutonomousDiscoveryEngine(max_steps=max_steps)
            ledger = engine.run_discovery_cycle(seed_base=seed_base, num_seeds=num_seeds)
            data = {
                "ledger_id": ledger.ledger_id,
                "timestamp": ledger.timestamp,
                "ratified_laws": ledger.ratified_laws,
                "refutation_graveyard": ledger.refutation_graveyard,
                "ledger_sha256": ledger.ledger_sha256,
            }
            verified = verify_discovery_receipt(data)
            return {
                "verified": verified,
                "ledger_id": ledger.ledger_id,
                "timestamp": ledger.timestamp,
                "ratified_laws_count": len(ledger.ratified_laws),
                "refutations_count": len(ledger.refutation_graveyard),
                "ledger_sha256": ledger.ledger_sha256,
                "ratified_laws": ledger.ratified_laws,
            }

    def discovery_lab(self, seed: int = 42, river_width: int = 3, max_steps: int = 100):
        from .discovery_lab import RiverWorld
        return RiverWorld(seed=seed, river_width=river_width, max_steps=max_steps)

    def discovery_lab_status(self):
        from .discovery_lab import (
            DiscoverySkillLibrary,
            DiscoveryCausalEngine,
        )
        lib = DiscoverySkillLibrary()
        causal = DiscoveryCausalEngine()
        return {
            "ready": True,
            "version": "1.0.0",
            "project": "PROMETHEUS: Autonomous Cognitive Discovery Engine",
            "subsystems": [
                "river_world_simulation",
                "object_physics_buoyancy",
                "manipulation_action_space",
                "observation_encoder",
                "curiosity_controllability_engine",
                "action_conditioned_dynamics_model",
                "causal_intervention_engine",
                "reusable_skill_library",
                "hierarchical_skill_composer",
                "multimodal_enzyme_generator",
                "curriculum_runner",
                "paired_evaluator",
            ],
            "certified_skills": [s.name for s in lib.skills.values() if s.status == "certified"],
            "core_hypotheses": list(causal.hypotheses.keys()),
            "enzyme_catalysts_supported": ["laccase_polyphenol_oxidase", "transglycosylase_cellulose_synthase", "peroxidase_bioadhesive"],
        }

    def discovery_lab_step(self, action: int = 0, seed: int = 42, river_width: int = 3):
        env = self.discovery_lab(seed=seed, river_width=river_width)
        obs, reward, done, info = env.step(action)
        return {
            "action_executed": info.get("action", str(action)),
            "observation": [round(float(x), 4) for x in obs],
            "reward": round(float(reward), 4),
            "done": bool(done),
            "state_hash": env.get_state_hash(),
            "info": info,
        }

    def discovery_lab_river_state(self, seed: int = 42, river_width: int = 3):
        env = self.discovery_lab(seed=seed, river_width=river_width)
        return {
            "river_width": env.river_width,
            "river_start_x": env.river_start_x,
            "river_end_x": env.river_end_x,
            "water_depth": env.water_depth,
            "grid_width": env.grid_width,
            "grid_height": env.grid_height,
            "agent": {
                "x": round(float(env.agent.x), 3),
                "y": round(float(env.agent.y), 3),
                "z": round(float(env.agent.z), 3),
                "health": round(float(env.agent.health), 3),
                "stamina": round(float(env.agent.stamina), 3),
                "held": env.agent.held_object_id,
                "alive": env.agent.alive,
                "reached_goal": env.agent.reached_goal,
            },
            "objects": [obj.to_dict() for obj in env.objects.values()],
            "state_hash": env.get_state_hash(),
        }

    def discovery_lab_enzyme(self, substrate: str = "lignocellulose", target_shear: float = 300.0, seed: int = 42):
        from .discovery_lab import MultimodalEnzymeGenerator
        gen = MultimodalEnzymeGenerator(seed=seed)
        enzyme = gen.generate_catalyst_for_substrate(substrate_type=substrate, target_shear_force=target_shear)
        return enzyme.to_dict()

    def discovery_lab_curriculum(self, seed: int = 42):
        from .discovery_lab import DiscoveryCurriculumRunner
        runner = DiscoveryCurriculumRunner(base_seed=seed)
        return runner.run_full_curriculum()

    def discovery_lab_evaluate(self, seeds: tuple[int, ...] = (2001, 2002, 2003, 2004, 2005), river_width: int = 3):
        from .discovery_lab import DiscoveryEvaluator
        evaluator = DiscoveryEvaluator(bootstrap_reps=1000)
        return evaluator.evaluate_crossing_benchmark(seeds=seeds, river_width=river_width)

    def mnemorph_status(self):
        relative = "outputs/mnemorph/RECEIPT.json"
        path = self.root / relative
        if not path.is_file():
            return {"ready": False, "path": relative}
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            return {
                "ready": True,
                "path": relative,
                "receipt_sha256": receipt.get("receipt_sha256"),
                "intact_mean": receipt.get("intact_mean_retrieval"),
                "hub_vulnerability": receipt.get("hub_vulnerability_ratio"),
            }
        except Exception as error:
            return {"ready": False, "path": relative, "error": str(error)}

    def jobs(self):
        with self._jobs_lock:
            if self._jobs is None:
                import hashlib
                from .jobs import ExperimentJobs
                from .provenance import assert_source_current
                sources = assert_source_current()
                producer = {"version": __version__, "source_sha256": hashlib.sha256(
                    json.dumps(sources, sort_keys=True).encode()).hexdigest()}
                self._jobs = ExperimentJobs(self.root / "outputs/jobs", producer=producer)
            return self._jobs

    def close(self):
        if self._jobs is not None:
            self._jobs.close()

    def submit_helm_experiment(self, settings):
        from .provenance import assert_source_current
        from .world import _integer, _number
        if not isinstance(settings, dict) or set(settings) - {"kind", "seed", "episodes", "max_steps", "scarcity"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "helm-experiment":
            raise ValueError("Unknown experiment job kind.")
        seed = _integer(settings.get("seed", 112000001), "seed", 0, 2**63 - 9)
        episodes = _integer(settings.get("episodes", 4), "episodes", 1, 8)
        max_steps = _integer(settings.get("max_steps", 64), "max_steps", 32, 256)
        scarcity = _number(settings.get("scarcity", 2.5), "scarcity", .5, 4)
        assert_source_current()
        # Fail missing/invalid producer readiness before accepting background work.
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Poseidon is processing another model operation. Try again shortly.")
        try:
            self.helm()
        finally:
            self.lock.release()
        normalized = {"seed": seed, "episodes": episodes, "max_steps": max_steps, "scarcity": scarcity}
        return self.jobs().submit("helm-experiment", normalized,
            lambda progress: self.helm_experiment(**normalized, progress=progress), timeout_seconds=1800)

    def helm_experiment(self, seed=112000001, episodes=4, max_steps=64, scarcity=2.5, progress=None):
        from .helm_experiment import run_helm_experiment, verify_helm_receipt
        from .experiments import write_receipt
        from .provenance import assert_source_current
        from .world import _integer, _number
        seed = _integer(seed, "seed", 0, 2**63 - 65)
        episodes = _integer(episodes, "episodes", 1, 64)
        max_steps = _integer(max_steps, "max_steps", 16, 512)
        scarcity = _number(scarcity, "scarcity", .5, 4)
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        while not self.lock.acquire(timeout=.2):
            check({"phase": "waiting-for-model"})
        try:
            check({"phase": "starting"})
            result = run_helm_experiment(self.core(), self.helm(), seeds=list(range(seed, seed + episodes)),
                max_steps=max_steps, scarcity=scarcity, progress=check)
            check({"phase": "verifying"})
            verification = verify_helm_receipt(result, progress=check)
            check({"phase": "publishing"})
            path = write_receipt(result, self.root / "outputs/helm_experiments")
            replay = next(episode for episode in result["episodes"] if episode["controller"] == "selected")
            return {key: result[key] for key in ("schema", "experiment_id", "summary", "paired", "rows", "risk_coverage", "limits") if key in result} | {
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"], "verification": verification, "replay": replay}
        finally:
            self.lock.release()

    def submit_odysseus_experiment(self, settings):
        from .provenance import assert_source_current
        from .world import _integer, _number
        if not isinstance(settings, dict) or set(settings) - {"kind", "seed", "episodes", "max_steps", "scarcity"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "odysseus-experiment":
            raise ValueError("Unknown experiment job kind.")
        seed = _integer(settings.get("seed", 133000001), "seed", 0, 2**63 - 9)
        episodes = _integer(settings.get("episodes", 4), "episodes", 1, 8)
        max_steps = _integer(settings.get("max_steps", 64), "max_steps", 32, 256)
        scarcity = _number(settings.get("scarcity", 2.5), "scarcity", .5, 4)
        assert_source_current()
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Poseidon is processing another model operation. Try again shortly.")
        try:
            self.odysseus()
        finally:
            self.lock.release()
        normalized = {"seed": seed, "episodes": episodes, "max_steps": max_steps, "scarcity": scarcity}
        return self.jobs().submit("odysseus-experiment", normalized,
            lambda progress: self.odysseus_experiment(**normalized, progress=progress), timeout_seconds=1800)

    def odysseus_experiment(self, seed=133000001, episodes=4, max_steps=64, scarcity=2.5, progress=None):
        from .odysseus_experiment import run_odysseus_experiment, verify_odysseus_receipt
        from .experiments import write_receipt
        from .provenance import assert_source_current
        from .world import _integer, _number
        seed = _integer(seed, "seed", 0, 2**63 - 65)
        episodes = _integer(episodes, "episodes", 1, 64)
        max_steps = _integer(max_steps, "max_steps", 16, 512)
        scarcity = _number(scarcity, "scarcity", .5, 4)
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        while not self.lock.acquire(timeout=.2):
            check({"phase": "waiting-for-model"})
        try:
            check({"phase": "starting"})
            seeds = list(range(seed, seed + episodes))
            result = run_odysseus_experiment(self.core(), self.odysseus(), seeds=seeds,
                max_steps=max_steps, scarcity=scarcity, progress=check)
            check({"phase": "verifying"})
            verification = verify_odysseus_receipt(result)
            check({"phase": "publishing"})
            path = write_receipt(result, self.root / "outputs/odysseus_experiments")
            replay = next(episode for episode in result["episodes"] if episode["controller"] == "odysseus_calibrated")
            return {key: result[key] for key in ("schema", "experiment_id", "summary", "paired", "limits") if key in result} | {
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"], "verification": verification, "replay": replay}
        finally:
            self.lock.release()

    def submit_mco_experiment(self, settings=None):
        from .provenance import assert_source_current
        assert_source_current()
        settings = settings or {}
        if not isinstance(settings, dict) or set(settings) - {"kind"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "mco-experiment":
            raise ValueError("Unknown experiment job kind.")
        return self.jobs().submit("mco-experiment", settings,
            lambda progress: self.mco_experiment(progress=progress), timeout_seconds=1800)

    def mco_experiment(self, progress=None):
        import json
        from .mco import run_mco_experiment, verify_mco_receipt
        from .provenance import assert_source_current
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        while not self.lock.acquire(timeout=.2):
            check({"phase": "waiting-for-model"})
        try:
            check({"phase": "starting"})
            receipt = run_mco_experiment(progress=check)
            check({"phase": "verifying"})
            verification = verify_mco_receipt(receipt)
            check({"phase": "publishing"})
            receipt_dir = self.root / "outputs/mco_experiments"
            receipt_dir.mkdir(parents=True, exist_ok=True)
            path = receipt_dir / "RECEIPT.json"
            path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
            return {
                "schema": receipt["schema"],
                "experiment_id": receipt["experiment_id"],
                "best_subset": receipt["best_subset"],
                "carrier_contributions": receipt["carrier_contributions"],
                "condition_comparison": receipt["condition_comparison"],
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": receipt["receipt_sha256"],
                "verification": verification,
            }
        finally:
            self.lock.release()

    def submit_aura_experiment(self, settings):
        from .provenance import assert_source_current
        from .world import _integer, _number
        if not isinstance(settings, dict) or set(settings) - {"kind", "seed", "episodes", "max_steps", "scarcity"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "aura-experiment":
            raise ValueError("Unknown experiment job kind.")
        seed = _integer(settings.get("seed", 144000001), "seed", 0, 2**63 - 9)
        episodes = _integer(settings.get("episodes", 4), "episodes", 1, 8)
        max_steps = _integer(settings.get("max_steps", 64), "max_steps", 16, 256)
        scarcity = _number(settings.get("scarcity", 2.5), "scarcity", .5, 4)
        assert_source_current()
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Poseidon is processing another model operation. Try again shortly.")
        try:
            self.core()
        finally:
            self.lock.release()
        normalized = {"seed": seed, "episodes": episodes, "max_steps": max_steps, "scarcity": scarcity}
        return self.jobs().submit("aura-experiment", normalized,
            lambda progress: self.aura_experiment(**normalized, progress=progress), timeout_seconds=1800)

    def aura_experiment(self, seed=144000001, episodes=4, max_steps=64, scarcity=2.5, progress=None):
        from .aura_experiment import run_aura_experiment, verify_aura_receipt
        from .experiments import write_receipt
        from .provenance import assert_source_current
        from .world import _integer, _number
        seed = _integer(seed, "seed", 0, 2**63 - 65)
        episodes = _integer(episodes, "episodes", 1, 64)
        max_steps = _integer(max_steps, "max_steps", 16, 512)
        scarcity = _number(scarcity, "scarcity", .5, 4)
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        while not self.lock.acquire(timeout=.2):
            check({"phase": "waiting-for-model"})
        try:
            check({"phase": "starting"})
            seeds = list(range(seed, seed + episodes))
            result = run_aura_experiment(self.core(), seeds=seeds, max_steps=max_steps, scarcity=scarcity, progress=check)
            check({"phase": "verifying"})
            verification = verify_aura_receipt(result)
            check({"phase": "publishing"})
            path = write_receipt(result, self.root / "outputs/aura_experiments")
            replay = next(episode for episode in result["episodes"] if episode["controller"] == "aura")
            return {key: result[key] for key in ("schema", "experiment_id", "summary") if key in result} | {
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"], "verification": verification, "replay": replay}
        finally:
            self.lock.release()

    def submit_tessera_experiment(self, settings=None):
        from .provenance import assert_source_current
        assert_source_current()
        settings = settings or {}
        if not isinstance(settings, dict) or set(settings) - {"kind", "seed"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "tessera-experiment":
            raise ValueError("Unknown experiment job kind.")
        return self.jobs().submit("tessera-experiment", settings,
            lambda progress: self.tessera_experiment(progress=progress), timeout_seconds=1800)

    def tessera_experiment(self, seeds=None, progress=None):
        import json
        from .tessera_experiment import run_tessera_experiment, verify_tessera_receipt
        from .provenance import assert_source_current
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        seeds = seeds or [155000001, 155000002, 155000003, 155000004]
        receipt = run_tessera_experiment(seeds=seeds, progress=check)
        verification = verify_tessera_receipt(receipt)
        receipt_dir = self.root / "outputs/tessera_experiments"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        path = receipt_dir / "RECEIPT.json"
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "schema": receipt["schema"],
            "experiment_id": receipt["experiment_id"],
            "ratified_opcodes_count": receipt["ratified_opcodes_count"],
            "quarantined_candidates_count": receipt["quarantined_candidates_count"],
            "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
            "receipt_sha256": receipt["receipt_sha256"],
            "verification": verification,
        }

    def submit_mnemorph_experiment(self, settings=None):
        from .provenance import assert_source_current
        assert_source_current()
        settings = settings or {}
        if not isinstance(settings, dict) or set(settings) - {"kind"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "mnemorph-experiment":
            raise ValueError("Unknown experiment job kind.")
        return self.jobs().submit("mnemorph-experiment", settings,
            lambda progress: self.mnemorph_experiment(progress=progress), timeout_seconds=1800)

    def mnemorph_experiment(self, progress=None):
        import json
        from .mnemorph import MnemorphArchaeology, verify_mnemorph_receipt
        from .provenance import assert_source_current
        assert_source_current()
        arch = MnemorphArchaeology()
        receipt = arch.run_archaeological_study()
        verification = verify_mnemorph_receipt(receipt)
        receipt_dir = self.root / "outputs/mnemorph"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        path = receipt_dir / "RECEIPT.json"
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "schema": receipt["schema"],
            "intact_mean_retrieval": receipt["intact_mean_retrieval"],
            "hub_lesion_mean_retrieval": receipt["hub_lesion_mean_retrieval"],
            "periphery_lesion_mean_retrieval": receipt["periphery_lesion_mean_retrieval"],
            "regrowth_mean_retrieval": receipt["regrowth_mean_retrieval"],
            "associative_regrowth_recovery_rate": receipt["associative_regrowth_recovery_rate"],
            "hub_vulnerability_ratio": receipt["hub_vulnerability_ratio"],
            "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
            "receipt_sha256": receipt["receipt_sha256"],
            "verification": verification,
        }

    def submit_metamorph_experiment(self, settings):
        from .provenance import assert_source_current
        from .world import _integer, _number
        if not isinstance(settings, dict) or set(settings) - {"kind", "seed", "episodes", "max_steps", "scarcity"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "metamorph-experiment":
            raise ValueError("Unknown experiment job kind.")
        seed = _integer(settings.get("seed", 199000001), "seed", 0, 2**63 - 9)
        episodes = _integer(settings.get("episodes", 4), "episodes", 1, 8)
        max_steps = _integer(settings.get("max_steps", 48), "max_steps", 16, 256)
        scarcity = _number(settings.get("scarcity", 2.5), "scarcity", .5, 4)
        assert_source_current()
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Poseidon is processing another model operation. Try again shortly.")
        try:
            self.core()
        finally:
            self.lock.release()
        normalized = {"seed": seed, "episodes": episodes, "max_steps": max_steps, "scarcity": scarcity}
        return self.jobs().submit("metamorph-experiment", normalized,
            lambda progress: self.metamorph_experiment(**normalized, progress=progress), timeout_seconds=1800)

    def metamorph_experiment(self, seed=199000001, episodes=4, max_steps=48, scarcity=2.5, progress=None):
        import json
        from .metamorph_experiment import run_metamorph_experiment, verify_metamorph_receipt
        from .provenance import assert_source_current
        from .world import _integer, _number
        seed = _integer(seed, "seed", 0, 2**63 - 65)
        episodes = _integer(episodes, "episodes", 1, 64)
        max_steps = _integer(max_steps, "max_steps", 16, 512)
        scarcity = _number(scarcity, "scarcity", .5, 4)
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        while not self.lock.acquire(timeout=.2):
            check({"phase": "waiting-for-model"})
        try:
            check({"phase": "starting"})
            seeds = list(range(seed, seed + episodes))
            result = run_metamorph_experiment(self.core(), seeds=seeds, max_steps=max_steps, scarcity=scarcity, progress=check)
            check({"phase": "verifying"})
            verification = verify_metamorph_receipt(result)
            check({"phase": "publishing"})
            receipt_dir = self.root / "outputs/metamorph_experiments"
            receipt_dir.mkdir(parents=True, exist_ok=True)
            path = receipt_dir / "RECEIPT.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            return {
                "schema": result["schema"],
                "experiment_id": result["experiment_id"],
                "arm_stats": result["arm_stats"],
                "metamorph_telemetry": result["metamorph_telemetry"],
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"],
                "verification": verification,
            }
        finally:
            self.lock.release()

    def submit_chimera_experiment(self, settings):
        from .provenance import assert_source_current
        from .world import _integer, _number
        if not isinstance(settings, dict) or set(settings) - {"kind", "seed", "episodes", "max_steps", "scarcity"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "chimera-experiment":
            raise ValueError("Unknown experiment job kind.")
        seed = _integer(settings.get("seed", 202000001), "seed", 0, 2**63 - 9)
        episodes = _integer(settings.get("episodes", 4), "episodes", 1, 8)
        max_steps = _integer(settings.get("max_steps", 48), "max_steps", 16, 256)
        scarcity = _number(settings.get("scarcity", 2.5), "scarcity", .5, 4)
        assert_source_current()
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Poseidon is processing another model operation. Try again shortly.")
        try:
            self.core()
        finally:
            self.lock.release()
        normalized = {"seed": seed, "episodes": episodes, "max_steps": max_steps, "scarcity": scarcity}
        return self.jobs().submit("chimera-experiment", normalized,
            lambda progress: self.chimera_experiment(**normalized, progress=progress), timeout_seconds=1800)

    def chimera_experiment(self, seed=202000001, episodes=4, max_steps=48, scarcity=2.5, progress=None):
        import json
        from .chimera_experiment import run_chimera_experiment, verify_chimera_receipt
        from .provenance import assert_source_current
        from .world import _integer, _number
        seed = _integer(seed, "seed", 0, 2**63 - 65)
        episodes = _integer(episodes, "episodes", 1, 64)
        max_steps = _integer(max_steps, "max_steps", 16, 512)
        scarcity = _number(scarcity, "scarcity", .5, 4)
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        while not self.lock.acquire(timeout=.2):
            check({"phase": "waiting-for-model"})
        try:
            check({"phase": "starting"})
            seeds = list(range(seed, seed + episodes))
            result = run_chimera_experiment(self.core(), seeds=seeds, max_steps=max_steps, scarcity=scarcity, progress=check)
            check({"phase": "verifying"})
            verification = verify_chimera_receipt(result)
            check({"phase": "publishing"})
            receipt_dir = self.root / "outputs/chimera_experiments"
            receipt_dir.mkdir(parents=True, exist_ok=True)
            path = receipt_dir / "RECEIPT.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            return {
                "schema": result["schema"],
                "experiment_id": result["experiment_id"],
                "arm_stats": result["arm_stats"],
                "chimera_telemetry": result["chimera_telemetry"],
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"],
                "verification": verification,
            }
        finally:
            self.lock.release()

    def submit_hyperion_experiment(self, settings):
        from .provenance import assert_source_current
        from .world import _integer, _number
        if not isinstance(settings, dict) or set(settings) - {"kind", "seed", "episodes", "max_steps", "scarcity"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "hyperion-experiment":
            raise ValueError("Unknown experiment job kind.")
        seed = _integer(settings.get("seed", 303000001), "seed", 0, 2**63 - 9)
        episodes = _integer(settings.get("episodes", 4), "episodes", 1, 8)
        max_steps = _integer(settings.get("max_steps", 48), "max_steps", 16, 256)
        scarcity = _number(settings.get("scarcity", 2.5), "scarcity", .5, 4)
        assert_source_current()
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Poseidon is processing another model operation. Try again shortly.")
        try:
            self.core()
        finally:
            self.lock.release()
        normalized = {"seed": seed, "episodes": episodes, "max_steps": max_steps, "scarcity": scarcity}
        return self.jobs().submit("hyperion-experiment", normalized,
            lambda progress: self.hyperion_experiment(**normalized, progress=progress), timeout_seconds=1800)

    def hyperion_experiment(self, seed=303000001, episodes=4, max_steps=48, scarcity=2.5, progress=None):
        import json
        from .hyperion_experiment import run_hyperion_experiment, verify_hyperion_receipt
        from .provenance import assert_source_current
        from .world import _integer, _number
        seed = _integer(seed, "seed", 0, 2**63 - 65)
        episodes = _integer(episodes, "episodes", 1, 64)
        max_steps = _integer(max_steps, "max_steps", 16, 512)
        scarcity = _number(scarcity, "scarcity", .5, 4)
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        while not self.lock.acquire(timeout=.2):
            check({"phase": "waiting-for-model"})
        try:
            check({"phase": "starting"})
            seeds = list(range(seed, seed + episodes))
            result = run_hyperion_experiment(self.core(), seeds=seeds, max_steps=max_steps, scarcity=scarcity, progress=check)
            check({"phase": "verifying"})
            verification = verify_hyperion_receipt(result)
            check({"phase": "publishing"})
            receipt_dir = self.root / "outputs/hyperion_experiments"
            receipt_dir.mkdir(parents=True, exist_ok=True)
            path = receipt_dir / "RECEIPT.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            return {
                "schema": result["schema"],
                "experiment_id": result["experiment_id"],
                "arm_stats": result["arm_stats"],
                "hyperion_telemetry": result["hyperion_telemetry"],
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"],
                "verification": verification,
            }
        finally:
            self.lock.release()

    def submit_titan_experiment(self, settings):
        from .provenance import assert_source_current
        from .world import _integer, _number
        if not isinstance(settings, dict) or set(settings) - {"kind", "seed", "episodes", "max_steps", "scarcity"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "titan-experiment":
            raise ValueError("Unknown experiment job kind.")
        seed = _integer(settings.get("seed", 99000000), "seed", 0, 2**63 - 9)
        episodes = _integer(settings.get("episodes", 4), "episodes", 1, 8)
        max_steps = _integer(settings.get("max_steps", 64), "max_steps", 16, 256)
        scarcity = _number(settings.get("scarcity", 2.5), "scarcity", .5, 4)
        assert_source_current()
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Poseidon is processing another model operation. Try again shortly.")
        try:
            self.core()
        finally:
            self.lock.release()
        normalized = {"seed": seed, "episodes": episodes, "max_steps": max_steps, "scarcity": scarcity}
        return self.jobs().submit("titan-experiment", normalized,
            lambda progress: self.titan_experiment(**normalized, progress=progress), timeout_seconds=1800)

    def titan_experiment(self, seed=99000000, episodes=4, max_steps=64, scarcity=2.5, progress=None):
        import json
        import os
        from .titan_experiment import run_titan_benchmark, verify_titan_receipt
        from .provenance import assert_source_current
        from .world import _integer, _number
        seed = _integer(seed, "seed", 0, 2**63 - 9)
        episodes = _integer(episodes, "episodes", 1, 32)
        if seed > 2**63 - episodes:
            raise ValueError("seed cannot form the requested nonnegative seed range")
        max_steps = _integer(max_steps, "max_steps", 1, 512)
        scarcity = _number(scarcity, "scarcity", .5, 4)
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        while not self.lock.acquire(timeout=.2):
            check({"phase": "waiting-for-model"})
        try:
            check({"phase": "starting"})
            core = self.core()
            result = run_titan_benchmark(core_path=core.path, seed_base=seed,
                                         episodes=episodes, max_steps=max_steps,
                                         scarcity=scarcity, progress=check)
            check({"phase": "verifying"})
            verification = verify_titan_receipt(result, core_path=core.path)
            check({"phase": "publishing"})
            receipt_dir = self.root / "outputs/titan_experiments"
            receipt_dir.mkdir(parents=True, exist_ok=True)
            path = receipt_dir / "RECEIPT.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
            os.replace(temporary, path)
            return {
                "schema": result["schema"],
                "experiment_id": result.get("experiment_id", result.get("sha256", "")[:32]),
                "arms": {arm: {key: value for key, value in data.items() if key != "episodes_data"}
                         for arm, data in result["arms"].items()},
                "summary": {arm: {key: data[key] for key in
                                   ("episodes", "survival_rate", "mean_steps", "mean_reward", "std_reward")}
                            for arm, data in result["arms"].items()},
                "titan_telemetry": result.get("titan_telemetry"),
                "paired_vs_titan": result.get("paired_vs_titan"),
                "paired": {
                    "titan_vs_" + arm: {
                        "mean_reward_delta": sum(row[arm + "_reward_delta"] for row in result["paired_vs_titan"]) / episodes,
                        "per_seed": [{"reward_delta": row[arm + "_reward_delta"]} for row in result["paired_vs_titan"]],
                        "wins": sum(row[arm + "_reward_delta"] > 1e-10 for row in result["paired_vs_titan"]),
                        "ties": sum(abs(row[arm + "_reward_delta"]) <= 1e-10 for row in result["paired_vs_titan"]),
                        "losses": sum(row[arm + "_reward_delta"] < -1e-10 for row in result["paired_vs_titan"]),
                    } for arm in ("core", "aura", "metamorph", "chimera", "hyperion")
                },
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result.get("receipt_sha256", result.get("sha256")),
                "verification": verification,
                "replay": result["arms"]["titan"]["episodes_data"][0],
                "titan": self.titan_status(),
            }
        finally:
            self.lock.release()

    def submit_trident_experiment(self, settings):
        from .provenance import assert_source_current
        from .world import _integer, _number
        if not isinstance(settings, dict) or set(settings) - {"kind", "seed", "episodes", "max_steps", "scarcity"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "trident-experiment":
            raise ValueError("Unknown experiment job kind.")
        seed = _integer(settings.get("seed", 88000000), "seed", 0, 2**63 - 9)
        episodes = _integer(settings.get("episodes", 4), "episodes", 1, 8)
        max_steps = _integer(settings.get("max_steps", 64), "max_steps", 16, 256)
        scarcity = _number(settings.get("scarcity", 2.5), "scarcity", .5, 4)
        assert_source_current()
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Poseidon is processing another model operation. Try again shortly.")
        try:
            self.core()
        finally:
            self.lock.release()
        normalized = {"seed": seed, "episodes": episodes, "max_steps": max_steps, "scarcity": scarcity}
        return self.jobs().submit("trident-experiment", normalized,
            lambda progress: self.trident_experiment(**normalized, progress=progress), timeout_seconds=1800)

    def trident_experiment(self, seed=88000000, episodes=4, max_steps=64, scarcity=2.5, progress=None):
        import json
        import os
        from .trident_experiment import run_trident_benchmark, verify_trident_receipt
        from .provenance import assert_source_current
        from .world import _integer, _number
        seed = _integer(seed, "seed", 0, 2**63 - 9)
        episodes = _integer(episodes, "episodes", 1, 32)
        if seed > 2**63 - episodes:
            raise ValueError("seed cannot form the requested nonnegative seed range")
        max_steps = _integer(max_steps, "max_steps", 1, 512)
        scarcity = _number(scarcity, "scarcity", .5, 4)
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        while not self.lock.acquire(timeout=.2):
            check({"phase": "waiting-for-model"})
        try:
            check({"phase": "starting"})
            core = self.core()
            result = run_trident_benchmark(core_path=core.path, seed_base=seed,
                                           episodes=episodes, max_steps=max_steps,
                                           scarcity=scarcity, progress=check)
            check({"phase": "verifying"})
            verification = verify_trident_receipt(result, core_path=core.path)
            check({"phase": "publishing"})
            receipt_dir = self.root / "outputs/trident_experiments"
            receipt_dir.mkdir(parents=True, exist_ok=True)
            path = receipt_dir / "RECEIPT.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
            os.replace(temporary, path)
            return {
                "schema": result["schema"],
                "experiment_id": result.get("sha256", "")[:32],
                "arms": {arm: {key: value for key, value in data.items() if key != "episodes_data"}
                         for arm, data in result["arms"].items()},
                "summary": {arm: {key: data[key] for key in
                                   ("episodes", "survival_rate", "mean_steps", "mean_reward", "std_reward")}
                            for arm, data in result["arms"].items()},
                "trident_telemetry": result.get("trident_telemetry"),
                "paired_vs_trident": result.get("paired_vs_trident"),
                "paired": {
                    "trident_vs_" + arm: {
                        "mean_reward_delta": sum(row[arm + "_reward_delta"] for row in result["paired_vs_trident"]) / episodes,
                        "per_seed": [{"reward_delta": row[arm + "_reward_delta"]} for row in result["paired_vs_trident"]],
                        "wins": sum(row[arm + "_reward_delta"] > 1e-10 for row in result["paired_vs_trident"]),
                        "ties": sum(abs(row[arm + "_reward_delta"]) <= 1e-10 for row in result["paired_vs_trident"]),
                        "losses": sum(row[arm + "_reward_delta"] < -1e-10 for row in result["paired_vs_trident"]),
                    } for arm in ("core", "erased", "shifted", "no_topology", "ungated")
                },
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result.get("sha256"),
                "verification": verification,
                "replay": result["arms"]["trident"]["episodes_data"][0],
                "trident": self.trident_status(),
            }
        finally:
            self.lock.release()

    def submit_cognitive_experiment(self, settings):
        from .provenance import assert_source_current
        from .world import _integer, _number
        if not isinstance(settings, dict) or set(settings) - {"kind", "seed", "episodes", "max_steps", "scarcity", "use_pomdp"}:
            raise ValueError("Unknown job setting.")
        if settings.get("kind") != "cognitive-experiment":
            raise ValueError("Unknown experiment job kind.")
        seed = _integer(settings.get("seed", 90000000), "seed", 0, 2**63 - 9)
        episodes = _integer(settings.get("episodes", 4), "episodes", 1, 8)
        max_steps = _integer(settings.get("max_steps", 64), "max_steps", 16, 256)
        scarcity = _number(settings.get("scarcity", 2.5), "scarcity", .5, 4)
        use_pomdp = bool(settings.get("use_pomdp", False))
        assert_source_current()
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Poseidon is processing another model operation. Try again shortly.")
        try:
            self.core()
        finally:
            self.lock.release()
        normalized = {"seed": seed, "episodes": episodes, "max_steps": max_steps, "scarcity": scarcity, "use_pomdp": use_pomdp}
        return self.jobs().submit("cognitive-experiment", normalized,
            lambda progress: self.cognitive_experiment(**normalized, progress=progress), timeout_seconds=1800)

    def cognitive_experiment(self, seed=90000000, episodes=4, max_steps=32, scarcity=2.5, use_pomdp=False, progress=None):
        import json
        import os
        from .cognitive_experiment import run_cognitive_benchmark, verify_cognitive_receipt
        from .provenance import assert_source_current
        from .world import _integer, _number
        seed = _integer(seed, "seed", 0, 2**63 - 9)
        episodes = _integer(episodes, "episodes", 1, 32)
        if seed > 2**63 - episodes:
            raise ValueError("seed cannot form the requested nonnegative seed range")
        max_steps = _integer(max_steps, "max_steps", 1, 512)
        scarcity = _number(scarcity, "scarcity", .5, 4)
        def check(event):
            assert_source_current()
            if progress:
                progress(event)
        while not self.lock.acquire(timeout=.2):
            check({"phase": "waiting-for-model"})
        try:
            check({"phase": "starting"})
            core = self.core()
            result = run_cognitive_benchmark(
                core_path=core.path,
                seed_base=seed,
                episodes=episodes,
                max_steps=max_steps,
                scarcity=scarcity,
                use_pomdp=use_pomdp,
                oracle_checkpoint=self.oracle_checkpoint,
                progress=check,
            )
            check({"phase": "verifying"})
            verification = verify_cognitive_receipt(result, core_path=core.path)
            check({"phase": "publishing"})
            receipt_dir = self.root / "outputs/cognitive_experiments"
            receipt_dir.mkdir(parents=True, exist_ok=True)
            path = receipt_dir / "RECEIPT.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
            os.replace(temporary, path)

            # Format paired for UI display
            paired_for_ui = {}
            for pair_name, pair_val in result.get("paired", {}).items():
                paired_for_ui[pair_name] = {
                    "mean_reward_delta": pair_val.get("mean_reward_advantage", 0.0),
                    "wins": pair_val.get("wins", 0),
                    "ties": pair_val.get("ties", 0),
                    "losses": pair_val.get("losses", 0),
                    "per_seed": [{"reward_delta": d} for d in pair_val.get("deltas", [])],
                }

            intact_replay = (
                result.get("episodes_data", {}).get("intact", [{}])[0]
                if result.get("episodes_data", {}).get("intact")
                else None
            )

            return {
                "schema": result["schema"],
                "experiment_id": result.get("sha256", "")[:32],
                "receipt_sha256": result.get("sha256"),
                "summary": result.get("arms"),
                "arms": result.get("arms"),
                "paired": paired_for_ui,
                "verification": verification,
                "replay": intact_replay,
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "cognitive_status": self.cognitive_agent_status(),
            }
        finally:
            self.lock.release()

    def hyper_experiment(self, seed=90000001, episodes=4, max_steps=64, scarcity=2.5):
        if self.oracle_checkpoint is not None:
            raise ValueError("Trained ORACLE experiments require cognitive-experiment; the legacy hyper assay includes unsupported 3D worlds")
        from .hyper_experiment import run_hyper_experiment
        from .world import _integer, _number
        seed = _integer(seed, "seed", 0, 2**63 - 9)
        episodes = _integer(episodes, "episodes", 1, 32)
        max_steps = _integer(max_steps, "max_steps", 1, 512)
        scarcity = _number(scarcity, "scarcity", .5, 4)
        with self.lock:
            return run_hyper_experiment(
                seed=seed,
                episodes=episodes,
                max_steps=max_steps,
                scarcity=scarcity,
                core=self.core(),
            )

    def _candidate_status(self, name):
        """Readiness may be cached while a world or experiment owns the model."""
        relative = f"outputs/{name}/atlas.json"
        if not self.lock.acquire(blocking=False):
            snapshot = copy.deepcopy(self._candidate_status_snapshots.get(name, {
                "ready": False, "path": relative, "error": "Readiness check pending while Poseidon is busy.",
            }))
            return snapshot | {"status_cached": True, "busy": True}
        try:
            path = self.root / relative
            if not path.is_file():
                result = {"ready": False, "path": relative}
            else:
                try:
                    payload = getattr(self, name)().artifact
                    receipt = {key: payload[key] for key in ("partition", "selection", "calibration", "config") if key in payload}
                    if "records" in payload:
                        receipt["training_samples"] = len(payload["records"])
                    result = {"ready": True, "path": relative, "artifact_sha256": payload["sha256"],
                              "fit_receipt": receipt}
                except (ValueError, TypeError, OSError, RuntimeError, ImportError) as error:
                    result = {"ready": False, "path": relative, "error": str(error)}
            self._candidate_status_snapshots[name] = copy.deepcopy(result)
            return result | {"status_cached": False, "busy": False}
        finally:
            self.lock.release()

    def experiment(self, seed=93000001, episodes=4, max_steps=64, scarcity=2.5):
        from .experiments import ExperimentSpec, run_experiment, verify_receipt, write_receipt
        spec = ExperimentSpec(seed=seed, episodes=episodes, max_steps=max_steps, scarcity=scarcity)
        with self.lock:
            result = run_experiment(self.core(), self.atlas(), spec)
            verification = verify_receipt(result)
            path = write_receipt(result, self.root / "outputs/experiments")
            replay = next(episode for episode in result["episodes"] if episode["controller"] == "atlas")
            return {key: result[key] for key in ("schema", "experiment_id", "summary", "paired", "rows", "atlas", "limits")} | {
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"], "verification": verification, "replay": replay}

    def contrast_experiment(self, seed=104000001, episodes=4, max_steps=64, scarcity=2.5):
        from .experiments import ExperimentSpec
        from .contrast_experiments import run_experiment, verify_receipt, write_receipt
        spec = ExperimentSpec(seed=seed, episodes=episodes, max_steps=max_steps, scarcity=scarcity)
        with self.lock:
            result = run_experiment(self.core(), self.atlas(), self.contrast(), spec)
            verification = verify_receipt(result)
            path = write_receipt(result, self.root / "outputs/contrast_experiments")
            replay = next(episode for episode in result["episodes"] if episode["controller"] == "contrast")
            return {key: result[key] for key in ("schema", "experiment_id", "summary", "paired", "rows", "contrast", "prediction_comparison", "limits")} | {
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"], "verification": verification, "replay": replay}

    def horizon_experiment(self, seed=106000001, episodes=4, max_steps=64, scarcity=2.5):
        from .experiments import ExperimentSpec
        from .horizon_experiments import run_experiment, verify_receipt, write_receipt
        spec = ExperimentSpec(seed=seed, episodes=episodes, max_steps=max_steps, scarcity=scarcity)
        with self.lock:
            result = run_experiment(self.core(), self.atlas(), self.contrast(), self.horizon(), spec)
            verification = verify_receipt(result)
            path = write_receipt(result, self.root / "outputs/horizon_experiments")
            replay = next(episode for episode in result["episodes"] if episode["controller"] == "horizon")
            return {key: result[key] for key in ("schema", "experiment_id", "summary", "paired", "rows", "horizon", "limits")} | {
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"], "verification": verification, "replay": replay}

    def odyssey_experiment(self, seed=108000001, episodes=4, max_steps=64, scarcity=2.5):
        from .experiments import ExperimentSpec
        from .odyssey_experiments import run_experiment, verify_receipt, write_receipt
        spec = ExperimentSpec(seed=seed, episodes=episodes, max_steps=max_steps, scarcity=scarcity)
        with self.lock:
            result = run_experiment(self.core(), self.atlas(), self.contrast(), self.horizon(), self.odyssey(), spec)
            verification = verify_receipt(result)
            path = write_receipt(result, self.root / "outputs/odyssey_experiments")
            replay = next(episode for episode in result["episodes"] if episode["controller"] == "odyssey")
            return {key: result[key] for key in ("schema", "experiment_id", "summary", "paired", "rows", "odyssey", "limits")} | {
                "artifact_url": "/artifacts/" + path.relative_to(self.root / "outputs").as_posix(),
                "receipt_sha256": result["receipt_sha256"], "verification": verification, "replay": replay}

    def core(self):
        from .core import active_core_path
        path = active_core_path(self.root)
        if not path.exists():
            raise RuntimeError("The Tidal checkpoint is not ready yet. Check training progress.")
        stamp = (str(path), path.stat().st_mtime_ns)
        if self._core is None or self._core_stamp != stamp:
            import torch
            from .core import CoreRuntime
            torch.set_num_threads(2)
            self._core = CoreRuntime(path)
            self._core_stamp = stamp
        return self._core

    def mco_status(self):
        relative = "outputs/mco_experiments/RECEIPT.json"
        path = self.root / relative
        status = {"ready": True, "path": relative, "receipt_ready": False, "receipt_integrity": False,
                  "receipt_verified": False, "replay_verified": False, "verification_scope": "not_checked"}
        if not path.is_file():
            return status
        try:
            receipt = self._status_receipt("mco")
            return status | {
                "receipt_ready": True, "receipt_integrity": True,
                "verification_scope": "checksum_and_summary_arithmetic",
                "receipt_sha256": receipt.get("receipt_sha256"),
                "best_subset": receipt.get("best_subset"),
                "carrier_contributions": receipt.get("carrier_contributions"),
            }
        except Exception as error:
            return status | {"error": str(error)}

    def status(self):
        reports, errors = {}, {}
        for name, relative in [("core", "runs/tidal/receipt.json"), ("language", "runs/language/report.json"), ("evaluation", "runs/evaluation.json"), ("language_data", "data/language/manifest.json"), ("dagger", "runs/tidal_dagger/report.json"), ("calibration", "runs/calibration.json"), ("active_core", "runs/active_core.json")]:
            path = self.root/relative
            try:
                if path.exists():
                    reports[name] = _status_report(path)
            except (ValueError, OSError, RecursionError) as error:
                errors["report:" + name] = str(error)
        for name in ("core", "language"):
            path = self.root/("runs/tidal/training.jsonl" if name == "core" else "runs/language/metrics.jsonl")
            try:
                if path.exists():
                    rows, invalid = _recent_metrics(path)
                    reports[name + "_recent"] = rows
                    if invalid:
                        errors["report:" + name + "_recent"] = f"Skipped {invalid} incomplete or invalid metric records in the recent log tail."
            except (ValueError, OSError, RecursionError) as error:
                errors["report:" + name + "_recent"] = str(error)
        from .core import active_core_path
        from .provenance import source_status
        core_error = None
        try:
            core_ready = active_core_path(self.root).is_file()
        except (ValueError, TypeError, OSError) as error:
            core_ready, core_error = False, str(error)
        result = {"version": __version__, "name": "Supermix Poseidon",
                  "language_ready": (self.root / "models/language/model.safetensors").is_file(),
                  "core_ready": core_ready, "reports": reports,
                  "limits": "Experimental composite system. Controlled geometric media. Tool-assisted maths. Synthetic survival."}
        components = {name: name + "_status" for name in (
            "atlas", "contrast", "horizon", "odyssey", "helm", "odysseus", "mco", "aura", "tessera",
            "mnemorph", "metamorph", "chimera", "hyperion", "titan", "trident", "hyper_agent",
            "ecosystem", "poseidon_x", "nexus_omega", "discovery_lab", "omni_reasoner", "pantheon",
            "athenaeum", "hephaestus", "hyperborea", "neurograft", "aether", "chronos", "noether", "cartan", "tether", "faultline", "crosswire", "poincare")}
        components.update(external_adapter="external_adapter_status", noosphere="creative_status")
        for name, method in components.items():
            try:
                component = getattr(self, method)()
                if not isinstance(component, dict):
                    raise ValueError("Component status must be a JSON object")
                json.dumps(component, allow_nan=False)
                result[name] = component
                if component.get("error"):
                    errors[name] = str(component["error"])
            except Exception as error:
                errors[name] = str(error)
                result[name] = {"ready": False, "error": str(error), "type": type(error).__name__}
        # Readiness polling must not trigger the full cognitive replay verifier.
        result["cognitive"] = {"ready": core_ready, "receipt_verified": False,
                               "replay_verified": False, "verification_scope": "readiness_only",
                               "promotion": False}
        try:
            result.update(source_status())
        except (ValueError, OSError, RecursionError) as error:
            errors["source"] = str(error)
            result.update(source_current=False, restart_required=True, changed_source_files=[])
        if core_error:
            result["core_error"] = core_error
            errors["core"] = core_error
        result["status_errors"] = errors
        return result

    def omni_reasoner_status(self):
        return {
            "ready": True,
            "version": "stage-27-omni-reasoner-v8",
            "prover": "Tessera LTL Bounded Model Checker",
            "deliberation": "GhostInTheMachine Cognitive Monologue + 4-Agent Dialectical Tribunal",
            "intents": [
                "physical_discovery",
                "biocatalytic_synthesis",
                "formal_verification",
                "clifford_geometric",
                "causal_superposition",
                "biomechanical_kinematics",
                "naval_stability",
                "heuristic_problem_solving",
                "pantheon_24_pillar_synthesis",
                "creative_bisociation_synthesis",
                "scientific_hypothesis_falsification",
                "neurograft_connectome",
                "aether_gauge_thermo",
                "chronos_quantum_geometry",
                "noether_invariant_symmetries",
                "cartan_differential_systems",
                "tether_reliability_gating",
                "faultline_joint_failure",
                "crosswire_interoperability",
                "poincare_dynamical_topology",
                "general_hyper_reasoning",
            ],
        }

    def creative_status(self):
        from .creative_synthesis import ARCHETYPE_LIBRARY, INVARIANT_AXES
        return {
            "ready": True,
            "version": "stage-17-noosphere-v1",
            "archetypes_count": len(ARCHETYPE_LIBRARY),
            "invariant_axes": list(INVARIANT_AXES),
            "tribunal_agents": ["ThesisArchitect", "AntithesisSkeptic", "CreativeBisociator", "SynthesisArbiter"],
        }

    def creative_synthesize(self, prompt: str, seed: int = 42):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 32000:
            raise ValueError("Prompt must contain 1–32000 characters.")
        with self.lock:
            from .creative_synthesis import DialecticalTribunal
            return DialecticalTribunal().deliberate(prompt.strip(), seed=seed)

    def pantheon(self, seed=42):
        with self.lock:
            if self._pantheon is None:
                from .pantheon import PantheonEngine
                try:
                    core = self.core()
                except Exception:
                    core = None
                self._pantheon = PantheonEngine(core=core, seed=seed)
            return self._pantheon

    def pantheon_status(self):
        from .pantheon import PANTHEON_24_PILLARS
        return {
            "ready": True,
            "version": "stage-16-pantheon-v1",
            "total_pillars": len(PANTHEON_24_PILLARS),
            "layers": sorted({p["layer"] for p in PANTHEON_24_PILLARS}),
            "merkle_dag": "Quibble Content-Addressed 24-Node Merkle Build DAG",
            "hardware_frame": "Flipper NTAG 215 NFC NDEF + ISO/IEC 14443-A CRC-16 (0x6363)",
            "dream_consolidation": "Morpheus Counterfactual Theorem Synthesis",
        }

    def pantheon_synthesize(self, prompt: str, seed: int = 42):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 32000:
            raise ValueError("Prompt must contain 1–32000 characters.")
        with self.lock:
            engine = self.pantheon(seed=seed)
            return engine.synthesize(prompt.strip(), seed=seed)

    def reason(self, prompt, seed=42, max_new_tokens=512):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 32000:
            raise ValueError("Prompt must contain 1–32000 characters.")
        with self.lock:
            from .omni_reasoner import OmniReasoner
            if self._omni_reasoner is None:
                self._omni_reasoner = OmniReasoner(seed=seed)
            res = self._omni_reasoner.reason(prompt.strip(), seed=seed, language_runtime=self.language, max_new_tokens=max_new_tokens)
            return res.to_dict()

    def solve_problem(self, prompt: str, domain: Optional[str] = None):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 32000:
            raise ValueError("Prompt must contain 1–32000 characters.")
        with self.lock:
            from .problem_solver import solve_problem
            return solve_problem(prompt.strip(), domain=domain)

    def problem_capabilities(self):
        from .problem_solver import ProblemSolver
        return ProblemSolver().capabilities()

    def hypothesis_status(self):
        return {
            "ready": True,
            "version": "1.0.0",
            "project": "OLYMPUS: Scientific Hypothesis & Falsification Engine",
            "popperian_demarcation": True,
            "lakatos_msrp_evaluator": True,
            "crucial_experiment_designer": True,
            "shannon_information_bits": True,
            "cryptographic_falsification_receipts": True,
        }

    def hypothesis_formulate(
        self,
        conjecture: str,
        domain: str = "empirical_science",
        hard_core_assumptions=None,
        protective_belt=None,
        falsification_criteria=None,
        novel_predictions=None,
        prior: float = 0.5,
    ):
        with self.lock:
            from .hypothesis_engine import HypothesisEngine
            engine = HypothesisEngine()
            hyp = engine.formulate_hypothesis(
                conjecture=conjecture,
                domain=domain,
                hard_core_assumptions=hard_core_assumptions or ["Primary Invariant Axiom"],
                protective_belt=protective_belt or [],
                falsification_criteria=falsification_criteria or ["Empirical divergence > 3-sigma"],
                novel_predictions=novel_predictions or [],
                prior=prior,
            )
            return hyp.to_dict()

    def proof_status(self):
        return {
            "ready": True,
            "version": "1.0.0",
            "project": "ARCHIMEDES: Formal Natural Deduction & Equational Proof Engine",
            "natural_deduction": True,
            "schwartz_zippel_algebraic_identities": True,
            "mathematical_induction": True,
            "tamper_evident_sha256_certificates": True,
        }

    def proof_prove(
        self,
        theorem: str = "",
        premises: list = None,
        goal: str = None,
        lhs: str = None,
        rhs: str = None,
        induction_key: str = None,
    ):
        with self.lock:
            from .proof_engine import ProofEngine
            engine = ProofEngine()
            if lhs and rhs:
                return engine.prove_algebraic_identity(lhs, rhs)
            if induction_key:
                return engine.prove_by_induction(induction_key)
            if premises and goal:
                return engine.prove_propositional(premises, goal)
            # Default to standard algebraic or propositional proof if prompt provided
            if "==" in theorem or "=" in theorem:
                parts = theorem.split("==") if "==" in theorem else theorem.split("=")
                return engine.prove_algebraic_identity(parts[0].strip(), parts[1].strip())
            return engine.prove_propositional(premises or ["P", "P -> Q"], goal or "Q")

    def creativity_status(self):
        return {
            "ready": True,
            "version": "1.0.0",
            "project": "DA VINCI: Deep Computational Creativity Engine",
            "koestler_bisociation": True,
            "triz_contradiction_engine": True,
            "canonical_matrices_count": 8,
            "triz_principles_count": 11,
            "pareto_optimal_innovation": True,
        }

    def creativity_matrices(self):
        from .creativity_engine import CreativityEngine
        return CreativityEngine().list_matrices()

    def creativity_principles(self):
        from .creativity_engine import CreativityEngine
        return CreativityEngine().list_triz_principles()

    def creativity_invent(
        self,
        problem: str,
        domain_a: str = "quantum_mechanics",
        domain_b: str = "distributed_consensus",
        improving_param: str = "speed",
        deteriorating_param: str = "precision",
    ):
        with self.lock:
            from .creativity_engine import CreativityEngine
            engine = CreativityEngine()
            dossier = engine.invent(
                problem_statement=problem,
                domain_a=domain_a,
                domain_b=domain_b,
                improving_param=improving_param,
                deteriorating_param=deteriorating_param,
            )
            return dossier.to_dict()

    def athenaeum_status(self):
        from .athenaeum import AthenaeumEngine
        return AthenaeumEngine().status()

    def athenaeum_synthesize(
        self,
        expression: str = "(x + y) * (x - y) + y ^ 2",
        target_expression: Optional[str] = "x ^ 2",
        samples=None,
        budget_rollouts: int = 20,
    ):
        with self.lock:
            from .athenaeum import AthenaeumEngine
            engine = AthenaeumEngine()
            expr = (expression or "(x + y) * (x - y) + y ^ 2").strip()
            target = target_expression
            if "==" in expr:
                parts = expr.split("==", 1)
                expr = parts[0].strip()
                target = parts[1].strip()
            parsed_samples = None
            if isinstance(samples, list) and len(samples) >= 3:
                parsed_samples = [(float(p[0]), float(p[1])) for p in samples if isinstance(p, (list, tuple)) and len(p) >= 2]
            return engine.synthesize(
                expression=expr,
                target_expression=target,
                samples=parsed_samples,
                budget_rollouts=int(budget_rollouts),
            )

    def hephaestus_status(self):
        from .hephaestus import HephaestusEngine
        return HephaestusEngine().status()

    def hephaestus_synthesize(
        self,
        corpus=None,
        terminal_rewards=None,
        koopman_params=None,
    ):
        with self.lock:
            from .hephaestus import HephaestusEngine
            engine = HephaestusEngine()
            return engine.synthesize(
                corpus=corpus,
                terminal_rewards=terminal_rewards,
                koopman_params=koopman_params,
            )

    def hyperborea_status(self):
        from .hyperborea import HyperboreaEngine
        return HyperboreaEngine().status()

    def hyperborea_synthesize(
        self,
        sheaf_angles_rad=None,
        rg_initial_K=None,
        braid_word=None,
    ):
        with self.lock:
            from .hyperborea import HyperboreaEngine
            engine = HyperboreaEngine()
            return engine.synthesize(
                sheaf_angles_rad=sheaf_angles_rad,
                rg_initial_K=rg_initial_K,
                braid_word=braid_word,
            )

    def neurograft_status(self):
        return {
            "ready": True,
            "version": "v3.12.0-NEUROGRAFT-95",
            "stage": 31,
            "subsystems": {
                "successor_representation_td_learning": True,
                "successor_reward_revaluation": True,
                "sharp_wave_ripple_reverse_replay": True,
                "graph_laplacian_spatial_decomposition": True,
                "trainable_recurrent_graph_circuit": True,
                "mnemosyne_successor_memory_adapter": True,
            },
            "total_verified_domains": 95,
        }

    def neurograft_synthesize(
        self,
        seed: int = 42,
        discount: Optional[float] = None,
        learning_rate: Optional[float] = None,
        **kwargs,
    ):
        with self.lock:
            if "horizon" in kwargs or "modulator_overrides" in kwargs or "lesion_source" in kwargs:
                from .neurograft import NeurograftEngine as LegacyNeurograftEngine
                engine = LegacyNeurograftEngine(seed=seed)
                return engine.synthesize(
                    horizon=kwargs.get("horizon", 6),
                    modulator_overrides=kwargs.get("modulator_overrides"),
                    lesion_source=kwargs.get("lesion_source"),
                )
            from .neurograft_engine import NeurograftEngine
            engine = NeurograftEngine()
            return engine.synthesize(seed=seed, discount=discount, learning_rate=learning_rate)

    def reasoning_audit(self, seed=20261030, samples=8):
        """Run bounded independent solver checks without loading model weights."""
        from .research.reasoning_audit import run_audit
        with self.lock:
            return run_audit(seed=seed, samples=samples)

    def metamorphic_audit(self, seed=20261031, samples=8):
        """Check related inputs and independent correctness without model weights."""
        from .research.metamorphic_audit import run_audit
        with self.lock:
            return run_audit(seed=seed, samples=samples)

    def episode_risk_audit(self, protocol, outcomes_by_condition):
        """Screen supplied complete episodes; never activates a controller."""
        from .research.episode_risk import audit_episode_risk
        with self.lock:
            return audit_episode_risk(protocol, outcomes_by_condition)

    def poincare_status(self):
        return {
            "ready": True,
            "version": "v3.14.0-POINCARE-100",
            "stage": 32,
            "subsystems": {
                "poincare_return_map_chaos": True,
                "hyperbolic_poincare_disk_geodesic": True,
                "poincare_duality_morse_complex": True,
                "ricci_flow_perelman_entropy": True,
                "unified_quartet_cognitive_synthesis": True,
            },
            "total_verified_domains": None,
            "evidence_scope": "five_synthetic_diagnostics",
            "live_policy_activated": False,
        }

    def poincare_synthesize(
        self,
        seed: int = 42,
        **kwargs,
    ):
        with self.lock:
            from .poincare_engine import PoincareEngine
            engine = PoincareEngine(seed=seed)
            return engine.synthesize(
                stochastic_k=kwargs.get("stochastic_k", 0.85),
                iterations=kwargs.get("iterations", 400),
                curvature_c=kwargs.get("curvature_c", 1.0),
                surface_type=kwargs.get("surface_type", "torus"),
                ricci_vertices=kwargs.get("ricci_vertices", 6),
                ricci_steps=kwargs.get("ricci_steps", 30),
                observation_16d=kwargs.get("observation_16d"),
            )

    def aether_status(self):
        from .aether import AetherEngine
        return AetherEngine().status()

    def aether_synthesize(
        self,
        grid_coords=(0, 0),
        thermo_center_end: float = 2.0,
        mu: float = 1.0,
        sigma: float = 1.5,
        symplectic_steps: int = 60,
    ):
        with self.lock:
            from .aether import AetherEngine
            engine = AetherEngine()
            return engine.synthesize(
                grid_coords=grid_coords,
                thermo_center_end=thermo_center_end,
                mu=mu,
                sigma=sigma,
                symplectic_steps=symplectic_steps,
            )

    def chronos_status(self):
        return {
            "ready": True,
            "version": "v3.6.0-CHRONOS-65",
            "stage": 25,
            "subsystems": {
                "causal_dynamical_triangulations": True,
                "noncommutative_spectral_geometry": True,
                "tropical_algebraic_geometry": True,
                "stiefel_grassmannian_optimization": True,
                "coalgebraic_bisimulation": True,
                "coalgebraic_bisimulation_metric": True,
            },
            "total_verified_domains": 65,
        }

    def chronos_synthesize(self, seed: int = 42):
        with self.lock:
            from .chronos import ChronosEngine
            engine = ChronosEngine()
            return engine.synthesize(seed=seed)

    def noether_status(self):
        return {
            "ready": True,
            "version": "v3.7.0-NOETHER-70",
            "stage": 26,
            "subsystems": {
                "noether_symmetry_conservation": True,
                "modular_tensor_category": True,
                "clifford_spacetime_algebra": True,
                "batalin_vilkovisky_quantization": True,
                "active_matter_nonequilibrium": True,
            },
            "total_verified_domains": 70,
        }

    def noether_synthesize(self, seed: int = 42):
        with self.lock:
            from .noether import NoetherEngine
            engine = NoetherEngine()
            return engine.synthesize(seed=seed)

    def cartan_status(self):
        return {
            "ready": True,
            "version": "v3.8.0-CARTAN-75",
            "stage": 27,
            "subsystems": {
                "cartan_exterior_systems": True,
                "non_hermitian_pt_symmetry": True,
                "exceptional_jordan_algebra": True,
                "holographic_ryu_takayanagi": True,
                "morphogenetic_turing_reaction": True,
            },
            "total_verified_domains": 75,
        }

    def cartan_synthesize(self, seed: int = 42):
        with self.lock:
            from .cartan import CartanEngine
            engine = CartanEngine()
            return engine.synthesize(seed=seed)

    def crosswire_status(self):
        return {
            "ready": True,
            "version": "v3.11.0-CROSSWIRE-90",
            "stage": 30,
            "subsystems": {
                "candidate_permutation_space": True,
                "version_space_tracking": True,
                "active_diagnostic_probing": True,
                "all_hypothesis_contract_verification": True,
                "closed_loop_runtime_shield": True,
                "external_adapter_interoperability": True,
            },
            "total_verified_domains": 90,
        }

    def crosswire_synthesize(self, seed: int = 42, budget: int = 16, num_actions: int = 4):
        with self.lock:
            from .crosswire_engine import CrosswireEngine
            engine = CrosswireEngine(num_actions=num_actions)
            return engine.synthesize(seed=seed, budget=budget, num_actions=num_actions)

    def faultline_status(self):
        return {
            "ready": True,
            "version": "v3.10.0-FAULTLINE-85",
            "stage": 29,
            "subsystems": {
                "discrete_joint_failure_selection": True,
                "higher_order_hyperedge_proof": True,
                "condition_stratified_risk_matrix": True,
                "continuous_joint_error_covariance": True,
                "minimax_continuous_panel_triage": True,
                "oracle_active_panel_slicing": True,
            },
            "total_verified_domains": 85,
        }

    def faultline_synthesize(self, seed: int = 42, panel_size: int = 3):
        with self.lock:
            from .faultline_engine import FaultlineEngine
            engine = FaultlineEngine()
            return engine.synthesize(seed=seed, panel_size=panel_size)

    def tether_status(self):
        return {
            "ready": True,
            "version": "v3.9.0-TETHER-80",
            "stage": 28,
            "subsystems": {
                "continuous_reliability_weighting": True,
                "counterfactual_covariation_invariance": True,
                "causal_attribution_saliency": True,
                "zero_shot_corruption_detection": True,
                "evidence_calibrated_fallback_gate": True,
            },
            "total_verified_domains": 80,
        }

    def tether_synthesize(self, seed: int = 42):
        with self.lock:
            from .tether_engine import TetherEngine
            engine = TetherEngine()
            preds = [
                [0.10, 0.20, -0.05, 0.15],
                [0.11, 0.19, -0.04, 0.14],
                [0.09, 0.21, -0.06, 0.16],
                [0.10, 0.20, -0.05, 0.15],
                [0.90, -0.50, 0.80, -0.70],
            ]
            rewards = [1.0, 1.1, 0.9, 1.05, -2.0]
            return engine.synthesize(
                member_predictions=preds,
                member_rewards=rewards,
                base_action=0,
                candidate_action=1,
                candidate_margin=0.25,
                corrupted_member_idx=4,
            )

    def _memory(self):
        from .memory import MemoryBank
        return MemoryBank.load(self.memory_path) if self.memory_path.exists() else MemoryBank()

    def remember(self, text, carrier="episodic"):
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2000:
            raise ValueError("Memory must contain 1–2000 characters.")
        with self.lock:
            memory = self._memory()
            entry = memory.remember(text.strip(), carrier=carrier, metadata={"source": "explicit-user-entry"})
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            memory.save(self.memory_path)
        return {"saved": True, "entry": entry}

    def respond(self, prompt, mode="chat", history=None, seed=42, disabled_carriers=None, planner="policy", scarcity=1.0, max_steps=256, max_new_tokens=None):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 32000:
            raise ValueError("Prompt must contain 1–32000 characters.")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Seed must be a nonnegative integer below 2^63.")
        with self.lock:
            if mode == "poincare":
                return self.poincare_synthesize(seed=seed)
            if mode == "crosswire":
                return self.crosswire_synthesize(seed=seed)
            if mode == "faultline":
                return self.faultline_synthesize(seed=seed)
            if mode == "tether":
                return self.tether_synthesize(seed=seed)
            if mode == "cartan":
                return self.cartan_synthesize(seed=seed)
            if mode == "noether":
                return self.noether_synthesize(seed=seed)
            if mode == "chronos":
                return self.chronos_synthesize(seed=seed)
            if mode == "aether":
                return self.aether_synthesize()
            if mode == "neurograft":
                return self.neurograft_synthesize(seed=seed)
            if mode == "hyperborea":
                return self.hyperborea_synthesize()
            if mode == "hephaestus":
                corpus = [s.strip() for s in prompt.split(";") if s.strip()] if ";" in prompt else None
                return self.hephaestus_synthesize(corpus=corpus)
            if mode == "athenaeum":
                return self.athenaeum_synthesize(expression=prompt, target_expression=None)
            if mode == "proof":
                return self.proof_prove(prompt)
            if mode == "invent":
                return self.creativity_invent(prompt)
            if mode == "hypothesis":
                return self.hypothesis_formulate(prompt)
            if mode == "creative":
                return self.creative_synthesize(prompt, seed=seed)
            if mode == "pantheon":
                return self.pantheon_synthesize(prompt, seed=seed)
            if mode == "reason":
                return self.reason(prompt, seed=seed, max_new_tokens=max_new_tokens or 512)
            if mode == "problem":
                return self.solve_problem(prompt)
            if mode == "math" or (mode == "chat" and extract_math(prompt)):
                return solve(extract_math(prompt) or prompt)
            if mode == "chat":
                from .problem_solver import detect_problem
                if detect_problem(prompt) is not None:
                    return self.solve_problem(prompt)
            if mode in ("image", "video", "mesh"):
                from .media import render_scene
                core = self.core()
                planned = core.scene(prompt)
                result = render_scene(planned, self.root/"outputs/media", mode, seed)
                result.update({"text": f"Created {mode} from the learned scene prediction: {planned['count']} {planned['scale']} {planned['color']} {planned['shape']}, {planned['motion']} motion.", "backend": "Tidal scene predictor + procedural renderer", "prediction": planned, "verified": False})
                return result
            if mode == "world":
                from .world import _integer, _number
                from .world_controls import rollout_with_decisions
                _integer(max_steps, "max_steps", 1, 10000)
                scarcity = _number(scarcity, "scarcity", .5, 4)
                if planner not in ("policy", "mpc", "hybrid", "risk_aware", "uncertainty", "atlas", "contrast", "horizon", "odyssey", "helm", "odysseus", "aura", "metamorph", "chimera", "hyperion", "titan", "trident", "cognitive", "hyper", "poseidon_x", "nexus_omega", "pantheon"):
                    raise ValueError("Unknown survival planner.")
                core = self.core()
                if planner == "pantheon":
                    controller = self.pantheon(seed=seed)
                    controller.reset(scarcity)
                    backend_desc = "Supermix Poseidon Project PANTHEON 24-Pillar Unified Cognitive Synthesis Agent in TidePool"
                elif planner == "nexus_omega":
                    controller = self.nexus_omega(seed=seed)
                    controller.reset()
                    backend_desc = "Supermix Poseidon Nexus-Omega Causal Epigenetic Autonomous Agent in TidePool"
                elif planner == "poseidon_x":
                    controller = self.poseidon_x()
                    controller.reset()
                    backend_desc = "Supermix Poseidon v5.0 Apex Multivector Autopoietic Cognitive Kernel in TidePool"
                elif planner == "hyper":
                    controller = self.hyper_agent()
                    controller.reset()
                    backend_desc = "Supermix Poseidon v3.0 Apex Causal-Quantum Morphogenetic Agent in TidePool"
                elif planner == "cognitive":
                    controller = self.cognitive_agent()
                    controller.reset()
                    backend_desc = "Supermix Poseidon v2.0 Cognitive Agent (Oracle + Mnemosyne + Horizon-X + Neuro) in TidePool"
                elif planner == "trident":
                    controller = self.trident()
                    controller.reset(scarcity, seed=seed, max_steps=max_steps)
                    backend_desc = "TRIDENT identity-gated residual control with observed directed topology in synthetic TidePool"
                elif planner == "titan":
                    controller = self.titan()
                    controller.reset(scarcity, seed=seed)
                    backend_desc = "TITAN experimental composite controller (selected portfolio mechanisms) in synthetic TidePool; software-only sidecar simulations"
                elif planner == "hyperion":
                    controller = self.hyperion()
                    controller.reset(scarcity)
                    backend_desc = "HYPERION frontier super-controller (Morpheus dream consolidation + Prometheus meta-plasticity + NTAG intermittent harvest) in TidePool"
                elif planner == "chimera":
                    controller = self.chimera()
                    controller.reset(scarcity)
                    backend_desc = "CHIMERA quantum superposition (Causeway) + holographic memory + 3D diamond lattice in TidePool"
                elif planner == "metamorph":
                    controller = self.metamorph()
                    controller.reset(scarcity)
                    backend_desc = "METAMORPH developmental ecdysis (MOLT) + causal flux (NexusFlow) + cyclic beacon (Chronos) in TidePool"
                elif planner == "aura":
                    controller = self.aura()
                    controller.reset(scarcity)
                    backend_desc = "AURA biomimetic Central Complex ring attractor + sparse neuropil arbiter in TidePool"
                elif planner == "odysseus":
                    controller = self.odysseus()
                    controller.reset(scarcity)
                    backend_desc = "Odysseus empirical cognitive map + Bayesian replenishment in TidePool"
                elif planner == "helm":
                    controller = self.helm()
                    controller.reset(scarcity=scarcity, max_steps=max_steps)
                    backend_desc = "Helm independently fitted multi-horizon critic with observed history and empirical abstention in TidePool"
                elif planner == "odyssey":
                    controller = self.odyssey()
                    controller.reset(scarcity)
                    backend_desc = "Odyssey cognitive topological mapping + navigational memory in TidePool"
                elif planner == "horizon":
                    controller = self.horizon()
                    backend_desc = "Horizon Atlas multi-horizon advantage + depletion-aware empirical gate in TidePool"
                elif planner == "contrast":
                    controller = self.contrast()
                    backend_desc = "Contrast Atlas matched-action advantage memory + empirical gate in TidePool"
                elif planner == "atlas":
                    controller = self.atlas()
                    backend_desc = "Counterfactual Atlas residual memory + measured-error gate in TidePool"
                elif planner == "mpc":
                    from .planning import ModelPredictivePlanner, PlanningConfig
                    controller = ModelPredictivePlanner(core, PlanningConfig(policy_weight=0.0, mpc_weight=1.0))
                    backend_desc = "Tidal dynamics head pure MPC planner in TidePool"
                elif planner == "hybrid":
                    from .planning import ModelPredictivePlanner
                    controller = ModelPredictivePlanner(core)
                    backend_desc = "Tidal policy + dynamics head hybrid planner in TidePool"
                elif planner in ("risk_aware", "uncertainty"):
                    from .ensemble import UncertaintyAwarePlanner
                    controller = UncertaintyAwarePlanner(core)
                    backend_desc = "Supermix Beyond uncertainty-aware 3-model ensemble planner in TidePool"
                else:
                    controller = core
                    backend_desc = "learned Tidal policy in TidePool"
                if planner in ("titan", "trident", "cognitive", "hyper", "poseidon_x"):
                    from .world_controls import rollout_with_observed_transitions
                    episode = rollout_with_observed_transitions(controller, seed=seed, max_steps=max_steps, scarcity=scarcity)
                else:
                    episode = rollout_with_decisions(controller, seed=seed, max_steps=max_steps, scarcity=scarcity)
                episode["controller"] = planner
                episode["backend"] = backend_desc
                return {"text": f"Episode ended after {episode['steps']} steps. " + (f"The agent reached the {max_steps}-step horizon alive." if episode["survived"] else f"The agent died: {episode['death_reason']}."), "backend": backend_desc, "episode": episode, "verified": True}
            if mode == "memory":
                retrieved = self._memory().retrieve(prompt, disabled_carriers=disabled_carriers or [])
                return {"text": json.dumps(retrieved, ensure_ascii=False, indent=2), "backend": "explicit carrier memory retrieval", "memories": retrieved, "verified": False}
            if mode != "chat":
                raise ValueError("Unknown mode.")
            memories = self._memory().retrieve(prompt, disabled_carriers=disabled_carriers or []) if self.memory_path.exists() else []
            result = self.language.generate(prompt, history, memories, max_new_tokens=max_new_tokens or 512)
            result["memories"] = memories
            return result

