"""Build a frozen, allowlisted Poseidon model release from a clean Git checkout."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "Kai9987kai/Supermix-Poseidon"
LANGUAGE_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
PREVIOUS_HUB_REVISION = "cfe2b0e6a3e5599eaac8c14b56a6ab3b47f5f953"
EXPERIMENTS = (
    "c260cb232f1635a1-949a615206d8.json",
    "875452fefbf277df-3b1edfb49993.json",
)
HORIZON_EXPERIMENTS = (
    "f83c77efdb0a70ad-ac7304a77e02.json",
    "7eb3ec0702f503b1-a4bac8a67483.json",
)
ODYSSEY_EXPERIMENTS = (
    "cdaf779874a04de6-7dc45ea640b3.json",
    "e492031f870f05da-c9ba9d72f3be.json",
    "3ed044592794f58c-29888499b43a.json",
    "50ce54c9e486ce8f-0fb6374e203a.json",
)
ODYSSEUS_EXPERIMENTS = (
    "1521d677b6f2f75b-c4035a1005d3.json",
)
AURA_EXPERIMENTS = (
    "d9e0d76bd4812c03-b489e9702cff.json",
)
TRAJECTORY_AUDITS = {
    "8491880ca77a2dba-c7284642c7ab.json": "3ed044592794f58c-29888499b43a.json",
    "8b0e6f16a03ed340-a22ce43a85ac.json": "50ce54c9e486ce8f-0fb6374e203a.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def model_card(revision: str, core_hash: str, atlas_hash: str, horizon_hash: str, contrast_hash: str, odyssey_hash: str, version: str, helm_hash: str = "", odysseus_hash: str = "", titan_hash: str = "", trident_hash: str = "") -> str:
    robustness_upgrade = """
The [31 October robustness upgrade](docs/PROJECT_UPGRADE_2026-10-31.md) repairs
exact decimal and scientific-notation input across the solvers and adds the
`metamorphic-audit` related-input tool and offline `episode-risk` screen. Two
fresh 200-case reasoning reports and two 320-answer related-input reports replay
against this release's source. The synthetic episode fixture recomputes supplied
labels to test the software; it is not controller performance evidence.
Historical v3.13 reasoning reports remain under their original paths and retain
their original bytes. Their old source identities are historical evidence and
are not claimed to replay against the current solver. Episode bounds require
the declared fixed-policy and sampling assumptions, do not estimate conditional
per-action risk, and do not activate or promote a controller.
""" if version == "3.14.0" else ""
    return fr"""---
language:
- en
license: other
license_name: mit-and-apache-2.0
license_link: https://huggingface.co/{REPO_ID}/blob/main/THIRD_PARTY.md
library_name: pytorch
base_model:
- HuggingFaceTB/SmolLM2-135M-Instruct
tags:
- world-models
- imitation-learning
- dagger
- mixture-of-experts
- model-predictive-control
- counterfactual-memory
- multi-horizon-advantage
- spatial-memory
- cognitive-mapping
- topological-navigation
- bayesian-replenishment
- empirical-transitions
- memory-carrier-observatory
- quantum-superposition
- vector-symbolic-architecture
- holographic-memory
- grid-cells
- causal-residuals
- identity-gating
- lora
- cpu
- experimental
- synthetic-simulation
---

# Supermix Poseidon v{version}: Local Research Workbench

Version {version} packages a local experimental workbench around the frozen Tidal core, with a separate pretrained conversation backend, optional synthetic controllers, and mathematical research tools. Runtime availability, replay integrity, and measured model performance are separate checks. Additional modules do not imply that the selected core was retrained or that a candidate was promoted.
{robustness_upgrade}

The [30 October reliability upgrade](docs/PROJECT_UPGRADE_2026-10-30.md) adds
independently generated reasoning checks, corrected controller adapters, explicit
caller evidence for the opt-in coordinator, and bounded Poincare diagnostics.
Changed ensemble predictors preserve the baseline until their own calibration
is available. Generated audit reports are finite software checks, not
general model accuracy or a statistical risk guarantee. Mean-curvature relaxation
is a surrogate, separate from geometric Ricci flow or Perelman entropy.

TRIDENT remains available as an opt-in identity-gated controller around the unchanged Tidal checkpoint. It corrects one-step predictions with recent observed prediction errors, admits directed travel only after enough local visits unless that transition has been observed, and requires an override to beat erased and time-shifted residual controls by configured margins. TRIDENT is experimental software for synthetic TidePool; it does not modify or train model weights.

The six-arm paired benchmark compares the frozen core, TRIDENT, erased residual, shifted residual, directed-support-gate-off, and residual-gate-off controls on identical seeds. The receipt stores actions and transitions and independently replays controller decisions and world outcomes. Its four-seed results are descriptive synthetic evidence, not proof of a general performance gain; they do not promote a model.

Additional modules are software simulations or heuristics: Archimedes computes synthetic fluid telemetry; Genesis runs an auxiliary toy ecology; S-Video encodes telemetry; Universal Modder applies static Python hooks; QuantumBot uses a classical state-vector simulator. They do not provide physical RF harvesting, NTAG hardware, analog video, hot-patching, quantum hardware, biological evidence or real-world survival capability. Causeway uses deterministic maximum-probability selection in this controller.

[Source commit](https://github.com/kai9987kai/Supermix-Poseidon/tree/{revision}) ·
[TRIDENT design](docs/TRIDENT_DESIGN.md) · [TRIDENT results](docs/TRIDENT_RESULTS.md) ·
[TITAN design](docs/TITAN_DESIGN.md) · [TITAN results](docs/TITAN_RESULTS.md) ·
[HYPERION results](docs/HYPERION_RESULTS.md) · [Morpheus design](docs/MORPHEUS_DESIGN.md) · [Prometheus design](docs/PROMETHEUS_DESIGN.md) · [Intermittent design](docs/INTERMITTENT_DESIGN.md) · [CHIMERA results](docs/CHIMERA_RESULTS.md) · [Causeway design](docs/CAUSEWAY_DESIGN.md) · [Release guide](docs/HUGGINGFACE_RELEASE.md)

## Components and activation

| Component | Artifact | Status |
|---|---|---|
| Tidal core: 327,230 parameters | `runs/tidal_dagger/core.pt` | Existing selected DAgger policy, default scene/control backend |
| Supervised baseline | `runs/tidal/core.pt` | Comparison checkpoint, original embedded training state |
| SmolLM2: 134,515,008 stored parameters | `models/language/model.safetensors` | Separate unchanged pretrained conversation backend |
| LoRA: 61,440 parameters | `runs/language/adapter/` | Experimental candidate, inactive by default |
| Poseidon-X | `poseidon/poseidon_x.py` | Experimental geometric and homeostatic controller; opt-in |
| Problem Forge | `poseidon/problem_solver.py` | Parsed exact/numerical tools and synthetic demonstrations; evidence scope varies by domain |
| Generated reasoning audit | `poseidon/research/reasoning_audit.py` | Independent numeric/structural checks, refusal controls, source binding and finite replay |
| Related-input reasoning audit | `poseidon/research/metamorphic_audit.py` | Independent answer checks and related-input consistency; replayable finite tools, not general model accuracy |
| Offline episode risk screen | `poseidon/research/episode_risk.py` | Fixed-policy complete-episode binary bounds under declared sampling assumptions; no activation or per-action risk claim |
| Poincare diagnostics | `poseidon/poincare_engine.py` | Five bounded numerical/synthetic diagnostics; no domain-count or controller-safety certification |
| Project NEUROGRAFT-II | `poseidon/neurograft_engine.py`, `poseidon/research/successor_memory_adapter.py` | Successor representations (SR-TD), zero-shot reward revaluation, SWR reverse replay, Graph Laplacian spatial embeddings & trainable graph circuits |
| Project CROSSWIRE | `poseidon/crosswire_engine.py`, `poseidon/research/crosswire_adapter.py` | Version space tracking, active diagnostic probing, all-hypothesis contract verification & closed-loop runtime shielding |
| Project FAULTLINE | `poseidon/faultline_engine.py`, `poseidon/research/faultline_adapter.py` | Condition-stratified joint failure selection, 3-bit parity hyperedge audit, continuous error covariance & minimax panel triage |
| Project TETHER | `poseidon/tether_engine.py`, `poseidon/research/tether_gate.py` | Continuous reliability weighting, 4-view causal invariance, input x gradient saliency, consensus isolation & fallback gate |
| CHRONOS, NOETHER, and CARTAN | `poseidon/chronos.py`, `poseidon/noether.py`, `poseidon/cartan.py` | Experimental mathematical constructions and synthetic telemetry |
| Project HYPERBOREA | `poseidon/hyperborea.py` | Cellular sheaf cohomology, Port-Hamiltonian Dirac structures, Wilsonian RG & Fibonacci anyon braiding |
| Project HEPHAESTUS | `poseidon/hephaestus.py` | Wake-sleep library synthesis, GFlowNet DAG sampling, Koopman EDMD & persistent homology |
| Project ATHENAEUM | `poseidon/athenaeum.py` | Equality saturation e-graphs, Process-Reward MCTS & SINDy equation discovery |
| Project ARCHIMEDES & DA VINCI | `poseidon/proof_engine.py`, `poseidon/creativity_engine.py` | Formal natural-deduction proof certificates & Koestler/TRIZ bisociative invention |
| Project OLYMPUS & PANTHEON | `poseidon/hypothesis_engine.py`, `poseidon/pantheon.py` | Popperian crucial experiment falsification & 24-pillar universal synthesis |
| Clifford Geometric Algebra | `poseidon/geometric_algebra.py` | Complete Cl(3,0) multivector engine, Rotor3D orientations & fluid vorticity |
| Autopoietic Affective System | `poseidon/autopoiesis.py` | Autopoietic homeostasis, PAD affective appraisal & micro-sleep self-repair |
| Hypergraph Theorem Discovery | `poseidon/hypergraph_discovery.py` | Neural-symbolic inductive logic programming with Laplace confidence |
| Continuous 3D OmniEcosystem | `poseidon/ecosystem_3d.py` | Multi-agent trophic dynamics (grazers, predators, symbionts) & duty-cycled pulse signaling |
| Tessera LTL Prover | `poseidon/tessera_prover.py` | Bounded synthetic contract and invariant checks |
| Universal Modder & Reactive Actors | `poseidon/universal_reactive.py` | Static Python hooks with rollback and reactive message passing |
| Active Inference & Free Energy | `poseidon/active_inference.py` | Expected free energy minimization, epistemic curiosity & reserve override |
| Metacognitive Monitor | `poseidon/metacognition.py` | Metacognitive error calibration & dynamic confidence monitoring |
| TRIDENT Identity-Gated Controller | `poseidon/trident.py` | Identity-gated, observed-topology causal residual controller around frozen core; opt-in |
| TRIDENT Paired Replay Receipt | `outputs/trident_experiments/RECEIPT.json` | Six-arm matched-seed controller and world replay; no promotion |
| TITAN Composite Controller | `poseidon/titan.py` | Selected project mechanisms around the frozen Tidal core; opt-in |
| TITAN Paired Replay Receipt | `outputs/titan_experiments/RECEIPT.json` | Six-arm matched-seed controller and world replay; no promotion |
| HYPERION Super-Controller | `poseidon/hyperion.py` | Separate opt-in experimental controller |
| Morpheus Consolidation & Replay | `poseidon/morpheus.py` | Simulated trace compression and replay summaries |
| Prometheus Meta-Plasticity | `poseidon/prometheus.py` | Rolling Shannon entropy meta-plasticity & 4-drive homeostatic allostasis |
| Intermittent Computing | `poseidon/intermittent.py` | Software model of harvested-energy state and compact frames |
| NexusSearch Federated Index | `poseidon/nexus_search.py` | Dual HRR associative vector & 3D diamond lattice spatial search |
| HYPERION Receipt | `outputs/hyperion_experiments/RECEIPT.json` | Paired multi-arm evaluation benchmark receipt |
| CHIMERA Super-Controller | `poseidon/chimera.py` | Unified quantum superposition, holographic memory & diamond lattice, opt-in |
| Causeway Decision Engine | `poseidon/causeway.py` | 6D complex superposition, non-Hermitian gain & Born collapse |
| Holographic Concept Memory | `poseidon/hologram.py` | 256-d HRR circular convolution binding & associative clean-up |
| Diamond Lattice Neuropil | `poseidon/lattice.py` | 3D FCC tetrahedral geodesic multi-scale grid cells |
| CHIMERA Receipt | `outputs/chimera_experiments/RECEIPT.json` | Paired multi-arm evaluation benchmark receipt |
| METAMORPH Controller | `poseidon/metamorph.py` | Unified developmental ecdysis, causal flux & cyclic sync controller, opt-in |
| MOLT Morphogenesis | `poseidon/molt.py` | Larval/pupa/imago developmental instars & exuvia hashing |
| NexusFlow Causal Flux | `poseidon/nexusflow.py` | Hydraulic replenishment gradients & superposition interference checking |
| Chronos & Ghost Auditor | `poseidon/chronos.py` | Simulated periodic ticks and counterfactual SDI tracking |
| METAMORPH Receipt | `outputs/metamorph_experiments/RECEIPT.json` | Paired multi-arm evaluation benchmark receipt |
| AURA Biomimetic Controller | `poseidon/aura.py` | 16-wedge CX-ring attractor & 128-KC sparse neuropil arbiter, opt-in controller |
| Tessera Macro-Commons | `outputs/tessera_experiments/RECEIPT.json` | Independent dual-lineage ratified macro-actions, opt-in commons |
| Mnemorph Memory Assays | `outputs/mnemorph/RECEIPT.json` | Typed structural lesion and associative regrowth receipts |
| Odysseus Navigator | `outputs/odysseus/atlas.json` | Empirical macro-action transitions & Bayesian replenishment LCB, opt-in candidate |
| Helm Critic v0.6 | `outputs/helm/atlas.json` | Fitted multi-horizon critic with observed history, opt-in candidate |
| Odyssey Atlas v5 | `outputs/odyssey/atlas.json` | Spatial cognitive map and replenishment-decay waypoint planner, opt-in candidate |
| Horizon Atlas v4 | `outputs/horizon/atlas.json` | Fitted multi-horizon return advantage, opt-in planner |
| Contrast Atlas | `outputs/contrast/atlas.json` | Matched-action residual memory, opt-in candidate |
| Counterfactual Atlas | `outputs/atlas/atlas.json` | Fitted residual memory, opt-in candidate |

Tidal uses 512 hashed lemma/bigram features, a 128-wide state, four gated experts,
and two recurrent updates within each decision. It predicts five controlled scene
attributes, six simulation actions and 16 state deltas. Its trained routing is
dense across four experts; experimental sparse/ensemble modules are separate.
The default protocol resets hidden state at each decision. Fluent conversation,
procedural geometry/rendering, external carrier storage and exact maths use
separate components. No unified multimodal, real-world survival, biological
connectome, consciousness or general autonomous-learning capability is claimed.

## Download and run

```bash
python -m pip install huggingface-hub
hf download {REPO_ID} --revision v{version} --local-dir Poseidon-release
cd Poseidon-release
python -m pip install -e .
python tools/verify_huggingface_release.py . --smoke
python -m poseidon reasoning-audit --seed 20261030 --samples 32 --output outputs/reasoning-audit.json
python -m poseidon verify-reasoning-audit outputs/reasoning-audit.json
python -m poseidon metamorphic-audit --seed 20261031 --samples 32 --output outputs/metamorphic.json
python -m poseidon verify-metamorphic-audit outputs/metamorphic.json
python -m poseidon episode-risk study.json --output outputs/episode-risk.json
python -m poseidon verify-episode-risk outputs/episode-risk.json
python -m poseidon serve --port 8787
```

Open http://127.0.0.1:8787. Use Python 3.10+ with a compatible CPU PyTorch 2.6+
installation. The snapshot includes weights and tokenizer; no additional model
download is needed for the workbench. Native checkpoints use
`torch.load(..., weights_only=True)` and language loading disables remote code.
`AutoModel.from_pretrained('{REPO_ID}')` is unsupported: use the packaged runtime.
For immutable provenance, pin the Hub commit SHA returned by the release upload.

```python
from poseidon.runtime import Poseidon
runtime = Poseidon('.')  # run from the downloaded release directory
print(runtime.respond('Survive', mode='world', seed=42, planner='policy'))
print(runtime.respond('Survive', mode='world', seed=42, planner='odyssey'))
print(runtime.respond('Survive', mode='world', seed=42, planner='horizon'))
```

Default language uses pinned `HuggingFaceTB/SmolLM2-135M-Instruct` at
`{LANGUAGE_REVISION}`. Explicitly test the candidate with
`python -m poseidon chat 'Explain why seasons change.' --adapter runs/language/adapter`.
Generation uses at most a 1,024-token local input budget and 256 generated tokens.
This release does not automatically activate the adapter, Atlas, Contrast, Horizon or Odyssey.

## Training and measured limits

The supervised Tidal run used 120,000 synthetic examples and 360,000 exposures.
DAgger selected round two using validation seeds; 769,600 exposures is the aggregate
over the four-round run, not a precise exposure count of the selected weights.
Stored evaluation reports describe controlled scene vocabulary and synthetic
TidePool survival. Reported 100% results do not imply real-world generalization.

Odyssey Atlas v0.5 constructs an episodic spatial and topological cognitive map
strictly from 16-d observation streams without simulator cheats or global coordinate
leakage. It tracks environmental replenishment dynamics
($\\lambda_{{food}} = 0.007 / scarcity, \\lambda_{{water}} = 0.014 / scarcity$), plans
multi-hop navigational routes via breadth-first search with physical travel cost
penalties, enforces pre-transit rest guards when stamina < 0.20, and prioritizes
shelters during impending storms. The map's scarcity now matches the selected
world settings. The v0.5.0 scarcity-4 suite used the default map setting of 2.5
in 12 episodes; its preserved measurements have that limitation.

Two fresh four-seed, ten-arm, 64-step suites used scarcity 2.5 and 4.0 with no
tuning between them. Calibrated Odyssey matched policy with zero overrides.
Ungated Odyssey's mean episode reward differences were +0.002331 and +0.041827,
but the stress result was dominated by one seed. The separate H=16 auditor
replayed 91 anchors, 546 actual multi-step branches and 8,712 branch transitions.
Six of 11 ungated overrides had negative H-step return advantages. These small
synthetic development suites do not establish superiority or a safety guarantee.
The candidate remains opt-in and the core policy remains the default.

The historical experiment audit forked six one-step reserve-utility branches;
it did not validate multi-step reward margins. The new audit records every
continuation action and replays returns without neural weights. It verifies
execution and arithmetic, not neural authorship. Its strict parent checks also
reconstruct maps, validate fitting partitions, paired cardinality, forecasts
and terminal states. See docs/TRAJECTORY_AUDIT.md and docs/trajectory-evaluation.json.

Horizon Atlas v0.4 indexes multi-horizon returns over H=16 lookahead steps across
12 training episodes (1,728 transitions), evaluates physical depletion guards, and
calibrates empirical advantage error radii across 8 disjoint calibration episodes
(1,152 transitions). Returns use undiscounted rewards. Empirical radii describe
the sampled fitting data and are not calibrated failure probabilities. Historical
episode measurements and their one-step audit limits are in docs/HORIZON_RESULTS.md.

The candidate LoRA ran 256 steps, saw 512 examples and 42,104 tokens, and changed
development loss from 0.960879 to 0.958487. That does not establish better conversation
or reasoning; the dataset overlaps the upstream model's instruction-training
distribution. The candidate remains inactive. The conversation model can hallucinate.

## Provenance and file integrity

Source revision: `{revision}`.
Active core file SHA-256: `{core_hash}`.
Atlas canonical payload SHA-256: `{atlas_hash}`.
Contrast canonical payload SHA-256: `{contrast_hash}`.
Horizon canonical payload SHA-256: `{horizon_hash}`.
Odyssey canonical payload SHA-256: `{odyssey_hash}`.
Helm canonical payload SHA-256: `{helm_hash}`.
TITAN receipt SHA-256: `{titan_hash}`.
TRIDENT receipt SHA-256: `{trident_hash}`.
`manifest.json` and `SHA256SUMS.txt` inventory the staged files. The manifest binds
source, weights, upstream provenance and the exact experiment receipts.
Checksums diagnose integrity; they do not prove scientific validity or authorship.
`SOURCE_README.md` is the project's operational README. Root `core.pt`,
`core_base.pt`, and `adapter/` are compatibility copies of the native layout.
User memory, raw language datasets, caches, logs, rejected Atlas artifacts and
language optimizer continuation state are excluded.

## Superseded historical release

The old `ensemble_adapted.pt` is archived as `legacy/ensemble_adapted.pt` and retained
in Hub history. Its archived bytes are inventoried for integrity, without any
capability validation, and are not loaded by the default runtime. Its former
promotion claim is invalidated. The old demonstration video is also archived.
`beyond_audit_receipt.json` now records supersession, while original evidence and
the previous model card are preserved under `legacy/`. The earlier card's claims
of authentic fly-connectome wiring, calibrated failure probabilities, learned
delayed-memory performance, zero-width survival intervals and verified ensemble
promotion are not supported by this release. See the repaired source documentation.

## License and attribution

Original Poseidon code, Tidal checkpoints and Atlas are MIT (`LICENSE`). Bundled
SmolLM2 weights/tokenizer, source card and related adapter retain Apache-2.0 terms
and attribution (`licenses/APACHE-2.0.txt`, `models/language/README.md`). The local
adapter used `HuggingFaceTB/smol-smoltalk` at
`f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`; its source card and selection manifest
are included, without raw dataset redistribution. Dependencies retain their own
licenses. See `THIRD_PARTY.md`. The Hub metadata uses a composite license label
to avoid relicensing upstream artifacts as MIT.
"""


def package(output: Path, repo_id: str = REPO_ID) -> dict:
    sys.path.insert(0, str(ROOT))
    from poseidon import __version__
    from tools.verify_huggingface_release import (
        BUILDABLE_RELEASE_VERSIONS, REASONING_AUDIT_PATHS, ROBUSTNESS_AUDIT_PATHS,
        reasoning_audit_paths, verify_reasoning_audits, verify_robustness_audits,
    )
    if __version__ not in BUILDABLE_RELEASE_VERSIONS:
        raise ValueError(f"unsupported version {__version__}")
    if repo_id != REPO_ID:
        raise ValueError("this release card is bound to Kai9987kai/Supermix-Poseidon")
    if git("status", "--porcelain"):
        raise ValueError("commit intended source changes before packaging; checkout must be clean")
    revision = git("rev-parse", "HEAD")
    output = output.resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("output must not contain or replace the source checkout")
    if output.exists():
        raise ValueError("output already exists; choose a fresh directory")

    sys.path.insert(0, str(ROOT))
    import torch
    from poseidon.atlas import CounterfactualAtlas
    from poseidon.contrast import ContrastAtlas
    from poseidon.horizon import HorizonAtlas
    from poseidon.odyssey import OdysseyAtlas
    from poseidon.core import CoreRuntime, active_core_path
    from poseidon.experiments import load_and_verify
    from poseidon.trajectory_audit import load_artifacts, load_json, verify_parent, verify_audit
    torch.set_num_threads(2)
    core_path = active_core_path(ROOT)
    core_hash = sha256(core_path)
    pointer = json.loads((ROOT / "runs/active_core.json").read_text(encoding="utf-8"))
    if pointer.get("sha256") != core_hash:
        raise ValueError("active pointer does not identify actual checkpoint bytes")
    core = CoreRuntime(core_path)
    atlas = CounterfactualAtlas.load(core, ROOT / "outputs/atlas/atlas.json")
    atlas_hash = atlas.artifact["sha256"]
    contrast = ContrastAtlas.load(core, ROOT / "outputs/contrast/atlas.json")
    contrast_hash = contrast.artifact["sha256"]
    horizon = HorizonAtlas.load(core, ROOT / "outputs/horizon/atlas.json")
    horizon_hash = horizon.artifact["sha256"]
    odyssey = OdysseyAtlas.load(core, ROOT / "outputs/odyssey/atlas.json")
    odyssey_hash = odyssey.artifact["sha256"]
    from poseidon.helm import HelmCritic, stable_json
    from poseidon.helm_experiment import verify_helm_receipt
    helm = HelmCritic.load(core, ROOT / "outputs/helm/atlas.json")
    helm_hash = helm.artifact["sha256"]
    evaluation = stable_json(ROOT / "docs/helm-evaluation.json")
    helm_names = evaluation.get("receipts")
    if (evaluation.get("status") != "completed-evaluation" or not isinstance(helm_names, list) or
            not 2 <= len(helm_names) <= 8 or len(set(helm_names)) != len(helm_names) or
            any(not isinstance(name, str) or Path(name).name != name or not name.endswith(".json") for name in helm_names)):
        raise ValueError("Helm release requires a frozen complete evaluation profile")
    helm_experiment_verification = {
        name: verify_helm_receipt(stable_json(ROOT / "outputs/helm_experiments" / name)) for name in helm_names}
    if any(stable_json(ROOT / "outputs/helm_experiments" / name)["critic"]["sha256"] != helm_hash for name in helm_names):
        raise ValueError("Helm evaluation is bound to a different fitted critic")

    from poseidon.odysseus import OdysseusAtlas
    from poseidon.odysseus_experiment import verify_odysseus_receipt
    from poseidon.mco import verify_mco_receipt
    odysseus = OdysseusAtlas.load(core, ROOT / "outputs/odysseus/atlas.json")
    odysseus_hash = odysseus.artifact["sha256"]
    odysseus_experiment_verification = {
        name: verify_odysseus_receipt(json.loads((ROOT / "outputs/odysseus_experiments" / name).read_text(encoding="utf-8")))
        for name in ODYSSEUS_EXPERIMENTS
    }
    mco_experiment_verification = {
        "RECEIPT.json": verify_mco_receipt(json.loads((ROOT / "outputs/mco_experiments/RECEIPT.json").read_text(encoding="utf-8")))
    }

    from poseidon.aura_experiment import verify_aura_receipt
    from poseidon.tessera_experiment import verify_tessera_receipt
    from poseidon.mnemorph import verify_mnemorph_receipt
    from poseidon.metamorph_experiment import verify_metamorph_receipt
    from poseidon.chimera_experiment import verify_chimera_receipt
    from poseidon.hyperion_experiment import verify_hyperion_receipt
    from poseidon.titan_experiment import verify_titan_receipt
    from poseidon.trident_experiment import verify_trident_receipt
    aura_experiment_verification = {
        name: verify_aura_receipt(json.loads((ROOT / "outputs/aura_experiments" / name).read_text(encoding="utf-8")))
        for name in AURA_EXPERIMENTS
    }
    tessera_experiment_verification = {
        "RECEIPT.json": verify_tessera_receipt(json.loads((ROOT / "outputs/tessera_experiments/RECEIPT.json").read_text(encoding="utf-8")))
    }
    mnemorph_experiment_verification = {
        "RECEIPT.json": verify_mnemorph_receipt(json.loads((ROOT / "outputs/mnemorph/RECEIPT.json").read_text(encoding="utf-8")))
    }
    metamorph_experiment_verification = {
        "RECEIPT.json": verify_metamorph_receipt(json.loads((ROOT / "outputs/metamorph_experiments/RECEIPT.json").read_text(encoding="utf-8")))
    }
    chimera_experiment_verification = {
        "RECEIPT.json": verify_chimera_receipt(json.loads((ROOT / "outputs/chimera_experiments/RECEIPT.json").read_text(encoding="utf-8")))
    }
    hyperion_experiment_verification = {
        "RECEIPT.json": verify_hyperion_receipt(json.loads((ROOT / "outputs/hyperion_experiments/RECEIPT.json").read_text(encoding="utf-8")))
    }
    titan_receipt_path = ROOT / "outputs/titan_experiments/RECEIPT.json"
    titan_receipt = json.loads(titan_receipt_path.read_text(encoding="utf-8"))
    titan_experiment_verification = {
        "RECEIPT.json": verify_titan_receipt(titan_receipt, core_path=core_path)
    }
    titan_receipt_hash = sha256(titan_receipt_path)
    trident_receipt_path = ROOT / "outputs/trident_experiments/RECEIPT.json"
    trident_receipt = json.loads(trident_receipt_path.read_text(encoding="utf-8"))
    trident_experiment_verification = {
        "RECEIPT.json": verify_trident_receipt(trident_receipt, core_path=core_path)
    }
    trident_receipt_hash = sha256(trident_receipt_path)
    reasoning_audit_verification = verify_reasoning_audits(ROOT, version=__version__)
    robustness_audit_verification = verify_robustness_audits(ROOT) if __version__ == "3.14.0" else None

    experiment_verification = {name: load_and_verify(ROOT / "outputs/experiments" / name) for name in EXPERIMENTS}
    evidence_artifacts = load_artifacts(ROOT)
    horizon_experiment_verification = {name: verify_parent(load_json(ROOT / "outputs/horizon_experiments" / name), evidence_artifacts) for name in HORIZON_EXPERIMENTS}
    odyssey_experiment_verification = {name: verify_parent(load_json(ROOT / "outputs/odyssey_experiments" / name), evidence_artifacts) for name in ODYSSEY_EXPERIMENTS}
    trajectory_audit_verification = {
        name: {"parent_path": "outputs/odyssey_experiments/" + parent,
               **verify_audit(load_json(ROOT / "outputs/trajectory_audits" / name),
                              load_json(ROOT / "outputs/odyssey_experiments" / parent), evidence_artifacts)}
        for name, parent in TRAJECTORY_AUDITS.items()
    }

    upstream = json.loads((ROOT / "models/language/manifest.json").read_text(encoding="utf-8"))
    if upstream["revision"] != LANGUAGE_REVISION:
        raise ValueError("unexpected language source revision")
    for name, expected in upstream["files"].items():
        if sha256(ROOT / "models/language" / name) != expected:
            raise ValueError(f"upstream language file changed: {name}")
    from huggingface_hub import hf_hub_download
    historical_assets = {name: Path(hf_hub_download(repo_id, name, revision=PREVIOUS_HUB_REVISION))
                         for name in (".gitattributes", "replay.mp4")}
    legacy_ensemble_hash = "820cfc0cadc9f28f887715a9487bc3f04a7071e8a21e23d5806429bd75ae59bd"
    if sha256(ROOT / "runs/beyond_adapted/ensemble_adapted.pt") != legacy_ensemble_hash:
        raise ValueError("historical ensemble bytes changed")

    copied_sources = {}

    def copy(relative: str, target: str | None = None) -> None:
        requested = ROOT / relative
        source = requested.resolve()
        if not source.is_relative_to(ROOT) or requested.is_symlink() or not source.is_file():
            raise ValueError(f"invalid source artifact: {relative}")
        destination = output / (target or relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = sha256(source)
        if relative in copied_sources and copied_sources[relative] != expected:
            raise ValueError(f"source artifact changed while packaging: {relative}")
        shutil.copyfile(source, destination)
        if sha256(destination) != expected or sha256(source) != expected:
            raise ValueError(f"source artifact changed while copying: {relative}")
        copied_sources[relative] = expected

    output.mkdir(parents=True, exist_ok=True)
    source_files = git("ls-files", "-z").split("\0")
    for name in source_files:
        if name:
            target = {"README.md": "SOURCE_README.md", ".gitignore": "SOURCE_GITIGNORE"}.get(name)
            copy(name, target)
    source_hashes = {name: sha256(ROOT / name) for name in source_files if name}
    # The Hub's preupload API applies incoming .gitignore to weight additions.
    # Source ignores training artifacts; a model release must include them.
    (output / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n.cache/huggingface/\n*.egg-info/\n", encoding="utf-8")

    artifacts = [
        "runs/tidal_dagger/core.pt", "runs/tidal/core.pt",
        "runs/language/adapter/adapter_model.safetensors",
        "outputs/atlas/atlas.json", "outputs/atlas/fit_receipt.json",
        "outputs/contrast/atlas.json", "outputs/contrast/fit_receipt.json",
        "outputs/horizon/atlas.json", "outputs/horizon/fit_receipt.json",
        "outputs/odyssey/atlas.json", "outputs/odyssey/fit_receipt.json",
        "outputs/helm/atlas.json", "outputs/helm/fit_receipt.json",
        "outputs/odysseus/atlas.json", "outputs/odysseus/fit_receipt.json",
        "outputs/mco_experiments/RECEIPT.json",
        "outputs/tessera_experiments/RECEIPT.json",
        "outputs/mnemorph/RECEIPT.json",
        "outputs/metamorph_experiments/RECEIPT.json",
        "outputs/chimera_experiments/RECEIPT.json",
        "outputs/hyperion_experiments/RECEIPT.json",
        "outputs/titan_experiments/RECEIPT.json",
        "outputs/trident_experiments/RECEIPT.json",
        "models/language/manifest.json", "data/language/manifest.json",
        "data/language/SOURCE_CARD.md",
    ] + [f"models/language/{name}" for name in upstream["files"]]
    artifacts += [f"outputs/experiments/{name}" for name in EXPERIMENTS]
    artifacts += [f"outputs/horizon_experiments/{name}" for name in HORIZON_EXPERIMENTS]
    artifacts += [f"outputs/odyssey_experiments/{name}" for name in ODYSSEY_EXPERIMENTS]
    artifacts += [f"outputs/trajectory_audits/{name}" for name in TRAJECTORY_AUDITS]
    # Historical v3.13 report bytes remain available, but their old sources are
    # not represented as replaying against the new version's exact solvers.
    artifacts += list(dict.fromkeys((*REASONING_AUDIT_PATHS, *reasoning_audit_paths(__version__))))
    if robustness_audit_verification is not None:
        artifacts += list(ROBUSTNESS_AUDIT_PATHS)
    artifacts += [f"outputs/helm_experiments/{name}" for name in helm_names]
    artifacts += [f"outputs/odysseus_experiments/{name}" for name in ODYSSEUS_EXPERIMENTS]
    artifacts += [f"outputs/aura_experiments/{name}" for name in AURA_EXPERIMENTS]
    for relative in artifacts:
        copy(relative)
    copy("docs/history/HF_MODEL_CARD_v0.1.md", "legacy/model-card-before-v0.2.md")
    copy("runs/beyond_adapted/ensemble_adapted.pt", "legacy/ensemble_adapted.pt")
    shutil.copyfile(historical_assets[".gitattributes"], output / ".gitattributes")
    # The Hub adds exact LFS rules for newly uploaded large JSON receipts. Stage
    # them before hashing so the remote commit cannot silently change this file.
    attributes_path = output / ".gitattributes"
    attributes = attributes_path.read_text(encoding="utf-8").rstrip("\n")
    rules = set(attributes.splitlines())
    for path in sorted(output.rglob("*.json")):
        if path.stat().st_size >= 10 * 1024 * 1024:
            rule = path.relative_to(output).as_posix() + " filter=lfs diff=lfs merge=lfs -text"
            if rule not in rules:
                attributes += "\n" + rule
                rules.add(rule)
    attributes_path.write_text(attributes + "\n", encoding="utf-8")
    shutil.copyfile(historical_assets["replay.mp4"], output / "legacy/replay-before-v0.2.mp4")

    # Original bytes remain available; a current envelope retracts the old promotion.
    superseded = json.loads((ROOT / "runs/beyond_adapted/audit_receipt.json").read_text(encoding="utf-8"))
    write_json(output / "legacy/beyond-audit-before-v0.2.json", superseded["historical_receipt"])
    write_json(output / "beyond_audit_receipt.json", superseded)
    (output / "legacy/README.md").write_text(
        "# Historical, superseded evidence\n\nThese documents preserve the pre-v0.2 release. "
        "They do not establish current ensemble promotion, calibrated risk, biological wiring, "
        "or learned recurrent-memory performance. Read the root model card and docs/ATLAS_RESULTS.md.\n",
        encoding="utf-8",
    )
    for source, target in {
        "runs/tidal_dagger/core.pt": "core.pt", "runs/tidal/core.pt": "core_base.pt",
        "runs/active_core.json": "active_core.json", "runs/tidal_dagger/report.json": "dagger_report.json",
        "runs/tidal/receipt.json": "receipt.json", "runs/tidal/data_manifest.json": "data_manifest.json",
        "runs/calibration.json": "calibration.json", "runs/evaluation.json": "evaluation.json",
    }.items():
        copy(source, target)
    write_json(output / "config.json", core.payload["config"])

    adapter_config = json.loads((ROOT / "runs/language/adapter/adapter_config.json").read_text(encoding="utf-8"))
    adapter_config.update(base_model_name_or_path="HuggingFaceTB/SmolLM2-135M-Instruct", revision=LANGUAGE_REVISION)
    adapter_card = """---
base_model: HuggingFaceTB/SmolLM2-135M-Instruct
license: apache-2.0
tags:
- peft
- lora
- experimental
---
# Poseidon candidate LoRA (inactive)

61,440 trainable parameters: rank 8, alpha 16, q_proj/v_proj on layers 26-29.
256 steps, 512 examples, 42,104 tokens; dev loss 0.960879 to 0.958487.
No improvement in conversation or reasoning has been established. Load explicitly
with the packaged runtime and the pinned SmolLM2 base; default inference omits it.
See runs/language/report.json, data/language/SOURCE_CARD.md and the root model card.
"""
    for directory in ("runs/language/adapter", "adapter"):
        copy("runs/language/adapter/adapter_model.safetensors", directory + "/adapter_model.safetensors")
        write_json(output / directory / "adapter_config.json", adapter_config)
        (output / directory / "README.md").write_text(adapter_card, encoding="utf-8")
    (output / "README.md").write_text(model_card(revision, core_hash, atlas_hash, horizon_hash, contrast_hash, odyssey_hash, __version__, helm_hash, odysseus_hash, titan_receipt_hash, trident_receipt_hash), encoding="utf-8")

    status_output = git("status", "--porcelain")
    if status_output or git("rev-parse", "HEAD") != revision:
        raise ValueError(f"source checkout changed while packaging: {status_output!r}")
    for name, expected in source_hashes.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"source changed while packaging: {name}")
    for name, expected in copied_sources.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"source artifact changed while packaging: {name}")
    if (sha256(core_path) != core_hash or
            sha256(ROOT / "outputs/atlas/atlas.json") != sha256(output / "outputs/atlas/atlas.json") or
            sha256(ROOT / "outputs/odyssey/atlas.json") != sha256(output / "outputs/odyssey/atlas.json") or
            sha256(ROOT / "outputs/odysseus/atlas.json") != sha256(output / "outputs/odysseus/atlas.json")):
        raise ValueError("core, Atlas, Odyssey, or Odysseus changed while packaging")

    files = [{"path": p.relative_to(output).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(output.rglob("*")) if p.is_file()]
    manifest = {
        "schema": "poseidon-huggingface-release-v1", "repo_id": repo_id,
        "version": __version__, "source_revision": revision, "files": files,
        "previous_hub_revision": PREVIOUS_HUB_REVISION,
        "historical_artifacts": ["legacy/ensemble_adapted.pt", "legacy/replay-before-v0.2.mp4",
                                 "legacy/beyond-audit-before-v0.2.json", "legacy/model-card-before-v0.2.md"],
        "core_sha256": core_hash, "atlas_artifact_sha256": atlas_hash,
        "contrast_artifact_sha256": contrast_hash,
        "horizon_artifact_sha256": horizon_hash,
        "odyssey_artifact_sha256": odyssey_hash,
        "helm_artifact_sha256": helm_hash,
        "odysseus_artifact_sha256": odysseus_hash,
        "upstream_language": upstream, "source_file_sha256": source_hashes,
        "release_transforms": {"README.md": "Hub model card; original becomes SOURCE_README.md",
                               ".gitignore": "Hub upload rules; original becomes SOURCE_GITIGNORE",
                               ".gitattributes": "historical LFS rules plus explicit large JSON receipt paths",
                               "runs/language/adapter/README.md": "authored inactive candidate card",
                               "runs/language/adapter/adapter_config.json": "portable upstream id and pinned revision"},
        "experiment_verification": experiment_verification,
        "horizon_experiment_verification": horizon_experiment_verification,
        "odyssey_experiment_verification": odyssey_experiment_verification,
        "trajectory_audit_verification": trajectory_audit_verification,
        "helm_experiment_verification": helm_experiment_verification,
        "odysseus_experiment_verification": odysseus_experiment_verification,
        "mco_experiment_verification": mco_experiment_verification,
        "aura_experiment_verification": aura_experiment_verification,
        "tessera_experiment_verification": tessera_experiment_verification,
        "mnemorph_experiment_verification": mnemorph_experiment_verification,
        "metamorph_experiment_verification": metamorph_experiment_verification,
        "chimera_experiment_verification": chimera_experiment_verification,
        "hyperion_experiment_verification": hyperion_experiment_verification,
        "titan_receipt_sha256": titan_receipt_hash,
        "titan_experiment_verification": titan_experiment_verification,
        "trident_receipt_sha256": trident_receipt_hash,
        "trident_experiment_verification": trident_experiment_verification,
        "reasoning_audit_verification": reasoning_audit_verification,
        "robustness_audit_verification": robustness_audit_verification,
        "build_environment": {name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "peft", "safetensors", "numpy", "scipy", "huggingface-hub")},
        "activation": {"core": "existing-dagger-selection", "language": "unchanged-upstream-base",
                       "language_adapter": "inactive-candidate", "atlas": "opt-in-experiment",
                       "contrast": "opt-in-experiment", "horizon": "opt-in-experiment",
                       "odyssey": "opt-in-experiment",
                       "helm": "opt-in-experiment",
                       "odysseus": "opt-in-experiment",
                       "mco": "factual-observatory-benchmark",
                       "aura": "opt-in-biomimetic-controller",
                       "tessera": "ratified-macro-commons",
                       "mnemorph": "archaeological-memory-assays",
                       "metamorph": "opt-in-metamorphic-controller",
                       "chimera": "opt-in-classical-composite-controller",
                       "hyperion": "opt-in-composite-controller",
                       "titan": "opt-in-experimental-composite-controller-no-promotion",
                       "trident": "opt-in-identity-gated-directed-residual-controller-no-promotion",
                       "poseidon_x": "opt-in-apex-geometric-autopoietic-kernel",
                       "autopoiesis": "homeostatic-pad-affect-and-epistemic-curiosity",
                       "geometric_algebra": "clifford-cl30-multivector-kinematics",
                       "hypergraph": "neural-symbolic-inductive-theorems",
                       "omniecosystem": "multi-agent-trophic-continuous-3d",
                       "tessera_prover": "ltl-invariant-safety-shield",
                       "universal_reactive": "static-python-hooks-with-rollback-and-reactive-message-passing",
                       "active_inference": "expected-free-energy-minimization",
                       "morpheus": "simulated-transition-trace-and-replay",
                       "prometheus": "synthetic-entropy-and-drive-heuristics",
                       "intermittent": "simulated-energy-state-and-compact-frame",
                       "nexus_search": "federated-associative-vector-lattice-indexing",
                       "external_adapter": "typed-3d-simulation-telemetry-protocol",
                       "nexus_omega": "causal-epigenetic-self-experimenting-discovery-agent",
                       "problem_forge": "parsed-exact-numerical-tools-and-synthetic-demonstrations-with-scoped-evidence",
                       "reasoning_audit": "finite-generated-suite-with-independent-checks-and-replay-no-statistical-guarantee",
                       "metamorphic_audit": "finite-related-input-tool-audit-with-independent-correctness-no-general-model-claim",
                       "episode_risk": "offline-complete-episode-screen-of-supplied-labels-no-action-risk-or-promotion",
                       "poincare": "bounded-numerical-synthetic-diagnostics-no-safety-certification",
                       "neurograft_ii": "successor-representation-swr-reverse-replay-graph-laplacian-trainable-circuit-engine",
                       "crosswire": "crosswire-version-space-active-diagnostic-probing-runtime-shield",
                       "faultline": "faultline-joint-failure-condition-stratified-panel-selection-continuous-covariance",
                       "tether": "tether-continuous-reliability-causal-invariance-fallback-gate",
                       "cartan": "cartan-moving-frames-pt-symmetry-albert-algebra-holographic-turing-engine",
                       "noether": "noether-symmetry-quantum-darwinism-hyperbolic-pde-berry-curv-aditi-engine",
                       "chronos": "relativity-cosmology-hawking-gravitational-wave-kerr-engine",
                       "aether": "gauge-field-nonequilibrium-thermo-fisher-rao-symplectic-open-game-engine",
                       "neurograft": "multispecies-connectome-ltc-trimonoaminergic-virtual-lesion-engine",
                       "hyperborea": "cellular-sheaf-dirac-port-hamiltonian-wilsonian-rg-anyon-engine",
                       "hephaestus": "wake-sleep-library-gflownet-koopman-edmd-tda-engine",
                       "athenaeum": "egraph-equality-saturation-prm-mcts-sindy-engine",
                       "archimedes": "natural-deduction-formal-proof-certificate-engine",
                       "da_vinci": "koestler-bisociation-and-triz-contradiction-engine",
                       "olympus": "popperian-crucial-experiment-falsification-engine",
                       "pantheon": "24-pillar-universal-cross-domain-synthesis-engine",
                       "noosphere": "dialectical-tribunal-and-fauconnier-turner-blending",
                       "legacy_ensemble": "historical-unvalidated-not-loaded"},
    }
    write_json(output / "manifest.json", manifest)
    sums = files + [{"path": "manifest.json", "sha256": sha256(output / "manifest.json")}]
    (output / "SHA256SUMS.txt").write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in sums), encoding="utf-8")
    return {"output": str(output), "source_revision": revision, "files": len(files) + 2,
            "bytes": sum(row["size"] for row in files), "manifest_sha256": sha256(output / "manifest.json"),
            "core_sha256": core_hash, "atlas_artifact_sha256": atlas_hash,
            "contrast_artifact_sha256": contrast_hash, "horizon_artifact_sha256": horizon_hash,
            "odyssey_artifact_sha256": odyssey_hash, "helm_artifact_sha256": helm_hash,
            "odysseus_artifact_sha256": odysseus_hash,
            "titan_receipt_sha256": titan_receipt_hash,
            "trident_receipt_sha256": trident_receipt_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()
    print(json.dumps(package(args.output, args.repo_id), indent=2))


if __name__ == "__main__":
    main()
