# SYNAPSE: Developmental Connectome Intelligence

**SYNAPSE** is an integrated cognitive, neuromorphic, and weight-space research system combining:
1. **GraftBrain** (`kai9987kai/Graft-brain`) — multi-source recurrent rate connectomes, signed Dale-law synaptic dynamics, adaptive graft plasticity, ellipsoid body heading compass, and sparse Kenyon cell associative memory.
2. **Supermix-Poseidon** (`kai9987kai/Supermix-Poseidon`) — trained Tidal DAgger MoE expert controller (327,230 parameters), recurrent memory state, and verified symbolic problem solving.
3. **Supermix-Expanse-v2** (`kai9987kai/Supermix-Expanse-v2`) — function-preserving dormant expert slots, native latent consolidation blocks, input-dependent graph gates, and evoked zero-subtracted connectome readouts.
4. **Weight-Space Programming & Autonomous Repair** — transaction-safe, bounded numeric weight modifications with parent fingerprinting, atomic rollback, and closed-loop failure diagnosis with writer-disconnected confirmation.
5. **Multi-Model Subcircuit Transplantation** — cross-model donor subcircuit extraction and ridge representation alignment evaluated across a full 6-arm factorial control suite (`correct`, `random`, `shuffled`, `irrelevant`, `disabled`, `permuted_host`).
6. **Causal Mediation & Biological Frontiers** — Pearlian counterfactual decomposition of Natural Direct (NDE) vs Natural Indirect (NIE) effects, Janelia Male CNS Dale's law compliance auditing, and metabolic wiring Pareto-optimal sparsification.
7. **Recursive Self-Improvement (RSI)** — inherited meta-improvement policies ($\phi$) whose accepted successors directly control subsequent candidate proposals, benchmarked on fresh-learner held-out tasks under identical budgets.

---

## Architecture Overview

SYNAPSE operates across coupled timescales:

```mermaid
flowchart TD
    W[Embodied World: 9 Task Backends] -->|155-d sensory obs| S[Retina rays, proprioception, resource cues]
    S --> B[Compass bump + Mushroom Body + Curiosity RND]
    B --> G[Grafted Connectome: 300 Neurons across 8 Sites]
    G --> A[Motor Decoder: 5 continuous commands] --> W
    G <-->|bidirectional residual bridge| P[Poseidon Tidal MoE: 4 Experts, 128-d Memory]
    P -->|Intent & routing| G
    G & P --> E[Episodic & Tool Memory Trace]
    E --> C[Weight-Space Programmer / RSI Meta-Policy]
    C --> T[Inactive Candidate Checkpoint]
    T --> V[Paired Retention & Adaptation Screen]
    V -->|accepted| G
    V -->|accepted| P
```

### 1. Fast Sensorimotor Loop (Every Step)
- Connectome graph: 300 nodes distributed across three biological profiles (`flywire` central brain, `manc` ventral nerve cord, and `optic-lobe` visual system) with 552 graft edges across 8 homolog/interface sites.
- Rate dynamics: $\tau \dot{r} = -r + \sigma(W_{base} r + W_{graft} r + I)$.
- Ellipsoid body ring attractor maintaining a 16-wedge heading direction bump on $S^1$.
- Mushroom body sparse Kenyon cell expansion (top-$k$ WTA) modulating dopaminergic valence.
- Output: 5 continuous motor commands (`drive`, `turn`, `reach`, `height`, `grip`) strictly gated by connectome activity.

### 2. Slower Cognition Loop (Every 8 Steps)
- Bidirectional neural bridge: connectome context modulates Tidal resource representations, while Tidal memory and intent modulate connectome input currents.
- Trained Tidal core checkpoint (`tidal_core.pt`, SHA-256 `a053104a...`) evaluates resource states and routes across 4 MoE experts.
- Shared episodic/tool memory records queries, reasoning traces, and environment feedback.

### 3. Weight-Space Programming & Autonomous Repair Loop
- **Named edit surfaces**: graft weights, router parameters, associative readout weights, and neural bridges. Pretrained Tidal weights remain frozen.
- **Transaction safety**: parent state verification, strict parameter budget bounds, finite checks, and atomic rollback.
- **Autonomous Repair**: closed-loop diagnostic rollouts &rarr; circuit failure diagnosis &rarr; weight-space program synthesis &rarr; inactive candidate screening &rarr; confirmation with writer disconnected.
- **Factorial Donor Transplantation**: extracts portable subcircuits, fits ridge representation alignment maps, and tests causal donor support against 5 rigorous control arms.
- **Pearlian Causal Mediation**: decomposes weight program effects into Natural Direct Effect (NDE on motor decoder) and Natural Indirect Effect (NIE mediated through connectome dynamics).
- **Biological Metabolic Sparsification**: models physical axonal wiring costs, spiking energy, and sweeps Pareto-optimal frontiers.
- **Dale's Principle Enforcement**: audits synaptic sign consistency and projects unconstrained updates onto the biological Dale's law cone.

---

## Master Capability Explorer

SYNAPSE exposes 67 typed, audited service capabilities across 16 architecture families:

| Capability ID | Family | Name | Scope / Intended Role |
|---|---|---|---|
| `reasoning.solve` | reasoning | Problem Forge Solver | Exact verified arithmetic & AST problem solving |
| `reasoning.cartan` | reasoning | Cartan Exterior Solver | Exterior differential systems & involutivity |
| `reasoning.noether` | reasoning | Noether Symmetry Analyzer | Variational symmetries & conservation laws |
| `reasoning.causal` | reasoning | Causal Do-Calculus Solver | Pearl do-calculus intervention estimation |
| `memory.retrieve` | memory | Episodic Memory Retrieval | Query tool & episodic memory store |
| `memory.store` | memory | Episodic Memory Storage | Explicit key/value record persistence |
| `memory.mnemosyne` | memory | Mnemosyne Working Memory | Multi-timescale decaying trace buffer |
| `memory.hologram` | memory | Holographic Vector Memory | HRR circular convolution binding |
| `core.scene` | core | Poseidon Tidal Core | Tidal DAgger expert controller inference |
| `core.tidal` | core | Tidal Core Metadata | MoE expert controller parameter inspection |
| `graph.substrate` | graph | Connectome Substrate | Recurrent connectome rate graph simulation |
| `graph.connectomes` | graph | Drosophila Neuropil Profiles | Connectome registry (flywire, manc, optic-lobe) |
| `graph.successor` | graph | Successor Solver | Bounded successor representation matrix |
| `embodiment.compass` | embodiment | Central Complex Compass | 16-wedge ring attractor heading bump |
| `embodiment.mushroom_body`| embodiment | Mushroom Body Learner | Sparse Kenyon cell valence plasticity |
| `embodiment.curiosity` | embodiment | RND Curiosity Bonus | Prediction error novelty bonus |
| `planning.oracle` | planning | Oracle Trajectory Planner | Forward trajectory rollout predictor |
| `nexus.phase_world` | nexus | NEXUS PhaseWorld POMDP | Discrete phase transition dynamics |
| `reliability.crosswire` | reliability | Crosswire Hypothesis Pruner| Version-space hypothesis pruning |
| `tools.archimedes` | tools | Archimedes Geometric Engine | Exact geometry & constraint evaluation |
| `expanse.male_cns` | expanse | Male CNS Core | Janelia male CNS 11,751 cell-type real connectome |
| `expanse.dormant_expert` | expanse | Dormant Expert Gate | Zero-initialized expert slot wake-up gate |
| `expanse.donor_graft` | expanse | Donor Ridge Map | Cross-model representation alignment map |
| `weightspace.causal_mediation` | weightspace | Causal Mediation Assay | Pearlian NDE/NIE decomposition of weight edits |
| `weightspace.metabolic_pareto` | weightspace | Metabolic Pareto Optimizer | Biological wiring cost & Pareto sparsification |
| `connectome.dales_law_audit` | connectome | Dale's Law Auditor | Janelia neurotransmitter sign compliance audit |
| `rsi.meta_evaluate` | rsi | RSI Meta-Evaluation Screen | Fresh-learner meta-policy benchmark |
| `language.smollm2` | language | SmolLM2 Interface | Optional local instruct tokenization |

---

## Quick Start

### Installation
SYNAPSE is built with Python 3.10+ (tested on Python 3.12) and PyTorch CPU:
```bash
python -m pip install -e .
```

### Launch Interactive Local Workbench
Start the local HTTP workbench server:
```powershell
.\launch.ps1 -Port 8765
# Or on Windows cmd:
Run-Synapse.cmd
```
Then navigate in your browser to:
```
http://127.0.0.1:8765
```

The workbench provides 5 interactive panes:
- **01 Live Workbench**: Real-time canvas rendering of connectome activation graphs, 3D world projections, ray-cast visual sensors, and routing telemetry.
- **02 Capabilities**: Interactive explorer to inspect schemas and invoke all 67 audited capabilities.
- **03 Weight Programs**: Author, preview, apply, and rollback bounded numeric patch packets, run autonomous closed-loop failure repairs, and execute factorial donor transplantation assays.
- **04 Experiments & RSI**: Configure and execute bounded background RSI studies with job monitoring and lineage graphs.
- **05 State Archive**: Save, load, and inspect exact agent and world session checkpoints.

### CLI Usage
```bash
# Inspect agent and connectome status
python -m synapse.cli status

# Run 100 simulation steps in the forage environment
python -m synapse.cli run --steps 100 --save my_checkpoint

# Train with active plasticity for 500 steps
python -m synapse.cli train --steps 500 --save trained_agent

# Run closed-loop autonomous failure diagnosis and repair
python -m synapse.cli autoprogram --diagnostic-episodes 2 --eval-episodes 2

# Run full factorial donor transplantation assay across 6 control arms
python -m synapse.cli transplant --donor-task forage --test-task delayed_match --episodes 2

# Invoke a specific capability via CLI
python -m synapse.cli capabilities --id weightspace.causal_mediation
```

---

## Running the Verification Suite

Run the full pytest suite:
```bash
python -m pytest
```

All 51 test suites pass with 100% verification across:
- Bidirectional neural coupling & bridge preparation
- Acquiring mushroom body learning and reversal
- Exact checkpoint state round-trip & inference freeze
- Bounded patch application, stale parent rejection, and atomic rollback
- Autonomous closed-loop failure diagnosis, synthesis, screening, and confirmation
- Factorial transplantation with all 6 experimental control arms
- Pearlian causal mediation analysis of weight programs (NDE vs NIE)
- Biological connectome metabolic cost and Pareto frontier sweeps
- Dale's principle sign consistency auditing & neurotransmitter projection
- Fresh-learner RSI policy inheritance & receipt verification
- Multi-backend world simulation across all 9 native tasks
- Loopback security, CSP headers, and input boundary validation
- Upstream capability catalog coverage (67 endpoints) & memory consumer wiring
