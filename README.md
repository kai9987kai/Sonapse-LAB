---
language:
- en
tags:
- connectome
- neuromorphic
- weight-space-programming
- continual-learning
- pytorch
- neurogenesis
- homeostasis
- dual-process-reasoning
license: apache-2.0
---

# SYNAPSE: Developmental Connectome Intelligence & Weight-Space Lab (SAIL Edition)

**Current evidence:** SYNAPSE is a CPU research workbench with an integrated
executed-transition learner and many standalone experiments. Module availability
does not establish embodied capability or biological fidelity. Planning is
experimental and off by default. Read the [design recovery audit](docs/design-audit.md),
[Antigravity contract audit](docs/antigravity-audit.md), [neural graft training specification](docs/training-graft.md),
[connectome source & verification guide](docs/connectome-data.md), and
[compute, training & connectome pilot report](docs/results/2026-11-07-compute-training-connectome.md)
for restored contracts, verified datasets, and measured pilot outcomes.

A unified, CPU-first embodied cognitive research platform combining signed recurrent neural graphs, a frozen Poseidon Tidal MoE DAgger controller, continuous online connectome neurogenesis, interoceptive homeostatic drive regulation, MiMoMix dual-process cognitive deliberation, experience-grounded prediction, and direct weight-space programming with inherited recursive self-improvement (RSI).

Now featuring the **Sonapse Adaptive Intelligence Laboratory (SAIL)** research track: utility-driven continual backpropagation, trainable student-teacher sleep consolidation, universal fast-weight memory (uTTT & Titans associative learning), recurrent multi-step world models with MPC planning ($H \in \{1, 3, 5, 10\}$), verifiable ReLIFT curriculum challenges, and causal connectome structural evolution.

Live telemetry, real-time 3D anatomical connectome visualization, and zero-warping virtual world observatory are served via a local web interface and audited CLI tools.

## Architecture

```mermaid
flowchart TD
    W[Complex Dynamic Virtual Ecology] -->|Observations, Threat, Friction, Light| S[Sensory Stream: Rangefinder, Contrast, Cues]
    S --> A[GraftBrain Recurrent Rate Connectome]
    A <-->|Residual Bridges| T[Frozen Poseidon Tidal MoE Controller]
    
    %% Astrocytic and Homeostatic Neuromodulation
    H[Homeostatic Drive Engine] -->|Vitals: Energy, Stamina, Health, Threat, Hunger| N[Neuromodulatory Field: DA, 5-HT, OA, ACh]
    N --> A
    N --> G[Astrocytic Calcium Reservoir]
    G --> A
    
    %% Fast Sensorimotor vs Deliberative Cognition
    A --> M[Fast Motor Actuators: Drive, Turn, Reach, Height, Grip]
    M --> R[MiMoMix Dual-Process Reasoner]
    A --> R
    T --> R
    R -->|System 1 Fast Reflex or System 2 ACT Mental Search| D[Deliberated Action]
    
    %% Experience & Outcome Planning
    E[Experience World Model Ensemble] --> P[Bounded 1–5 Step Experience Planner]
    SM[Surprise Episodic Outcome Memory] --> P
    D --> P
    P --> X[Executed Action]
    X -->|Dynamic Throttle by Stamina Fatigue| W
    
    %% Structural & Weight-Space Plasticity
    A -->|Online Welford Covariance| NG[Archimedes Connectome Neurogenesis]
    N -->|Dopamine RPE & Astrocyte Calcium Gating| NG
    NG -->|Synaptogenesis & Apoptosis under Dale's Law| A
    
    %% Weight Space Programming & RSI
    X --> EXP[Rollout Experience & Causal Diagnostics]
    EXP --> WSP[Weight-Space Programmer & Transaction Ledger]
    WSP -->|Validated Inactive Patch & Atomic Rollback| A
    WSP --> RSI[Inherited Meta-Improvement Policy]
```

## System Components & Evidence Boundaries

| Subsystem | Architectural Role | Governing Dynamics & Evidence Boundary |
|---|---|---|
| **Recurrent Connectome Substrate** | Fast sensorimotor rate dynamics | Signed base connectivity across FlyWire, MANC, and Optic Lobe profiles, 8 inter-module graft sites, Dale's law compliance, and ellipsoid body compass bump. |
| **Tidal MoE Controller** | Pretrained goal & intent donor | Pinned Poseidon Tidal core (`tidal_core.pt`, SHA-256 `a053104a...`), 4 experts, 128 hidden width, 2 recurrent steps, 6 categorical intent classes (`rest`, `forage`, `drink`, `shelter`, `explore`, `flee`). Kept frozen during ordinary execution. |
| **Archimedes Neurogenesis Engine** | Dynamic structural plasticity | Continuous online Welford covariance tracking; apoptosis of inactive synapses ($|w| < \theta_{\text{prune}}$); synaptogenesis of high-covariance neural pairs gated by Dopamine RPE and Astrocytic $[\text{Ca}^{2+}]$; strict Dale's law sign preservation. |
| **Supermix Homeostatic Drives** | Interoceptive physiological regulation | Continuous tracking of 6 bodily vitals (Energy, Stamina, Health, Threat, Exposure, Hunger); dynamic neuromodulator coupling ($\text{DA}, 5\text{-HT}, \text{OA}, \text{ACh}$); physical forward thrust limits under stamina fatigue. |
| **MiMoMix Cognitive Core** | Dual-process cognitive deliberation | Epistemic trigger monitoring free energy, surprisal, and hazard proximity; fast reflexive System 1 intuition vs. multi-step System 2 Adaptive Computation Time (ACT) mental rollouts with attention sink null-slots and ponder penalties. |
| **Experience World Model** | Prequential dynamics prediction | 3-member deterministic bootstrap RLS ensemble predicting sensor summary deltas and reward conditioned on live graph/Tidal representations. Disagreement and leverage provide uncalibrated heuristics. |
| **Surprise Episodic Memory** | Error-gated outcome storage | Bounded capacity (128 entries); selective error/coverage admission; pure distance and recency weighted retrieval; non-mutating reads. |
| **Complex Dynamic Ecology** | Embodied virtual simulation | Continuous 2D/3D physics environment featuring autonomous moving predator drones with threat auras, variable-friction terrain zones (ice, mud, sand), dynamic oscillating/locked gates, and diurnal ambient illumination cycles. |
| **Weight-Space Programming** | Safe transactional parameter editing | Bounded edit packets, parent hash validation, candidate evaluation, exact state rollback, and closed-loop automated diagnosis/repair. |
| **Recursive Self-Improvement (RSI)** | Meta-level learning of learning | Versioned improvement policies $\phi_g$, support/query task isolation, shared archive controls, candidate meta-validation, and fresh-learner held-out evaluations. |
| **Zero-Warping Observatory & HUD** | Real-time scientific visualization | Isotropic 3D isometric and top-down radar views with mathematical aspect ratio locking; real-time 3D anatomical connectome observatory with OrbitControls; direct WASD teleoperation; real-time Vitals HUD meters. |
| **Continual Backpropagation** | Utility-guided plasticity & replacement | Continuous utility tracking $u_i(t) = \beta u_i(t-1) + (1-\beta)|r_i|\sum_j |W_{ji}|$; resets mature stagnant units while clearing Adam moments and preserving Dale's law. |
| **Trainable Sleep Engine** | Student-teacher consolidation | Minimizes $\mathcal{L}_{\text{sleep}} = \lambda_1\mathcal{L}_{\text{task}} + \lambda_2\mathcal{L}_{\text{distill}} + \lambda_3\mathcal{L}_{\text{replay}} + \lambda_4\mathcal{L}_{\text{reg}}$ with SHY synaptic downscaling and evaluation rollback gating. |
| **Universal Fast-Weight Memory** | Test-time associative learning | Shared associative matrix $M \in \mathbb{R}^{d_k \times d_v}$ with self-supervised inner loss $\mathcal{L}_{\text{inner}}$, surprise-gated outer-product updates, and surprise gating. |
| **Recurrent Multistep Planner** | Learned dynamics & MPC control | Recurrent GRU state transition, reward, observation, and uncertainty heads; MPC action trajectory optimization over horizons $H \in \{1, 3, 5, 10\}$ with penalty for epistemic variance. |
| **Verifiable Curriculum Engine** | Verifiable progression & ReLIFT | Procedural sequential subgoals (`CUE_DISCOVERED`, `GATE_TRIGGERED`, `BARRIER_PASSED`, `TARGET_REACHED`) with cryptographic SHA-256 process receipts. |
| **Causal Connectome Evolution** | Interventional structural changes | Paired counterfactual rollouts measuring $\Delta J(m) = \mathbb{E}[R \mid do(m)] - \mathbb{E}[R \mid do(m_0)]$ penalized by wiring, compute, and catastrophic forgetting costs. |
| **Tripartite Uncertainty Decomposition** | Calibrated risk & OOD detection | Epistemic ensemble variance + Heteroscedastic Gaussian NLL aleatoric head + Mahalanobis OOD latent distance; Expected Calibration Error (ECE) and monotonic risk-coverage analysis. |
| **Action-Grounded Efference Self-Model** | Sensorimotor agency & reafference | Forward internal model predicting sensory delta $\Delta \hat{s}_t = M_{\text{efference}}(s_t, a_t)$; decomposes discrepancy into Self-Authorship Index $A_t \in [0, 1]$, External Disturbance $E_t$, and Actuator Slip $F_t$; benchmarked against matched scientific controls (`scrambled_timing`, `shuffled_action`, `ablated`). |
| **Supermix Research Fabric & 6-Arm Study** | Cross-ecosystem contracts & comparative study | Typed, versioned contracts across Poseidon, NEUROGRAFT, Tether, FAULTLINE, and GhostInTheMachine with cryptographic SHA-256 evaluation receipts; standard 6-arm curriculum study (`frozen_actor`, `online_rl`, `actor_replay`, `continual_backprop`, `connectome_evolution`, `composed_system`) evaluating retention ($\ge 95\%$), forgetting, and latency. |
| **Plasticity Monitor & Reusable Skills** | Representation rank & macro discovery | Continuous representation Effective Rank $\text{erank}(X)$, Stable Rank $\text{srank}(X)$, dead neuron monitoring, and TESSERA-inspired macro skill discovery. |
| **Verifier-Gated Bounded RSI** | Bounded self-improvement & failure archive | AIDE² / DGM / AlphaEvolve pattern: AST range checks, deterministic tests, support/query evaluation, LCB dominance ($LCB(\Delta) > \epsilon$), canary regression gating ($\le 1\%$), and immutable Failure Archive. |
| **World Model v2 & CEM Planner** | Standalone neural dynamics experiment | $M=3$ latent ensemble and CEM demonstrator. The live actor instead uses the persisted experience model below. CLI demonstrations do not record fabricated realized outcomes. |
| **Experience MPC** | Integrated short-horizon prediction and control | Pure deterministic ensemble rollouts over horizons 1–5, bounded local action changes, fixed neural context, repeated-baseline comparator, uncertainty/support abstention, and a 256 ensemble-transition work budget. No terminal/value head or calibrated safety claim. |
| **9-Arm Connectome Causal Assay** | Biological vs matched controls | Causal benchmark evaluating real Drosophila wiring against 8 matched nulls (degree-, sign-, community-rewired, random, learned, unsigned, local-only, brain-cord motif) calculating formal Cohen's $d$. |
| **Coconut Latent Deliberation** | Tokenless continuous recurrence | Continuous hidden-state thought recurrence ($h^{\text{think}}_{i+1} = F(h^{\text{think}}_i, z_t)$), dynamic ponder halting, and anytime compute-depth curves. |
| **Masked Diffusion Repair Sidecar** | Non-autoregressive repair & infill | Iterative masked diffusion repair of failing code, configuration, or action trajectory constraints with executable verifier in the loop. |
| **Machine-Verifiable DPO Engine** | Direct preference optimization | First-party machine-checked preference pairs (safety, return, unit tests) optimized via DPO without ungrounded self-judging loops. |
| **Unified Evaluation Scorecard** | Standalone metric calculations | Combines supplied returns, compute, Brier/ECE and failure inputs. Its CLI fixture is explicitly illustrative; it does not measure deployment performance. |
| **Landmark Connectome Catalog (8 Datasets)** | Whole-animal multi-connectome substrate | Full integration of 8 datasets: MaleCNS v1.0, BANC, FlyEM Optic Lobe, FANC, FlyWire FAFB v783, MANC, Larval Drosophila, and Hemibrain with regional neuromodulation. |
| **Multi-Connectome Hub** | Scalable cross-region routing | Unifies brain, optic, and cord neuropils with 93 biological interface sites, high-capacity scaling (>40,000 neurons/sec on CPU), and transactional weight-space editing. |
| **Frontier Apex Omnibus Mode** | Full architectural & environmental integration | Single unified command `python -m synapse.cli omnibus` running all 13 complex environments (`forage`, `maze`, `push`, `delayed_match`, `ecosystem`, `predator_evasion`, `dynamic_arena`, `puzzle_chamber`, `apex_megasite`, `stealth_labyrinth`, `kinetic_foundry`, `continental_biosphere`, `chrono_tidepool_citadel`) across **4,000 neurons** (`1334/1333/1333`) with efference self-model, uncertainty calibration, plasticity rank monitoring, and cryptographic SHA-256 evaluation receipts. |
| **Executed-Transition Neural Graft** | Bounded residual actor, TD critic & sensory predictor | Zero-initialized residual policy (`synapse/neural_graft.py`) conditioned on 155-d observations, 64-d graph/donor features, 5-d baseline actions, and an optional 128-neuron MaleCNS sparse connectome prior (`load_programmed_graft`). Supports parallel rollout pretraining (`pretrain_runtime`) and online clipped policy optimization without mutating actor identity during auxiliary pretraining. |
| **Active Dendritic Liquid Time-Constant (LTC) Sub-Graft** | Two-compartment apical/basal integration | Continuous-time ODE compartments (`synapse/dendritic_ltc.py`) with input-dependent liquid time-constants $\tau(x, c) \in [0.15, 4.0]$, NMDA coincidence plateau potentials, and local three-factor eligibility traces. |
| **Oscillatory Phase-Coherence & Kuramoto Synchrony** | Multi-band ($\theta/\beta/\gamma$) cross-neuropil routing | Coupled Kuramoto phase oscillators (`synapse/oscillatory_coherence.py`) across FlyWire, MANC, and Optic-Lobe neuropils implementing Communication-Through-Coherence (CTC) gain modulation. |
| **Poseidon DV-GEM Belief Memory & Reserve Guard** | Decision-value gated episodic memory | Recurrent GRU belief encoder (`synapse/dvgem_tether.py`), support/conflict screening, predictive viability reserve floor guard, decision-value admission gate ($\Delta Q - 0.5\,\text{conflict} > \theta_{\text{admit}}$), and 6-arm matched lesion assay (`intact`, `similarity`, `erased`, `shifted`, `donor`, `baseline`). |
| **Project TETHER Specialist Gate, Successor Features & Poincaré/Lyapunov Monitor** | Specialist coordination, $\psi(s,a)$ & hyperbolic geometry | Continuous specialist reliability weighting & fallback gate $\tau \in [0, 1]$ (`synapse/dvgem_tether.py`), TD(0) Successor Features $Q(s, a) = \psi(s, a)^\top w_r$, Poincaré Ball $\mathbb{B}^3$ hyperbolic neuropil embedding, and Finite-Time Lyapunov Exponent $\lambda_{\max}$ stability tracking. |
| **Verified Neuron-Level Connectome Caches** | Checksum-verified local connectome ingestion | Independent source-backed caches (`synapse/connectome_assets.py`) for `malecns`, `flywire`, `manc`, `optic-lobe`, `banc`, `hemibrain`, and `larval`, keeping catalog metadata, cell-type aggregates, induced subgraphs, and synthetic proxies explicitly distinct. |
| **Live Weight-Space Screen & Bounded RSI** | Matched cloned-world numeric patch screening | Screens numeric weight edits (`screen_live_patch`, `run_live_rsi` in `synapse/live_programming.py`) on isolated cloned worlds across query and retention seeds before atomic application. |


## Run Locally

Requires Python 3.10+ with PyTorch CPU, NumPy, and SciPy:

New workbench sessions use **4,000 total neurons**, allocated 1334/1333/1333 across
the three default sources (FlyWire, MANC, Optic Lobe). The graph view includes all 4,000. Use
`python -m synapse --total-neurons 4000 serve` or set **Total neurons** in the
session controls. The older `--nodes 16` option still selects 16 per source for
small experiments and compatible checkpoints. Run the 11-stage omnibus curriculum with
`python -m synapse.cli omnibus`.

```powershell
python -m pip install -e .
python -m pytest -q
.\launch.ps1 -Port 8765
```

Open the local workbench at [http://127.0.0.1:8765](http://127.0.0.1:8765) or launch via `Run-Synapse.cmd`.

### Interactive Workbench Features
- **Zero-Warping World Views**: True isotropic aspect ratio rendering for both 3D isometric projection and top-down radar world views, ensuring square cells remain square under all window resizes.
- **Complex Ecology Telemetry**: Real-time visualization of terrain friction zones (ice, mud, sand), moving predator drones with hazard danger rings, dynamic gates, day/night ambient lighting, and resource targets.
- **Direct Keyboard Teleoperation**: Control the embodied agent in real-time using WASD or Arrow Keys, with Spacebar for object grab/place.
- **3D Anatomical Connectome Observatory**: Interactive 3D point-cloud and synaptic edge visualization with OrbitControls (left-click rotate, right-click pan, scroll zoom) and click-to-inspect individual neuron activity.
- **Homeostatic Vitals & Cognition HUD**: Real-time meters for Energy, Stamina, Health, Threat, Exposure, and Hunger, paired with MiMoMix System 1 / System 2 deliberation badges and ACT ponder steps.
- **Scientific Assay Panels**: Causal Circuit Atlas, Weight-Space Hypernetwork synthesis, Flagship evolutionary benchmarks, State Archives, and Audited Capabilities Explorer.
- **Counterfactual Plan Inspector**: Select a horizon, inspect pure forecasts, compare candidate returns and uncertainty, and see why the planner retains its baseline. Previews never advance the world or fit the model.
- **Learning Contracts**: Manual actions can train the same executed-outcome learner. Drive/turn coordinates and fatigue limits apply consistently; frozen evaluation preserves learned topology and structural statistics. Tool memory belongs to each checkpointed runtime.

## Command Line Interface (CLI)

SYNAPSE provides comprehensive CLI commands for headless execution, experiments, and inspection:

```powershell
# Inspection & Baseline Execution
python -m synapse.cli --nodes 16 status
python -m synapse.cli --nodes 16 run --steps 100
python -m synapse.cli --nodes 16 train --steps 100 --save trained_agent

# Supermix Beyond Capabilities
python -m synapse.cli --nodes 16 neurogenesis
python -m synapse.cli --nodes 16 homeostasis
python -m synapse.cli --nodes 16 mimomix --ponder 4
python -m synapse.cli --nodes 16 ecology --task ecosystem

# Audited Upstream Capabilities (142 audited capabilities across 47 families)
python -m synapse.cli capabilities --id structural.neurogenesis
python -m synapse.cli capabilities --id motivation.homeostasis
python -m synapse.cli capabilities --id reasoning.mimomix
python -m synapse.cli capabilities --id world.complex_ecology
python -m synapse.cli capabilities --id learning.continual_backprop
python -m synapse.cli capabilities --id memory.fast_weights

# SAIL 6-Upgrade Research & Development Suite
python -m synapse.cli continual-backprop
python -m synapse.cli sleep-train --sleep-steps 10
python -m synapse.cli fast-memory
python -m synapse.cli multistep-plan --horizon 3
python -m synapse.cli curriculum-verify
python -m synapse.cli causal-growth
python -m synapse.cli flagship-milestone --milestone-steps 20

# Deep Research Architecture & Control Experiments
python -m synapse.cli rsi-verify --patch-type config
python -m synapse.cli world-model-cem --cem-horizon 5
python -m synapse.cli connectome-assay --assay-trials 3
python -m synapse.cli coconut-think --think-depth 4
python -m synapse.cli diffusion-repair --repair-iters 3
python -m synapse.cli dpo-align --pairs 5
python -m synapse.cli unified-eval --eval-task forage

# Landmark Connectome Datasets & High-Capacity Scaling
python -m synapse.cli connectome-inventory
python -m synapse.cli connectome-datasets
python -m synapse.cli multi-connectome --nodes-per-source 32
python -m synapse.cli scale-neurons --targets 64 128 256 512

# Neural Graft Pretraining, Active Training & Bounded Live RSI
python -m synapse.cli --compute-threads 4 --neural-graft --programmed-connectome pretrain --steps 256 --updates 64 --batch-size 128 --workers 2 --save pretrained-graft
python -m synapse.cli --compute-threads 4 --neural-graft --active-training --programmed-connectome --connectome-mode prefer_authentic train --steps 64 --save online-graft
python -m synapse.cli --compute-threads 4 live-rsi --steps 24 --candidates 2 --save rsi-screened

# Automated Weight-Space Programming & Closed-Loop Repair
python -m synapse.cli --nodes 16 autoprogram --diagnostic-episodes 3 --eval-episodes 3 --save repaired_agent
python -m synapse.cli --nodes 16 transplant --donor-task forage --test-task delayed_match
python -m synapse.cli --nodes 16 metamorphic
python -m synapse.cli --nodes 16 spectral
python -m synapse.cli --nodes 16 hyperprogram --target motor.weight

# Measured Adaptation Experiments
python -m synapse.cli --output outputs evidence --name my-pilot --seeds 7 11 19 --tasks forage push --support-steps 64 --query-steps 64
python -m synapse.cli adaptation-verify outputs/adaptation-studies/my-pilot
python -m synapse.cli evidence --receipt outputs/adaptation-studies/my-pilot

# Paired actual-world planning comparison (actor, horizon 1, horizon 3)
python -m synapse.cli --nodes 16 --output outputs planning-study --name planning-pilot --seeds 23 29 31 --tasks forage push ecosystem --support-steps 96 --query-steps 48 --horizons 1 3
python -m synapse.cli planning-verify outputs/planning-studies/planning-pilot
```

## Measured Experiments & Verification

Empirical evidence studies run on actual native and complex environments with strictly separated support and query worlds:
- **Support Adaptation**: Shared exploratory support trajectories condition the learner.
- **Query Freeze**: All model and memory learning is strictly frozen during query episodes.
- **Replay & Audit Receipts**: Full cryptographic digests, parameter matrices, RNG states, and execution timings are recorded in atomic JSON receipts.

Planning pilots fit a shared learner on actual support trajectories and compare
frozen actors in independent paired query worlds. Saved checkpoints and receipts
bind raw transitions, source fingerprints and arithmetic. Verification detects
inconsistency; it does not authenticate a consistently rewritten receipt or
re-execute the simulation. Negative differences and abstentions remain visible.

Sleep consolidation uses observation-conditioned reset-state imitation and accepts
only candidates passing a caller-supplied finite retention screen. Missing or
failed screens reject the candidate. Untrained predictive/MiMoMix action corrections
are opt-in through `experimental_cognition`; older checkpoints default to disabled.
The original flagship continual benchmark remains a labeled synthetic surrogate.

The integrated planner draws on [PETS](https://arxiv.org/abs/1805.12114) for ensemble
trajectory propagation and [MBPO](https://arxiv.org/abs/1906.08253) for limiting model
rollouts. It uses online linear dynamics and deterministic search; it implements
neither paper's complete algorithm or reported guarantees. An online Temporal Difference
value critic $V(s)$ resolves finite-horizon truncation bias when `value_guided_planning`
is selected.

## Frontier Capabilities & Research Additions

The runtime provides toggleable research implementations of recent frontier techniques:
- **Dormant Expert Dynamic Lifecycle (`synapse/dormant_lifecycle.py`)**:
  Recruits dynamic capacity when graph delta exceeds novelty thresholds. Experts advance
  through validated states (`DORMANT` $\to$ `INITIALIZING` $\to$ `ACTIVATING` $\to$ `ACTIVE` $\to$ `CONSOLIDATING` $\to$ `RETIRED`)
  with orthogonal initialization, gradual activation ramping, directional cosine contribution testing,
  and native parameter distillation with slot recycling.
- **Value-Guided MPC Critic (`synapse/experience.py`, `synapse/experience_planner.py`)**:
  Eliminates horizon truncation bias in short MPC trajectories ($H \in [1, 5]$) by learning an online TD critic
  $V(s) \approx \mathbb{E}[\sum_{\tau=0}^\infty \gamma^\tau r_\tau]$ alongside the transition ensemble.
- **Two-Channel Evoked Readout (`synapse/agent.py`)**:
  Separates resting manifold activity $m = O(r_{\text{null}})$ from task-evoked perturbation $d = O(r_{\text{actual}}) - O(r_{\text{null}})$
  with input-dependent gating $\alpha$, exposing full telemetry for persistent state dynamics.
- **Continual Backpropagation (`synapse/continual_backprop.py`)**:
  Applies Dohare et al. (Nature 2024) selective unit reinitialization based on feature utility to prevent
  loss of plasticity during continual adaptation.
- **Universal Fast-Weight Memory (`synapse/fast_weight_memory.py`)**:
  Implements dual-speed associative learning (Titans / uTTT) with surprise-gated inner-loop updates.
- **Continuous Latent Deliberation (`synapse/latent_reasoner.py`)**:
  Applies Coconut continuous latent-space thinking when variational free energy or prediction error exceeds bounds.

All options are configurable in `SynapseConfig`, controllable via the web workbench HUD and settings, and fully preserved across mid-episode state checkpoints.

## Upstream Lineage & Research Grounding

SYNAPSE unifies innovations from:
- **GraftBrain**: Signed recurrent neural dynamics, biological neuropil topologies, mushroom body associative learning, and embodied planar physics.
- **Supermix-Poseidon**: Trained Tidal MoE DAgger controller, predictive horizons, causal intervention analyzers, and multi-expert routing.
- **Supermix-Expanse**: Function-preserving dormant experts, native consolidation, evoked graph readouts, and male-CNS connectome data.
- **Archimedes v93**: Continuous covariance-driven structural neurogenesis, synaptogenesis, and apoptosis.
- **Supermix Beyond**: Interoceptive homeostatic drive regulation, MiMoMix dual-process cognitive deliberation, and complex multi-hazard virtual ecology.

All source code is released under the Apache 2.0 license. Detailed source manifests and hashes are preserved in `synapse/source_manifest.json`.
