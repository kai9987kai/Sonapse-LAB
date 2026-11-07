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
- **SANCF — Sonapse Adaptive Neural Communication Fabric (`synapse/neural_comm.py`, `synapse/neural_comm_channels.py`, `synapse/neural_comm_topology.py`)**:
  Additive virtual neural communication fabric operating alongside traditional scalar plastic grafts (`gamma=0.0` zero-init baseline). Features continuous, uniform quantized, VQ codebook, orthogonal mode (DCT/Hadamard/learned basis), and impaired (latency queue, drop, burst, ARQ) channel models with regional message encoders, decoders, and communication-through-coherence phase gating.
- **8,000 Biological Neuron Connectome Scaling (`synapse/config.py`, `synapse/multi_connectome.py`)**:
  Scales high-capacity Drosophila multi-connectome substrate to 8,000 biological neurons partitioned across FlyWire (2,667), MANC (2,667), and FlyEM Optic Lobe (2,666), loading 427,390 authentic biological synapses alongside 128 MaleCNS motor priors and an auxiliary 256-neuron Dale-compliant Overflow Mini-Brain.
- **Non-Equilibrium Neural Thermodynamics & Arrow-of-Time Dissipation (`synapse/thermodynamics.py`)**:
  Quantifies broken detailed balance, net state-transition probability flux, and non-equilibrium entropy production rate $\dot{S} = \sum J_{ij} \ln(P_{ij} / P_{ji}) \ge 0$, tracking thermodynamic information efficiency $\eta = \Delta I_{\text{task}} / (\dot{S} + \epsilon)$ (Lynn & Bassett, Nature Physics 2021; Deco et al., Cell Reports 2022).
- **Homeostatic Metaplasticity & BCM Sliding Threshold Engine (`synapse/metaplasticity_bcm.py`)**:
  Implements Bienenstock-Cooper-Munro sliding modification threshold $\theta_M(t) = (1 - \alpha) \theta_M(t-1) + \alpha (r_i / r_0)^2$ and Turrigiano multiplicative homeostatic scaling $w \leftarrow w \cdot (r_{\text{target}} / \bar{r})^\gamma$, preventing runaway excitation or quiescence across 8,000 neurons while guaranteeing exact replay determinism.
- **Dynamic Transfer Entropy & Partial Information Decomposition (`synapse/transfer_entropy.py`)**:
  Measures directed transfer entropy $\text{TE}_{X \to Y} = I(Y_{t+1}; X_t \mid Y_t)$ between Sensory (Optic Lobe), Central (FlyWire), and Motor (MANC) connectomes, decomposing joint mutual information into synergistic vs. redundant components to quantify emergent multi-connectome integration.
- **3D Causal & Transfer-Entropy Geodesic Streamlines in Observatory (`synapse/observatory.py`, `synapse/web/app.js`)**:
  Renders directed signal highways directly into the WebGL Observatory canvas with real-time thermodynamic dissipation and transfer entropy HUD gauges.
- **Chebyshev Spectral Graph Diffusion Engine (`synapse/graph_diffusion.py`)**:
  Computes multi-scale topological signal diffusion across the 6,500-neuron connectome via K=4 Chebyshev polynomial recurrence over the normalized graph Laplacian in O(E) time, with zero-initialized readout gating for exact frozen baseline preservation.
- **Non-Linear Multi-Branch Dendritic dSpikes (`synapse/dendritic_ltc.py`)**:
  Multi-compartmental dendritic model featuring 4 localized branches (proximal basal, distal oblique, apical trunk, apical tuft) with supralinear calcium/sodium dSpike event tracking and NMDA plateau potentials (Gidon et al. Science 2020).
- **Precision-Weighted Hierarchical Active Inference (`synapse/active_inference.py`)**:
  Expected Free Energy (EFE) decision engine balancing pragmatic goal-directed reward utility against epistemic Bayesian information gain, with acetylcholine observation precision and dopamine policy precision weighting (Friston et al. 2023).
- **Directional Causal Graph Discovery & Path Tracing (`synapse/causal_discovery.py`)**:
  DirectLiNGAM structural equation causal discovery and Pearlian interventional path tracing, revealing top causal signal transmission highways from sensory neuropils to motor effectors.
- **Hierarchical 3D Level-Of-Detail (LOD) Observatory (`synapse/observatory.py`, `synapse/web/app.js`)**:
  Generates multi-level anatomical cluster hierarchies (LOD 0: 6,500 individual neurons; LOD 1: 48 spatial neuropil clusters; LOD 2: global specimen hubs) for smooth 60 FPS WebGL visualization.
- **Scientific Non-Claims & Physical Boundary Disclaimers**:
  Oscillatory modes in SANCF and `oscillatory_coherence.py` are abstract coupled mathematical modes, **not** literal biophysical membrane rhythms. Digital orthogonal channels are mathematical basis projections, not physical orbital angular momentum, radio waves, or quantum communication channels.
- **Opt-in 4D Environment (`synapse/world4d.py`)**:
  `world4d_tesseract_bypass` features 4-spatial-dimensional hyper-torus physics, 6-plane hyper-rotations ($XY, XZ, XW, YZ, YW, ZW$), 155D multi-plane shadow projection adapters, and a 4D obstacle bypass maze.
- **Permissioned Digital Tool Arena (`synapse/digital_tool_arena.py`)**:
  `digital_tool_arena` provides a strictly closed, permissioned synthetic document retrieval sandbox with zero shell, subprocess, socket, or host filesystem access.
- **Background Simulation Worker & Uncapped SPS (`synapse/sim_worker.py`)**:
  Server-owned background thread operating via re-entrant locks for batch stepping and uncapped throughput execution.
- **Causal Connectome Discovery Lab (`synapse/causal_connectome_assay.py`, `synapse/motif_discovery.py`, `synapse/connectome_interventions.py`)**:
  Extracts 10 canonical microcircuit motifs (feedforward loop, feedback inhibition, reciprocal bridge, mutual excitation, lateral inhibition, cross-region relay, converging hub, diverging broadcaster, chain relay, ring oscillator) and evaluates 10 causal graph interventions with cross-region transferability assays.
- **Multi-Timescale Plasticity Coordinator (`synapse/adaptation_coordinator.py`, `synapse/multitimescale_learning.py`)**:
  Harmonizes fast-weight associative updates (fast), neural graft active replay & DPO (medium), and dormant structural lifecycle (slow) with live effective rank and condition number tracking.
- **Meta-Learning of Synaptic Plasticity Rules (`synapse/meta_plasticity.py`, `synapse/plasticity_rule_learner.py`)**:
  Meta-learns parameterized synaptic update functions $f_\phi(\text{pre}, \text{post}, \delta, \text{DA}, \text{ACh}, u, c)$ respecting Dale's law and weight magnitude bounds, evaluated against classical Hebbian baselines.
- **Rigorous Criticality Engine (`synapse/criticality_vsa.py`)**:
  Upgrades connectome reservoir criticality with Maximum-Likelihood Hill estimator, log-likelihood ratio tests against exponential alternatives, and 95% bootstrap confidence intervals.
- **Causal World Model with Pearlian Do-Calculus (`synapse/causal_world_model.py`, `synapse/intervention_planner.py`, `synapse/causal_experiment_assay.py`)**:
  Disentangles passive environmental drift from agent interventional effects $P(s' \mid \text{do}(a))$ across 4 competing hypotheses with active information-gain planning.
- **Relational & Holographic Working Memory (`synapse/relational_memory.py`, `synapse/compositional_retrieval.py`, `synapse/memory_interference_assay.py`)**:
  HRR circular convolution role-filler binding with codebook cleanup memory and empirical interference capacity benchmarks.
- **Verifier-Gated Recursive Architectural Improvement (`synapse/verifier_rsi.py`)**:
  4-stage verification gate battery (static validity, numerical stability, benchmark non-regression, paired bootstrap testing) with cryptographic DAG lineage tracking and automated rollback.
- **Event-Driven Sparse Update Engine (`synapse/event_driven_backend.py`, `synapse/energy_efficiency_assay.py`)**:
  Delta-thresholded event propagation engine tracking synaptic operations (SynOps) and theoretical energy speedup over dense matrix multiplication.
- **18-Arm Factorial Ablation Engine & Unified Benchmark Suite (`synapse/ablation_engine.py`, `synapse/benchmark_suite.py`)**:
  18-condition ablation study isolating each architectural mechanism with bootstrap confidence intervals and reproducible scorecard receipts.

All options are configurable in `SynapseConfig`, controllable via the web workbench HUD and settings, and fully preserved across mid-episode state checkpoints.

## Upstream Lineage & Research Grounding

SYNAPSE unifies innovations from:
- **GraftBrain**: Signed recurrent neural dynamics, biological neuropil topologies, mushroom body associative learning, and embodied planar physics.
- **Supermix-Poseidon**: Trained Tidal MoE DAgger controller, predictive horizons, causal intervention analyzers, and multi-expert routing.
- **Supermix-Expanse**: Function-preserving dormant experts, native consolidation, evoked graph readouts, and male-CNS connectome data.
- **Archimedes v93**: Continuous covariance-driven structural neurogenesis, synaptogenesis, and apoptosis.
- **Supermix Beyond**: Interoceptive homeostatic drive regulation, MiMoMix dual-process cognitive deliberation, and complex multi-hazard virtual ecology.

All source code is released under the Apache 2.0 license. Detailed source manifests and hashes are preserved in `synapse/source_manifest.json`.
