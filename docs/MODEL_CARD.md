# SYNAPSE Model Card

## Model Details
- **Model Name**: SYNAPSE Integrated Connectome & Cognitive Agent
- **Version**: 0.2.0 (Supermix Beyond & Archimedes v93 Edition)
- **Model Type**: Hybrid Neuromorphic Connectome / Transformer MoE Controller with Structural Neurogenesis, Homeostatic Regulation, Dual-Process Deliberation, and Direct Weight-Space Programming
- **Architecture**:
  - **Sensory Encoder**: Linear projection mapping 173 input dimensions (155-d observation + 16-d compass bump + 2-d mushroom body valence) to connectome sensory nodes.
  - **Neural Connectome Substrate**: Signed recurrent rate graph (`GraftedBrain`) combining three source families (`flywire`, `manc`, `optic-lobe`), signed Dale synapses, homeostatic balance, and 8 inter-module graft sites.
    - **Deep Research Verifier-Gated Bounded RSI Engine**:
    - AIDE² / DGM / AlphaEvolve pattern: Schema/AST checking, deterministic tests, support/query world isolation, LCB dominance ($LCB(\Delta) > \epsilon$), canary regression gating ($\le 1\%$), and dual Promoted Lineage vs. Failure Archive.
  - **Deep Research World Model v2 with CEM & Fallback Shield**:
    - $M=3$ latent dynamics ensemble with JEPA representation loss, Cross-Entropy Method (CEM) trajectory optimization across horizons $H \in \{1, 3, 5, 10\}$, invariant Fallback Shield to frozen Tidal controller, and model exploitation tracking ($G_{\text{exploit}} = R_{\text{predicted}} - R_{\text{realised}}$).
  - **Deep Research 9-Arm Connectome Causal Controls**:
    - Causal assay evaluating real biological Drosophila connectome wiring against 8 matched control topologies (degree-preserving, sign-preserving, community-preserving, random sparse, learned sparse, unsigned, local-only, brain-cord motif) calculating formal Cohen's $d$ effect sizes.
  - **Deep Research Coconut Continuous Latent Deliberation**:
    - Continuous hidden-state thought recurrence without token bottleneck ($h^{\text{think}}_{i+1} = F(h^{\text{think}}_i, z_t)$), dynamic ponder halting, and anytime compute-depth curves.
  - **Deep Research Masked Diffusion Repair & Infilling Sidecar**:
    - Iterative non-autoregressive masked diffusion repair of failing code, configuration, or action trajectory constraints with executable verifier in the loop.
  - **Deep Research Machine-Verifiable DPO Engine**:
    - First-party machine-checked preference pairs (safety, return, unit tests) optimized via Direct Preference Optimization.
  - **Deep Research Unified Evaluation Scorecard**:
    - Joint deployment score $S = R - \alpha C - \beta U - \gamma F$ balancing task return, compute cost, calibration (Brier / ECE), and safety failures.
  - **SAIL Continual Backpropagation & Utility Replacement**:
    - Utility metric: $u_i(t) = \beta u_i(t-1) + (1-\beta)|r_i|\sum_j |W_{ji}|$
    - Resets low-utility mature units, zeros corresponding Adam first (`exp_avg`) and second (`exp_avg_sq`) moments, and reinitializes with Dale's law sign preservation.
  - **SAIL Trainable Student-Teacher Sleep Consolidation**:
    - Optimizes $\mathcal{L}_{\text{sleep}} = \lambda_1\mathcal{L}_{\text{task}} + \lambda_2\mathcal{L}_{\text{distill}} + \lambda_3\mathcal{L}_{\text{replay}} + \lambda_4\mathcal{L}_{\text{regularise}}$.
    - Employs Synaptic Homeostasis (SHY) downscaling and test-candidate evaluation gating before committing weights.
  - **SAIL Universal Fast-Weight Associative Memory (uTTT & Titans)**:
    - Shared fast associative matrix $M \in \mathbb{R}^{d_k \times d_v}$ with self-supervised reconstruction loss $\mathcal{L}_{\text{inner}}$, surprise gating, and bounded memory capacity.
  - **SAIL Recurrent Multi-Step World Model & MPC Planner**:
    - Recurrent GRU transition model $s_{t+1} = f_\theta(s_t, a_t)$ with observation, reward, and ensemble epistemic uncertainty estimation.
    - Model Predictive Control evaluates simulated rollouts across horizons $H \in \{1, 3, 5, 10\}$ with variance penalties.
  - **SAIL Causal Connectome Evolution**:
    - Evaluates prospective mutations via counterfactual paired rollouts $\Delta J(m) = \mathbb{E}[R \mid do(m)] - \mathbb{E}[R \mid do(m_0)]$.
    - Multi-objective fitness function enforces wiring length, metabolic cost, and backward transfer constraint.
  - **Donor Cognitive Core**: Pinned trained Poseidon Tidal MoE DAgger controller (`tidal_core.pt`, SHA-256 `a053104a...`), 327,230 parameters, 4 experts, 128 hidden width, 2 recurrent steps.
  - **Bridges**: Bidirectional residual projections (`graph_to_resource` and `donor_to_graph`) synchronizing graph context with cognitive intent.
  - **Archimedes Connectome Neurogenesis Engine**:
    - Online Welford covariance tracking: $\text{Cov}_{ij} \leftarrow \lambda \text{Cov}_{ij} + (1-\lambda)(r_i - \bar{r}_i)(r_j - \bar{r}_j)$.
    - Apoptosis: eliminates stagnant synapses where $|w_{ij}| < \theta_{\text{prune}} = 0.02$.
    - Synaptogenesis: spawns new connections between highly co-active neural pairs ($\text{Cov}_{ij} > \theta_{\text{grow}}$) gated by Dopamine RPE ($\delta_{\text{DA}}$) and Astrocytic calcium spillover ($[\text{Ca}^{2+}]$).
    - Dale's Law Enforcement: strictly enforces excitatory/inhibitory sign compliance on newborn connections based on presynaptic cell type.
  - **Supermix Homeostatic Drive Engine**:
    - Continuous tracking of 6 bodily vitals: Energy ($E$), Stamina ($S$), Health ($H$), Threat ($T$), Exposure ($X$), and Hunger ($U$).
    - Dynamically modulates neuromodulatory gains:
      $$\text{DA} \leftarrow \text{DA}_0 \cdot (0.6 + 0.4 E)(1.0 - 0.5 T)$$
      $$5\text{-HT} \leftarrow 5\text{-HT}_0 \cdot (0.5 + 0.5 H)(1.0 + 0.8 T)$$
      $$\text{OA} \leftarrow \text{OA}_0 \cdot (0.7 + 0.3 S)(1.0 + 1.2 T)$$
      $$\text{ACh} \leftarrow \text{ACh}_0 \cdot (0.5 + 0.5 X)$$
    - Imposes physical fatigue limits on motor thrust: $a_{\text{drive}} \leftarrow \min(a_{\text{drive}}, S)$.
  - **MiMoMix Dual-Process Cognitive Reasoner**:
    - System 1 (Intuitive Reflex): Fast low-latency motor execution driven directly by connectome and Tidal output.
    - Epistemic Trigger: Activates deliberative System 2 whenever free energy, surprisal, or threat exceeds threshold $\tau_{\text{deliberate}} = 0.45$.
    - System 2 (Deliberative ACT Mental Rollout): Multi-step mental search with Adaptive Computation Time (ACT), attention sink null-slots, and ponder cost penalty ($\lambda_{\text{ponder}} = 0.02$ per step).
  - **Expanse-inspired Grafts**: 2 untrained expert slots recruited via novelty thresholding, and a zero-initialized native consolidation block.
  - **Motor Decoder**: Linear projection from 64-d fused feature representation to 5 continuous motor actuators (`drive`, `turn`, `reach`, `height`, `grip`), modulated by homeostatic fatigue constraints and deliberated actions.
  - **Experience Model**: Three bootstrap RLS members predict 14 sensor-summary deltas and actual reward, conditioned on current graph/donor summaries.
  - **Outcome Memory**: At most 128 actual state/action/reward records with surprise/coverage admission and pure distance/recency reads.
  - **Vision**: Small seeded visual encoder and projection affecting live actions; receives optional supervised preparation gradients and is checkpointed.

## Feature Representation Schema (v1, 64-dimensional)
| Indices | Subsystem | Content |
|---|---|---|
| `[0:32]` | Graph Context | Geometrically pooled connectome activation (absolute or evoked) |
| `[32:48]` | Tidal Memory | Pretrained MoE hidden memory state slice |
| `[48:50]` | Mushroom Body | 2-d approach/avoidance valence readout |
| `[50:56]` | Tidal Intent | 6-class softmax action distribution (`rest`, `forage`, `drink`, `shelter`, `explore`, `flee`) |
| `[56:64]` | Compass | First 8 wedges of the ellipsoid body ring attractor |

## Weight Edit Surfaces & Permissions
| Parameter Surface | Editable in Patch | Trainable in Preparation | Frozen |
|---|---|---|---|
| `brain.graft_w` | Yes | Yes | No |
| `plasticity.router.*` | Yes | Yes | No |
| `mb.w_mbon` | Yes | Yes (Hebbian/Dopamine) | No |
| `encoder.*` | Yes | Yes | No |
| `donor_to_graph.*` | Yes | Yes | No |
| `graph_to_resource.*`| Yes | Yes | No |
| `motor.*` | Yes | Yes | No |
| `dormant.*` | Yes | No | No (manual patch surface) |
| `native.*` | Yes | No gradients; heuristic copying | No |
| `vision_encoder.*`, `vis_to_sensory.*` | Yes | Yes, with supplied camera demonstrations | No |
| `neurogenesis.*` | Yes | Dynamic online structural updates | No |
| `homeostasis.*` | Yes | Continuous interoceptive regulation | No |
| `mimomix.*` | Yes | Deliberative ACT rollout parameters | No |
| `fast_memory.*` | Yes | Yes (online associative updates) | No |
| `world_model.*` | Yes | Yes (prequential & multi-step) | No |
| `continual_backprop.*`| Yes | Dynamic utility-based replacement | No |
| `tidal.*` (Tidal Core)| **No** | **No** | **Frozen** |
| `curiosity.*` | **No** | Yes (RND updates) | No |

## Evaluation and Intended Usage
- **Primary Use**: Embodied reinforcement learning, cue-delay navigation, multi-agent predator evasion, terrain-aware locomotion, homeostatic survival, and recursive self-improvement research in synthetic continuous environments.
- **Out-of-Scope**: High-stakes physical robotics without hardware safety bounds, medical diagnosis, or claims of biological consciousness.

The recurrent graph uses synthetic proxy topologies. Checkpoints preserve all learned models, covariance arrays, RNG states, homeostatic vitals, neurogenesis history, and pending transitions. Exact mid-episode replay is verified and tested bit-for-bit across all world backends.
