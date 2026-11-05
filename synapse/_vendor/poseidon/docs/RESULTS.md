# Supermix Poseidon: Measured Results

This document records the empirical results measured on the local workstation for Supermix Poseidon v0.1. All numbers reflect actual runs executed with reproducible random seeds and documented data contracts.

## 1. Tidal Core Training

The Tidal core was trained on CPU using the documented contract (`configs/core.json`, 4 threads, AdamW lr=1e-3, batch size 256):

- **Data composition**: 60,000 synthetic scene descriptions + 60,000 simulator transition records = 120,000 unique examples.
- **Training schedule**: 3 epochs = 360,000 total exposures (1,407 gradient steps).
- **Model parameters**: 327,230 parameters (~0.33M).
- **Training time**: ~31.2 seconds on local CPU.
- **Data tensor SHA-256**: `20937c06443d9a0a1c92c279f76d47f1982eee288fcc0aca638233a69caff0aa`

### Development Split Progression (4,096 examples)

| Epoch | Scene Exact Acc | Action Acc (Teacher agreement) | Majority Action Baseline | Dynamics MSE | Zero-Delta Baseline |
|---|---|---|---|---|---|
| Untrained | 0.0% | 74.63% | 74.63% | 0.06227 | 0.02028 |
| Epoch 1 | 100.0% | 89.40% | 74.63% | 0.00550 | 0.02028 |
| Epoch 2 | 100.0% | 94.48% | 74.63% | 0.00447 | 0.02028 |
| Epoch 3 | 100.0% | 91.53% | 74.63% | 0.00429 | 0.02028 |

---

## 2. Frozen Held-Out Scene Evaluation

Evaluated on 2,048 scene prompts across 77 disjoint semantic groups held out from training (`test` split, seed 771122):

| Attribute | Vocabulary Size | Accuracy |
|---|---|---|
| Shape (`cube`, `sphere`, `pyramid`, `cylinder`) | 4 | **100.0%** (2048/2048) |
| Color (`red`, `blue`, `green`, `yellow`, `purple`, `cyan`) | 6 | **100.0%** (2048/2048) |
| Motion (`still`, `orbit`, `bounce`, `spin`) | 4 | **100.0%** (2048/2048) |
| Count (`1`, `2`, `3`) | 3 | **100.0%** (2048/2048) |
| Scale (`small`, `medium`, `large`) | 3 | **100.0%** (2048/2048) |
| **Complete Scene Exact Match** | 4×6×4×3×3 = 864 tuples | **100.0%** (2048/2048) |

Lemma normalisation (`blake2b-lemma-words-bigrams-l2-v2`) ensures that inflections (e.g., plurals like *spheres*, participles like *orbiting*, synonyms like *stationary*) map to the exact same feature vector as the canonical lemmas.

---

## 3. Synthetic Survival Simulation (TidePool World)

Episodes run for a maximum of 256 steps. Evaluation is conducted across 24 fixed held-out seeds (`91000001` to `91000024`), comparing the learned policy against a random baseline and the scripted heuristic teacher.

### Standard Scarcity (scarcity = 1.0, 24 test seeds)

| Controller | Survival Rate | Mean Steps | Mean Reward |
|---|---|---|---|
| Random Walk | 12.5% (3/24) | 123.5 / 256 | 5.16 |
| Scripted Heuristic | 100.0% (24/24) | 256.0 / 256 | 25.71 |
| Tidal Core (Base BC) | 100.0% (24/24) | 256.0 / 256 | 25.66 |
| **Tidal Core (+ DAgger)** | **100.0%** (24/24) | **256.0 / 256** | **25.68** |

### High Scarcity (scarcity = 2.0, 24 test seeds)

| Controller | Survival Rate | Mean Steps | Mean Reward |
|---|---|---|---|
| Random Walk | 4.2% (1/24) | 100.6 / 256 | 3.38 |
| Scripted Heuristic | 100.0% (24/24) | 256.0 / 256 | 25.69 |
| Tidal Core (Base BC) | 95.8% (23/24) | 252.5 / 256 | 25.10 |
| **Tidal Core (+ DAgger)** | **100.0%** (24/24) | **256.0 / 256** | **25.68** |

---

## 4. DAgger Policy Improvement (Ross et al. 2011)

Under standard behavior cloning (BC), the agent only trains on trajectories the teacher executes. Under extreme environmental stress, small compounding errors take the agent into states the teacher never visited (covariate shift).

To test and resolve this, DAgger was run across 4 rounds (40 episodes/round, decaying teacher probability $\beta_i = 0.5^i$) on an independent validation namespace (`78 << 23 | i`, 48 seeds) spanning scarcity up to 4.0:

### Extreme Scarcity Validation (scarcity = 4.0, 48 validation seeds)

| Model Iteration | Scarcity 1.0 Survival | Scarcity 2.0 Survival | Scarcity 4.0 Survival | Validation Aggregate Score | Decision |
|---|---|---|---|---|---|
| Parent (Base BC) | 100.0% | 100.0% | 18.75% (9/48) | 72.92% | Baseline |
| DAgger Round 1 ($\beta=0.50$) | 100.0% | 100.0% | 79.17% (38/48) | 93.06% | Improved |
| **DAgger Round 2 ($\beta=0.25$)** | **100.0%** | **100.0%** | **97.92% (47/48)** | **99.31%** | **Accepted & Promoted** |
| DAgger Round 3 ($\beta=0.125$) | 100.0% | 100.0% | 95.83% (46/48) | 98.61% | Overtrained |
| DAgger Round 4 ($\beta=0.0625$) | 100.0% | 100.0% | 43.75% (21/48) | 81.25% | Rejected |
| Scripted Teacher (Upper bound) | 100.0% | 100.0% | 100.0% (48/48) | 100.0% | Ground truth |

**Outcome**: Round 2 increased extreme-scarcity survival from **18.75% to 97.92%**, effectively closing the covariate shift gap. Because Round 2 strictly beat the parent, it was automatically promoted via `runs/active_core.json` to serve as the active runtime checkpoint.

---

## 5. Calibration (Temperature Scaling - Guo et al. 2017)

Temperature scaling fits a single positive scalar $T$ per classification head on the dev split to minimize negative log-likelihood:

- Because all 5 scene heads achieved 100.0% accuracy on the dev split, the NLL minimizer has no finite optimum ($T \to 0$).
- The calibration module (`poseidon.calibrate`) detects this degenerate condition and leaves $T = 1.0$ with explicit diagnostic status:
  `"degenerate: every dev prediction correct, NLL minimiser has no finite optimum"`.
- This avoids arbitrary over-sharpening while retaining full transparency in the checkpoint metadata.

---

## 6. Advanced Exact Mathematics & Symbolic Solver

Evaluated on arithmetic, linear equations, quadratic equations, and multi-variable linear systems:

- **Arithmetic expressions** (`(a + b) * c`): 100/100 correct (100.0%).
- **Linear equations** (`a*x + b = c`): 100/100 correct (100.0%).
- **Quadratic equations** (`a*x^2 + b*x + c = 0`):
  - Solves rational roots ($x = 2, 3$), double roots ($x = 3$), real radical roots ($x \sim \pm 1.414$ with exact simplified radical $\sqrt{D}$), and complex conjugate roots ($p \pm qi$).
  - **Self-verification**: Every root is substituted back into the original AST polynomial expression to verify exact equality to 0.
- **Linear systems of 2 equations** (`a1*x + b1*y = c1, a2*x + b2*y = c2`):
  - Solves exact rational pairs $(x, y)$ using Cramer's rule / 2×2 matrix determinant.
  - Detects inconsistent (no solution) and dependent (infinitely many solutions) systems.
  - **Self-verification**: Evaluated on both equations before returning.
- **Implementation**: Bounded exact rational numbers via Python's AST parser; never invokes `eval()`.

---

## 7. Model-Predictive Control (MPC) Planning with Dynamics Head

The Tidal core's action-conditioned transition delta head $\Delta(o, a)$ was harnessed for multi-step model-based planning (`poseidon.planning.ModelPredictivePlanner`):

- **Horizon-2 Batched Imagination**: Evaluates all 36 candidate 2-step future branches ($6 \times 6$) in two batched tensor forward passes on CPU (taking **~2.47s for an entire 256-step episode**).
- **Physical Survival Viability**: Evaluates predicted vital reserves (health, energy, hydration, stamina) and penalizes predicted lethal hazards, exposure spikes, and starvation.
- **Hybrid Decision-Making**: Fuses the reactive policy prior $\pi(a|o)$ with normalized rollout value estimates $Q_{\text{MPC}}(o, a)$.
- **Empirical Performance**:
  - Successfully survives full 256-step horizons in TidePool under scarcity $2.0$ with maximum health ($1.0$).
  - Exposed via CLI (`python -m poseidon world --planner mpc` and `--planner hybrid`).

---

## 8. Hybrid Dense-Lexical Carrier Memory (RRF)

Memory retrieval was upgraded from pure BM25 lexical matching to Reciprocal Rank Fusion (`poseidon.memory.MemoryBank`):

- **Dense Semantic Hashing**: Leverages Blake2b lemma bigram hash features (512-dim L2-normalized vectors) from `poseidon.core` to measure semantic cosine similarity.
- **Reciprocal Rank Fusion (RRF, Cormack et al. 2009)**:
  $$\text{RRF}(d) = \frac{I(d \in \text{BM25})}{60 + \text{rank}_{\text{BM25}}(d)} + \frac{I(d \in \text{dense})}{60 + \text{rank}_{\text{dense}}(d)}$$
- **Semantic Generalization**: Successfully retrieves relevant notes when queries use synonyms or paraphrases without exact token overlap (e.g. retrieving `"filtration unit"` notes for query `"water supply"`), while retaining pure BM25 backward-compatibility for deterministic exact keyword workflows.

---

## 9. Media Artifact Generation

Software-rendered geometric assets generated from learned Tidal predictions:

- **Image mode**: 512×384 PNG generated in ~10 ms.
- **Video mode**: 2.4-second 10 fps animation exported simultaneously as animated GIF (320×240) and MP4 video (`imageio-ffmpeg`).
- **Mesh mode**: 3D geometric meshes exported as OBJ, MTL, and binary glTF (`.gltf`).
- All media files include deterministic cryptographic hashes (`sha256`) and provenance metadata.

---

## 10. Language Model & Adaptation

- **Base conversation**: HuggingFaceTB/SmolLM2-135M-Instruct (pinned commit `12fd25f77366fa6b3b4b768ec3050bf629380bac`).
- **LoRA experiment**: 256 steps on 2,048 deduplicated conversation pairs (42,104 tokens consumed).
  - Baseline dev loss: 0.9609
  - Candidate dev loss: 0.9585
  - **Status**: Candidate retained at `runs/language/adapter` but kept **inactive by default**. A 0.0024 dev loss delta does not establish general conversational improvement without broader evaluation.

---

## 11. Test Suite Verification

- **Total unit & integration tests**: **69 passed in 10.77s** (`pytest -q`).
- **Test Coverage**:
  - Core architecture, feature hashing, lemma normalization.
  - Checkpoint serialization and active core pointer (`runs/active_core.json`).
  - DAgger on-policy data collection, trajectory aggregation, and gated promotion.
  - Temperature scaling and calibration diagnostics.
  - Exact rational arithmetic, quadratic equations, radical reduction, complex roots, and 2-variable linear systems.
  - Model-predictive control planner, multi-step unrolling, and batching.
  - Carrier memory: pure BM25, dense cosine, and hybrid RRF retrieval.
  - Server HTTP endpoints (`/api/status`, `/api/respond`, `/api/remember`).
