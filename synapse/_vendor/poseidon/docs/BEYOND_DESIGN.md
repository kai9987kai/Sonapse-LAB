# Supermix Beyond — Adaptive Cognitive World Model

> Scope correction, 9 October 2026: this document is a research agenda. The default
> ensemble contains partially warm-started priors and its hand-written failure
> score is not calibrated. "Authentic" graph labels refer to synthetic modular
> topology, without imported biological connectivity. The episodic recall assay
> checks explicit record storage/retrieval; learned recurrence is not evaluated.
> The v0.2 auditor now measures actual semantic scenes and matched incumbent
> dynamics/survival, reports eligibility, and never activates a candidate.
> [Counterfactual Atlas](ATLAS_DESIGN.md) provides the implemented extension and
> source-bound execution evidence.
## Architectural Specification and Research Agenda

**Status**: Active Research Implementation  
**Codebase**: `poseidon/`  
**License**: MIT (Core implementation and experimental modules)

---

## 1. Executive Summary and Vision

Supermix Beyond represents an evolution from isolated capability demonstrations (language chat, static procedural geometry, reactive behavior cloning) toward an **adaptive cognitive world model** that learns from experience, predicts consequences with epistemic uncertainty, maintains cross-step memory, and improves its behavior over time.

By integrating the architectural rigor of **Supermix Expanse** (connectome wiring experiments, ablation methodologies, distillation) with the operational runtime of **Supermix Poseidon** (learned environment dynamics, survival policy, MPC planning, CPU-bounded execution), Supermix Beyond investigates the central research hypothesis:

> **Core Hypothesis**: *Can a small, locally executable AI system with persistent cross-step memory, learned environmental dynamics, and uncertainty-aware planning adapt to unfamiliar environments more effectively and safely than a purely reactive or uncalibrated single-model baseline?*

---

## 2. Six Experimental Upgrades

```
+-------------------------------------------------------------------------------+
|                         SUPERMIX BEYOND ARCHITECTURE                          |
+-------------------------------------------------------------------------------+
|                                                                               |
|  [ Environment State / 3D WorldLab ]                                          |
|            |                                                                  |
|            v                                                                  |
|  +-------------------------------------+   +-------------------------------+  |
|  |     Tidal Shared Latent Trunk       |   |    Persistent Memory Store    |  |
|  |   - Hashed Blake2b Embeddings       |   |  - Cross-Step Recurrent (h_t) |  |
|  |   - Sparse Adaptive MoE (Top-2)     |<->|  - Episodic Event Replay      |  |
|  |   - Early-Exit Recurrent Depth      |   |  - Delayed Recall Association |  |
|  +-------------------------------------+   +-------------------------------+  |
|            |                                                                  |
|            +-----------------------------------+                              |
|            |                                   |                              |
|            v                                   v                              |
|  +------------------------+        +----------------------------------------+ |
|  | Prior Policy π(a|s)    |        | 3-Model Dynamics Ensemble f_θ(s, a)    | |
|  | Fast Reactive Intuition|        | Head 1: f_θ1  | Head 2: f_θ2 | Head 3  | |
|  +------------------------+        +----------------------------------------+ |
|            |                                   |                              |
|            |                                   v                              |
|            |                       +----------------------------------------+ |
|            |                       | Epistemic Disagreement: U(s, a)        | |
|            |                       | Calibrated Failure Probability: P(fail)| |
|            |                       +----------------------------------------+ |
|            \                                   /                              |
|             \                                 /                               |
|              v                               v                                |
|        +--------------------------------------------+                         |
|        |    Risk-Sensitive MPC Planner              |                         |
|        |    Q*(s, a) = E[R] - λ U(s, a) - β P(fail) |                         |
|        +--------------------------------------------+                         |
|                              |                                                |
|                              v                                                |
|                      [ Selected Action ]                                      |
|                              |                                                |
|            +-----------------+-----------------+                              |
|            |                                   |                              |
|            v                                   v                              |
|  [ Execution in World ]          [ Surprise-Prioritized Buffer ]              |
|                                  - High Disagreement U(s, a)                  |
|                                  - Execution Failure / Hazard Events          |
|                                                |                              |
|                                                v                              |
|                                  [ Gated Promotion Auditor ]                  |
|                                  - Zero-Regression Gate                       |
|                                  - Frozen Evaluation Partition                |
+-------------------------------------------------------------------------------+
```

---

### Experiment A: Predictive World Model with Epistemic Uncertainty (`poseidon/ensemble.py`)

A single deterministic forward head $\hat{s}_{t+1} = s_t + \Delta_\theta(s_t, a_t)$ can be confidently wrong in out-of-distribution or sparsely explored states. Over multi-step imagination horizons, compounding prediction errors lead to catastrophic decisions.

Beyond upgrades the dynamics system to an ensemble of $K = 3$ independently parameterized dynamics heads:

$$\hat{s}_{t+1}^{(k)} = s_t + \Delta_{\theta_k}(s_t, a_t), \quad k \in \{1, 2, 3\}$$

The epistemic uncertainty (model disagreement) is computed across ensemble members:

$$U(s_t, a_t) = \frac{1}{K} \sum_{k=1}^K \left\| \hat{s}_{t+1}^{(k)} - \bar{s}_{t+1} \right\|^2$$

Coupled with a calibrated failure detector estimating survival hazards $P(\text{failure} \mid s_t, a_t)$, the planner optimizes a risk-sensitive objective:

$$a_t^* = \arg\max_a \left[ \mathbb{E}[R \mid a] - \lambda U(s_t, a) - \beta P(\text{failure} \mid a) \right]$$

---

### Experiment B: Biological Neural Circuitry (`poseidon/connectome.py`)

Inspired by connectome research in Supermix Expanse, Beyond implements a 256-node recurrent control module constrained by synaptic graph topologies:

1. **Authentic Modular Connectome**: Incorporates functional neuropil modules (sensory, integrative, central complex, motor), log-normal degree distributions, and rich-club recurrent hub connectivity.
2. **Degree-Preserving Shuffled Control**: Generated via the Maslov-Sneppen double-edge swap algorithm. Rewires edges $(u \to v, x \to y \implies u \to y, x \to v)$ to preserve exact in-degree and out-degree distributions of every individual node while destroying modular hierarchy and clustering.
3. **Erdős–Rényi Random Control**: Matches exact non-zero edge density ($\sim 6.28\%$).

The recurrent dynamics follow a leaky-integrator biological membrane equation:

$$h_{t+1} = (1 - \alpha) h_t + \alpha \tanh\left( W_{\text{in}} x_t + (W_{\text{rec}} \odot M_{\text{adj}}) h_t + b \right)$$

---

### Experiment C: Persistent Memory and Episodic Learning (`poseidon/persistent_memory.py`)

Standard reactive networks reset activations at each step. Beyond decouples memory into two systems:

1. **Cross-Step Recurrent Memory ($h_t$)**: Maintains an evolving internal hidden state across sequential environment steps:
   $$z_t = \sigma(W_z [o_t, a_{t-1}, h_{t-1}])$$
   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tanh(W_h [o_t, a_{t-1}, r_t \odot h_{t-1}])$$
2. **Episodic Memory Store**: Bounded ring buffer recording $(t, x, y, o_t, a_t, r_t, \text{uncertainty}, \text{discoveries})$. Enables content-addressable delayed retrieval based on spatial proximity and resource queries.

---

### Experiment D: Sparse Mixture of Experts with Adaptive Depth (`poseidon/sparse_moe.py`)

Rather than executing all experts unconditionally, Beyond tests dynamic sparse gating:

1. **Sparse Top-2 Gating**: Routes each latent state to the top-2 scoring experts and normalizes their weights, bypassing unselected expert MLPs.
2. **Switch/MoSE Load-Balancing Loss**:
   $$\mathcal{L}_{\text{aux}} = E \sum_{i=1}^E f_i P_i$$
3. **Adaptive Recurrent Depth (Early Exit)**: Computes router confidence margin $M = p_{\text{top1}} - p_{\text{top2}}$. If the decision is clear ($M > \tau_{\text{exit}}$), execution terminates after 1 recurrent pass, saving CPU cycles for ambiguous states.

---

### Experiment E: Surprise-Driven Adaptation Loop (`poseidon/adaptation.py`)

Continuous self-improvement without catastrophic forgetting:

1. **Prioritized Surprise Buffer**: Samples transitions with probability:
   $$P(i) \propto \left( U(s_i, a_i) + \|\hat{s}_{i+1} - s_{i+1}\| + 2.0 \cdot \mathbb{I}(\text{failure}) \right)^\alpha$$
2. **Promotion Auditor**: Candidate models must clear frozen gates:
   - Held-out scene accuracy $\ge 98.0\%$
   - Held-out dynamics MSE $\le 0.040$
   - Multi-seed survival rate $\ge$ baseline control
   - Epistemic calibration: $U(s, a)$ must positively correlate with true prediction error ($r \ge 0.20$).

---

### Experiment F: Interactive 3D World Generation (`poseidon/scene_graph.py`)

Moves beyond isolated shape generation to an **object-centric scene graph**:

- **Entities**: Explicit 3D meshes with transform $(x, y, z)$, scale, material, mass, and velocity.
- **Relational Edges**: Dynamic spatial and causal links (`orbiting`, `sheltering`, `on_top_of`).
- **Interactive Physics**: Updates entity positions under gravitational, orbital, and agent impulse forces.
- **Wavefront OBJ Exporter**: Generates complete multi-object 3D environments viewable locally or imported into Unity.

---

## 3. Unity WorldLab Integration Blueprint

To test cross-environment transfer beyond TidePool:

1. **Environment Interface**:
   - Unity communicates via local IPC/sockets (or Unity ML-Agents Python API).
   - Sensor inputs map to a structured 16-d or 32-d observation vector (vitals, raycast lidar, scent sensors, hazard indicators).
2. **Action Translation**:
   - Discrete macro actions map directly to Unity character controller impulses (`Explore` $\to$ navmesh waypoint traversal; `Flee` $\to$ retreat from hostile tag; `Rest`/`Shelter` $\to$ bunker entry).
3. **WorldLab Scene Construction**:
   - The scene graph from `poseidon/scene_graph.py` exports directly to `.obj` or JSON, enabling procedural layout instantiation within Unity scenes.
