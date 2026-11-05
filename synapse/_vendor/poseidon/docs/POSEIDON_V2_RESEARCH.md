# Supermix Poseidon v2.0: Experimental Prediction, Memory and Planning Modules

**Project:** kai9987kai/Supermix-Poseidon; engineering documentation with no external institutional authorship claimed.
**System Version:** `2.0.0`  
**Audited Revision:** `97eebc637865a815b458c50fb7ef8200ef1c61df` (11 October 2026)
**Current Receipt Schema:** `poseidon-cognitive-experiment-v3` (legacy v2 checksum-only receipts are rejected)
**Core Baseline Weights:** Frozen Tidal Core (`runs/tidal_dagger/core.pt`), SHA-256 `a053104ae63dc7f86baa15b4b34e46730e1ccb92117f221801fb21f90d2d6c27`.

**Evidence status, 12 October 2026:** This document contains the original module designs. The [learning foundation report](research/learning-foundation-2026-10-12.md) and [residual prediction results report](research/residual-prediction-results-2026-10-12.md) take precedence for current behavior: explicit trained ORACLE integration, linear-residual state anchoring, frozen-policy fallback, one-step planning, full controller/environment replay, isolated parameter search and measured candidate limitations. The earlier [audit](research/architecture-audit-2026-10-11.md) is preserved as historical evidence. The [operation design](superpowers/specs/2026-10-11-learning-transfer-design.md) specifies remaining learned-memory and structural-transfer work.

---

## 1. Abstract

Supermix Poseidon v2.0 adds experimental modules to the trained Tidal core and existing controllers. Their usefulness as a combined learning system remains to be established. The development questions include:
1. Saturated synthetic test environments obscuring statistical significance.
2. Uncalibrated heuristic variance metrics masquerading as epistemic uncertainty.
3. Rigid risk penalties that provoked the **"pessimism trap"**—causing agents facing critical starvation or dehydration to freeze or default to doomed actions because all high-reward harvesting attempts carried uncertainty penalties.
4. Absence of genuine temporal context integration and value-gated episodic memory.
5. Vulnerability to catastrophic forgetting during continual policy adaptation.

Poseidon v2.0 includes six experimental mechanisms:
- **ORACLE**: Independently initialized runtime heads, a disagreement/error curve and recursive trajectory imagination. The separate research trainer implements whole-episode bootstrap and held-out calibration; its pilot is inactive.
- **MNEMOSYNE**: A 3-tier memory hierarchy with a GRU, priority-weighted episodic retrieval and spatial resource estimates. Five intervention modes exist; a causal control-benefit assay is pending.
- **HORIZON-X**: Multi-horizon planning (default $H \in \{1,4,8,16\}$) with reserve-dependent risk penalties. The intended response to depletion needs measured comparison against alternatives.
- **ADAPT**: Evidence-gated continual learning pipeline utilizing residual policy perturbation networks, anti-forgetting anchor datasets ($\le 2\%$ allowable regression), and audited promotion gates.
- **NEURO**: Neuromodulated sparse connectome routing with small-world topological graph masks and metabolic gain scaling.
- **WORLDLAB**: Synthetic partially observed TidePool variants with fog, sensor noise, tidal effects and a simulation bridge contract.

The v3 cognitive receipt verifier establishes source/model binding, complete pairing, exact controller/environment replay and recomputed summaries. Reproducibility does not by itself establish improved performance.

---

## 2. Core Subsystem Architectures

### 2.1 POSEIDON ORACLE: Calibrated Predictive World Model Ensemble

The legacy diagnostic ORACLE contains $K$ independently initialized dynamics heads $\theta_1, \dots, \theta_K$. Cognitive coordinators now optionally load the distinct trained research ensemble. The formula describes the shared state-prediction interface:

$$\hat{s}_{t+1}^{(k)} = s_t + \Delta_{\theta_k}(s_t, a_t)$$

Each head features dedicated prediction branches:
- State delta: $\Delta s \in \mathbb{R}^{16}$
- Immediate reward: $\hat{r} \in \mathbb{R}$
- Resource depletion cap delta: $\Delta c \in \mathbb{R}^2$
- Catastrophic failure probability: $\hat{p}_{\text{fail}} \in [0, 1]$

#### Epistemic Uncertainty Estimation
Raw predictive disagreement is an ensemble diagnostic; whether it tracks epistemic error must be tested:

$$U_{\text{raw}}(s_t, a_t) = \frac{1}{K} \sum_{k=1}^K \|\Delta_{\theta_k}(s_t, a_t) - \bar{\Delta}\|^2$$

#### Empirical Calibration
To prevent ungrounded variance assumptions, ORACLE fits an empirical power-law calibration curve mapping raw disagreement to observed validation prediction error:

$$U_{\text{cal}}(s, a) = a \cdot U_{\text{raw}}^b + c$$

The implemented fitting routine uses log-space regression and an offset, not quantile regression. Default buffers are present even when `is_calibrated=false`. Its `ece` output is mean absolute mismatch between fitted error estimates and observed errors, not a probability calibration statistic. The raw runtime output name `calibrated_uncertainty` does not itself establish calibration.

#### Trajectory Imagination
ORACLE recursively projects hypothetical action sequences $[a_0, a_1, \dots, a_{H-1}]$ forward in time:

$$\hat{s}_{t+1} = \text{clamp}(\hat{s}_t + \bar{\Delta}(\hat{s}_t, a_t), 0, 1)$$
$$\hat{R}_{\text{tot}} = \sum_{t=0}^{H-1} \hat{r}_t, \quad P_{\text{surv}} = \prod_{t=0}^{H-1} (1 - \hat{p}_{\text{fail}, t})$$

---

### 2.2 POSEIDON MNEMOSYNE: 3-Tier Cognitive Memory

MNEMOSYNE structures synthetic memory into three software components:

```
[Observation o_t] ---> [Prediction Error e_t]
        |                       |
        v                       v
+-------------------------------------------------------+
| Tier 1: Working Memory (GRU Cell)                     |
| h_t = GRU([o_t, a_{t-1}, e_{t-1}], h_{t-1})           |
+-------------------------------------------------------+
        |
        v
+-------------------------------------------------------+
| Tier 2: Value-Gated Episodic Store                    |
| Priority P(i) ~ (alpha_r |r_i| + alpha_e e_i) * sim  |
+-------------------------------------------------------+
        |
        v
+-------------------------------------------------------+
| Tier 3: Topological Spatial Cognitive Map            |
| Bayesian Replenishment LCB(t), Patch Graph P(v|u, a)  |
+-------------------------------------------------------+
```

1. **Tier 1 (Working Memory)**: A recurrent GRU cell integrating observation $o_t$, prior action $a_{t-1}$, and world-model prediction surprise $e_{t-1} = \|o_t - \hat{o}_t\|_2$, maintaining continuous latent context $h_t \in \mathbb{R}^{64}$.
2. **Tier 2 (Episodic Store)**: Capacity-bounded buffer storing experienced transitions $(o_t, a_t, r_t, o_{t+1}, e_t, h_t)$. Retrieval queries calculate priority-weighted relevance:
   $$P(i) \propto \left(\alpha_r |r_i| + \alpha_e e_i + \alpha_{\text{rec}} \gamma^{t - t_i}\right) \cdot \frac{q \cdot o_i}{\|q\| \|o_i\|}$$
3. **Tier 3 (Spatial Map)**: Online Bayesian estimation of patch replenishment rates:
   $$\text{LCB}_{\text{food}}(t) = \min\left(C_{\text{food}}, o_{\text{obs}} + \Delta t \cdot \max(0, \mu_{\text{food}} - z \cdot \sigma_{\text{food}})\right)$$

#### Matched Lesion Assay Protocol
MNEMOSYNE implements five intervention modes intended for a future causal memory assay. Their presence and unit tests do not prove memory utility:
- `intact`: All three tiers fully functioning.
- `erased`: Memory reset to zero at every tick (context wiped).
- `shifted`: Circular roll of memory embeddings (preserves vector norm and variance while severing semantic alignment).
- `random_retrieval`: Episodic recall selects uniform random past events.
- `memory_free`: Memory subsystems bypassed (reactive policy baseline).

---

### 2.3 POSEIDON HORIZON-X: Dynamic Risk-Modulated Multi-Horizon Planner

Standard risk-sensitive planners penalize epistemic uncertainty using a fixed multiplier $\lambda_0$:

$$V(a) = \mathbb{E}[R] - \lambda_0 U(s, a) - \beta P_{\text{fail}}$$

**The Pessimism Trap:** When an agent's food or water collapses to critical levels ($\le 0.05$), remaining in the current depleted patch guarantees death. However, traveling toward distant unvisited patches carries high predictive uncertainty $U(s, a)$. Under a static $\lambda_0$, the planner assigns extreme negative utility to movement, paralyzing the agent into certain starvation.

#### Dynamic Reserve Modulation
HORIZON-X computes a real-time metabolic reserve index:

$$\rho_t = \text{health}_t \cdot \min(\text{energy}_t, \text{hydration}_t)$$

The uncertainty penalty is dynamically modulated:

$$\lambda_t = \lambda_0 \cdot \max(\mu_{\min}, \rho_t)$$

- When reserves are abundant ($\rho_t \approx 1.0$), $\lambda_t \to \lambda_0$, enforcing strict risk aversion.
- When reserves collapse ($\rho_t \to 0$), $\lambda_t$ scales down to $\mu_{\min} \cdot \lambda_0$ (e.g. $10\%$ of base risk penalty), enabling bold exploratory travel when starvation is imminent.

#### Multi-Horizon Synthesis
Candidate actions are projected across horizons $H \in \{1, 4, 8, 16\}$:

$$V^{(H)}(a) = \frac{1}{H} R^{(H)}(a) - \lambda_t U^{(H)}(a) - \beta P_{\text{fail}}^{(H)}(a)$$
$$S(a) = \sum_{H} w_H V^{(H)}(a)$$

An override against the frozen Tidal baseline is issued if and only if $S(a^*) - S(a_{\text{base}}) > \tau_{\text{override}}$.

---

### 2.4 POSEIDON ADAPT: Continual Learning with Anchor Retention

To allow post-deployment policy improvement without destabilizing the core:
1. **Residual Policy Perturbation**:
   $$\pi_{\text{adapted}}(a | s) = \text{Softmax}\left(\text{Logits}_{\text{Tidal}}(s) + \alpha \cdot \Delta \pi_\phi(s)\right)$$
   where $\Delta \pi_\phi$ is zero-initialized, ensuring identity mapping at origin.
2. **Anti-Forgetting Anchor Dataset**:
   Maintains a distribution of canonical reference states and frozen Tidal action probabilities. During training on the online replay buffer, the loss includes an explicit KL constraint:
   $$\mathcal{L} = \mathcal{L}_{\text{replay}} + \lambda_{\text{anchor}} D_{\text{KL}}\left(p_{\text{Tidal}}(s) \parallel p_{\text{adapted}}(s)\right)$$
3. **Formal Promotion Gate**:
   A candidate adapter is audited on held-out validation seeds. It is promoted if and only if:
   - Mean return advantage $\Delta \bar{R} \ge +0.04$
   - Anchor distribution regression $\le 2.0\%$

---

### 2.5 POSEIDON WORLDLAB: POMDP Benchmark & External Bridge

WorldLab adds synthetic challenges intended to probe benchmark saturation:
- **Fog of War**: When weather severity $> 0.65$, scent gradients from adjacent patches are masked.
- **Sensor Noise**: Gaussian noise $\mathcal{N}(0, \sigma^2)$ injected into vital and sensory channels.
- **Tidal Oscillations**: Patch water tables oscillate sinusoidally:
  $$W_{\text{eff}}(t) = W_{\text{base}} \cdot \left(1.0 + A_{\text{tide}} \sin\left(\frac{2\pi t}{T_{\text{tide}}}\right)\right)$$
- **WorldBridgeProtocol**: JSON/TCP contract for external simulation adapters. Actual compatibility requires unit, action, episode/tick and lifecycle validation for each adapter.

---

## 3. Experimental Protocol & Verification

The benchmark evaluates six matched arms over identical pseudo-random seeds:
1. `intact`: Full Poseidon v2.0 Cognitive Agent
2. `erased`: Working memory erased every step
3. `shifted`: Episodic memory circularly rolled
4. `random_retrieval`: Random episodic recall
5. `memory_free`: Pure reactive baseline (ablated memory)
6. `frozen_tidal`: Frozen unaugmented Tidal baseline

The benchmark produces a JSON receipt. Its current verifier validates only the recomputable digest; replay, semantic identity and outcome acceptance are pending.

---

## 4. Summary of Version 2.0 Deliverables

| Subsystem | Module | Key Innovation |
| :--- | :--- | :--- |
| **ORACLE** | `poseidon/oracle.py` | Initialized runtime heads, fitted error curve, recursive imagination; trained research artifact is separate |
| **MNEMOSYNE** | `poseidon/mnemosyne.py` | 3-tier memory (working GRU, episodic store, spatial map), 5 matched lesion assays |
| **HORIZON-X** | `poseidon/horizon_x.py` | Multi-horizon risk planner, dynamic metabolic reserve modulation ($\lambda_t$) |
| **ADAPT** | `poseidon/adapt.py` | Residual adapter, anti-forgetting anchor dataset ($D_{\text{KL}}$), promotion gate |
| **NEURO** | `poseidon/neuro.py` | Watts-Strogatz small-world connectome mask, neuromodulated gain routing |
| **WORLDLAB** | `poseidon/worldlab.py` | POMDP TidePool (fog-of-war, tides, sensor noise), WorldBridge TCP/JSON protocol |
| **AGENT** | `poseidon/cognitive_agent.py`| Master coordinator unifying all subsystems around frozen Tidal core |
| **EXPERIMENT**| `poseidon/cognitive_experiment.py`| 6-arm benchmark and checksum verifier; replay and causal memory evidence pending |
