# Poseidon Stage 6: Controlled Adaptation and Information Seeking

**Date:** 13 October 2026  
**Status:** Implemented, verified, and audited  
**Milestone:** Stage 6 of `docs/superpowers/specs/2026-10-11-learning-transfer-design.md`  

---

## 1. Executive Summary

This milestone delivers two scientific advancements for the Poseidon cognitive architecture:
1. **Candidate 7 Full-Assay Audit & Promotion:**
   - Evaluated candidate parameter mutation `override_threshold = 0.045` (promoted from `0.040`) across **32 fresh, disjoint out-of-distribution seeds** (`90600000..90600031`) under calibrated ORACLE dynamics.
   - Demonstrated a **`+0.7174` mean reward improvement** and an absolute **`+9.38%` survival gain** ($65.62\% \rightarrow 75.00\%$), while pruning 87 spurious planner overrides.
   - Verified across the full repository test bank and promoted to `poseidon/horizon_x.py`.

2. **Stage 6 Architecture: Controlled Adaptation & Information Seeking:**
   - **Targeted Epistemic Probes vs. Random Equal-Budget Probes:** Formulated active-inference information seeking that prioritizes high epistemic value and future state space dispersion. Compared against equal-budget random exploration probes under identical step/compute limits.
   - **Candidate-Only Sleep/Replay Adaptation Sessions:** Built `CandidateAdapterSession` to guarantee atomic isolation. Updates apply exclusively to candidate weights; any cancelled or unverified session performs complete fail-closed rollback, leaving the incumbent policy and frozen core 100% untouched.
   - **Independent Retention & Adaptation Audit Gate:** Formulated `RetentionAndAdaptationAuditor` enforcing anti-forgetting anchors ($D_{KL}$ retention $\ge 98\%$, regression $\le 2\%$), positive return advantage ($\Delta R \ge \delta_{min}$), survival non-inferiority, and cryptographic SHA-256 receipts.

---

## 2. Candidate 7 Verification: 32-Seed Disjoint Audit

### 2.1 Audit Protocol
- **Incumbent Baseline:** `HorizonXConfig(override_threshold=0.040)`
- **Candidate 7:** `HorizonXConfig(override_threshold=0.045)`
- **Evaluation World:** `POMDPTidePool(POMDPConfig(scarcity=2.8, sensor_noise_std=0.02, fog_threshold=0.65))`
- **Audit Seeds:** Disjoint range $[90600000, 90600031]$ (32 seeds, 64 total rollouts)
- **Frozen Invariants:** Core checkpoint [`runs/tidal_dagger/core.pt`](../../runs/tidal_dagger/core.pt), ORACLE checkpoint [`runs/search_oracle/oracle.pt`](../../runs/search_oracle/oracle.pt).

### 2.2 Empirical Results

| Metric | Baseline ($0.040$) | Candidate 7 ($0.045$) | Delta |
| :--- | :---: | :---: | :---: |
| **Mean Total Reward** | $0.2918$ | **$1.0092$** | **$+0.7174$** |
| **Survival Rate** | $65.62\%$ (21/32) | **$75.00\%$ (24/32)** | **$+9.38\%$** |
| **Total Steps Survived** | 823 | **876** | **$+53$ steps** |
| **Total Planner Overrides** | 612 | **525** | **$-87$ overrides** |
| **Seed Improvement Profile** | — | — | **11 improved, 2 regressed, 19 neutral** |

### 2.3 Scientific Interpretation
In Horizon-X model predictive control, an override is triggered when:
$$\Delta V = V_{\text{candidate}} - V_{\text{base}} > \text{override\_threshold}$$
At $\text{override\_threshold} = 0.040$, near-boundary noise in value estimations frequently prompted low-margin interventions that interrupted base policy foraging. Raising the threshold to $0.045$ prunes 87 noisy interventions, ensuring the controller only intervenes when the predictive model holds high confidence in avoiding hazards.

---

## 3. Stage 6 Architecture: Information Seeking & Controlled Adaptation

### 3.1 Epistemic Information Seeking
Traditional RL balances exploration and exploitation via stochastic action perturbations ($\epsilon$-greedy or entropy regularization). In Poseidon, information seeking is formalized under the Free Energy Principle:
$$G(a) = \underbrace{D_{KL}[q(s'|a) \parallel p(s')]}_{\text{Pragmatic Error}} - \underbrace{\mathbb{E}_q[I(s'; o' | a)]}_{\text{Epistemic Information Gain}}$$

The `EpistemicInformationProber` evaluates candidate actions by their expected epistemic reduction and state dispersion:
```python
score = epistemic_drive * act_bonus + 0.2 * base_logits[act]
```
The benchmark framework (`compare_probes`) executes paired episodes comparing targeted epistemic probes against equal-budget uniform random exploration, recording:
- **Epistemic Uncertainty Reduction:** $\Delta U = U_{\text{initial}} - U_{\text{final}}$
- **State Space Coverage Volume:** $\mathcal{V} = \frac{1}{T} \sum_{t=1}^T \|s_t - \bar{s}\|_2$
- **Action Entropy:** $\mathcal{H}(A) = -\sum_{a} p(a) \log p(a)$

### 3.2 Atomic Candidate Sessions (`CandidateAdapterSession`)
To ensure continuous learning does not degrade production stability:
1. Training updates never touch the incumbent policy in-place.
2. A candidate session deep-copies the residual adapter weights:
   $$\pi_{\text{cand}}(a | s) = \text{Softmax}(\text{Logits}_{\text{core}}(s) + \alpha \Delta \pi_\phi(s))$$
3. Batches from replay buffer or dream consolidator update only $\Delta \pi_\phi$.
4. If training is cancelled or validation fails, `session.cancel()` drops the candidate weights. The incumbent remains identical to its pre-session state.

### 3.3 Independent Retention & Adaptation Auditor
Before admitting any candidate adapter into deployment, `RetentionAndAdaptationAuditor` executes a strict three-tier verification:
1. **Anti-Forgetting Retention Gate:** Evaluates policy agreement across canonical anchor states:
   $$R_{\text{anchor}} = \frac{1}{|\mathcal{D}_{\text{anchor}}|} \sum_{s \in \mathcal{D}} \mathbb{I}[\arg\max \pi_{\text{cand}}(s) = \arg\max \pi_{\text{core}}(s)] \ge 0.98$$
2. **Return Advantage Gate:** Evaluates on held-out seeds to ensure $\Delta R \ge \delta_{\min}$.
3. **Survival Non-Inferiority Gate:** Ensures $\Delta S \ge 0.0$ (no survival regression).
4. **Cryptographic Receipt Ledger:** Produces a signed JSON receipt containing candidate weights SHA-256 and full audit parameters.

---

## 4. Verification & Governance Summary

- **Promoted Parameter:** `override_threshold = 0.045` in [`poseidon/horizon_x.py`](../../poseidon/horizon_x.py).
- **New Module:** [`poseidon/information_seeking.py`](../../poseidon/information_seeking.py).
- **New Unit Tests:** [`tests/test_information_seeking.py`](../../tests/test_information_seeking.py) (5/5 passing).
- **Core Invariant:** Frozen Tidal policy [`runs/tidal_dagger/core.pt`](../../runs/tidal_dagger/core.pt) remains bitwise identical.
- **Server Health:** Port 8787 synchronized (`source_current: True`, `restart_required: False`).
