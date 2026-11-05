# Horizon Atlas design

Horizon Atlas is an opt-in multi-horizon advantage controller for Poseidon's synthetic TidePool world. It complements the frozen Tidal core with an associative memory of multi-step undiscounted returns under core policy execution, models action return advantages over a lookahead horizon \(H = 16\), applies observable resource-depletion guards, and gates interventions using descriptive error radii from held-out episodes.

The learned Tidal core remains the default controller. Fitting and running Horizon Atlas do not alter neural checkpoint weights, modify the active-core pointer, or promote a candidate.

This document details the architecture, mathematical formulations, calibration protocol, and experiment assay. The implementation is in [`poseidon/horizon.py`](../poseidon/horizon.py), [`poseidon/horizon_experiments.py`](../poseidon/horizon_experiments.py), and [`poseidon/runtime.py`](../poseidon/runtime.py).

---

## 1. Motivation and Problem Formulation

In Poseidon v0.2 ([`Atlas v2`](ATLAS_DESIGN.md)) and v0.3 ([`Contrast Atlas`](CONTRAST_DESIGN.md)), candidate interventions evaluated one-step reserve utility increments:
\[
\Delta U_1(a) = U(y(o, a)) - U(y(o, \pi(o)))
\]
While Contrast Atlas achieved a 2.5% reduction in transition prediction MSE and 100% positive audited one-step utility at overrides, mean episode reward under high scarcity remained slightly below or tied with the base policy baseline ($-0.0055$ delta).

### Why One-Step Utility Fails to Improve Long-Term Reward
1. **Myopic Greed vs. Resource Preservation**: A 1-step utility function prioritizes immediate ingestion ($forage$ or $drink$) even when current reserves are adequate, consuming local patch reserves prematurely.
2. **The Resource Depletion Trap**: When a patch's food or water level falls below the gathering threshold ($obs[6] < 0.04$ or $obs[7] < 0.04$), executing $forage$ or $drink$ produces zero yield, expends metabolic stamina/energy, and risks fatal depletion loops. The core policy frequently falls into this trap under severe scarcity ($scarcity \ge 2.5$).
3. **Compound Horizons**: A decision to $explore$ or move toward shelter yields negative 1-step utility (energy expenditure without immediate replenishment) but yields positive cumulative return over an extended horizon by discovering replenished patches or evading exposure.

---

## 2. Multi-Horizon Advantage Modeling

### Trajectory Returns
For an observation $o_t$ at step $t$ in episode $\tau$, the implementation uses undiscounted multi-step return under core policy execution over horizon $H$. In the notation below, $\gamma=1$; no configurable discount is applied:
\[
G_t^{(H)} = \sum_{k=0}^{H-1} \gamma^k r_{t+k}
\]
where $r_{t+k}$ is the scalar survival reward received from the environment at step $t+k$.

### Counterfactual Action Return and Advantage
At anchor snapshot $S_t$, taking action $a \in \{0, \dots, 5\}$ transitions the environment to $S'_{t+1} = \text{step}(S_t, a)$, yielding immediate reward $r_0(a)$ and next observation $o'_{1}(a)$. Subsequent steps $k = 1, \dots, H-1$ follow the frozen core policy:
\[
a_k = \arg\max_a \pi_{\text{core}}(a \mid o_k), \quad S'_{t+k+1} = \text{step}(S'_{t+k}, a_k)
\]
The cumulative $H$-step trajectory return for action $a$ is:
\[
Q^{(H)}(S_t, a) = r_0(a) + \sum_{k=1}^{H-1} \gamma^k r_k(a)
\]
The true counterfactual multi-horizon advantage of action $a$ over the incumbent policy action $p = \pi_{\text{core}}(o_t)$ is:
\[
A^{(H)}(S_t, a) = Q^{(H)}(S_t, a) - Q^{(H)}(S_t, p)
\]
By construction, $A^{(H)}(S_t, p) = 0$.

---

## 3. Architecture of Horizon Atlas

```
                        Observation (16-d)
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       Frozen Tidal Policy            k-NN Associative Memory
     π(o) -> incumbent action p        Query per action a in {0..5}
               │                       Distance: weighted L2 on reserves
               │                               │
               │                               ▼
               │                     Retrieved Advantage Ä_H(a)
               │                               │
               └───────────────┬───────────────┘
                               ▼
               Depletion Trap & Vitality Filter
               - obs[6] < 0.04 -> ban forage
               - obs[7] < 0.04 -> ban drink
               - near-lethal reserves -> enforce vital action
                               │
                               ▼
               Calibrated Advantage Margin Test
               Ä_H(a*) - ε_cal > margin (0.02)
                               │
                      ┌────────┴────────┐
                 Pass │                 │ Fail / Fallback
                      ▼                 ▼
             Execute Action a*     Execute Incumbent Policy p
```

### Associative Trajectory Memory
Rather than fitting an unconstrained parametric neural regressor—which drifts out of distribution and hallucinates positive advantage on doomed trajectories—Horizon Atlas uses a non-parametric associative memory of validated trajectory anchors.

Each record in the memory stores:
- `observation`: normalized 16-dimensional observation vector $o$.
- `action`: the initial branch action $a \in \{0, \dots, 5\}$.
- `policy_action`: incumbent action $p$.
- `multi_step_return`: the measured $H$-step undiscounted return $Q^{(H)}(o, a)$.
- `advantage`: the relative advantage $A^{(H)}(o, a) = Q^{(H)}(o, a) - Q^{(H)}(o, p)$.
- `survived_horizon`: boolean indicating survival through $H$ steps.

### Metric Distance and Querying
Given an observation $o$, for each candidate action $a \in \{0, \dots, 5\}$:
The distance to stored record $(o_i, a)$ is evaluated across observation dimensions with vital reserve weighting:
\[
d(o, o_i) = \sqrt{\sum_{j=1}^{16} w_j (o[j] - o_i[j])^2}
\]
Weights $w$ emphasize vital reserves (health, energy, hydration, stamina) and resource availability (food, water) while downweighting background channels (weather, time, scent).

The $k$-nearest records (default $k = 5$) are retrieved within maximum support radius $R_{\text{support}} = 0.40$. The point advantage estimate $\hat{A}^{(H)}(o, a)$ is the distance-weighted average:
\[
\hat{A}^{(H)}(o, a) = \sum_{i \in \mathcal{N}_k(o, a)} \frac{d(o, o_i)^{-1}}{\sum_{j} d(o, o_j)^{-1}} A^{(H)}(o_i, a)
\]

---

## 4. Physical Vitality Constraints and Depletion-Trap Guard

A machine-learned advantage estimate can be brittle near boundaries. Horizon Atlas incorporates strict physical guards:

### 1. Depletion-Trap Detection
In TidePool, foraging or drinking when local patch resources are exhausted yields $0.00$ intake while incurring metabolic cost.
```python
if action == FORAGE and observation[6] < 0.04:  # Food exhausted
    banned_actions.add(FORAGE)
if action == DRINK and observation[7] < 0.04:   # Water exhausted
    banned_actions.add(DRINK)
```
If the incumbent policy proposes a banned depletion action, the controller immediately searches for an un-banned vitality action (e.g., $explore$ to leave the depleted patch, or $rest$).

### 2. Critical Vital Shortfall Override
If any vital reserve falls below its emergency threshold:
- $\text{energy} < 0.12 \implies$ prioritize replenishment or rest.
- $\text{hydration} < 0.12 \implies$ prioritize drinking (if water available) or relocation.
- $\text{stamina} < 0.08 \implies$ ban high-exertion exploration.
- $\text{threat} > 0.60 \implies$ prioritize shelter or fleeing.

---

## 5. Held-Out Calibration and Empirical Margin Gate

### Partitions
Horizon Atlas enforces strict, non-overlapping seed partitions:
- **Training partition** (default 12 seeds, 1,728 transitions): generates trajectory anchor records for memory.
- **Calibration partition** (default 8 seeds, 1,152 transitions): measures empirical advantage estimation errors.
- **Audit partition** (disjoint, starting at `109000001` or `110000001`): independent multi-controller evaluation.

### Advantage Error Calibration
On calibration episodes, at each anchor step, the controller computes the point advantage $\hat{A}^{(H)}(o, a)$ and compares it against the true audited multi-horizon counterfactual advantage $A^{(H)}_{\text{true}}(o, a)$:
\[
e_{\text{anchor}} = \max_{a} \max\left(0, \hat{A}^{(H)}(o, a) - A^{(H)}_{\text{true}}(o, a)\right)
\]
\[
e_{\text{episode}} = \max_{t \in \text{episode}} e_{\text{anchor}, t}
\]
The calibrated advantage error radius $\varepsilon_{\text{adv}}$ is computed as the 90th percentile of episode-maximum errors:
\[
\varepsilon_{\text{adv}} = \text{Quantile}_{0.90}\left(\{ e_{\text{episode}, m} \}_{m=1}^M \right)
\]

### Calibrated Gate Decision Rule
A non-incumbent action $a^* \ne p$ can override the incumbent policy if and only if:
1. $a^*$ is not banned by the depletion-trap or vitality guards.
2. $a^*$ has at least $k_{\text{min}}$ neighbors within support radius $R_{\text{support}}$.
3. The empirical lower bound on advantage strictly exceeds the override margin:
\[
\text{Margin}(a^*) = \hat{A}^{(H)}(o, a^*) - \varepsilon_{\text{adv}} > \delta_{\text{margin}} \quad (\delta_{\text{margin}} = 0.02)
\]
If any condition fails, the controller falls back to the incumbent policy $p$, recording the explicit `fallback_reason`.

---

## 6. Experimental Assay and Replay Verification

### Nine Paired Controllers
The horizon experiment suite evaluates nine controllers on identical seeds and environments:
1. `policy`: frozen core Tidal policy (DAgger).
2. `neural_mpc`: two-step neural world model lookahead.
3. `atlas_v2`: residual counterfactual memory with absolute vital bounds.
4. `contrast`: matched-action residual memory with 1-step paired utility gate.
5. `horizon`: full Horizon Atlas v4 (multi-horizon memory + depletion guards + calibrated gate).
6. `horizon_no_memory`: Horizon Atlas with memory disabled (relies on depletion guards and baseline prior).
7. `horizon_ungated`: Horizon Atlas with calibration radius set to 0 (uncalibrated point advantage).
8. `heuristic`: simulator task teacher.
9. `random`: uniform random control.

### Weight-Independent Replay Verifier
Because Horizon Atlas and the audit harness operate deterministically through Blake2b-keyed event randomness, every experiment receipt can be replayed and verified without PyTorch weights, GPUs, or neural runtimes:
```powershell
python -m poseidon verify-horizon outputs/horizon_experiments/<receipt>.json
```
The verifier re-simulates every transition, audits the six counterfactual physical branches, verifies every probability vector and advantage margin, and confirms cryptographic receipt hashes.
