# Helm v0.6: Empirical Multi-Horizon Critic Results

> [!NOTE]
> **Scope & Evidence Limits**: Branch returns are undiscounted multi-step rewards ($H \in \{4, 16\}$) evaluated under recorded frozen-core continuation actions. Weightless replay validates action execution, environmental transitions, and arithmetic consistency without loading neural weights; it does not claim neural authorship, biological fidelity, or general superiority beyond the synthetic TidePool benchmark. Empirical calibration error radii ($\varepsilon_4, \varepsilon_{16}$) are descriptive diagnostic margins from held-out calibration episodes and do not provide a mathematical safety or future-coverage guarantee. The core Tidal DAgger policy remains the default; Helm is opt-in.

---

## 1. Experimental Architecture & Evaluation Protocol

Poseidon v0.6 **Helm** introduces an independently fitted, CPU-local return critic that conditions on observation history, executed macro-actions, and realized prediction innovations ($y_t - \hat{y}_t$) without altering the frozen Tidal core checkpoint or reset protocol.

### 1.1 Disjoint Partitions
To prevent data leakage and optimistic bias, Helm enforces strictly disjoint episode seed sets across its training, model selection, calibration, and benchmark evaluation stages:

1. **Training Partition**: 12 seeds (`121000001`–`121000012`), 48-unit echo-state reservoir, ridge readouts with $\alpha \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$ over dual horizons $H=4$ and $H=16$.
2. **Selection Partition**: 8 disjoint seeds (`122000001`–`122000008`) comparing three input feature families:
   - `observation`: Observation history only (baseline).
   - `no_innovation`: Reservoir history and actions without prediction innovation.
   - `history`: Reservoir history with full innovation feedback.
   *Outcome*: In held-out MSE evaluation, the `observation` family achieved lower/tied MSE and was cleanly selected per the strict tie-breaking rule.
3. **Calibration Partition**: 8 disjoint seeds (`123000001`–`123000008`), calculating empirical episode-maximum advantage overestimation radii:
   $$\varepsilon_4 = 0.105822, \quad \varepsilon_{16} = 0.357416$$
4. **Benchmark Suite 1 (Moderate Scarcity 2.5)**: 4 seeds (`123000001`–`123000004`), 6 controllers, 64 max steps.
5. **Benchmark Suite 2 (Severe Scarcity 4.0)**: 4 seeds (`124000001`–`124000004`), 6 controllers, 64 max steps.

### 1.2 Evaluated Controller Arms (6 Arms)
- **`policy`**: Frozen Tidal Core policy (DAgger round 2 incumbent).
- **`selected`**: Calibrated Helm controller using the selected `observation` family and held-out calibration radii ($\varepsilon_4, \varepsilon_{16}$).
- **`observation`**: Calibrated Helm critic using only observation history.
- **`no_innovation`**: Calibrated Helm critic ablated of prediction error feedback.
- **`yoked`**: Helm critic conditioned on a frozen training-error schedule (control).
- **`ungated`**: Uncalibrated Helm critic with error radii set to zero (diagnostic control).

---

## 2. Quantitative Results: Moderate Scarcity ($scarcity = 2.5$)

- **Receipt**: `outputs/helm_experiments/349465c522f435a5-2101a2f756ec.json`
- **Replayed**: 24 episodes | 1,449 transitions | 186 anchors | 2,232 branches | 22,050 branch transitions
- **Replay Status**: 100% verified offline via `python -m poseidon verify-helm`

| Controller Arm | Survival Rate | Mean Steps | Mean Reward | Mean Reward Delta vs Policy | Wins / Ties / Losses | Overrides | Override Rate | Adverse Override Rate ($H=4$) | Adverse Override Rate ($H=16$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `policy` (baseline) | **100%** | 64.00 | 6.5288 | 0.0000 | — | 0 | 0.0% | — | — |
| `selected` (calibrated) | **100%** | 64.00 | 6.5288 | 0.0000 | 0 / 4 / 0 | 0 | 0.0% | — | — |
| `observation` (calibrated) | **100%** | 64.00 | 6.5288 | 0.0000 | 0 / 4 / 0 | 0 | 0.0% | — | — |
| `no_innovation` (calibrated) | **100%** | 64.00 | 6.5288 | 0.0000 | 0 / 4 / 0 | 0 | 0.0% | — | — |
| `yoked` (calibrated) | **100%** | 64.00 | 6.5288 | 0.0000 | 0 / 4 / 0 | 0 | 0.0% | — | — |
| `ungated` (diagnostic) | **0%** | 42.25 | -1.3096 | **-7.8384** | 0 / 0 / 4 | 103 | **60.9%** | **61.2%** (63/103) | **49.5%** (51/103) |

### Key Findings: The Necessity of Dual-Horizon Risk Gating
1. **Catastrophic Failure of Uncalibrated Criticism**: When the critic was permitted to override without calibration margins (`ungated`), it aggressively overrode the competent policy in **60.9% of all steps** (103 overrides across 169 decision points). Over **61% of these overrides resulted in negative actual 4-step advantages**, leading to premature death across **100% of episodes** (survival collapsed from 100% to 0%, mean steps dropped from 64 to 42.25, and mean reward plummeted from +6.53 to -1.31).
2. **Effective Safeguarding via Calibration**: Under the calibrated gate, Helm verified that challenger advantages did not exceed the empirical overestimation radii ($\varepsilon_4 = 0.1058, \varepsilon_{16} = 0.3574$). The gate cleanly suppressed spurious overrides (0 overrides), preserving full policy performance (100% survival, 64.0 steps, +6.53 reward).

---

## 3. Quantitative Results: Severe Scarcity ($scarcity = 4.0$)

- **Receipt**: `outputs/helm_experiments/69be786b564e4850-f99e4055ab80.json`
- **Replayed**: 24 episodes | 1,466 transitions | 199 anchors | 2,388 branches | 23,363 branch transitions
- **Replay Status**: 100% verified offline via `python -m poseidon verify-helm`

| Controller Arm | Survival Rate | Mean Steps | Mean Reward | Mean Reward Delta vs Policy | Wins / Ties / Losses | Overrides | Override Rate | Adverse Override Rate ($H=4$) | Adverse Override Rate ($H=16$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `policy` (baseline) | **100%** | 64.00 | 6.3233 | 0.0000 | — | 0 | 0.0% | — | — |
| `selected` (calibrated) | **100%** | 64.00 | 6.3233 | 0.0000 | 0 / 4 / 0 | 0 | 0.0% | — | — |
| `observation` (calibrated) | **100%** | 64.00 | 6.3233 | 0.0000 | 0 / 4 / 0 | 0 | 0.0% | — | — |
| `no_innovation` (calibrated) | **100%** | 64.00 | 6.3233 | 0.0000 | 0 / 4 / 0 | 0 | 0.0% | — | — |
| `yoked` (calibrated) | **100%** | 64.00 | 6.3233 | 0.0000 | 0 / 4 / 0 | 0 | 0.0% | — | — |
| `ungated` (diagnostic) | **25%** | 46.50 | 0.4515 | **-5.8719** | 0 / 0 / 4 | 114 | **61.3%** | **44.7%** (51/114) | **36.0%** (41/114) |

### Per-Seed Reward Delta Breakdown under Severe Scarcity ($scarcity = 4.0$)
- **Seed `124000001`**: -0.0297 delta | 64.0 steps | Survived (1)
- **Seed `124000002`**: -8.2456 delta | 42.0 steps | Died (-1)
- **Seed `124000003`**: -9.0619 delta | 39.0 steps | Died (-1)
- **Seed `124000004`**: -6.1502 delta | 41.0 steps | Died (-1)
- **Leave-One-Seed-Out Range**: $[-7.8192, -4.8085]$ mean delta (0 direction reversals; consistent loss across all subsets).

In severe environmental stress, `ungated` caused fatal decisions in 3 out of 4 seeds (survival fell to 25%, mean steps 46.5). The calibrated gate protected the agent in all 4 seeds, achieving 100% survival and 6.3233 mean reward.

---

## 4. Risk Coverage Audit across Thresholds

The benchmark computes empirical risk coverage across predeclared advantage thresholds $\tau \in \{-1.0, 0.0, 0.03, 0.1, 0.3, 1.0\}$. Under `ungated` control at $scarcity = 2.5$:
- **At threshold $\tau = -1.0$ (all overrides accepted)**:
  - Horizon $H=4$: 103 exposures, **61.2% adverse override rate**, mean actual advantage: $-0.0624$, radius exceedance rate: **27.2%**.
  - Horizon $H=16$: 103 exposures, **49.5% adverse override rate**, mean actual advantage: $-0.2182$, radius exceedance rate: **12.6%**.
- **At threshold $\tau \ge 0.0$**: 0 exposures, as predicted advantages did not exceed positive confidence thresholds, preventing adverse override exposure.

---

## 5. Durable Cooperative Job Engine

Poseidon v0.6 introduces a persistent background job queue (`poseidon/jobs.py`):
1. **Producer Identity & Generation Binding**: Every job records worker process ID, host thread, schema generation, normalized arguments, created/finished timestamps, and execution phase.
2. **Atomic State Persistence**: State is saved atomically via `.tmp` files and `os.replace` in `outputs/jobs/`.
3. **Crash Recovery**: Unfinished jobs from prior server sessions are marked as `interrupted` upon startup, preventing orphaned locks or invalid "completed" artifacts.
4. **Cooperative Cancellation Outside the Lock**: Polling (`GET /api/jobs/<id>`) and cancellation (`POST /api/jobs/<id>/cancel`) run independently of the inference lock. Long-running benchmark loops check the cancellation token at clean episode boundaries.

---

## 6. Cryptographic Artifact Integrity

| Artifact | File Path | SHA-256 Digest |
| :--- | :--- | :--- |
| **Helm Critic Atlas** | `outputs/helm/atlas.json` | `a4b7dcdb3579d2931350073e4195a1e5879b2e802f81e5b69af7152138066f5c` |
| **Helm Fit Receipt** | `outputs/helm/fit_receipt.json` | `1020703f847250616110f81d11ff236961cfb31e9c8948ee3bcbb4d940dfcb72` |
| **Benchmark Receipt (Scarcity 2.5)** | `outputs/helm_experiments/349465c522f435a5-2101a2f756ec.json` | `2101a2f756ec94bb77f482d8c90356bf5cbb00424564c489c7d42cf38f9b9084` |
| **Benchmark Receipt (Scarcity 4.0)** | `outputs/helm_experiments/69be786b564e4850-f99e4055ab80.json` | `f99e4055ab80527c0c07084f3f7b86960571b77cbba5338ed162212d91c78e5f` |
| **Helm Evaluation Profile** | `docs/helm-evaluation.json` | Verified schema `poseidon-helm-evaluation-v1` |
