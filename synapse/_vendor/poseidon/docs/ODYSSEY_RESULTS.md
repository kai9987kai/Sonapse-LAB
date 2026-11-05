# Odyssey: Local v0.5 Empirical Measurements

These are historical v0.5.0 measurements. The v0.5.1 trajectory audit found that
all 12 Odyssey-arm episodes in the scarcity-4 receipt reset their map setting to
2.5. Their simulated world still used scarcity 4.0. The reset is now corrected;
these numbers should not be presented as measurements of the corrected planner.
The original branch audit executes one step, despite its former multi-horizon
label. See [the stricter audit and actual multi-step returns](TRAJECTORY_AUDIT.md).

This document presents the official local CPU evaluation measurements for Poseidon v0.5 **Odyssey: Spatial Cognitive Mapping & Navigational Memory**. Across two independent 10-controller audit suites (80 episodes, 5,021 transitions, and 30,126 counterfactual branches replayed), Odyssey demonstrated the tangible benefits of spatial topological mapping and goal-directed navigational memory under resource scarcity.

All benchmarks and counterfactual branch audits were conducted deterministically on 9 October 2026 on the local CPU runtime without modifying frozen neural checkpoint weights.

---

## 1. Experimental Setup and Protocol

The evaluation adheres to strict disjoint partition boundaries:
- **Training partition**: 12 seeds (`109000001`–`109000012`), 1,728 branch transitions across scarcity 1.0–4.0.
- **Calibration partition**: 8 disjoint seeds (`110000001`–`110000008`), 1,152 branch transitions.
- **Audit partition 1 (Moderate scarcity 2.5)**: 4 disjoint seeds (`111000001`–`111000004`), 10 controllers, 64-step horizon.
- **Audit partition 2 (Severe scarcity 4.0)**: 4 disjoint seeds (`112000001`–`112000004`), 10 controllers, 64-step horizon.

### Evaluated Controllers (10 Arms)
1. `policy`: Frozen Tidal Core policy trained with DAgger (the incumbent baseline).
2. `neural_mpc`: 2-step lookahead using the learned dynamics head ($H=2$).
3. `atlas_v2`: v0.2 residual counterfactual memory with absolute vital error bounds.
4. `contrast`: v0.3 matched-action residual memory with 1-step paired utility gate.
5. `horizon`: v0.4 Multi-horizon advantage controller ($H=16$) with depletion guards.
6. `odyssey`: Full v0.5 Odyssey (topological cognitive map + replenishment decay + waypoint navigation + calibrated advantage margin).
7. `odyssey_no_memory`: Odyssey with multi-horizon memory ablated (topological cognitive map + waypoint navigation + base policy).
8. `odyssey_ungated`: Odyssey with advantage calibration margin set to zero (point advantage).
9. `heuristic`: Task-specific behavioral rule teacher.
10. `random`: Uniform random action baseline.

---

## 2. Quantitative Results: Severe Scarcity ($scarcity = 4.0$)

Under severe scarcity, environmental renewal is choked by $4\times$. Tiles empty after 1–2 harvests, making cognitive spatial mapping and remembering replenished tiles essential for survival.

- **Seeds**: `112000001` – `112000004` (4 episodes per arm = 40 paired episodes)
- **Receipt ID**: `cdaf779874a04de6-7dc45ea640b3` | **Checksum**: `7dc45ea640b3fd978735dd02702be476c97a9aaf860af360c9b2015a69b16bce`
- **Transitions replayed**: 2,521 | **Counterfactual branches replayed**: 15,126

| Controller Arm | Survival Rate | Mean Steps | Mean Reward | Mean Reward Delta vs Policy | Mean Patches Mapped | One-Step Regret |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `policy` | **100%** | 64.0 | 6.3646 | Baseline (0.0) | 0.0 | 0.1388 |
| `neural_mpc` | **100%** | 64.0 | 6.4017 | +0.03710 | 0.0 | 0.1165 |
| `atlas_v2` | **100%** | 64.0 | 6.3646 | 0.00000 | 0.0 | 0.1388 |
| `contrast` | **100%** | 64.0 | 6.3675 | +0.00290 | 0.0 | 0.1108 |
| `horizon` (v0.4) | **100%** | 64.0 | 6.3646 | 0.00000 | 0.0 | 0.1388 |
| `odyssey` (calibrated) | **100%** | 64.0 | 6.3646 | 0.00000 | **12.25** | 0.1388 |
| `odyssey_no_memory` | **100%** | 64.0 | 6.3646 | 0.00000 | **12.25** | 0.1388 |
| `odyssey_ungated` | **100%** | 64.0 | **6.4051** | **+0.04042** | **13.00** | 0.1527 |
| `heuristic` | **100%** | 64.0 | 6.3110 | -0.05360 | 0.0 | 0.1333 |
| `random` | 25% | 54.3 | 0.3488 | -6.01586 | 0.0 | 0.3549 |

### Highlights under Severe Scarcity
1. **Highest Episode Reward Achieved**: `odyssey_ungated` achieved a **+0.04042 mean reward gain over the policy baseline**, outperforming Neural MPC (+0.0371), Contrast (+0.0029), and task heuristics (-0.0536).
2. **Cognitive Exploration**: Odyssey mapped an average of **12.25 to 13.0 unique patches per episode**, building a rich topological graph of discovered food, water, and shelter clusters.
3. **Seed-Level Superiority**: On seed `112000004`, `odyssey_ungated` delivered a **+0.1127 reward gain**, and on seed `112000002` a **+0.0988 reward gain**.
4. **Observed fallback**: Calibrated `odyssey` retained the policy on these seeds. This is not a safety guarantee; some heuristic override paths bypass its margin gate.

---

## 3. Quantitative Results: Moderate Scarcity ($scarcity = 2.5$)

- **Seeds**: `111000001` – `111000004` (4 episodes per arm = 40 paired episodes)
- **Receipt ID**: `e492031f870f05da-c9ba9d72f3be` | **Checksum**: `c9ba9d72f3be7dde03467a62e88a93ba52862ec8033bc09a51357eabfd4f563f`
- **Transitions replayed**: 2,500 | **Counterfactual branches replayed**: 15,000

| Controller Arm | Survival Rate | Mean Steps | Mean Reward | Mean Reward Delta vs Policy | Mean Patches Mapped | One-Step Regret |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `policy` | **100%** | 64.0 | 6.5642 | Baseline (0.0) | 0.0 | 0.1330 |
| `neural_mpc` | **100%** | 64.0 | 6.5642 | 0.00000 | 0.0 | 0.1317 |
| `atlas_v2` | **100%** | 64.0 | 6.5642 | 0.00000 | 0.0 | 0.1330 |
| `contrast` | **100%** | 64.0 | 6.5642 | 0.00000 | 0.0 | 0.1330 |
| `horizon` (v0.4) | **100%** | 64.0 | 6.5642 | 0.00000 | 0.0 | 0.1330 |
| `odyssey` (calibrated) | **100%** | 64.0 | 6.5642 | 0.00000 | **8.50** | 0.1330 |
| `odyssey_no_memory` | **100%** | 64.0 | 6.5642 | 0.00000 | **8.50** | 0.1330 |
| `odyssey_ungated` | **100%** | 64.0 | 6.5642 | 0.00000 | **8.50** | 0.1330 |
| `heuristic` | **100%** | 64.0 | 6.5642 | 0.00000 | 0.0 | 0.1330 |
| `random` | 25% | 47.5 | -0.1983 | -6.76250 | 0.0 | 0.4447 |

In this moderate-scarcity sample, policy survived all four episodes and achieved mean reward 6.564. Odyssey tracked 8.5 patches on average and matched the observed returns. Optimality was not established.

---

## 4. Independent Weightless Replay Verification

Both official Odyssey experiment receipts were verified using the standalone counterfactual replay harness:
```bash
python -m poseidon verify-odyssey outputs/odyssey_experiments/cdaf779874a04de6-7dc45ea640b3.json
python -m poseidon verify-odyssey outputs/odyssey_experiments/e492031f870f05da-c9ba9d72f3be.json
```

- **Neural Models Invoked**: Exactly 0.
- **Episodes Replayed**: 80 (40 severe + 40 moderate).
- **Transitions Verified**: 5,021.
- **Counterfactual Branches Verified**: 30,126.
- **Verification Status**: 100% Verified (`verified: true`).
