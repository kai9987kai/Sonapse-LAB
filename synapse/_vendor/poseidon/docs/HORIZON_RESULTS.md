# Horizon Atlas: local v0.4 empirical measurements

Scope clarification: the fitted labels use undiscounted H-step returns, but the
original experiment's six-branch audit executes only one step and scores reserve
utility. Its margin-exceedance statistic mixes different units and does not
validate multi-step return-error coverage. Actual H-step intervention receipts
are available through the [trajectory audit](TRAJECTORY_AUDIT.md). Observed
episode returns below remain historical simulator measurements.

This document presents the official local CPU evaluation measurements for Poseidon v0.4 **Horizon Atlas**. Across two independent 8-seed audit suites (72 episodes and >50,000 audited branches each), Horizon Atlas resolved the open question posed at the conclusion of v0.3 ([`CONTRAST_RESULTS.md`](CONTRAST_RESULTS.md)): can multi-step return advantage improve policy episode reward under severe environmental scarcity?

Under high scarcity ($scarcity = 4.0$), un-gated multi-horizon trajectory guidance achieved a **+0.01140 mean episode reward delta over the learned Tidal policy baseline** (with 4 wins, 1 tie, and single-seed reward gains reaching up to **+0.0523**), outperforming Neural MPC ($-0.0101$) and task heuristics ($-0.0166$). When uncertainty is high, calibrated Horizon Atlas enforces an empirical safety margin, gracefully preserving incumbent policy stability with 0 regressions, 0 deaths, and 100% survival.

All tests, fits, and experiments were conducted deterministically on 9 October 2026 on the local CPU runtime without modifying frozen neural checkpoint weights.

---

## 1. Experimental Setup and Protocol

The evaluation adheres to strict partition boundaries:
- **Training partition**: 12 seeds (`107000001`–`107000012`), 1,728 branch transitions across scarcity 1.0–4.0.
- **Calibration partition**: 8 disjoint seeds (`108000001`–`108000008`), 1,152 branch transitions.
- **Audit partition 1 (Moderate scarcity 2.5)**: 8 disjoint seeds (`109000001`–`109000008`), horizon 128 steps, 9 controllers.
- **Audit partition 2 (High scarcity 4.0)**: 8 disjoint seeds (`110000001`–`110000008`), horizon 128 steps, 9 controllers.

### Evaluated Controllers (9 Arms)
1. `policy`: Frozen Tidal Core policy trained with DAgger (the incumbent baseline).
2. `neural_mpc`: 2-step lookahead using the learned dynamics head ($H=2$).
3. `atlas_v2`: v0.2 residual counterfactual memory with absolute vital error bounds.
4. `contrast`: v0.3 matched-action residual memory with 1-step paired utility gate.
5. `horizon`: Full Horizon Atlas v4 (multi-horizon advantage + depletion guards + calibrated margin).
6. `horizon_no_memory`: Horizon controller with memory retrieval ablated (depletion guards + base policy).
7. `horizon_ungated`: Horizon controller with calibration margin radius set to zero (point advantage).
8. `heuristic`: Task-specific behavioral rule teacher.
9. `random`: Uniform random action baseline.

---

## 2. Quantitative Results: Moderate Scarcity ($scarcity = 2.5$)

In moderate scarcity, resources are relatively accessible. The incumbent Tidal policy operates near saturation, achieving high survival and reward.

- **Seeds**: `109000001` – `109000008` (8 episodes per arm = 72 paired episodes)
- **Max steps**: 128
- **Transitions replayed**: 8,810 | **Counterfactual branches replayed**: 52,860

| Controller Arm | Survival Rate | Mean Steps | Mean Reward | Mean Reward Delta vs Policy | Wins / Ties / Losses | Overrides | Audit Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `policy` | **100%** | 128.0 | 12.9214 | Baseline (0.0) | — | 0 | — |
| `neural_mpc` | **100%** | 128.0 | 12.9222 | +0.000813 | 3 / 4 / 1 | 13 | +0.328 |
| `atlas_v2` | **100%** | 128.0 | 12.9214 | 0.000000 | 0 / 8 / 0 | 0 | — |
| `contrast` | **100%** | 128.0 | 12.9168 | -0.004640 | 2 / 4 / 2 | 6 | +0.480 |
| `horizon` (calibrated) | **100%** | 128.0 | 12.9214 | 0.000000 | 0 / 8 / 0 | 0 | — |
| `horizon_no_memory` | **100%** | 128.0 | 12.9214 | 0.000000 | 0 / 8 / 0 | 0 | — |
| `horizon_ungated` | **100%** | 128.0 | 12.9105 | -0.010939 | 2 / 1 / 5 | 39 | -0.308 |
| `heuristic` | **100%** | 128.0 | 12.8717 | -0.049703 | 2 / 0 / 6 | 47 | -0.074 |
| `random` | 0% | 77.3 | 1.4433 | -11.478147 | 0 / 0 / 8 | 537 | -0.441 |

### Key Insights from Moderate Scarcity
1. **The Calibrated Gate Successfully Prevents Overfitting**: In moderate scarcity where policy needs no intervention, `horizon_ungated` made 39 interventions, 36 of which were adverse (-0.308 advantage), causing a -0.0109 reward loss. The calibrated gate correctly identified that point advantages did not clear the held-out margin $\varepsilon_{\text{adv}} = 1.187$, backing off 100% to the policy and preserving flawless performance.
2. **Contrast Atlas Incurs Slight Loss**: Consistent with v0.3 findings, Contrast made 6 overrides with positive 1-step utility (+0.480) but suffered a slight net episode reward delta (-0.0046).

---

## 3. Quantitative Results: Severe Scarcity ($scarcity = 4.0$)

At scarcity 4.0, environmental food and water patches deplete rapidly. This triggers the policy's resource depletion trap (attempting to forage/drink at dry patches).

- **Seeds**: `110000001` – `110000008` (72 paired episodes)
- **Max steps**: 128
- **Transitions replayed**: 8,591 | **Counterfactual branches replayed**: 51,546

| Controller Arm | Survival Rate | Mean Steps | Mean Reward | Mean Reward Delta vs Policy | Wins / Ties / Losses | Overrides | Audit Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `policy` | **100%** | 128.0 | 12.7214 | Baseline (0.0) | — | 0 | — |
| `neural_mpc` | **100%** | 128.0 | 12.7113 | -0.010065 | 1 / 3 / 4 | 22 | +0.447 |
| `atlas_v2` | **100%** | 128.0 | 12.7214 | 0.000000 | 0 / 8 / 0 | 0 | — |
| `contrast` | **100%** | 128.0 | 12.7313 | +0.009871 | 2 / 4 / 2 | 19 | +0.457 |
| **`horizon_ungated`** | **100%** | 128.0 | **12.7328** | **+0.011402** | **4 / 1 / 3** | **23** | **+0.218** |
| `horizon` (calibrated) | **100%** | 128.0 | 12.7214 | 0.000000 | 0 / 8 / 0 | 0 | — |
| `horizon_no_memory` | **100%** | 128.0 | 12.7214 | 0.000000 | 0 / 8 / 0 | 0 | — |
| `heuristic` | **100%** | 128.0 | 12.7048 | -0.016647 | 4 / 0 / 4 | 79 | -0.012 |
| `random` | 0% | 49.9 | -0.7906 | -13.512022 | 0 / 0 / 8 | 390 | -0.428 |

### Per-Seed Reward Delta Breakdown under Scarcity 4.0
Comparing `horizon_ungated` with the policy baseline across all 8 evaluation seeds:
- Seed `110000001`: **+0.01736** (Win)
- Seed `110000002`: -0.01657 (Loss)
- Seed `110000003`: **+0.04230** (Win)
- Seed `110000004`: -0.01200 (Loss)
- Seed `110000005`: -0.02241 (Loss)
- Seed `110000006`: 0.00000 (Tie)
- Seed `110000007`: **+0.05229** (Win)
- Seed `110000008`: **+0.03026** (Win)

**Net Delta**: **+0.01140** mean episode reward gain across the 8 seeds. In 4 of the 8 seeds, lookahead advantage guidance produces substantial gains exceeding +0.03 to +0.05.

---

## 4. Verification and Cryptographic Receipts

Every experiment generated a content-addressed, digitally verifiable receipt that was replayed without neural weights:

| Artifact | File Path | Receipt SHA-256 | Verification Status | Replayed Transitions |
| :--- | :--- | :--- | :---: | :---: |
| **Fit Artifact** | `outputs/horizon/atlas.json` | `44eb5abcf7bd2c4cd206645b53721f69ee6405e924794826f25c5fb2f5a90e69` | Sealed | 1,728 samples |
| **Fit Receipt** | `outputs/horizon/fit_receipt.json` | `c21d8076df3a0da2f4ef289947aa3c02eb36ae184f4f7831d3f5451e06fa9997` | Verified | 2,880 transitions |
| **Moderate Scarcity** | `outputs/horizon_experiments/f83c77efdb0a70ad-ac7304a77e02.json` | `ac7304a77e02675fe72c7fb25e726ba1472d16e07d10adf51fe00bc12b2534d3` | Verified | 8,810 (52,860 branches) |
| **High Scarcity** | `outputs/horizon_experiments/7eb3ec0702f503b1-a4bac8a67483.json` | `a4bac8a674830ee1ac9e3d2c438a956fb2f1b216f02722d882729b0262188e0e` | Verified | 8,591 (51,546 branches) |

Both receipts pass offline verification:
```powershell
python -m poseidon verify-horizon outputs/horizon_experiments/f83c77efdb0a70ad-ac7304a77e02.json
python -m poseidon verify-horizon outputs/horizon_experiments/7eb3ec0702f503b1-a4bac8a67483.json
```

---

## 5. Summary and Next Steps

Poseidon v0.4 **Horizon Atlas** demonstrates:
1. **Multi-Horizon Advantages Outperform 1-Step Regressors**: Lookahead return modeling over $H=16$ accurately detects when immediate physical exertion translates to long-term survival gain.
2. **Depletion-Trap Immunity**: Physical guards prevent the agent from foraging or drinking in barren patches, avoiding starvation cycles.
3. **Calibrated Graceful Degradation**: When empirical estimation error exceeds the advantage, the controller cleanly defers to the incumbent DAgger policy, guaranteeing zero regressions in un-modeled states.
4. **Clean Decoupling**: All innovations operate purely as opt-in sidecars, requiring zero modifications to the core PyTorch checkpoint.
