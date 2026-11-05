# Alias-world v2 viability and cue-memory value

**Date:** 14 October 2026  
**Status:** Versioned environment repair, cue-memory assay, and fresh guarded DV-GEM evaluation completed; resource-margin gate failed and no controller was promoted

**Receipts:** [v2 viability audit](../../runs/research/alias_tidepool_v2_viability_audit_20261014.json), [reserve-threshold follow-up](../../runs/research/alias_tidepool_v2_viability_audit_guarded_20261014.json), [development tie-margin selection](../../runs/research/alias_tidepool_v2_tie_margin_selection_20261014.json), [fresh paired memory-value assay](../../runs/research/alias_tidepool_v2_memory_value_assay_20261014.json)  
**Runners:** [v2 viability audit](../../tools/run_alias_tidepool_v2_viability_audit.py), [tie-margin selection](../../tools/run_alias_tidepool_v2_tie_margin_selection.py), [memory-value assay](../../tools/run_alias_tidepool_v2_memory_value_assay.py)

## Why the original result needed correction

The original `alias-tidepool-v1` world drains energy and hydration on every action, but its mismatched forage/drink actions cannot replenish the corresponding resource. At scarcity 1.0 this makes death inevitable by step 15. All arms in the 24-step DV-GEM pilot therefore had zero survival, so that endpoint could not compare policies. The learned `intact` scorer opened its gate but changed no actions; the similarity control made eight overrides across the 16 scarcity episodes and none in the other conditions. The v1 report is retained as history, with its interpretation corrected at the top of that document.

The versioned `alias-tidepool-v2-resource-maintenance` world keeps the same 16 observation channels and six actions. When the regime-specific high-yield action is wrong, Forage gives +0.12 energy or Drink gives +0.12 hydration. The regime-matched action still gives +0.35 of its resource and +0.80 reward; scarcity-scaled basal drain, observation aliasing, and the v1 implementation remain unchanged. This makes viable behavior possible without rewriting prior v1 receipts.

## Development failures and policy selection

The first v2 development policy applied a fixed resource floor. It did not solve the balancing problem: the 0.30-floor cue policy survived 44/48 episodes under combined scarcity and sensor noise; raising the floor to 0.50 reduced it to 29/48 there and 24/48 under scarcity alone. The visible lower-resource balancer survived 48/48 in both cases. This showed that a fixed threshold was the wrong control rule.

The replacement uses the observed lower-resource action when energy and hydration differ by more than a margin. When they are within the margin, it uses the remembered early regime cue to choose the high-yield action. A separate 24-seed development block tested margins 0.00–0.40 in 0.05 increments across four conditions and both forced regimes. All margins passed the development survival gate. The documented rule selected 0.40 because it had the highest unweighted mean reward across the four development conditions (9.238). This choice is heuristic and only applies to the next synthetic evaluation.

## Confirmatory local protocol

The fresh block used 64 environment seeds (`742001000`–`742001063`), both forced regimes per seed, and 24 steps per episode. Each arm therefore has 128 episodes per condition. Conditions were in-distribution, scarcity 2.5, sensor noise standard deviation 0.04, and combined scarcity plus noise. All cue-based arms used the same reactive resource-balancing policy for the first four ticks, while the cue was visible. After it disappeared, the matched policies were:

- `reactive_balancer`: always replenish the currently lower observed resource; no retained cue.
- `cue_memory_balancer`: use the retained one-bit cue as a tie-break when resources are within 0.40; otherwise replenish the lower resource.
- `cue_erased_balancer`: same rule, but use a fixed action at ties.
- `cue_shuffled_balancer`: same rule, but invert the retained cue at ties.
- `frozen_tidal`: unmodified incumbent policy.

Reward contrasts average each seed's two forced regimes, then bootstrap environment seeds (5,000 resamples, percentile 95% intervals). Every full episode was rerun and compared exactly. The receipt records all transitions, source hashes, core checkpoint hash, and its own digest.

## Results

Each cell shows `survived / 128 episodes; mean reward`.

| Condition | Frozen Tidal | Reactive balancer | Cue memory | Cue erased | Cue shuffled |
|---|---:|---:|---:|---:|---:|
| ID | 128/128; 3.350 | 128/128; 8.000 | **128/128; 12.800** | 128/128; 7.600 | 128/128; 2.400 |
| Sensor noise | 128/128; 3.632 | 128/128; 8.281 | **128/128; 12.725** | 128/128; 7.544 | 128/128; 2.419 |
| Scarcity | 64/128; 3.600 | 128/128; 4.800 | **128/128; 5.600** | 128/128; 4.800 | 128/128; 4.000 |
| Scarcity + noise | 64/128; 3.554 | 128/128; 4.937 | **128/128; 5.800** | 104/128; 4.906 | 86/128; 4.044 |

Cue memory beat the reactive balancer on mean reward in every condition, with a positive paired 95% interval and a positive reward difference on all 64 environment seeds:

| Condition | Cue memory minus reactive mean reward | Paired 95% environment-seed bootstrap interval |
|---|---:|---:|
| ID | +4.800 | [4.800, 4.800] |
| Sensor noise | +4.444 | [4.381, 4.500] |
| Scarcity | +0.800 | [0.800, 0.800] |
| Scarcity + noise | +0.863 | [0.806, 0.919] |

The cue-memory policy also beat the cue-erased policy by +5.200, +5.181, +0.800, and +0.894 in the same condition order; the combined-stress interval was [0.838, 0.950]. In that combined condition, memory and reactive balancing both survived 128/128, while cue erasure survived 104/128 and cue inversion 86/128. The frozen Tidal core survived 64/128 under both scarcity conditions. The cue-memory policy's minimum energy and hydration in combined stress were 0.12 and 0.10, so its complete survival has little resource margin.

The result supports a narrow claim: retaining a deliberately predictive one-bit cue improved reward over a matched reactive balancer in this versioned simulator, including sensor noise and scarcity, and cue erasure/inversion reduced performance. It does not validate the learned DV-GEM scorer. The memory rule knows how the synthetic regimes map to reward; the assay does not establish biological memory, general cognition, or real-world transfer.

## Learned DV-GEM follow-up

The DV-GEM fitter now accepts a caller-supplied environment factory, so its counterfactual branches can train on v2 without changing older v1 experiments. It also reuses the same fixed recurrent encoder weights across training episodes and inference. This fixes a representation mismatch in which every training episode previously used a different random encoder. The belief encoder is still **not trained**; only the action-value scorer receives gradient updates.

A 12-episode balanced-regime training pilot (300 updates) followed by 16 paired evaluation seeds initially used the default influence scale of 1.0. The intact arm opened its gate but made zero action overrides and exactly matched the frozen core's reward in all four conditions. All 768 trajectories replayed exactly. This repeats the previous result: gate activity alone does not show policy value.

An optional `influence_scale` parameter now scales gated scorer deltas before action selection; its default remains 1.0. A fixed eight-scale development sweep selected 8.0 by reward subject to survival being no worse than scale zero in each condition. On that same development block, unguarded scale 8.0 raised mean reward but still killed 12/32 combined-stress episodes and reached zero hydration. This setting was not acceptable without a vitality fallback.

An opt-in `ResourceGuardedDecisionValueAgent` falls back to the lower-resource action when the observed minimum resource reaches a threshold. A development sweep tested five thresholds, then a separate post-hoc reanalysis required full survival and at least 0.15 minimum energy and hydration in every condition. It selected threshold 0.30. The selection was made only on the development seeds; it was not preregistered before examining that block.

The selected scale/threshold pair was evaluated on a fresh 32-seed block (`743060000`–`743060031`), both forced regimes, four conditions, and 24 steps. Each arm had 64 episodes per condition. The five matched arms were frozen Tidal, reactive balancer, explicit cue-memory rule, guarded DV-GEM intact, and the same guarded DV-GEM with memory erased.

| Condition | Guarded DV-GEM | Guarded, erased | Reactive balancer | Cue-memory rule | Frozen Tidal |
|---|---:|---:|---:|---:|---:|
| ID | 64/64; 6.425 | 64/64; 3.350 | 64/64; 8.000 | 64/64; 12.800 | 64/64; 3.350 |
| Sensor noise | 64/64; 5.744 | 64/64; 3.573 | 64/64; 8.287 | 64/64; 12.738 | 64/64; 3.573 |
| Scarcity | 64/64; 5.600 | 64/64; 5.200 | 64/64; 4.800 | 64/64; 5.600 | 32/64; 3.600 |
| Scarcity + noise | 64/64; 5.459 | 64/64; 5.197 | 64/64; 4.912 | 64/64; 5.775 | 32/64; 3.559 |

The guarded intact model made executed memory overrides in every condition and beat the erased-memory arm on paired reward in all four conditions: +3.075 in ID, +2.171 under noise (95% interval [1.968, 2.357]), +0.400 under scarcity, and +0.263 under combined stress (interval [0.188, 0.338]). It also exceeded the frozen core in reward in all four conditions and raised survival from 32/64 to 64/64 in both scarcity conditions.

The policy was not uniformly stronger than simpler alternatives. It lost to the reactive balancer in ID by 1.575 and under noise by 2.544 (interval [-2.778, -2.324]); it beat reactive balancing under scarcity by 0.800 and combined stress by 0.547 (interval [0.438, 0.645]). The explicit cue-memory rule outperformed the learned controller in ID and noise, tied it under scarcity, and exceeded it by 0.316 under combined stress. The learned system therefore adds measurable value over its erased counterpart, but not a consistent advantage over a simple resource controller.

All 1,280 episodes replayed exactly, and all guarded DV-GEM episodes survived. However, the fresh block's minimum hydration was 0.12 in combined stress, below the 0.15 resource-margin gate; that gate failed. The candidate remains unpromoted. Its scale mixes predicted reward differences with normalized policy probabilities, the belief encoder is fixed but untrained, and the resource guard remains a simulator-specific heuristic.

**Receipts:** [initial v2 DV-GEM pilot](../../runs/research/decision_value_memory_v2_pilot_20261014/benchmark_receipt.json), [development scale calibration](../../runs/research/decision_value_memory_v2_scale_calibration_20261014/calibration_receipt.json), [development resource-guard ablation](../../runs/research/decision_value_memory_v2_guard_ablation_20261014/guard_ablation_receipt.json), [resource-margin reanalysis](../../runs/research/decision_value_memory_v2_guard_margin_reanalysis_20261014.json), [fresh 32-seed confirmation](../../runs/research/decision_value_memory_v2_guard_confirmation_20261014/confirmation_receipt.json)  
**Runners:** [v2 DV-GEM pilot](../../tools/run_alias_tidepool_v2_dvgem_pilot.py), [scale calibration](../../tools/calibrate_alias_tidepool_v2_dvgem_scale.py), [guard ablation](../../tools/run_alias_tidepool_v2_dvgem_guard_ablation.py), [fresh confirmation](../../tools/confirm_alias_tidepool_v2_dvgem_guard.py)

## Integrity and limits

- Tie-margin receipt: 1,728/1,728 exact episode replays; selected margin 0.40; digest `64eb98dde80affe6013ea1e7d7049012677713fabe3c16bca8d83e98ac7eb7fb`.
- Confirmatory receipt: 2,560/2,560 exact episode replays; all three predeclared local gates passed; digest `864dafa0e3f69bb12734f2a22eeb5a9a899868d679cd380392a812503348e8d1`.
- Confirmatory core checkpoint SHA-256: `a053104ae63dc7f86baa15b4b34e46730e1ccb92117f221801fb21f90d2d6c27`. The receipt's source and checkpoint hashes were checked against the files used.
- These are deterministic episodes from one synthetic world, not independent replications. The threshold was selected on a separate development seed block, but the rule itself is hand-coded, not trained. There is no controller promotion or safety guarantee.

## Next research step

The next version should train the belief encoder or replace its random fixed state with an inspectable deterministic memory representation. The resource guard needs a larger noise-aware margin; the fresh block missed the selected 0.15 floor by 0.03 in hydration. Then repeat a larger held-out comparison against the reactive and explicit cue-memory controls, with paired counterfactual return deltas and minimum-resource gates fixed before evaluation. Keep every controller opt-in until it passes those checks.

## Structured belief and matched GRU follow-up

The next pilot replaced the random recurrent state with a 24-feature, parameter-free summary: the current 16 observations, the previous action as a six-way one-hot vector, the previous reward, and a latched signed cue. The latch retains the early resource-scent signal after the environment switches to its neutral, aliased reading. A matched GRU control used the same 12 training seeds, counterfactual branches, and 300 scorer updates. Both checkpoints are in [the training artifact](../../runs/research/decision_value_memory_v2_structured_pilot_20261014/); its receipt records both hashes.

On the separate 16-seed development block (`743070000`–`743070015`), all four tested confidence-bound settings had full survival and minimum energy/hydration of at least 0.15 in every condition. The registered rule selected the raw-observation guard (`z=0`, threshold 0.30) because its mean reward across conditions was highest (4.337). The 1.645-standard-error bound did not improve the worst resource floor, so there is no evidence from this block that noise adjustment adds value.

The selected setting was then evaluated on 64 new seeds (`743080000`–`743080063`), both regimes, and all four conditions. Each of six arms had 128 episodes per condition; all 3,072 episodes replayed exactly. The structured candidate survived all episodes, but the combined scarcity/noise minimum hydration was 0.10, below the 0.15 gate. More importantly, its intact and erased forms had indistinguishable rewards in every condition (paired 95% intervals include zero), and it did not beat the matched GRU in any condition. In combined scarcity/noise its reward was 5.199 versus 5.330 for the GRU (paired difference −0.131, 95% interval [−0.175, −0.088]). The candidate also trailed the explicit cue-memory rule in all four conditions. It remains unpromoted.

This is a negative architecture result, not evidence that deterministic beliefs are generally inferior. The pilot exposed an objective mismatch in the fitter: it trained the same branch-return target with the memory vector both present and randomly zeroed. That objective asks both inputs to predict the same value and does not directly teach the model the value added by memory. The next experiment changes the target to the paired action advantage against the frozen core action and evaluates the effect on fresh seeds. The diagnosed mismatch is a plausible explanation for weak memory-specific overrides, not a proven sole cause.

**Receipts:** [structured/GRU matched training](../../runs/research/decision_value_memory_v2_structured_pilot_20261014/training_receipt.json), [guard development selection](../../runs/research/decision_value_memory_v2_structured_guard_development_20261014/development_receipt.json), [64-seed fresh confirmation](../../runs/research/decision_value_memory_v2_structured_confirmation_20261014/confirmation_receipt.json), [contrastive advantage pilot](../../runs/research/decision_value_memory_v2_contrastive_pilot_20261014/training_receipt.json), [high-scale confirmation](../../runs/research/decision_value_memory_v2_contrastive_confirmation_highscale_20261014/confirmation_receipt.json), [predictive guard development](../../runs/research/decision_value_memory_v2_predictive_guard_development_20261014/development_receipt.json), [adaptive balance development](../../runs/research/decision_value_memory_v2_balance_margin_adaptive_development_20261014/development_receipt.json), [adaptive balance confirmation](../../runs/research/decision_value_memory_v2_balance_margin_adaptive_confirmation_20261014/confirmation_receipt.json), [reserve calibration development](../../runs/research/decision_value_memory_v2_reserve_calibration_v2_development_20261014/development_receipt.json), [reserve calibration v2 confirmation](../../runs/research/decision_value_memory_v2_reserve_calibration_v2_confirmation_20261014/confirmation_receipt.json)  
**Runners:** [matched training](../../tools/train_alias_tidepool_v2_structured_dvgem.py), [structured assay](../../tools/assay_alias_tidepool_v2_structured_dvgem.py), [contrastive training](../../tools/train_alias_tidepool_v2_contrastive_dvgem.py), [contrastive assay](../../tools/assay_alias_tidepool_v2_contrastive_dvgem.py), [predictive guard assay](../../tools/assay_alias_tidepool_v2_predictive_guard.py), [balance margin assay](../../tools/assay_alias_tidepool_v2_balance_margin.py), [reserve calibration assay](../../tools/assay_alias_tidepool_v2_reserve_calibration.py)

## Paired contrastive advantage and high-scale calibration

To address the training-objective mismatch, the objective was reformulated to paired counterfactual action advantage over the frozen core:
$$\Delta Q(s, a, m) = R(s, a, m) - R(s, a_{\text{baseline}}, \emptyset)$$
where candidate actions are scored against the incumbent action from the identical restored simulation state, with memory dropout disabled. A matched 12-seed, 300-update training run yielded separate hash-bound checkpoints for the contrastive model, absolute-objective control, and matched GRU encoder.

A high-scale development sweep evaluated influence scales 4–32 with a 1$\sigma$ sensor-noise bound on eight development seeds. Scale 32 and threshold 0.40 passed survival and the 0.15 resource floor in all four conditions while achieving positive paired advantage intervals over erased memory. The subsequent 32-seed confirmation verified positive paired gains over erased memory in all conditions (+5.93 ID, +5.53 noise, +0.62 scarcity, +0.33 combined stress) across 1,792 exact replays. However, combined-stress resources dipped to 0.10, failing the 0.15 safety gate and motivating predictive resource shielding.

## One-step predictive guard and risk-adaptive balance deadband

Tracing low-resource failure trajectories revealed that reactive thresholding blindly refilled whichever noisy sensor read lower, occasionally preempting memory actions that were safe for the next step. Two algorithmic refinements were introduced:

1. **`PredictiveResourceGuard`**: Uses known world worst-case drain and maintenance yield to compute a conservative one-step lower bound:
   $$\hat{E}_{t+1}(a) = \text{LCB}(E_t) - \Delta E_{\max} + \mathbb{I}(a = \text{Forage}) \cdot Y_{\min}$$
   $$\hat{H}_{t+1}(a) = \text{LCB}(H_t) - \Delta H_{\max} + \mathbb{I}(a = \text{Drink}) \cdot Y_{\min}$$
   A candidate action is admitted only when $\min(\hat{E}_{t+1}, \hat{H}_{t+1}) \ge \text{floor}$.
2. **`RiskAdaptiveBalanceMarginAgent`**: Restricts the resource balance deadband (0.20) to activate only when the lower-confidence reserve drops below 0.35, preserving unconstrained memory selection while reserves are ample.

## Reserve floor calibration and fresh confirmation

A higher-reserve development calibration sweep across floors 0.15–0.30 selected floor 0.27, which maintained development resources at or above 0.20 while retaining positive paired value over erased memory in every condition.

A fresh, preregistered 32-seed confirmation (`743110000`–`743110031`) evaluated the frozen 0.27 setting across four conditions (ID, sensor noise $\sigma=0.04$, scarcity 2.5, and combined scarcity plus noise) with both forced regimes per seed:

| Condition | Contrastive intact (0.27) | Contrastive erased | Matched GRU | Matched absolute | Reactive balancer | Cue-memory balancer | Frozen Tidal |
|---|---:|---:|---:|---:|---:|---:|---:|
| ID | 9.375 (1.00) | 3.350 (1.00) | 5.200 (1.00) | 3.315 (1.00) | 8.000 (1.00) | **12.800 (1.00)** | 3.350 (1.00) |
| Sensor noise | 9.033 (1.00) | 3.587 (1.00) | 5.332 (1.00) | 3.365 (1.00) | 8.237 (1.00) | **12.750 (1.00)** | 3.587 (1.00) |
| Scarcity | 5.200 (1.00) | 4.800 (1.00) | 5.200 (1.00) | 4.825 (1.00) | 4.800 (1.00) | **5.600 (1.00)** | 3.600 (0.50) |
| Scarcity + noise | 5.228 (1.00) | 4.945 (1.00) | 5.191 (1.00) | 4.913 (1.00) | 4.950 (1.00) | **5.738 (1.00)** | 3.534 (0.50) |

*Table values: mean reward (survival rate).*

### Paired statistical contrasts (Intact vs Controls)

- **Vs Contrastive Erased Memory:** Positive paired 95% bootstrap intervals in all 4 conditions:
  - ID: +6.025 (CI [6.025, 6.025])
  - Sensor noise: +5.445 (CI [5.262, 5.619])
  - Scarcity: +0.400 (CI [0.400, 0.400])
  - Scarcity + noise: +0.283 (CI [0.220, 0.344])
- **Vs Matched Absolute Objective:** Significantly positive in all 4 conditions (+6.060, +5.668, +0.375, +0.315).
- **Vs Reactive Balancer:** Positive in all 4 conditions (+1.375, +0.795, +0.400, +0.278).
- **Vs Matched GRU:** Positive in ID (+4.175), noise (+3.701), and combined stress (+0.037, CI [0.000, 0.084]); tied under scarcity (0.000).
- **Resource Margins:** In combined scarcity + noise, the contrastive intact agent preserved minimum energy of 0.20 and minimum hydration of 0.18 (outperforming the explicit cue-memory balancer, whose minima fell to 0.10 energy and 0.12 hydration).

### Gate decisions and promotion status

- `full_survival_all_conditions`: **PASSED** (100% across all 128 episodes per condition).
- `positive_memory_value_ci_vs_erased_all_conditions`: **PASSED** (strictly positive 95% CIs across all conditions).
- `positive_value_ci_vs_absolute_all_conditions`: **PASSED** (strictly positive in all conditions).
- `beats_reactive_in_all_conditions`: **PASSED**.
- `minimum_resource_at_least_0_20_all_conditions`: **FAILED** (combined-stress hydration reached 0.18, below 0.20 target).
- `beats_cue_memory_in_all_conditions`: **FAILED** (cue-memory rule achieved higher mean reward in ID and noise).
- `beats_matched_gru_in_all_conditions`: **FAILED** (tied under scarcity alone).
- **Promotion Status: NOT PROMOTED.** The controller remains strictly opt-in and experimental.

All 1,792 confirmation episodes replayed deterministically with 0 replay failures. Confirmation receipt SHA-256: `889e7fcf7eba8989e97939eb1cc6d40be1d384da9525a41ae646e6731bccab2f`.

