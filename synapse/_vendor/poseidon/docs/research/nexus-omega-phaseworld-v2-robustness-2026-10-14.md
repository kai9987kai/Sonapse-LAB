# NEXUS-Ω phaseworld viability, stress, and schedule-shift study

**Date:** 14 October 2026  
**Status:** Synthetic 128-step stress and schedule-shift studies completed; exploratory, no controller promotion

**Receipts:** [32-step refinement](../../runs/nexus_omega/phaseworld_v2_viability_assay_refined_20261014.json), [64-step robustness replication](../../runs/nexus_omega/phaseworld_v2_robustness_h64_20261014.json), [128-step robustness extension](../../runs/nexus_omega/phaseworld_v2_robustness_h128_20261014.json), [five-tick stress run](../../runs/nexus_omega/phaseworld_v2_storm5_h128_20261014.json), [five-tick source-bound repeat](../../runs/nexus_omega/phaseworld_v2_storm5_h128_sourcebound_20261014.json), [eight-tick adaptive stress pilot](../../runs/nexus_omega/phaseworld_v2_storm8_h128_adaptive_20261014.json), [eight-tick phase-learning confirmation](../../runs/nexus_omega/phaseworld_v2_storm8_h128_phaselearn_20261014.json), [hidden interval-shift baseline](../../runs/nexus_omega/phaseworld_v3_interval_shift_h128_20261014.json), [two-pulse interval-shift comparison](../../runs/nexus_omega/phaseworld_v3_interval_shift_h128_twopulse_20261014.json), [severe-shift guard pilot](../../runs/nexus_omega/phaseworld_v3_severe_shift_confirmed_period_pilot_20261014.json), [boundary-latch confirmation](../../runs/nexus_omega/phaseworld_v3_severe_shift_boundary_latch_h128_20261014.json), [64-seed reserve-aware schedule-shift confirmation](../../runs/nexus_omega/phaseworld_v3_severe_shift_resource_guard_h128_20261014.json), [paired guard-threshold ablation with full-episode replay hashes](../../runs/nexus_omega/phaseworld_v3_guard_threshold_ablation_20261014.json)
**Runners:** [v2 viability assay](../../tools/run_nexus_phaseworld_v2_viability_assay.py), [v3 schedule-shift assay](../../tools/run_nexus_phaseworld_v3_drift_assay.py), [paired guard-threshold ablation](../../tools/run_nexus_phaseworld_v3_guard_threshold_ablation.py)

## Question

The original `POMDPPhaseWorld` drains hydration on every action and offers no replenishment action. A three-tick pulse-memory controller therefore still dies from starvation, making survival an invalid outcome for comparing controllers. The versioned `nexus-phaseworld-v2` environment preserves the eight observation channels and four actions, while adding +0.12 hydration to Rest before the unchanged -0.05 basal drain. The original environment and its receipts remain untouched.

The first v2 run exposed two model/controller errors: the inherited prior predicted health recovery from storm Shelter even though the environment provides none, and the safety fallback always chose Shelter even when Rest had the higher lower-confidence score and replenished hydration. After those were corrected, a 32-step run found that all aligned agents survived, but an initially hidden storm could leave health at 0.2 and energy near zero. A 128-step run extended nominal-horizon survival, then a five-tick storm stress pilot showed that fixed one- and three-tick surprise persistence still failed. The current v2 refinement estimates storm duration relative to the observed pulse and permits tightly screened resupply when the model believes the storm has cleared.

## V2 changes

- The v2 factored prior removes the inherited +0.05 Shelter health gain during storms and models Rest's +0.12 hydration recovery.
- The safety planner keeps its original Shelter fallback by default. The v2 agent opts into a pair of fallback actions (Shelter, Rest) and selects the action with the higher model lower-confidence score.
- The v2 agent starts in Rest until it observes a pulse anchor. A shorter observed pulse interval immediately lowers its provisional period estimate; two near-matching observations in either direction confirm the period before it can tighten the storm-duration window. The inferred storm duration is capped at `confirmed interval - 1` (up to eight ticks).
- A health residual at or below -0.25 extends the pulse-relative storm estimate by five ticks. A shock without a pulse latches Rest until a new pulse anchors phase.
- During a cue or inferred storm, a heuristic guard selects Rest below hydration 0.98 or energy 0.75, and Shelter otherwise. Calm Forage recovery remains screened by low energy, adequate hydration, no cue/memory/regime storm belief, and predicted vital-state recovery. Guard overrides and recovery admissions are logged separately.
- If the expected next pulse is near or overdue but absent, a phase-boundary latch holds Rest until a new pulse reanchors the cycle, covering schedule shifts that begin late in an existing cycle.
- The v1 profile does not use duration inference or calm-recovery admission.

These are explicit structured heuristics. The neural residual ensemble remains randomly initialized; this is not a trained world model. The lower-confidence score is an uncalibrated heuristic and carries no conformal or safety guarantee.

The design follows portfolio lessons without importing upstream code: prediction-error feedback as a control signal from [GhostInTheMachine](https://github.com/kai9987kai/GhostInTheMachine), matched deterministic interventions from [Causeway](https://github.com/kai9987kai/Causeway) and [MOLT](https://github.com/kai9987kai/MOLT), and explicit separation of evidence admission from promotion as in [TESSERA](https://github.com/kai9987kai/TESSERA).

## Protocol

The original v1/v2 blocks used 64 environment seeds, two fixed agent initializations, full action/reward/observation/vital traces, and exact deterministic replay. The interval-shift assays use the same 64 environment seeds and two fixed agent initializations in both estimator versions. Controls include the v1 structured prior, fixed-duration trace policy, reactive cue policy, and privileged oracle. The trace is a hand-coded control; the oracle is a viability check, not a learned policy or baseline.

| Run | Environment seeds | Agent initializations | Horizon | Replay |
|---|---|---|---:|---:|
| Initial v2 refinement | 99201–99264 | 55101, 55102 | 32 | 448/448 |
| One-step shock update replication | 99301–99364 | 55201, 55202 | 64 | 448/448 |
| One-step shock update extension | 99401–99464 | 55301, 55302 | 128 | 448/448 |
| Pulse-age duration stress | 99701–99764 | 55601, 55602 | 128 | 576/576 |
| Eight-tick phase-learning confirmation | 100501–100564 | 56201, 56202 | 128 | 576/576 |
| Hidden 8-to-10 interval shift, original estimator | 100801–100864 | 56501, 56502 | 128 | 448/448 |
| Same shift, two-pulse estimator | 100801–100864 | 56501, 56502 | 128 | 448/448 |
| Severe 10x6-to-8x7 shift with boundary latch | 101201–101264 | 56801, 56802 | 128 | 448/448 |

Blocks use fresh seed ranges except the source-bound repeat and the paired estimator comparison. The interval-shift pair reran the same 64 environment seeds and two agent initializations with both estimator versions. Earlier receipts retain hashes for their source revisions. Each receipt binds its agent, models, environment, and runner and records exact replay.

## Results

| h64 arm | Survival | Mean reward | Mean terminal health | Mean terminal energy | Mean terminal hydration |
|---|---:|---:|---:|---:|---:|
| Legacy v1 trace-memory on v1 | 0/64 | 3.892 | 0.000 | 0.950 | -0.170 |
| V2 reactive cue-only | 0/64 | -1.131 | 0.000 | 0.893 | 0.728 |
| V2 hand-coded pulse trace | 64/64 | 14.009 | 0.756 | 0.950 | 0.881 |
| V2 NEXUS, v1 prior, init 55201 | 0/64 | 5.384 | 0.000 | 0.499 | -0.077 |
| V2 NEXUS, v1 prior, init 55202 | 0/64 | 5.384 | 0.000 | 0.499 | -0.077 |
| V2 NEXUS, aligned prior, init 55201 | 64/64 | 15.858 | 0.788 | 0.905 | 0.457 |
| V2 NEXUS, aligned prior, init 55202 | 64/64 | 15.940 | 0.788 | 0.913 | 0.421 |

At h128, the same aligned profile again survived every episode at both fresh initializations. Its mean reward was 30.881 (init 55301) and 30.773 (55302); mean terminal health was 0.788 for both, terminal energy 0.883/0.895, and terminal hydration 0.416/0.459. Across all h128 episodes, minimum health was 0.60, minimum energy 0.75, and minimum hydration 0.24/0.30. The per-episode hydration change averaged -0.024 and +0.027 across steps 65–128, after a larger first-half drawdown of -0.560 and -0.568. The shock update fired in 34 of 64 environment seeds at each initialization. The v1-prior comparison and cue-only control survived 0/64, while the hand-coded pulse-memory control survived 64/64; all 448 traces replayed exactly. Paired v2-minus-v1 reward changes were +25.655 (95% interval [24.988, 26.368]) and +25.548 ([24.925, 26.222]). This supports a stable resource trajectory over the measured second half, not indefinite viability.

The v2 agent completed all 64 steps in every episode. Across both initializations, the lowest observed health was 0.60, the lowest energy was 0.75, and the lowest hydration was 0.26–0.32. The unexpected-health update activated in 34 of 64 environment seeds for each initialization; the two initializations agreed on which episodes triggered it. The paired v2-minus-v1 reward changes were +10.474 (95% environment-seed bootstrap interval [9.837, 11.164]) and +10.556 ([9.930, 11.225]). Each initialization had 64/64 pairs in which only the aligned v2 prior survived. These are two fixed initializations on one synthetic environment, not independent model-family replications.

The earlier h32 refinement, before the shock update, also achieved 64/64 survival for each aligned initialization. It revealed why horizon and resource margins matter: its minimum health fell to 0.20, one initialization reached energy 0.05, and the other reached 0.28 minimum hydration. The shock-recovery rule raised the h64 minimum health to 0.60 and avoided near-empty energy and hydration on the new seed block.

### Five-tick hidden-storm stress

The stress run changed the storm schedule from 3 of every 6 actions to 5 of every 6, while retaining the v2 hydration mechanics and a 128-step horizon. The aligned NEXUS profile survived 64/64 seeds at both initializations. Both runs had mean reward 29.070, mean terminal health 0.294, mean terminal energy 0.607, and mean terminal hydration 0.471. The lowest observed health/energy/hydration were 0.20/0.30/0.28 and 0.20/0.35/0.28. The duration estimate reached its configured maximum of five ticks in all episodes; each initialization recorded 113 shock events and 672 screened calm-recovery admissions across its 64 episodes.

| Five-tick storm arm | Survival | Mean steps | Mean reward |
|---|---:|---:|---:|
| V2 NEXUS, aligned prior, init 55601 | 64/64 | 128.0 | 29.070 |
| V2 NEXUS, aligned prior, init 55602 | 64/64 | 128.0 | 29.070 |
| V2 NEXUS, v1 prior, each initialization | 0/64 | 17.5 | 2.434 |
| Fixed 3-tick pulse trace | 0/64 | 6.0 | -2.420 |
| Duration-matched 5-tick pulse trace | 35/64 | 71.4 | 6.039 |
| Privileged oracle-state controller | 64/64 | 128.0 | 13.843 |

The oracle chooses Rest during the simulator's hidden storm phase and Forage otherwise. It checks that this harsher environment admits a viable policy; its privileged information makes it an upper bound, not an agent baseline. The duration-matched pulse controller still fails when episodes begin partway through a storm before the first cue. In contrast, the NEXUS candidate completed all episodes and exceeded the oracle's reward, but its mean terminal health is only 0.294 and its minimum health is 0.20. This demonstrates performance inside the simulator, not a robust safety margin. The paired candidate-minus-v1-prior reward change was +26.635 for each initialization (95% environment-seed bootstrap intervals [26.424, 26.845] and [26.424, 26.843]); all 64 pairs favored the candidate on survival. Every one of 576 complete trajectories replayed exactly with zero failures. The source-bound repeat produced identical episode records and arm summaries, and its source hashes match the current files.

### Eight-tick hidden-storm adaptive stress (initial guard)

An eight-tick storm schedule (8 ticks storm out of 10-tick cycle, horizon 128, seeds 100201–100264, agent initializations 56001 and 56002) tested the first adaptive hazard guard. The aligned NEXUS-Ω agent achieved 64/64 survival across both initializations with 0 replay failures across all 576 complete trajectories. In contrast, the legacy v1 prior survived 0/64 (survival delta +1.0, paired sign test p = 1.08e-19, 95% bootstrap CI [1.0, 1.0]). The paired reward delta was +3.232 (95% CI [3.070, 3.391]) for initialization 56001 and +3.180 (95% CI [3.030, 3.330]) for initialization 56002. This version reached only 0.20 minimum health, which motivated the cold-start and pulse-period refinement below.

### Eight-tick pulse-phase learning confirmation

A fresh eight-of-ten storm block (128-step horizon, seeds 100501–100564, initializations 56201 and 56202) evaluated the revised phase learner. It waits in Rest for a pulse at cold start, estimates cycle interval from pulse recurrences, and extends storm duration from late health residuals. Both initializations estimated an eight-tick storm and ten-tick pulse interval in every episode.

| Eight-tick phase-learning arm | Survival | Mean reward | Mean steps |
|---|---:|---:|---:|
| V2 NEXUS, aligned prior, init 56201 | 64/64 | 14.128 | 128.0 |
| V2 NEXUS, aligned prior, init 56202 | 64/64 | 14.128 | 128.0 |
| V2 NEXUS, v1 prior, each initialization | 0/64 | 2.287 | 17.4 |
| Duration-matched eight-tick pulse trace | 28/64 | 4.940 | 57.7 |
| Reactive cue-only | 0/64 | -2.547 | 4.0 |
| Privileged oracle-state controller | 64/64 | 15.341 | 128.0 |

Both aligned initializations had mean terminal health 0.60, energy 0.95, and hydration 0.930; minimum observed values were health 0.60, energy 0.90, and hydration 0.85. Each recorded 64 health-shock updates across 64 episodes. The guard activated on 6,602 of 8,192 transitions and overrode 6,595 planner actions per initialization, so this is a conservative hybrid controller rather than evidence of a learned optimal policy. Candidate-minus-v1-prior reward was +11.841 for each initialization (95% environment-seed bootstrap intervals [11.668, 12.012] and [11.666, 12.010]); all 64 pairs favored the candidate on survival. Its self-hash and all nine source hashes match; all 576 stored episodes replayed exactly. Receipt digest: `ec9914a0b72b0ed5bbdbdcf8d96b21d2a0a5cd41a3379e293538b3ee0408c7af`.

### Hidden cycle-length shift

The versioned v3 environment changed the pulse interval once at step 64, from 8 to 10, while keeping storm duration at six steps. It preserved the current cycle's fractional phase, then generated observations under the new schedule. The original and revised estimators used the same 64 environment seeds (100801–100864) and the same two agent initializations (56501, 56502), with full trace replay in both blocks.

| Hidden interval-shift arm | Survival | Mean total reward | Mean post-shift reward | Mean steps |
|---|---:|---:|---:|---:|
| V2 NEXUS, two-pulse estimator | 128/128 | 14.788 | 10.077 | 128.0 |
| V2 NEXUS, v1 prior | 0/128 | 1.845 | 0.000 | 16.9 |
| Fixed six-tick trace | 37/64 | 12.891 | 6.953 | 88.3 |
| Privileged oracle | 64/64 | 21.089 | 12.289 | 128.0 |

The first estimator block identified the new interval after 26.453 steps on average. Requiring two near-matching longer intervals reduced that lag to 16.453 (10 steps faster) with 100% estimate coverage. It immediately lowers its estimate on shorter intervals; the two-pulse rule applies only to upward changes. In this particular interval-increase assay, the candidate's actions and reward traces were identical between estimator versions: the period estimate changed faster, but this did not alter decisions. The phase-tracking improvement is measured; a policy benefit from it is not established. The candidate's minimum observed health was 0.20 despite complete survival. The revised receipt digest is `c1061ebd5148505fa4cc6f93f9e14c430eaf79e548e32fa2bb3256a36ff07e50`; its self-hash and source hashes match, and all 448 trajectories replayed exactly.

### Joint interval and duration expansion

A separate hidden shift at step 64 changed the schedule from a six-of-ten storm to a seven-of-eight storm. This couples higher hazard frequency with a longer storm, so it does not isolate either change. The candidate used provisional and confirmed period estimates plus the boundary latch. On 64 fresh environment seeds and two initializations, it completed 128/128 episodes; the duration-matched trace and v1-prior agent survived 0/64 and 0/128, while the privileged oracle survived 64/64.

| Severe-shift arm | Survival | Mean reward | Mean post-shift reward | Mean terminal health |
|---|---:|---:|---:|---:|
| V2 NEXUS, reserve-aware thresholds (energy 0.80, hydration 0.40) | 128/128 | 15.941 | 3.720 | 0.600 |
| V2 NEXUS, previous thresholds (energy 0.75, hydration 0.98) | 128/128 | 8.141 | 3.200 | 0.600 |
| V2 NEXUS, v1 prior | 0/128 | 1.791 | 0.000 | 0.000 |
| Fixed six-tick trace | 0/64 | 8.431 | -0.976 | 0.000 |
| Privileged oracle | 64/64 | 18.245 | 6.000 | 1.000 |

The paired ablation held environment seeds and both agent initializations fixed while changing only shelter resource thresholds. The reserve-aware threshold raised mean total reward by 7.800 points (95% environment-seed bootstrap CI 7.669–7.931) and mean post-shift reward by 0.520 (CI 0.408–0.638). Both profiles survived 128/128 episodes and made zero post-shift storm-forage actions. Minimum health stayed 0.60; minimum energy and hydration fell from 0.90/0.85 to 0.75/0.65. Guard overrides fell from 15,380 to 12,052 of 16,384 transitions (21.6% fewer), while the boundary latch count stayed at 1,510. The new interval estimate covered all episodes with mean lag 12.969 steps.

The candidate receipt replayed 448/448 trajectories. The threshold ablation independently replayed 128/128 episodes for each profile by hashing complete observations and transitions. Source hashes and receipt self-hashes match. Candidate receipt digest: `729f9a786064029ecd75173856d97e9d85860883daed59d388edce5e7e6cfac3`; ablation receipt digest: `8687946cf829671f1cafb23b9969f8c07213772fff696dcd22cbb5139a374f1b`.

## Interpretation and limits

The evidence now includes fixed-schedule stress, an interval increase, and one joint frequency/duration expansion. The reserve-aware guard recovered reward while preserving survival and zero storm forages on this block, at the cost of lower minimum energy and hydration. Evidence remains bounded to synthetic schedules, two fixed initializations, 64 seeds per main stress block, and 128 steps. Isolated duration-only changes, repeated or gradual regime changes, noisy or missed pulses, resource changes, and adversarial schedules remain untested. The thresholds are uncalibrated, not safety guarantees or controller promotion.

The assay does not establish real-world safety, biological memory, learned causal inference, or independent replication. It does not promote the v2 profile for deployment. The complete episode traces, source hashes, replay counts, and receipt digests are retained in the machine-readable receipts linked above.
