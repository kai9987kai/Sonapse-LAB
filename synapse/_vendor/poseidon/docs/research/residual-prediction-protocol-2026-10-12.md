# Frozen linear anchor and learned residual: prediction protocol

This bounded continuation of the learning-foundation experiment asks whether a learned correction improves the strongest existing state predictor. It is a prediction experiment. The failed 632000000 controller and its evidence are preserved; the default frozen Tidal controller remains unchanged.

The new `linear-residual-v1` option freezes the training-only action-conditioned affine ridge state model and adds an ensemble residual. Each residual mean head starts at zero. The state anchor and residual multiplier are serialized, hashed, validated and replayed with the fitted artifact. The original `mlp-v1` remains the default fit architecture. Historical configs without an architecture field retain that meaning.

Existing joint NLL selects a checkpoint on selection episodes. After this, the residual multiplier is selected from `[0, 0.25, 0.5, 1]` by equally weighted episode state MSE on that selection split, with ties favoring the smaller multiplier. Calibration happens after both choices are frozen. Multiplier zero retains the linear anchor; it is not evidence of a neural improvement. Only residual means are scaled. Aleatoric variance is not scaled away, and the shared anchor's fitting uncertainty is not represented by ensemble disagreement.

`tools/run_residual_pilot.py` registers and archives source/protocol before fitting. It fits both architectures before collecting any final test episodes. Budgets for both: 128 train, 32 selection, 64 calibration episodes, 96-step maximum, 1,024 updates, batch 128, five 128-unit members and two CPU Torch threads. This matches data and update counts, not FLOPs or elapsed time. There is one independently seeded training lineage per architecture; a separate single-network comparison remains unimplemented.

| Partition | Inclusive seeds |
| --- | --- |
| Train | 652000000–652000127 |
| Selection | 653000000–653000031 |
| Calibration | 654000000–654000063 |
| ID prediction test | 655000000–655000063 |
| Scarcity 2.5 test | 656000000–656000063 |

Both architectures share exactly these partitions for paired comparison. These ranges differ from the previous consumed pilot. After this run, all test seeds are consumed; reusing them reproduces evidence but cannot validate a candidate tuned on its results. The registered `episode-mixture-v2` generation and existing baselines are retained. No test-based tuning or retries with changed settings are permitted in this pilot.

The new prediction screen requires full receipt replay. Both conditions must have a strictly positive lower 95% paired bootstrap bound for state-MSE gain over the linear baseline and a nonnegative lower bound for reward-MAE gain over the action-mean baseline. Missing, invalid or contradictory evidence cannot qualify. Passing this screen does not promote a controller: intervals are descriptive, not corrected for multiple comparisons or training-lineage variability. Death Brier score, interval widths and behavior subgroups remain separate diagnostics, not calibrated survival guarantees.

The one-step HorizonX action score uses reward and interval width; it does not directly value the predicted next-state mean. The full CognitiveAgent additionally uses state means in its homeostatic score (default blend 0.05) and predicted-health guard. This pilot does not run control or attribute control gains to state prediction. A later control experiment needs separately frozen controller settings, policy-query support, paired survival/reward tests and ablations separating state, reward and interval effects.

This design uses established model-based learning ideas without a novelty claim. [PETS](https://arxiv.org/abs/1805.12114) combines probabilistic ensembles with trajectory sampling; Poseidon's present one-step score is a different mechanism. [MOPO](https://arxiv.org/abs/2005.13239) studies offline model exploitation and uncertainty penalties; its return guarantees are not established by Poseidon's interval-width heuristic. The immediately useful portfolio lessons remain baseline-preserving inactive capacity, disjoint proposal/audit evidence and action-aligned predictions, as recorded in the [portfolio review](portfolio-refresh-2026-10-11.md).

Run from the repository root with installed dependencies:

```powershell
python -m tools.run_residual_pilot --output outputs/residual-pilot-652000000 --seed 652000000
```

The output directory must not already exist. The run is bounded by fixed episode/update counts and performs no activation, publication or controller mutation.
