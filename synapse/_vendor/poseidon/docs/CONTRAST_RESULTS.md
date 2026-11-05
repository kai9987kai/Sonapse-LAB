# Contrast Atlas: local v0.3 measurements

On two fresh eight-seed suites, selected residual memory reduced full-state
forecast MSE by **1.93% and 2.50%** against the base neural predictor. The paired
gate made 34 action overrides, all with positive audited one-step utility
advantages. Mean episode reward was slightly lower than the policy baseline in
both suites, and survival was unchanged. These results justify an inspectable
experimental candidate, not activation as the default controller.

The measurements were made on 9 October 2026 on the local CPU. The selected
DAgger checkpoint, original Atlas v2 artifact, simulator, default policy and
language backend were preserved. This v0.3 development work has not been published
to Hugging Face. The frozen `v0.2.0` release remains separate.

## Frozen fit and audit protocol

The fit used 12 training episodes, eight selection episodes and eight calibration
episodes, each with 24 anchors over a 128-step horizon. Scarcity cycled through
1, 2 and 3. It produced 1,728 training, 1,152 selection and 1,152 calibration
branch transitions. Each anchor contributes six correlated action branches.

The selector admitted corrections for 14 of 16 channels. Full-state selection
MSE changed from `0.00590533` to `0.00557184` (5.65% lower); this is a selection
measurement, not an independent audit result. The frozen weights were:

| Weight | Channels |
| --- | --- |
| 1 | Health, energy, stamina, exposure, weather severity, episode progress |
| 0.75 | Hydration, last action |
| 0.5 | Threat, food, water, shelter quality, resource scent |
| 0.25 | Temperature |
| 0 | Daylight, terrain difficulty |

The selected-memory paired radius was `0.3818907287`; the base and unfiltered
radii were `0.5230224337` and `0.3566411387`. With eight calibration episodes, the
observed 90th-percentile episode maximum is the largest observed episode error.
It is descriptive and provides no future coverage or safety guarantee. Selection
weights, utility coefficients, support threshold and gate margin were frozen
before either audit; neither suite was used to retune them.

| Suite | Seeds | Scarcity | Horizon | Controllers | Episodes |
| --- | --- | ---: | ---: | ---: | ---: |
| Moderate | 105000001–105000008 | 2.5 | 128 | 9 | 72 |
| Higher scarcity | 106000001–106000008 | 4.0 | 128 | 9 | 72 |

Scarcity 4 exceeds the fitted range. All seeds are disjoint from the recorded
Contrast and Atlas v2 fitting partitions. This is a small synthetic development
study; separation from these partitions does not establish independence from
every historical neural training example.

## Forecasts on identical snapshots

Each suite supplies 1,024 incumbent-policy anchors and 6,144 actual action
branches. All three predictors receive the same observations and actions. MSE
averages squared normalized errors across all sixteen channels and branches.

| Predictor | Moderate MSE | Higher-scarcity MSE |
| --- | ---: | ---: |
| Base neural | 0.00752069 | 0.00739369 |
| Unfiltered residual memory | 0.00833250 | 0.00795409 |
| Selected residual memory | **0.00737526** | **0.00720856** |

Unfiltered correction worsened full-state error by 10.79% and 7.58% relative to
the base. Feature selection therefore addresses a measured weakness of applying
every retrieved correction. The gain for the selected predictor is modest and
specific to these audited states. Per-channel errors are retained in the
[machine-readable summary](contrast-evaluation.json).

## Action changes and episode outcomes

All nonrandom arms survived all eight episodes in each suite; random survived
none. The resulting survival ceiling limits conclusions about better survival.
Overrides below mean deviations from the frozen core's policy at the arm's own
visited state. Controller trajectories can diverge after an override.

| Controller | Moderate overrides | Moderate reward delta | Higher-scarcity overrides | Higher-scarcity reward delta |
| --- | ---: | ---: | ---: | ---: |
| Policy | 0 | 0 | 0 | 0 |
| Neural MPC | 15 | -0.019032 | 40 | +0.028218 |
| Atlas v2 | 0 | 0 | 0 | 0 |
| Contrast paired | 7 | -0.025148 | 27 | -0.005474 |
| Contrast absolute bounds | 0 | 0 | 0 | 0 |
| Contrast without memory | 0 | 0 | 1 | +0.004620 |
| Contrast unfiltered | 10 | -0.084905 | 31 | -0.014344 |
| Heuristic | 86 | -0.016454 | 147 | -0.008848 |
| Random | 440 | -12.820258 | 305 | -13.947378 |

Reward deltas are mean within-seed differences against policy. Its mean rewards
were `12.884033` and `12.829192`. Contrast's reported bootstrap intervals were
`[-0.061914, +0.001651]` and `[-0.038147, +0.030759]`, from 1,000 resamples of
eight paired differences. Its win/tie/loss counts were `1/5/2` and `3/1/4`.
These finite-sample intervals do not establish population superiority. Exact
zero intervals for arms with identical observed returns describe this sample;
they do not imply zero general uncertainty.

Contrast's seven moderate overrides had mean actual one-step utility advantage
`+0.480824`; its 27 higher-scarcity overrides had mean advantage `+0.449546`.
None had a negative audited advantage. No proposed-action advantage exceeded
the selected-memory radius at its 2,048 visited states. These are observations,
not guarantees. The utility is a fixed reserve score with a policy-prior term,
not episode reward. Positive immediate gains coexisted with lower mean episode
reward: longer-horizon consequences remain unresolved.

Contrast decisions averaged 6.19 and 5.29 ms, versus policy's 1.30 and 1.29 ms.
Controllers have unequal compute budgets. Decision timing excludes evaluator
forks and matched prediction probes. The two audit processes ran concurrently
on the PC, so these timings are descriptive rather than isolated speed tests.

## Receipts and validation

The two receipts independently replayed 144 episodes, **17,234 executed
transitions and 103,404 counterfactual branches**, without loading model weights.
The verifier rebuilds decision arithmetic, simulator outcomes and aggregate
errors. It rejects re-signed changes to action branches, candidate identities,
margins, probe vectors, summaries, fit metadata, completion and promotion flags.
Replay consistency does not establish neural authorship of recorded forecasts.

- Moderate receipt: `outputs/contrast_experiments/3bf46b0111c2f8ef-2f9821149c7f.json`.
- Higher-scarcity receipt: `outputs/contrast_experiments/00cec956b1e574b6-be6ece882333.json`.
- Canonical Contrast artifact: `7ddbb687b5638c52b1f9cfa27fcace6ce801a549ba5227515f87bfa5b4f45002`.
- Core file SHA-256: `a053104ae63dc7f86baa15b4b34e46730e1ccb92117f221801fb21f90d2d6c27`.

The summary records full experiment IDs, receipt file and canonical checksums,
source identities, settings, all nine outcomes, paired differences and every
forecast channel. Large execution receipts remain local generated artifacts.
Repeating the entire fit in a fresh process reproduced the same canonical
artifact exactly. Both original v0.2 receipts still replayed: 96 episodes and
11,079 transitions. The final Python suite passed **172 tests**; compilation,
JavaScript syntax and whitespace checks passed. Live browser checks are recorded
under `outputs/contrast/qa/`.

```powershell
python -m poseidon contrast-fit
python -m poseidon contrast-experiment --seed 105000001 --episodes 8 --max-steps 128 --scarcity 2.5
python -m poseidon contrast-experiment --seed 106000001 --episodes 8 --max-steps 128 --scarcity 4.0
python -m poseidon verify-contrast outputs/contrast_experiments/3bf46b0111c2f8ef-2f9821149c7f.json
python -m poseidon verify-contrast outputs/contrast_experiments/00cec956b1e574b6-be6ece882333.json
```

Exact content-addressed filenames depend on source bytes and recorded timing.
Use the filename emitted by your own run. See [design and access boundaries](CONTRAST_DESIGN.md).
The next useful research question is whether longer-horizon, separately fitted
advantage models can improve episode return while retaining inspectable controls.
This one-step candidate supplies evidence for that question rather than an
answer to it.
