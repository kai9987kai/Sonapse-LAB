# Experience adaptation: implementation and measured results

6 November 2026. Local CPU development studies; no remote publication or controller
promotion. The upgrade adds an executed-transition model, bounded surprise memory,
and an optional one-step planner. Engineering integration is tested; a general
embodied performance advantage is not established.

## Changes that matter

- Prediction and outcome memory now consume actual executed actions, rewards and
  next sensors. They are persistent, inspectable and frozen during query scoring.
- A bootstrap RLS ensemble learns state/action interactions and reward. Covariance
  is bounded and restored with its RNG. Graph/donor activity conditions the model;
  it does not claim to simulate future neural states.
- Selective memory writes retain actual outcomes in at most 128 slots. Retrieval
  is deterministic and pure. It is inspired by fast-memory research, not Titans.
- Planning keeps the original action available and bounds alternatives to one
  coordinate changed by 0.35. Support prediction skill and a conservative advantage
  margin gate its use. These are uncalibrated engineering heuristics.
- Restore preserves live graft parameter/Adam bindings and the neuromodulatory
  reward baseline/history. Camera fusion affects behavior, supports supervised
  preparation gradients and persists visual parameters.
- CLI/capability evidence no longer supplies invented ablation arrays. Measured
  receipts retain signed differences, exact pairing, raw records, costs, source
  and dependency fingerprints, and query freeze checks. Optional guard validation
  evaluates cloned candidates and binds approvals to the assessed delta/target.

The [research review](../research/2026-11-06-experience-learning-review.md) links
primary sources and distinguishes implemented mechanisms from their inspiration.

## Initial native-world development pilot

Six independent actor/world units: seeds 7, 11, 19 crossed with forage and push;
64 shared exploratory support transitions and 64 separately seeded query steps
per unit. Support trains prediction/memory only; the untrained actor backbone is
fixed. All query learning is disabled. Rewards are native cumulative fixed-step
returns, including resets, rather than task-success percentages.

| Arm | Mean return | Mean reward-prediction MSE | Planner use |
|---|---:|---:|---:|
| Actor only | -0.602578 | 0.000205 | 0% |
| Model | -0.842084 | 0.002090 | 100% |
| Model + memory | -0.842084 | 0.002090 | 100% |
| Shuffled support rewards | -0.408805 | 0.000667 | 98.7% |

Model minus actor: **-0.239506**, descriptive paired bootstrap interval
**[-1.591752, +0.864120]**. Memory actions matched model-only actions. This is a
negative/inconclusive development result, and the sham control doing better is
an additional reason not to promote it.

This result motivated the gate described in the design revision: recent pre-update
reward MSE must beat a running reward-average predictor by 5%; candidate advantage
must clear twice the disagreement-plus-feature-leverage margin. Neither threshold
is a calibrated guarantee. The original receipt remains unchanged.

## Subsequent frozen-gate native evaluation

Ten new actor/world units: seeds 23, 29, 31, 37, 41 crossed with forage and push;
128 shared support transitions and 96 frozen query steps. The algorithm/gate were
fixed before inspecting these outcomes. This remains a development evaluation,
with only two task types and an untrained recurrent actor. Changed seeds and
support/query budgets prevent attributing cross-study differences solely to the gate.

| Arm | Mean return | Mean reward-prediction MSE | Planner use |
|---|---:|---:|---:|
| Actor only | -1.424430 | 0.000228 | 0% |
| Model | -1.517917 | 0.000421 | 32.8% |
| Model + memory | -1.460402 | 0.000530 | 32.8% |
| Shuffled support rewards | -1.424430 | 0.001160 | 0% |

| Paired return difference | Mean | Descriptive 95% interval |
|---|---:|---|
| Model minus actor | -0.093487 | [-0.361867, +0.162004] |
| Memory minus actor | -0.035972 | [-0.492617, +0.393951] |
| Memory minus model | +0.057515 | [-0.160552, +0.279427] |

Memory changes behavior here, but neither experimental arm establishes superiority
over the original actor. The gate suppresses shuffled-reward planning in this
sample; it does not eliminate true-model regressions or model exploitation. Keep
planning **off by default**. No further tuning used these query scores.

Costs: 1,280 support world steps, 1,280 shuffled-support replay actor calls, 3,840
query world steps, 2,560 model updates including sham replay, and zero query
updates. Runtime was 106.33 seconds on this machine during concurrent validation.
All 40 query learned-state freeze checks and 10 checkpoint round trips passed.

Intervals resample seed/task units, never time steps. They are small-sample,
descriptive bootstrap intervals over fixed task types, without a confirmatory
claim or multiple-comparison adjustment.

## Controlled reversal and recurrence probe

This separate synthetic diagnostic uses three hidden reward regimes (+1, -1, +1)
and a declared nonlinear action-dependent reward. Three seeds receive 64 support
and 64 query samples per phase. It is not an embodied task or a trained
continual-backpropagation system. Current and contradictory old-regime outcomes
are scored independently; no phase label enters the learner.

| Phase | Initial model reward | Initial memory reward | Gated model reward | Gated memory reward |
|---|---:|---:|---:|---:|
| Stable | 0.186271 | 0.189200 | 0.174560 | 0.175177 |
| Reversal | 0.128300 | 0.141147 | 0.000000 | 0.010665 |
| Recurrence | 0.147844 | 0.141685 | 0.004537 | 0.016795 |

The zero-action baseline earns zero. Before gating, memory helps reversal by
0.012847 but worsens recurrence by 0.006159 relative to model-only. Both learners'
prequential reversal MSE exceeds the zero predictor (model 0.210497, memory
0.177802, zero 0.149389). The gate leaves this underlying error unchanged but
abstains on many later-phase actions, sharply reducing useful adaptation too.
This is a concrete conservatism/adaptation tradeoff, not a universal improvement.
Old-stable performance also degrades after reversal; contradictory hidden regimes
cannot be inferred from identical visible states without further outcome history.
All nine phase freeze and checkpoint checks passed in each probe. Three seeds do
not justify inferential significance claims.

## Reproduce and inspect

- Original raw receipt/checkpoints: `outputs/adaptation-studies/development-20261106/`.
- Subsequent raw receipt/checkpoints: `outputs/adaptation-studies/independent-20261106/`.
- Original/gated stream records: `outputs/continual-probe-20261106.json` and
  `outputs/continual-probe-gated-20261106.json`.
- Compact tracked summaries: `2026-11-06-measured-summary.json` alongside this report.

```powershell
python -m synapse.cli adaptation-verify outputs/adaptation-studies/independent-20261106
python -m synapse.cli --output outputs evidence --name reproduce --seeds 23 29 31 37 41 --tasks forage push --support-steps 128 --query-steps 96
python -m synapse.continual_probe --output outputs/reproduced-stream.json
python -m pytest -q
```

Output folders are ignored by Git and exclusive names prevent overwriting a study.
Aggregate verification of both native receipts passed (24 and 40 records). The
older receipt correctly reports source drift after the gate change. This verifies
record arithmetic and file integrity, not an independent rerun of the world or
authenticity against someone rewriting every hash.

## Validation and remaining limits

Fresh full suite: **198 passed in 103.07 seconds**. JavaScript syntax and whitespace
checks passed. Regression tests cover live optimizer updates after restore, camera
causality/training, frozen inference, bounded planning/memory, nonfinite rejection,
exact pending-outcome continuation, real pilot reproducibility, source drift,
tamper detection and changed-delta/stale-target rejection.

The workbench mode API passed tests and a real HTTP smoke check collected 32
transitions, switched to memory mode, then preserved model/memory across two
frozen steps. Visual browser verification was blocked when
Computer Use could not verify the browser URL; no visual-completion claim is made.

Legacy flagship transfer, analytic recursive-learning and curriculum surrogate
objectives are not fixed by this release and cannot substantiate embodied transfer
or recursive improvement. Arbitrary legacy topology changes can invalidate graph
metadata/buffers. Supervised preparation still differs from the full live policy;
dormant experts are not trained continual-backpropagation adapters. The old neural
world-model confidence head remains uncalibrated. These are documented limitations,
not evidence of completed research goals.
