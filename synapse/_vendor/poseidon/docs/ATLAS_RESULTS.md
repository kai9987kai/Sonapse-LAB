# Counterfactual Atlas v0.2: measured local results

Measured on 9 October 2026 with the local CPU runtime. This is a bounded synthetic
development experiment, not a confirmatory or biological result. The new sidecar
is fitted locally; no core/language weights were changed or promoted by this
upgrade. Publication is recorded separately by the GitHub commit and the Hugging
Face release manifest; it does not change the scope of these measurements.

## What improved

The Atlas learns action-conditioned residuals from exact six-action interventions
at frozen TidePool anchors. It retrieves those experiences to correct neural
predictions and exposes their identities and support distances. Its calibrated
error budget and conservative override rule preserve incumbent decisions when a
proposed improvement is unresolved. The workbench now shows paired controls,
candidate futures at each replay step, fallback reasons and exportable receipts.

The initial greedy planner was rejected after only 1/8 survival in each of two
development suites, despite improving the calibration prediction metric. This
failure motivated the conservative rule: a challenger's pessimistic predicted
value must exceed the incumbent's optimistic value plus 0.02. Error margins are
empirical diagnostics, without guaranteed future coverage. The initial artifacts
and unsuccessful receipts remain available locally for inspection.

## Fitting and prediction measurements

The fit uses 12 training episode seeds (81000001–81000012) and six separate
calibration seeds (82000001–82000006), cycling scarcity 1, 2 and 3. There are 24
anchors per episode and six counterfactual actions per anchor: 1,728 training
transitions and 864 calibration transitions. These are 288/144 correlated anchors,
not 1,728/864 independent episodes. No evaluation seed overlaps either partition.

The metric below is maximum absolute error over the first five vital observation
channels, averaged over calibration transitions. The error radius is its observed
90th-percentile order statistic per action. Every action has 144 calibration rows.

| Action | Neural base mean error | Memory correction mean error | Base radius | Corrected radius |
|---|---:|---:|---:|---:|
| Rest | 0.024849 | 0.018272 | 0.056652 | 0.032895 |
| Forage | 0.053569 | 0.044653 | 0.120884 | 0.077490 |
| Drink | 0.045633 | 0.037374 | 0.094748 | 0.065193 |
| Shelter | 0.043099 | 0.030290 | 0.085502 | 0.059110 |
| Explore | 0.033515 | 0.029217 | 0.059325 | 0.050317 |
| Flee | 0.036306 | 0.031186 | 0.069451 | 0.050168 |

The average decreases from **0.039495 to 0.031832 (19.4%)**. This is a specific
calibration metric; it does not establish overall world-model or survival gain.

## Fresh paired episodes after the revision

Each suite has eight new seeds, six controllers and a 128-step horizon. Controllers
share event-keyed exogenous draws, not hidden simulator state. The Atlas receives
only observations. The memory-erased control uses the base predictions and its
separately measured error radii. The neural MPC arm is the policy/dynamics hybrid
with default weights 0.60/0.40; it is not pure MPC.

| Controller | Scarcity 2.5 survival | Mean reward | Scarcity 4.0 survival | Mean reward |
|---|---:|---:|---:|---:|
| Learned policy | 8/8 | 12.887615 | 7/8 | 11.108038 |
| Neural MPC | 8/8 | 12.902347 | 7/8 | 10.734850 |
| Counterfactual Atlas | 8/8 | 12.887615 | 7/8 | 11.108038 |
| Atlas without memory | 8/8 | 12.887615 | 7/8 | 11.108038 |
| Scripted heuristic | 8/8 | 12.897844 | 8/8 | 12.807523 |
| Random control | 3/8 | 5.139977 | 0/8 | -0.963888 |

The Atlas accepted **zero overrides** across these suites: its action sequences
matched the incumbent. Thus survival and reward parity demonstrate preservation
on these seeds, not superior control. The identical paired reward differences and
zero bootstrap interval reflect identical actions, not certain population parity.
The scripted heuristic remains stronger on the stress suite. Atlas decisions also
cost more CPU time than direct policy inference (about 9.2–9.6 versus 2.1–2.2 ms
here); no efficiency gain is claimed and compute budgets differ.

Fresh on-policy mean maximum vital error was 0.021259 versus 0.021799 at scarcity
2.5 (2.5% reduction), and 0.033541 versus 0.034562 at scarcity 4.0 (3.0% reduction).
However, full 16-channel mean episode prediction MSE was **worse** with memory:
0.004924 versus 0.004452, and 0.006430 versus 0.005701 respectively. The memory
correction therefore shows a small vital-specific gain, with no demonstrated
general transition-model advantage. Further work should test feature-specific
correction admission and planning benefit on independently frozen conditions.

## Exact identities and replay

- Current checkpoint file SHA-256:
  `a053104ae63dc7f86baa15b4b34e46730e1ccb92117f221801fb21f90d2d6c27`.
- Atlas artifact SHA-256:
  `db9a4bc78e738ca55e61ec0511ea48814b13ed0ba48f8264f8c43cfda7205dc8`.
- Scarcity 2.5: seeds 96000001–96000008; experiment
  `c260cb232f1635a1bf402fa2c0e9bd7e30776b40a9928c572548c572e4af1ef8`;
  local receipt `outputs/experiments/c260cb232f1635a1-949a615206d8.json`.
- Scarcity 4.0: seeds 97000001–97000008; experiment
  `875452fefbf277dfa4e9af8ff6c69af93e39d1168d2126c591712b2297da1130`;
  local receipt `outputs/experiments/875452fefbf277df-3b1edfb49993.json`.

Both receipts passed independent same-version action replay: **96 episodes and
11,079 transitions**. Verification checks initial/final snapshots, every action,
observation, reward, terminal outcome, prediction-error arithmetic, partitions,
paired summaries and integrity digests without loading model weights. It does not
prove model authorship, probability calibration or optimal action selection.
Download the original JSON bytes; numeric normalization by another serializer can
change its strict canonical checksum. The active-core pointer's historical stale
hash was corrected for release; these receipts already bound the actual checkpoint
bytes directly.

Source hashes and every controller configuration are retained inside each receipt.
Source, checkpoint or atlas changes during execution abort publication. Generated
artifacts are excluded from Git source control; the Hugging Face model snapshot
packages the fitted sidecar and these two final receipts. Source-only checkouts must
fit their own sidecar. Repeated use of these suites should be called development
testing, with new final audit seeds for any later promotion.

## Software verification and repaired evidence

The full suite passes **122 tests**. Python compilation, JavaScript syntax and
`git diff --check` pass. Chromium checks at 1440×1000 and 390×844 cover status,
policy default, the six-arm experiment, pending state, comparisons, receipt
download, replay/scrubbing, six candidate futures, explicit Atlas survival,
invalid input budgets and responsive results. Missing-fit presentation uses a
status fixture; actual missing/corrupt artifacts and checkpoint replacement are
covered by isolated runtime/API tests. Browser checks report no page exceptions.

The promotion auditor now checks real scene prompts/semantic test groups and
matched incumbent survival/dynamics. Missing, constant or nonfinite calibration
evidence cannot pass. Eligibility is separate from activation. Adaptation records
unchanged core identity, actual ensemble changes, disjoint collection/audit seeds,
anchor mixing and independent head bootstrap exposures. The episodic cue assay
now performs actual retrieval and matched donor-record swaps with equal action
supports, explicitly labeling itself as software storage rather than learned
recurrent memory. Legacy catastrophic-death parsing and survival intervals are
corrected; the older Beyond report is marked as historical and superseded.

## Portfolio lineage

All 24 supplied sources were reviewed through the three existing lineage reports:
[cognition](research-core.md), [worlds and memory](research-world.md), and
[systems](research-systems.md). REA's removed replay implementation is now labeled
historical. The implemented synthesis particularly uses [Mnemorph's compiler/auditor
boundary](https://github.com/kai9987kai/Mnemorph/blob/4fc3353b4aa4ea3e2c7b24798121ee65deda6939/src/core/compiler.js),
[Causeway's named-event pairing](https://github.com/kai9987kai/Causeway/blob/e25ea9e6a5a70f60f5a6ee36756aceb3c6c9baac/src/model.js),
[QuantumBot's matched controls](https://github.com/kai9987kai/QuantumBot/blob/42e4fd637da2df1da5241da484ebc2c625050e72/quantumbot/benchmark.py),
and [GenesisEngine's source-bound experiment evidence](https://github.com/kai9987kai/GenesisEngine/blob/a43099ef21e91b75b812e3eb883cb4f3813ab7c7/src/genesis/experiments.py).
No source code or checkpoints from those projects were copied by this upgrade.
The integration is distinctive within Poseidon; global methodological novelty is
not established.
