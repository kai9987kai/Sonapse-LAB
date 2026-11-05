# Contrast Atlas design

Contrast Atlas is an opt-in, one-step controller for Poseidon's synthetic TidePool world. It adds a fitted residual store beside the frozen Tidal core, chooses how much correction to admit for each state channel, and compares a proposed action with the core policy using a calibrated action-difference diagnostic. The learned policy remains the default. Fitting and running Contrast do not train the neural checkpoint, change the active-core pointer, or promote a candidate.

This document describes the implemented mechanism and experiment protocol. It contains no measured performance claim. The implementation is in [contrast.py](../poseidon/contrast.py), [contrast_experiments.py](../poseidon/contrast_experiments.py), and [runtime.py](../poseidon/runtime.py). The original [Atlas v2 design](ATLAS_DESIGN.md) remains a separate baseline.

## Data and access boundaries

TidePool exposes sixteen normalized observations and six macro actions: rest, forage, drink, shelter, explore, and flee. The sixteen channels describe reserves, local conditions, resources, the previous action, and episode progress. Contrast plans from this observation alone. Its `plan(observation)` interface receives no simulator snapshot, episode seed, teacher action, or future outcome.

Offline fitting and evaluation have broader access. They clone a frozen simulator snapshot and take each of the six actions once. Those six successor states share the same starting state and event-keyed exogenous randomness. This is a synthetic intervention assay; it does not demonstrate causal identification in an unknown physical environment.

Four episode partitions have different jobs:

| Partition | Default CLI seeds | Role |
| --- | --- | --- |
| Residual training | `101000001` through `101000012` | Store action-conditioned transition residuals. |
| Feature selection | `102000001` through `102000008` | Select sixteen correction weights with equal weight per episode. |
| Calibration | `103000001` through `103000008` | Measure action vital errors and episode-max advantage overestimation after selection is frozen. |
| Audit | Starts at `104000001`; four episodes by default | Compare the nine controllers and audit their recorded decisions. |

These are configuration defaults, not observed sample counts. The fitting API rejects overlap between training, selection, and calibration seeds. The experiment rejects audit seeds that overlap any recorded fitting partition of Contrast or Atlas v2. This separation does not by itself establish independence from every historical neural training example.

The default fitter schedules up to twenty-four anchor snapshots per episode over a 128-step horizon. Episodes that terminate early contribute fewer anchors. It cycles the configured scarcity levels `1.0`, `2.0`, and `3.0`. A local seeded RNG follows a mixture of teacher actions, core actions, and random actions with probabilities `0.7`, `0.2`, and `0.1`; the teacher is used only for offline collection. Receipts distinguish episodes, anchors, and six-action branch transitions. Branches and anchors within an episode are correlated.

## Action-conditioned residual retrieval

For a stored transition with observation `o`, action `a`, actual successor `y`, and frozen neural prediction `b(o,a)`, the residual is `y - b(o,a)`. At planning time, Contrast queries only the records for the requested action. Distance is the root mean squared difference across the sixteen observation channels.

The default retriever takes five nearest records, with stable ordering for ties, and averages their residuals using normalized weights proportional to `1 / (distance + 1e-6)`. Adding that average to the current neural prediction produces the raw corrected prediction. The artifact records the source identities used by the query.

Each channel then applies its frozen shrinkage weight:

```text
final[j] = clip(base[j] + alpha[j] * (raw[j] - base[j]), 0, 1)
```

Zero retains the neural prediction. One admits the whole retrieved residual. Intermediate weights admit a fraction of the correction. The weights are global per channel for the fitted candidate; they are not retrained during a world episode.

## Episode-balanced feature selection

For each channel, the selector evaluates the default grid `{0, 0.25, 0.5, 0.75, 1}` on the selection partition. It first averages squared successor-state errors within each episode, then averages those episode means. A longer episode therefore does not get more weight merely because it produced more anchors.

The selector chooses the coefficient with the smallest episode-balanced MSE. Ties prefer the smaller coefficient. A nonzero coefficient must strictly improve on the zero-correction baseline and satisfy the configured relative improvement threshold, `min_selection_gain=0.02` by default. Otherwise, that channel gets zero. The artifact records `selection.alphas`, `selection.base_mse`, `selection.admitted_mse`, the episode count, and the branch sample count.

This is finite-grid empirical selection. The relative threshold is a design rule; it is not a hypothesis test or a guarantee that the selected correction will improve a new episode. Calibration episodes do not choose these coefficients. If an audit is used to revise the candidate or its settings, that audit becomes development evidence and a revised candidate needs a fresh audit suite.

## Calibration and action comparison

Calibration evaluates three frozen prediction modes: selected memory (`memory`), full residual correction (`unfiltered`), and zero residual correction (`base`). Each mode has its own diagnostics.

For each action, the vital error is the largest absolute prediction error among health, energy, hydration, stamina, and exposure. The artifact stores the observed quantile of these branch-level errors, their mean, and supported calibration counts. These pooled action diagnostics differ from the episode-max paired diagnostic described below.

The decision score is the fixed `reserve_utility` function. It rewards health, energy, hydration, and stamina, subtracts exposure and near-threshold shortfall penalties, and adds a small fixed contribution from the current policy probability. Its default linear terms are:

```text
2.0 * health + 1.6 * energy + 1.9 * hydration + 0.8 * stamina
- 1.2 * exposure - 2.0 * shortfall + 0.15 * policy_probability
```

The shortfall thresholds and weights are defined in `reserve_utility`. This score is not the environment's reward or the expected return of a complete episode.

At a calibration anchor, let `p` be the core's incumbent policy action. For every action `a`, Contrast compares the predicted utility difference with the utility difference of the two actual simulator branches:

```text
predicted_advantage(a) = predicted_utility(a) - predicted_utility(p)
actual_advantage(a)    = actual_utility(a) - actual_utility(p)
anchor_error          = max(0, max_a(predicted_advantage(a) - actual_advantage(a)))
episode_error         = max_anchor(anchor_error)
```

The paired radius is the observed quantile of the calibration episode errors, separately for each prediction mode. With `n` episodes and quantile `q`, the implementation uses the sorted observation at zero-based index `ceil(q*n)-1`, clipped to the available range. The default `q` is `0.90`. The artifact retains the individual `episode_max_errors`, so the aggregation is inspectable.

This radius is a descriptive order statistic. It has no conformal finite-sample correction, no hypothesis-testing guarantee, and no assurance of coverage under a different policy's visited states. Taking episode maxima acknowledges within-episode dependence; it does not establish the assumptions needed for a safety or risk-control theorem.

The default `paired` gate proposes the trusted action with the highest point utility. An action is called `trusted` in the implementation only when its nearest record is within `support_radius=0.25`, it has some supported calibration evidence, and its action vital-error radius is at most `max_vital_error=0.20`. This label denotes these checks, not safety.

An override also requires the incumbent to pass those checks. The proposed action's point advantage minus the mode-specific paired radius must strictly exceed `override_margin=0.02`. Otherwise, Contrast retains the incumbent. It records the proposal, executed action, support checks, and fallback reason.

The `absolute` control uses the same selected corrections but ranks pessimistic reserve utility and compares the challenger with the incumbent's optimistic utility. Vital error radii lower positive reserve channels and raise exposure for the pessimistic score, with the converse for the optimistic score. This reproduces an absolute-bound decision test for comparison with the paired-difference gate.

## Nine controls and a matched prediction probe

Every experiment uses the same seed set, scarcity, and horizon for all controls. Event-keyed randomness aligns exogenous events after paths diverge. Execution order rotates across seeds. The controls are:

| Receipt arm | Mechanism |
| --- | --- |
| `policy` | Frozen learned policy; the incumbent baseline. |
| `neural_mpc` | Existing two-step neural planner, with its default `0.60` policy and `0.40` dynamics combination. |
| `atlas_v2` | Original residual Atlas with its absolute-bound gate. |
| `contrast` | Selected channel corrections and paired gate. |
| `contrast_absolute` | Selected corrections and absolute gate. |
| `contrast_no_memory` | Zero residual correction and base-mode calibration; support geometry is retained. |
| `contrast_unfiltered` | Unit correction weights and unfiltered-mode calibration; paired gate. |
| `heuristic` | Simulator task heuristic, labeled explicitly. |
| `random` | Event-keyed random action control. |

The no-memory arm disables residual contributions. It retains the stored observation geometry used for support checks; it is not a deletion of every artifact record or a learned-memory recall test.

The evaluator branches all six actions at each controller's actual pre-action snapshot, after the controller has chosen. It records actual one-step utility advantages, adverse overrides, one-step regret, and paired-radius exceedances. Evaluator snapshots and branches are not passed back into planning. Decision latency and evaluator audit latency are measured separately.

Prediction errors along different controllers' trajectories mix prediction quality with differences in visited states. The experiment therefore also probes `base`, `unfiltered`, and selected `memory` predictions at exactly the same incumbent-policy observations, against all six actual successors. `prediction_comparison` reports shared anchor and branch counts, sixteen per-channel MSEs, and full-state MSE. This matched probe isolates the three prediction modes at those policy anchors. It does not establish that a predictor remains accurate on another controller's changed trajectory.

Reward deltas compare each control with the policy on the same seed. The reported `ci95` is a descriptive fixed-seed bootstrap interval over episode deltas, with 1,000 resamples. Small suites, different compute budgets, and the one-step utility surrogate limit interpretations of these comparisons.

## Commands and API

Run commands from the repository root. Both fitted artifacts are required for the nine-arm experiment because Atlas v2 is a baseline:

```powershell
python -m poseidon atlas-fit
python -m poseidon contrast-fit
python -m poseidon contrast-experiment --seed 104000001 --episodes 4 --max-steps 64 --scarcity 2.5
python -m poseidon world --planner contrast --seed 104000001 --max-steps 128 --scarcity 2.5
python -m poseidon serve --port 8787
```

The fitting commands write `outputs/atlas/atlas.json` and `outputs/contrast/atlas.json`; Contrast also writes `outputs/contrast/fit_receipt.json`. The experiment exports a content-addressed JSON receipt under `outputs/contrast_experiments/`. Use the emitted `artifact_url` to find the actual filename. A saved receipt can be checked with `python -m poseidon verify-contrast "outputs/contrast_experiments/<receipt-filename>.json"`, replacing the placeholder with that filename.

`GET /api/status` exposes `contrast.ready`, `contrast.path`, `contrast.artifact_sha256`, and `contrast.fit_receipt` with `training_samples`, `partition`, `selection`, `calibration`, and `config`. Readiness flags are nested: `contrast.busy`, `contrast.status_cached`, `atlas.busy`, and `atlas.status_cached`. Source freshness fields `source_current`, `restart_required`, and `changed_source_files` are top-level. While compute holds the runtime lock, readiness may be a cached snapshot; the workbench labels that state. A source-change banner blocks recorded experiments until the process is restarted.

`POST /api/contrast-experiment` accepts only `seed`, `episodes`, `max_steps`, and `scarcity`. The browser endpoint allows 1–8 paired episodes and a 32–256-step horizon. It returns `schema`, `experiment_id`, `summary`, `paired`, `rows`, `contrast`, `prediction_comparison`, `limits`, `artifact_url`, `receipt_sha256`, `verification`, and the first Contrast episode as `replay`. The original six-arm endpoint remains `/api/experiment`.

For a single replayable world episode, send `/api/respond` a JSON body with `mode: "world"` and `planner: "contrast"`, plus the desired `prompt`, `seed`, `scarcity`, and `max_steps`. The Survival UI sends its explicit resource scarcity and episode horizon settings, defaulting to `1.0` and `256`. Its `episode.trajectory` records the actual decisions. Contrast decision evidence includes `policy_action`, `proposed_action`, `override_accepted`, `point_advantage`, `advantage_error_radius`, `empirical_advantage_margin`, `gate`, `feature_alphas`, `fallback_reason`, and six `candidates`. Candidate evidence contains base and final successor predictions, support distance, vital error radius, policy probability, utility scores, and retrieved source identities.

The workbench presents the nine-arm comparison, matched prediction probe, actual episode replay, paired margins, and per-channel weights. Selecting Contrast in the comparison panel does not change the default Survival controller. Receipt links download the original bytes: parsing and reserializing JSON can change number representations and invalidate its checksum.

## Integrity and verification scope

The fitted artifact binds the actual checkpoint file hash, the neural model parameter digest, the world version, and finite bounded JSON content. It validates schema, partitions, selection evidence, calibration evidence, and complete six-action anchors. The runtime rejects an incompatible or mutated core and caches candidates only while their file and core identity remain current.

Before recording an experiment, a source guard compares the current Python source tree with a snapshot captured when Poseidon entered the process. The protocol commits source hashes, model and checkpoint identities, both fitted artifact hashes and partitions, planner configuration, and the seed schedule. A further identity check rejects changes during execution. This detects ordinary workspace edits; it is not authenticated code attestation.

The receipt verifier checks checksums and protocol structure, replays actual actions, recomputes all six simulator branches, validates utility and decision-evidence arithmetic, and rebuilds episode summaries and matched-probe error statistics. It can do this without neural model weights. Consequently, successful verification establishes simulator and evidence consistency, not authorship of every recorded prediction or agreement with a separately rerun neural controller.

No mode provides a real-world survival guarantee, a multi-step planning theorem, biological interpretation, automatic promotion, or proof of broad superiority. Accepted local overrides, lower matched MSE, reward differences, and survival rates remain distinct measurements.

## Primary research context

The mechanism combines established ideas in a small inspectable instrument. The references below motivate distinctions in the design; their theorems are not transferred to Contrast.

- [Strens and Moore, *Policy Search using Paired Comparisons* (2002)](https://www.jmlr.org/papers/volume3/strens02a/strens02a.pdf) studies policy comparisons using shared scenarios and controlled randomness, including overfitting to fixed scenarios. It motivates paired seeds and a separate audit suite. Contrast's observation-level one-step advantage gate is a different mechanism.
- [Laroche, Trichelair, and Tachet des Combes, *Safe Policy Improvement with Baseline Bootstrapping* (2019)](https://arxiv.org/abs/1712.06924) uses baseline bootstrapping under uncertainty and provides guarantees for its stated algorithms and setting. Contrast adopts the practical idea of retaining the incumbent when local evidence is unresolved. It does not implement SPIBB or claim its policy-improvement guarantee.
- [Angelopoulos, Bates, Candès, Jordan, and Lei, *Learn then Test* (2022)](https://arxiv.org/abs/2110.01052) formulates predictive risk control through multiple hypothesis testing. It is a reference point for what a formal calibration claim requires. Contrast uses a selection threshold and observed error quantiles without that testing construction.
- [Angelopoulos, Bates, Fisch, Lei, and Schuster, *Conformal Risk Control*](https://arxiv.org/abs/2208.02814) develops control of expected monotone losses under its specified procedure. Contrast's adaptive trajectory decisions and descriptive episode-max radius are not that procedure. The term "calibration" here does not imply conformal coverage or controlled expected risk.
