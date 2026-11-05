# Poseidon architecture and evidence audit — 11 October 2026

This audit identifies the next engineering work from actual code and bounded probes. It does not establish a new capability or approve model promotion. The diagnostic run began and ended at `97eebc637865a815b458c50fb7ef8200ef1c61df`; the working branch was `codex/auto-improve-overnight`. Earlier exploration began at `f1ce131`, before another local process committed v3 and OmniPlane modules. The inspected v2 and research ORACLE files did not change between those commits.

The active frozen core was `runs/tidal_dagger/core.pt`, SHA-256 `a053104ae63dc7f86baa15b4b34e46730e1ccb92117f221801fb21f90d2d6c27`. The tracked [diagnostic record](architecture-audit-2026-10-11.json) preserves the measured results. The reproducible probe is [tools/audit_cognitive_contracts.py](../../tools/audit_cognitive_contracts.py); generated output goes to the ignored `outputs/architecture-audit-2026-10-11/` directory.

## Proven wiring and verification gaps

| Finding | Evidence | Consequence and required correction |
| --- | --- | --- |
| Runtime and trained research ORACLE are different models. | `cognitive_agent.py:127` constructs `poseidon.oracle.OracleEnsemble` without an artifact path. `hyper_agent.py` does the same. The probe reports three heads and `is_calibrated=false`. `research/oracle_model.py` and its saved `oracle.pt` have a different input/output interface. | Introduce one explicit artifact-bound prediction interface. Fresh initialization must be labeled untrained and cannot supply calibrated planner evidence. |
| Memory and sparse routing do not enter the v2 planner's decision input. | `cognitive_agent.py:172-193` computes memory and router output, then calls `planner.plan(obs_list)`. The outputs are consumed by decision telemetry at lines 280-287. The planner receives current observation and its own ORACLE. | Train and wire memory into prediction/return estimation before claiming a useful lesion assay. Trace each subsystem's actual decision path. |
| Five memory modes produced exactly the same short trajectory. | At seed `447000000`, eight steps, every mode produced `[2,2,1,0,0,0,0,0]` and reward `0.9568093581466558`. | This supports the wiring concern, but one seed alone cannot prove memory is inert or estimate its performance. |
| Cognitive verification accepts empty fabricated evidence. | `cognitive_experiment.py:226-234` checks only a recomputable JSON hash. A forged object with an invalid schema, empty `episodes_data` and invented performance passed. | Validate schema, complete pairs, source/model binding, transition replay, terminal outcomes and recomputed summaries. A checksum establishes byte consistency only. |
| Research fit verification accepts contradictory resealed calibration claims. | Original fit data and calibration replayed successfully. Changing the receipt's duplicated `calibration.coverage` to `0.99`, preserving the artifact and resealing the receipt, also passed `replay_data=True`. `oracle_training.py:474-502` compares model identity and replays artifact calibration, but does not bind all duplicated receipt fields to the artifact. | Require semantic equality of manifest, training, calibration, config and activation claims; test wrong-on-purpose fixtures. |
| Metacognitive residuals are shifted by one transition. | `cognitive_agent.py:347-364` compares the previous prediction to the current transition's successor, then computes the current transition prediction. | Cache predictions before execution with episode/tick/action identity, and compare to that exact successor once. |
| Terminal metrics can use stale/noisy observations. | `cognitive_agent.py:412-426` breaks before assigning `obs=next_obs` and reports `final_health=obs[0]`. Survival uses sensed health rather than `info['death']`. Snapshot capture precedes environment reset. | Take the initial snapshot after reset, report terminal successor state, and distinguish true death from truncation and sensed health. These are source findings; the short probe's health remained 1.0 and did not expose this case. |
| Existing parameter search is selection evidence only. | `tools/auto_improver.py` repeatedly selects on four fixed seeds and calls repository-wide `git restore .` when rejecting. The report records 64 hypotheses, 0 accepted, reward `3.3232` before and after. | Isolate candidates, seed proposal RNG, retain raw pairs and use untouched final audit seeds. Never restore another task's changes. No performance gain was established by that run. |

## The trained research pilot is real but limited

Both original fit and evaluation receipts replayed successfully against the current research sources. Candidate artifact SHA-256: `73cd9a991b7f64ed8745c760edf7d0cf94d8a5e5f8f4b6ad1e9715b55357c39b`; model-state SHA-256: `504d2ddcd34bc73e7aea3cb1d6be3b48ac5582d6ccd6f87e1e8dd053c36c61bc`. It remains inactive and unpromoted.

| Metric | ID: 32 episodes, scarcity 1.25 | Scarcity shift: 32 episodes, scarcity 2.5 |
| --- | ---: | ---: |
| Candidate state-delta MSE | 0.0169613 | 0.0184074 |
| Persistence state-delta MSE | 0.0222422 | 0.0235647 |
| Candidate reward MAE | 0.0125581 | 0.0144184 |
| Training action-mean reward MAE | 0.00701757 | 0.00794195 |
| Candidate death Brier score | 0.00101787 | 0.00163937 |
| Training-prior death Brier score | 0.000000106 | 0.000000106 |
| Mean state interval half-width | 1.23025 | 1.44107 |

State prediction improves over persistence, while reward and death prediction lose to their simple comparators. There are zero observed deaths in the stored fit, selection, calibration and evaluation splits. Death discrimination therefore cannot be established. Large intervals can cover observations while conveying little decision information: many normalized feature ranges are only `[0,1]`.

Calibration uses a cycle of scarcity values `(0.8,1.0,1.25,1.5)` while ID testing fixes scarcity at `1.25`. The stored coverage is empirical; the episode mixture is not identical to this tested subgroup. No unqualified conformal guarantee should be attached to it or to the shifted condition. The evaluator tests predictions under the data-collection policy, not improved control by an ORACLE-driven agent.

The research trainer already samples fixed whole-episode bootstrap draws per member (`oracle_training.py:274-283`). `train_adaptation.py` instead samples rows within minibatches. The next work should extend the existing research implementation and avoid claiming trajectory bootstrap is still missing there.

## New v3/OmniPlane modules

`764727c` and `97eebc6` added SCM, classical amplitude scoring, morphogenesis, continuous 3D simulation, phase analysis, introspective text and sleep replay. Selected dependency tracing shows HyperAgent still instantiates the untrained runtime ORACLE. These new modules were not included in the 14-test audit below. Their names and docstrings are not evidence of do-calculus identification, quantum advantage, Lyapunov stability, biological sleep or useful transfer. Each needs its own actual-input, output, learning, reset and control-benefit contract. Online/sleep weight changes also need explicit episode isolation and artifact identity during evaluation.

Before this audit, `docs/POSEIDON_V2_RESEARCH.md` attributed authorship to Google DeepMind and described a complete scientifically validated platform. Neither attribution nor that validation was supported by the inspected evidence. Its claimed quantile regression was inconsistent with the runtime's log-space fitted curve. This audit corrects those statements and adds explicit implementation/evidence status to that document and the README.

## Validation performed

The targeted suite passed **14 tests in 44.84 seconds**: `test_oracle_research.py`, `test_cognitive_agent.py`, `test_cognitive_experiment.py`, and `test_oracle.py`. The first invocation had 11 passes and three fixture setup errors because the sandbox's default temporary directory was unwritable. Repeating with `tempfile.tempdir` and pytest `--basetemp` inside the workspace resolved those environment errors. Existing passing tests do not cover the contradictory receipt or causal-wiring findings above.

To rerun the diagnostics from the project root in PowerShell:

```powershell
python tools/audit_cognitive_contracts.py
```

The empty cognitive-receipt probe needs installed project dependencies but no trained weights. The wiring probe requires the active Tidal checkpoint. Historical replay and fit-contradiction probes require the saved `outputs/oracle-fit-442000000/` and `outputs/oracle-eval-442000000/` artifacts at their bound identities. Missing artifacts are reported as unavailable, not as successful verification. A fresh checkout can run the independent receipt probe; it cannot reproduce historical training/evaluation without those artifacts.

Tests ran fixture-scale training only. No new release candidate was fitted, activated, committed, pushed or published by this audit. See the [operation design](../superpowers/specs/2026-10-11-learning-transfer-design.md) for the proposed engineering and experiment gates.
