# Research-grounded evidence and reliability upgrade

This local upgrade improves the CPU research workbench across reasoning,
controller adapters, numerical diagnostics, and the browser. The frozen Tidal
checkpoint is preserved and candidates remain opt-in. Source v3.13.0 is not a
publication or a new trained-model release.

## What changed

The generated reasoning audit checks six families against independent numeric
and structural expectations: exact arithmetic, linear equations, determinants,
gcd, Hanoi move replay and edit distance. Unsupported and invalid controls test
the tool router's refusal behavior. Unexpected exceptions count as errors rather
than successful abstentions. Wrong numbers in the answer cannot pass merely
because the expected value occurs in explanatory prose.

Acceptance means a nonempty answer with the solver's verification flag;
correctness is measured independently. Reports show coverage, accepted-answer
error, false self-verification, missed answerable cases and correct refusals.
Risk is undefined when nothing is accepted. These are descriptive metrics on a
finite generated suite, not calibrated probabilities or a general language-model
abstention benchmark. Six tested families do not verify every Problem Forge tool.

Reports bind generated cases and source fingerprints. Integrity checking checks
the saved contents and case manifest; replay executes the suite again. A process
whose Python sources changed since import refuses to create a fresh audit.
Fingerprints detect changes; they do not authenticate evidence or prove answers.
The older `tools/reasoning_bench.py` is explicitly a display-token smoke test and
now returns a failing exit code when its checks fail.

The standalone quartet coordinator fits its panel from supplied historical
member errors. Decisions require actual predictions, interface state, margin,
risk estimate and calibration identity. Missing evidence preserves the baseline.
It checks every surviving interface hypothesis and separates internal/external
action identifiers. Calibration provenance remains caller-declared and
unverified. Unknown successor values and stability are reported as unknown.
The coordinator is not activated in the runtime's default controller.

TETHER computes reward variance with consistent batch/member dimensions.
TETHER reweighting and FAULTLINE panel slicing cannot inherit calibration from
the original full ensemble; planning falls back until the changed predictor has
its own calibration. FAULTLINE's canonical synthetic audit cannot select members
of an unrelated ORACLE artifact. These corrections do not establish performance
gains.

CROSSWIRE reads actual TidePool health, reserve, exposure and resource fields,
with goal/hazard IDs matching its transition model. This remains a heuristic
abstraction. Successor memory uses terrain, shelter and resource scent for its
state hash and triggers replay on REST (0), rather than FLEE (5).

Poincaré APIs honor iterations, curvature and seeds; reject unsupported,
nonfinite or numerically unrepresentable inputs; fix rectangular matrix ranks
and Betti numbers; and stabilize hyperbolic distances. Mean-curvature relaxation
is labeled a surrogate, separate from geometric Ricci flow or Perelman entropy.
Finite orbit classifications and reserve heuristics are not KAM theorems or
policy safety proofs. Five synthetic diagnostics replace the invented
100-domain certification count. The browser distinguishes availability from
computed diagnostics and starts unrun telemetry without example results.

## Use

Restart a running server after Python source changes. Open **Independent
reasoning audit**, choose a seed and 1–32 checks per family, and run it.
Six generated families and eight refusal controls give 56 default cases or 200
cases at the maximum budget.

```powershell
python -m poseidon reasoning-audit --seed 20261030 --samples 32 --output outputs/reasoning-audit.json
python -m poseidon verify-reasoning-audit outputs/reasoning-audit.json
```

`verify-reasoning-audit --skip-replay` checks integrity without execution and
leaves `replay_verified` false. POST integer `seed` and `samples` fields to
`/api/reasoning/audit` for the same report. Unknown fields, boolean counts and
excessive budgets are rejected. No model weights are loaded.

## Research used

Primary sources checked on 30 October 2026 informed these choices. Their
performance results and guarantees are not Poseidon's.

| Source | Applied idea | Boundary |
|---|---|---|
| [TAPS, revised 26 October 2026](https://arxiv.org/abs/2610.16318) | Measure answering and selection separately through coverage and accepted error | No learned confidence estimator or TAPS reproduction |
| [AbstentionBench, June 2025](https://arxiv.org/abs/2506.09038) | Include unanswerable controls and measure refusals | Finite tool controls, separate from its language tasks |
| [Conformal prediction after data-dependent model selection, revised April 2026](https://arxiv.org/abs/2408.07066) | Changed member selection cannot inherit the previous predictor's calibration | No new conformal calibrator; unsupported planning falls back |
| [WorldTest, revised May 2026](https://arxiv.org/abs/2510.19788) | Separate prediction fit, interface abstraction and control outcomes | No new multistep model or controller promotion |
| [Valid Stopping in Adaptive Generator-Verifier Loops, October 2026](https://arxiv.org/abs/2610.06432) | Cheap proxies/fingerprints need independent correctness checks | Its statistical stopping method is not implemented |

## Validation

Regression tests reproduced defects before their fixes. Independent review found
additional displayed-answer, extreme-number, matrix-validation and stale-source
gaps; each received a regression and correction. Integrated results are recorded
below after verification.

Two generated runs (seeds 20261030 and 20261031, 32 checks per family) each tested
192 solvable inputs and eight refusal controls, with no wrong accepted answer,
unexpected exception, missed answerable case or false acceptance. Both replayed
exactly. These are additional software checks, not an untouched statistical
confirmation protocol.

Final integrated verification on 30 October 2026:

- `python -m pytest -q`: **1,296 passed**, 406.27 seconds, with native CPU
  thread budgets set to one (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`). No tests were deselected.
- `node --check poseidon/web/app.js` and the shipped Node browser harness:
  **46 passed**. Actual browser interaction also ran the 200-case audit and
  displayed 192 correct answers/eight refusals with accurate evidence labels.
- Two separate 200-case generated audits replayed exactly. Saved reports are
  under `outputs/research/evidence-upgrade-20261030/` (local generated outputs).
- The v3.13.0 wheel built offline without resolving dependencies; all **191**
  packaged source/asset files matched the checkout. An isolated extracted-wheel
  process ran/replayed its generated audit successfully.
- The legacy display-token smoke suite passed 109/109; this is separate from
  independent correctness checks and is not a scientific accuracy result.
- An independent temporary-package probe changed Python source after import;
  audit creation and replay refused cached implementations, even after the saved
  report was re-fingerprinted with the changed disk hashes.
- The checked local server reports source current, core/language available and
  no status errors. This validation was performed on `codex/evidence-reliability-upgrade` before publication.

The initial full run overlapped final review edits and was rejected by the source
guard. The fresh process result above is the complete final verification.
