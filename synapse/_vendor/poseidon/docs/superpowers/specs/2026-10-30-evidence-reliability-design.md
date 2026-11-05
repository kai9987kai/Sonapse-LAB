# Research-grounded reliability upgrade

Date: 30 October 2026. The user asked to improve the whole project with recent
research and explicitly delegated scope and implementation judgment.

## Intent and scope

Improve the existing CPU-local Python workbench across reasoning, controller
integration, numerical demonstrations, and the browser. Preserve the frozen
Tidal checkpoint, inactive candidates, existing local work, and all negative
results. This is a bounded software upgrade, not a claim to have improved every
experimental subsystem or established model performance.

## Approaches considered

1. Add more named architectures or mathematical domains: broadens demonstrations
   but leaves the observed evidence and integration defects unresolved.
2. Train/promote another controller: requires a separate frozen experiment and
   more evidence than the current integration provides.
3. Repair evidence flow and add independent generated checks: chosen by the
   implementer under delegated judgment because it addresses reproduced bugs and
   makes subsequent experiments more informative.

## Design

- Replace invented coordinator forecasts, risk, random panels, and synthetic
  successor rewards with explicit caller evidence and missing-evidence fallback.
  Fit specialist selection separately from decisions. Use the entire configured
  interface hypothesis space at the actual supplied state. Never fold an action
  identifier into another action. Do not activate this coordinator by default.
- Correct actual observation/action semantics in CROSSWIRE and successor-memory
  adapters, and correct TETHER reward batch dimensions. Preserve baseline fallback
  where predictive evidence is unavailable.
- Fix Poincare numerical API/input defects. Distinguish finite computations,
  toy diagnostics, and fingerprints from mathematical proofs or live safety.
- Add a deterministic reasoning audit with independently generated expected
  answers and exact/structural checks for arithmetic, linear equations, 2x2
  determinants, gcd, Hanoi move replay, and edit distance. Include unsupported
  and malformed inputs. Report answer correctness, coverage, accepted-only risk,
  false self-verification, and correct refusals. No learned confidence or
  statistical certification is implied by these descriptive metrics.
- Expose the bounded audit through CLI, runtime, HTTP, and a browser report with
  explicit evidence scope. Record source/case/report fingerprints and allow
  integrity checking and deterministic replay.

## Research basis

Primary sources checked on 30 October 2026:

- TAPS, revised 26 October 2026: https://arxiv.org/abs/2610.16318 . Separating
  answering from selection motivates reporting coverage and accepted-only error.
- AbstentionBench, 10 June 2025: https://arxiv.org/abs/2506.09038 . Negative cases
  test unsupported and unanswerable inputs alongside solvable cases.
- Conformal prediction after data-dependent model selection, revised 16 April
  2026: https://arxiv.org/abs/2408.07066 . Panel selection cannot silently inherit
  uncertainty calibration for a different ensemble.
- WorldTest, revised 7 May 2026: https://arxiv.org/abs/2510.19788 . Prediction,
  counterfactual/control queries and actual task outcomes are different evidence.
- Valid Stopping in Adaptive Generator-Verifier Loops, 5 October 2026:
  https://arxiv.org/abs/2610.06432 . A cheap verifier/fingerprint is not a
  ground-truth oracle. This upgrade does not implement its statistical method.

These papers motivate mechanisms and measurement choices. Their results and
guarantees are not Poseidon results. No new model weights or dependencies needed.

## Validation and boundaries

Regression tests must first reproduce the defect. Verify actual adapter tensor
shapes and TidePool observation/action names; adversarial complete hypothesis
checks; malformed/nonfinite numerical inputs; checker mutation rejection and
seed reproducibility; HTTP/CLI/browser behavior; full Python suite; Node harness;
source-package parity. Replay binds the local code and finite generated suite,
not unseen questions, real-world control, novelty, or general intelligence.

## Implementation choices

Reuse the dirty checkout on a dedicated branch to preserve unfinished work.
Implement independent code areas in parallel, review the combined diff, and leave
publication and checkpoint promotion outside this request. Source version identity
will not be treated as a published release.
