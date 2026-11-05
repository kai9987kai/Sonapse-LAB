# Research robustness upgrade

Date: 31 October 2026.

## Intent and authority

The user requested further innovation and improvement across Poseidon based on
new research. The implementer selects this bounded upgrade under the delegated
scope, building on the current CPU-local Python workbench. No checkpoint
promotion, publication, dependency installation or new trained-model claim is
part of this work.

## Findings and approaches

Inspection reproduced a verified-answer defect: decimal constants are rounded
through Python floats before exact rational arithmetic. The generated audit
currently covers small independent cases, without related-input robustness.
Controller calibration identities remain caller declarations; episode evidence
cannot be substituted for per-action risk.

Considered approaches:

1. Train another predictor/controller. Requires a distinct frozen experiment and
   does not repair the observed exact-answer defect.
2. Expand named reasoning domains. Adds breadth without testing robustness of
   the existing tools.
3. Repair decimal semantics, add metamorphic checks, and implement a separately
   scoped held-out episode screen. Selected by the implementer because all three
   yield directly inspectable behavior and useful evidence on a local CPU.

## Design

- Read numeric AST constants from their source lexemes in every exact solver
  path. Preserve long decimal and scientific notation as rational numbers;
  bound excessive numeric input and retain unsupported-input refusals.
- Add five seeded paired-input reasoning families: decimal arithmetic, scaled
  linear equations, determinant row swaps, gcd symmetry and edit symmetry.
  Check each answer independently and separately record relation consistency,
  acceptance, refusal and unexpected errors. Custom routers cannot replay as the
  built-in solver. Saved reports regenerate cases and recompute metrics.
- Add an offline episode risk protocol and report. Bind declared policy,
  predictor, panel and thresholds; require disjoint fitting, selection and
  calibration episode identifiers and complete declared conditions. Compute
  one-sided Clopper-Pearson bounds with Bonferroni allocation across conditions.
  Eligibility requires every bound to meet the predeclared episode risk limit.
  Verifiers recompute from raw binary outcomes. Digests do not authenticate data,
  prove independent sampling or prove that the protocol preceded collection.
- Expose both audits through CLI, runtime, HTTP and the existing workbench.
  Browser reports distinguish independent correctness, relation consistency and
  episode risk. Export reports for verification. Use existing bounded HTTP input.
- Preserve default control, checkpoints, inactive candidates and negative results.
  No statistical episode bound is used as conditional per-step risk.

## Research basis

Primary sources checked on 31 October 2026:

- [LGMT, revised 2 July 2026](https://arxiv.org/abs/2605.23965): related-input
  consistency motivates paired transformations. This is a bounded arithmetic
  tool audit, not a reproduction of its first-order logic or LLM experiments.
- [Calibration-risk routing, revised 17 October 2026](https://arxiv.org/abs/2610.01001):
  distinct fit, selection and calibration partitions motivate explicit contracts.
- [Calibrated for Which Policy?](https://arxiv.org/abs/2610.35877): changed
  deployment policies can change calibration scope even for a fixed predictor.
  This upgrade does not implement weighted conformal transport.
- [Learn then Test](https://arxiv.org/abs/2110.01052): testing a fixed algorithm's
  risk motivates the held-out screening workflow. We implement a single fixed
  protocol and elementary binary bounds, not adaptive policy selection.

## Validation

Regression-first checks for precision, adversarial constant-wrong answers,
mutation/resealing rejection, split overlaps and malformed numeric/JSON input.
Check CLI exit codes, HTTP validation, browser rendering and report export.
Run the full Python suite, shipped Node harness, independent replay and wheel
source/asset parity after all source edits. All results remain finite local
software evidence; binomial claims additionally require the declared sampling
and fixed-policy assumptions, which software cannot establish.
