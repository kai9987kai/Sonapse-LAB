# Evidence reliability implementation plan

> **For agentic workers:** Use regression-first development and independent review.

**Goal:** Repair evidence flow and provide an independently checked local reasoning audit.

**Architecture:** Existing Python/CPU runtime with isolated audit helpers and
explicit caller evidence. CLI, HTTP and browser consume the same audit report.

**Tech stack:** Python, pytest, existing PyTorch, standard-library exact arithmetic,
existing browser JavaScript and Node harness.

**Spec:** ../specs/2026-10-30-evidence-reliability-design.md

## Global constraints

- Preserve existing local changes and the frozen Tidal checkpoint.
- No default candidate promotion, new dependencies, or published release claims.
- Fingerprint integrity and replay are separate checks.
- Keep malformed inputs bounded and reject nonfinite numbers.

## Review focus

- Missing evidence must not create an admitted override.
- Batch sizes unequal to ensemble sizes must retain runtime tensor shapes.
- Empty acceptance sets must have undefined selective risk, not zero error.
- An expected number in explanatory prose must not pass a wrong answer.
- Source or report mutation must invalidate the corresponding replay claim.

## Tasks

1. [x] Coordinator: regression tests, explicit evidence, frozen empirical panel,
   complete interface checks, fingerprint verification; focused tests.
2. [x] Adapters: reproduce tensor/semantic defects, repair actual observation and
   action handling, regression checks against owning world/model contracts.
3. [x] Numerical diagnostics: reproduce parameter/geometry/rank validation bugs,
   repair calculations and scope labels; property and rejection checks.
4. [x] Audit module: deterministic generated cases, independent answer checkers,
   correct/refused/wrong and risk/coverage summaries, fingerprints and replay;
   test checker mutations, nonfinite payloads, no-accept cases, seed stability.
5. [x] Integration: CLI/runtime/API/browser audit report with bounded inputs,
   HTTP and Node tests; label the older token benchmark as display smoke and
   return failure exit status on regressions.
6. [x] Review and verify: combined diff, full pytest, Node/browser inspection,
   generated audit runs with distinct seeds, source packaging and documentation.

## Execution record

The user delegated implementation choices. Independent workers own coordinator,
adapters and numerical diagnostics; the primary worker owns the audit and its
integration. No commits include preexisting unfinished work without final review.

Completed: 1,296 Python tests, 46 Node/browser-handler checks, two exact 200-case
audit replays, and parity for all 191 packaged source/asset files. The isolated
wheel also replayed its audit. Final review defects received regressions and
corrections. Full results: `docs/PROJECT_UPGRADE_2026-10-30.md`.
