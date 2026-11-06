# Experience-grounded adaptation implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans. Track completion below.

**Goal:** Connect learned prediction and surprise memory to the actor and produce measured, replayable evidence.
**Architecture:** A persistent RLS ensemble and bounded episodic memory feed a conservative one-step planner. Runtime supplies actual executed transitions. Paired native worlds test frozen actors after shared support.
**Tech stack:** Python, NumPy, PyTorch, pytest; no new dependencies.
**Spec:** `docs/superpowers/specs/2026-11-06-experience-grounded-adaptation-design.md`.

## Global constraints

CPU only; no vendor modifications, paid compute, publication, or remote push. Planner off by default. Query evaluation never updates model, memory, or backbone. Heuristic uncertainty carries no calibration guarantee. Existing synthetic surrogate scores are not embodied evidence.

## Review focus

- Restore must update the exact live Adam parameters, including after topology changes.
- Frozen query and memory reads must leave learned state unchanged.
- Nonfinite/shape-invalid inputs must fail before mutation.
- Shared support and paired query seeds must not expose query rewards to fitting.
- Hashes must cover raw scores and source/configuration, and report regressions.

## Tasks

- [x] 1. Write regressions for a learned ensemble predicting true action effects, cold-start abstention, deterministic restore, bounded planning and frozen inference; implement `synapse/experience.py`.
- [x] 2. Write regressions for outcome memory recall, surprise admission, bounded capacity, finite validation, pure reads and restore; implement `synapse/surprise_memory.py`.
- [x] 3. Write regressions on optimizer live identities, camera action effects/persistence, learned-state freeze and runtime transition attribution. Integrate in agent/runtime/config and fix checkpoint/vision defects.
- [x] 4. Replace invented evidence with paired recorded native-world experiments and a controlled drift stream; correct statistics and CLI; test tamper detection, frozen scoring, reproducibility and mismatched samples.
- [x] 5. Run full tests and development experiments; review diffs independently, fix concrete findings, update README/research/results docs with measured outcomes and evidence boundaries.

## Execution record

Initial audit: clean `main` at `63a4aca`; new branch `codex/experience-grounded-adaptation`. Numerical replay of the old RSI receipt succeeded (450 rows); it is historical evidence, not a fresh neural experiment. Implementation tasks are delegated by file ownership and the main agent integrates and verifies the whole change.

Final full suite: 198 passed. Two native studies (24 and 40 frozen query records)
and two controlled reversal probes completed. Initial native regression motivated
a support prediction gate; the subsequent independent-seed pilot still did not
establish superiority. Planning remains off by default. Independent review found
and prompted a fix for guard approval reuse with a changed delta/target; tested
digests now bind both. No additional P1/P2 findings remained in the reviewed diff.
Visual browser verification was blocked by the Computer Use URL check; API tests
and JS syntax checks passed. No commits, remote pushes, or publication performed.
