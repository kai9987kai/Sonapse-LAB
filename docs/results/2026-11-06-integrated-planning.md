# Integrated planning and restored execution contracts

6 November 2026. Local work on `codex/integrated-planning-contracts`, from `d76b72e`.

## What changed

The earlier specs remain in Git. Their missing behavioral commitments are now
covered by regressions: executed manual actions train the actual learner, physiology
advances once, drive/turn are consistent, the final action respects fatigue, frozen
queries preserve structural statistics and parameters, and rewired active grafts
update executing buffers and replay. Legacy growth rollback works again.

The persistent experience ensemble now supports pure, bounded 1–5 step MPC with
member-specific states, fixed neural context, a repeated-baseline comparator,
prequential reward/dynamics gates, and heuristic uncertainty penalties. Horizon 1
retains the earlier action rule and reports its compute. Planning stays off by
default. Untrained predictive/MiMo corrections are opt-in.

The workbench adds horizon controls, forecast inspection, action comparisons,
candidate model returns and explicit abstention reasons. Manual-learning and pure
preview paths were exercised in the browser; static inline-style CSP errors were
removed. Runtime tool memory is isolated and checkpointed. Sleep fits its actual
native/motor path on reset-state replay and rejects missing, failed or nonfinite
retention screens. Known literal capability/CLI fixtures and the legacy flagship
surrogate are explicitly labeled; fabricated realized outcomes were removed.

## Validation and review

Fresh full suite: **322 passed in 117.12 seconds** (starting suite: 228).
Python compilation, JavaScript syntax and diff checks passed. Independent review
found four additional defects; all received observed failing regressions and fixes:
legacy shape rollback, inactive-slot birth reports, nonfinite fallback JSON and
malformed receipt step counts. Root also corrected missing one-step compute counts.

## Measured development pilot

Shared actual-world support: seeds 23/29/31 × forage/push/ecosystem, 96 steps per
unit. Three paired frozen query arms run 48 steps each: 864 support transitions,
1296 query transitions, zero query fitting calls. These seeds were reused during
development; this is not a preregistered held-out confirmation study. The final
receipt verified with no source-drift warnings when the pilot was recorded.
The subsequent 1000-neuron sizing patch changed `agent.py`, `cli.py`, `config.py`,
`runtime.py`, and `server.py`; verification against the later source correctly
reports those differences. The original measurements and fingerprints remain frozen.

actual ComplexWorld transitions; shared exploratory support; frozen paired queries; heuristic short-horizon planner

| Frozen arm | Mean return | Planner use | Model evaluations / step |
|---|---:|---:|---:|
| actor | -0.523362 | 0.000 | 0.0 |
| horizon_1 | -0.461640 | 0.262 | 8.7 |
| horizon_3 | -0.513340 | 0.377 | 96.2 |

Paired return differences from actor:
- horizon_1_minus_actor: 0.061722; descriptive interval [-0.028561039945590463, 0.21861434293002258]
- horizon_3_minus_actor: 0.010021; descriptive interval [-0.19648778592317767, 0.22776726318898405]

finite development pilot; shared task types; no automatic promotion; no general performance claim

Bootstrap units are seed/task runs, not individual transitions. Shared support changes graph and memory; query fitting is disabled. The planner has no terminal/value head, uses lossy sensory summaries, and may abstain. Receipts check integrity and arithmetic, not authenticity.

Both descriptive paired intervals include zero. Horizon 3 uses substantially more
model evaluations and does not establish an advantage over horizon 1 or the actor.
Negative seed/task differences are retained in the receipt and summary. Planner
support gates and defaults were not promoted based on these scores.

## Research grounding and limits

[PETS](https://arxiv.org/abs/1805.12114) motivates ensemble trajectory propagation;
[MBPO](https://arxiv.org/abs/1906.08253) motivates short model rollouts. This CPU
implementation uses online linear dynamics and deterministic bounded search,
not either complete algorithm. It has no terminal/value model and no calibration,
safety, biological fidelity or general learning claim. A synthetic delayed-credit
regression demonstrates a learned three-step dependency separately from world returns.

Dense structural covariance remains a scaling limit. Several older modules remain
standalone experiments; the dormant-expert training/activation/retirement lifecycle
and replacement of the flagship surrogate remain open, as recorded in
[the design audit](../design-audit.md). Old checkpoints cannot recover graph-only
rewires that were never saved. New checkpoints use executing buffers as authority.

[Summary and source fingerprints](2026-11-06-integrated-planning-summary.json).
Full raw receipt and support checkpoints are retained locally in
`outputs/planning-studies/integrated-planning-final-20261106/`.
