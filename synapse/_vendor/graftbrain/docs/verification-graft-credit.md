# Graft credit and maturation — verification, 2026-11-04

Historical increment record. The [validation continuation](verification-graft-continuation.md)
adds diagnostics and fresh confirmation; source-match statements below refer to
the files verified for these recorded runs, not later source changes.

The approved increment is implemented locally on `codex/adaptive-graft-plasticity`.
At this verification checkpoint, work was uncommitted and no publication,
release or push had been requested. See [publication notes](publication.md) for
later repository delivery.

## What changed

Graft recruitment can now use coactivity, objective-gradient sensitivity, or
private-seeded random selection. The gradient hook observes effective candidate
weights before the mask chain rule, sums signed derivatives over the rollout,
and finalizes one absolute-gradient EMA after backward. Dormant records still
transmit no current and receive zero raw-weight gradient. Scores describe the
combined actor-critic loss, not direct reward attribution.

Regrown records can start at zero and remain protected until a configured number
of learner updates has elapsed. Initial active records start mature. Protection
can reduce actual turnover, while every directional bucket retains its exact
active budget. Changed records' eligibility, pending Hebbian updates and Adam
moments are cleared. Interventions have separate counters and an independent RNG.

The agent saves a `rewiring_enabled` control for isolated fixed-support recovery
comparisons. Checkpoints retain scores, ages, evidence, cohorts and growth RNG.
Previous schema-1 checkpoints migrate only to neutral compatibility defaults:
coactivity recruitment, original signed initialization, minimum age zero. Graph
identity and budgets are validated before loading mutable score/age tensors.
Next-decision replay applies at completed learner-update boundaries under
identical subsequent inputs/RNG; pending rollouts and full worlds are not replayed.

The CLI exposes `--graft-growth`, `--graft-growth-init` and `--graft-min-age`.
Viewer telemetry includes finite evidence coverage, protected/mature records,
newborn/incumbent gradient scales, actual event turnover and selection metadata.
Older backends show unavailable fields rather than fabricated diagnostics.

## Research basis and limits

- [RigL, ICML 2020](https://proceedings.mlr.press/v119/evci20a.html) motivates
  gradient-informed recruitment and zero growth initialization. This lab retains
  different pruning, recurrent dynamics and learning objectives; it is not an
  exact RigL reproduction.
- [Lillo and Cheney, preprint v3, 2026-10-03](https://arxiv.org/abs/2605.15435v3)
  motivates checking integration time and gradient signal. Its experiments
  concern units/channels; our edge-age rule is an engineering hypothesis.
- [Ma et al., 2025 preprint](https://arxiv.org/abs/2510.12096v1) motivates isolating
  module-specific sparse interventions rather than resetting the whole network.
- [Patterson et al., JMLR 2024](https://jmlr.org/papers/v25/23-0183.html) informs
  paired empirical evaluation, frozen policies and independent-seed accounting.

Sources were selected for this CPU lab and checked through 2026-11-03. No claim
of a new sparse-training algorithm, biological fidelity, or generally best
architecture follows from these adaptations.

## Verification

`python -m pytest -q`: **129 passed in 72.68 seconds**.

Coverage includes finite-difference dormant sensitivity, zero dormant raw
gradients/current, signed aggregation before EMA, finite/overflow handling,
loss-based ranking at equal coactivity, private randomness, age protection,
directional budgets, zero initialization, optimizer/trace resets, cohort
diagnostics, schema migration and next decisions, learner/frozen-state behavior,
CLI overrides, balanced scenario selection, matched arms, paired deltas, source
preservation, seed bounds, refusal to overwrite artifacts, and probe replay.

The independent reviewer found two material issues: oversized probe seeds could
claim an output directory before failing, and last-event selection telemetry
was omitted from the viewer. The seed issue was reproduced before repair and
now rejects API/CLI values outside the supported unsigned 64-bit range before
output creation. Viewer event rows now include actual swaps, eligibility,
protection, growth rule and recruitment score.

`node tests/check_viewer_credit.cjs` passed against the actual formatter/DOM text
renderer using a known event, an older snapshot and absent telemetry. JavaScript
syntax, Python compilation and diff checks passed. The reviewer independently
checked a three-microstep brain derivative against finite differences and found
dormant raw gradients remained zero.

An isolated server on port 8774 served gradient-mode HTTP/WebSocket telemetry.
Its captured snapshot had 41 active records, 30 protected and 11 mature, 76
gradient updates and a three-record last event with a finite recruitment score.
[Captured synthetic snapshot](evidence/graft-credit-snapshot.json).
The new browser webview could not attach; new-panel visual verification remains
unavailable. Actual browser inspection of the existing live dashboard and its
site table succeeded without errors. The isolated server was stopped, and only
its temporary model binaries were removed.

## Balanced RL pilot: no observed improvement

```powershell
python -m graftbrain benchmark-graft-credit --out runs/graft-credit-balanced-20261104 --train-steps 352 --tasks go_nogo --seeds 0 --eval-seeds 1000 --max-neurons 400 --microsteps 1
```

Seven arms used the same 800-node proxy graph, 1,590 candidate records, 952
initially active records, model initialization, 16-step rollout, zero growth
initialization, and training budget: 2,464 total environment steps. Coactivity,
gradient and random recruitment were each tested at minimum ages 0/20; the
seventh arm disabled rewiring but retained routing and ordinary learning.

The evaluation anchor was resolved **before policies ran** to go world seed 1002
and no-go world seed 1000. These same scenarios were evaluated before and after
training. Each arm failed the go condition and succeeded on no-go in both
phases: balanced completion was 1/2, mean return 0.43, paired return change zero.
The zero-action reference also failed go and succeeded no-go. No-go success
therefore does not demonstrate learning.

| Controls | Learned swaps per arm | Pre/post mean return change |
|---|---:|---:|
| Coactivity, ages 0 and 20 | 180 | 0 |
| Gradient, ages 0 and 20 | 180 | 0 |
| Random, ages 0 and 20 | 180 | 0 |
| Routing without rewiring | 0 | 0 |

This single-training-seed, short pilot does not establish an RL advantage for
gradient growth or maturation. Evaluation episodes are repeated measurements,
not independent trained agents. No inferential confidence interval is claimed.

Artifacts: [manifest](../runs/graft-credit-balanced-20261104/manifest.json),
[episodes](../runs/graft-credit-balanced-20261104/episodes.json),
[summary](../runs/graft-credit-balanced-20261104/summary.json),
[structural events](../runs/graft-credit-balanced-20261104/structural_events.json).
The manifest reports `source_changed_during_run: false`, with source hashes,
versions, paired initial model/support fingerprints and all seven outcomes.

## Supervised current probe: bounded positive recovery result

```powershell
python -m graftbrain probe-graft-credit --out runs/graft-credit-probe-20261104 --seeds 0 1 2 --train-steps 80 --recovery-steps 80
```

Each declared seed trained on one synthetic current target, then cloned the
trained state into sham, support replacement with adaptation, and the identical
replacement with rewiring disabled. All branches received the same target switch
and recovery inputs. Budgets remained six active records in each direction
(12/24 total); source states were unchanged. The fixed-support branch continued
ordinary weight learning and age/evidence tracking, so its comparison isolates
the availability of structural swaps.

| Seed | Final replacement + adaptation MSE | Final replacement + fixed-support MSE | Final sham + adaptation MSE |
|---|---:|---:|---:|
| 0 | 0.349845 | 0.494824 | 0.399116 |
| 1 | 0.106660 | 0.898947 | 0.280236 |
| 2 | 0.274289 | 0.805518 | 0.306440 |

Adaptive final current error was lower than the paired fixed-support control in
all three declared seeds. Support replacement also helped the switched target
immediately in these seeds; it is not necessarily damage. Rewiring produces
periodic error spikes, visible in the complete recovery trajectories. These
results concern one small hand-declared supervised problem; they are not RL task
performance, general robustness, retention, or biological evidence.

Artifacts: [manifest](../runs/graft-credit-probe-20261104/manifest.json),
[raw trajectories](../runs/graft-credit-probe-20261104/curves.json),
[summary](../runs/graft-credit-probe-20261104/summary.json),
[recovery figure](../runs/graft-credit-probe-20261104/recovery.png),
[vector figure](../runs/graft-credit-probe-20261104/recovery.svg).
Computational source hashes match the final files. The figure was generated
from all recorded recovery points using the included optional matplotlib script.

## Live-run preservation and remaining boundaries

The original process remained PID 4272, started 2026-11-03 23:18:44, on port 8770;
its learning continued past update 658 during validation. It was never paused,
restarted or sent configuration commands. Its loaded backend remains the earlier
coactivity implementation until a separately requested restart/new run.

The original `state/graftbrain.pt` and journal retain these SHA-256 hashes:

```text
BDB74CE2A931D92CF04DA0B0A773F61350AC4CA1CEDB85190580FB4CB91FCFF6  state/graftbrain.pt
A2F89473F0BF854488C0FF2132A78E07146A2D1C4CE7237EE2A80A291D5C456C  state/graftbrain_journal.jsonl
```

The active `state/adaptive-grafts.pt` continues changing through its own normal
autosaves; tests/experiments never write it. PPO replacement, memory-module
repairs, full-world exact replay, arbitrary endpoint growth, and importing new
anatomical data remain outside this selected increment. Candidate-sized tensors
remain computed; no speedup claim is made.

To try the new policy separately while retaining the current run:

```powershell
python -m graftbrain run --fresh --graft-mode adaptive --graft-growth gradient --graft-growth-init zero --graft-min-age 20 --state state/gradient-grafts.pt --port 8771
```
