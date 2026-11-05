# Trajectory evidence audit: local v0.5.1

Poseidon now independently reconstructs cognitive-map evidence and records real
multi-step action interventions. This makes two previously separate claims
inspectable: whether a map follows the recorded observations, and whether a
chosen action improves an H-step reward trajectory under policy continuation.
The default controller and neural weights remain unchanged.

## Corrected behavior and historical limits

Odyssey's episode-boundary reset previously replaced explicitly selected scarcity
with its configured default. A strict reconstruction of the published
scarcity-4 receipt found **12 Odyssey-arm episodes using map scarcity 2.5**.
The simulated worlds used scarcity 4. The reset now preserves an explicit
setting, and each direct world request initializes the map with that request's
scarcity. Consecutive episodes with different settings start with a fresh map.

The historical Horizon and Odyssey experiment branch audits execute **one step**,
despite their multi-horizon scope label. They compare fixed reserve utility,
while the model's advantage labels use H-step episode rewards. Subtracting those
different units does not establish multi-step prediction-error coverage. Those
receipts remain historical execution evidence; this patch adds separate audits
without rewriting them.

Fitted Horizon returns are **undiscounted** sums; the earlier README's gamma 0.96
claim was incorrect. Calibration also does not reproduce the complete navigation
history or every heuristic bias. Pre-transit, storm and depletion override paths
can bypass the margin gate. No calibration or observed fallback is a safety
guarantee. Waypoint ranking uses a discovered graph, but the six macro actions do
not execute a specified BFS route to a target patch.

## Strict parent verification

`verify-evidence` accepts a Horizon or Odyssey experiment receipt together with
the matching four fitted sidecars under `outputs/`. It checks their canonical
checksums, checkpoint/model identities and all recorded fitting partitions. Audit
seeds must be disjoint from every controller's fitting partitions. Embedded fit
metadata must agree with the actual bound artifact.

Each expected controller/seed pair must occur exactly once in episodes and
outcome rows. The verifier replays observations, executed actions, state telemetry,
terminal snapshots and flags, six one-step branches, forecast squared errors and
aggregate results. Finite nonnegative timing values and bounded trace lengths
are required.

For Odyssey arms it reconstructs `CognitiveMap` from observations and actions,
rechecking patch counts, edges, resource estimates, current-patch metadata and
the selected waypoint. Historical map-setting mismatches are reported explicitly
instead of being treated as measurements of the corrected controller. This
verifies observation-derived arithmetic, not that the recorded predictions or
continuation actions were authored by a particular neural model.

## Real H-step intervention receipts

`audit-trajectory` samples policy, Horizon and Odyssey arms every 32 steps and at
**every observed override**. The default budget is 256 anchors; exceeding it
rejects the run rather than silently dropping overrides. Each frozen snapshot
branches into all six initial actions. Subsequent steps use the frozen core
policy until the chosen horizon or episode termination.

The audit records every branch action, actual summed reward and terminal outcome.
It compares the executed branch return with the incumbent branch return at the
same snapshot. These are counterfactual H-step returns under policy continuation,
not the return of an entire subsequently adaptive Odyssey trajectory. The chosen
H is explicit; its default 16 matches the fitted label horizon.

`verify-trajectory` reconstructs parent snapshots and all recorded branch actions
without loading neural weights. It rejects altered returns, missing continuation
actions, duplicate or missing anchors, changed parent identities and modified
summaries. Checksums link the evidence; they do not authenticate authorship.

```powershell
python -m poseidon verify-evidence outputs/odyssey_experiments/RECEIPT.json
python -m poseidon audit-trajectory outputs/odyssey_experiments/RECEIPT.json --horizon 16 --stride 32
python -m poseidon verify-trajectory outputs/trajectory_audits/AUDIT.json --parent outputs/odyssey_experiments/RECEIPT.json
```

Use the emitted content-addressed filename. `--root` can point to a matching
artifact workspace. The 128 MiB input bound and anchor/horizon budgets limit
the local work. No command changes weights, the active-core pointer or planner
defaults.

## Fresh corrected measurements

Two fresh suites used four seeds each, ten controllers and a 64-step horizon.
Seeds `113000001`–`113000004` used scarcity 2.5; `114000001`–`114000004` used
scarcity 4.0. They are disjoint from all four sidecar fitting partitions. There
was no tuning between these suites.

| Measure | Moderate | Scarcity 4 |
| --- | ---: | ---: |
| Paired episodes | 40 | 40 |
| Executed transitions replayed | 2,521 | 2,505 |
| Cognitive-map transitions reconstructed | 768 | 768 |
| Map scarcity mismatches | **0** | **0** |
| Real H-step anchors | 44 | 47 |
| Real H-step branches replayed | 264 | 282 |
| Branch transitions replayed | 4,224 | 4,488 |
| Calibrated Odyssey mean reward delta | 0 | 0 |
| Ungated Odyssey mean reward delta | +0.002331 | +0.041827 |
| Ungated overrides | 4 | 7 |
| Negative H-step override advantages | **2** | **4** |

Ungated overrides had mean actual 16-step return advantages `+0.011907` and
`+0.000255`. Six of eleven individual overrides were adverse on this measure.
Positive mean episode differences therefore do not imply consistently beneficial
interventions. The scarcity-4 episode difference was dominated by one seed
(`+0.195738`); the other differences were `-0.016071`, `0`, and `-0.012360`.
Four seeds provide limited evidence, and no superiority conclusion is made.
Calibrated Odyssey matched policy with zero overrides. Both survived all eight
episodes; survival remains saturated at this short horizon.

In total, the new auditor replayed **91 anchors, 546 real multi-step branches and
8,712 branch transitions**. The complete outcome and source identities are in
[trajectory-evaluation.json](trajectory-evaluation.json). Large receipts are
generated artifacts included in the matching Hugging Face release:

- Moderate parent: `outputs/odyssey_experiments/3ed044592794f58c-29888499b43a.json`.
- Scarcity-4 parent: `outputs/odyssey_experiments/50ce54c9e486ce8f-0fb6374e203a.json`.
- Moderate audit: `outputs/trajectory_audits/8491880ca77a2dba-c7284642c7ab.json`.
- Scarcity-4 audit: `outputs/trajectory_audits/8b0e6f16a03ed340-a22ce43a85ac.json`.

The Python suite passed **243 tests**, including re-signed evidence tampering,
weightless replay and successive world-setting regression cases. Compilation,
JavaScript syntax and whitespace checks passed. Live desktop and mobile browser
checks verified a real 32-step scarcity-4 world, map-setting parity, ten-controller
result rendering, nine paired differences and return-specific decision columns.
Comparison rendering used the actual corrected receipt as a fixture; it was not
a second independent experiment.

The workbench now reads the Horizon/Odyssey result schema correctly. Episode-total
decision time is converted to per-decision time for display; paired deltas and
win/tie/loss counts are populated; absent uncertainty intervals remain absent.
Return candidates display their advantage and local filter instead of empty
one-step state predictions. Mapped-patch counts are visible alongside outcomes.

The **v0.5.1 release profile** binds these corrected receipts, replay tools and
unchanged weights to a clean Git source revision. Package verification checks
all file bytes, exercises the bundled models offline and replays the trajectory
evidence. The audit strengthens the instrument's evidence and exposes remaining
planner weaknesses; it does not promote the candidate.
