# Graft validation continuation — 2026-11-04

The continuation strengthens the approved graft evidence tools and repairs an
age-counter boundary defect. The ordinary neural algorithm, objective, teacher
fixture, growth settings and candidate graph remain unchanged. At this
verification checkpoint, work was local and uncommitted; no publication or
live-run restart had been performed. See [publication notes](publication.md)
for later repository delivery.

## Changes and verification

- `min_age` must fit signed int64. Active ages saturate at its maximum instead
  of wrapping negative. Three tests reproduced the accepted-config overflow,
  incorrect protection and failed checkpoint restoration before repair.
- RL episode rows now record initial/minimum/final target distance, actual planar
  path length, mean absolute drive/turn, mean squared clipped command norm, and
  saturation fraction. Reference, training and frozen evaluations use the same
  reporter-only trace. No information is added to the agent or its objective.
- Paired behavior deltas supplement reward/outcome comparisons and preserve
  historical rows without those fields. Distances use synthetic world units;
  command norm is dimensionless, not physical energy. Closer is not universally
  desirable, particularly during no-go.
- The supervised probe now reports normalized trajectory AUC, the final-window
  mean, and old-target observations. Structural-event MSE is measured immediately
  before and after the edit, after ordinary optimization; it is no longer
  conflated with the optimizer's step-to-step effect.
- Probe manifests declare phase-specific event targets, metric definitions,
  status and start/end fingerprints. The source-training event target is the
  before-switch teacher; recovery events use the after-switch teacher.

Fresh full-suite verification: **143 passed**. After review corrected the
phase-specific metadata wording, the relevant probe suite again passed **22
tests**. Focused tests also cover clipped commands, known forward displacement,
stationary references, paired zero changes, old rows, episode resets, frozen
state/RNG, trajectory integration, tail windows and isolated edit effects.
Independent review found no remaining important algorithm or reporting issue.

## Why the earlier flat reward was insufficient

Go/no-go returns are constant over trajectories that never enter the approach
radius. A bounded replay showed parameter and action changes despite identical
returns, with worse closest approach on the go condition. Thus zero return
change did not mean no learning, and it did not establish useful learning.

The earlier supervised endpoint at update80 also coincides with rewiring, whose
error spikes make a single endpoint sensitive to cycle phase. Full trajectories
were preserved previously; they now have explicit primary/secondary summaries.

## Frozen confirmation on 17 declared runs

The [local protocol](graft-credit-confirmation-20261104.json) was hash-frozen
before execution, with [checksum](graft-credit-confirmation-20261104.json.sha256):

```text
bf2cbd07162845f5209383358f1f0b921885402ec36c237c5b54eb044d1ace69
```

An initial, unfrozen draft proposed seeds3–19. Review identified short unit probes
at3/4/5; before any declared80+80 confirmation, the namespace was offset to
**1000003–1000019**, keeping the same count, budgets and analysis. Selection did
not use confirmation outcomes. Exploratory runs0–2 remain separate.

All17 runs used80 source-training and80 recovery updates, the same fixed teacher
switch,12/24 active records, paired sham/replacement branches, and unchanged
gradient/age20 settings. Every declared seed was retained. Computational hashes
remained stable, source states were unchanged, and directional budgets held.

The primary metric was normalized trapezoidal recovery MSE area (updates0..80),
**fixed-support minus adaptive** within each seed; positive favors adaptation.

| Declared metric | Mean paired difference | Positive / negative seeds |
|---|---:|---:|
| Primary: normalized recovery AUC | 0.319617 | 17 / 0 |
| Secondary: final10-update mean | 0.395874 | 17 / 0 |
| Secondary: final update | 0.522163 | 16 / 1 |

The primary mean's95% percentile bootstrap interval is **[0.257920,0.384436]**,
using10,000 resamples of the17 paired confirmation seeds and the declared
bootstrap seed20261104. This uncertainty is conditional on this one fixed teacher
fixture; it does not span other tasks, graphs, shifts or biological systems.
No p-value is claimed.

Seed1000015 had a worse adaptive endpoint by0.002085 MSE despite positive area
and final-window differences. All negative results remain in the raw data and
[analysis](../runs/graft-credit-confirmation-20261104/analysis.json).

Among136 recovery edits in the replacement+adaptive branch,75 immediately
increased current error and61 decreased it; the largest increase was0.618367.
Mean full-trajectory benefit therefore does not mean every edit is beneficial.

Old-target final MSE averaged2.431599 for replacement+adaptation and1.977942 for
replacement+fixed-support. The same inputs receive conflicting old/new labels
without a context cue, so these are fitting tradeoffs—not a general forgetting
or contextual retention test. This probe also does not isolate gradient versus
other recruitment rules or the specific benefit of age20.

Artifacts: [manifest](../runs/graft-credit-confirmation-20261104/manifest.json),
[all trajectories](../runs/graft-credit-confirmation-20261104/curves.json),
[summary](../runs/graft-credit-confirmation-20261104/summary.json),
[confirmation figure](../runs/graft-credit-confirmation-20261104/confirmation.png),
[vector figure](../runs/graft-credit-confirmation-20261104/confirmation.svg).
The [analysis script](analysis/graft_confirmation.py) independently recomputes
metrics from all raw points, verifies seed coverage, budgets and frozen hashes,
and records its own hash with the bootstrap results.

## Longer descriptive RL study: still no task benefit

```powershell
python -m graftbrain benchmark-graft-credit --out runs/graft-credit-behavior-20261104 --train-steps 640 --tasks forage go_nogo --seeds 0 --eval-seeds 1000 --max-neurons 400 --microsteps 1
```

Fourteen arms used7 matched controls across2 tasks, one independent training
seed,640 training steps per arm (8,960 total), and common frozen scenarios. There
were41 learner updates for forage and42 for go/no-go, reaching updates where
first-generation age20 records can become eligible again. This is descriptive
evidence only, not a replicated RL performance estimate.

| Arm | Forage return change | Forage final-distance change | Go/no-go return change |
|---|---:|---:|---:|
| Coactivity, age0 | -0.837118 | +0.837118 | 0 |
| Coactivity, age20 | -0.782176 | +0.782176 | 0 |
| Gradient, age0 | -0.885622 | +0.885622 | 0 |
| Gradient, age20 | -0.858420 | +0.858420 | 0 |
| Random, age0 | -0.828343 | +0.828343 | 0 |
| Random, age20 | -0.780200 | +0.780200 | 0 |
| Routing without rewiring | -1.077067 | +1.077067 | 0 |

Every forage arm moved roughly30 more world units during evaluation and ended
farther from its target; increased movement was not useful progress. Every
go/no-go arm still failed go and succeeded no-go before/after training. On go,
closest approach worsened by about0.321 world units even though return stayed
flat. These observations establish no RL-task advantage for the graft policies.

Gradient+age20 performed320 learned swaps on forage and332 on go/no-go; other
rewiring arms performed358–360. Exact active budgets remained952/1,590 candidate
records. Reduced turnover is an observed mechanism effect, not a performance
gain. No algorithm was retuned to remove the unfavorable task outcomes.

Artifacts: [manifest](../runs/graft-credit-behavior-20261104/manifest.json),
[episodes and behavior](../runs/graft-credit-behavior-20261104/episodes.json),
[summary](../runs/graft-credit-behavior-20261104/summary.json),
[structural events](../runs/graft-credit-behavior-20261104/structural_events.json).
The completed manifest reports unchanged start/end computational hashes.

## Preservation and boundaries

Process4272, started2026-11-03 at23:18:44, continued on port8770 beyond update2073
during validation. No controls, pause or restart were sent. Its own adaptive
checkpoint continues normal autosaves; no experiment writes it. The original
`state/graftbrain.pt` and journal remain at their previously recorded hashes.

Previous verification reports remain historical run records. Their source-match
statements apply to the files verified at those times, not the later continuation.
Full-world exact replay, memory repairs, learner replacement, arbitrary endpoint
growth and general biological/reliability claims remain outside this increment.
