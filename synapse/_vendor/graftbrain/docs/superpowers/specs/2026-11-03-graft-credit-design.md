# Learning-informed graft growth and connection maturation

Status: approved by the user on 2026-11-04. User selected deeper adaptive graft
routing and structural plasticity for the research-led upgrade.

## Intent and approach

Improve the evidence used to recruit dormant graft connections, and give newly
activated connections time to integrate. Keep the fixed candidate pool, frozen
base anatomy, existing actor-critic, local CPU runtime, and exact directional
budgets. The live process on port 8770 must continue without pause or restart;
tests/experiments use new output paths and never write its checkpoint.

Considered approaches: further coactivity tuning is cheap but does not use the
learning objective; arbitrary endpoint/neuron growth would expand the anatomical
and checkpoint problem. Recommended: optional loss-gradient growth plus a
minimum connection age, compared against coactivity and seeded random growth.
This is an adaptation of prior methods, not a new sparse-learning algorithm.

## Research basis, verified through 2026-11-03

- Evci et al., RigL, ICML 2020: gradient-informed recruitment with zero-initialized
  growth motivates a gradient selection control. Our existing coactivity pruning
  and recurrent actor-critic differ from the published implementation.
  https://proceedings.mlr.press/v119/evci20a.html
- Lillo and Cheney, *On the Stability of Growth in Structural Plasticity*,
  preprint v3, 2026-10-03: newly inserted units can receive inadequate learning
  signal. Edge residency and cohort diagnostics are proposed adaptations; the
  paper's unit/channel results do not validate them for this graph.
  https://arxiv.org/abs/2605.15435v3
- Ma et al., 2025-10-14 preprint: sparse-training effects differ between RL
  modules. Restrict structural changes to artificial grafts and retain matched
  controls rather than applying resets across the ANN and biological graph.
  https://arxiv.org/abs/2510.12096v1
- Patterson et al., JMLR 2024: controlled empirical RL methodology motivates
  paired pre/post measurements, independent training seeds, and frozen policies.
  https://jmlr.org/papers/v25/23-0183.html

Recent connectome-controller work was reviewed as context (FlyGM, 2026,
https://arxiv.org/abs/2602.17997v3). Its simulated biomechanical results are not
evidence that this synthetic-proxy lab has biological fidelity or task benefits.

## Mechanism

Extend `PlasticityConfig` with `growth_rule={coactivity,gradient,random}`,
`growth_init={original,zero}`, and nonnegative integer `min_age`.
Defaults remain coactivity/original/0, preserving previous behavior. Research
examples explicitly select gradient/zero/20; interval remains 10 updates.

For each training action, register a hook on its effective edge-weight vector
`effective = weight * mask * site_gain` entering the recurrent current. The
upstream derivative can be nonzero for dormant coordinates. Multiply it by the
action's detached site gain, **without the active mask**, and sum signed
derivatives across the rollout. This estimates sensitivity to a latent raw edge
coefficient. Raw dormant `weight.grad` must remain zero.

Accumulate detached derivatives in one edge-sized pending buffer during backward,
then update an absolute-gradient EMA once per completed optimizer update.
Do not update the EMA inside callbacks, which would make scores depend on
backward traversal order. Collect finite-score coverage and active/newborn versus
incumbent gradient summaries. No retained autograd graph may enter checkpoints
or telemetry. Empty graphs and frozen/no-grad evaluation are safe no-ops.

Gradient growth ranks dormant candidates by that EMA, skips missing/nonpositive
evidence, and uses stable candidate-index ties. Coactivity retains its existing
rule. Random growth uses a private checkpointed RNG and never advances the
global action-sampling stream. All policies share the current pruning utility
`abs(weight) * coactivity` to isolate the growth decision. Selection magnitude
means sensitivity, not predicted improvement in reward or guaranteed loss drop.

Track active connection age in complete learner updates. Initial active records
start mature. Each update advances active ages; inactive ages are zero. Only
records with age >= min_age can be pruned. Newly activated records start at zero.
If insufficient mature records exist, reduce actual swaps rather than violate
the budget or protection. Zero initialization provides a selectable control;
original signed initialization remains the compatibility default. Clear traces
and optimizer moments for every changed record as before.

Save gradient scores/evidence counts, ages, RNG, configuration, cohort diagnostics,
and event counters. Migrate pre-upgrade plasticity checkpoints by filling only
new neutral/default state; never silently change their growth policy. Validate
old candidate identities and budgets before tensor copying. At completed learner
update boundaries, identical subsequent inputs, action/local RNG, and optimizer
state must reproduce the next growth decision after resume. Arbitrary mid-rollout
autosaves remain model-state restarts: pending rollouts/autograd graphs are not
recovered and full-world continuation is not promised.

## Integration and diagnostics

The agent brackets backward with score accumulation/finalization, then performs
optimizers, Hebbian application and the existing post-update structural event.
Support changes remain outside all retained rollout computations. A saved
`rewiring_enabled` flag defaults true and can disable swaps in isolated recovery
controls while allowing ordinary weight/router learning and diagnostics.

CLI adds validated `--graft-growth`, `--graft-growth-init`, `--graft-min-age`.
Resume compares only explicit options. New viewer telemetry shows selection
rule, protected/mature active counts, gradient evidence coverage, newborn versus
incumbent gradient scale, actual turnover, and last event's selection metadata.
Old-runtime snapshots lacking new fields show unavailable diagnostics gracefully.

## Evidence tools

Add a focused `benchmark-graft-credit` command. Matched arms compare coactivity,
gradient, and seeded random growth at min_age 0 and 20, plus a routing-only
no-rewire control. Hold initial graph/support, initialization policy (zero),
training steps, learner rollout, site routing and all other settings fixed.
Default research budget: 384 training steps per arm, rollout 16, small graph.
Expose tasks, training/evaluation seeds, training steps and graph budget.

Choose balanced go/no-go scenarios before running agents; record condition and
scenario identity. Use the same frozen scenarios before and after training and
report paired return changes, task outcomes, and separate go/no-go counts.
Include zero-action references so no-go success cannot masquerade as learning.
Keep incomplete training episode returns separate from completed returns.
Record configuration, exact executed source-file hashes, library versions,
candidate/support digests, structural events, per-seed results and raw episodes.
Refuse existing output directories; do not select seeds by observed reward.

Add a small deterministic supervised graft-current probe with a declared target
and a target switch. Compare learning and error recovery for sham, equal-budget
support disruption with adaptation, and disruption with rewiring disabled.
Interventions use independent RNG, preserve bucket budgets and clear changed
traces/moments. This controlled probe tests the mechanism, not embodied policy
performance. Report negative or inconclusive results without selecting only
successful seeds. Never equate edge movement or gain spread with a task benefit.

## Required validation

- Dormant effective derivatives agree with finite differences on a deterministic
  current/brain fixture while masked raw gradients and dormant signal stay zero.
- Candidate ranking follows a known loss even when coactivity is held constant.
- Zero-growth insertion leaves immediate current unchanged; mature records alone
  are eligible; budgets hold when protection limits turnover.
- Score collection is independent of hook traversal order within floating-point
  tolerance, finite, and absent
  during frozen evaluation; private randomness leaves action RNG unchanged.
- New and pre-upgrade checkpoint restoration; saved next decision and CLI partial
  override semantics; unchanged legacy/coactivity behavior under defaults.
- Source checkpoint/agent is unchanged by reference, pre/post and recovery
  evaluations. An unchanged policy reports zero paired learning change.
- Full existing suite, bounded real research pilot, API/viewer syntax/telemetry,
  and continued health of the existing live process.

## Scope and builder choices

Keep PPO replacement, memory repairs, new datasets, arbitrary endpoint generation,
full-world exact replay, cloud publication and long training campaigns outside
this increment. Routine class names, EMA decay (proposal 0.9), fixture geometry,
output filenames, plotting style and bounded pilot sizes are builder choices.

## Builder review

A fresh independent reviewer found no remaining material planning questions
after clarifying the update-boundary resume guarantee and floating-point
tolerance. This is a design review, not experimental evidence of effectiveness.
