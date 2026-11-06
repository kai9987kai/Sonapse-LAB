# Integrated planning and learning contracts

The user authorizes substantial research-informed improvement across architecture,
learning, experiments, and the interactive workbench. Routine choices below are
builder decisions. Existing standalone experiments remain available with accurate
evidence labels. No remote publication or dependency changes are required.

## Audit and alternatives

The original connectome/weight-space spec remains in Git. Lost commitments are
behavioral: frozen query learning, complete topology restoration, persistent
downstream use, and measured rather than literal evidence. Adding more isolated
modules would increase this gap. Replacing the whole actor with an untrained
neural world model would discard working controls. Extend the actual executed
transition learner and repair the acting boundaries instead.

## Design

1. Establish one validated five-coordinate action contract: drive, turn, reach,
   height, grip. Validate before world mutation, enforce fatigue on drive after
   planning, and attribute learning to the executed action, including manual input.
   Advance physiology once per real transition. Frozen inference must not update
   learned topology or structural sufficient statistics. Restore graph endpoints
   and optimizer-bound topology together.
2. Add a deterministic, bounded receding-horizon experience planner (horizons
   1..5; default remains 1 and planning remains off). Extend the existing bootstrap
   RLS ensemble with pure member predictions and prequential dynamics diagnostics.
   Search short action sequences near the actor baseline, compare to a repeated
   baseline rollout, propagate member-specific states, penalize disagreement and
   poor support, and abstain when prediction gates fail. Retain conditioning
   coordinates; never advance the real actor/world during imagination. The model
   is lossy, lacks a terminal/value model, and uncertainty is heuristic.
3. Provide a pure plan preview on the current observation and previous executed
   action. Expose horizon, reasons for abstention, candidate returns, uncertainty,
   predicted trajectory, and compute counts through the API and workbench.
4. Run paired actual-world comparisons after shared support with frozen query
   learning. Save raw scores, seeds, model state digests, configuration, source
   fingerprints, and timing. Report regressions and abstentions. Keep synthetic
   delayed-credit tests distinct from embodied performance.
5. Restore meaningful evidence contracts for identified literal capability
   responses and document the older design audit with owning-source references.

## Research and deviations

- PETS (Chua et al., 2018), https://arxiv.org/abs/1805.12114 motivates ensemble
  trajectory propagation. This implementation uses online linear models and
  deterministic bounded search, not PETS probabilistic neural dynamics.
- MBPO (Janner et al., 2019), https://arxiv.org/abs/1906.08253 motivates short
  rollouts and explicit model-use limits. No monotonic-improvement result applies.
- DreamerV3, https://arxiv.org/abs/2301.04104 motivates coupling prediction and
  behavior; this work does not implement Dreamer's latent actor-critic training.
- Plan2Explore, https://arxiv.org/abs/2005.05960 treats ensemble disagreement as an
  exploration signal; here it is a conservative ranking penalty, not calibrated
  probability or an exploration claim.

## Verification

Behavioral regressions must fail before fixes. Check manual attribution, single
physiology update, action coordinates and post-plan governor, frozen structural
state, changed-endpoint replay, delayed reward planning, cold-start/uncertainty
abstention, pure previews, checkpoint compatibility, API validation, and measured
study receipt tampering. Run targeted checks, full pytest, JS syntax, bounded
actual-world studies, and browser verification. Preserve negative results.
