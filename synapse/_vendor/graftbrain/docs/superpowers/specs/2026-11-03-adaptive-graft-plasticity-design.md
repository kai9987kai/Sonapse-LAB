# Adaptive graft routing and structural plasticity

Status: approved by the user on 2026-11-03; implementation authorized.

## Intent and decisions

The user requested an advance across GraftBrain, then selected new neural
capabilities and specifically **adaptive graft routing + structural plasticity**.
The goal is to let the neural interface change which artificial graft pathways
carry activity and which candidate synapses are active during learning.

Keep Python, PyTorch, the CPU runtime, the existing graph builder, and the local
viewer. These implementation choices and the numerical defaults below are the
builder's proposals. Preserve the currently modified checkpoint and journal;
all verification runs use temporary paths or a new experiment directory.

## Approaches

1. Working memory and reward-driven associative learning: fits the cognitive
   tasks, but the user chose a different lead capability.
2. Predictive world model and surprise-driven exploration: adds an auxiliary
   learning objective, but was not selected.
3. Adaptive graft routing and structural plasticity: selected by the user.
   Implement bounded support adaptation within the existing graft candidate
   pool. This preserves anatomical site constraints and parameter shapes.
   Generating arbitrary new endpoints would require defining new anatomical
   admissibility rules and would complicate checkpoint compatibility; defer it.

This is artificial sparse connectivity adaptation, not biological growth or a
claim to discover new anatomical connections. A mask changes the active graph;
the underlying candidate endpoints remain fixed.

## Existing behavior and constraints

- `graft.py` builds candidate graft edges and sites once. `brain.py` currently
  weights every candidate edge on every forward pass.
- Bridge sites can include both source-to-target directions. Preserve a budget
  separately for each `(site, source connectome, target connectome)` bucket.
- The base connectome buffers remain frozen and byte-for-byte unchanged.
- The present learner takes one gradient step on stored action likelihoods.
  Its policy ratio is identically one, so it is a one-pass actor-critic with GAE,
  rather than demonstrated multi-epoch PPO. This upgrade will describe it
  accurately; replacing the full learner is outside this increment.
- The current overflow path discards a transition, and terminal reward modulation
  is omitted. Repair those paths as prerequisites to meaningful graft learning.
- The existing 90-trial oracle script prints 10/10 for all tasks, but asserts
  nothing. Its webcam metric counts positive return as success even though the
  environment returns `timeout`. Oracle solvability does not establish learning.
- Existing persistence restarts world episodes. This increment promises routing
  and structural state restoration, not exact continuation of the entire world
  and outstanding autograd rollout.

## Neural architecture

Add a focused `graftbrain/plasticity.py` module owned by the agent/brain.

### Context-dependent routing

A small linear router reads the current public observation vector (retina,
proprioception, task indicator, cue, reward, progress), never `World.st` or hidden
answers. It emits one logit per candidate graft site. Initialize its parameters
to zero so the initial site gains are neutral:

`gate_s(observation) = 2 * sigmoid(router(observation)_s)`.

For candidate edge `e`, the effective weight is

`effective_w_e = graft_w_e * active_mask_e * gate_site(e)`.

The router learns from the existing policy/value gradient through recurrent
microsteps. Add a small mean-gate regularizer (`1e-3 * (mean(gate)-1)^2`) to
discourage uniform suppression or amplification. Keep a separate Adam optimizer
for router parameters (`lr=1e-4`) so the old optimizer parameter layout can load.
No graft sites means an empty/no-op router and finite zero routing diagnostics.
Instantiate the router inside an isolated RNG context so adding it does not
change shared network initialization or action sampling for the control arms.
Use one effective-weight helper for forward dynamics, `graft_current`, and flow
telemetry.

Route once per world action and hold that action's gains across its microsteps.
Store routing tensors with the rollout until the update finishes. No persistent
router telemetry may retain an autograd graph after the update.

### Budgeted structural plasticity

For each nonempty directional site bucket with `E_b` candidate edge records,
activate `max(1, min(E_b, round(density * E_b)))` records. Proposed density: 0.6.
Use a local seeded generator and stable candidate-index tie breaking. Duplicate
candidate endpoints remain separate edge records; report record counts as such.

Maintain a detached, unmasked pre/post coactivity EMA for every candidate,
including dormant candidates. Dormant edges must have a path to a nonzero growth
score. The EMA updates once per world action, not once per microstep.

After each complete optimizer update, increment a persistent structural clock.
Every ten updates, after a two-update warmup:

1. Within each bucket, rank active edges by `abs(weight) * coactivity_ema`.
2. Prune `floor(0.1 * active_budget)` edges, bounded by the number of dormant
   candidates. Tiny buckets without sufficient active/dormant records no-op.
3. Grow the same number of previously dormant candidates with highest coactivity.
   Exclude just-pruned records from immediate regrowth in that event.
4. Initialize regrown weights from their original candidate signed magnitudes.
5. Clear eligibility, pending Hebbian updates, and Adam moment tensors for both
   pruned and regrown edge indices. Inactive edges receive no Hebbian update.
6. Assert that each bucket's active budget is unchanged.

No active mask or graft weight changes while its retained rollout autograd
graphs are still required. Apply structural changes only after backpropagation,
both optimizer steps, and pending Hebbian application finish.

The first version continues computing candidate-sized tensors. A sparse active
support is a neural constraint here; it is not evidence of a wall-clock speedup
or lower memory use.

## Modes, integration, and compatibility

Expose `--graft-mode {legacy,static,routing,structural,adaptive}`:

- `legacy`: all candidates active, gains fixed at one; compatibility default.
- `static`: fixed sparse support, gains one; matched-budget baseline.
- `routing`: same initial sparse support, learned site gains; no swaps.
- `structural`: same sparse budget, prune/regrow, gains one.
- `adaptive`: learned routing plus prune/regrow.

Expose density, rewire interval, turnover, and warmup with validated bounds.
`--fresh --graft-mode adaptive --state state/adaptive-grafts.pt` starts an isolated
adaptive run. The dashboard displays the saved mode and authoritative counts.

Save mask, bucket budgets/identity, coactivity EMA, update clock, swap counters,
router weights, its optimizer state, settings, and local generator state.
Validate candidate endpoint/site identity on restore, rather than accepting only
matching tensor lengths. Resume adaptive checkpoints with their saved settings;
reject conflicting explicit settings with an actionable message.

Legacy checkpoints load in `legacy` mode with all edges active and unit gains.
Require a fresh run for changing mode in this increment; never silently sparsify
an existing trained checkpoint. No verification run writes the user's original
checkpoint or journal.

Add telemetry to each existing site row: candidate/active counts, gate, effective
flow, and swap count. Add a compact panel for mode, total active budget, last
rewire update, and routing spread. The candidate topology endpoint must label
candidates and expose active support separately. Use current masked/gated weights
when reporting signal flow.

## Evidence and validation

Add discoverable regression tests for actual neural behavior:

- Legacy mode reproduces the original forward calculation at zero neural noise.
- Dormant edges contribute no current and do not receive weight plasticity.
- Router gradients are finite/nonzero in a controlled graph and learn different
  gains for two observable contexts in a small supervised routing probe.
- Structural events change support, conserve every directional bucket budget,
  keep base anatomy fixed, clear optimizer/traces, and never regrow just-pruned
  candidates immediately. Zero sites, one edge, and fully active buckets no-op.
- Saving/restoring the new module and optimizers reproduces its next structural
  decision and effective graft current given identical observations/rates.
- Old agent state loads without optimizer parameter-group mismatch.
- Learner transition accounting includes rollout-boundary and terminal rewards.
- CLI rejects invalid mode/density/interval/turnover settings; snapshots and the
  browser show candidate versus active connectivity consistently.

Provide a bounded, reproducible `benchmark-plasticity` command. Run five modes
with paired graph/training seeds and the same tasks, environment-step budgets,
and candidate graphs. The four sparse arms share their initial masks. Report
return, actual terminal outcomes, active counts, support turnover, gate spread,
and wall-clock cost. Development seeds and held-out evaluation seeds are listed
in a saved manifest; frozen evaluation performs no learning, Hebbian changes,
RND training, router updates, or structural swaps. Store per-episode JSON data,
configuration, code revision/dirty status, and summaries. Small CPU pilots are
mechanism checks; do not interpret them as proof of general task advantage.
Evaluation also freezes homeostatic bias adaptation and coactivity/eligibility
accumulation, resets episodic neural/compass state between episodes, and uses
policy mean actions. Recurrent firing activity may evolve normally within an
episode. Preserve learned weights, router settings, and structural support.

Update README/PAPER with the new mechanism, exact commands, and evidence limits.
Label the existing unsupported ablation percentages as unverified historical
claims instead of reusing them as support for the new capability.

## Research grounding

Magnitude pruning and growth under a fixed active budget have prior art:

- Evci et al., *Rigging the Lottery: Making All Tickets Winners*, ICML 2020:
  https://proceedings.mlr.press/v119/evci20a.html
- Mocanu et al., *Scalable training of artificial neural networks with adaptive
  sparse connectivity inspired by network science*, Nature Communications 2018:
  https://www.nature.com/articles/s41467-018-04316-3

This proposal uses candidate coactivity rather than RigL's dormant-edge gradient
growth rule. It is an engineering extension to this lab, not a claim of a new
sparse-training algorithm or of demonstrated biological fidelity.

## Scope boundary and builder choices

Include only learning-path repairs needed by the adaptive mechanism, new neural
components, their CLI/runtime/viewer integration, evidence tooling, and docs.
Memory modules, arbitrary endpoint generation, full PPO replacement, full-world
exact resume, new datasets, cloud publication, and long training campaigns are
outside this increment. Routine class names, tensor organization, fixture sizes,
pilot step budgets, and visual styling are left to the builder.

## Independent builder review

A fresh software-engineering reviewer read this design against the existing
source on 2026-11-03 and reported: "nothing important left". This is a design
completeness review, not implementation verification or scientific validation.
