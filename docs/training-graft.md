# Executed-transition neural graft

SYNAPSE can train an additional residual policy while retaining its existing
connectome, donor, plasticity, memory, planning and workbench pathways. The new
policy consumes the current 155 sensory coordinates, the actual 64 graph/donor
features, the five baseline action coordinates and 32 optional fixed-connectome
conditioning coordinates. Its final action head starts at exactly zero. A query
therefore initially reproduces the baseline action, and queries never consume
exploration RNG or update learned state.

The added network has separate actor, value and transition-prediction paths.
The baseline action receives a correction of at most 0.25 per coordinate, then
the existing final action bounds and physiology governor still apply. Learning
uses Gaussian latent exploration followed by a bounded tanh correction. Frozen
queries use the latent mean. The live graph remains an input to the actor; this
does not replace it with an independent controller.

## What is actually trained

`NeuralResidualGraft.observe` stores only a proposal paired with its actual
executed action, next observation, reward and termination flag. Receipts are
single use. Manual actions and proposals changed by a later planner/governor
can train prediction and value, but they do not train the policy. The replay has
a fixed capacity and a separately checkpointed RNG.

Active policy training uses a clipped likelihood-ratio objective over eligible
samples collected with the current policy version. The advantage is the observed
reward plus a terminal-masked next-observation value estimate, minus the current
value estimate. The value path deliberately sees observation only, so bootstrapping
does not require inventing future graph features. It remains a partial-observation
approximation. Policy samples are retired when that policy is updated; arbitrary
old replay does not become fresh on-policy evidence.

The separate auxiliary path predicts 14 visible sensory-summary coordinates and
the actual reward from the current observation and executed action. Value fitting
uses a stopped-gradient TD target. `train_replay(actor=False)` trains only those
two paths: actor weights and deterministic action output remain unchanged.

This is a compact implementation inspired by residual control and clipped policy
optimization. It does not reproduce either paper's full experimental protocol.
Residual action addition follows the decomposition studied in
[Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201).
The clipped recent-policy objective follows
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

## Pretraining and the live world

`pretrain_from_transitions` accepts recorded executed transitions and validates
the complete batch on a candidate before replacing resident graft state.
`pretrain_runtime` clones the runtime, suppresses that clone's active auto-fit,
executes real simulator transitions, and then fits prediction/value heads. It
can transfer only the resulting graft state. The user's live world, episode,
physiology and existing actor weights are not advanced by collection.

Asynchronous callers use `apply=False`, retain the supplied parent graft digest,
and apply the candidate under their runtime lock only if that digest still
matches. Cancellation never applies a partial candidate. Cloned trajectories
are excluded from policy-gradient updates because the cloned legacy actor may
have adapted during collection.

The helper reports recorded transitions, actual optimizer and actor updates,
prediction losses, CPU time and elapsed time. Timing measurements are returned
as observations and are excluded from learned checkpoint state. The checkpoint
includes network tensors, Adam moments, replay, pending receipt, policy version,
fixed-prior identity, training counters and RNG state. Restore keeps the network
and Parameter objects bound to the host's editable module map.

## Fixed connectome conditioning

`bind_connectome(pre, post, weights, n, provenance)` accepts a source-backed
signed sparse graph and records its digest and provenance. Deterministic sensory
drive propagates for two normalized, bounded rate steps and is pooled into 32
conditioning coordinates. These fixed weights are not described as learned.
Their being measured data does not establish that this sensory mapping models
an animal or improves control. Changing the prior invalidates earlier policy
likelihoods. Without a prior these coordinates are zero.

## CPU budget and evidence limits

`configure_torch` applies an explicit process-wide CPU thread budget and reports
the actual PyTorch setting, logical CPU count, build and CUDA availability.
Training uses CPU tensors. Larger batches and repeated useful optimizer updates
can use additional CPU resources; a thread count alone does not establish higher
utilization or greater task performance. Inter-op thread count is left unchanged
because PyTorch does not allow repeatedly resetting it after work starts.

The tests cover bounded/pure queries, action-contingent policy gradients,
intervention exclusion, prediction-loss reduction, frozen-policy pretraining,
exact checkpoint continuation, atomic input rejection and live-world isolation.
The controlled gradient fixture is a functional test. It is not a new benchmark,
an environment score, a general-intelligence result or evidence of real-world
reliability. A useful performance claim still requires matched frozen-policy,
no-pretraining and ablated-prior controls on held-out task seeds.
