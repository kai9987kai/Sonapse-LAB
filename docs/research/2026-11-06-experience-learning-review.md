# Experience learning research review

Reviewed 6 November 2026. This review checked primary paper pages and selected
SYNAPSE implementation paths. It is not a reproduction of the papers or an
exhaustive survey. The algorithms below are small CPU experiments, with their
deviations stated explicitly. Their usefulness for this architecture requires
measured intervention controls.

## What the existing implementation established

The pre-change `IntegratedAgent` coupled the proxy graph, frozen Tidal model,
mushroom body, compass, and motor readout, but did not call the standalone
`world_model.py`, `imagination.py`, `multiscale_memory.py`, or `curriculum.py`
modules. The existing world-model training loss fitted latent transitions and
reward only; its uncertainty, sensory decoder, and resource heads were not fitted
by that method. A positive value from its uncertainty head was therefore not
evidence of calibrated epistemic uncertainty.

The dormant expert matrices were absent from the agent's Adam parameter list.
Novelty recruited fixed random experts under `no_grad`, and consolidation copied
their matrices into a native matrix. That operation did not establish learned
expert recruitment. `learning.py` fitted a small labeled cue readout using supplied
support labels; success there did not by itself demonstrate embodied adaptation.

The old curriculum used generated observations and random training rewards. Its
fixed-scenario score combined action components with action-independent random
reward. It could test repeatability of that surrogate, but could not establish
environmental task generalization. A public constant seed and mutable Python
objects also do not make an evaluator cryptographically sealed or tamper-proof.

These findings favor an outcome-attributed learning loop and measured controls
before adding more independent biological modules.

## Primary research and feasible adaptations

| Primary source | Mechanism relevant here | CPU experiment and limits |
| --- | --- | --- |
| [Dohare et al., Loss of plasticity in deep continual learning, Nature, published 21 August 2024](https://www.nature.com/articles/s41586-024-07711-7) | Continual backpropagation selectively reinitializes low-utility mature hidden units while continuing gradient learning. The paper distinguishes loss of plasticity from forgetting old tasks. | A trained `64 -> 32 ReLU -> 5` motor residual could compare ordinary training, matched random replacement, and utility replacement. Resample incoming weights, zero outgoing weights, and reset the affected optimizer moments. This adapter and a simplified EMA contribution utility would be adaptations. This release does not implement continual backpropagation, and untrained expert recruitment is not equivalent to it. |
| [Sun et al., Learning to (Learn at Test Time): RNNs with Expressive Hidden States, submitted 5 July 2024](https://arxiv.org/abs/2407.04620) | A recurrent state can itself be a model whose state transition is a self-supervised gradient update. The paper develops learned inner-loop tasks and efficient TTT layers. | A small writable memory is an affordable experimental direction. A nearest-neighbor store with hand-chosen gates is not a TTT layer and does not inherit the paper's end-to-end learning or language-model results. |
| [Behrouz et al., Titans: Learning to Memorize at Test Time, submitted 31 December 2024](https://arxiv.org/abs/2501.00663) | Neural memory updates combine gradient-based surprise, momentum, and forgetting; retrieval is integrated with the surrounding sequence architecture. Its arXiv identifier begins `2501`, although the primary page records a 2024 submission. | The implemented bounded outcome store adopts only the broad motivation of selective writes and recency. Its error gate is a scalar reward-prediction residual, not the gradient-based surprise of Titans. It has no neural memory optimizer, momentum, outer-loop meta-learning, attention integration, or language-model experiment. It is explicitly not a Titans reproduction. |
| [Chua et al., Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models, 2018](https://arxiv.org/abs/1805.12114) | PETS combines probabilistic dynamics ensembles with trajectory sampling for model-based control. It provides a mature baseline mechanism even though it predates the recent papers. | A small ensemble predicting actual transition deltas and reward can expose disagreement and support short planning. The selected RLS bootstrap approximation is not PETS: it has no deep probabilistic dynamics model or particle trajectory propagation. Ensemble agreement alone can be confidently wrong. |
| [Jeong et al., Reflect-then-Plan: Offline Model-Based Planning through a Doubly Bayesian Lens, submitted 6 June 2025](https://arxiv.org/abs/2506.06261) | RefPlan updates a belief over environment dynamics from observations and incorporates uncertainty into planning through marginalization. | A persistent CPU ensemble can test whether online model adaptation helps after drift. Bootstrap disagreement and covariance leverage are heuristics, not this paper's Bayesian inference procedure, posterior, or calibrated confidence. |
| [Lin et al., TD-M(PC)^2: Improving Temporal Difference MPC Through Policy Constraint, submitted 5 February 2025](https://arxiv.org/abs/2502.03550) | The paper diagnoses value overestimation caused by mismatch between planner-generated data and a learned policy prior, and studies policy regularization. | Keeping the original action among candidates and bounding proposed changes are conservative engineering choices. They are not the paper's policy regularizer or a proof against model exploitation. The chosen one-step system has no SAC-style bootstrapped value head. |
| [Wen et al., Mind the Execution Gap: Action-Semantic Mismatch in World-Model Control, submitted 5 October 2026](https://arxiv.org/abs/2610.06582) | This recent preprint diagnoses mismatches between commanded and applied actions in recurrent dynamics attribution and future planning timelines. | Training records must pair each observed transition with the action actually executed. A future delay/loss stress test would be useful. This citation motivates the attribution contract; it does not establish an asynchronous-control failure or a correction result in SYNAPSE. |

Paper benchmark outcomes do not establish a benefit for the graph/Tidal actor.
The selection prioritizes inspectable mechanisms and CPU feasibility; it is not
a claim that these are universally the newest or best methods.

## Implemented bounded actual-outcome memory

The integrated experience model uses three Poisson-bootstrap recursive
least-squares members, not deep neural dynamics. Its basis is bias, state, action,
state/action products and squared actions. Default forgetting is .995, applied on
every observed outcome even for members with a zero bootstrap draw. A covariance
trace cap of 10000 prevents unexcited directions growing indefinitely. This cap
departs from exact unbounded RLS. Fourteen visible sensor summary coordinates are
predicted; two current graph/donor summaries condition prediction without a claim
to predict their future evolution.

The first native pilot regressed on average. Before the next independent-seed
evaluation, a support-only gate was added: EMA pre-update reward MSE must be below
.95 times the EMA error of a running reward-average predictor. A candidate must
also clear a margin of twice its and the baseline's ensemble reward disagreement
plus reward-RMSE-scaled covariance leverage of the action feature difference.
These constants are engineering choices, not confidence coverage, calibration or
safety guarantees. The controlled probe finds that the gate can suppress useful
adaptation after reversal. The [measured results](../results/2026-11-06-experience-adaptation.md)
retain the original negative pilot, subsequent independent seeds, and this tradeoff.

The planner is explicitly one-step. It changes at most one motor coordinate by
.35 and preserves the original action as a candidate. Predicted sensor deltas are
diagnostics rather than a trained long-horizon value model. None of PETS, RefPlan,
Dreamer, Titans or end-to-end TTT has been reproduced by this implementation.

`SurpriseEpisodicMemory` stores raw finite `(state, executed_action, actual_reward,
write_timestamp)` records. Its default dimensions are 16 state values and 5 action
values; capacity is 128. It stores observed reward, not a model residual that
would become stale as the model changes. The signed pre-update prediction error
is used only for deciding whether to write. There is no RNG.

The metric is

```
d(s,a;si,ai)^2 = mean((tanh(s/state_scale)-tanh(si/state_scale))^2)
                 + mean((tanh(a/action_scale)-tanh(ai/action_scale))^2).
```

State and action groups have equal total weight, regardless of their dimensions.
The default scales are one. Tanh bounds numerical ranges and makes every distance
finite, but compresses large inputs and can alias distinct saturated states. The
representation and metric are engineering assumptions, not learned similarity.

For each real outcome, validate all shapes and finite values before changing
state. Let `e = min(abs(prediction_error), 1e6)`. Compute the write threshold from
the *previous* statistics:

```
threshold = min(1e6, max(error_floor, mean + min(multiplier,1e6)*sqrt(variance)))
mean' = mean + (1-decay)*(e-mean)
variance' = decay*(variance + (1-decay)*(e-mean)^2).
```

The caps stabilize the admission statistics; they do not clip actual rewards.
Defaults are `decay=.95`, `error_floor=.001`, and `multiplier=1`. The low floor
recognizes that native one-step rewards can be small; it remains a scale-sensitive
hyperparameter. Admission occurs for the first eight valid outcome updates, a
previously uncovered key farther than `.45` from every stored key, or `e` strictly
greater than the adaptive threshold. The baseline updates even when a write is
rejected. This gate is not a statistical change-point detector or a calibrated
test; early outliers can temporarily suppress later surprise admissions.

An admitted key within `.10` of its nearest neighbor replaces that record with
the actual new outcome. Otherwise it appends while space remains, then evicts the
oldest timestamp. Nearest-distance and age ties choose the first stored slot.
Replacement discards the previous outcome rather than estimating a stationary
reward distribution. This lets a changed reward overwrite an obsolete one, but
can amplify noisy outcomes. The algorithm cannot infer an unobserved task reversal.

For retrieval, select the four nearest keys and weight them by

```
w_i = exp(-.5*(distance_i/.35)^2) * 2^(-age_i/128).
reward = sum(w_i*actual_reward_i)/sum(w_i)
confidence = max(w_i).
```

Reward averaging rescales values internally to avoid overflow on extreme finite
rewards. Missing or numerically zero support returns zero reward and confidence.
Distance is the nearest-key distance; an empty store returns the finite maximum
metric distance `sqrt(8)`. Confidence lies in `[0,1]`, decays with distance and age,
and is a support weight rather than a probability of correctness. Neighbor count
does not automatically inflate it. Multiple outcomes from one region are not
independent evidence of reliability.

Age counts valid real-outcome updates, including rejected writes. Reading does
not advance this clock or alter learned state, statistics, counters, or records.
Export contains all records, timestamps, admission statistics, counters, and the
configuration in detached JSON-compatible data. Import validates compatibility,
finite arrays, occupancy, timestamps, and counter consistency before installing
anything. Restoring the same checkpoint gives the same subsequent decisions;
there is no hidden random state to reconstruct.

## Selected integration experiment

The design pairs this memory with a small recursive-least-squares bootstrap
ensemble and conservative one-step action selection. Record the *pre-update*
reward error, then fit only completed transitions and write their actual rewards.
The state summary includes sensory and live actor features. Frozen evaluation
must freeze model and memory; recurrent actor state may still advance normally.

This design intentionally does not claim multi-step imagination, calibrated
epistemic or aleatoric uncertainty, a Bayesian posterior, continual backpropagation,
model weight consolidation, physical safety, or general intelligence. A fixed
coordinate adjustment bound does not establish safety. Good transition fit does
not establish improved task reward, and improved reward does not establish that
the connectome wiring caused the improvement.

Shared support trajectories and separate query seeds compare actor-only,
model-only, model-plus-memory, and shuffled-reward controls. Query rewards must
never enter fitting. Retain per-seed/task raw returns and negative paired
differences, runtime cost, configuration/source hashes, and checkpoint/replay
checks. A controlled reward reversal and recurrence stream tests memory recall
and post-feedback adaptation separately from embodied-world performance. Stable
recall of a contextual mapping is different from recognizing an unobserved
regime. The local development pilots are not preregistered confirmation.

## Verification record

Before implementation, `python -m pytest tests/test_surprise_memory.py -q` failed
all 32 new cases because `synapse.surprise_memory` did not exist. After
implementation, the same command passed all 32 in 0.50 seconds. Cases exercise
small-reward action ranking and observed reversal, predictable-write rejection,
novel coverage, bounded eviction, recency and distance support, pure reads,
invalid-input atomicity, JSON checkpoint continuation, malformed-restore atomicity,
configuration validation, and extreme finite numeric inputs. These are functional
contracts, not evidence of better embodied task performance. Integration results
and whole-project verification belong in the measured results report.
