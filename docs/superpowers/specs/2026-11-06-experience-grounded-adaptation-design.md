# Experience-grounded adaptation

The user authorized research-informed architectural improvements, implementation,
and evaluation of SYNAPSE. Routine design choices below are the builder's choices.

## Findings and approaches

The existing CLI evidence command supplies hard-coded scores; its report is not a
measured ablation. The saved RSI pilot is numerically replayable but all final
intact methods reach 1.0 and recursive-minus-fixed is zero. Several other modules
are standalone demonstrations. The acting world model is absent, dormant experts
are untrained, camera fusion is zero and unarchived, and restoration replaces a
parameter without rebinding Adam.

Options considered: add further isolated biological modules; expand the synthetic
RSI search; or connect prediction and memory to actual transitions with measured
controls. Choose the third because it addresses the missing learning/action loop
and the reliability of evidence. Preserve the existing broad exploratory modules,
but do not use their surrogate scores as embodied performance evidence.

## Architecture

1. `experience.py`: a deterministic CPU recursive-least-squares bootstrap ensemble
   predicts encoded next-state deltas and actual reward from state/action features.
   Inputs include sensory summaries and the live graph/Tidal feature summary.
   Predictions expose model disagreement and covariance leverage as heuristics,
   never calibrated probabilities. Track pre-update prediction errors.
2. `surprise_memory.py`: bounded, deterministic, error-gated episodic outcome memory.
   Read similar state/action outcomes without mutation; write only after actual
   outcomes. Retrieve reward with distance/coverage and recency weighting. It stores
   actual rewards rather than stale model residuals. This is inspired by fast
   memory research, not an implementation of Titans or TTT.
3. One-step conservative action planning keeps the original action among candidates,
   limits every coordinate to a 0.35 change, penalizes disagreement and poor
   coverage, and abstains before sufficient support. A one-step horizon is explicit;
   no long-horizon imagination or safety guarantee is claimed.
4. Agent/runtime integration trains only from `(state, executed action, reward,
   next state)` when learning is requested. `learn=False` freezes learned model and
   memory. Checkpoints contain ensemble weights/covariance, bootstrap RNG, memory,
   counters, and pending transition. Planning is experimental and off by default.
5. Repair parameter/optimizer identity during restore. Register vision modules and
   make camera input causally effective; persist visual parameters. Preserve legacy
   checkpoint loading where unambiguous. Add regression tests on live parameters.

## Interfaces

`ExperienceWorldModel(state_dim=16, action_dim=5, ensemble_size=3, seed=7)` provides
`predict(state, actions)`, `update(state, action, next_state, reward, done=False)`,
`plan(state, baseline, memory=None) -> (action, diagnostics)`, `export_state()`,
`import_state(state)`, and `summary()`. Prediction dictionaries contain `reward`,
`delta`, `disagreement`, and `coverage` arrays. `encode_experience_state(obs,
features=None)` returns 16 finite floats. `SurpriseEpisodicMemory` provides
`update(state, action, reward, prediction_error)`, `retrieve(state, actions)` returning
`reward`/`confidence`/`distance` arrays, and export/import/summary methods.

## Experiments and evidence

Replace the CLI evidence demonstration with actual paired, action-dependent native
world runs. Share support trajectories across actor-only, model-only, model+memory,
and shuffled-reward controls. Use separate support/query world seeds, freeze all
learning during query, save raw per-seed/task returns, costs, configuration, hashes,
and checkpoint/replay checks. Bootstrap paired seed/task units, retain negative
differences, validate finite matched arrays, and avoid unsupported p-value claims.
Add a controlled reversal/recurrence stream to test memory and adaptation separately
from embodied task scores. Small experiments are development pilots, not confirmation.

## Constraints and completion

Use current Python/NumPy/PyTorch dependencies and CPU. No paid compute, source vendor
edits, publication or remote push. Work in the supplied checkout on a codex branch.
Run failing behavioral regressions first, targeted checks, full tests, and fresh
bounded experiments. Document research links, exact algorithm deviations, observed
results, limitations, and known surrogate-only legacy assays.

## Evidence-driven revision

The first native development pilot (seeds 7/11/19, forage/push, support64/query64)
found model-minus-actor mean return -0.239506, with descriptive paired interval
[-1.591752, 0.864120]. Memory actions matched model-only actions. Before the next
independent-seed run, add a fixed gate: recent pre-update reward MSE must be below
0.95 times the running-mean predictor's MSE. Candidate reward advantage must exceed
twice the sum of candidate/baseline ensemble disagreement plus reward-RMSE-scaled
covariance leverage of the action feature difference. These are engineering
thresholds, not calibrated confidence guarantees. Planning stays off by default.
The next run uses seeds 23/29/31/37/41 and support128/query96; its different budget
prevents attributing cross-study changes solely to the gate.

Graph/donor summaries are conditioning coordinates, excluded from predicted sensor
dynamics. Members forget on every outcome, including zero-bootstrap draws. A
covariance trace cap of 10000 prevents unexcited directions growing without bound;
this stabilizer departs from exact unbounded RLS.
