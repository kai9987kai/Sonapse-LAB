# ORACLE: episode-trained, calibrated world-model candidate

ORACLE is Poseidon's first separately trained predictive model artifact. It is
an opt-in experiment; the Tidal policy, default planner, and v1.3.0 checkpoint
remain untouched. It predicts a TidePool observation delta, realized reward,
and a binary death event from the current observation and executed action. It
does not select actions or establish a control improvement.

## Why this experiment is structured this way

The earlier `WorldModelEnsemble` is useful as a planner prior, but its default
members are initialized/warm-started predictors rather than a demonstrated,
held-out trained ensemble. ORACLE starts independent members, samples complete
training episodes with replacement for each member, and never moves transitions
between partitions. This measures learning from observed trajectories while
preserving episode identity.

The run has three fit-time partitions:

| Partition | Use |
| --- | --- |
| Train | Optimize model weights using episode-bootstrap samples. |
| Selection | Choose the saved update by held-out predictive likelihood. |
| Calibration | Set separate state and reward interval widths from one maximum standardized-error score per complete episode. |

Final evaluation is a separate command and uses new seed namespaces. The
in-distribution (ID) condition uses scarcity 1.25. The shift condition uses
scarcity 2.5, outside the training scarcity cycle. Both conditions compare the
same model predictions to train-only persistence/state and action-mean/reward
baselines on identical observed transitions. Episode-paired bootstrap
intervals describe uncertainty across those sampled episodes.

The conformal scores are maxima over every transition and target dimension in
an episode. The reported coverage is therefore **whole-episode coverage**,
not per-feature marginal coverage. It is calibrated only against the ID
calibration distribution. OOD coverage is diagnostic and is not guaranteed.
The binary death output has a Brier score but is not probability-calibrated.

This follows two durable benchmark lessons: generalization needs genuinely
separate environment instances, and an in-distribution score alone can hide
overfitting. Procgen formalized train/test level separation across procedural
environment diversity; TD-MPC2 is a relevant example of jointly trained latent
dynamics, reward, and value components at much larger scale. ORACLE is a small
CPU-local supervised dynamics experiment, not an implementation or reproduction
of TD-MPC2. See [Procgen](https://cdn.openai.com/procgen.pdf) and
[TD-MPC2](https://arxiv.org/abs/2310.16828).

## Run

From the repository root, fit a new candidate, evaluate it on fresh seeds, and
replay both receipts:

```powershell
python -m poseidon.cli oracle-fit --oracle-train-episodes 32 --oracle-selection-episodes 8 --oracle-calibration-episodes 12 --oracle-updates 256
python -m poseidon.cli oracle-evaluate outputs/oracle-fit-442000000/oracle.pt --eval-episodes 100
python -m poseidon.cli verify-oracle-fit outputs/oracle-fit-442000000/fit_receipt.json
python -m poseidon.cli verify-oracle-evaluation outputs/oracle-eval-442000000/evaluation_receipt.json
```

Use a different `--seed` and `--output` for a new run. Fit/evaluation commands
refuse nonempty output directories so previous artifacts and evidence remain
intact. A small `--eval-episodes` value is a wiring smoke test, not a reliable
performance estimate. The recommended 100-episode run is still only evidence
about this registered TidePool task.

`fit_receipt.json` records exact fit seed sets, deterministic transition
fingerprints, source hashes, bootstrap draws, the selected update, and
calibration scores. `evaluation_receipt.json` records the new test seed sets,
transition fingerprints, paired per-episode metrics, bootstrap intervals,
source and artifact hashes, and limitations. The verifier regenerates episodes
and recomputes the recorded evidence. Neither receipt authorizes promotion.

## Interpretation limits

- This is a deterministic synthetic survival world; new seeds are unseen
  episodes, not a new environment family.
- Scarcity 2.5 is one controlled shift, not broad OOD coverage.
- One-step prediction does not establish multi-step rollout fidelity, planning
  value, better survival, or transfer to Unity or physical systems.
- Predictions over executed teacher-plus-exploration actions measure that data
  distribution. They do not imply uniform accuracy over every legal action in
  every state.
- A held-out metric is exploratory until the run is preregistered, repeated
  with larger independent seed sets, and confirmed without selecting settings
  on those final test episodes.

The next research step after a successful ORACLE fit is a separately frozen
candidate-planning experiment: use only the trained model's reward and
transition outputs, compare a fixed planner with the incumbent policy on
paired fresh episodes, and count overrides, adverse interventions, compute,
and survival. That control experiment must have its own untouched evaluation
seeds; ORACLE's prediction test does not substitute for it.
