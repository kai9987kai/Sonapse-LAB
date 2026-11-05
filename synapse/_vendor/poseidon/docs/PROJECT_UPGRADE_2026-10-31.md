# Exact reasoning and research robustness

This local source upgrade repairs exact numeric semantics and adds two bounded
research tools. It preserves the frozen Tidal checkpoint and default control.
The v3.14.0 release packages these software changes without training new weights
or promoting a controller.

## Exact decimals

Python's AST converts decimal tokens to binary floats. Previously Poseidon used
those rounded values to construct rational numbers and could return a wrong
answer with `verified=True`: `9007199254740993.0 - 9007199254740992.0` returned
`0`. Numeric constants now retain their original spelling, so this returns `1`.
The fix applies across arithmetic, equations, systems, polynomials, calculus and
matrices. Scientific notation is supported within the existing numeric budget;
excessive exponents are rejected before large integers are constructed.

## Related-input reasoning checks

The browser's **Reasoning robustness** panel generates pairs across five
families: long-decimal arithmetic, scaled linear equations, determinant row
swaps, gcd symmetry and edit-distance symmetry. Each answer has an independent
numeric or structural reference. Relation consistency and answer correctness are
different measurements: a constant wrong answer can be consistent while failing
every independent check. Jointly correct pairs require both correctness and a
valid relation. Refusals, exceptions and unavailable relations remain separate.

```powershell
python -m poseidon metamorphic-audit --seed 20261031 --samples 32 --output outputs/metamorphic.json
python -m poseidon verify-metamorphic-audit outputs/metamorphic.json
```

One to 32 pairs per family gives 10 to 320 checked answers. The command exits
with code 1 for wrong answers, refusals or exceptions. A saved report binds the
generated cases and source files; verification recomputes the stored answer
checks and can execute the suite again. `--skip-replay` checks the saved content
without executing the tools. Custom Python routers are recorded separately and
cannot replay as the built-in solver. These finite checks concern local exact
tools, not the conversation model or the full Problem Forge catalog.

## Held-out episode screen

The new `poseidon.research.episode_risk` module screens a fixed policy from
supplied complete-episode binary outcomes. The event is **at least one adverse
override during an episode**. Every prespecified episode must be included,
including episodes without overrides. Fit, selection and calibration identifiers
must be disjoint; every declared condition must have complete outcomes.

Create a protocol before collecting audit outcomes:

```python
from poseidon.research.episode_risk import make_episode_risk_protocol

protocol = make_episode_risk_protocol(
    policy_sha256=policy_manifest_hash,
    predictor_sha256=predictor_artifact_hash,
    panel_sha256=selected_panel_hash,
    thresholds={"margin": 0.15, "trust": 0.60},
    required_conditions=["id", "scarcity"],
    fit_episode_ids=fit_ids,
    selection_episode_ids=selection_ids,
    calibration_episode_ids=audit_ids,
    max_episode_risk=0.10,
    confidence=0.95,
)
```

The hashes and identifiers above must describe the actual study. The policy
manifest should include execution settings, environment, episode horizon and
adverse-event evaluator as well as the controller. Declaring different hashes
does not prove that different code or weights were executed. Save a JSON object
with `protocol` and `outcomes_by_condition`; each condition contains rows such as
`{"episode_id": 101, "adverse_event": false}`. These are supplied labels,
not outcomes inferred by Poseidon from model predictions.

```powershell
python -m poseidon episode-risk study.json --output outputs/episode-risk.json
python -m poseidon verify-episode-risk outputs/episode-risk.json
```

The screen computes exact one-sided Clopper-Pearson upper bounds, splitting the
failure probability across declared conditions using Bonferroni. Eligibility
requires every bound to meet the fixed limit. Zero observed events does not mean
zero risk: 10 event-free episodes with one condition and 95% confidence have an
upper bound of about 25.89%. Screening exits 1 when evidence does not meet the
limit. Verification exits 1 for an invalid report, and always recomputes the
statistics even for a valid report whose evidence was insufficient.

The browser's **Held-out episode risk screen** imports the same study JSON,
up to 64 KiB; the CLI permits larger bounded studies. Both browser exports retain
the original server JSON text, preserving float representations and 64-bit
episode IDs for subsequent verification. POST the same inputs to
`/api/research/episode-risk`; use `/api/reasoning/metamorphic-audit` for related
reasoning checks. HTTP rejects duplicate JSON keys and nonfinite numbers.

Bounds assume independent, identically distributed Bernoulli episodes within
each condition and a policy fixed before collection. Content checks do not prove
those assumptions, the freeze time, data origin, execution identity or event
labels. Reusing the audit for candidate search or repeated protocols requires a
separate statistical design. This screen is neither conditional per-action risk
nor live controller calibration, and it does not activate or promote a policy.

## Research basis

Primary research checked on 31 October 2026:

| Source | Applied idea | Boundary |
|---|---|---|
| [LGMT, revised 2 July 2026](https://arxiv.org/abs/2605.23965) | Evaluate relations between transformed problems | Five exact tool families, not its FOL/LLM experiments |
| [Calibration-risk routing, revised 17 October 2026](https://arxiv.org/abs/2610.01001) | Keep fitting, selection and calibration distinct | No world-model refit or policy deployment |
| [Calibrated for Which Policy?](https://arxiv.org/abs/2610.35877) | Calibration scope depends on policy and sampling law | No weighted conformal algorithm or shift guarantee |
| [Learn then Test](https://arxiv.org/abs/2110.01052) | Test a fixed algorithm's risk with held-out outcomes | Classical binary bounds for one protocol, no adaptive policy search |

These sources motivate the design. Their reported results and guarantees are not
Poseidon results. Research breadth or publication recency alone does not make a
method suitable for this local workbench.

## Validation

Regression-first tests reproduced the precision defect, missing interfaces and
export serialization issue. Adversarial checks include constant-wrong answers,
re-fingerprinted mutations, custom-router replay, split overlap, missing
conditions, extreme numbers and duplicate JSON keys. Final integrated results
are recorded below after source edits and independent review.

- Final v3.14.0 `python -m pytest -q`: **1,451 passed in 791.91 seconds**, with native CPU
  thread budgets set to one (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`). No tests were deselected. This was a fresh process
  after all implementation, release-profile and version-assertion fixes were settled.
- The shipped Node workbench harness passed **50 checks**; JavaScript syntax
  validation passed. Independent final review reported no remaining material
  issues after export precision, numeric workload and semantic-metadata fixes.
- A real browser ran the maximum **320-answer / 160-pair** suite: all answers
  and relations passed, with no refusals or exceptions. Its downloaded report
  passed independent content checking and full replay in a fresh CLI process.
- A **synthetic UI fixture** supplied 20 event-free episodes in each of two
  conditions. Its simultaneous 95% upper bounds were approximately **16.84%**,
  correctly failing a 10% limit. The actual browser download recomputed exactly,
  including episode IDs above JavaScript's safe-integer range. This checks the
  software, not a controller's performance or real adverse-event rate.
- The wheel built without resolving dependencies. All **193 packaged source and
  asset files** matched the checkout. A process launched from an isolated wheel
  extraction ran and replayed an additional 20-answer suite.
- The checked local server reported current source, no status errors, and
  available core/language artifacts. The temporary verification server was stopped.

Local proof artifacts and the full test log are under
`outputs/research/robustness-upgrade-20261031/`. They are generated local outputs,
not controller-performance evidence. The follow-up publication request prepares
v3.14.0 using the release procedure, fresh source-bound audits and an isolated
packaged smoke check. Existing v3.13.0 history and artifacts remain preserved.
