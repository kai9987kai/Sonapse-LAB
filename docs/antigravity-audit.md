# Antigravity integration audit and repaired contracts

Audited the inherited changes at `ddf7406` on 6–7 November 2026. The three recent
commits (`c3e66e8`, `cffa430`, `ddf7406`) are retained. This audit distinguishes
executable integration from stronger claims in their completion reports. It is a
targeted source review, not verification of every vendored research architecture.

The original [developmental design](superpowers/specs/2026-11-05-connectome-weightspace-design.md)
and [integrated planning design](superpowers/specs/2026-11-06-integrated-planning-design.md)
are present. Earlier headings saying all capabilities were complete did not
establish their behavioral obligations. The [design audit](design-audit.md)
remains useful historical context, but its later dormant-lifecycle resolution
paragraph overstated fitted initialization and gradient distillation.

| Inherited addition | What actually executes | Evidence boundary |
| --- | --- | --- |
| Larger workbench graph | Exact configurable total, including 2,000-node default sessions and full node display | More nodes do not establish better behavior. The inherited live actor constructed synthetic profiles, even though a separate loader exposed anatomical data. Source-backed import work must report each source and representation separately. |
| Dormant lifecycle | Explicit states, gradual activation, output-norm monitoring, weight EMA merge and slot recycling | Initialization is unfitted orthogonal random weights. Output norm is not held-out causal utility. EMA merging is not gradient-based distillation. These limitations are now explicit in telemetry. |
| Continual backprop and fast memory | Optional live updates and downstream feature use in `IntegratedAgent` | Updates and callability do not demonstrate retained skills or an adaptation advantage. |
| Efference self-model | Online sensory-delta prediction using the executed action | Authorship, disturbance and slip are heuristic scores. A low discrepancy alone does not identify causal agency. The old control test trained only the active comparator, so it was not a matched trained-model control study. |
| Tripartite uncertainty | Trainable ensemble/NLL heads and an OOD detector | The inherited Omnibus constructed an untrained ensemble. Online training is now connected to successive real actor features, but held-out calibration and reliable selective prediction remain unestablished. |
| Reusable skills | Candidate action motifs mined from rewarded observed trajectories | Discovery is not validated transfer. No automatic macro execution is claimed. The live library can be inspected via `OmnibusRunner.skill_library` and its summary. |
| Research fabric | Typed observation/action records, audit helpers and integrity hashes | Named adapters are not proof that separate projects cooperate in the motor policy. Hashes bind recorded values; they do not establish measurement validity. |
| Complex environments | Currents, conveyors, elevation forces, predator coordination, searchlights and tripwires in `ComplexWorld` | Unit checks exercise mechanics. End-to-end solvability and trained task success need separate evaluations. Isometric rendering is not a general 3D physics simulator. |
| Omnibus curriculum | One learning actor across configurable task stages, with real-world steps and sleep | The original two-step-per-task report is a smoke run, not evidence of useful learning, retention or calibrated uncertainty. Its retention/rank claims are superseded by the protocol below. |

## Repaired state and evaluation boundaries

Frozen inference no longer ages, recruits, activates or updates contribution
statistics in dormant slots. Learning calls retain the original lifecycle.
Self-model checkpoints now include optimizer moments, control RNG, action history,
recent receipts and the authorship threshold, so continued learning and telemetry
can replay exactly. Dormant slot recycling uses its own checkpointed Torch RNG.
Construction no longer resets the process-wide Torch RNG.
Legacy self-model checkpoints still load, but their omitted optimizer/telemetry
history cannot be reconstructed.

Macro-library checkpoints now retain the discovery window, settings, metadata and
RNG. A nonempty Omnibus library no longer crashes the status endpoint by iterating
dictionary keys as skill objects. Candidate macro metadata records observed
trajectory evidence and explicitly leaves transfer unverified.

Before ten representation samples, rank is unavailable (`null`, `WARMUP`), rather
than reporting an invented full-rank value of 64. Measured rank is computed from
the observed centered activity window once there are enough samples. The monitor
recommends interventions; it does not itself execute those interventions.

Omnibus now advances synthetic physiology once for each executed transition. It
trains uncertainty only when a subsequent genuine neural feature supplies the
target for the previous state/action/reward tuple. Episode and stage boundaries
discard pending pairs. No identity target or speculative mutation of the live
actor is used. These auxiliary predictions remain uncalibrated.

Every baseline, post-stage and sleep-candidate probe uses independent actor clones,
the same explicit per-task/per-episode world seeds, the same maximum step budget,
reset recurrent state and reset synthetic physiology. Probe outcomes do not
update the live learner. The sleep evaluator actually executes the candidate in
the world; the inherited lambda returned cached training scores regardless of the
candidate. Its engineering acceptance tolerance remains a mean return decrease of
at most 0.05 across seen tasks, not a statistical or per-task preservation guarantee.

Receipts retain the exact configuration, evaluation protocol/seeds, before-stage
scores, post-acquisition scores, task score matrix, final scores, sleep screens,
uncertainty update count and evaluation work. The entire Omnibus receipt payload
is hash-bound. Fabric hashes now also bind adaptation speed and schema version.

The descriptive backward-retention metric is one minus the clipped mean positive
return drop, normalized by `max(1, abs(acquisition return))`, over tasks before the
last stage. The raw matrix is the primary result. Signed task returns are not
comparable using an arbitrary `+10` offset or by comparing a two-step training
fragment with a thirty-step probe. This metric is not a claim of general lifelong
learning. Success rates are unavailable when no full episode completes; positive
partial reward is not silently counted as task success.

## Study corrections and remaining limits

The inherited `ControlledSixArmStudy` used arm-specific training seeds, duplicated
continual-backprop tracking and called sleep without an evaluator. The four-arm
flagship also used arm-specific seeds and unscreened sleep. Both now share training
seeds across arms, evaluate isolated candidates on phase-matched query episodes,
advance physiology and run actual candidate-aware sleep screens. The six-arm
study enables structural remodeling only in evolution/composed arms through the
actor's explicit `structural_learning` switch. CBP updates belong to the integrated
agent, once per observed transition. Its optional uncertainty sidecar now predicts
actual next observation channels, rather than a copied current latent. The
four-arm full bundle enables the actor's real downstream fast-memory pathway.

All arms retain their names for compatibility. The `online_rl` arm is inherited
reward-modulated Hebbian learning; `fixed_gradient` is reward-scaled self-imitation.
Neither should be presented as a standard policy-gradient RL baseline. Other
inherited actor learning mechanisms remain common within the learning bundles.
The world-transition budgets match; optimizer and sleep computation differ and
are reported separately. Negative transfer is unavailable without an isolated-task
control. The legacy `adaptation_speed` field is mean post-stage return, not time to
mastery. These development studies require frozen confirmation before any benefit
claim. Synthetic flagship mode remains explicitly a surrogate.

The older `BoundedRSIEngine` in `rsi_verifier.py` simulates unit-test outcomes and
support/query deltas with random draws. Receipts and summaries now explicitly carry
`simulated=true`, `measured=false` and `evidence_kind="illustrative"`. Stale-parent
patches and invalid/nonfinite input are rejected; receipt hashes bind the stage
records as well as headline demo scores. Its LCB and promotion outputs remain
demo values, not measured world evidence. Actual numeric patch screens belong to the
separate live programming path. Neither repeated weight edits nor bounded search
establish improvement of an inherited improvement mechanism.

The inherited anatomical artifact is a maleCNS **cell-type aggregate** with 11,751
type nodes and 3,830,931 type-level edges. Homology flags in that archive do not
turn it into independent FlyWire, MANC or optic-lobe exports. Catalog availability,
installed full sources, sampled runtime graphs and modelling assumptions must be
reported independently. Eight catalog entries do not mean eight installed full
neuron-level connectomes.

## Verification

The initial nine contract regressions were observed failing before repairs;
additional tests reproduced the missing matched evaluator and untrained Omnibus
uncertainty path. The initial focused suite passed **36 tests** covering the new
regressions plus existing advanced-frontier, self-model, Omnibus, macro-library,
fabric and plasticity tests. A final focused run passed **32 tests** including the
expanded study/RSI regressions, private slot-recycling RNG and exact advanced-feature
checkpoint continuation. A separate runtime/Omnibus contract run passed **42 tests**.
These checks establish the tested state/evaluation contracts. They do not establish
a behavioral advantage, biological fidelity,
general intelligence or a useful pretrained integrated policy. The final release
must additionally pass the consolidated project suite and browser/API checks.
