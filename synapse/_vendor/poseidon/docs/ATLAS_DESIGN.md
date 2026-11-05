# Poseidon Counterfactual Atlas

## Intent and scope

The requested upgrade synthesizes the 24 linked projects into a useful extension of
the existing local, CPU-only Poseidon system. Success means a runnable new memory
and planning mechanism, matched controls, inspectable decisions, and independently
replayable outcomes. It does not require changing or promoting the language/core
weights or asserting global novelty.

## Architecture

1. Fork a TidePool snapshot into all six actions. Store action-conditioned neural
   prediction residuals from these matched interventions in a bounded atlas.
2. Retrieve nearby residuals to correct the existing neural dynamics head. Keep
   record identities and distances visible. Calibrate descriptive error radii on
   disjoint episode seeds, separately for corrected and memory-erased predictions.
3. Plan from observations alone. Use support distance and measured error to decide
   whether to trust a prediction; fall back to the incumbent policy when support
   is inadequate. An override must exceed the incumbent action's optimistic value
   using the challenger's pessimistic value plus a configured margin. Ambiguous
   counterfactual advantages retain the incumbent. The online controller cannot
   inspect simulator snapshots.
4. Compare incumbent policy, neural MPC, atlas, memory-erased atlas and scripted
   heuristic on exactly paired seeds. Preserve actions, terminal states, prediction
   errors and decision evidence; identify source, configuration, checkpoint and
   atlas contents by SHA-256.
5. Export receipts that an independent simulator replay checks without loading a
   model. Display paired differences and descriptive intervals in the workbench.

This combines retrieval provenance (NexusSearch/Supermix), memory interventions
(MOLT/MCO/Mnemorph), event-keyed contrasts (Causeway/TESSERA), and explicit execution
contracts (NexusFlow/REA/Odysseus). It is an engineering synthesis, not a claim that
k-nearest-neighbor residual models or model-predictive control are new methods.

## Implementation and verification

- Atlas module: deterministic fitting, disjoint partitions, safe bounded loading,
  exact checkpoint identity, per-action errors and explicit memory ablation.
- Experiment module: validated bounded specification, frozen source identity,
  raw paired episodes, deterministic bootstrap, replay verifier, atomic exports.
- Integration: CLI fit/experiment/verify commands, opt-in survival planner, local
  HTTP endpoint and comparison UI. Existing policy remains the default.
- Correct legacy promotion checks and the explicit-store recall controls. Mark
  historical Beyond findings that lack matching evidence as superseded.
- Validate with unit/integration tests, fresh bounded real-checkpoint experiments,
  independent replay, browser interaction and a final source diff check.

## Boundaries

Calibration is descriptive on procedurally generated, correlated trajectories;
it provides no distribution-free coverage guarantee for adaptive rollouts. Low
prediction error is not proof of improved survival. Evaluation seeds must not
overlap atlas fitting/calibration seeds. Repeatedly inspecting a frozen suite
turns it into development evidence. Candidates never activate themselves.
