# Runtime memory reconsolidation integration

This implementation connects the existing reconsolidation engine to episodic records that the experience planner actually scores. It is an engineering integration, not evidence that reconsolidation improves task performance.

## Runtime path

1. The planner records the selected first action's nearest episodic slot identity and timestamp, along with the distance-weighted neighborhood reward estimate and confidence used for scoring. In multi-step planning, this attribution comes from the selected path's first simulated step; memory use at later imagined steps is not treated as a live recall.
2. After the world returns an outcome, runtime supplies the executed action. A materially changed action, replaced slot, or out-of-radius context prevents a revision.
3. The usual episodic-memory admission rule runs first. If it admits the observed outcome, ordinary memory handling remains authoritative. If it rejects the outcome, a moderate scaled discrepancy can open or update a bounded labile trace. The discrepancy mapping `abs(error) / (1 + abs(error))` is a heuristic, not a calibrated probability.
4. When that trace is eligible, the nearest recalled slot can accept the exact observed state, action, and reward in place of the rejected write. Historical outcomes are not blended into synthetic reward values.
5. Labile traces are bounded, decay on learned transitions, and persist through agent checkpoints. The runtime status endpoint reports these agent-owned traces. An explicit request with `trace_id` and `prediction_error` remains a separate synthetic probe.

The feature requires the existing experience planner and surprise memory settings. It does not introduce a new always-on learning path.

## Operator-facing behavior

`GET /api/ecology` now reads a snapshot. `POST /api/ecology/step` advances the virtual ecology, so dashboard polling does not mutate ecology state. The dashboard labels STDP, allostasis, planner memory, and synthetic probes according to their actual subsystem scope.

## Evidence status

No performance advantage is established by this change. The match radius, confidence threshold, error rescaling, destabilization/extinction thresholds, and action-change gate are bounded engineering heuristics. A controlled comparison would be needed before making a learning or biological-fidelity claim.
