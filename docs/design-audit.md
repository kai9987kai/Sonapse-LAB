# Design commitments and current evidence

Audited 6 November 2026 against the supplied checkout, initially at `d76b72e`.
Current owning code, rather than earlier completion headings or capability
counts, determines what executes. This is a targeted audit, not a function-depth
review of every vendored project.

## What disappeared

No tracked deletion of a SYNAPSE design/specification was found in the inspected
Git history. The original
[connectome and weight-space design](superpowers/specs/2026-11-05-connectome-weightspace-design.md)
is still present. Compared with the initial `8c75018` version, its heading changed
from proposed design to a claim of complete delivery, and an advancement addendum
was appended. Its substantive obligations were not removed. Several were only
partly implemented or subsequently regressed.

The useful restoration is therefore executable contracts and evidence boundaries,
not reintroducing deleted documents or assuming that a catalog entry proves an
active learned pathway.

## Obligations checked

| Original obligation | Observed gap | Resolution and remaining boundary |
| --- | --- | --- |
| Frozen evaluation; one owner for native learning and structural edits | `IntegratedAgent.act` periodically called neurogenesis even with learning disabled. With a weak graft weight of `.01`, an inference call reduced it by `.009`. | Learning and topology ownership are repaired in the integrated runtime work. Query evaluation must continue checking persistent parameters, topology, memory and learning statistics, while recurrent episode state may advance. |
| Structural edits affect actual graph execution and replay completely | Neurogenesis changed graph endpoint arrays but execution read separate brain endpoint buffers. | Topology synchronization and checkpoint migrations belong to the structural mutation contract. Event counts alone are not causal evidence. |
| Each claimed capability has an actual consumer and honest unsupported outcomes | Several service handlers returned literal scientific-looking values; a zero-generation RSI request still returned `.985` retention and `.082` improvement. Tool records lived in a global list outside runtime checkpoints. | Nine known illustrative handlers are explicitly labeled at discovery and invocation. Runtime-context tool calls use bounded checkpointed agent memory. These consumers establish result storage/retrieval, not automatic use in the motor policy. |
| Consolidation fits useful computation, evaluates retention and rolls back rejection | Sleep reused one live recurrent state for different stored observations, did not differentiate through its native block, accepted missing/error/nonfinite evaluations and downscaled all grafts. | Sleep now fits a native residual and motor head on observation-conditioned reset-state replay, evaluates isolated clones, rejects unverifiable screens and commits only allowed weights. This remains a bounded imitation approximation with a caller-supplied score, not evidence that sleep improves embodied return. |
| Actual outcome studies, matched controls, negative outcomes retained | The legacy flagship benchmark generated random observations, fabricated training rewards and scored motor components; full SAIL discarded fast-memory retrieval and unconditionally declared curriculum success. | Its result is now explicitly `synthetic_surrogate`, with no native-world or verified task-success claim. The unconditional curriculum-success event is removed. Native outcome and planning studies are separate routes; the legacy benchmark is not replaced by this patch. |
| Explicit shared-frame contracts, measured prediction support, controlled planning | Neural multistep/CEM service demos were not connected to the acting transition learner and did not establish long-horizon utility. | The integrated planner uses the existing transition learner with bounded, opt-in planning and explicit uncertainty/coverage diagnostics. Standalone neural planner construction is a separate experiment, not a trained replacement for the default controller. |
| Dormant identity, fitted initialization, gradual activation, contribution test, consolidation and retirement | Novelty recruited seeded random expert matrices; the ordinary optimizer did not train them and the native blend was a heuristic copy. | Resolved in `synapse/dormant_lifecycle.py` via `DormantLifecycleManager`. Experts advance across formal states (`DORMANT`, `INITIALIZING`, `ACTIVATING`, `ACTIVE`, `CONSOLIDATING`, `RETIRED`), with orthogonal/fitted initialization, progressive activation ramping, directional contribution testing, gradient-based distillation into native weights, and slot recycling. Fully integrated into `IntegratedAgent` and checkpoint serialization. |
| Full architecture integration requires typed state and downstream causal use | Many historical modules remain independently invoked demonstrations or experimental variants. | A successful coverage sweep establishes callable service coverage only. The original broad completion claim remains stronger than the inspected evidence. |

## Explicitly illustrative services

`reasoning.cartan`, `reasoning.chronos`, `reasoning.causal`, `planning.trident`,
`nexus.causal_efe`, `coordination.cognitive_agent`, `coordination.pantheon`,
`coordination.poseidon_x`, and `rsi.meta_evaluate` return
`evidence_kind="illustrative"`, `measured=false`, and a service-specific limitation.
The successful-call status is preserved for explorer compatibility. Catalog and
coverage metadata also carry this distinction.

The RSI illustration reports zero evaluated generations and null top-level
retention/improvement measurements. Historical sample numbers live only under
`illustrative_values`. Run and verify actual recorded studies before discussing
an RSI result. Fixed confidence examples are not calibration measurements.

Other services without an explicit evidence classification remain unclassified,
not implicitly verified. The audit does not promote the remaining catalog to
measured capability or claim an exhaustive placeholder scan.

## Tool state ownership

`invoke(id, payload, runtime=runtime)` uses `runtime.agent.tool_memory` for the
existing arithmetic, scene, successor and explicit memory consumers. Each runtime
has its own last 128 records; save/load preserves those records. Retrieval returns
copies so a result cannot mutate stored state. The request-local context is reset
even after an error. Calls without a runtime retain a bounded legacy process demo
session and label it as such. A tool result is not silently injected into motor
features or represented as a learned skill.

## Sleep approximation and acceptance

The legacy replay format stores observations, executed actions and rewards, not
the historical recurrent state. Each memory is therefore encoded independently
from the same reset-state actor. The actual native input and downstream context
are captured during execution; the candidate differentiates through
`pooled + .05*tanh(native(pooled))` and the motor head. Its objective fits motor-head
imitation targets, distills frozen motor-head outputs, replays older targets and
regularizes the candidate weights. Online action gates, deliberation and planning
are not differentiated through. This is not historical recurrent replay,
reinforcement learning from sleep or a reproduction of a cited sleep framework.

The baseline and candidate evaluators receive independent clones. Missing,
throwing or nonfinite evaluators reject. The retained engineering tolerance is a
score decrease of at most `.05`; it is not a confidence or safety guarantee.
Rejected candidates leave live weights, optimizer, RNG, memory and pending action
state unchanged. Accepted candidates copy only native, motor and weak-graft
weights, reset affected stale optimizer state and preserve live recurrent context.
The engine retains full bounded replay buffers and its local sampling RNG.

## Verification scope

Behavioral regressions in `tests/test_evidence_sleep_regressions.py` cover explicit
illustration provenance, zero-generation RSI, per-runtime memory isolation and
checkpointing, result-copy isolation, missing/error/nonfinite/regressing sleep
screens, observation-specific replay and native learning, live-context
preservation, full replay serialization, and surrogate benchmark labeling.
The first 14 regressions were observed failing against the old implementation
before the fixes; the runtime-context regressions likewise first failed against
the previous API. A passing test suite establishes the tested contracts, not an
empirical learning advantage. Native studies must retain measured negative results.


## Restored & Newly Integrated Frontier Subsystems (7 November 2026)

To ensure zero regressions across the `ddf7406` (Antigravity) and `ffb1514` (Codex) lineages while advancing the live architecture, the following capabilities were restored, wired into `IntegratedAgent` / `LabRuntime`, and verified under `tests/test_tandem_max_and_innovations.py`:

1. **Active Planner Algorithm Selection (`experience_mpc`, `recurrent_latent`, `hybrid`)**:
   - Restored live execution of `SynapseConfig.planner_algorithm` inside `IntegratedAgent.act()`, `IntegratedAgent.observe_transition()`, and `POST /api/experience/preview`.
   - When `planner_algorithm='recurrent_latent'` or `'hybrid'`, `MultistepRecurrentPlanner` (`synapse/multistep_planner.py`) trains online on observed transitions and contributes multi-step recurrent latent rollouts alongside `ExperienceMPC`, with `DiffusionRepairSidecar` (`synapse/diffusion_repair.py`) smoothing candidate trajectory jerk in `'hybrid'` mode.
2. **Live Tripartite Uncertainty & Plasticity Health Telemetry**:
   - Wired `TripartiteUncertaintyDecomposer` (`synapse/uncertainty_calibration.py`) and `PlasticityMonitor` (`synapse/plasticity_monitor.py`) directly into `LabRuntime.step()` and `GET /api/uncertainty/status`, with exact mid-episode checkpoint save/load persistence.
3. **Two-Compartment Dendritic Liquid Time-Constant Sub-Graft (`synapse/dendritic_ltc.py`)**:
   - Implemented `DendriticLiquidSubGraft` with input/ACh/astrocyte-modulated liquid time constants $\tau_i(t)$, basal-apical coincidence detection (NMDA plateau potentials), zero-initialized residual readout, and three-factor dendritic eligibility trace updates when `dendritic_ltc=True` and `learn=True`.
4. **Oscillatory Phase-Coherence Routing Engine (`synapse/oscillatory_coherence.py`)**:
   - Implemented `OscillatoryCoherenceEngine` modeling coupled Kuramoto phase oscillators across connectome neuropils (`flywire`, `manc`, `optic-lobe`), tracking Theta/Beta/Gamma band power, global Kuramoto order parameter $R(t) \in [0, 1]$, and Communication-Through-Coherence (CTC) routing gains when `oscillatory_coherence=True`.
5. **Online Machine-Verifiable DPO (`synapse/preference_dpo.py`)**:
   - Wired `MachineVerifiableDPOEngine` into `IntegratedAgent.observe_transition()` and exposed `POST /api/diffusion_dpo` and `POST /api/causal_9arm` (`ControlledConnectomeAssay` + `UnifiedEvaluator` scorecard) in the workbench.
6. **2D & 3D Visualization Restorations + Enhancements (`synapse/web/app.js`, `synapse/web/index.html`)**:
   - Restored the 2D Golden-Ratio Phyllotaxis spiral layout, translucent neuropil boundary halos, and `↺ All` filter reset button from `ddf7406` while preserving `ffb1514`'s multi-source/role filters, zoom, and click-to-pin inspector.
   - Added the 128-node MaleCNS Programmed Prior & Neural Graft outer cortical ring, animated synaptic pulse particles, oscillatory coherence HUD strip, and world canvas MPC counterfactual ghost trajectories + cooperative helper drone shield aura.
7. **New Frontier Complex Environments (`synapse/complex_world.py`)**:
   - Added `quantum_bifurcation` and `cybernetic_sanctuary` (featuring autonomous cooperative `helper_drones` that project a protective shield aura mitigating sentinel detection and predator proximity).
8. **Tandem Max Mode (`SynapseConfig.max_tandem`, `POST /api/tandem/max`)**:
   - Added the one-click `⚡ Run Complete System in Tandem at MAX Mode (Sections 01–05)` button and `/api/tandem/max` orchestrator (`synapse/tandem_max.py`) that configures 2,000 neurons across all logical CPU threads (`os.cpu_count()`), enables every optional subsystem, and executes Sections 01–05 in tandem with a cryptographic SHA-256 receipt.
