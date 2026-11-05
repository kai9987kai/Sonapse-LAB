# Poseidon: world and memory design evidence

Inspected on 2026-10-08. This is a source review of the linked repositories, not a rerun of their experiments. GitHub repository metadata, the listed README files, source files and root LICENSE files were retrieved over HTTPS at the revisions below. No source code, weights, training examples or assets from these projects are copied into Poseidon by this review.

| Project | Inspected revision | Root license |
| --- | --- | --- |
| [3D Animal Simulator / EvoSim](https://github.com/kai9987kai/3d-animal-simulator-Hybrid-Agent) | `b7dd934770380a28717ddc82e3dfa7f127732a75` | Apache-2.0 |
| [GenesisEngine](https://github.com/kai9987kai/GenesisEngine) | `a43099ef21e91b75b812e3eb883cb4f3813ab7c7` | MIT |
| [FLY-DIAMOND-NEXUS](https://github.com/kai9987kai/FLY-DIAMOND-NEXUS) | `0c8d4100b21d8889bacdef1a9c6dfc20b3424242` | MIT |
| [QuantumBot](https://github.com/kai9987kai/QuantumBot) | `42e4fd637da2df1da5241da484ebc2c625050e72` | MIT |
| [Causeway](https://github.com/kai9987kai/Causeway) | `e25ea9e6a5a70f60f5a6ee36756aceb3c6c9baac` | MIT |
| [MOLT](https://github.com/kai9987kai/MOLT) | `8ac04b1839b82d4e4d83166f1938bc6e1150197a` | MIT |
| [Memory Carrier Observatory](https://github.com/kai9987kai/MEMORY-CARRIER-OBSERVATORY) | `523c10b46fa26acd53cd8468f95d3cbf55011437` | MIT |
| [Mnemorph](https://github.com/kai9987kai/Mnemorph) | `4fc3353b4aa4ea3e2c7b24798121ee65deda6939` | MIT |
| [TESSERA](https://github.com/kai9987kai/TESSERA) | `85e740452824a0ae089332e47d3473db879ba70c` | MIT |

License entries describe the inspected root files. Separate dependencies and datasets can have separate licenses. Conceptual attribution does not imply importing a project's implementation or inheriting its measured capability.

## Concrete lessons

### EvoSim

The current README describes a fixed-step ecosystem with complete schema-7 checkpoints, energy-aware feeding, local vegetation renewal, a census and paired intervention rollouts. It explicitly limits ecological interpretation. Source inspection covered `src/lab-core.js`, `src/persistence.js` and `src/experiments.js`: random generators, checkpoint validation and isolated comparisons are separate concerns. Poseidon should likewise separate environment stepping from rendering, validate a complete snapshot before replacing state, and distinguish survival from merely reaching an episode boundary. Pairing a seed alone does not guarantee aligned subsequent random draws after action paths diverge. [Inspected source tree](https://github.com/kai9987kai/3d-animal-simulator-Hybrid-Agent/tree/b7dd934770380a28717ddc82e3dfa7f127732a75/src)

### GenesisEngine

`README.md`, `src/genesis/simulation.py`, `src/genesis/neural.py` and `src/genesis/experiments.py` describe and implement a deterministic developmental toy world. The experiment module hashes canonical manifests, source files and endpoint snapshots, and refuses a result directory that already contains files. It verifies source stability across an experiment. Its documentation distinguishes newborn endpoint state from evaluated cohort outcomes. For Poseidon: preserve episode identity, terminal reason, source/data identity and independent experiment worlds. A learned world predictor should be judged on held-out transitions, separately from policy survival. The project does not establish biological realism or general intelligence. [Experiment implementation](https://github.com/kai9987kai/GenesisEngine/blob/a43099ef21e91b75b812e3eb883cb4f3813ab7c7/src/genesis/experiments.py)

### FLY-DIAMOND-NEXUS

Inspected both root and `DIAMOND-SIM-FLY/README.md`, plus `DIAMOND-SIM-FLY/src/fly-brain-engine.js`, `src/fly-environment.js` and `docs/supermix-bridge.md`. The nested application contains a 22-module extension alongside the earlier 16-module baseline; the root README is not a complete description of that nested version. Shared connections coexist with per-agent state. The bridge documentation distinguishes bounded advice from telemetry availability. Poseidon should retain an explicit learned-controller interface, preserve per-environment state, and never silently substitute a teacher when neural inference fails. These are synthetic modules; neither their names nor a telemetry endpoint establish imported connectome behavior or validated language-model control. [Nested application](https://github.com/kai9987kai/FLY-DIAMOND-NEXUS/tree/0c8d4100b21d8889bacdef1a9c6dfc20b3424242/DIAMOND-SIM-FLY)

### QuantumBot

Inspected `README.md`, `quantumbot/benchmark.py`, `policies.py`, `persistence.py` and `world.py`. The benchmark enumerates every condition/seed pair and retains raw run outcomes. The README reports comparisons with linear, MLP, scripted and random policies and explicitly reports no quantum advantage. The practical lesson is to include a strong simple controller, random control and multiple held-out episode seeds. Additional architecture complexity cannot count as an improvement without a matched measurement. Parameter count, environment steps and compute budget are distinct quantities. Poseidon's survival controller will be classical; quantum terminology adds no verified benefit here. [Benchmark implementation](https://github.com/kai9987kai/QuantumBot/blob/42e4fd637da2df1da5241da484ebc2c625050e72/quantumbot/benchmark.py)

### Causeway

Inspected `README.md` and `src/model.js`. The source's `eventUniform(seed, week, actor, channel)` hashes a named event into a uniform draw. Diverging policy paths therefore continue to share named exogenous weather and agent events. Exported receipts retain the randomization method and per-seed outcomes. Poseidon can independently implement the principle using its own event keys and hashing: weather for `(seed, tick, channel)` and location-specific encounters for `(seed, tick, location, channel)`. Use raw paired differences, not a single average trajectory, to compare controllers. Causeway's watershed is synthetic and its provenance links do not validate the rules. [Event-keyed model](https://github.com/kai9987kai/Causeway/blob/e25ea9e6a5a70f60f5a6ee36756aceb3c6c9baac/src/model.js)

### MOLT

Inspected `README.md` and `src/engine.js`. The two arms share schedules but place traces in different carriers. Head removal deletes one arm's stored traces under an explicit rule. The model reports trace retention, behavior, energy-threshold survival and a Brier diagnostic separately; exported aggregates are not full organism event histories. For Poseidon, distinguish storing a memory from retrieving and successfully using it. Provide explicit carrier deletion and context-transfer operations with provenance. A successful deletion/retention experiment tests this software's memory design; it says nothing about biological memory survival. [Memory ecology engine](https://github.com/kai9987kai/MOLT/blob/8ac04b1839b82d4e4d83166f1938bc6e1150197a/src/engine.js)

### Memory Carrier Observatory

Inspected `README.md` and `src/engine.js`. Its neural/body/habitat/group carriers support individual removal, all sixteen subsets and six transfer contexts. The source computes inclusion-exclusion coefficients per seed before summarizing them. Leave-one-pair-out ranges are influence checks, not new replications. Poseidon's analogous four explicit stores should support reproducible retrieval ablation without modifying the source bank. Carrier names describe software storage provenance, not biological substrates. Retrieval scores must not be represented as calibrated truth probabilities. [Subset analysis implementation](https://github.com/kai9987kai/MEMORY-CARRIER-OBSERVATORY/blob/523c10b46fa26acd53cd8468f95d3cbf55011437/src/engine.js)

### Mnemorph

Inspected `README.md`, `src/core/compiler.js`, `experiment.js`, `receipt.js` and `configs/benchmark-smoke.json`. Compiler inputs reject auditor-only truth fields; the experiment pipeline keeps independent withheld scoring. The README correctly says public replay wiring can still reveal a mechanism, so this is a data-flow separation rather than secrecy. Poseidon should train with teacher labels only in the training pipeline and invoke its neural policy directly in evaluation. Split by episode seed, not shuffled transition row, and preserve raw predictions with labels only in evaluator output. A runnable smoke profile is wiring evidence, not confirmatory research. [Compiler boundary](https://github.com/kai9987kai/Mnemorph/blob/4fc3353b4aa4ea3e2c7b24798121ee65deda6939/src/core/compiler.js)

### TESSERA

Inspected `README.md`, `src/engine.js` and `benchmarks/transfer-suite-v1.json`. Opcode publication checks the finite 4-bit input domain exhaustively; ratified admission requires evidence from at least two independent founder lineages and two task families. Transfer starts fresh seeded populations while preserving only frozen learned vocabulary. For Poseidon, evaluate reusable mechanisms on fresh compositions or environment seeds, and label exact checks by their actual finite domain. A symbolic arithmetic verifier can check a generated answer, but that does not establish neural reasoning across mathematics. TESSERA documents established prior art and makes no global novelty claim. [Finite-domain engine](https://github.com/kai9987kai/TESSERA/blob/85e740452824a0ae089332e47d3473db879ba70c/src/engine.js)

## Poseidon implementation requirements derived from this review

1. A bounded synthetic survival environment with observable needs, resources, weather and threat; survival must depend on action. Save the entire state and random-event identity for exact same-version continuation.
2. Separate generated transition training, frozen seed-based validation, and fresh paired survival evaluation. Compare learned, heuristic and random policies under the same exogenous events. Report scarcity shift separately.
3. No teacher or oracle fallback during learned evaluation. A failed or unavailable trained controller must be reported as unavailable.
4. Four memory carriers with explicit provenance, deterministic retrieval, bounded storage, reversible ablation and transfer. Personal facts enter memory only through an explicit user request in the calling runtime.
5. Keep raw episodes, death reasons, horizon truncation, prediction errors and configuration in machine-readable receipts. A pleasant visualization is not a training receipt.
6. Describe innovation as an experimental integration of these ideas. Synthetic survival, generated media and verified arithmetic are separate capabilities and require separate evidence.
