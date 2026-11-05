# Core research and lineage

Inspected live on 8 October 2026. These are architectural lessons, not copied
implementations or claims that merging projects transfers their capabilities.
Repository descriptions were checked against the files listed below. Reported
historical experiment results are the repositories' own results, not new
independent replications. The implementation in Poseidon is original.

| Project | Inspected revision | Files inspected | Licensing observed |
|---|---|---|---|
| [NexusSearch](https://github.com/kai9987kai/NexusSearch) | `538edcba595889090946a51ce03f72ac9fb72202` | README; `src/engine/nx_search.c`; tree | MIT |
| [prometheus-alpha](https://github.com/kai9987kai/prometheus-alpha) | `417009e5548453338761732152b7b8ec30ffa79a` | README; `src/prometheus/tissue.py`; `src/prometheus/train_v7.py`; tree | MIT |
| [morpheus](https://github.com/kai9987kai/morpheus) | `5db6f98d605b80245ed9ff5e2cc9d42151830a14` | README; `src/morpheus/tissue.py`; `src/morpheus/rsi.py`; tree | MIT |
| [GhostInTheMachine](https://github.com/kai9987kai/GhostInTheMachine) | `239f5de6452f78f3960de07c393268eeac142d03` | README; `src/ghost_v7/engine.py`; tree | MIT |
| [supermix-archimedes](https://github.com/kai9987kai/supermix-archimedes) | `1952cf1c92cb506fff60f9ee0c055c2365988e5e` | README; `archimedes/src/archimedes_core.py`; `models/supermix-v93/README.md`; tree | MIT code; inherited component terms still matter |
| [Supermix-expanse](https://github.com/kai9987kai/Supermix-expanse) | `3fd091e9cc3387b2c17da5ce7514831d77e2af44` | README; `LICENSE-MODEL.md`; tree | MIT code; composite model/data terms |
| [Supermix-Expanse-v2](https://github.com/kai9987kai/Supermix-Expanse-v2) | `33e38a279c86bccdb1e5680f15f193d61a3f71cb` | README; `LICENSE-MODEL.md`; `expanse/src/source_sampler.py`; `expanse/checkpoints/mixture-20260927-v2/REPORT.md`; tree | MIT code; BioMedLM-derived weights/outputs carry BigScience OpenRAIL-M restrictions; Qwen donor Apache-2.0; connectome attribution applies |
| [Supermix](https://github.com/kai9987kai/Supermix) | `6ea1c2cc41e3705b4b899b764a001cec180d155d` | README; `source/mimomix_core.py`; `source/build_omni_corpus.py`; `source/build_native_image_pairs_v36.py`; `docs/V55_MEMORY_AUTHORITY_AND_ANSWER_RECEIPTS.md`; tree | MIT repository; verify individual data/model provenance before importing |

## Mechanisms that inform Poseidon

**NexusSearch.** Its active immutable-snapshot search combines exact numeric
filtering, lexical BM25 ranking, exhaustive cosine similarity and deterministic
reciprocal-rank fusion. It includes no embedding model. Poseidon takes the lesson
of explicit, inspectable retrieval records: recovered knowledge is supporting
context with provenance, not permission to issue instructions. It does not claim
to embed the C engine or its experimental ANN modules.
[Inspected source](https://github.com/kai9987kai/NexusSearch/blob/538edcba595889090946a51ce03f72ac9fb72202/src/engine/nx_search.c).

**Prometheus.** A shared PyTorch cell rule learns synthetic conditioning through
state changes within a lifetime. Its stress trainer pairs turnover, noise and
extra-training controls using the same generated lives. The evidence distinguishes
decodable state from state the regrown controller actually uses. Poseidon adopts
paired simulator seeds and perturbation/control thinking. A recurrent hidden
vector alone does not establish regenerative memory, biological fidelity or
cross-model memory transfer.
[Stress trainer](https://github.com/kai9987kai/prometheus-alpha/blob/417009e5548453338761732152b7b8ec30ffa79a/src/prometheus/train_v7.py).

**Morpheus.** The implemented cell rule is a small NumPy network with a learned
forward model; training derivatives are implemented in NumPy. Its self-edit loop
separates proposer randomness, selection samples, anti-forgetting anchors and an
auditor's held-out suite. Poseidon learns action-conditioned state changes and
preserves a frozen evaluation partition. An accurate next-state head is a narrow
prediction result, not evidence of awareness or beneficial autonomous self-editing.
[Audit loop](https://github.com/kai9987kai/morpheus/blob/5db6f98d605b80245ed9ff5e2cc9d42151830a14/src/morpheus/rsi.py).

**GhostInTheMachine.** Its agent engine separates network, action and authorship
random streams, retains replayable mismatch vectors, and compares authored versus
donor action streams. The project explicitly rejects consciousness interpretations.
Poseidon's dynamics head receives the action actually executed, while its policy
target is stored separately. Simulator comparisons must preserve the same world
seed and report death/survival, not mistake a prediction-error diagnostic for success.
[Agent engine](https://github.com/kai9987kai/GhostInTheMachine/blob/239f5de6452f78f3960de07c393268eeac142d03/src/ghost_v7/engine.py).

**Archimedes.** Its 47.6M-parameter model combines a v93 trunk, grafted v87 experts,
FlyCore and separate v48/v38 branches. Its native image branch is 64 by 64 RGB;
upscaling does not increase native resolution. A committed v93 model card says it
was not trained for conversation and its connectome ablations did not demonstrate
use of those branches. Poseidon therefore tests each output head and makes no
automatic capability claim from architectural connectivity.
[Core](https://github.com/kai9987kai/supermix-archimedes/blob/1952cf1c92cb506fff60f9ee0c055c2365988e5e/archimedes/src/archimedes_core.py),
[v93 limits](https://github.com/kai9987kai/supermix-archimedes/blob/1952cf1c92cb506fff60f9ee0c055c2365988e5e/models/supermix-v93/README.md).

**Expanse and Expanse-v2.** Distillation and grafting added code, biomedical and
connectome data, but loss reductions did not establish broad capabilities. In the
frozen mixture pilot all three compared checkpoints scored 4 of 72 strict final
answers despite a small development-loss gain. The sampler stores RNG state,
row identities and source probabilities to make continuation reproducible.
Poseidon uses a resumable sampler and reports scene accuracy, action agreement,
dynamics error and actual survival separately. No Expanse checkpoint or donor
data is imported into Poseidon's new core.
[Pilot](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/checkpoints/mixture-20260927-v2/REPORT.md),
[sampler](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/src/source_sampler.py),
[model terms](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/LICENSE-MODEL.md).

**Supermix.** Its language core contains hybrid attention, mixtures and latent
thinking machinery, while its corpus generator documents failures caused by
repeated dialogue and narrow arithmetic templates. Multiple phrasings and a
deterministic answer checker address different failure modes. Poseidon combines
many controlled phrasings with semantic-group holdout and keeps exact-math tool
results separate from unaided neural reasoning. A memory record is admitted for
specific uses, rather than promoted to general instruction authority.
[Corpus lessons](https://github.com/kai9987kai/Supermix/blob/6ea1c2cc41e3705b4b899b764a001cec180d155d/source/build_omni_corpus.py),
[memory contract](https://github.com/kai9987kai/Supermix/blob/6ea1c2cc41e3705b4b899b764a001cec180d155d/docs/V55_MEMORY_AUTHORITY_AND_ANSWER_RECEIPTS.md).

## Existing data and model assets, and why they are not silently reused

Archimedes lists 3,743 replay rows and 4,000 Fly rows, with a 7,356/387 train/dev
split in its recorded run. Its corpora cover limited generated science, arithmetic
and code tracing. Expanse ships teacher corpora and connects to published
Archimedes, Qwen coder and BioMedLM sources, but their composite terms and reported
capability limits travel with derived checkpoints. Supermix contains generators,+native-image branches and experimental adapters; repository presence is not proof
that a particular checkpoint is available, reliable, or the active release.

Poseidon's core instead trains from its own synthetic scene descriptions and
simulator trajectories. A separate pretrained conversational backbone requires
its own exact revision, license, adaptation-data manifest and evaluation receipt.
The resulting application can be useful while still being a multi-component
system. It is not a single foundation model that has learned unrestricted video,
image generation, arbitrary 3D geometry or general reasoning.

## Feasible shared-core experiment

The new TidalCore hashes words and adjacent pairs with stable BLAKE2 features,
projects text and 16 simulation observations into a common hidden space, and
iterates a gated mixture of four learned experts. It has five discrete scene
attribute heads, a six-action controller and an action-conditioned state-delta
head. The scene planner learns which primitives to request; a deterministic
renderer turns that request into pixels, animated frames and mesh files.

Training uses disjoint complete scene tuples and disjoint simulator episode
seeds. Receipts count unique generated examples separately from repeated epoch
exposures. Checkpoints store validated architecture hashes, optimizer state,
sampler order, RNG state and the next batch cursor. Evaluation compares untrained
weights, class-majority action agreement and zero-delta predictions before a
separate paired survival benchmark. These controls test this specific prototype;
they do not yet establish improvement over the earlier Supermix family.
