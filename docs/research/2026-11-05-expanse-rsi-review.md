# Expanse source review and recursive-improvement extensions

Reviewed 5 November 2026 for SYNAPSE. This is a source-and-record review, not a fresh reproduction of model results. No model was trained or executed for this review.

## Inspected source

Repository: [Supermix-Expanse-v2](https://github.com/kai9987kai/Supermix-Expanse-v2/tree/33e38a279c86bccdb1e5680f15f193d61a3f71cb), default `main`, pinned revision **`33e38a279c86bccdb1e5680f15f193d61a3f71cb`**. Despite its repository name, this tree contains the v1–v4 model line and later controlled-continuation and input-evoked experiments.

Read the README and repository tree, relevant model/graft/connectome/consolidation/sampler implementations, experimental designs and reports, test inventory, and vendored structural-growth code. Review depth was targeted rather than an exhaustive audit of every execution path. The reported test totals below are historical repository records, not tests rerun here. A local Expanse checkout or checkpoint has not been established by this review; source and recorded artifact availability are sufficient for the design.

## Mechanisms worth adopting

| Mechanism | Actual implementation | Adaptation for SYNAPSE |
|---|---|---|
| Dormant expert growth | `ExpanseSparseMoE` preserves the active routing width when adding inactive slots, avoiding changes caused by top-k tie-breaking. | Typed birth operation: allocate a dormant expert, verify unchanged host behavior, then measure activation and utility. |
| Donor translation | Student/teacher representation maps are fitted by ridge regression on aligned single-token examples; donor neurons are selected, mapped, locally refined and assigned load-matched routing biases. | Architecture-specific transfer operator with held-out fit diagnostics, rejection and distillation fallback. Fitting is training; do not label it direct gradient-free writing. |
| Progressive expert activation | New expert biases are annealed toward calibrated participation targets. | Measure actual traffic and retained-skill effects while opening new pathways, rather than treating existence as use. |
| Input-dependent graph gates | A static gate plus `tanh(W RMSNorm(h))` controls the graph residual; the dynamic projection starts at zero. | Learn which observations require which graph pathways and expose gate intervention controls. |
| Evoked readout | Subtract the same core's separately normalized zero-drive write from its driven write. The implemented mode requires reset state. | Remove a constant-response shortcut in a stateless diagnostic; develop a separately tested persistent-state variant. |
| Causal graph attachment | Token-local sensory input, explicit masks and persistent decode-position handling avoid the old future-token/padding leak. | Test chronology, padding, session state ownership and batched/incremental parity in any language-facing graft. |
| Native latent consolidation | Zero-gated latent MoE blocks learn from a temporary heterogeneous teacher-fusion bank, which is absent from the saved runtime. | Consolidate useful donor/tool computations into the resident model, then remove teachers and ablate the new blocks. |
| Structural growth primitives | Vendored neurogenesis prunes edges, splits modules, creates covariance-selected connections/taps, clones overloaded experts and resets affected optimizer moments. | Reuse as candidate mutation primitives in isolated snapshots, with explicit inheritance and independent evaluation. |
| Controlled sampling and continuation | SourceSampler saves exact sampling state and exposure; incompatible resume contracts are rejected. | Match generations on actual data/token exposures and account for rejected searches, not just nominal training steps. |

Primary source links: [core and routing](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/src/expanse_core.py), [donor fitting](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/src/donor_graft.py), [connectome laboratory](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/src/connectome_lab.py), [consolidation](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/src/consolidation_v2.py), [structural growth](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/supermix-archimedes/archimedes/src/neurogenesis.py), [sampler](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/src/source_sampler.py).

## Findings that change the design

1. **Transfer is an empirical question.** The repository reports weak teacher–student maps, with only one of 32 grafted experts reproducing its source neuron group adequately on held-out tokens. Most acquired knowledge was attributed to distillation. Record donor-specific fidelity and behavioral donor-removal/shuffle controls; a successful post-training model does not by itself establish transferred donor computation. [README](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/README.md).

2. **Added machinery can receive credit for ordinary extra training.** The v2 consolidation gain largely survived zeroing its new blocks. SYNAPSE needs a continuation control matched for optimizer steps/data and a contribution test after consolidation. [v2 design and results](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/V2_DESIGN.md).

3. **A connectome branch can function mainly as an adapter or bias.** Historical CNS tests found most gain survived removing all edges, and real wiring did not outperform degree-preserving rewiring under those protocols. The fresh stricter audit marks H-CNS-004 inconclusive because paired cohort evidence is incomplete; preserve the distinction from its original historical verdict. [CNS results](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/CNS_LAB_RESULTS.md), [later evidence interpretation](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/EVOKED_CNS_DESIGN.md).

4. **Evoked output is promising instrumentation, not a demonstrated performance breakthrough.** In the two-seed pilot, evoked real wiring reached 0.425105 development nats/character and shuffled wiring 0.424951. The real arm slightly regressed on replay; no generated-answer benchmark or model promotion followed. Zero-edge cancellation demonstrates an architectural property rather than biological superiority. [Evoked design and results](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/EVOKED_CNS_DESIGN.md).

5. **Lower language loss is not improved problem-solving.** The matched 80-step continuation pilot left parent, natural and balanced models at 4/72 strict final answers. Its score includes answer-format/completion requirements, although most failures were parsed wrong numbers. Use task outcomes as primary metrics and loss/format diagnostics separately. [Raw-result report](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/expanse/checkpoints/mixture-20260927-v2/REPORT.md).

6. **Growth is not an RSI acceptance mechanism.** Vendored neurogenesis can mutate the live model and flag a witness-loss change; its witness check does not perform rejection or rollback. Defaults disable growth, and the inspected Expanse trainer does not invoke that controller. Reuse operators only after adapting them to SYNAPSE's candidate transaction and evaluation protocol. [Neurogenesis source](https://github.com/kai9987kai/Supermix-Expanse-v2/blob/33e38a279c86bccdb1e5680f15f193d61a3f71cb/supermix-archimedes/archimedes/src/neurogenesis.py).

## Integration constraints

- Expanse's token stream has width 320 and Archimedes-specific MoE structures. Tidal's observation stream has width 128 and a different task/action contract. New semantic and neural interfaces are required; raw tensor averaging is inappropriate.
- The donor implementation supports particular Qwen2/SwiGLU and BioMedLM/GPT-2 transformations. The GELU sigmoid approximation uses a coefficient of 1.702; do not generalize an exact-equivalence claim to arbitrary GELU implementations or nonlinear representations.
- Its learned latent memory table is a trainable key/value parameter store, distinct from persistent episodic memory.
- Its male-CNS graph aggregates **11,751 cell types and 3,830,931 type-level connections**. The repository supplies the aggregate graph and reconstruction procedure. This is a real-data-derived representation, not a neuron-by-neuron simulation of the full biological CNS. Verify file and source receipts before importing it. This adds an anatomical-data experiment alongside GraftBrain's synthetic proxies.
- Preserve tokenizer/context semantics; token-aligned donor fitting is not directly applicable to motor-control observations. Use paired sensorimotor functional probes when transferring to SYNAPSE's control graph.
- Mutable forward-hook context and cached graph state need per-session ownership. Migrate selected checkpoints into explicit tensor/config artifacts rather than automatically executing inherited pickle loaders.
- Separate code, graph-data and derived-weight provenance. The repository records MIT code, CC-BY graph data and composite model-weight terms. Reusing its algorithms does not require immediately loading the full 132M-parameter model or its multi-gigabyte teachers.

## New research directions proposed for SYNAPSE

### Persistent memory and current-input contribution

Expanse's evoked mode rejects persistent state. For SYNAPSE, compare two transitions from the same cloned pre-step state `r`:

`r_actual = G(r, u)` and `r_null = G(r, 0)`.

Read `memory_baseline = O(r_null)` and `input_effect = O(r_actual) - O(r_null)` with the same output-normalization convention applied separately before subtraction. Independently gate the two contributions; commit only `r_actual` to live state. The baseline contains both history and learned bias, so test zeroed/shuffled history separately. This is a proposed local counterfactual decomposition, not a unique mechanistic explanation or validated extension. It roughly doubles the probed graph work; cost belongs in comparisons. It does not inherit zero-edge cancellation without the same port/path assumptions.

### A life cycle for useful grafts

Treat a graft as a candidate with measurable stages: dormant identity → bounded activation → contribution measurement → consolidation → possible retirement. Utility is determined by task/retention outcomes, donor dependence and compute cost. Traffic or gradient magnitude alone is insufficient. Keep graph-dependent, ordinary-adapter and failed-transfer outcomes distinct rather than awarding all gains to biology.

### Recursive improvement of the improver

Keep the acting system's state `theta` separate from the improvement mechanism `phi`. The latter includes diagnostic-probe selection, replay/source-mixture policy, proposal ranking, allowed plasticity-rule coefficients and a small optional neural editor. The current `phi_g` proposes an edit to `theta`, itself, or eventually both. An accepted `phi_(g+1)` must actually produce subsequent descendants.

Measure each frozen improver on identical fresh learners and previously unseen learning problems. This prevents accumulated task answers and growing archives from masquerading as an improved learning algorithm. An unchanged external language model may remain a proposal primitive; that supports bounded system-level RSI, not a claim that the external model rewrote its own weights.

## RSI prior art and limits on novelty

- [STOP](https://arxiv.org/abs/2310.02304) improves an improver's scaffolding using a fixed language model. It explicitly distinguishes this from changing the underlying model.
- [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) empirically evaluates agent self-modifications and branching archives. An archive and repeated self-editing alone are established ideas.
- [Self-referential weight matrices](https://arxiv.org/abs/2202.05780) modify their own neural matrices at runtime; self-modification does not itself demonstrate progressively better improvement ability.
- [Learned optimizers that can train themselves](https://arxiv.org/abs/2009.11243) motivate testing optimization ability across fresh learner tasks.

The narrower proposed contribution is an inherited neural/structural improvement policy whose effectiveness is grounded in measured graft contribution and whose successors improve fresh connectome-coupled learners at comparable total cost. This remains a hypothesis. The revised main design specifies the evaluation, lineage, mutation surfaces and completion criteria.

## Review outcome

Adopt Expanse's concrete graft lifecycle, causal/state-handling discipline, consolidation operator and reproducibility machinery as engineering inputs. Treat topology benefits, donor transfer and recursive improvement as separately testable hypotheses. Expanse's negative and inconclusive results are design inputs, not defects to conceal or claims to copy into SYNAPSE.
