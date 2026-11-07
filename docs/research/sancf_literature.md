# SANCF & Frontier Connectome Architecture — Literature Review (2024–2026)

**Date:** 2026-11-07  
**Baseline Commit:** `880bb4a0da05ca783738a16723b0a629f14e5517`

---

## 1. Graph Message Passing, Oversquashing, Oversmoothing & Rewiring

1. **Deeper with Riemannian Geometry: Overcoming Oversmoothing and Oversquashing for Graph Foundation Models** (NeurIPS 2025)
   - *URL:* https://proceedings.nips.cc/paper_files/paper/2025/hash/e4e5d01075906f9d8522911a736aa5d9-Abstract-Conference.html
   - *Core Mechanism:* Uses negatively/positively curved Riemannian manifold embeddings (including hyperbolic Poincaré projections for hierarchical hubs) to preserve representation rank across deep message-passing steps.
   - *Competing Methods & Limitations:* Euclidean residual connections still suffer rank collapse when spectral gap is small; manifold projections add per-step trigonometric/logarithmic map overhead and do not by themselves create new task information if source features are uninformative.
   - *Testable Rationale in SANCF:* We pair low-rank inter-region communication with explicit effective-rank and stable-rank monitoring (`comm_effective_rank`) so any claim of reduced oversmoothing is directly measured rather than assumed.

2. **The Effectiveness of Curvature-Based Rewiring and the Role of Hyperparameters in GNNs Revisited** (ICLR 2025)
   - *URL:* https://proceedings.iclr.cc/paper_files/paper/2025/hash/ea7c7e886d48dd3a3a799819c8195012-Abstract-Conference.html
   - *Core Mechanism:* Demonstrates that Ollivier-Ricci and Forman-Ricci curvature rewiring gains often shrink or vanish once baseline hyperparameters (dropout, layer norm, residual scale, width) are properly tuned.
   - *Limitations:* Adding arbitrary shortcut edges can degrade graph locality (over-smoothing) and obscure whether gains come from topology or extra parameter capacity.
   - *Testable Rationale in SANCF:* Every topology strategy (`random`, `anatomical_interface`, `structural_centrality`, `functional_perturbation`, `learned_low_rank`, `hybrid_causal`) is benchmarked against a matched-parameter/matched-budget control (`A1` / `A7`) with frozen base connectome edges.

3. **Mitigating Over-Squashing in GNNs by Spectrum-Preserving Sparsification** (ICML 2025, Liang et al.)
   - *URL:* https://proceedings.mlr.press/v267/liang25a.html
   - *Core Mechanism:* Selects sparse inter-community bridges that preserve the normalized Laplacian spectrum while bounding node fan-in.
   - *Limitations:* Exact spectral sparsification scales superlinearly if recomputed every step on dense matrices.
   - *Testable Rationale in SANCF:* `neural_comm_topology.py` precomputes degree/participation/community bridge candidates offline with strict fan-in and route budgets.

4. **PANDA: Expanded Width-Aware Message Passing Beyond Rewiring** (ICML 2024, Choi et al.)
   - *URL:* https://proceedings.mlr.press/v235/choi24f.html
   - *Core Mechanism:* Expands message dimensionality (`D > 1`) across bottleneck bridges rather than mutating the underlying graph adjacency.
   - *Limitations:* Wider messages increase compute/memory linearly in `D` and only help when the task requires multi-feature cross-region coordination that a scalar edge bottlenecks.
   - *Testable Rationale in SANCF:* SANCF's `MessageEncoder` / `MessageDecoder` sends a `D`-dimensional latent vector (`D in {4, 16, 32}`) alongside unmodified scalar grafts, evaluated on controlled narrow-vs-wide cue-fusion benchmarks.

5. **Multi-Track Message Passing** (ICML 2024, Pei et al.) & **Adaptive Message Passing** (ICML 2025, Errica et al.)
   - *URLs:* https://proceedings.mlr.press/v235/pei24a.html · https://proceedings.mlr.press/v267/errica25a.html
   - *Core Mechanism:*Separates parallel representation tracks and learns per-step halting/routing gates so regions communicate only when context demands it.
   - *Limitations:* Unregularized gates can saturate at 1.0 (wasting bandwidth) or collapse to 0.0 (dead channel).
   - *Testable Rationale in SANCF:* Zero-initialized residual gain $\gamma$ with explicit bitrate ($R_{\text{bits}}$), compute ($R_{\text{compute}}$), and entropy regularization.

---

## 2. Bandwidth-Limited & Orthogonal-Mode Communication

6. **Bandwidth-Efficient Multi-Agent Communication through Information Bottleneck and Vector Quantization** (ICRA 2026)
   - *URL:* https://ieeexplore.ieee.org/abstract/document/11696376
   - *Core Mechanism:* Combines a variational/deterministic information bottleneck with Vector Quantization (VQ codebook + straight-through estimator) to compress inter-module messages under strict bit budgets.
   - *Limitations:* Codebook collapse occurs if EMA dead-code revival or commitment loss is omitted; numeric bit-width does not equal task-relevant mutual information.
   - *Testable Rationale in SANCF:* `QuantisedChannel` implements both uniform $b$-bit (`2/4/8/16`) quantization and VQ codebooks (`16/64/256` entries) with EMA utilization tracking, STE gradients, and exact header + index + scale bit accounting.

7. **Capacity Limits of Spatially Multiplexed Free-Space Communication** (Nature Photonics 2015, Zhao et al.) & **Orbital Angular Momentum Modes Do Not Increase Channel Capacity** (New Journal of Physics 2015, Andersson et al.)
   - *URLs:* https://www.nature.com/articles/nphoton.2015.214 · https://doi.org/10.1088/1367-2630/17/4/043040
   - *Core Finding:* Orthogonal spatial modes (such as OAM or Hermite/Laguerre-Gaussian modes) are a change of basis within existing spatial degrees of freedom; they **do not** create free Shannon capacity beyond conventional MIMO/orthogonal basis decompositions under equal power and aperture constraints.
   - *Testable Rationale in SANCF:* `OrthogonalModeChannel` (`M_{ij}(t) = \sum_{k=1}^K a_{ij,k}(t)\psi_k`) is explicitly implemented and documented as a **digital orthogonal-mode representation analogy** (Walsh-Hadamard / DCT / Stiefel orthonormal basis with mode-selective filtering and crosstalk robustness testing), never as physical wave multiplexing or superdense quantum capacity.

---

## 3. Communication Through Coherence (CTC), Criticality & Vector Symbolic Architectures

8. **Rhythms for Cognition: Communication through Coherence** (Neuron 2015, Fries)
   - *Core Mechanism:* Phase alignment between sending and receiving neural populations modulates effective synaptic gain.
   - *Limitations in Artificial Models:* Abstract coupled phase oscillators ($\theta/\beta/\gamma$ proxies) in discrete-time rate networks are **not** biophysical membrane-conductance rhythms and must never invert Dale's law excitatory/inhibitory neurotransmitter polarity.
   - *Testable Rationale in SANCF:* NeuroWave phase gating multiplies non-negative coherence gains $\in [0, 1]$ while strictly preserving underlying E/I sign constraints and comparing against random-phase, shuffled-phase, and frozen-phase controls.

9. **Power-Law Distributions in Empirical Data** (SIAM Review 2009, Clauset, Shalizi & Newman) & **Neuronal Avalanches in Neocortical Circuits** (J. Neurosci. 2003, Beggs & Plenz)
   - *Core Finding:* Ratio of consecutive active counts is an insufficient estimator of branching dynamics under external drive, and power-law exponents cannot be claimed without maximum-likelihood fitting, Kolmogorov-Smirnov goodness-of-fit testing, log-likelihood ratios against exponential/lognormal alternatives, and sufficient event counts.
   - *Testable Rationale in SANCF:* `synapse/criticality_vsa.py` returns `"insufficient_data"` when avalanche event count $< 12$, computes log-likelihood ratios against exponential/lognormal distributions, bootstrap confidence intervals, perturbation susceptibility, and autocorrelation decay.

10. **Holographic Reduced Representations** (IEEE Trans. Neural Networks 1995, Plate)
    - *Core Mechanism:* Circular convolution binds role-filler pairs in $\mathbb{R}^D$ with $\mathcal{O}(D \log D)$ FFT complexity.
    - *Limitations:* Superposition SNR degrades as $\mathcal{O}(1/\sqrt{M})$ with $M$ bound items, requiring explicit interference and capacity assays (`synapse/memory_interference_assay.py`).

---

## 4. Grafted Brain Sections: Selective State-Space (Mamba), Diffusion Policies & Offline/Online RL

11. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (Gu & Dao, COLM 2024)
    - Input-dependent selection gates $(\Delta_t, B_t, C_t)$ allow linear-time recurrent state spaces to selectively latch task-relevant cues while forgetting distractors. Used in `SelectiveStateSpaceMambaSection`.

12. **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** (Chi et al., RSS 2023 / IJRR 2024)
    - Models conditional multimodal action distributions via iterative score-based denoising steps. Used in `GraftedMiniDiffusionPolicyModel`.

13. **Hopfield Networks is All You Need** (Ramsauer et al., ICLR 2021)
    - Continuous modern Hopfield energy function with inverse-temperature $\beta$ provides exponential pattern storage and single-step associative retrieval. Used in `HopfieldEnergyAssociativeSection`.

14. **Offline Reinforcement Learning with Implicit Q-Learning (IQL)** (Kostrikov et al., ICLR 2022) & **Soft Actor-Critic (SAC)** (Haarnoja et al., ICML 2018)
    - Expectile value regression ($\tau$-expectile) avoids querying out-of-support actions during replay updates, while maximum-entropy twin-Q critics prevent value overestimation. Used in `MultiParadigmRLTrainer` and `NeuralGraft` audit improvements.
