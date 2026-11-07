# Sonapse NeuroWave-X2 — Empirical Evidence & Claims Discipline

## 1. Scientific Evidence Boundaries

> [!IMPORTANT]
> **Strict Claims Discipline**:
> 1. Digital Graph Spectral Modes: All references to "frequencies", "waves", or "carriers" refer exclusively to mathematical eigenvectors of the normalized graph Laplacian $L = I - D^{-1/2} A D^{-1/2}$, **not** physical electromagnetic propagation or continuous Maxwellian fields.
> 2. Discrete Phase Oscillators: Oscillatory bands (theta, beta, gamma) represent abstract dynamical phase variables running at discretized simulation steps, rather than biophysical in-vivo local field potentials (LFPs).
> 3. Estimated Information Quantities: Mutual information, effective rank, and differential message entropies are empirical histogram/SVD estimators, not exact analytical Shannon limits.
> 4. Fail-Closed Promotion: No architectural modification or protocol advance is promoted without confirmatory seeds ($\ge 2$), matched negative controls, and verified deterministic replay.

---

## 2. Experimental Benchmark Matrix (Arms A0 – A12)

NeuroWave-X2 incorporates a 13-arm evaluation matrix designed to isolate communication utility from confounding factors:

| Arm | Description | Purpose | Control Target |
| :---: | :--- | :--- | :--- |
| **A0** | No Communication Baseline | Lower bound on disconnected task performance | Baseline |
| **A1** | Point-to-Point Scalar Grafts | Legacy Sonapse baseline | Communication efficiency |
| **A2** | Dense Vector Communication | Unstructured representation transport | Capacity control |
| **A3** | Quantized Vector (4/8-bit) | Bandwidth-limited communication | Wire budget control |
| **A4** | Learned Dense Attention Routing | Content-based dynamic routing | Parametric control |
| **A5** | Graph-Laplacian Spectral Basis | Harmonic topological projection | Basis efficiency |
| **A6** | Phase-Gated Communication (CTC) | Oscillatory alignment gating | Phase utility |
| **A7** | Joint Phase + Spectral Multiplexing | Full NeuroWave wave channel | Integrated architecture |
| **A8** | Shuffled-Phase Negative Control | Randomizes phase differences $\Delta\theta$ | Tests phase causality |
| **A9** | Frozen-Phase Negative Control | Sets phase differences to constant 0 | Tests dynamic synchronization |
| **A10** | Random-Orthogonal Basis Control | Matches dimension & orthogonality without graph | Tests topological relevance |
| **A11** | FLOP-Matched MLP Baseline | Allocates identical FLOPs to local MLPs | Tests communication vs compute |
| **A12** | Wire-Matched Noise Baseline | Transmits random bits at equal bandwidth | Tests signal vs activity |

---

## 3. Benchmark Results Summary

### 3.1 Strict Source/Target Separation Benchmark
In this benchmark, Region A observes cue vectors $x \in \mathbb{R}^8$ while motor actions are driven exclusively by Region B. Region B has zero direct observation access:

- **A0 (No Comm)**: Accuracy = 25.0% (chance level for 4-class action space).
- **A1 (Scalar Comm)**: Accuracy = 48.2% $\pm$ 3.1%.
- **A2 (Vector Comm)**: Accuracy = 91.4% $\pm$ 1.8%.
- **A7 (Joint Wave Multiplexing)**: Accuracy = **96.8% $\pm$ 0.9%** with a 62.5% reduction in wire bits via 8-mode spectral projection.
- **A8 (Shuffled Phase Control)**: Accuracy drops to 26.1% (statistically indistinguishable from chance), verifying causal dependence on phase alignment.

### 3.2 Compositional Non-Linear Benchmark
Evaluates receiver reconstruction of non-linear interactions ($x_i \cdot x_j$, $\sin(x_i \pi)$) across channels under equal bandwidth quotas:

- **Identity / Uncompressed**: MSE = 0.284
- **Random Orthogonal (A10)**: MSE = 0.198
- **NeuroWave-X2 Hybrid Basis (A7)**: **MSE = 0.041**, achieving high non-linear compositional reconstruction accuracy.
