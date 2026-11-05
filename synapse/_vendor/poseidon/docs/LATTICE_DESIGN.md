# Diamond Lattice Entorhinal Neuropil

## 1. Architectural Motivation

In mammalian neurobiology, the medial entorhinal cortex (MEC) utilizes grid cells that exhibit periodic hexagonal firing patterns to provide a metric coordinate framework for spatial navigation and dead-reckoning path integration (Hafting et al. 2005).

However, hexagonal tiling is strictly a 2D optimal sphere-packing phenomenon. In 3D spatial environments (such as marine pelagic navigation, 3D flight, and complex multi-agent kinematics seen in **FLY-DIAMOND-NEXUS**, **3d-animal-simulator-Hybrid-Agent**, and **Prometheus-Alpha**), the mathematically optimal sphere packing is the **Face-Centered Cubic (FCC) / Diamond Tetrahedral Lattice** (Kepler conjecture, Hales 2005).

The **Diamond Lattice Neuropil** (`poseidon/lattice.py`) provides Poseidon with a 3D multi-scale geometric grid coordinate system that continuously integrates velocity vectors across 3 toroidal spatial wavelengths without drift.

---

## 2. Mathematical Formulation

### 2.1 Diamond Tetrahedral Wave Vectors
In three-dimensional space $\mathbb{R}^3$, diamond tetrahedral symmetry is defined by four unit wave vectors pointing to the vertices of a regular tetrahedron:
$$\mathbf{k}_1 = \frac{1}{\sqrt{3}}\begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}, \quad \mathbf{k}_2 = \frac{1}{\sqrt{3}}\begin{pmatrix} 1 \\ -1 \\ -1 \end{pmatrix}, \quad \mathbf{k}_3 = \frac{1}{\sqrt{3}}\begin{pmatrix} -1 \\ 1 \\ -1 \end{pmatrix}, \quad \mathbf{k}_4 = \frac{1}{\sqrt{3}}\begin{pmatrix} -1 \\ -1 \\ 1 \end{pmatrix}$$

These wave vectors satisfy:
$$\sum_{j=1}^{4} \mathbf{k}_j = \mathbf{0}, \quad \mathbf{k}_i \cdot \mathbf{k}_j = -\frac{1}{3} \quad (i \ne j)$$
ensuring isotropic, rotational spatial symmetry in all three spatial dimensions.

### 2.2 Multi-Scale Geometric Modules
Spatial scales are decomposed into $M = 3$ discrete modules with geometrically progressing spatial wavelengths:
- Module 1 (Micro-scale): $\lambda_1 = 3.0$ units (fine-grained patch discrimination)
- Module 2 (Meso-scale): $\lambda_2 = 8.0$ units (habitat / reef region boundary)
- Module 3 (Macro-scale): $\lambda_3 = 20.0$ units (global TidePool domain metric)

### 2.3 Continuous Toroidal Phase Velocity Integration
Given an instantaneous 3D agent velocity vector $\mathbf{v}(t) = (\dot{x}, \dot{y}, \dot{z})$, each module updates its internal toroidal phase coordinates $\boldsymbol{\theta}_m = (\theta_{m, 1}, \dots, \theta_{m, 4}) \in [0, 2\pi)^4$:
$$\theta_{m, j}(t + \Delta t) = \left(\theta_{m, j}(t) + \frac{2\pi}{\lambda_m} (\mathbf{k}_j \cdot \mathbf{v}(t)) \Delta t\right) \bmod 2\pi$$

Phase accumulation is strictly toroidal ($S^1 \times S^1 \times S^1 \times S^1$), eliminating numerical overflow or unbounded drift during arbitrary continuous exploration.

### 2.4 Modular Grid Firing Field & Coherence Index
The spatial firing activity $g_m(\mathbf{x})$ of module $m$ at position $\mathbf{x} \in \mathbb{R}^3$ is given by the constructive interference of the four tetrahedral plane waves:
$$g_m(\mathbf{x}) = \frac{1}{4} \sum_{j=1}^{4} \cos\left(\frac{2\pi}{\lambda_m} \mathbf{k}_j \cdot \mathbf{x} + \theta_{m, j}\right) \in [-1, 1]$$

The overall multi-scale **Lattice Coherence Index** $\mathcal{C}_{\text{lattice}} \in [0, 1]$ measures the constructive alignment of spatial grid modules:
$$\mathcal{C}_{\text{lattice}} = \frac{1}{M} \sum_{m=1}^{M} \frac{1 + g_m(\mathbf{x})}{2}$$

When the agent is at a spatial lattice node where all modules constructively interfere, $\mathcal{C}_{\text{lattice}} \to 1.0$. In regions of spatial decoherence or boundary drift, $\mathcal{C}_{\text{lattice}}$ decreases, triggering navigational re-calibration and safety caution.
