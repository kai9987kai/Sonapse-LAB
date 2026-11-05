# Holographic Reduced Representations (HRR) Associative Concept Memory

## 1. Architectural Motivation

Standard agent memory mechanisms typically rely on one of two extremes:
1. **Parametric Weight Updates:** Slow gradient descent on neural weights (catastrophic forgetting, high latency).
2. **Explicit Slot-Based Buffers:** Growing dictionaries of key-value embeddings (linear search complexity, unbounded memory footprint).

Inspired by the holographic paradigms in **Mnemorph**, **TESSERA**, and **GenesisEngine**, Poseidon introduces **Holographic Concept Memory** (`poseidon/hologram.py`), rooted in **Holographic Reduced Representations (HRR)** and **Vector Symbolic Architectures (VSA)** (Plate 1994, Kanerva 1988).

HRR enables constant-size ($D = 256$), distributed, content-addressable associative memory where:
- Variable-value pairs are bound using **circular convolution**.
- Multiple bound associations are bundled into a single unified trace vector using **vector superposition**.
- Stored items are retrieved through **circular correlation (unbinding)** followed by vocabulary clean-up resonance.

---

## 2. Mathematical Formulation

### 2.1 Vector Space & Random Hypervectors
Hypervectors $\mathbf{x} \in \mathbb{R}^D$ are drawn from independent Gaussian distributions $\mathcal{N}(0, 1/D)$ and normalized to unit Euclidean length $\|\mathbf{x}\| = 1$. In high-dimensional spaces ($D \ge 256$), randomly drawn vectors are quasi-orthogonal with overwhelming probability:
$$\mathbb{E}[\mathbf{x} \cdot \mathbf{y}] = 0, \quad \operatorname{Var}[\mathbf{x} \cdot \mathbf{y}] = \frac{1}{D}$$

### 2.2 Circular Convolution Binding ($\circledast$)
Binding a role/variable $\mathbf{x}$ to a filler/value $\mathbf{y}$ is performed via discrete circular convolution:
$$z_n = (\mathbf{x} \circledast \mathbf{y})_n = \sum_{k=0}^{D-1} x_k \, y_{(n - k) \bmod D}$$

In the frequency domain using the Fast Fourier Transform (FFT):
$$\mathbf{z} = \mathcal{F}^{-1}\Big(\mathcal{F}(\mathbf{x}) \odot \mathcal{F}(\mathbf{y})\Big)$$
where $\odot$ is the Hadamard (element-wise) product. Circular convolution is:
- Associative: $(\mathbf{x} \circledast \mathbf{y}) \circledast \mathbf{w} = \mathbf{x} \circledast (\mathbf{y} \circledast \mathbf{w})$
- Commutative: $\mathbf{x} \circledast \mathbf{y} = \mathbf{y} \circledast \mathbf{x}$
- Preserves vector dimension ($D \to D$) without dimensionality inflation.
- Produces a vector $\mathbf{z}$ that is quasi-orthogonal to both inputs: $\mathbf{z} \cdot \mathbf{x} \approx 0, \mathbf{z} \cdot \mathbf{y} \approx 0$.

### 2.3 Circular Correlation Unbinding ($\circledast^\dagger$)
Given the bound trace $\mathbf{z} = \mathbf{x} \circledast \mathbf{y}$, querying with cue $\mathbf{x}$ recovers a noisy reconstruction $\mathbf{y}' \approx \mathbf{y}$:
$$(\mathbf{x} \circledast^\dagger \mathbf{z})_n = \sum_{k=0}^{D-1} x_k \, z_{(k + n) \bmod D} = \mathcal{F}^{-1}\Big(\mathcal{F}^*(\mathbf{x}) \odot \mathcal{F}(\mathbf{z})\Big)$$
where $\mathcal{F}^*$ denotes the complex conjugate. The reconstruction satisfies:
$$\mathbf{y}' = \mathbf{x} \circledast^\dagger (\mathbf{x} \circledast \mathbf{y}) = (\mathbf{x} \circledast^\dagger \mathbf{x}) \circledast \mathbf{y} \approx \mathbf{y} + \boldsymbol{\eta}$$
where $\boldsymbol{\eta}$ is zero-mean quasi-orthogonal residual noise.

### 2.4 Trace Superposition & Temporal Decay
Multiple bound representations are accumulated into a single memory trace $\mathbf{M} \in \mathbb{R}^D$ with exponential forgetting rate $\lambda \in (0, 1]$:
$$\mathbf{M}_t = \lambda \mathbf{M}_{t-1} + \mathbf{z}_t = \lambda \mathbf{M}_{t-1} + (\mathbf{x}_t \circledast \mathbf{y}_t)$$

Normalizing $\mathbf{M}_t$ preserves bounded numerical scale across arbitrarily long horizons:
$$\mathbf{M}_t \leftarrow \frac{\mathbf{M}_t}{\max(1.0, \|\mathbf{M}_t\|)}$$

### 2.5 Clean-Up Vocabulary Resonance
To extract discrete concepts from the noisy unbinded vector $\mathbf{y}'$, the vector is matched against the clean-up codebook $\mathcal{V} = \{\mathbf{v}_1, \dots, \mathbf{v}_K\}$ via cosine resonance:
$$\operatorname{sim}(\mathbf{y}', \mathbf{v}_k) = \frac{\mathbf{y}' \cdot \mathbf{v}_k}{\|\mathbf{y}'\| \, \|\mathbf{v}_k\|}$$
The concept with the highest resonance above detection threshold $\theta_{\text{recall}} = 0.15$ is retrieved.
