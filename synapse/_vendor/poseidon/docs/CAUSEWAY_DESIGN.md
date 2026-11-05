# Causeway: Quantum Superposition Decision Engine

## 1. Architectural Motivation

Classical decision-making architectures in agentic reinforcement learning collapse probability distributions prematurely, discarding intermediate cognitive phase relationships, constructive/destructive interference between mutually reinforcing or conflicting intentions, and continuous Hamiltonian energy landscapes.

Drawing from the principles of **Causeway**, **QuantumBot**, and **GhostInTheMachine**, the **Causeway Decision Engine** (`poseidon/causeway.py`) represents the discrete action manifold $\mathcal{A} = \{0, 1, 2, 3, 4, 5\}$ of Poseidon as a 6-dimensional complex quantum state vector:
$$|\psi(t)\rangle = \sum_{a=0}^{5} \alpha_a(t) |a\rangle, \quad \alpha_a(t) \in \mathbb{C}, \quad \sum_{a=0}^{5} |\alpha_a(t)|^2 = 1$$

Actions do not exist as isolated classical choices. Instead, they evolve continuously in superposition on a complex Hilbert sphere, where phase rotations, potential channeling, and wave interference guide the agent's emergent survival behavior.

---

## 2. Formal Mathematical Formulation

### 2.1 State Vector & Density Matrix
Each action eigenstate $|a\rangle$ is assigned a complex amplitude:
$$\alpha_a = u_a + i v_a$$
The probability of observing action $a$ follows the Born rule:
$$p_a = |\alpha_a|^2 = u_a^2 + v_a^2$$
The density operator is given by $\rho = |\psi\rangle \langle\psi|$, with diagonal entries $\rho_{aa} = p_a$ and off-diagonal coherence terms $\rho_{ab} = \alpha_a \alpha_b^*$.

### 2.2 Non-Hermitian Potential Evolution & Unitary Phase Rotation
To allow the agent's multi-channel potential landscape $\mathbf{V} \in \mathbb{R}^6$ (derived from physiological drives, causal fluxes, and topological gradients) to amplify promising action trajectories while accumulating phase, the state evolves according to a combined non-Hermitian gain and unitary phase operator:

$$\alpha_a(t + \Delta t) = \alpha_a(t) \cdot \exp\Big(\gamma (V_a - \bar{V})\Big) \cdot \exp\Big(-i (V_a + \phi_a) \Delta t\Big)$$

where:
- $\bar{V} = \frac{1}{6} \sum_{a=0}^{5} V_a$ is the mean baseline potential.
- $\gamma > 0$ is the non-Hermitian potential amplification gain parameter ($\gamma = 1.0$).
- $\phi_a \in [-\pi, \pi]$ is an action-specific phase bias from external oscillations (e.g. Chronos cyclic beaconing).
- After evolution, the state vector is re-normalized on the unit sphere:
$$\alpha_a \leftarrow \frac{\alpha_a}{\sqrt{\sum_{k=0}^{5} |\alpha_k|^2}}$$

### 2.3 Pairwise Wave Interference
Interference between distinct candidate actions $a$ and $b$ is computed directly from complex inner products:
$$I(a, b) = 2 \text{Re}(\alpha_a^* \alpha_b) = 2 (u_a u_b + v_a v_b)$$

- **Constructive Interference ($I(a, b) > 0$):** Actions share phase alignment and reinforce mutual execution probability.
- **Destructive Interference ($I(a, b) < -0.25$):** Antagonistic actions (e.g., flee vs. forage) possess opposing phase components, signaling cognitive conflict.

### 2.4 Von Neumann Decoherence Entropy & Coherence Length
Superposition dispersion is quantified via von Neumann entropy:
$$S(\rho) = -\sum_{a=0}^{5} p_a \ln (p_a + \epsilon)$$
- $S \approx \ln(6) \approx 1.79$: Maximum uncertainty; equal superposition across all branches.
- $S \to 0$: Near-pure state; decisive collapse towards a single action eigenstate.

Quantum coherence length measures the magnitude of off-diagonal quantum correlations:
$$L_c = \frac{1}{|\mathcal{A}| - 1} \sum_{a \ne b} |\alpha_a^* \alpha_b|$$

### 2.5 Born Rule Peak-Amplitude Collapse
Upon measurement, the agent collapses the superposition to the maximum-likelihood action eigenstate:
$$a^* = \arg\max_{a \in \mathcal{A}} |\alpha_a|^2$$
ensuring decisive action selection under acute survival pressure while retaining the entire evolution telemetry in the replay log.
