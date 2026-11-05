# NexusFlow: Directed Acyclic Causal Flux & Heuristic Routing

## 1. Architectural Motivation

Inspired by `NexusSearch`, `Causeway`, `nexusflow`, and `QuantumBot`, **NexusFlow** (`poseidon/nexusflow.py`) provides directed causal flux routing and coherent superposition arbitration over the agent's topological spatial graph in the TidePool environment.

Traditional planners either greedily choose the nearest resource or perform static graph searches. NexusFlow models the environment as an open hydraulic potential network, computing continuous **causal flux** across replenishment corridors while actively detecting destructive directional wave interference.

---

## 2. Mathematical Formulation

### 2.1 Waypoint Potentials
Each visited spatial waypoint $i = (x_i, y_i)$ possesses a continuous replenishment potential:
$$P_i = \max\Big(0, R_i - \kappa \cdot \text{Depletion}_i\Big)$$
where $R_i$ is the patch replenishment rate and $\kappa = 0.45$ is the harvesting depletion penalty coefficient.

### 2.2 Continuous Causal Flux
The directed causal flux $\Phi_{ij}$ between waypoint $i$ and waypoint $j$ obeys exponential spatial distance decay:
$$\Phi_{ij} = \exp\left(-\frac{\|\vec{x}_j - \vec{x}_i\|_2}{\tau}\right) \cdot \max(0, P_j - P_i)$$
where $\tau = 2.50$ is the characteristic spatial decay scale.

### 2.3 Potential Gradient Flow
The topological force vector $\vec{F}_i$ acting on an agent at waypoint $i$ aggregates flux along unit direction vectors $\vec{u}_{ij}$:
$$\vec{F}_i = \sum_{j \ne i} \Phi_{ij} \cdot \vec{u}_{ij}, \quad \vec{u}_{ij} = \frac{\vec{x}_j - \vec{x}_i}{\|\vec{x}_j - \vec{x}_i\|_2}$$

### 2.4 Coherent Superposition Wave Interference
When multiple candidate directional policies are proposed (e.g. AURA biomimetic heading $\theta_a$ vs NexusFlow potential vector $\theta_b$), NexusFlow models their arbitration as a coherent wave interference term:
$$I(a, b) = 2 \sqrt{V(a) \cdot V(b)} \cos(\theta_a - \theta_b)$$
- **Constructive Alignment ($\cos(\Delta \theta) \ge -0.2$):** Both signals reinforce navigation along the shared gradient.
- **Destructive Conflict ($\cos(\Delta \theta) < -0.2$):** Divergence exceeds $\approx 101^\circ$, indicating dangerous spatial bifurcation. NexusFlow suppresses greedy exploration and activates involuntary vertical safety maneuvers.
