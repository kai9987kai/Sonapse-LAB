# PROMETHEUS: Meta-Plasticity & Homeostatic Drive Self-Regulation

## 1. Motivation & Principles

Inspired by `prometheus-alpha`, **Prometheus** provides real-time online meta-parameter adaptation and homeostatic drive balancing without gradient descent or retraining the frozen Tidal core.

In standard reinforcement learning and imitation policies, hyperparameters such as potential gains, memory retention rates, and exploration temperatures are static. In volatile, highly scarce environments (scarcity $\ge 2.5$), fixed parameters cause either hyper-fixation on depleted resource patches or excessive wandering.

Prometheus dynamically regulates:
1. **Causeway Non-Hermitian Gain** ($\gamma_t$).
2. **Holographic Memory Decay** ($\lambda_t$).
3. **Multi-Drive Physiological Allostasis** ($U_{\text{homeo}}$).

---

## 2. Mathematical Formalization

### 2.1 Rolling Scarcity Entropy
Over a sliding window $W = 16$ of recent observations, Prometheus tracks the Shannon entropy of vital resource indicators (energy, hydration, stamina):
$$H(S) = -\sum_{k=1}^K p_k \log_2(p_k + \epsilon)$$
where $p_k$ is the empirical probability density over normalized resource depletion bins.

### 2.2 Dynamic Gain & Decay Adaptation
When environmental entropy is elevated (high chaos / resource volatility), Prometheus amplifies the non-Hermitian gain in Causeway:
$$\gamma_t = \gamma_0 \cdot \big(1.0 + \beta \cdot H(S)\big)$$
Higher gain enables the agent to decisively escape local reward traps and accelerate probability concentration.

Simultaneously, the holographic memory retention decay is modulated:
$$\lambda_t = \lambda_0 \cdot \big(1.0 - 0.15 \cdot \text{std}(H(S))\big)$$
During chaotic volatility, decay accelerates slightly to prevent stale navigational dead-ends from dominating decision manifolds.

### 2.3 Homeostatic Drive Allostasis
Prometheus continuously evaluates four physiological drives:
- **Hunger Drive:** $D_{\text{hunger}} = \max(0, 1.0 - \text{energy})$
- **Thirst Drive:** $D_{\text{thirst}} = \max(0, 1.0 - \text{hydration})$
- **Fatigue Drive:** $D_{\text{fatigue}} = \max(0, 1.0 - \text{stamina})$
- **Exposure Drive:** $D_{\text{exposure}} = \text{exposure} \cdot (1.0 \text{ if health} < 0.8 \text{ else } 0.5)$

The dominant drive $\arg\max_k D_k$ directly injects homeostatic bias into the action potentials, ensuring life-critical physiological equilibrium.
