# Odysseus Navigator v0.7

Odysseus is an empirical cognitive navigation architecture designed for TidePool survival environments without modifying the frozen Tidal core neural weights. It resolves the core limitations of static spatial graphs by introducing online Bayesian replenishment estimation, empirical macro-action transition matrices, and physiological stamina rest guards.

---

## Architecture & Mathematical Foundations

### 1. Online Bayesian Replenishment Estimator
Rather than assuming deterministic linear replenishment, Odysseus maintains independent Bayesian posterior estimators for each patch's food and water replenishment rate:
$$\lambda \sim \mathcal{N}(\mu_\lambda, \sigma^2_\lambda)$$

- **Prior Formulation**: Prior mean $\mu_0 = 0.008$, prior variance $\sigma^2_0 = 0.05^2$.
- **Revisit Observation**: When patch $p$ is revisited after $\Delta t$ ticks since last seen, observing recovered resource level $R_t \in [0, 1]$ relative to depleted level $R_{\text{dep}}$ yields an empirical rate observation:
  $$\hat{\lambda} = \frac{\max(0, R_t - R_{\text{dep}})}{\Delta t}$$
- **Posterior Update**: With observation noise $\sigma^2_{\text{obs}} = 0.02^2$:
  $$\mu_{\text{post}} = \frac{\sigma^2_{\text{obs}} \mu_0 + \sigma^2_0 \hat{\lambda}}{\sigma^2_0 + \sigma^2_{\text{obs}}}, \quad \sigma^2_{\text{post}} = \frac{\sigma^2_0 \sigma^2_{\text{obs}}}{\sigma^2_0 + \sigma^2_{\text{obs}}}$$
- **Lower Confidence Bound (LCB)**: Navigational utility evaluates conservative availability discounted by parameter $z = 1.645$ (95% conservative one-sided bound):
  $$\text{LCB}(R) = \min(R_{\text{cap}}, R_{\text{last}} + \max(0, \mu_\lambda - z \sigma_\lambda) \cdot \Delta t)$$

### 2. Empirical Macro-Action Transition Distributions
Odysseus abandons static geometric topology and directly models empirical destination distributions under macro-actions explore ($a=4$) and flee ($a=5$):
$$\hat{P}(P_{\text{dest}} \mid P_{\text{src}}, a) = \frac{N(P_{\text{src}}, a, P_{\text{dest}})}{\sum_{p'} N(P_{\text{src}}, a, p')}$$
Waypoint routing discounts candidate patches by their observed reachability confidence $\hat{P}$, preventing suicide migrations into inaccessible terrain.

### 3. Physiological Pre-Transit Stamina Rest Guard
In TidePool physics, executing strenuous travel actions ($a \in \{4, 5\}$) when stamina is depleted ($\text{stamina} < 0.20$) causes acute exhaustion mortality. Odysseus enforces a hard pre-transit safety invariant:
$$\text{If } \text{stamina} < 0.20 \text{ and proposed action} \in \{4, 5\} \implies \text{override to } \text{rest } (a=0)$$

### 4. Calibration & Overestimation Radius
During disjoint calibration partitions, empirical one-step net utility gains over policy incumbent are compared against realized transition rewards to measure overestimation error:
$$\text{radius} = \text{quantile}_{0.90}(\{ \max(0, \hat{U}_{\text{pred}} - R_{\text{actual}}) \})$$
Waypoint navigational overrides are accepted only if net utility clears this empirical threshold $\hat{U} > \text{radius}$.

---

## CLI & Verification Commands

```powershell
# Fit Odysseus empirical error calibration radius
python -m poseidon fit-odysseus --train-episodes 4 --calibration-episodes 4 --max-steps 32 --scarcity 2.5

# Run multi-arm paired benchmark
python -m poseidon odysseus-experiment --episodes 4 --max-steps 64 --scarcity 2.5

# Weightless replay verification of experiment receipt
python -m poseidon verify-odysseus outputs/odysseus_experiments/1521d677b6f2f75b-c4035a1005d3.json

# Interactive rollout with Odysseus controller
python -m poseidon world --planner odysseus --seed 133000001 --max-steps 64 --scarcity 2.5
```

---

## Diagnostic Controls & Arms

The paired evaluation protocol tests 6 distinct arms on identical worlds and seeds:
1. `policy`: Frozen Tidal DAgger policy baseline.
2. `odysseus_calibrated`: Full Odysseus with Bayesian LCB routing, empirical transition probabilities, stamina rest guard, and calibrated error margin.
3. `odysseus_ungated`: Odysseus navigational overrides without calibration radius hurdle.
4. `odysseus_no_memory`: Odysseus with memory ablations.
5. `odyssey`: Prior topological map controller (v0.5 baseline).
6. `random`: Uniform stochastic action selection negative control.
