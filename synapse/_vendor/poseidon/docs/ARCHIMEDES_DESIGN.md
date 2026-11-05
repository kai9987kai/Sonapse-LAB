# Archimedes Hydrodynamics & Buoyancy Engine Design

## 1. Overview and Theoretical Formulation

The **Archimedes Hydrodynamic & Buoyancy Engine** (`poseidon/archimedes.py`) integrates hydrodynamic fluid mechanics and buoyancy principles derived from `supermix-archimedes` and `Supermix-Expanse` into the synthetic TidePool survival substrate.

The engine establishes physical fluid interactions without modifying or retraining the frozen Tidal core policy:
1. **Seawater Density Stratification $\rho(S, T)$**: Local fluid density is governed by salinity $S \in [0, 1]$ and temperature $T \in [0, 1]$:
   $$\rho(S, T) = \rho_0 \left(1 + \alpha_S S - \beta_T (T - 0.5)\right)$$
   where baseline seawater density $\rho_0 = 1.025\text{ kg/L}$, salinity coefficient $\alpha_S = 0.028$, and thermal expansion coefficient $\beta_T = 0.003$.
2. **Archimedes Hydrostatic Buoyancy $F_b$**: The buoyant force exerted on the agent displacement volume $V_{\text{disp}}$ is:
   $$F_b = \rho(S, T) \cdot V_{\text{disp}} \cdot g$$
   Internal gas bladder vesicle inflation $v_{\text{inflation}} \in [0, 1]$ dynamically modulates displacement volume $V_{\text{disp}} = V_0 (0.8 + 0.4 v_{\text{inflation}})$.
3. **Net Vertical Acceleration & Depth Modulation**:
   $$F_{\text{net}} = F_b - m \cdot g - 0.5 C_d \rho v_z |v_z| A$$
   yielding depth updates across the water column $z \in [0, 1]$ (surface to benthos).
4. **Harmonic Horizontal Tidal Surge Currents $\mathbf{u}(t)$**:
   $$\mathbf{u}(t) = \left[ U_0 \sin\left(\frac{2\pi t}{\tau}\right), U_0 \cos\left(\frac{2\pi t}{\tau}\right) \right]$$
   where tidal period $\tau = 48\text{ steps}$ and maximum surge velocity $U_0 = 1.2\text{ m/s}$.
5. **Directional Locomotion Modulation**:
   Swimming aligned with tidal currents confers a stamina discount ($\le 40\%$), while opposing currents incur an energetic drag surcharge ($\le 50\%$).

---

## 2. API Reference

```python
from poseidon.archimedes import ArchimedesHydrodynamicEngine

engine = ArchimedesHydrodynamicEngine(tide_period_steps=48, max_surge_velocity=1.2)
engine.reset()

# Step hydrodynamic simulation
fluid_info = engine.step(
    salinity=0.55,
    temperature=0.48,
    depth_action=0,        # -1 = dive, 0 = neutral, 1 = surface
    locomotion_action=1,   # Cardinal action [1..4]
)
print("Fluid density:", fluid_info["fluid_density"])
print("Buoyancy force:", fluid_info["buoyancy_force"])
print("Surge velocity:", fluid_info["surge_velocity"])
print("Stamina multiplier:", fluid_info["stamina_cost_multiplier"])
```

---

## 3. Empirical Verification

Unit test coverage in `tests/test_archimedes.py` validates:
- Salinity stratification increases fluid density monotonically.
- Vesicle inflation regulates neutral buoyancy and vertical equilibrium.
- Harmonic tidal currents cycle accurately across the 48-step period.
- Directional stamina discounts apply symmetrically with current alignment.
