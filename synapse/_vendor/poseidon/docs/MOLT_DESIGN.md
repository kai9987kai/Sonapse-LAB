# MOLT: Metamorphic Online Life-Stage Transmutation & Morphogenesis

## 1. Architectural Motivation

In nature, marine organisms and insects do not operate under static morphological or sensory-motor constraints across their lifespans. Instead, they undergo **metamorphosis and ecdysis (molting)**, drastically reorganizing their bodily configuration, sensory aperture, and metabolic expenditure:
- **Juvenile / Larval Instar:** Maximizes local resource foraging efficiency and growth rate, at the expense of high vulnerability to environmental stressors (e.g. tidal wave exposure).
- **Diapause / Pupation:** Enters an acute quiescent metabolic arrest, heavily hardening against external physical exposure while internal somatic reorganization takes place.
- **Adult / Imago Dispersal:** Expands sensory receptive horizon, achieving high-speed hydrodynamic/aerodynamic travel and heightened spatial dispersal at the expense of higher basal metabolic maintenance costs.

In Poseidon, **MOLT** (`poseidon/molt.py`) introduces online developmental life-stage transmutation into the synthetic TidePool environment without modifying the frozen Tidal core network weights (`runs/tidal_dagger/core.pt`).

---

## 2. Formal Mathematical Formulation

### 2.1 Morphological Profiles
Each life stage $S \in \{\text{LARVAL}, \text{PUPA}, \text{IMAGO}\}$ defines a static morphological parameter tuple:
$$\mathcal{M}(S) = \langle \gamma_{\text{sensory}}, \xi_{\text{vuln}}, \beta_{\text{forage}}, \mu_{\text{basal}}, \sigma_{\text{stamina}}, \rho_{\text{horizon}} \rangle$$

| Parameter | Larval (Instar 0) | Pupa (Instar 1) | Imago (Instar 2) | Biological Role |
| :--- | :---: | :---: | :---: | :--- |
| $\gamma_{\text{sensory}}$ | $1.35$ | $0.50$ | $1.10$ | Sensory amplification on proximal resource signals |
| $\xi_{\text{vuln}}$ | $1.25$ | $0.35$ | $0.85$ | Chitinous hardening multiplier against tidal exposure |
| $\beta_{\text{forage}}$ | $+0.25$ | $0.00$ | $+0.10$ | Extra energetic conversion from foraging |
| $\mu_{\text{basal}}$ | $0.85$ | $0.40$ | $1.20$ | Basal metabolic energy decay rate multiplier |
| $\sigma_{\text{stamina}}$ | $0.90$ | $2.00$ | $0.65$ | Stamina expenditure per movement action |
| $\rho_{\text{horizon}}$ | $1.00$ | $0.50$ | $1.75$ | Spatial perception radius |

### 2.2 Transmutation Triggers and Biomass Accumulation
Biomass accumulation integrates excess energetic surplus above baseline homeostatic equilibrium:
$$\Delta B(t) = \max(0, E(t) - E_{\text{baseline}}) \cdot \eta$$

1. **Larva $\to$ Pupa:** When cumulative biomass $B(t) \ge \Theta_{\text{larva}} = 3.0$ and stamina $\ge 0.50$, the agent sheds its larval cuticle, initiating diapause.
2. **Pupa $\to$ Imago:** After $T_{\text{pupa}} \ge 4$ developmental ticks, somatic remodeling is complete, triggering emergence into the adult imago instar.

### 2.3 Exuvia Digest Binding
Each ecdysis event sheds an immutable cryptographic artifact called an **exuvia**:
$$\text{Exuvia} = \text{SHA-256}\Big(S_{\text{from}} \,\|\, S_{\text{to}} \,\|\, t \,\|\, B(t) \,\|\, H(t) \,\|\, \sigma(t)\Big)$$
ensuring that life-stage transitions remain verifiable, tamper-proof, and content-addressed.
