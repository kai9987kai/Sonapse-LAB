# GraftBrain: Closed-Loop Embodied Reinforcement Learning via Multi-Connectome Cross-Wiring and Bio-Inspired Circuit Modules

**Kai Piper**  
*Independent Research / Laboratory for Connectomic Intelligence*  
`kai9987kai`

---

## Abstract

**GraftBrain** is a CPU-local engineering lab that couples directed neural graphs, artificial cross-graph grafts, simplified compass and Mushroom Body modules, and an embodied virtual environment. Its default graph contains synthetic statistical proxies named for eight *Drosophila* connectome families. Explicit FlyWire, neuPrint, or CSV imports are required to use reconstructed data; default proxy edges are not measured biological synapses.

This increment adds observation-dependent gains at graft sites and structural plasticity within a fixed candidate pool. A persistent mask changes active support while conserving each directional site's edge-record budget. Coactivity scores include dormant candidates, enabling prune/regrow decisions after complete learning updates. The existing learner is a one-pass actor-critic with Generalized Advantage Estimation (GAE), plus three-factor Hebbian updates; it is not a demonstrated multi-epoch PPO implementation. Five modes provide dense compatibility, fixed-support, routing-only, structural-only, and combined controls. The implementation and bounded evaluation protocol enable mechanism checks. No general task improvement, biological fidelity, sparse-compute speedup, or new sparse-training algorithm is claimed. Historical throughput and ablation figures retained below are unverified because supporting run artifacts are unavailable.

---

## 1. Introduction

A central objective of computational neuroscience and artificial intelligence is to understand how biological connectivity gives rise to adaptive behavior. For decades, connectionist models have relied on idealized topologies—such as multilayer perceptrons, convolutional layers, and self-attention transformers—optimizing weights from unconstrained random initializations (LeCun et al., 2015; Vaswani et al., 2017). Concurrently, connectomics has undergone a technological revolution: advances in automated serial-section electron microscopy, volumetric computer vision, and distributed human proofreading have produced complete, synapse-level wiring diagrams of the nervous system of *Drosophila melanogaster* (Dorkenwald et al., 2024; Schlegel et al., 2024; Scheffer et al., 2020; Takemura et al., 2023).

Despite the availability of these wiring diagrams, significant computational barriers remain:
1. **Compartmentalized Disconnection**: Public datasets exist in isolation—such as the brain-only FlyWire dataset, the ventral nerve cord (MANC/FANC) datasets, and the optic lobe connectome. In a living organism, behavior arises from continuous, bidirectional loop dynamics traversing optic neuropils, central decision centers, descending premotor tracts through the neck connective, and ascending proprioceptive feedback from the nerve cord.
2. **Closed-Loop Integration**: Coupling graph activity to sensory input and motor feedback is an engineering requirement for this lab's behavioral experiments.
3. **Plasticity vs. Structure**: Connectome data provide a snapshot of fixed structural connectivity, but biological intelligence relies on synaptic plasticity, neuromodulation, and ongoing credit assignment.

GraftBrain connects proxy or imported graphs using targeted role-based interfaces, shared cell-type bridges, and reciprocal central links. These artificial grafts connect separate graphs computationally; their endpoints do not establish connections between the original biological specimens.

Furthermore, we couple the grafted core with specialized biological circuits—specifically the **Central Complex (CX) ring attractor** (Seelig & Jayaraman, 2015; Hulse et al., 2021) and the **Mushroom Body (MB)** associative learning network (Aso et al., 2014)—and train the entire system closed-loop in a 3D physics environment.

---

## 2. Related Work

### 2.1 Whole-Brain Drosophila Connectomics
The completion of the FlyWire FAFB whole-brain connectome (Dorkenwald et al., 2024; Schlegel et al., 2024) and the Janelia Hemibrain (Scheffer et al., 2020) provides an unprecedented catalog of over 139,000 neurons and 50 million synaptic connections, annotated by cell type and neurotransmitter predictions. Shiu et al. (2024) constructed a leaky integrate-and-fire (LIF) whole-brain simulation based on FlyWire, demonstrating that resting-state dynamics and localized sensory activations spread through identifiable anatomical pathways. In sensory processing, Lappalainen et al. (2024) demonstrated that connectome-constrained recurrent neural networks predict neural responses in the fly visual system (T4/T5 motion detection pathways) with greater biological accuracy than unconstrained artificial models.

### 2.2 Central Complex Ring Attractors and Vector Navigation
In *Drosophila*, spatial navigation and heading estimation are computed in the central complex, specifically within the ellipsoid body (EB) and protocerebral bridge (PB). Experimental and theoretical studies have established that E-PG (Ellipsoid body-Protocerebral bridge-Gall) compass neurons form a continuous ring attractor network (Kim et al., 2017; Turner-Evans et al., 2020). An localized bump of neural activity represents the insect's current head direction in azimuthal space. This bump is continuously updated via two complementary mechanisms: self-motion integration (angular velocity signals delivered by P-EN neurons) and visual landmark anchoring (delivered by ring neurons).

### 2.3 Mushroom Body Plasticity and Neuromodulation
Associative learning and cognitive memory in insects reside primarily in the mushroom body (Cognigni et al., 2018). High-dimensional sensory inputs from projection neurons are converted into a sparse, decorrelated representation across thousands of Kenyon Cells (KCs) via claw morphology and winner-take-all feedback inhibition (Litwin-Kumar et al., 2017). Dopaminergic neurons (DANs) compartmentalize the mushroom body lobes into distinct functional units that signal reward (e.g., PAM cluster) or punishment/aversive states (e.g., PPL1 cluster). DAN activation alters the synaptic weight between active KCs and Mushroom Body Output Neurons (MBONs), directly driving approach or avoidance behavior.

### 2.4 Adaptive Sparse Connectivity

Sparse Evolutionary Training (SET) evolves initially sparse artificial connectivity during learning ([Mocanu et al., 2018](https://www.nature.com/articles/s41467-018-04316-3)). RigL updates sparse support using weight magnitudes and infrequent gradient calculations ([Evci et al., 2020](https://proceedings.mlr.press/v119/evci20a.html)). GraftBrain's bounded candidate-pool extension uses pre/post coactivity to score dormant candidates, rather than RigL's dormant-edge gradient growth rule. It is an engineering extension to this lab, without a sparse-training novelty claim. Results from those prior methods do not establish results for this implementation.

---

## 3. Multi-Connectome Ingestion and Grafting Architecture

### 3.1 Mathematical Formulation of Connectome Datasets
Each proxy or imported source $k \in \{1, \dots, K\}$ is represented as a directed, signed, attributed graph:
$$\mathcal{G}_k = (\mathcal{V}_k, \mathcal{E}_k, \mathbf{W}_k, \mathbf{S}_k, \mathbf{T}_k, \mathbf{R}_k)$$
where:
* $\mathcal{V}_k$ is the set of $N_k$ nodes (reconstructed neurons only when an actual dataset is imported).
* $\mathcal{E}_k \subseteq \mathcal{V}_k \times \mathcal{V}_k$ is the set of directed synaptic connections $(i, j)$ from presynaptic neuron $i$ to postsynaptic neuron $j$.
* $\mathbf{W}_k \in \mathbb{R}_{\ge 0}^{|\mathcal{E}_k|}$ represents source edge weights. Proxy weights are synthetic; imported weights depend on the source and preprocessing.
* $\mathbf{S}_k \in \{-1, +1\}^{|\mathcal{E}_k|}$ represents synaptic sign determined by predicted neurotransmitter identity:
  $$\mathbf{S}_k(i, j) = \begin{cases} -1 & \text{if } \text{NT}(i) \in \{\text{GABA}, \text{Glutamate}\} \\ +1 & \text{if } \text{NT}(i) \in \{\text{Acetylcholine}\} \end{cases}$$
  *(In the insect central nervous system, glutamate gates inhibitory glutamate-gated chloride channels [GluCl], acting primarily as an inhibitory neurotransmitter).*
* $\mathbf{T}_k$ is the cell-type classification string (e.g., `E-PG`, `P-EN`, `L1-L5`, `DNp01`).
* $\mathbf{R}_k \in \{\text{sensory}, \text{visual}, \text{central}, \text{descending}, \text{ascending}, \text{motor}\}$ denotes the macroscopic functional role.

To bound CPU workload, imported releases can be sub-sampled to a configured neuron budget (default $N_{\text{budget}} = 1,500$ per source) by ranking neurons according to total synaptic degree:
$$\deg(i) = \sum_{j} \mathbf{W}(i, j) + \sum_{j} \mathbf{W}(j, i)$$

Table 1 lists the eight source families represented by the proxy names. It is background metadata, not evidence that a run imported all these releases; inspect that run's source specifications and provenance.

| Dataset Name | Anatomical Region | Specimen Sex / Stage | Structural Modality | License |
|---|---|---|---|---|
| **FlyWire (FAFB v783)** | Whole Brain | Adult Female | Serial-section TEM | CC-BY-NC-4.0 |
| **Hemibrain v1.2.1** | Central Brain | Adult Female | FIB-SEM | CC-BY-4.0 |
| **MaleCNS v1.0** | Brain + Nerve Cord | Adult Male | FIB-SEM | CC-BY-4.0 |
| **BANC** | Brain + Nerve Cord | Adult Female | Serial-section TEM | CC-BY-4.0 |
| **Optic Lobe** | Visual Neuropils | Adult Female | FIB-SEM | CC-BY-4.0 |
| **MANC v1.2.1** | Ventral Nerve Cord | Adult Male | FIB-SEM | CC-BY-4.0 |
| **FANC** | Ventral Nerve Cord | Adult Female | Serial-section TEM | Custom / Academic |
| **Larval Connectome** | Full CNS | 1st Instar Larva | Serial-section TEM | CC-BY-4.0 |

*Table 1: Drosophila connectome source families. Actual import availability and licenses must be checked for the chosen release.*

### 3.2 Multi-Site Grafting Formulation
Let $\mathcal{V} = \bigcup_{k=1}^K \mathcal{V}_k$ denote the unified population of $N = \sum_{k=1}^K N_k$ nodes, indexed globally using index offsets $O_k = \sum_{m=1}^{k-1} N_m$. Within-source connections form the frozen base matrix $\mathbf{W}_{\text{base}}$. Between-source connections form the artificial candidate graft records $\mathcal{E}_{\text{graft}}$ with trainable weights $\mathbf{W}_{\text{graft}}$.

Grafting connections are established across three distinct structural mechanisms:

#### 1. Anatomical Interface Sites
Connections are instantiated to reflect the known macro-architecture of the insect nervous system:
* **Descending Pathways (Neck Connective)**: Brain descending neurons ($\mathbf{R}_a = \text{descending}$) project into nerve-cord premotor and motor circuits ($\mathbf{R}_b \in \{\text{motor}, \text{central}\}$):
  $$\mathcal{E}_{\text{desc}} = \{(u, v) \mid u \in \mathcal{V}_a^{\text{desc}}, v \in \mathcal{V}_b^{\text{motor}}\}$$
* **Ascending Pathways**: Nerve-cord ascending neurons ($\mathbf{R}_b = \text{ascending}$) project into central brain targets ($\mathbf{R}_a = \text{central}$), providing proprioceptive and motor-state feedback.
* **Optic Projection Pathways**: Optic lobe visual output neurons ($\mathbf{R}_a = \text{visual}$) project into central brain visual and anterior neuropils.

#### 2. Homologous Cell-Type Sites
Where distinct datasets share standardized anatomical cell-type nomenclature (e.g., mutual E-PG or Kenyon Cell annotations across FlyWire, Hemibrain, and MaleCNS), shared cell-type subsets $\mathcal{T}_{a, b} = \mathbf{T}_a \cap \mathbf{T}_b$ are linked via reciprocal homologous coupling:
$$\mathcal{E}_{\text{homolog}} = \{(u, v) \mid \mathbf{T}_a(u) = \mathbf{T}_b(v)\}$$

#### 3. Distributed Reciprocal Central Bridges
To allow distributed, non-bottlenecked cross-talk across distinct central neuropils, $K_{\text{bridge}} = 3$ independent, spatially separated clusters of central neurons ($\mathbf{R} = \text{central}$) are reciprocally cross-wired between each pair of connectomes.

Graph counts depend on the source specifications, seed, and budget. The runtime reports candidate edge records separately from active records. Duplicate endpoints remain separate records and are not counted as distinct anatomical synapses.

### 3.3 Observation-Dependent Routing and Budgeted Structural Plasticity

A linear router reads the current public observation $o_t$, without access to hidden world answers. For site $s$, its gain is

$$g_s(o_t) = 2\,\operatorname{sigmoid}(A_s o_t + b_s), \qquad W^{\mathrm{eff}}_e(t) = W_e\,m_e\,g_{s(e)}(o_t).$$

Router parameters start at zero, so initial gains equal one. Gains are computed once per world action and held across its recurrent microsteps. Forward dynamics, graft current, and flow telemetry use the same effective-weight rule. A separate Adam optimizer with learning rate $10^{-4}$ trains the router through the existing policy/value gradient; a $10^{-3}(\operatorname{mean}(g)-1)^2$ regularizer discourages uniform gain drift. Zero-site graphs have an empty router and finite zero diagnostics.

For each nonempty bucket $b=(\text{site},\text{source graph},\text{target graph})$ containing $E_b$ candidate records, the active mask budget is

$$B_b=\max(1,\min(E_b,\operatorname{round}(d E_b))), \qquad d=0.6\text{ by default}.$$

A locally seeded generator selects the initial mask, shared across the four sparse control modes. Detached unmasked pre/post coactivity EMAs update once per action, including dormant records. After a two-update warmup, structural modes attempt a rewire every ten complete optimizer updates. Within each bucket they prune up to $\lfloor0.1B_b\rfloor$ low-$|W_e|\cdot\mathrm{coactivity}_e$ active records, bounded by the number of dormant records, and grow the same number of high-coactivity previously dormant records. Stable candidate indices break ties; just-pruned records cannot immediately regrow. Tiny or fully active buckets may no-op.

Regrown weights recover their original candidate signed magnitudes. Pruned and regrown records have eligibility, pending Hebbian changes, and Adam moments cleared. Dormant records receive no Hebbian weight update. Structural changes occur only after backpropagation, both optimizer steps, and pending Hebbian application finish, preserving the retained rollout graph. Each directional budget is conserved and the frozen base remains unchanged.

The modes are `legacy` (all candidates, unit gains), `static` (fixed sparse mask), `routing` (fixed sparse mask plus learned gains), `structural` (sparse prune/regrow, unit gains), and `adaptive` (both mechanisms). This changes active support among fixed endpoints; it does not generate arbitrary connections or model biological growth. Candidate-sized tensors remain allocated and computed, so support sparsity alone establishes neither faster execution nor lower memory use.

Checkpoint state includes mask and candidate identity, bucket budgets, coactivity, structural clock, swap counters, router and optimizer state, settings, and local generator state. Restore validates candidate endpoints/sites. Legacy checkpoints retain all-active unit-gain behavior; changing a saved run's mode requires a fresh run. Resume restores neural plasticity state, not an exact world episode or outstanding autograd trajectory.

```powershell
python -m graftbrain run --fresh --graft-mode adaptive --state state/adaptive-grafts.pt
python -m graftbrain run --state state/adaptive-grafts.pt
```

---

### 3.4 Learning-Informed Recruitment and Protected Integration Time

The optional gradient policy observes upstream derivatives of the effective candidate-weight vector before propagation through the active mask. Multiplication by that action's detached site gain gives latent raw-weight sensitivity without making dormant records transmit activity. Signed derivatives sum over each retained rollout; an absolute-gradient EMA is finalized once after backward. Gradient recruitment requires positive, finite evidence. This is sensitivity to the implemented combined actor-critic objective through within-action recurrent microsteps, not direct reward attribution or cross-world-step credit.

All growth policies retain the same coactivity-weight pruning utility. Regrowth can use original signed weights or zero initialization; the latter follows an aspect of [RigL](https://proceedings.mlr.press/v119/evci20a.html), while our graph, objective and pruning rule differ. Newly activated records can be protected by a minimum age in completed learner updates. Budgets remain fixed even when protection reduces actual turnover. Cohort diagnostics track recently activated versus incumbent gradient scales, finite evidence and structural events.

Residency is a proposed edge-level adaptation of concerns raised by [Lillo and Cheney's October 2026 preprint](https://arxiv.org/abs/2605.15435v3), which studies growth and integration of units/channels. Its results do not establish effectiveness for our recurrent grafts. [Module-specific dynamic-sparsity findings in RL](https://arxiv.org/abs/2510.12096v1) motivate isolating the graft intervention from ANN resets. Neither source establishes a task-performance advantage here.

The new benchmark compares coactivity, gradient and random recruitment at ages 0/20 against a matched no-rewiring control, with common zero initialization. Balanced frozen go/no-go scenarios, zero-action references, paired pre/post changes and independent training-seed identities prevent absolute no-go success from masquerading as learning. A separate supervised target-switch probe measures current-error changes under sham and equal-budget support replacement, with and without rewiring. Such replacement may help immediately; it is not necessarily damage. These controlled synthetic measurements and source fingerprints are engineering evidence, not biological validation.

## 4. Dynamical Rate Model and Biological Sub-Circuits

### 4.1 Vectorized Dynamical Rate Model
Neural activation is modeled as a continuous, bounded firing-rate dynamical system evaluated over $M = 4$ microsteps per simulation time-step $\Delta t = 0.1\text{ s}$. For neuron $i \in \{1, \dots, N\}$, the total synaptic current $x_i$ and updated firing rate $r_i \in [0, 1]$ are governed by:

$$x_i(t) = \sum_{j \in \mathcal{N}_{\text{base}}(i)} \mathbf{W}_{\text{base}}(j, i) r_j(t) + \sum_{k \in \mathcal{N}_{\text{graft}}(i)} \mathbf{W}^{\mathrm{eff}}_{\text{graft}}(k, i, t) r_k(t) + b_i + I_{\text{drive}, i}(t)$$

$$r_i(t + \delta\tau) = (1 - \alpha) r_i(t) + \alpha \cdot \text{ReLU}(\tanh(x_i(t))) + \sigma \xi_i(t)$$

where $\alpha = 0.35$ is the leak rate, $b_i$ is intrinsic baseline excitability, $\sigma = 0.01$ is background Gaussian membrane noise $\xi_i \sim \mathcal{N}(0, 1)$, and $I_{\text{drive}, i}$ represents injected sensory drive.

```
Algorithm 1: Vectorized Connectome Microstep via index_add
--------------------------------------------------------------------------------
Input: Presynaptic rates r in [0, 1]^N, drive I_drive in R^N, bias b in R^N
Output: Updated post-synaptic rates r' in [0, 1]^N

1: x <- zeros(N)
2: x <- x.index_add(dim=0, index=base_post, tensor=base_w * r[base_pre])
3: x <- x.index_add(dim=0, index=graft_post, tensor=effective_graft_w * r[graft_pre])
4: x <- x + b + I_drive
5: r' <- (1 - alpha) * r + alpha * clamp(tanh(x), min=0.0)
6: if noise > 0 then r' <- r' + noise * randn(N)
7: r' <- clamp(r', min=0.0, max=1.0)
8: return r'
```

### 4.2 Homeostatic Plasticity Regulation
In recurrent networks of biological scale, unbalanced positive feedback loops can precipitate either quiescent extinction ($r \to 0$) or pathological saturation ($r \to 1$). To maintain the connectome within a responsive dynamic regime, we implement continuous homeostatic firing rate adaptation:
$$\bar{r}_i(t) = (1 - \beta_{\text{homeo}}) \bar{r}_i(t - 1) + \beta_{\text{homeo}} r_i(t)$$
$$b_i(t + 1) = \text{clamp}\left(b_i(t) + \eta_{\text{homeo}} \left(r^* - \bar{r}_i(t)\right), -0.5, +0.5\right)$$
where $\beta_{\text{homeo}} = 0.01$, $\eta_{\text{homeo}} = 0.0005$, and $r^* = 0.12$ is the target mean population firing rate.

### 4.3 Central Complex (CX) Ring Attractor Compass
To provide the agent with a robust internal coordinate frame for spatial navigation, we model the Ellipsoid Body (EB) as a continuous ring attractor comprising $K_{\text{wedge}} = 16$ discrete angular columns covering $\theta \in [-\pi, +\pi)$.

The internal heading angle $\theta_{\text{bump}}$ is updated dynamically via:
$$\theta_{\text{bump}}(t + 1) = \theta_{\text{bump}}(t) + \omega(t) \Delta t + \kappa_{\text{vis}} \Delta\theta_{\text{landmark}}$$
where $\omega(t)$ is the angular velocity from proprioception, and $\Delta\theta_{\text{landmark}}$ is the visual angular discrepancy between the heading bump and the brightest/nearest visual contrast detected by the retina.

The activation $c_k$ of compass wedge $k$ with preferred angle $\phi_k = -\pi + k \frac{2\pi}{16}$ follows a circular Gaussian tuning curve:
$$c_k = \frac{\exp\left( - \frac{\text{dist}_{S^1}(\phi_k, \theta_{\text{bump}})^2}{2 \sigma_{\text{cx}}^2} \right)}{\sum_{m=1}^{16} \exp\left( - \frac{\text{dist}_{S^1}(\phi_m, \theta_{\text{bump}})^2}{2 \sigma_{\text{cx}}^2} \right)}$$
with width $\sigma_{\text{cx}} = 0.45\text{ rad}$. The resulting 16-dimensional compass state $\mathbf{c} \in \mathbb{R}^{16}$ is injected directly into central brain and descending steering pathways.

### 4.4 Mushroom Body (MB) Associative Sub-Circuit
For associative memory and delayed-response cognitive tasks, we implement a modeled mushroom body:
1. **Sensory Claw Expansion**: Environmental cue vectors $\mathbf{u}_{\text{cue}} \in \mathbb{R}^4$ (encoding cue color and active task flags) are projected onto a population of $N_{\text{KC}} = 64$ Kenyon Cells via a fixed pseudo-random claw projection matrix $\mathbf{W}_{\text{claw}} \in \mathbb{R}^{64 \times 4}$.
2. **Winner-Take-All (WTA) Sparsity**: To mirror biological Kenyon cell coding ($~5-10\%$ sparsity; Litwin-Kumar et al., 2017), we enforce top-$k$ Winner-Take-All gating ($k = 4$, yielding exactly $6.25\%$ active KCs):
   $$a_i = \begin{cases} \text{ReLU}\left((\mathbf{W}_{\text{claw}} \mathbf{u}_{\text{cue}})_i\right) & \text{if } (\mathbf{W}_{\text{claw}} \mathbf{u}_{\text{cue}})_i \in \text{Top-}k \\ 0 & \text{otherwise} \end{cases}$$
3. **Mushroom Body Output Readout**: Plastic synaptic weights $\mathbf{W}_{\text{MBON}} \in \mathbb{R}^{2 \times 64}$ project KCs to two opponent output neurons: $\text{MBON}_+$ (approach) and $\text{MBON}_-$ (avoidance):
   $$\mathbf{y}_{\text{MBON}} = \mathbf{W}_{\text{MBON}} \mathbf{a}$$
4. **Dopaminergic Neuromodulation**: Synaptic eligibility traces $\mathbf{E}_{\text{MB}} \in \mathbb{R}^{2 \times 64}$ track paired pre- and post-synaptic co-activation:
   $$\mathbf{E}_{\text{MB}}(t) = 0.85 \mathbf{E}_{\text{MB}}(t - 1) + \tanh(\mathbf{y}_{\text{MBON}}(t)) \mathbf{a}(t)^T$$
   When primary rewards or temporal-difference errors $\delta_{\text{TD}}$ occur, simulated dopaminergic neurons (DANs) modulate the weights:
   $$\mathbf{W}_{\text{MBON}} \leftarrow \text{clamp}\left(\mathbf{W}_{\text{MBON}} + \eta_{\text{DAN}} \begin{bmatrix} +\delta_{\text{TD}} \\ -\delta_{\text{TD}} \end{bmatrix} \odot \mathbf{E}_{\text{MB}}, -1.0, +1.0\right)$$

---

## 5. Differentiable Neural Interface and Agent Embodiment

### 5.1 Multimodal Sensory Processing
The agent receives a heterogeneous sensory vector $\mathbf{o} \in \mathbb{R}^{155}$:
* **Ray-Cast Retina**: 33 rays distributed over a $120^\circ$ horizontal field of view ($[-60^\circ, +60^\circ]$) with range $d_{\max} = 9.0\text{ m}$. Each ray returns 4 channels: normalized depth $1 - d/d_{\max}$ and normalized RGB surface reflectance ($33 \times 4 = 132$ dimensions).
* **Proprioception**: 8 channels encoding arm reach ($[0.25, 0.9]\text{ m}$), arm height ($[0.0, 0.75]\text{ m}$), gripper aperture ($[0.0, 1.0]$), held object flag, linear velocity $v$, angular velocity $\omega$, and heading components $(\sin\theta, \cos\theta)$.
* **Task Context**: One-hot task indicator vector ($\mathbb{R}^9$), cue vector ($\mathbb{R}^4$), previous reward, and normalized episode progress.
* **Camera Visual Stream**: Live webcam stream (or synthetic stimulus) downsampled to $2 \times 48 \times 64$ (grayscale luminance + motion energy), processed via a 2-stage convolutional network into a 32-dimensional embedding $\mathbf{z}_{\text{cam}} \in \mathbb{R}^{32}$.

### 5.2 Sensorimotor Encoder and Decoder
The artificial neural interface connects the external world to the biological connectome:
* **Encoder MLP**: Maps the combined input $[\mathbf{o}, \mathbf{z}_{\text{cam}}, \mathbf{c}_{\text{compass}}, \mathbf{y}_{\text{MBON}}] \in \mathbb{R}^{205}$ through a 256-unit hidden layer to drive up to $N_{\text{sensory}} = 256$ designated sensory and visual nodes.
* **Decoder MLP**: Reads the firing rates of $N_{\text{motor}} = 128$ motor/descending neurons, $N_{\text{probe}} = 256$ distributed central probe neurons, and the 2 MBON channels to parameterize a Gaussian policy distribution $\pi_\theta(\mathbf{a} \mid \mathbf{s}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma})$ over 5 continuous motor actuators:
  $$\mathbf{a} = [\text{drive } v, \text{turn } \omega, \Delta\text{reach}, \Delta\text{height}, \text{grip aperture}] \in [-1, 1]^5$$
  and estimates state value $V_\phi(\mathbf{s})$.

---

## 6. Optimization and Learning Architecture

GraftBrain trains the artificial neural interface, plastic graft weights, and enabled router using a **one-pass GAE actor-critic** plus **three-factor Hebbian plasticity**. The new mechanism uses this existing learner; replacing it with a full PPO trainer is outside this increment.

### 6.1 One-Pass Actor-Critic with GAE
For a rollout trajectory of length $T = 32$, Generalized Advantage Estimation (Schulman et al., 2015) computes advantage estimates $\hat{A}_t$:
$$\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$$
$$\hat{A}_t = \delta_t + \gamma\lambda(1-d_t)\hat{A}_{t+1}$$
where $\gamma = 0.98$ and $\lambda = 0.95$.

The policy loss uses the retained sampled-action log likelihoods with detached, normalized advantages:

$$L_{\pi}=-\operatorname{mean}_t\big[\log\pi_\theta(a_t\mid s_t)\,\widetilde{A}_t\big],$$
$$L_{\mathrm{total}}=L_{\pi}+0.5\operatorname{SmoothL1}(V_\phi,\widehat{R})-0.01\operatorname{mean}(\mathcal H(\pi_\theta))+L_{\mathrm{route}}.$$

Gradients pass through the retained recurrent microsteps. There is one update per completed rollout and no old-policy reevaluation or repeated optimization epochs. The earlier implementation's stored likelihood ratio was identically one at its single update; describing that as effective clipped multi-epoch PPO overstated the implemented behavior. The policy gradient is now written directly. Rollout-boundary transitions and terminal reward modulation are included before the update.

### 6.2 Three-Factor Hebbian Plasticity on Graft Synapses
In parallel with gradient descent, active graft records accumulate a simplified three-factor Hebbian update:
$$\mathbf{E}_{\text{graft}}(t) = 0.90 \mathbf{E}_{\text{graft}}(t - 1) + \mathbf{r}_{\text{post}}(t) \odot \mathbf{r}_{\text{pre}}(t)$$
$$\Delta \mathbf{W}_{\text{graft}} = \eta_{\text{heb}} \cdot \text{clamp}(\delta_{\text{TD}}, -2, +2) \cdot \left(\mathbf{E}_{\text{graft}} - \bar{\mathbf{E}}_{\text{graft}}\right)$$
where $\delta_{\text{TD}}$ acts as a global broadcast modulation signal. The active mask gates eligibility and pending updates; weight changes are applied only when retained autograd graphs no longer require the previous weights. This is an artificial approximation, not a measured biological learning rule.

### 6.3 Intrinsic Curiosity Bonus (Random Network Distillation)
To facilitate autonomous exploration in sparse-reward environments (e.g., complex mazes and object manipulation), an intrinsic reward bonus $r_{\text{int}}$ is generated using Random Network Distillation (Burda et al., 2018):
$$r_{\text{int}}(s_t) = \min\left(0.2, \, 0.5 \cdot \|\hat{f}_\psi(s_t) - f(s_t)\|_2^2\right)$$
$$r_{\text{total}}(s_t) = r_{\text{ext}}(s_t) + \beta_{\text{int}} r_{\text{int}}(s_t)$$
where $f$ is a fixed, randomly initialized target network, $\hat{f}_\psi$ is an online predictor network trained via mean-squared error, and $\beta_{\text{int}} = 0.05$.

---

## 7. Experimental Setup and Cognitive Benchmark Suite

The lab provides a virtual environment with planar motion, simplified object manipulation, and ray-cast vision. Its nine tasks are listed below; their presence does not establish that a trained policy solves them.

```
+-----------------------------------------------------------------------------------------+
|                                    GRAFTBRAIN TASK SUITE                                |
+-----------------------+----------------------+--------------------+---------------------+
| Task Name             | Domain               | Description        | Success Criterion   |
+-----------------------+----------------------+--------------------+---------------------+
| 1. forage             | Spatial Navigation   | Open-field search  | dist(agent, goal)   |
|                       |                      | for random target  | < 0.55 m            |
| 2. maze               | Spatial Navigation   | Random DFS maze    | dist(agent, exit)   |
|                       |                      | traversal (6x6)    | < 0.60 m            |
| 3. push               | Object Manipulation  | Push heavy block   | dist(box, pad)      |
|                       |                      | to target pad      | < 0.35 m            |
| 4. grasp_place        | Fine Motor Control   | Grasp cube, lift,  | Object deposited    |
|                       |                      | deposit into bin   | in target bin       |
| 5. peg_insert         | Fine Motor Control   | Align peg, lift,   | Object inserted     |
|                       |                      | insert into hole   | in target hole      |
| 6. delayed_match      | Working Memory       | Cue -> 20-step     | Correct colored     |
|                       |                      | delay -> choice    | object selected     |
| 7. go_nogo            | Inhibitory Control   | Green: approach    | Reach / withhold    |
|                       |                      | Red: withhold 80st | correctly           |
| 8. button_sequence    | Working Memory       | Observe 3-color    | Press 3 buttons     |
|                       |                      | cue, press order   | in exact sequence   |
| 9. webcam_chase       | Closed-Loop Tracking | Chase orb steered  | Maintain tracking   |
|                       |                      | by live webcam     | proximity < 1.0 m   |
+-----------------------+----------------------+--------------------+---------------------+
```

*Table 2: Benchmark tasks evaluating spatial, motor, and cognitive capabilities.*

### 7.1 Solvability Verification via Scripted Oracle
Scripted oracle probes are available separately from policy training. Earlier output printed `10/10` for nine tasks, but that script contained no assertions and treated positive tracking return as success even when the terminal outcome was `timeout`. Those historical rows therefore do not verify all-task terminal completion, and oracle performance never establishes learned policy performance.

---

## 8. Evidence Status and Evaluation Protocol

**The historical figures in Sections 8.1 and 8.2 are unverified.** No accompanying run artifacts establish their hardware, seeds, input graphs, task outcomes, sampling uncertainty, or measurement procedure. They are retained as historical claims for auditability, not as results supporting this implementation or the adaptive-graft capability.

### 8.1 Unverified Historical Throughput Claims

Earlier prose reported the following figures for a claimed 12,000-node graph with 252,447 base records, 95,920 graft records, and four microsteps. The measurements have not been reproduced here.

```
+-----------------------------------------------------------------------------------------+
|                        FORWARD + BACKWARD MICROSTEP LATENCY (N=12,000)                  |
+--------------------------------------+---------------------------+----------------------+
| Formulation                          | Latency per Step (ms)     | Speedup Factor       |
+--------------------------------------+---------------------------+----------------------+
| PyTorch Sparse COO (torch.sparse.mm) | 28.29 ms                  | 1.00x (Baseline)     |
| PyTorch Sparse CSR (torch.sparse.mm) | 40.36 ms                  | 0.70x                |
| Vectorized index_add (GraftBrain)    |  7.90 ms                  | 3.58x                |
+--------------------------------------+---------------------------+----------------------+
```

*Table 3: Unverified historical latency and speedup claims; no supporting run artifacts.*

The same earlier text claimed 16.85 to 7.75 seconds for 100 closed-loop steps and approximately 14 steps/second on CPU. These figures do not establish a present throughput or comparative speedup. In particular, the adaptive mask still operates on candidate-sized tensors.

### 8.2 Unverified Historical Ablation Claims

The following percentages and convergence counts appeared in earlier prose without supporting experiment artifacts. The labels are preserved from that historical table, including its PPO terminology.

```
+-----------------------------------------------------------------------------------------+
|                         ABLATION STUDY ACROSS REPRESENTATIVE TASKS                      |
+------------------------------------+------------------+-----------------+---------------+
| Architecture Configuration         | Maze (Success %) | Match (Success%)| Steps to Conv |
+------------------------------------+------------------+-----------------+---------------+
| Base Connectome + PPO Only         | 18.4%            | 31.2%           | ~45,000       |
| + Homeostatic Regulation           | 24.6%            | 33.5%           | ~38,000       |
| + Central Complex (CX) Compass     | 58.2%            | 34.0%           | ~22,000       |
| + Mushroom Body (MB) Circuit       | 59.1%            | 68.7%           | ~14,000       |
| Full System (+ RND Curiosity)      | 67.4%            | 72.1%           | ~11,000       |
+------------------------------------+------------------+-----------------+---------------+
```

*Table 4: Unverified historical ablation percentages and convergence counts; no supporting run artifacts.*

No causal effect of the compass, Mushroom Body, curiosity, or homeostasis follows from this undocumented table. Increases in maze or delayed-match success, faster convergence, resistance to catastrophic interference, and stability beyond 50,000 steps remain unverified claims.

### 8.3 Bounded Adaptive-Graft Evaluation

The `benchmark-plasticity` command runs `legacy`, `static`, `routing`, `structural`, and `adaptive` with paired graph/training seeds, identical task and step budgets, and identical candidate graphs. Sparse arms share their initial mask. A saved manifest lists development and held-out evaluation seeds, configuration, and code revision/dirty status; episode records preserve returns and actual terminal outcomes. Summaries report active counts, support turnover, gain spread, and wall-clock cost.

Held-out evaluation uses policy mean actions and resets episodic neural/compass state between episodes. Learning, Hebbian modulation, RND training, router updates, structural swaps, homeostatic bias adaptation, and coactivity/eligibility accumulation are frozen; firing activity may still evolve within each episode. Learned weights, router parameters, and structural support are retained. This prevents evaluation-time adaptation from being confused with frozen-policy performance.

An explicit bounded pilot command is:

```powershell
python -m graftbrain benchmark-plasticity --out runs/plasticity-pilot --train-steps 512 --tasks forage go_nogo --seeds 0 --eval-seeds 1000 1001 --max-neurons 400 --microsteps 2
```

The output path must be new. These defaults use two synthetic proxies (FlyWire and MANC), 512 training environment steps per mode/task and training seed, and two held-out world seeds. Budget-truncated training episodes are nonterminal `budget_exhausted` records; evaluation outcomes use terminal success/fail/timeout labels. No adaptive-graft task-performance result is asserted in this document. Small synthetic CPU pilots support only the mechanisms and budgets actually checked, not general reliability or a task advantage.

---

## 9. Discussion and Limitations

### 9.1 Point-Neuron Abstraction vs. Biophysical Spiking
GraftBrain models neural activation using a bounded continuous firing-rate equation rather than multi-compartmental Hodgkin-Huxley or conductance-based Leaky Integrate-and-Fire (LIF) models. Rate equations abstract away spike-timing-dependent plasticity (STDP) and dendritic branch computations. They permit:
1. Exact differentiability, enabling gradient backpropagation through recurrent connectome dynamics using standard autograd engines.
2. Vectorized CPU simulation whose wall-clock cost can be measured for a specific graph and run. No universal real-time throughput is established here.

### 9.2 Biological Grafting vs. Unconstrained Synthetic Topologies
Whether imported connectome constraints improve an agent over unconstrained MLPs, transformers, or matched synthetic graphs remains an experimental question. Heavy-tailed degree distributions, modularity, and role separation are candidate inductive biases. The present evidence does not establish resistance to catastrophic forgetting, better recurrent stability, or an advantage from biological connectivity. Default synthetic proxies further limit any anatomical interpretation.

### 9.3 Bounded Support, State, and Evaluation

The router sees public observations, while structural adaptation searches only the prebuilt candidate pool. A fixed per-bucket record budget is an implementation invariant, not a measure of biological synaptic cost. Duplicate endpoints, simplified signs, artificial cross-specimen grafts, and continuous firing rates limit anatomical claims. Checkpoint restoration covers neural/plasticity state and its next decisions under identical inputs; full-world exact continuation is not provided. Five-mode paired controls and held-out episodes are necessary for comparative claims, and short synthetic pilots remain mechanism checks.

---

## 10. Conclusion

GraftBrain provides a local testbed for embodied graph dynamics with artificial multi-site grafts. Observation-dependent routing and budget-conserving support adaptation extend that testbed while preserving fixed candidate endpoints and frozen base wiring. Reproducible mechanism tests and paired frozen-policy evaluation can assess the implementation. Historical undocumented figures do not establish improved learning, biological fidelity, or computational acceleration.

---

## References

* Aso, Y., et al. (2014). The neuronal architecture of the mushroom body provides a logic for associative learning. *eLife*, 3, e04577.
* Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018). Exploration by random network distillation. *arXiv preprint arXiv:1810.12894*.
* Cognigni, P., Felsenberg, J., & Waddell, S. (2018). Do the right thing: neural network mechanisms of memory formation, expression and update in *Drosophila*. *Current Opinion in Neurobiology*, 52, 18–29.
* Dorkenwald, S., et al. (2024). Neuronal wiring diagram of an adult brain. *Nature*, 634(8032), 124–138.
* Evci, U., Gale, T., Menick, J., Castro, P. S., & Elsen, E. (2020). [Rigging the Lottery: Making All Tickets Winners](https://proceedings.mlr.press/v119/evci20a.html). *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 2943–2952.
* Hulse, B. K., et al. (2021). A connectome of the *Drosophila* central complex reveals network motifs suitable for flexible navigation and goal-directed behavior. *eLife*, 10, e66039.
* Kim, S. S., Rouault, H., Druckmann, S., & Jayaraman, V. (2017). Ring attractor dynamics in the *Drosophila* central brain. *Science*, 356(6340), 849–853.
* Lappalainen, J. K., et al. (2024). Connectome-constrained networks predict neural activity in the *Drosophila* visual system. *Nature*, 634(8032), 166–174.
* LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.
* Litwin-Kumar, A., Harris, K. D., Axel, R., Sompolinsky, H., & Abbott, L. F. (2017). Optimal degrees of synaptic connectivity for mushroom body function. *Neuron*, 93(5), 1153–1164.
* Mocanu, D. C., et al. (2018). [Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science](https://www.nature.com/articles/s41467-018-04316-3). *Nature Communications*, 9, 2383.
* Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to neuromorphic hardware. *IEEE Signal Processing Magazine*, 36(6), 51–63.
* Scheffer, L. K., et al. (2020). A connectome and analysis of the adult *Drosophila* central brain. *eLife*, 9, e57443.
* Schlegel, P., et al. (2024). Whole-brain annotation and multi-connectome mapping of *Drosophila*. *Nature*, 634(8032), 139–152.
* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
* Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*.
* Seelig, J. D., & Jayaraman, V. (2015). Neural dynamics for landmark orientation and angular path integration. *Nature*, 521(7551), 186–191.
* Shiu, P. K., et al. (2024). A leaky integrate-and-fire computational model of the adult *Drosophila* brain. *Nature*, 634(8032), 153–165.
* Takemura, S., et al. (2023). A connectome of the male *Drosophila* ventral nerve cord. *bioRxiv*, 2023-06.
* Turner-Evans, D. B., et al. (2020). The neuroanatomical ultrastructure and function of a heading direction circuit in *Drosophila*. *Neuron*, 108(1), 145–163.
* Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
