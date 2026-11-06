# SYNAPSE Data Card

## Dataset & Environment Specifications

### 1. Embodied Simulation Worlds

SYNAPSE supports both the 9 native GraftBrain simulation tasks and 4 advanced frontier environments implemented in `ComplexWorld`:

#### Native Tasks
1. `forage`: Locate and consume food targets while avoiding boundary hazards and managing energy decay.
2. `maze`: Multi-corridor navigation requiring working memory and wall avoidance.
3. `push`: Manipulate external objects into designated target zones.
4. `grasp_place`: Position arm endpoint, engage grip, and transport objects.
5. `peg_insert`: High-precision 3D alignment and insertion task.
6. `delayed_match`: Encode color/spatial cue, navigate through an occlusion delay, and select matching resource.
7. `go_nogo`: Discriminative action selection conditioned on cue frequency/polarity.
8. `button_sequence`: Sequential activation of multi-state targets in ordered permutations.
9. `webcam_chase`: Track an orb that can be driven by webcam motion in the upstream environment.

#### Frontier Complex Ecology Tasks (`ComplexWorld`)
10. `ecosystem`: Comprehensive virtual biosphere combining autonomous moving predator drones, variable terrain friction patches (ice, mud, sand), food targets, dynamic gates, and day/night illumination cycles.
11. `predator_evasion`: High-threat evasion arena where the agent must out-maneuver pursuit drones, conserve stamina, and utilize terrain friction to escape predation.
12. `dynamic_arena`: Multi-room navigational maze featuring harmonic oscillating gates and variable friction surfaces testing spatial mapping under traction variations.
13. `puzzle_chamber`: Multi-stage logic environment where the agent must press interactive floor triggers/buttons to open pressure-locked security gates protecting high-yield food caches.


#### SAIL Verifiable Curriculum & ReLIFT Tasks (`synapse.verified_curriculum`)
14. `verifiable_relift`: Procedurally generated sequential challenge stages with strictly verifiable intermediate subgoals:
    - `CUE_DISCOVERED`: Visual recognition and approach toward target color cues.
    - `GATE_TRIGGERED`: Exact physical manipulation of spatial floor switches/buttons.
    - `BARRIER_PASSED`: Safe traversal through dynamic barrier passages without collision.
    - `TARGET_REACHED`: Successful goal acquisition and consumption.
    - Each transition and subgoal verification is cryptographically anchored by a SHA-256 execution receipt.

#### SAIL 4-Arm Continual Learning Milestone (`synapse.flagship_benchmark`)
- Standardized sequential curriculum evaluating catastrophic forgetting:
  $$\text{Task}_1 (\text{forage}) \rightarrow \text{Task}_2 (\text{delayed\_match}) \rightarrow \text{Task}_3 (\text{push}) \rightarrow \text{Task}_4 (\text{ecosystem})$$
- Evaluates four comparative arms:
  1. `fixed_gradient`: Standard Adam backpropagation baseline (prone to catastrophic forgetting).
  2. `dynamic_plasticity`: Structural neurogenesis + Hebbian weight plasticity.
  3. `continual_sleep`: Trainable student-teacher sleep consolidation.
  4. `full_developmental_sail`: Full integration (Continual Backprop + Fast Memory + Sleep + MPC Planning).

### 2. Dynamic Physics & Ecological Mechanics

- **Autonomous Predator Drones**: Drones execute continuous pursuit vectors toward the agent at speed $v_{\text{predator}} = 0.04$, generating a continuous distance-dependent threat field:
  $$\text{Threat}(d) = \max\left(0, 1 - \frac{d}{R_{\text{hazard}}}\right)$$
- **Terrain Friction Zones**: Continuous rectangular patches modifying effective friction and stamina expenditure:
  - *Ice*: Low friction ($0.1\times$), high slip, low stamina cost.
  - *Mud*: High friction ($2.5\times$), low slip, high stamina cost.
  - *Sand*: Moderate resistance ($1.8\times$).
- **Dynamic & Puzzle Gates**:
  - *Harmonic Oscillating Gates*: Cycle between open (passable) and closed (solid barrier) states on fixed periodic intervals.
  - *Button-Locked Gates*: Remain solid barriers until the agent depresses an associated trigger button in the environment.
- **Diurnal Ambient Illumination Cycle**: Sine-wave ambient lighting modulating visual rangefinder rays and sensory contrast:
  $$I_{\text{ambient}}(t) = 0.5 + 0.4 \cos\left(\frac{2\pi t}{T_{\text{day}}}\right)$$

### 3. Observation Structure (155 continuous float32 values)
- `[0:33]`: Ray depth sensor (33 visual rangefinder rays across 120° FOV), modulated by ambient light and ray occlusion.
- `[33:66]`: Red channel visual contrast.
- `[66:99]`: Green channel visual contrast.
- `[99:132]`: Blue channel visual contrast.
- `[132:136]`: Normalized arm reach, arm height, grip, and whether an object is held.
- `[136:140]`: Normalized forward/angular velocity, sine and cosine of body heading.
- `[140:149]`: One-hot native/frontier task identifier.
- `[149:153]`: Environmental associative cues for Mushroom Body olfactory memory.
- `[153]`: Previous observed reward.
- `[154]`: Current time divided by episode horizon.

### 4. Connectome Topologies
- **FlyWire Profile**: Synthetic model of central Drosophila brain neuropils with 25% inhibitory GABAergic/glutamatergic fractions.
- **MANC Profile**: Ventral nerve cord motor and ascending/descending interneuron clusters.
- **Optic Lobe Profile**: Visual projection columns with retinotopic receptive fields.
- **Janelia Male CNS v1.0 (Optional External Data)**: Full male fly CNS connectome (11,751 cell types, 3,830,931 type-to-type edges, CC BY 4.0), loaded from `malecns_types.npz`.

### 5. RSI Benchmarks & Task Partitions
- Synthetic feature evaluation splits:
  - Development families `0-1`: Initial policy search and hyperparameter calibration.
  - Validation families `2-3`: Intermediate meta-selection and mutation screening.
  - Sealed families `4-5`: Strictly isolated query tasks evaluated post-adaptation with external writer disabled.
