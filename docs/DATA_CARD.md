# SYNAPSE Data Card

## Dataset & Environment Specifications

### 1. Embodied Simulation Worlds (GraftBrain Suite)
The agent operates across 9 continuous 2D/3D physics backends:
1. `forage`: Locate and consume food targets while avoiding boundary hazards and managing energy decay.
2. `maze`: Multi-corridor navigation requiring working memory and wall avoidance.
3. `push`: Manipulate external objects into designated target zones.
4. `grasp_place`: Position arm endpoint, engage grip, and transport objects.
5. `peg_insert`: High-precision 3D alignment and insertion task.
6. `delayed_match`: Encode color/spatial cue, navigate through an occlusion delay, and select matching resource.
7. `go_nogo`: Discriminative action selection conditioned on cue frequency/polarity.
8. `button_sequence`: Sequential activation of multi-state targets in ordered permutations.
9. `auto`: Adaptive curriculum cycling through tasks based on success rates.

### 2. Observation Structure (155 continuous float32 values)
- `[0:33]`: Ray depth sensor (33 visual rangefinder rays across 120° FOV).
- `[33:66]`: Red channel visual contrast.
- `[66:99]`: Green channel visual contrast.
- `[99:132]`: Blue channel visual contrast.
- `[132:136]`: Proprioception (agent x, y position, velocity, and body heading $\theta$).
- `[136:140]`: Arm state (reach extension, vertical height, grip pressure, contact sensor).
- `[140:149]`: Homeostatic indicators (energy, hydration, stamina, exposure).
- `[149:153]`: Environmental cues (4-dimensional associative cues for Mushroom Body).
- `[153:155]`: Task metadata (current progress fraction, reward feedback).

### 3. Connectome Topologies
- **FlyWire Profile**: Synthetic model of central Drosophila brain neuropils with 25% inhibitory GABAergic/glutamatergic fractions.
- **MANC Profile**: Ventral nerve cord motor and ascending/descending interneuron clusters.
- **Optic Lobe Profile**: Visual projection columns with retinotopic receptive fields.
- **Janelia Male CNS v1.0 (Optional External Data)**: Full male fly CNS connectome (11,751 cell types, 3,830,931 type-to-type edges, CC BY 4.0). Available when `malecns_types.npz` is loaded.

### 4. RSI Benchmarks
- Synthetic feature evaluation splits:
  - Development families `0-1`: Initial policy search and hyperparameter calibration.
  - Validation families `2-3`: Intermediate meta-selection and mutation screening.
  - Sealed families `4-5`: Strictly isolated query tasks evaluated post-adaptation with external writer disabled.
