# SYNAPSE Data Card

## Dataset & Environment Specifications

### 1. Embodied Simulation Worlds (GraftBrain Suite)
The vendored world has nine native tasks with planar body physics and an arm-height
coordinate. Browser rendering is 3D; it is not a general 3D physics engine:
1. `forage`: Locate and consume food targets while avoiding boundary hazards and managing energy decay.
2. `maze`: Multi-corridor navigation requiring working memory and wall avoidance.
3. `push`: Manipulate external objects into designated target zones.
4. `grasp_place`: Position arm endpoint, engage grip, and transport objects.
5. `peg_insert`: High-precision 3D alignment and insertion task.
6. `delayed_match`: Encode color/spatial cue, navigate through an occlusion delay, and select matching resource.
7. `go_nogo`: Discriminative action selection conditioned on cue frequency/polarity.
8. `button_sequence`: Sequential activation of multi-state targets in ordered permutations.
9. `webcam_chase`: Track an orb that can be driven by webcam motion in the upstream environment. No live camera collection was used in this study.

`auto` is a curriculum mode rather than an additional native task.

### 2. Observation Structure (155 continuous float32 values)
- `[0:33]`: Ray depth sensor (33 visual rangefinder rays across 120° FOV).
- `[33:66]`: Red channel visual contrast.
- `[66:99]`: Green channel visual contrast.
- `[99:132]`: Blue channel visual contrast.
- `[132:136]`: Normalized arm reach, arm height, grip, and whether an object is held.
- `[136:140]`: Normalized forward/angular velocity, sine and cosine of body heading.
- `[140:149]`: One-hot native task identifier (the order in `graftbrain.world.TASKS`).
- `[149:153]`: Environmental cues (4-dimensional associative cues for Mushroom Body).
- `[153]`: Previous observed reward.
- `[154]`: Current time divided by the episode horizon.

These coordinates come from `World.observe`; agent x/y and physiological resources
are not directly present in this observation vector. SYNAPSE's resource frame is
an explicitly synthetic adapter. The experience-model encoder excludes the task
one-hot and previous-reward coordinates, summarizing visible rays, motion, cue
contrasts, time and current graph/donor activity instead. The full original actor
still receives the complete 155-dimensional observation, including task identity.

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

### 5. Measured Experience Studies (6 November 2026)

Native development experiments use separately seeded support and query worlds,
fixed task types, shared exploratory support, and frozen query learning. Original
seeds were 7/11/19; the subsequent frozen-gate study used 23/29/31/37/41. Only forage
and push were evaluated for comparative performance. Query reward is never used
for fitting or policy selection. Training the predictor on actual rewards is
support adaptation, not zero-shot performance.

The separate controlled stream is synthetic: hidden reward signs +1/-1/+1 create
reversal and recurrence, with labels available to the scorer only. It is not
biological data or an embodied transfer experiment. Raw local receipts, source
fingerprints, costs and signed results are described in the
[results report](results/2026-11-06-experience-adaptation.md). No live camera data,
personal data, or paid external compute was collected.
