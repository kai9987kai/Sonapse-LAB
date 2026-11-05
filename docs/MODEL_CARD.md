# SYNAPSE Model Card

## Model Details
- **Model Name**: SYNAPSE Integrated Connectome & Cognitive Agent
- **Version**: 0.1.0
- **Model Type**: Hybrid Neuromorphic Connectome / Transformer MoE Controller with Direct Weight-Space Programming
- **Architecture**:
  - **Sensory Encoder**: Linear projection mapping 173 input dimensions (155-d observation + 16-d compass bump + 2-d mushroom body valence) to connectome sensory nodes.
  - **Neural Connectome Substrate**: Signed recurrent rate graph (`GraftedBrain`) combining three source families (`flywire`, `manc`, `optic-lobe`), signed Dale synapses, homeostatic balance, and 8 inter-module graft sites.
  - **Donor Cognitive Core**: Pinned trained Poseidon Tidal MoE DAgger controller (`tidal_core.pt`, SHA-256 `a053104a...`), 327,230 parameters, 4 experts, 128 hidden width, 2 recurrent steps.
  - **Bridges**: Bidirectional residual projections (`graph_to_resource` and `donor_to_graph`) synchronizing graph context with cognitive intent.
  - **Expanse Grafts**: 2 dormant expert slots recruited via novelty thresholding, and a zero-initialized native consolidation block.
  - **Motor Decoder**: Linear projection from 64-d fused feature representation to 5 continuous motor actuators (`drive`, `turn`, `reach`, `height`, `grip`), gated by average connectome activity.

## Feature Representation Schema (v1, 64-dimensional)
| Indices | Subsystem | Content |
|---|---|---|
| `[0:32]` | Graph Context | Geometrically pooled connectome activation (absolute or evoked) |
| `[32:48]` | Tidal Memory | Pretrained MoE hidden memory state slice |
| `[48:50]` | Mushroom Body | 2-d approach/avoidance valence readout |
| `[50:56]` | Tidal Intent | 6-class softmax action distribution (`rest`, `forage`, `drink`, `shelter`, `explore`, `flee`) |
| `[56:64]` | Compass | First 8 wedges of the ellipsoid body ring attractor |

## Weight Edit Surfaces & Permissions
| Parameter Surface | Editable in Patch | Trainable in Preparation | Frozen |
|---|---|---|---|
| `brain.graft_w` | Yes | Yes | No |
| `plasticity.router.*` | Yes | Yes | No |
| `mb.w_mbon` | Yes | Yes (Hebbian/Dopamine) | No |
| `encoder.*` | Yes | Yes | No |
| `donor_to_graph.*` | Yes | Yes | No |
| `graph_to_resource.*`| Yes | Yes | No |
| `motor.*` | Yes | Yes | No |
| `dormant.*` | Yes | Yes | No |
| `native.*` | Yes | Yes (Consolidation) | No |
| `tidal.*` (Tidal Core)| **No** | **No** | **Frozen** |
| `curiosity.*` | **No** | Yes (RND updates) | No |

## Evaluation and Intended Usage
- **Primary Use**: Embodied reinforcement learning, cue-delay navigation, memory consolidation, and recursive self-improvement research in synthetic continuous environments.
- **Out-of-Scope**: High-stakes physical robotics without hardware safety bounds, medical diagnosis, or claims of biological consciousness.
