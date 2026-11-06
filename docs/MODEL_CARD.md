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
  - **Expanse-inspired Grafts**: 2 untrained expert slots recruited via novelty thresholding, and a zero-initialized native consolidation block. Their matrices are not optimized by preparation; copying recruited random matrices is not learned continual plasticity.
  - **Motor Decoder**: Linear projection from 64-d fused feature representation to 5 continuous motor actuators (`drive`, `turn`, `reach`, `height`, `grip`), gated by average connectome activity.
  - **Experience Model**: Three bootstrap RLS members predict 14 sensor-summary deltas and actual reward, conditioned on current graph/donor summaries. Heuristic disagreement, bounded covariance leverage and prequential prediction skill gate optional one-step local planning. Off by default.
  - **Outcome Memory**: At most 128 actual state/action/reward records with surprise/coverage admission and pure distance/recency reads. This is not a neural Titans/TTT implementation.
  - **Vision**: Small seeded visual fusion affects live actions; visual encoder/projection receive optional supervised preparation gradients and are checkpointed. Recognition accuracy has not been benchmarked.

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
| `dormant.*` | Yes | No | No (manual patch surface) |
| `native.*` | Yes | No gradients; heuristic copying | No |
| `vision_encoder.*`, `vis_to_sensory.*` | Yes | Yes, with supplied camera demonstrations | No |
| `tidal.*` (Tidal Core)| **No** | **No** | **Frozen** |
| `curiosity.*` | **No** | Yes (RND updates) | No |

## Evaluation and Intended Usage
- **Primary Use**: Embodied reinforcement learning, cue-delay navigation, memory consolidation, and recursive self-improvement research in synthetic continuous environments.
- **Out-of-Scope**: High-stakes physical robotics without hardware safety bounds, medical diagnosis, or claims of biological consciousness.

The recurrent graph uses synthetic proxy topologies. Tidal's upstream training
does not establish competence of the coupled untrained bridges/motor head. Ordinary
checkpoints preserve learned model/memory, live optimizer bindings, visual weights
and pending outcomes; arbitrary legacy topology mutations remain unsupported when
metadata and buffers diverge. Supervised preparation approximates parts of the
live recurrent policy. Current native development experiments do not establish a
planner advantage; see [measured results](results/2026-11-06-experience-adaptation.md).
