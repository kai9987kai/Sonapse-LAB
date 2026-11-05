# Supermix Poseidon v0.1

The requested system supports conversation, mathematical problem solving, basic image and video generation, basic 3D assets, and a policy that can act in a survival simulation. The local target is an eight-core Snapdragon Windows PC with 16 GB RAM and no CUDA GPU. This is an experimental composite model, not a foundation model trained from scratch or a claim of general intelligence.

## Architecture

1. **Language:** pinned Apache-2.0 SmolLM2-135M-Instruct weights provide pretrained conversation. A local LoRA experiment uses a provenance-tracked public conversation subset. Activation requires an explicit validation result; an unvalidated adapter is not silently used.
2. **Tidal core:** a newly trained compact shared expert network encodes prompt tokens and a 16-dimensional observation. Its learned scene heads predict shape, color, motion, count and scale. Its policy predicts six survival actions; a separate action-conditioned head learns next-observation deltas. Router diagnostics expose how expert use changes by task. Recurrence is within each decision, not persistent hidden memory.
3. **Scene representation:** one bounded representation drives a software image renderer, animation and OBJ/glTF exports. The renderer executes predicted parameters. These are basic synthetic shapes, not diffusion-generated photographs or arbitrary semantic 3D objects.
4. **Memory:** explicitly recorded facts have carrier identity, provenance and reversible retrieval ablations. Retrieved content is data, not privileged instructions. Persistent external memory is distinct from weight updates.
5. **Reasoning:** the language model handles open prompts; a bounded arithmetic and linear-equation solver supplies independently checked results. Tool-assisted accuracy is reported separately from raw language-model accuracy.
6. **Environment:** a seeded toy survival world exposes only observations, supports snapshot replay, and has train/held-out episode seeds. Neural inference never calls the demonstrator as a hidden fallback.

## Training and evidence

The initial target is 120,000 generated scene/action examples. Public conversation rows are downloaded with a pinned revision, filtered and deduplicated before optional language adaptation. Prepared rows, rows consumed, tokens consumed, and inherited pretraining are different counts. Training uses atomic checkpoints, data identity checks, bounded CPU threads, and machine-readable receipts. Evaluation keeps semantic scene groups and environment episode seeds disjoint. Random, heuristic and ablated baselines reveal whether the learned component adds value.

Success means runnable local checkpoints, actual training receipts, held-out evaluation, usable exported media, responsive chat UI, and honest capability limits. The project repositories are design evidence, not a corpus to copy wholesale. License and source lineage are documented in the research notes.

## Implementation sequence

Research all 24 supplied repositories; implement independent core, scene and world modules in parallel; prepare language artifacts and runtime; generate and train the local curriculum; evaluate the trained checkpoints and failure cases; integrate and verify browser workflows; record measured capability and remaining limitations.

## Adjacent research

Gato (https://arxiv.org/abs/2205.06175) motivates sharing representations across tasks; Poseidon does not reproduce its token architecture or scale. DreamerV3 (https://arxiv.org/abs/2301.04104) motivates predictive world-model diagnostics; Poseidon's supervised dynamics head is not Dreamer or imagination-based reinforcement learning. SmolLM2 (https://arxiv.org/abs/2502.02737) supplies pretrained language ability. The combination is a portfolio-inspired engineering experiment, not a claim of first discovery.
