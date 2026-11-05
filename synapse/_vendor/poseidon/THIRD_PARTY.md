# Third-party materials and design lineage

## Materials actually downloaded for runtime/training

- [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, Apache-2.0 according to its model card. Local source card: `models/language/README.md`. Runtime uses safetensors and disables remote model code.
- [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, Apache-2.0 according to its dataset card. Local source card: `data/language/SOURCE_CARD.md`. Its synthetic/public upstream sources retain their attributions; see the card. The original shard and selected split hashes are recorded.
- PyTorch, Transformers, PEFT, safetensors, NumPy, Pillow, Hugging Face Hub, PyArrow, pytest and optional imageio/FFmpeg retain their own licenses. An FFmpeg binary's compiled codecs can impose additional distribution terms. This project does not relicense them.

Apache-2.0 license text: https://www.apache.org/licenses/LICENSE-2.0

## Repositories used as design evidence

All 24 links supplied by the user are covered in the three research documents. These are source reviews and independently implemented ideas, not wholesale imported source or training data. Public accessibility is not permission to relicense a repository. In particular, restricted inherited model weights and unlicensed repositories were not imported into Poseidon. Research documents identify gaps or uncertain status rather than silently inferring implementations from repository names.

## Research context

- [SmolLM2 paper](https://arxiv.org/abs/2502.02737): compact pretrained language backbone and data-centric training.
- [A Generalist Agent](https://arxiv.org/abs/2205.06175): adjacent shared-model work; Poseidon uses separate learned heads and a language component rather than implementing Gato.
- [DreamerV3](https://arxiv.org/abs/2301.04104): adjacent predictive world-model work; Poseidon does not implement its reinforcement learning algorithm.
- [Hugging Face PEFT](https://huggingface.co/docs/peft/package_reference/lora): LoRA adapter implementation.

Prior-project inspiration does not establish novelty priority, scientific validity, or inherited performance.
