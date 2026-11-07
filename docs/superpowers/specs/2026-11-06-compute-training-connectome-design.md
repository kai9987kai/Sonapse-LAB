# Additive compute, training, and connectome integration

The user authorizes an additive upgrade after catching up with Antigravity, then
publication to the existing GitHub and Hugging Face repositories. Antigravity is
finished editing. Baseline is clean `ddf7406`, including `c3e66e8` and `cffa430`.
Preserve the 2000-neuron frontier configuration, environments, older feature APIs,
checkpoints where unambiguous, bounded planning, and transactional weight edits.

## Findings and choices

The current baseline passes 376 tests. It nevertheless pins PyTorch to one thread,
does not train an added motor policy with ordinary live learning, mutates dormant
state in frozen inference, and has incomplete optional-module checkpointing.
Omnibus sleep screens do not inspect candidates and several metrics are surrogate.
Live graph sources are proxy profiles even when independent data tools exist.

An eight-core Snapdragon X Plus, 16 GB RAM and Adreno GPU are available. Installed
PyTorch is CPU-only. Use configurable CPU threads and batched training rather than
claiming unavailable GPU training. Keep at least 2 GB free disk. Original full
MaleCNS Feather assets exist outside this checkout and must remain read-only.
The bundled type matrix and MaleCNS homolog annotations are not whole FlyWire/MANC
neuron graphs. Acquire public full sources where feasible, verify provenance and
report missing sources honestly.

## Architecture

1. Add a zero-initialized bounded residual neural graft. Inputs are the existing
   observation, 64 graph/donor features, baseline action, and a fixed connectome
   sidecar. Separate actor, observation-value and action-conditioned dynamics
   objectives preserve actor identity during auxiliary pretraining. Its optimizer,
   bounded replay, policy receipts and RNG belong to its own checkpoint state.
   Active policy gradients use real executed rewards and eligible exploratory
   samples. Manual/intervened transitions train prediction/critic but not an
   incorrectly attributed policy loss. Final existing action bounds still apply.
2. User-triggered pretraining collects real cloned-world transitions and fits
   batches; live learning can schedule small bounded batches. Surface actual
   samples, optimizer steps, losses, CPU/wall time and configured threads. Preserve
   existing learning systems; no invented performance scores or forced CPU load.
3. Add verified full-source inventory and portable import/downloader metadata.
   Source-backed graph subsets and a fixed weighted connectome graft identify
   original neuron IDs, scope, signs and hashes. Keep raw datasets local. Missing
   datasets never silently become authentic proxies. Strict mode fails closed;
   legacy exploratory proxies remain explicitly available.
4. Repair Antigravity evaluation/checkpoint defects, matched Omnibus baselines,
   skill inspection, fast-memory autograd leakage and full receipt integrity.
   Preserve feature availability while distinguishing trained, heuristic and
   unmeasured pathways.
5. Improve the 3D viewer with clear source provenance, activity-based sizing,
   source/role filters, edge budgets and graft visibility; preserve teleoperation,
   mode switching and environment mechanics. Add training/compute controls and
   safe live weight-programming/RSI inspection against current state.

## Research grounding

- Residual RL, https://arxiv.org/abs/1812.03201: additive correction around an
  existing controller. This implementation is not a reproduction or a guarantee.
- Net2Net, https://arxiv.org/abs/1511.05641: preserve existing function when adding
  capacity. A zero output head preserves initial deterministic actions here.
- RLPD, https://arxiv.org/abs/2302.02948: mixing recorded and online experience;
  this smaller CPU learner does not reproduce RLPD's ensemble SAC algorithm.
- AWR, https://arxiv.org/abs/1910.00177: distinguish value fitting and policy
  learning on real actions; copied self-actions alone do not demonstrate progress.

## Validation and publication

Behavioral regressions first: identity initialization; actual gradient updates;
manual attribution; frozen inference; checkpoint continuation; pretraining actor
freeze; full source versus type/proxy distinctions; sparse prior causality;
candidate-dependent evaluation; receipt tampering; UI controls. Run full tests,
browser checks, real training and bounded paired evaluations. Retain negative
results. Commit and publish the integrated tree only after fresh verification.
GitHub and Hugging Face have separate histories; preserve both without force-push.
