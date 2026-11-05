"""Reproducible CPU training for Poseidon's scene, control and dynamics heads."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .core import CoreConfig, SCENE_LABELS, TidalCore, config_hash, hash_features, load_checkpoint, save_checkpoint

DATA_SCHEMA = "poseidon-joint-data-v1"
TRAIN_SCHEMA = "poseidon-joint-training-v1"


def training_state(optimizer, sampler, *, epoch: int, batch: int, examples_seen: int, **extra) -> dict:
    return {"schema": TRAIN_SCHEMA, "optimizer": optimizer.state_dict(),
            "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(), "epoch": epoch, "batch": batch,
            "examples_seen": examples_seen, **extra}


def restore_training_state(state, optimizer, sampler) -> None:
    if state.get("schema") != TRAIN_SCHEMA:
        raise ValueError("unsupported training state schema")
    optimizer.load_state_dict(state["optimizer"])
    sampler.set_state(state["sampler_rng"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _label_indices(example: dict) -> list[int]:
    labels = example["labels"]
    if isinstance(labels, dict):
        return [int(labels[name]) for name in SCENE_LABELS]
    return [int(value) for value in labels]


def build_dataset(examples: int, seed: int, config: CoreConfig) -> tuple[dict, dict]:
    from .media import generate_scene_examples
    from .world import generate_transitions

    n_scene = examples // 2
    n_world = examples - n_scene
    dev_count = min(4096, max(128, examples // 20))
    scenes = {"train": generate_scene_examples(n_scene, seed=seed, split="train"),
              "dev": generate_scene_examples(dev_count, seed=seed + 1, split="dev")}
    worlds = {"train": generate_transitions(n_world, seed=seed),
              "dev": generate_transitions(dev_count, seed=seed + 1_000_000)}
    scene_groups = {split: {str(row["group"]) for row in rows} for split, rows in scenes.items()}
    episode_seeds = {split: {int(row["episode_seed"]) for row in rows} for split, rows in worlds.items()}
    if scene_groups["train"] & scene_groups["dev"]:
        raise ValueError("train/dev scene semantic groups overlap")
    if episode_seeds["train"] & episode_seeds["dev"]:
        raise ValueError("train/dev episode seeds overlap")
    data, manifests = {}, {}
    fingerprint = hashlib.sha256()
    for split in ("train", "dev"):
        sr, wr = scenes[split], worlds[split]
        prompts = [row["prompt"] for row in sr]
        if len(set(prompts)) != len(prompts):
            raise ValueError(f"duplicate {split} scene prompts")
        obs = torch.tensor([row["obs"] for row in wr], dtype=torch.float32)
        next_obs = torch.tensor([row["next_obs"] for row in wr], dtype=torch.float32)
        split_data = {"features": hash_features(prompts, config.hash_buckets),
                      "labels": torch.tensor([_label_indices(row) for row in sr], dtype=torch.long),
                      "obs": obs, "delta": next_obs - obs,
                      "actions": torch.tensor([row["action"] for row in wr], dtype=torch.long),
                      "teacher": torch.tensor([row.get("teacher_action", row["action"]) for row in wr], dtype=torch.long)}
        for key, tensor in split_data.items():
            fingerprint.update((split + key).encode())
            fingerprint.update(tensor.numpy().tobytes())
        data[split] = split_data
        manifests[split] = {"scene_rows": len(sr), "unique_scene_prompts": len(set(prompts)),
                            "scene_semantic_groups": len(scene_groups[split]),
                            "world_transitions": len(wr), "world_episodes": len(episode_seeds[split]),
                            "teacher_action_counts": torch.bincount(split_data["teacher"], minlength=6).tolist(),
                            "executed_action_counts": torch.bincount(split_data["actions"], minlength=6).tolist()}
    manifest = {"schema": DATA_SCHEMA, "seed": seed, "requested_training_examples": examples,
                "tensor_sha256": fingerprint.hexdigest(), "splits": manifests,
                "scene_split": "complete semantic tuples held out with stable hash, independent of wording seed",
                "world_split": "disjoint episode seed namespaces", "scene_group_overlap": 0,
                "world_episode_overlap": 0, "source_type": "procedural scenes and synthetic simulator teacher trajectories",
                "external_natural_media_examples": 0,
                "generators": {name: file_hash(Path(__file__).with_name(name)) for name in ("media.py", "world.py")}}
    return data, manifest


def _batch(data: dict, scene_indices: torch.Tensor, world_indices: torch.Tensor) -> tuple:
    features = data["features"][scene_indices]
    ns, nw = len(scene_indices), len(world_indices)
    joined_features = torch.cat((features, torch.zeros(nw, features.shape[1])), dim=0)
    observations = torch.cat((torch.zeros(ns, 16), data["obs"][world_indices]), dim=0)
    actions = torch.cat((torch.zeros(ns, dtype=torch.long), data["actions"][world_indices]))
    return joined_features, observations, actions


def loss_for_batch(model, data, scene_indices, world_indices) -> tuple[torch.Tensor, dict]:
    features, obs, actions = _batch(data, scene_indices, world_indices)
    ns = len(scene_indices)
    result = model(features, obs, actions)
    zero = result["action"].sum() * 0
    scene_loss = sum(F.cross_entropy(result["scene"][name][:ns], data["labels"][scene_indices, j])
                     for j, name in enumerate(SCENE_LABELS)) / len(SCENE_LABELS) if ns else zero
    action_loss = F.cross_entropy(result["action"][ns:], data["teacher"][world_indices]) if len(world_indices) else zero
    dynamics_loss = F.mse_loss(result["delta"][ns:], data["delta"][world_indices]) if len(world_indices) else zero
    load_penalty = ((result["routing"].mean(0) - 1 / model.config.experts) ** 2).mean()
    total = scene_loss + action_loss + 2 * dynamics_loss + 0.01 * load_penalty
    return total, {"scene_loss": float(scene_loss.detach()), "action_loss": float(action_loss.detach()),
                   "dynamics_mse": float(dynamics_loss.detach()), "loss": float(total.detach())}


@torch.inference_mode()
def evaluate(model: TidalCore, data: dict, batch_size: int = 256) -> dict:
    model.eval()
    scene_correct = torch.zeros(len(SCENE_LABELS), dtype=torch.long)
    scene_exact = 0
    action_correct, delta_squared, baseline_squared = 0, 0.0, 0.0
    route_sum = torch.zeros(model.config.experts)
    ns, nw = len(data["features"]), len(data["obs"])
    for start in range(0, ns, batch_size):
        result = model(data["features"][start:start + batch_size])
        predicted = torch.stack([result["scene"][name].argmax(-1) for name in SCENE_LABELS], -1)
        correct = predicted == data["labels"][start:start + batch_size]
        scene_correct += correct.sum(0)
        scene_exact += int(correct.all(-1).sum())
        route_sum += result["routing"].sum(0)
    for start in range(0, nw, batch_size):
        observations = data["obs"][start:start + batch_size]
        result = model(torch.zeros(len(observations), model.config.hash_buckets), observations, data["actions"][start:start + batch_size])
        action_correct += int((result["action"].argmax(-1) == data["teacher"][start:start + batch_size]).sum())
        targets = data["delta"][start:start + batch_size]
        delta_squared += float((result["delta"] - targets).square().sum())
        baseline_squared += float(targets.square().sum())
        route_sum += result["routing"].sum(0)
    counts = torch.bincount(data["teacher"], minlength=6)
    return {"scene_attribute_accuracy": {name: int(scene_correct[j]) / max(1, ns) for j, name in enumerate(SCENE_LABELS)},
            "scene_exact_accuracy": scene_exact / max(1, ns), "scene_examples": ns,
            "action_accuracy": action_correct / max(1, nw), "world_examples": nw,
            "majority_action_accuracy": int(counts.max()) / max(1, nw),
            "dynamics_mse": delta_squared / max(1, nw * 16),
            "zero_delta_mse": baseline_squared / max(1, nw * 16),
            "mean_expert_weights": (route_sum / max(1, ns + nw)).tolist(),
            "interpretation": "Controlled synthetic held-out groups; action agreement is not a survival result."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/tidal"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--examples", type=int, default=120000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", nargs="?", const="auto")
    args = parser.parse_args(argv)
    if args.examples < 32 or args.epochs < 1 or args.batch_size < 2 or args.threads < 1 or args.save_every < 1:
        parser.error("examples>=32, epochs>=1, batch-size>=2, threads>=1 and save-every>=1 required")
    if not (0 < args.learning_rate < 1):
        parser.error("learning-rate must be between zero and one")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    cfg = CoreConfig(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else CoreConfig()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "core.pt"
    if checkpoint.exists() and not args.resume:
        raise ValueError("output already contains core.pt; select another output or --resume")
    contract = {"examples": args.examples, "seed": args.seed, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "threads": args.threads, "config_hash": config_hash(cfg),
                "generators": {name: file_hash(Path(__file__).with_name(name)) for name in ("media.py", "world.py")}}
    data_path = out / "training_data.pt"
    manifest_path = out / "data_manifest.json"
    if args.resume and data_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != contract:
            raise ValueError("resume data/config contract changed")
        if file_hash(data_path) != manifest["cache_sha256"]:
            raise ValueError("training data cache hash mismatch")
        data = torch.load(data_path, weights_only=True, map_location="cpu")
    else:
        print(json.dumps({"event": "building_data", "examples": args.examples}), flush=True)
        data, manifest = build_dataset(args.examples, args.seed, cfg)
        temp = data_path.with_suffix(".tmp")
        torch.save(data, temp)
        os.replace(temp, data_path)
        manifest.update(contract=contract, cache_sha256=file_hash(data_path))
        atomic_json(manifest_path, manifest)
    model = TidalCore(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    sampler = torch.Generator().manual_seed(args.seed + 81)
    initial_epoch, initial_batch, seen, global_step = 0, 0, 0, 0
    scene_order = world_order = None
    receipt = {"schema": TRAIN_SCHEMA, "status": "training", "examples_seen": 0,
               "unique_training_examples": args.examples, "data_tensor_sha256": manifest["tensor_sha256"],
               "contract": contract, "parameters": sum(p.numel() for p in model.parameters()),
               "limitations": ["controlled synthetic scene vocabulary", "simulator-specific policy", "uncalibrated confidence", "not a conversational or pixel-generating neural model"]}
    if args.resume:
        resume_path = checkpoint if args.resume == "auto" else Path(args.resume)
        model, payload = load_checkpoint(resume_path)
        if payload.get("receipt", {}).get("contract") != contract:
            raise ValueError("resume checkpoint training contract changed")
        if payload["receipt"].get("data_tensor_sha256") != manifest["tensor_sha256"]:
            raise ValueError("resume checkpoint data hash mismatch")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        state = payload["training"]
        restore_training_state(state, optimizer, sampler)
        initial_epoch, initial_batch, seen = state["epoch"], state["batch"], state["examples_seen"]
        global_step = state.get("global_step", 0)
        scene_order, world_order = state.get("scene_order"), state.get("world_order")
        receipt = payload["receipt"]
        receipt["status"] = "training"
    else:
        receipt["untrained_dev"] = evaluate(model, data["dev"], args.batch_size)
    tr = data["train"]
    ns, nw = len(tr["features"]), len(tr["obs"])
    bs_scene, bs_world = args.batch_size // 2, args.batch_size - args.batch_size // 2
    batches = max(math.ceil(ns / bs_scene), math.ceil(nw / bs_world))
    started = time.monotonic()
    log_path = out / "training.jsonl"

    def publish(epoch, next_batch, so=None, wo=None):
        receipt.update(examples_seen=seen, global_step=global_step, completed_epochs=epoch,
                       elapsed_seconds_this_invocation=time.monotonic() - started)
        state = training_state(optimizer, sampler, epoch=epoch, batch=next_batch, examples_seen=seen,
                               scene_order=so, world_order=wo, global_step=global_step)
        save_checkpoint(checkpoint, model, receipt=receipt, training=state)
        atomic_json(out / "receipt.json", receipt)

    with log_path.open("a", encoding="utf-8") as log:
        for epoch in range(initial_epoch, args.epochs):
            if epoch != initial_epoch or initial_batch == 0 or scene_order is None:
                scene_order, world_order = torch.randperm(ns, generator=sampler), torch.randperm(nw, generator=sampler)
            first = initial_batch if epoch == initial_epoch else 0
            model.train()
            for batch in range(first, batches):
                si = scene_order[batch * bs_scene:(batch + 1) * bs_scene]
                wi = world_order[batch * bs_world:(batch + 1) * bs_world]
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = loss_for_batch(model, tr, si, wi)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss; previous checkpoint retained")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                seen += len(si) + len(wi)
                global_step += 1
                if global_step % 20 == 0 or batch + 1 == batches:
                    row = {"event": "train", "epoch": epoch + 1, "batch": batch + 1,
                           "batches": batches, "step": global_step, "examples_seen": seen,
                           "elapsed_seconds": time.monotonic() - started, **metrics}
                    log.write(json.dumps(row) + "\n"); log.flush()
                    print(json.dumps(row), flush=True)
                if global_step % args.save_every == 0:
                    publish(epoch, batch + 1, scene_order, world_order)
            evaluation = evaluate(model, data["dev"], args.batch_size)
            row = {"event": "eval", "epoch": epoch + 1, "examples_seen": seen, **evaluation}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps(row), flush=True)
            receipt["dev"] = evaluation
            receipt["status"] = "completed" if epoch + 1 == args.epochs else "training"
            publish(epoch + 1, 0)
            initial_batch = 0
    print(json.dumps({"event": "complete", "checkpoint": str(checkpoint), "examples_seen": seen}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
