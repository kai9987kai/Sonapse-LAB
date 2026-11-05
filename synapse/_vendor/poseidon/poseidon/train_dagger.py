"""DAgger for the Tidal survival policy (Ross, Gordon & Bagnell 2011, arXiv:1011.0686).

Behaviour cloning only sees states the scripted teacher visits, so small policy
errors compound into unfamiliar states. DAgger rolls out the *learned* policy,
asks the visible-state teacher to label the states actually reached, aggregates
those labels with the original data and retrains. The teacher is used only to
label training data; it is never consulted at inference or evaluation time.

A candidate is promoted (runs/active_core.json) only if it beats the parent on a
separate validation seed namespace. Final claims use evaluate.py's test seeds.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from statistics import fmean

import torch

from .core import ACTIVE_CORE_POINTER, TidalCore, load_checkpoint, save_checkpoint
from .train_core import atomic_json, file_hash, loss_for_batch
from .world import TidePool, rollout, teacher_action

DAGGER_SCHEMA = "poseidon-dagger-v1"
COLLECT_NAMESPACE = 77      # (77 << 23) | (round << 16) | episode
VALIDATION_NAMESPACE = 78   # disjoint from training (42/1000042) and evaluate.py (91000001+)
SCARCITIES = (0.8, 1.0, 1.5, 2.0, 3.0, 4.0)


def greedy_policy(model: TidalCore):
    zeros = torch.zeros(1, model.config.hash_buckets)

    @torch.inference_mode()
    def act(observation) -> int:
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        return int(model(zeros, obs)["action"].argmax(-1)[0])
    return act


def collect_on_policy(policy, episodes: int, round_index: int, beta: float,
                      max_steps: int = 192, namespace: int = COLLECT_NAMESPACE) -> list[dict]:
    """Roll out a beta-mixture of teacher/learner; label every visited state with the teacher."""
    if not (0 <= round_index < 128 and 1 <= episodes < 65536 and 0.0 <= beta <= 1.0):
        raise ValueError("invalid DAgger collection arguments")
    rng = random.Random(f"dagger:{namespace}:{round_index}")
    records = []
    for episode in range(episodes):
        episode_seed = (namespace << 23) | (round_index << 16) | episode
        scarcity = SCARCITIES[episode % len(SCARCITIES)]
        env = TidePool(episode_seed, scarcity=scarcity, max_steps=max_steps)
        obs = env.observe()
        while not env.done:
            label = teacher_action(obs)
            action = label if rng.random() < beta else int(policy(obs))
            nxt, _, _, info = env.step(action)
            records.append({"obs": obs, "action": action, "teacher_action": label, "next_obs": nxt,
                            "episode_seed": episode_seed, "scarcity": scarcity, "death": info["death"]})
            obs = nxt
    return records


def survival_score(policy, seeds: list[int], max_steps: int = 256) -> dict:
    out = {}
    for scarcity in (1.0, 2.0, 4.0):
        rows = [rollout(policy, seed, max_steps, scarcity, include_trajectory=False) for seed in seeds]
        out[str(scarcity)] = {"survival_rate": fmean(r["survived"] for r in rows),
                              "mean_steps": fmean(r["steps"] for r in rows),
                              "mean_reward": fmean(r["reward"] for r in rows)}
    out["score"] = fmean(v["survival_rate"] for k, v in out.items() if k != "score")
    out["mean_steps"] = fmean(v["mean_steps"] for k, v in out.items() if k not in ("score", "mean_steps"))
    return out


def better(candidate: dict, incumbent: dict) -> bool:
    if candidate["score"] != incumbent["score"]:
        return candidate["score"] > incumbent["score"]
    return candidate["mean_steps"] > incumbent["mean_steps"]


def aggregate(train: dict, records: list[dict]) -> dict:
    if not records:
        return train
    obs = torch.tensor([r["obs"] for r in records], dtype=torch.float32)
    nxt = torch.tensor([r["next_obs"] for r in records], dtype=torch.float32)
    data = dict(train)
    data["obs"] = torch.cat((train["obs"], obs))
    data["delta"] = torch.cat((train["delta"], nxt - obs))
    data["actions"] = torch.cat((train["actions"], torch.tensor([r["action"] for r in records], dtype=torch.long)))
    data["teacher"] = torch.cat((train["teacher"], torch.tensor([r["teacher_action"] for r in records], dtype=torch.long)))
    return data


def fine_tune(model: TidalCore, data: dict, *, steps: int, batch_size: int, lr: float, generator: torch.Generator) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ns, nw = len(data["features"]), len(data["obs"])
    bs_scene, bs_world = batch_size // 4, batch_size - batch_size // 4  # scene replay prevents forgetting
    model.train()
    last = {}
    for _ in range(steps):
        si = torch.randint(ns, (bs_scene,), generator=generator)
        wi = torch.randint(nw, (bs_world,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss, last = loss_for_batch(model, data, si, wi)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite DAgger loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return last


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", type=Path, default=Path("runs/tidal/core.pt"))
    parser.add_argument("--data", type=Path, default=Path("runs/tidal/training_data.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/tidal_dagger"))
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-seeds", type=int, default=48)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args(argv)
    if not (1 <= args.rounds <= 32 and 1 <= args.episodes <= 4096 and 1 <= args.steps <= 100000
            and 8 <= args.batch_size <= 4096 and 4 <= args.validation_seeds <= 1000):
        parser.error("arguments out of range")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    root = Path(".").resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    model, payload = load_checkpoint(args.parent)
    model.eval()
    train = torch.load(args.data, weights_only=True, map_location="cpu")["train"]
    validation_seeds = [(VALIDATION_NAMESPACE << 23) | i for i in range(args.validation_seeds)]
    generator = torch.Generator().manual_seed(args.seed)

    parent_val = survival_score(greedy_policy(model), validation_seeds)
    teacher_val = survival_score(teacher_action, validation_seeds)
    print(json.dumps({"event": "parent_validation", **parent_val, "teacher": teacher_val}), flush=True)
    best_state, best_val, best_round = copy.deepcopy(model.state_dict()), parent_val, 0
    aggregated, history, collected = train, [], 0
    for round_index in range(1, args.rounds + 1):
        beta = 0.5 ** round_index  # standard decaying teacher mixture; beta_1 = 0.5
        records = collect_on_policy(greedy_policy(model), args.episodes, round_index, beta)
        disagreement = fmean(r["action"] != r["teacher_action"] for r in records)
        collected += len(records)
        aggregated = aggregate(aggregated, records)
        metrics = fine_tune(model, aggregated, steps=args.steps, batch_size=args.batch_size,
                            lr=args.learning_rate, generator=generator)
        val = survival_score(greedy_policy(model), validation_seeds)
        row = {"event": "round", "round": round_index, "beta": beta, "new_states": len(records),
               "aggregated_world_states": len(aggregated["obs"]), "executed_vs_teacher_disagreement": disagreement,
               "train_metrics": metrics, "validation": val}
        history.append(row)
        print(json.dumps(row), flush=True)
        if better(val, best_val):
            best_state, best_val, best_round = copy.deepcopy(model.state_dict()), val, round_index

    model.load_state_dict(best_state)
    accepted = best_round > 0
    parent_receipt = payload.get("receipt", {})
    receipt = {"schema": DAGGER_SCHEMA, "status": "completed", "method": "DAgger (Ross et al. 2011), beta_i = 0.5^i",
               "parent_checkpoint": str(args.parent.resolve()), "parent_sha256": file_hash(args.parent),
               "parent_receipt": parent_receipt, "contract": parent_receipt.get("contract"),
               "examples_seen": parent_receipt.get("examples_seen", 0) + args.rounds * args.steps * args.batch_size,
               "dagger_states_collected": collected, "rounds": args.rounds, "selected_round": best_round,
               "accepted": accepted, "parent_validation": parent_val, "teacher_validation": teacher_val,
               "selected_validation": best_val, "history": history,
               "validation_seeds": f"({VALIDATION_NAMESPACE} << 23) | [0, {args.validation_seeds})",
               "seconds": time.monotonic() - started,
               "limitations": ["teacher labels bound the target behaviour; DAgger reduces covariate shift, not teacher error",
                               "synthetic simulator only", "model selection used validation seeds, not evaluate.py test seeds"]}
    checkpoint = out / "core.pt"
    save_checkpoint(checkpoint, model, receipt=receipt)
    atomic_json(out / "report.json", {k: v for k, v in receipt.items() if k != "parent_receipt"})
    pointer = root / ACTIVE_CORE_POINTER
    if accepted and not args.no_promote:
        atomic_json(pointer, {"checkpoint": checkpoint.relative_to(root).as_posix(),
                              "sha256": file_hash(checkpoint), "reason": "DAgger candidate beat parent on validation seeds",
                              "parent_validation": parent_val["score"], "selected_validation": best_val["score"]})
    elif pointer.exists():
        try:
            stale = (root / json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]).resolve() == checkpoint
        except (ValueError, KeyError, OSError):
            stale = False
        if stale:
            pointer.unlink()  # this directory's earlier promotion no longer holds
    print(json.dumps({"event": "complete", "accepted": accepted, "selected_round": best_round,
                      "parent_score": parent_val["score"], "selected_score": best_val["score"],
                      "promoted": accepted and not args.no_promote}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
