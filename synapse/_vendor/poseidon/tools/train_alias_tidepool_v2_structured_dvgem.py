"""Train the inspectable structured-cue DV-GEM candidate on local CPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseidon.core import active_core_path
from poseidon.research.alias_world_v2 import ObservationAliasTidePoolV2
from poseidon.research.contracts import ALIAS_WORLD_V2_VERSION, canonical_bytes
from poseidon.research.decision_value_memory import (
    STRUCTURED_CUE_BELIEF_FEATURES,
    DecisionValueMemoryConfig,
    fit_decision_value_memory,
)


TRAIN_SEED_START = 743020000
TRAIN_SEED_COUNT = 12
UPDATES = 300
MAX_STEPS = 24
TORCH_SEED = 743020001


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists; preserving prior evidence: {output}")
    train_seeds = list(range(TRAIN_SEED_START, TRAIN_SEED_START + TRAIN_SEED_COUNT))
    regime_by_seed = {seed: index % 2 for index, seed in enumerate(train_seeds)}
    common_config = dict(
        capacity=128,
        top_k=4,
        admission_threshold=0.04,
        shrinkage_alpha=0.50,
        hidden_dim=48,
        horizons=(4, 8),
    )
    structured_cfg = DecisionValueMemoryConfig(
        **common_config,
        belief_dim=len(STRUCTURED_CUE_BELIEF_FEATURES),
        belief_encoder_type="structured-cue-v1",
    )
    gru_cfg = DecisionValueMemoryConfig(**common_config, belief_dim=24, belief_encoder_type="gru-v1")
    core_path = active_core_path().resolve()
    output.mkdir(parents=True)
    source_paths = sorted((ROOT / "poseidon").rglob("*.py")) + [Path(__file__).resolve()]
    source_hashes = {p.relative_to(ROOT).as_posix(): _sha256(p) for p in source_paths}
    protocol = {
        "schema": "poseidon-alias-v2-structured-dvgem-training-protocol-v1",
        "world_version": ALIAS_WORLD_V2_VERSION,
        "train_seed_range": [train_seeds[0], train_seeds[-1]],
        "training_regime_by_seed": {str(k): v for k, v in regime_by_seed.items()},
        "max_steps": MAX_STEPS,
        "updates": UPDATES,
        "memory_config": vars(structured_cfg),
        "matched_gru_control_config": vars(gru_cfg),
        "belief_features": list(STRUCTURED_CUE_BELIEF_FEATURES),
        "belief_encoder_parameters": 0,
        "scorer_parameters_trainable": True,
        "torch_seed": TORCH_SEED,
        "torch_threads": 2,
        "core_checkpoint_path": str(core_path),
        "core_checkpoint_sha256": _sha256(core_path),
        "source_sha256": source_hashes,
        "evaluation_and_promotion": False,
    }
    protocol["protocol_sha256"] = hashlib.sha256(canonical_bytes(protocol)).hexdigest()
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with zipfile.ZipFile(output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, path.relative_to(ROOT).as_posix())

    def ensure_source() -> None:
        for relative, digest in source_hashes.items():
            if _sha256(ROOT / relative) != digest:
                raise RuntimeError(f"source changed during training; evidence retained: {relative}")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    started = time.monotonic()
    try:
        ensure_source()

        def environment_factory(seed: int, max_steps: int, cue_horizon: int):
            return ObservationAliasTidePoolV2(
                seed=seed,
                max_steps=max_steps,
                cue_horizon=cue_horizon,
                forced_regime=regime_by_seed[seed],
            )

        torch.manual_seed(TORCH_SEED)
        structured_memory = fit_decision_value_memory(
            train_seeds=train_seeds,
            config=structured_cfg,
            updates=UPDATES,
            environment_factory=environment_factory,
        )
        torch.manual_seed(TORCH_SEED)
        gru_memory = fit_decision_value_memory(
            train_seeds=train_seeds,
            config=gru_cfg,
            updates=UPDATES,
            environment_factory=environment_factory,
        )
        ensure_source()
        structured_path = output / "structured_memory_model.pt"
        gru_path = output / "matched_gru_control_model.pt"
        torch.save(structured_memory.state_dict(), structured_path)
        torch.save(gru_memory.state_dict(), gru_path)
        model_hashes = {
            "structured_memory_model.pt": _sha256(structured_path),
            "matched_gru_control_model.pt": _sha256(gru_path),
        }
        receipt = {
            "schema": "poseidon-alias-v2-structured-dvgem-training-receipt-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "complete": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "protocol_sha256": protocol["protocol_sha256"],
            "model_sha256": model_hashes,
            "source_sha256": source_hashes,
            "core_checkpoint_sha256": protocol["core_checkpoint_sha256"],
            "learned": {
                "structured_belief_encoder": False,
                "structured_scorer": True,
                "matched_gru_belief_encoder": False,
                "matched_gru_scorer": True,
            },
            "evaluation_performed": False,
            "promotion": False,
            "limits": [
                "Training uses a deterministic local synthetic world and a fixed, hand-specified belief feature map.",
                "The scorer is trained from short counterfactual branches; no independent policy evaluation is included in this receipt.",
            ],
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        (output / "training_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        return {"training_receipt": str(output / "training_receipt.json"), "model_sha256": model_hashes}
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/research/decision_value_memory_v2_structured_pilot_20261014",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
