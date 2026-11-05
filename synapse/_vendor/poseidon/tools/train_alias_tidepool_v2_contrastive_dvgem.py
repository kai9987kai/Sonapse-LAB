"""Train matched structured-belief and GRU controllers on paired advantages."""
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


TRAIN_SEED_START = 743030000
TRAIN_SEED_COUNT = 12
UPDATES = 300
MAX_STEPS = 24
TORCH_SEED = 743030001
SCORER_TARGET_TYPE = "counterfactual-advantage-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists; preserving prior evidence: {output}")
    seeds = list(range(TRAIN_SEED_START, TRAIN_SEED_START + TRAIN_SEED_COUNT))
    regime_by_seed = {seed: index % 2 for index, seed in enumerate(seeds)}
    common = {
        "capacity": 128,
        "top_k": 4,
        "admission_threshold": 0.04,
        "shrinkage_alpha": 0.50,
        "hidden_dim": 48,
        "scorer_target_type": SCORER_TARGET_TYPE,
        "horizons": (4, 8),
    }
    structured_cfg = DecisionValueMemoryConfig(
        **common,
        belief_dim=len(STRUCTURED_CUE_BELIEF_FEATURES),
        belief_encoder_type="structured-cue-v1",
    )
    gru_cfg = DecisionValueMemoryConfig(**common, belief_dim=24, belief_encoder_type="gru-v1")
    absolute_cfg = DecisionValueMemoryConfig(
        **{**common, "scorer_target_type": "absolute-q-v1"},
        belief_dim=len(STRUCTURED_CUE_BELIEF_FEATURES),
        belief_encoder_type="structured-cue-v1",
    )
    core_path = active_core_path().resolve()
    output.mkdir(parents=True)
    source_paths = sorted((ROOT / "poseidon").rglob("*.py")) + [Path(__file__).resolve()]
    source_hashes = {p.relative_to(ROOT).as_posix(): _sha256(p) for p in source_paths}
    protocol = {
        "schema": "poseidon-alias-v2-counterfactual-advantage-training-protocol-v1",
        "world_version": ALIAS_WORLD_V2_VERSION,
        "train_seed_range": [seeds[0], seeds[-1]],
        "training_regime_by_seed": {str(k): v for k, v in regime_by_seed.items()},
        "max_steps": MAX_STEPS,
        "updates": UPDATES,
        "scorer_target_type": SCORER_TARGET_TYPE,
        "target_definition": "four-step branch return for candidate action minus four-step branch return for frozen-core action from the same restored snapshot",
        "memory_dropout": False,
        "structured_memory_config": vars(structured_cfg),
        "matched_gru_config": vars(gru_cfg),
        "matched_absolute_objective_config": vars(absolute_cfg),
        "belief_features": list(STRUCTURED_CUE_BELIEF_FEATURES),
        "torch_seed_reset_per_architecture": TORCH_SEED,
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
        structured = fit_decision_value_memory(seeds, structured_cfg, UPDATES, environment_factory=environment_factory)
        torch.manual_seed(TORCH_SEED)
        gru = fit_decision_value_memory(seeds, gru_cfg, UPDATES, environment_factory=environment_factory)
        torch.manual_seed(TORCH_SEED)
        absolute = fit_decision_value_memory(seeds, absolute_cfg, UPDATES, environment_factory=environment_factory)
        ensure_source()
        model_paths = {
            "structured_contrastive_model.pt": output / "structured_contrastive_model.pt",
            "matched_gru_contrastive_control_model.pt": output / "matched_gru_contrastive_control_model.pt",
            "matched_structured_absolute_control_model.pt": output / "matched_structured_absolute_control_model.pt",
        }
        torch.save(structured.state_dict(), model_paths["structured_contrastive_model.pt"])
        torch.save(gru.state_dict(), model_paths["matched_gru_contrastive_control_model.pt"])
        torch.save(absolute.state_dict(), model_paths["matched_structured_absolute_control_model.pt"])
        model_hashes = {name: _sha256(path) for name, path in model_paths.items()}
        receipt = {
            "schema": "poseidon-alias-v2-counterfactual-advantage-training-receipt-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "complete": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "protocol_sha256": protocol["protocol_sha256"],
            "model_sha256": model_hashes,
            "source_sha256": source_hashes,
            "core_checkpoint_sha256": protocol["core_checkpoint_sha256"],
            "scorer_target_type": SCORER_TARGET_TYPE,
            "evaluation_performed": False,
            "promotion": False,
            "limits": [
                "Training targets are synthetic four-step paired branch returns against the frozen core action.",
                "The belief map remains hand-specified and only the decision-value scorers are trained.",
                "The GRU and absolute-target checkpoints are matched training controls, not external baselines.",
                "No policy evaluation is included in this training receipt.",
            ],
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        receipt_path = output / "training_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return {"training_receipt": str(receipt_path), "model_sha256": model_hashes, "elapsed_seconds": receipt["elapsed_seconds"]}
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/research/decision_value_memory_v2_contrastive_pilot_20261014",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
