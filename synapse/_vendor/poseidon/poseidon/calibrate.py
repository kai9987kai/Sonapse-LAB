"""Temperature scaling for the Tidal scene heads (Guo et al. 2017, arXiv:1706.04599).

One positive scalar per head is fitted on the dev split by minimising negative
log-likelihood; argmax predictions are unchanged. If a head is 100% correct on
dev, the NLL minimiser is degenerate (T -> 0), so that head is left at T = 1
and reported as such rather than silently over-sharpened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .core import SCENE_LABELS, TidalCore, active_core_path, load_checkpoint, save_checkpoint
from .train_core import atomic_json, file_hash

T_MIN, T_MAX = 0.05, 20.0


def expected_calibration_error(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    confidence, predicted = probabilities.max(-1)
    correct = (predicted == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            total += float(mask.float().mean() * (confidence[mask].mean() - correct[mask].mean()).abs())
    return total


@torch.inference_mode()
def scene_logits(model: TidalCore, features: torch.Tensor, batch_size: int = 512) -> dict[str, torch.Tensor]:
    model.eval()
    parts: dict[str, list] = {name: [] for name in SCENE_LABELS}
    for start in range(0, len(features), batch_size):
        result = model(features[start:start + batch_size])
        for name in SCENE_LABELS:
            parts[name].append(result["scene"][name])
    return {name: torch.cat(chunks) for name, chunks in parts.items()}


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, str]:
    if bool((logits.argmax(-1) == labels).all()):
        return 1.0, "degenerate: every dev prediction correct, NLL minimiser has no finite optimum"
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    value = float(log_t.detach().exp().clamp(T_MIN, T_MAX))
    return value, "fitted"


def calibrate(checkpoint: Path, data_path: Path, report_path: Path | None = None) -> dict:
    model, payload = load_checkpoint(checkpoint)
    dev = torch.load(data_path, weights_only=True, map_location="cpu")["dev"]
    logits = scene_logits(model, dev["features"])
    temperatures, report = {}, {}
    for j, name in enumerate(SCENE_LABELS):
        labels = dev["labels"][:, j]
        value, status = fit_temperature(logits[name], labels)
        temperatures[name] = value
        report[name] = {"temperature": value, "status": status,
                        "dev_accuracy": float((logits[name].argmax(-1) == labels).float().mean()),
                        "dev_nll_before": float(F.cross_entropy(logits[name], labels)),
                        "dev_nll_after": float(F.cross_entropy(logits[name] / value, labels)),
                        "dev_ece_before": expected_calibration_error(logits[name].softmax(-1), labels),
                        "dev_ece_after": expected_calibration_error((logits[name] / value).softmax(-1), labels)}
    calibration = {"method": "temperature-scaling (Guo et al. 2017)", "split": "dev", "scene_temperature": temperatures,
                   "data_sha256": file_hash(data_path), "heads": report}
    save_checkpoint(checkpoint, model, receipt=payload.get("receipt", {}), training=payload.get("training") or {},
                    calibration=calibration)
    if report_path:
        atomic_json(report_path, {"checkpoint": str(checkpoint.resolve()), **calibration})
    return calibration


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="defaults to the active core")
    parser.add_argument("--data", type=Path, default=Path("runs/tidal/training_data.pt"))
    parser.add_argument("--report", type=Path, default=Path("runs/calibration.json"))
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint or active_core_path(".")
    result = calibrate(checkpoint, args.data, args.report)
    print(json.dumps({"checkpoint": str(checkpoint), "temperatures": result["scene_temperature"],
                      "status": {k: v["status"] for k, v in result["heads"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
