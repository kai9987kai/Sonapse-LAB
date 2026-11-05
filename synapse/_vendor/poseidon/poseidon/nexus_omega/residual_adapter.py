"""Fail-closed runtime wrapper for an admitted contextual residual receipt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContextualResidualAdapter:
    """Action-conditioned forecast correction bound to an audited assay receipt."""

    def __init__(self, center: np.ndarray, scale: np.ndarray, coefficients: np.ndarray, shrinkage: float):
        self.center = np.asarray(center, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.shrinkage = float(shrinkage)
        if self.center.shape != (8,) or self.scale.shape != (8,) or np.any(self.scale <= 0):
            raise ValueError("adapter requires finite 8-channel center and positive scale")
        if self.coefficients.shape != (36, 8) or not np.isfinite(self.coefficients).all():
            raise ValueError("adapter coefficients must be finite with shape (36, 8)")
        if not np.isfinite(self.center).all() or not np.isfinite(self.scale).all():
            raise ValueError("adapter normalization values must be finite")
        if not np.isfinite(self.shrinkage) or not 0.0 < self.shrinkage <= 1.0:
            raise ValueError("adapter shrinkage must be in (0, 1]")

    @classmethod
    def from_receipt(cls, path: str | Path) -> "ContextualResidualAdapter":
        receipt_path = Path(path).resolve()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != "poseidon-nexus-contextual-residual-assay-v1":
            raise ValueError("unsupported contextual residual receipt schema")
        claimed_digest = receipt.get("receipt_sha256")
        payload = dict(receipt)
        payload.pop("receipt_sha256", None)
        actual_digest = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        if not isinstance(claimed_digest, str) or claimed_digest != actual_digest:
            raise ValueError("contextual residual receipt digest mismatch")
        holdout = receipt.get("holdout", {})
        if holdout.get("admitted") is not True:
            raise ValueError("contextual residual candidate did not pass its untouched holdout gate")
        if holdout.get("runtime_activated") is not False:
            raise ValueError("receipt must describe an inactive candidate")

        root = Path(__file__).resolve().parents[2]
        expected_sources = {
            "assay": root / "tools/run_nexus_contextual_residual_assay.py",
            "causal_world_model": root / "poseidon/nexus_omega/causal_world_model.py",
            "benchmark": root / "poseidon/nexus_omega/benchmark.py",
        }
        recorded = receipt.get("source_sha256", {})
        if any(recorded.get(key) != _file_sha256(source) for key, source in expected_sources.items()):
            raise ValueError("source changed since the contextual residual assay")

        selected = receipt.get("selection", {}).get("selected_candidate", {})
        if selected.get("kind") != "contextual_ridge":
            raise ValueError("admitted receipt does not select a contextual ridge adapter")
        ridge_key = str(float(selected["ridge"]))
        coefficients = receipt["fit"]["contextual_coefficients_by_ridge"][ridge_key]
        return cls(
            center=np.asarray(receipt["fit"]["observation_center"], dtype=np.float64),
            scale=np.asarray(receipt["fit"]["observation_scale"], dtype=np.float64),
            coefficients=np.asarray(coefficients, dtype=np.float64),
            shrinkage=float(selected["shrinkage"]),
        )

    def apply(self, observation: np.ndarray, action: int, prediction: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        pred = np.asarray(prediction, dtype=np.float64)
        if obs.shape != (8,) or pred.shape != (8,):
            raise ValueError("contextual residual adapter requires 8-channel observation and prediction")
        if type(action) is not int or not 0 <= action < 4:
            raise ValueError("action must be an integer in [0, 3]")
        normalized = np.clip((obs - self.center) / self.scale, -5.0, 5.0)
        features = np.zeros(36, dtype=np.float64)
        start = action * 9
        features[start] = 1.0
        features[start + 1:start + 9] = normalized
        return (pred + self.shrinkage * (features @ self.coefficients)).astype(np.float32)


class CalibratedFactoredWorldModel:
    """Optional wrapper applying a validated adapter; default model stays untouched."""

    def __init__(self, model: Any, adapter: ContextualResidualAdapter):
        if getattr(model, "d_obs", None) != 8 or getattr(model, "n_actions", None) != 4 or getattr(model, "n_members", None) != 3:
            raise ValueError("adapter was evaluated only for the 8-observation, 4-action, 3-member model")
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_adapter", adapter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_model", "_adapter"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._model, name, value)

    def predict_factored(self, obs: np.ndarray, mem: np.ndarray, action: int):
        result = self._model.predict_factored(obs, mem, action)
        corrected = self._adapter.apply(np.asarray(obs, dtype=np.float32), action, result.predicted_obs)
        result.predicted_obs = corrected
        result.endogenous_pred = corrected[:4].copy()
        result.exogenous_pred = corrected[4:].copy()
        # Comparator feedback should score the forecast actually presented to
        # the planner, rather than the hidden uncorrected base prediction.
        self._model.last_prediction = corrected.copy()
        return result

