"""Canonical state hashing and the DSAG-only checkpoint envelope."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from typing import Any

import torch


class DSAGCheckpointError(ValueError):
    """Raised when a DSAG checkpoint is incomplete, stale, or corrupt."""


def _feed(hasher: Any, value: Any) -> None:
    if value is None:
        hasher.update(b"N")
    elif isinstance(value, bool):
        hasher.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        payload = str(value).encode("ascii")
        hasher.update(b"I" + struct.pack(">I", len(payload)) + payload)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise DSAGCheckpointError("checkpoint contains a non-finite float")
        hasher.update(b"F" + struct.pack(">d", value))
    elif isinstance(value, str):
        payload = value.encode("utf-8")
        hasher.update(b"S" + struct.pack(">I", len(payload)) + payload)
    elif isinstance(value, bytes):
        hasher.update(b"Y" + struct.pack(">Q", len(value)) + value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        hasher.update(b"T" + struct.pack(">I", len(header)) + header)
        hasher.update(tensor.numpy().tobytes())
    elif isinstance(value, Mapping):
        hasher.update(b"D" + struct.pack(">I", len(value)))
        for key in sorted(value, key=lambda item: str(item)):
            _feed(hasher, str(key))
            _feed(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update((b"L" if isinstance(value, list) else b"U") + struct.pack(">I", len(value)))
        for item in value:
            _feed(hasher, item)
    else:
        raise DSAGCheckpointError(f"unsupported checkpoint value: {type(value).__name__}")


def canonical_state_digest(value: Any) -> str:
    hasher = hashlib.sha256()
    _feed(hasher, value)
    return hasher.hexdigest()


def seal_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    if "state_digest" in payload:
        raise DSAGCheckpointError("checkpoint payload already contains state_digest")
    return {**payload, "state_digest": canonical_state_digest(payload)}


def verify_checkpoint(checkpoint: dict[str, Any]) -> None:
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_digest"), str):
        raise DSAGCheckpointError("missing DSAG checkpoint digest")
    payload = {key: value for key, value in checkpoint.items() if key != "state_digest"}
    actual = canonical_state_digest(payload)
    if actual != checkpoint["state_digest"]:
        raise DSAGCheckpointError("DSAG checkpoint digest mismatch")
