"""Schema helpers for truthful DSAG observability."""
from __future__ import annotations

from typing import Any


def namespace_metrics(graft_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Attach a stable graft namespace without implying biological units."""
    if not graft_id:
        raise ValueError("graft_id is required")
    return {f"dsag/{graft_id}": {"schema_version": 1, **metrics}}
