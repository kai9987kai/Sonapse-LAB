"""Detect ordinary workspace edits after the package entered this process.

A disk hash taken just before an experiment cannot identify already imported
Python code. Capture the package at first import instead, and require an unchanged
source tree before producing an experiment receipt. This is an integrity guard,
not authenticated code attestation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


class StaleSourceError(RuntimeError):
    pass


def capture_sources(directory: str | Path | None = None) -> dict[str, str]:
    root = Path(directory) if directory is not None else Path(__file__).parent
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))}


def source_status() -> dict:
    import poseidon
    baseline = getattr(poseidon, "_SOURCE_SNAPSHOT", None)
    current = capture_sources()
    changed = sorted(set(current) | set(baseline or {})) if baseline is None else sorted(
        name for name in set(current) | set(baseline) if current.get(name) != baseline.get(name)
    )
    return {"source_current": baseline is not None and not changed,
            "restart_required": baseline is None or bool(changed), "changed_source_files": changed}


def assert_source_current() -> dict[str, str]:
    state = source_status()
    if not state["source_current"]:
        raise StaleSourceError("Source changed since this process imported Poseidon. Restart the server or command before running experiments.")
    import poseidon
    return dict(poseidon._SOURCE_SNAPSHOT)
