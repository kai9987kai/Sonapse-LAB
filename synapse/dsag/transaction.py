"""Atomic candidate work for the DSAG-local model, queues, clock, and RNG."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable

from .bridge import DSAGBusChannel


class DSAGCandidateRejected(RuntimeError):
    """Raised when a candidate fails its explicit validation gate."""


@dataclass(frozen=True)
class DSAGTransactionReceipt:
    schema_version: int
    transaction_id: str
    parent_checkpoint_digest: str
    candidate_checkpoint_digest: str
    spec_digest: str
    topology_digest: str
    accepted: bool


class DSAGTrainingTransaction:
    """Snapshot all DSAG-owned weights/state and require an explicit commit.

    The host world is intentionally outside this object. To make a whole-run
    transaction, pair this with the host runtime's existing checkpoint and
    rollback boundary.
    """

    def __init__(self, channel: DSAGBusChannel, validator: Callable[[DSAGBusChannel], bool] | None = None):
        if not isinstance(channel, DSAGBusChannel):
            raise TypeError("channel must be a DSAGBusChannel")
        self.channel = channel
        self.validator = validator
        self._snapshot: dict | None = None
        self._committed = False
        self.receipt: DSAGTransactionReceipt | None = None

    def __enter__(self) -> "DSAGTrainingTransaction":
        if self._snapshot is not None:
            raise RuntimeError("DSAG transaction cannot be entered twice")
        self._snapshot = self.channel.export_state()
        return self

    def commit(self) -> DSAGTransactionReceipt:
        if self._snapshot is None or self._committed:
            raise RuntimeError("DSAG transaction is not open")
        candidate = self.channel.export_state()
        if candidate["spec_digest"] != self._snapshot["spec_digest"]:
            self.rollback()
            raise DSAGCandidateRejected("candidate changed its frozen DSAG spec")
        passed = True if self.validator is None else bool(self.validator(self.channel))
        if not passed:
            self.rollback()
            raise DSAGCandidateRejected("candidate failed its explicit validator")
        parent_digest = self._snapshot["state_digest"]
        candidate_digest = candidate["state_digest"]
        transaction_id = hashlib.sha256((parent_digest + candidate_digest).encode("ascii")).hexdigest()[:24]
        self.receipt = DSAGTransactionReceipt(
            schema_version=1,
            transaction_id=transaction_id,
            parent_checkpoint_digest=parent_digest,
            candidate_checkpoint_digest=candidate_digest,
            spec_digest=self.channel.spec.spec_digest,
            topology_digest=self.channel.model.topology_digest,
            accepted=True,
        )
        self._committed = True
        return self.receipt

    def rollback(self) -> None:
        if self._snapshot is None:
            raise RuntimeError("DSAG transaction has not been entered")
        self.channel.import_state(self._snapshot)
        self._committed = False

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None or not self._committed:
            self.rollback()
        return False
