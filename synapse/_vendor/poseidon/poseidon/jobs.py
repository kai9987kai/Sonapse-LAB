"""Bounded, durable cooperative experiments with one local worker.

Cancellation and deadlines are checked at progress checkpoints and again before
result publication. Python threads cannot interrupt an uncooperative function;
callers must invoke progress between bounded chunks and while waiting for locks.
Recovery never executes saved work or promotes an interrupted partial result.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import copy
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import time
import uuid


SCHEMA = "poseidon-job-v1"
MAX_PENDING = 4
MAX_SMALL_BYTES = 16 * 1024
MAX_RESULT_BYTES = 8 * 1024 * 1024
MAX_STATE_BYTES = MAX_RESULT_BYTES + 4 * MAX_SMALL_BYTES
TERMINAL = frozenset({"completed", "cancelled", "failed"})
_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_KIND = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")


class JobCancelled(RuntimeError):
    """A cooperative checkpoint observed cancellation."""


class JobTimeout(RuntimeError):
    """The single absolute operation deadline elapsed."""


def _utc(timestamp=None):
    return datetime.fromtimestamp(time.time() if timestamp is None else timestamp, timezone.utc).isoformat()


def _json_copy(value, limit, label, *, object_only=False):
    if object_only and not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise ValueError(f"{label} must contain finite JSON-compatible values") from error
    if len(encoded) > limit:
        raise ValueError(f"{label} exceeds its {limit}-byte budget")
    return json.loads(encoded)


class ExperimentJobs:
    """One manager per output directory; work receives a cooperative callback."""

    def __init__(self, output_dir, *, producer, max_history=32):
        if type(max_history) is not int or not 1 <= max_history <= 128:
            raise ValueError("max_history must be an integer from 1 to 128")
        self.producer = _json_copy(producer, MAX_SMALL_BYTES, "producer", object_only=True)
        self.generation = uuid.uuid4().hex
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self._condition = threading.Condition(threading.RLock())
        self._records = {}
        self._work = {}
        self._deadlines = {}
        self._queue = deque()
        self._closed = False
        self._recover()
        self._worker = threading.Thread(target=self._run, name="poseidon-experiment-jobs", daemon=True)
        self._worker.start()

    def _path(self, job_id):
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
            raise KeyError(job_id)
        return self.output_dir / (job_id + ".json")

    def _persist(self, record):
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_STATE_BYTES:
            raise ValueError("Job state exceeds its byte budget")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=self.output_dir, prefix=".job-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path(record["id"]))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _recover(self):
        for path in self.output_dir.glob("*.json"):
            if not _JOB_ID.fullmatch(path.stem):
                continue
            try:
                before = path.lstat()
                if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_size > MAX_STATE_BYTES:
                    continue
                with path.open("rb") as handle:
                    initial = os.fstat(handle.fileno())
                    raw = handle.read(MAX_STATE_BYTES + 1)
                    after = os.fstat(handle.fileno())
                current = path.lstat()
                # On Windows, path stat and CRT handle fstat can expose different
                # ctime meanings. Compare ctime within each API, not across them.
                signature = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                if (len(raw) > MAX_STATE_BYTES or any(signature(info) != signature(before) for info in (initial, after, current)) or
                        before.st_ctime_ns != current.st_ctime_ns or initial.st_ctime_ns != after.st_ctime_ns):
                    continue
                record = json.loads(raw)
                if (not isinstance(record, dict) or record.get("schema") != SCHEMA or
                        record.get("id") != path.stem or record.get("state") not in TERMINAL | {"queued", "running"} or
                        not isinstance(record.get("created_at"), str) or not _JOB_ID.fullmatch(str(record.get("generation", "")))):
                    continue
                _json_copy(record.get("settings"), MAX_SMALL_BYTES, "settings", object_only=True)
                _json_copy(record.get("producer"), MAX_SMALL_BYTES, "producer", object_only=True)
                _json_copy(record.get("progress"), MAX_SMALL_BYTES, "progress", object_only=True)
                _json_copy(record.get("result"), MAX_RESULT_BYTES, "result")
                if record["state"] in {"queued", "running"}:
                    record.update(state="failed", result=None, finished_at=_utc(), updated_at=_utc(),
                                  error={"type": "InterruptedJob", "message": "The previous process ended before this job completed. Partial work was not resumed."})
                    self._persist(record)
                elif record["state"] != "completed":
                    record["result"] = None
                self._records[record["id"]] = record
            except (OSError, ValueError, TypeError, KeyError, RecursionError):
                # Invalid files remain available for inspection and are never executed.
                continue
        self._prune(self.max_history)

    def _prune(self, target):
        while len(self._records) > target:
            terminal = [record for record in self._records.values() if record["state"] in TERMINAL]
            if not terminal:
                return
            oldest = min(terminal, key=lambda record: (record["created_at"], record["id"]))
            self._path(oldest["id"]).unlink(missing_ok=True)
            del self._records[oldest["id"]]

    def submit(self, kind, settings, work, timeout_seconds=300):
        if not isinstance(kind, str) or not _KIND.fullmatch(kind):
            raise ValueError("Job kind must be a bounded name without path separators")
        if not callable(work):
            raise ValueError("Job work must be callable")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or
                not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 1800):
            raise ValueError("Job timeout must be finite and from 1 to 1800 seconds")
        normalized = _json_copy(settings, MAX_SMALL_BYTES, "settings", object_only=True)
        with self._condition:
            if self._closed:
                raise RuntimeError("Experiment job manager is closed")
            if sum(record["state"] == "queued" for record in self._records.values()) >= MAX_PENDING:
                raise RuntimeError("Experiment job queue is full")
            self._prune(self.max_history - 1)
            if len(self._records) >= self.max_history:
                raise RuntimeError("Experiment job history is full of active work")
            job_id = uuid.uuid4().hex
            now = time.time()
            record = {"schema": SCHEMA, "id": job_id, "kind": kind, "state": "queued",
                      "settings": normalized, "producer": copy.deepcopy(self.producer), "generation": self.generation,
                      "created_at": _utc(now), "started_at": None, "updated_at": _utc(now), "finished_at": None,
                      "timeout_seconds": timeout_seconds, "deadline_at": _utc(now + timeout_seconds),
                      "cancel_requested": False, "progress": {}, "error": None, "result": None}
            self._persist(record)
            self._records[job_id] = record
            self._work[job_id] = work
            self._deadlines[job_id] = time.monotonic() + timeout_seconds
            self._queue.append(job_id)
            self._condition.notify_all()
            return copy.deepcopy(record)

    def list(self):
        with self._condition:
            return [copy.deepcopy({key: value for key, value in record.items() if key != "result"})
                    for record in sorted(self._records.values(), key=lambda record: (record["created_at"], record["id"]), reverse=True)]

    def get(self, job_id):
        self._path(job_id)
        with self._condition:
            return copy.deepcopy(self._records[job_id])

    def _interruption(self, job_id):
        record = self._records[job_id]
        if record["cancel_requested"] or self._closed:
            return JobCancelled("Cancellation was requested; partial work is not completed evidence")
        if time.monotonic() >= self._deadlines[job_id]:
            return JobTimeout("The absolute experiment deadline elapsed")
        return None

    def _finish(self, job_id, *, result=None, error=None):
        record = self._records[job_id]
        interruption = self._interruption(job_id)
        if interruption is not None:
            error = interruption
        state = "cancelled" if isinstance(error, JobCancelled) else "failed" if error is not None else "completed"
        record.update(state=state, result=result if state == "completed" else None,
                      error=None if error is None else {"type": type(error).__name__, "message": str(error)[:2000]},
                      updated_at=_utc(), finished_at=_utc())
        self._persist(record)
        self._work.pop(job_id, None)
        self._deadlines.pop(job_id, None)
        self._condition.notify_all()

    def cancel(self, job_id):
        self._path(job_id)
        with self._condition:
            record = self._records[job_id]
            if record["state"] in TERMINAL:
                return copy.deepcopy(record)
            record.update(cancel_requested=True, updated_at=_utc())
            if record["state"] == "queued":
                self._finish(job_id, error=JobCancelled("Queued job cancelled"))
            else:
                self._persist(record)
            self._condition.notify_all()
            return copy.deepcopy(record)

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._queue)
                if self._closed:
                    return
                job_id = self._queue.popleft()
                record = self._records.get(job_id)
                if record is None or record["state"] != "queued":
                    continue
                interruption = self._interruption(job_id)
                if interruption is not None:
                    self._finish(job_id, error=interruption)
                    continue
                record.update(state="running", started_at=_utc(), updated_at=_utc())
                self._persist(record)
                work = self._work[job_id]

            def progress(event=None):
                with self._condition:
                    interruption = self._interruption(job_id)
                    if interruption is not None:
                        raise interruption
                    if event is not None:
                        record.update(progress=_json_copy(event, MAX_SMALL_BYTES, "progress", object_only=True), updated_at=_utc())
                        self._persist(record)

            try:
                progress()
                result = work(progress)
                progress()
                bounded = _json_copy(result, MAX_RESULT_BYTES, "result")
                with self._condition:
                    self._finish(job_id, result=bounded)
            except Exception as error:
                with self._condition:
                    self._finish(job_id, error=error)

    def close(self):
        with self._condition:
            if not self._closed:
                self._closed = True
                for record in list(self._records.values()):
                    if record["state"] not in TERMINAL:
                        self.cancel(record["id"])
                self._condition.notify_all()
        if threading.current_thread() is not self._worker:
            self._worker.join(timeout=0.25)
