"""Durable job behavior, including publication after cooperative interruption."""
import json
import threading
import time

import pytest


def manager(path, **kwargs):
    from poseidon.jobs import ExperimentJobs
    return ExperimentJobs(path, producer={"version": "test", "source_sha256": "source-a"}, **kwargs)


def terminal(jobs, job_id, timeout=4):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        record = jobs.get(job_id)
        if record["state"] in {"completed", "cancelled", "failed"}:
            return record
        time.sleep(0.01)
    pytest.fail("Job did not reach a terminal state")


def test_completed_result_survives_restart_and_snapshots_are_detached(tmp_path):
    jobs = manager(tmp_path)
    try:
        job = jobs.submit("helm-experiment", {"seed": 3}, lambda progress: (
            progress({"phase": "episode", "completed": 1, "total": 1}) or {"verified": True, "count": 1}
        ))
        record = terminal(jobs, job["id"])
        assert record["state"] == "completed"
        assert record["result"] == {"verified": True, "count": 1}
        assert record["progress"]["completed"] == 1
        record["result"]["count"] = 999
        assert jobs.get(job["id"])["result"]["count"] == 1
        assert "result" not in jobs.list()[0]
        saved = json.loads((tmp_path / (job["id"] + ".json")).read_text())
        assert saved["state"] == "completed"
        assert saved["producer"]["source_sha256"] == "source-a"
    finally:
        jobs.close()
    restarted = manager(tmp_path)
    try:
        recovered = restarted.get(job["id"])
        assert recovered["result"]["count"] == 1
        assert recovered["generation"] != restarted.generation
    finally:
        restarted.close()


def test_pending_cancel_never_executes_work(tmp_path):
    entered, release, invoked = threading.Event(), threading.Event(), threading.Event()
    jobs = manager(tmp_path)
    try:
        jobs.submit("blocking", {}, lambda progress: (entered.set(), release.wait(3), {})[-1])
        assert entered.wait(1)
        queued = jobs.submit("queued", {}, lambda progress: invoked.set() or {})
        cancelled = jobs.cancel(queued["id"])
        assert cancelled["state"] == "cancelled"
        assert cancelled["result"] is None
        release.set()
        time.sleep(0.05)
        assert not invoked.is_set()
    finally:
        release.set()
        jobs.close()


def test_running_cancel_cannot_be_suppressed_into_completed_evidence(tmp_path):
    entered, release = threading.Event(), threading.Event()
    jobs = manager(tmp_path)
    def work(progress):
        entered.set()
        release.wait(3)
        try:
            progress({"phase": "partial"})
        except RuntimeError:
            pass
        return {"verified": True, "partial": True}
    try:
        job = jobs.submit("interruptible", {}, work)
        assert entered.wait(1)
        assert jobs.cancel(job["id"])["cancel_requested"] is True
        release.set()
        record = terminal(jobs, job["id"])
        assert record["state"] == "cancelled"
        assert record["result"] is None
    finally:
        release.set()
        jobs.close()


def test_absolute_deadline_rejects_result_without_another_progress_callback(tmp_path):
    jobs = manager(tmp_path)
    try:
        job = jobs.submit("late", {}, lambda progress: (time.sleep(1.08), {"verified": True})[-1], timeout_seconds=1)
        record = terminal(jobs, job["id"])
        assert record["state"] == "failed"
        assert record["error"]["type"] == "JobTimeout"
        assert record["result"] is None
    finally:
        jobs.close()


def test_queue_time_uses_the_same_deadline_and_expired_work_is_not_started(tmp_path):
    entered, release, invoked = threading.Event(), threading.Event(), threading.Event()
    jobs = manager(tmp_path)
    try:
        jobs.submit("blocking", {}, lambda progress: (entered.set(), release.wait(3), {})[-1])
        assert entered.wait(1)
        queued = jobs.submit("expiring", {}, lambda progress: invoked.set() or {}, timeout_seconds=1)
        time.sleep(1.08)
        release.set()
        record = terminal(jobs, queued["id"])
        assert record["error"]["type"] == "JobTimeout"
        assert record["started_at"] is None
        assert not invoked.is_set()
    finally:
        release.set()
        jobs.close()


def test_single_worker_and_bounded_pending_queue(tmp_path):
    entered, release = threading.Event(), threading.Event()
    jobs = manager(tmp_path)
    order = []
    try:
        jobs.submit("blocking", {}, lambda progress: (entered.set(), release.wait(3), {})[-1])
        assert entered.wait(1)
        pending = [jobs.submit("queued", {"index": i}, lambda progress, i=i: order.append(i) or {}) for i in range(4)]
        with pytest.raises(RuntimeError, match="queue"):
            jobs.submit("overflow", {}, lambda progress: {})
        assert order == []
        release.set()
        for job in pending:
            assert terminal(jobs, job["id"])["state"] == "completed"
        assert order == [0, 1, 2, 3]
    finally:
        release.set()
        jobs.close()


def test_history_prunes_old_terminal_states_and_files(tmp_path):
    jobs = manager(tmp_path, max_history=2)
    try:
        ids = []
        for index in range(3):
            job = jobs.submit("short", {}, lambda progress: {})
            ids.append(job["id"])
            terminal(jobs, job["id"])
        assert len(jobs.list()) == 2
        with pytest.raises(KeyError):
            jobs.get(ids[0])
        assert not (tmp_path / (ids[0] + ".json")).exists()
        assert len(list(tmp_path.glob("*.json"))) == 2
    finally:
        jobs.close()


def test_interrupted_recovery_preserves_original_producer_and_never_resumes(tmp_path):
    jobs = manager(tmp_path)
    job = jobs.submit("short", {"seed": 7}, lambda progress: {})
    saved = terminal(jobs, job["id"])
    jobs.close()
    saved.update(state="running", result=None, finished_at=None)
    (tmp_path / (job["id"] + ".json")).write_text(json.dumps(saved))
    from poseidon.jobs import ExperimentJobs
    restarted = ExperimentJobs(tmp_path, producer={"version": "new", "source_sha256": "source-b"})
    try:
        recovered = restarted.get(job["id"])
        assert recovered["state"] == "failed"
        assert recovered["error"]["type"] == "InterruptedJob"
        assert recovered["producer"]["source_sha256"] == "source-a"
        assert recovered["generation"] == saved["generation"]
        assert recovered["result"] is None
    finally:
        restarted.close()


def test_invalid_json_progress_and_result_fail_without_completion(tmp_path):
    jobs = manager(tmp_path)
    try:
        for work in [lambda progress: progress({"bad": float("nan")}), lambda progress: {"bad": float("inf")}, lambda progress: "x" * (9 * 1024 * 1024)]:
            job = jobs.submit("invalid", {}, work)
            record = terminal(jobs, job["id"])
            assert record["state"] == "failed"
            assert record["result"] is None
    finally:
        jobs.close()


@pytest.mark.parametrize("timeout", [0, 1801, True, float("nan")])
def test_timeout_and_settings_are_bounded(tmp_path, timeout):
    jobs = manager(tmp_path)
    try:
        with pytest.raises(ValueError):
            jobs.submit("invalid", {}, lambda progress: {}, timeout_seconds=timeout)
        with pytest.raises(ValueError):
            jobs.submit("invalid", {"huge": "x" * 17000}, lambda progress: {})
        with pytest.raises(ValueError):
            jobs.submit("../escape", {}, lambda progress: {})
        assert jobs.list() == []
    finally:
        jobs.close()


def test_close_cancels_pending_and_rejects_new_work(tmp_path):
    entered, release = threading.Event(), threading.Event()
    jobs = manager(tmp_path)
    job = jobs.submit("blocking", {}, lambda progress: (entered.set(), release.wait(3), {})[-1])
    assert entered.wait(1)
    pending = jobs.submit("queued", {}, lambda progress: {"verified": True})
    jobs.close()
    release.set()
    assert terminal(jobs, job["id"])["state"] == "cancelled"
    assert jobs.get(pending["id"])["state"] == "cancelled"
    with pytest.raises(RuntimeError, match="closed"):
        jobs.submit("closed", {}, lambda progress: {})
