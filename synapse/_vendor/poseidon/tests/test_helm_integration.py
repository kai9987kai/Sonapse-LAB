import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from poseidon.runtime import Poseidon
from poseidon.server import make_handler


@pytest.fixture
def job_server(tmp_path):
    runtime = Poseidon(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, 8787))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield runtime, f"http://127.0.0.1:{server.server_port}"
    runtime.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def post(url, payload):
    return urlopen(Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"}), timeout=3)


def test_job_request_and_unknown_identity_are_strict(job_server):
    runtime, url = job_server
    for settings in ({}, {"kind": "unknown"}, {"kind": "helm-experiment", "episodes": True},
                     {"kind": "helm-experiment", "max_steps": 512},
                     {"kind": "helm-experiment", "scarcity": "4"},
                     {"kind": "helm-experiment", "surprise": 1}):
        with pytest.raises(HTTPError) as error:
            post(url + "/api/jobs", settings)
        assert error.value.code == 400
    with pytest.raises(HTTPError) as error:
        urlopen(url + "/api/jobs/missing", timeout=3)
    assert error.value.code == 404
    with pytest.raises(HTTPError) as error:
        post(url + "/api/jobs/missing/cancel", {})
    assert error.value.code == 404


def test_status_and_cancellation_do_not_wait_for_inference_lock(job_server):
    runtime, url = job_server
    # Initialize optional backend imports before measuring responsiveness under
    # a held inference lock; cold PyTorch import is unrelated to lock contention.
    runtime.status()
    started = threading.Event()
    released = threading.Event()
    def work(progress):
        with runtime.lock:
            started.set()
            while not released.wait(.01):
                progress({"phase": "test-model-work"})
        return {"verified": True}
    job = runtime.jobs().submit("test", {}, work)
    try:
        assert started.wait(2)
        status = json.load(urlopen(url + "/api/status", timeout=2))
        assert status["helm"]["busy"] and status["helm"]["status_cached"]
        listing = json.load(urlopen(url + "/api/jobs", timeout=2))
        assert listing["jobs"][0]["id"] == job["id"]
        response = json.load(post(url + "/api/jobs/" + job["id"] + "/cancel", {}))
        assert response["cancel_requested"] or response["state"] == "cancelled"
        with pytest.raises(HTTPError) as error:
            post(url + "/api/jobs", {"kind": "helm-experiment"})
        # Busy work rejects promptly; a completed cancellation may instead reveal
        # the deliberately absent fit, which is also an explicit rejection.
        assert error.value.code == 409
    finally:
        released.set()


def test_world_helm_resets_actual_episode_settings_and_records_decisions(tmp_path, monkeypatch):
    runtime = Poseidon(tmp_path)
    class Critic:
        def __init__(self):
            self.resets = []
        def reset(self, **settings):
            self.resets.append(settings)
        def plan(self, observation):
            return {"action": 0, "policy_action": 0, "overridden": False, "step": observation[15]}
        def act(self, observation):
            raise AssertionError("Recorder must call plan only once")
    critic = Critic()
    monkeypatch.setattr(runtime, "core", lambda: object())
    monkeypatch.setattr(runtime, "helm", lambda: critic)
    result = runtime.respond("Survive", mode="world", planner="helm", scarcity=4., max_steps=16)
    assert critic.resets == [{"scarcity": 4., "max_steps": 16}]
    assert result["episode"]["controller"] == "helm"
    assert all(event["decision"]["action"] == event["action"] for event in result["episode"]["trajectory"])


def test_runtime_rejects_job_when_model_busy_without_waiting(tmp_path):
    runtime = Poseidon(tmp_path)
    locked = threading.Event()
    release = threading.Event()
    def hold_model():
        with runtime.lock:
            locked.set()
            release.wait(2)
    thread = threading.Thread(target=hold_model)
    thread.start()
    try:
        assert locked.wait(1)
        with pytest.raises(RuntimeError, match="processing another"):
            runtime.submit_helm_experiment({"kind": "helm-experiment"})
    finally:
        release.set()
        thread.join(timeout=2)
