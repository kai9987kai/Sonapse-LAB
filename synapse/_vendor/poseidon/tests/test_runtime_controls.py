"""World settings, retained decisions and nonblocking runtime status."""
from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
import sys
import threading
import time
import types
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from poseidon.runtime import Poseidon
from poseidon.server import make_handler
from poseidon.world import ACTIONS, rollout, teacher_action
from poseidon.world_controls import rollout_with_decisions


class RecordingPlanner:
    def __init__(self):
        self.plan_calls = 0
        self.act_calls = 0
        self.shared = {}

    def plan(self, observation):
        self.plan_calls += 1
        action = teacher_action(observation)
        self.shared.update(action=action, action_name=ACTIONS[action], call=self.plan_calls)
        return self.shared

    def act(self, observation):
        self.act_calls += 1
        return teacher_action(observation)


def test_world_retains_one_independent_plan_per_transition():
    planner = RecordingPlanner()
    episode = rollout_with_decisions(planner, seed=41, max_steps=7)
    assert planner.plan_calls == episode["steps"] == 7
    assert planner.act_calls == 0
    for index, event in enumerate(episode["trajectory"], 1):
        assert event["decision"]["call"] == index
        assert event["decision"]["action"] == event["action"]
    planner.shared["call"] = -1
    assert episode["trajectory"][0]["decision"]["call"] == 1


def test_world_default_uses_act_and_preserves_previous_trace_shape():
    planner = RecordingPlanner()
    episode = rollout(planner, seed=41, max_steps=7)
    assert planner.act_calls == episode["steps"] == 7
    assert planner.plan_calls == 0
    assert all("decision" not in event for event in episode["trajectory"])


def test_runtime_world_honors_settings_and_records_atlas_decisions(tmp_path, monkeypatch):
    runtime = Poseidon(tmp_path)
    planner = RecordingPlanner()
    monkeypatch.setattr(runtime, "core", lambda: planner)
    monkeypatch.setattr(runtime, "atlas", lambda: planner)
    result = runtime.respond("Survive", "world", seed=71, planner="atlas", scarcity=3.5, max_steps=9)
    episode = result["episode"]
    assert episode["scarcity"] == 3.5
    assert episode["max_steps"] == episode["steps"] == 9
    assert len(episode["trajectory"]) == planner.plan_calls == 9
    assert planner.act_calls == 0
    assert "9-step horizon" in result["text"]


@pytest.mark.parametrize("settings", [
    {"max_steps": 0}, {"max_steps": True}, {"max_steps": 10001},
    {"scarcity": .4}, {"scarcity": True}, {"scarcity": float("nan")},
])
def test_invalid_world_settings_fail_before_loading_core(tmp_path, monkeypatch, settings):
    runtime = Poseidon(tmp_path)
    monkeypatch.setattr(runtime, "core", lambda: pytest.fail("invalid settings loaded a model"))
    with pytest.raises(ValueError):
        runtime.respond("Survive", "world", **settings)


def test_world_cli_passes_explicit_and_default_settings(tmp_path, monkeypatch, capsys):
    from poseidon import cli
    calls = []
    selected_oracles = []

    class Runtime:
        def __init__(self, *args, oracle_checkpoint=None):
            selected_oracles.append(oracle_checkpoint)

        def respond(self, *args, **kwargs):
            calls.append(kwargs)
            return {"ok": True}

    monkeypatch.setattr(cli, "Poseidon", Runtime)
    monkeypatch.setattr(sys, "argv", ["poseidon", "world", "--scarcity", "3.5", "--max-steps", "777"])
    cli.main()
    monkeypatch.setattr(sys, "argv", ["poseidon", "world"])
    cli.main()
    monkeypatch.setattr(sys, "argv", ["poseidon", "world", "--planner", "cognitive",
                                     "--oracle-checkpoint", "candidate/oracle.pt"])
    cli.main()
    capsys.readouterr()
    assert calls[0]["scarcity"] == 3.5 and calls[0]["max_steps"] == 777
    assert calls[1]["scarcity"] == 1.0 and calls[1]["max_steps"] == 256
    assert selected_oracles == [None, None, "candidate/oracle.pt"]
    assert calls[2]["planner"] == "cognitive"


def test_readiness_status_does_not_queue_behind_world_work(tmp_path, monkeypatch):
    runtime = Poseidon(tmp_path)
    path = tmp_path / "outputs/atlas/atlas.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")

    class Atlas:
        artifact = {"sha256": "a" * 64, "records": [1, 2], "partition": {}, "calibration": {}}

    monkeypatch.setattr(runtime, "atlas", lambda: Atlas())
    ready = runtime.atlas_status()
    assert ready["ready"] is True
    runtime.status()
    held, release = threading.Event(), threading.Event()

    def hold_runtime():
        with runtime.lock:
            held.set()
            release.wait(3)

    worker = threading.Thread(target=hold_runtime)
    worker.start()
    assert held.wait(1)
    try:
        started = time.monotonic()
        result = runtime.status()
        elapsed = time.monotonic() - started
        assert elapsed < .5
        assert result["atlas"]["ready"] is True
        assert result["atlas"]["status_cached"] is True
        assert result["atlas"]["busy"] is True
        assert result["contrast"]["ready"] is False
    finally:
        release.set()
        worker.join(2)


def test_missing_contrast_is_explicit_and_world_contrast_is_distinct(tmp_path, monkeypatch):
    runtime = Poseidon(tmp_path)
    assert runtime.contrast_status()["ready"] is False
    planner = RecordingPlanner()
    monkeypatch.setattr(runtime, "core", lambda: planner)
    monkeypatch.setattr(runtime, "contrast", lambda: planner)
    result = runtime.respond("Survive", "world", planner="contrast", max_steps=4)
    assert result["episode"]["controller"] == "contrast"
    assert planner.plan_calls == 4 and planner.act_calls == 0


def test_contrast_runtime_exports_verified_receipt_in_distinct_directory(tmp_path, monkeypatch):
    runtime = Poseidon(tmp_path)
    core, atlas, contrast = object(), object(), object()
    monkeypatch.setattr(runtime, "core", lambda: core)
    monkeypatch.setattr(runtime, "atlas", lambda: atlas)
    monkeypatch.setattr(runtime, "contrast", lambda: contrast)
    replay = {"controller": "contrast", "trajectory": [{"action": 1}]}
    result = {"schema": "contrast-test", "experiment_id": "b" * 64,
              "summary": {}, "paired": {}, "rows": [], "contrast": {}, "limits": [],
              "prediction_comparison": {"memory": {"mse": .01}},
              "episodes": [replay], "receipt_sha256": "c" * 64}
    seen = []

    def run(received_core, received_atlas, received_contrast, spec):
        assert (received_core, received_atlas, received_contrast) == (core, atlas, contrast)
        seen.append(spec)
        return result

    def write(receipt, directory):
        assert receipt is result
        directory.mkdir(parents=True)
        path = directory / "receipt.json"
        path.write_text(json.dumps(receipt))
        return path

    module = types.ModuleType("poseidon.contrast_experiments")
    module.run_experiment = run
    module.verify_receipt = lambda receipt: {"verified": receipt is result}
    module.write_receipt = write
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exported = runtime.contrast_experiment(episodes=1, max_steps=8)
    assert seen[0].seed == 104000001
    assert exported["replay"] is replay
    assert exported["verification"]["verified"] is True
    assert exported["prediction_comparison"] == result["prediction_comparison"]
    assert exported["artifact_url"] == "/artifacts/contrast_experiments/receipt.json"
    assert (tmp_path / "outputs/contrast_experiments/receipt.json").is_file()


def test_contrast_cli_fit_partitions_and_receipt_output(tmp_path, monkeypatch, capsys):
    from poseidon import cli
    calls = []

    class Candidate:
        def save(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"sha256":"test"}')

    class ContrastAtlas:
        @classmethod
        def fit(cls, core, **kwargs):
            calls.append(kwargs)
            return Candidate(), {"schema": "fit-test", "selection_rows": ["raw"], "calibration_rows": ["raw"]}

    class Runtime:
        def __init__(self, *args):
            pass

        def core(self):
            return object()

    module = types.ModuleType("poseidon.contrast")
    module.ContrastAtlas = ContrastAtlas
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cli, "Poseidon", Runtime)
    monkeypatch.setattr(sys, "argv", ["poseidon", "contrast-fit", "--root", str(tmp_path)])
    cli.main()
    printed = json.loads(capsys.readouterr().out)
    assert calls[0]["train_seeds"] == list(range(101000001, 101000013))
    assert calls[0]["selection_seeds"] == list(range(102000001, 102000009))
    assert calls[0]["calibration_seeds"] == list(range(103000001, 103000009))
    assert "selection_rows" not in printed and "calibration_rows" not in printed
    assert json.loads((tmp_path / "outputs/contrast/fit_receipt.json").read_text())["selection_rows"] == ["raw"]


@pytest.fixture
def control_server(tmp_path, monkeypatch):
    runtime = Poseidon(tmp_path)
    monkeypatch.setattr(runtime, "core", lambda: RecordingPlanner())
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, 8787))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}", runtime
    server.shutdown()
    server.server_close()
    worker.join(2)


def post(url, payload):
    return urlopen(Request(url + "/api/respond", json.dumps(payload).encode(),
                           {"Content-Type": "application/json"}), timeout=5)


def test_world_api_settings_and_budget_validation(control_server):
    url, _ = control_server
    result = json.load(post(url, {"prompt": "Survive", "mode": "world", "scarcity": 2.75, "max_steps": 3}))
    assert result["episode"]["scarcity"] == 2.75
    assert result["episode"]["steps"] == result["episode"]["max_steps"] == 3
    for settings in ({"max_steps": 513}, {"max_steps": 0}, {"max_steps": True}, {"scarcity": 4.1}):
        with pytest.raises(HTTPError) as error:
            post(url, {"prompt": "Survive", "mode": "world", **settings})
        assert error.value.code == 400


def test_contrast_api_has_own_endpoint_and_bounded_settings(control_server, monkeypatch):
    url, runtime = control_server
    calls = []

    def experiment(seed, episodes, max_steps, scarcity):
        calls.append((seed, episodes, max_steps, scarcity))
        return {"ok": True}

    monkeypatch.setattr(runtime, "contrast_experiment", experiment, raising=False)

    def request(payload):
        return urlopen(Request(url + "/api/contrast-experiment", json.dumps(payload).encode(),
                               {"Content-Type": "application/json"}), timeout=5)

    result = json.load(request({"episodes": 1, "max_steps": 32, "scarcity": 3.5}))
    assert result["ok"] is True
    assert calls == [(104000001, 1, 32, 3.5)]
    for settings in ({"episodes": 9}, {"episodes": True}, {"max_steps": 31}, {"extra": 1}):
        with pytest.raises(HTTPError) as error:
            request(settings)
        assert error.value.code == 400


def test_stale_source_is_visible_and_blocks_experiment_requests(control_server, monkeypatch):
    import poseidon
    url, runtime = control_server
    monkeypatch.setattr(poseidon, "_SOURCE_SNAPSHOT", {})
    monkeypatch.setattr(runtime, "contrast_experiment", lambda *args: pytest.fail("stale process ran experiment"))
    status = json.load(urlopen(url + "/api/status", timeout=5))
    assert status["source_current"] is False
    assert status["restart_required"] is True
    assert "runtime.py" in status["changed_source_files"]
    with pytest.raises(HTTPError) as error:
        urlopen(Request(url + "/api/contrast-experiment", b'{"episodes":1,"max_steps":32}',
                        {"Content-Type": "application/json"}), timeout=5)
    assert error.value.code == 409
    body = json.load(error.value)
    assert body["restart_required"] is True
    assert "restart" in body["error"].lower()


def test_odyssey_api_has_own_endpoint_and_bounded_settings(control_server, monkeypatch):
    url, runtime = control_server
    calls = []

    def experiment(seed, episodes, max_steps, scarcity):
        calls.append((seed, episodes, max_steps, scarcity))
        return {"ok": True}

    monkeypatch.setattr(runtime, "odyssey_experiment", experiment, raising=False)

    def request(payload):
        return urlopen(Request(url + "/api/odyssey-experiment", json.dumps(payload).encode(),
                               {"Content-Type": "application/json"}), timeout=5)

    result = json.load(request({"episodes": 1, "max_steps": 32, "scarcity": 3.5}))
    assert result["ok"] is True
    assert calls == [(110000001, 1, 32, 3.5)]
    for settings in ({"episodes": 9}, {"episodes": True}, {"max_steps": 31}, {"extra": 1}):
        with pytest.raises(HTTPError) as error:
            request(settings)
        assert error.value.code == 400


@pytest.mark.parametrize("candidate", ["atlas", "contrast", "horizon", "odyssey"])
def test_malformed_core_pointer_is_reported_without_breaking_status(tmp_path, candidate):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/active_core.json").write_text('{"checkpoint":null}')
    path = tmp_path / f"outputs/{candidate}/atlas.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    result = Poseidon(tmp_path).status()
    assert result["core_ready"] is False
    assert result["core_error"]
    assert result[candidate]["ready"] is False
    assert result[candidate]["error"]


def test_status_api_returns_json_error_when_readiness_check_fails(control_server, monkeypatch):
    url, runtime = control_server

    def broken_status():
        raise OSError("Model filesystem unavailable")

    monkeypatch.setattr(runtime, "status", broken_status)
    with pytest.raises(HTTPError) as error:
        urlopen(url + "/api/status", timeout=5)
    assert error.value.code == 500
    body = json.load(error.value)
    assert body["type"] == "OSError"
    assert "filesystem" in body["error"]
