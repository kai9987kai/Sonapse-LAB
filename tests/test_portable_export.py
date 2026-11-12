import base64
from copy import deepcopy
import hashlib
from types import SimpleNamespace
import threading
import time
import urllib.request

import numpy as np
import pytest
import torch

from synapse.portable_export import PortableReplayRecorder, write_portable_html
from synapse.server import Workbench, create_server
from synapse.stateio import load_state, save_state_snapshot


def test_snapshot_archive_round_trips_with_existing_state_loader(tmp_path):
    path = tmp_path / "portable.pt"
    value = {
        "weights": torch.tensor([0.25, -0.5]),
        "rng": np.random.default_rng(17).bit_generator.state,
        "history": [(1, "step"), (2, "step")],
    }

    identity = save_state_snapshot(path, value)

    assert len(identity) == 64
    assert torch.equal(load_state(path)["weights"], value["weights"])
    assert load_state(path)["history"] == value["history"]


def test_portable_html_embeds_checkpoint_and_escapes_snapshot_json(tmp_path):
    checkpoint = tmp_path / "state.pt"
    checkpoint.write_bytes(b"checkpoint\x00payload")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    destination = tmp_path / "Sonapse_Portable_Simulation_20261111_120000_ab12cd34.html"
    overview = {"telemetry": "</script><script>alert(1)</script>", "steps": 3}

    result = write_portable_html(
        destination,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        overview=overview,
        replay_history={"samples": []},
        metadata={"step": 3, "continuation": "native-only"},
    )

    html = destination.read_text(encoding="utf-8")
    assert base64.b64encode(b"checkpoint\x00payload").decode() in html
    assert "sonapse-portable-html-v1" in html
    assert "\u003c/script\u003e" in html
    assert "https://" not in html and "http://" not in html
    assert result["checkpoint_sha256"] == checkpoint_sha256
    assert result["html_bytes"] == destination.stat().st_size
    assert result["html_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_portable_html_rejects_wrong_checkpoint_digest(tmp_path):
    checkpoint = tmp_path / "state.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="checkpoint checksum"):
        write_portable_html(
            tmp_path / "bad.html",
            checkpoint_path=checkpoint,
            checkpoint_sha256="0" * 64,
            overview={},
            replay_history={"samples": []},
            metadata={},
        )


class _PortableRuntime:
    def __init__(self):
        self.config = SimpleNamespace(seed=9, task="forage", to_dict=lambda: {"seed": 9, "task": "forage"})
        self.total_steps = 0
        self.agent = SimpleNamespace(
            brain=SimpleNamespace(n=4, r=torch.zeros(4)),
            graph=SimpleNamespace(base_pre=[0, 1], graft_pre=[2]),
        )
        self.transition_taps = []

    def checkpoint_payload(self):
        return {"schema": 1, "steps": self.total_steps, "weight": torch.tensor([float(self.total_steps)])}

    def snapshot(self):
        return {
            "steps": self.total_steps,
            "config": self.config.to_dict(),
            "graph": {
                "neurons": 4,
                "nodes": [
                    {"id": i, "origin": 0, "role": "test", "activity": float(self.agent.brain.r[i])}
                    for i in range(4)
                ],
                "edges": [[0, 1, 0.5], [1, 2, -0.25]],
            },
        }

    def step(self, steps=1, learn=False, action=None):
        for _ in range(steps):
            previous = np.zeros(1, dtype=np.float32)
            self.agent.brain.r = torch.full((4,), float(self.total_steps + 1) / 10)
            for tap in tuple(self.transition_taps):
                tap(previous, np.zeros(5, dtype=np.float32), 0.25, previous, False)
            self.total_steps += 1
        return self.snapshot()


def test_export_job_captures_checkpoint_records_replay_and_downloads_html(tmp_path):
    runtime = _PortableRuntime()
    app = Workbench(runtime=runtime, output_root=tmp_path)
    with app.lock:
        runtime.step(2)
    started = app.post("/api/export", {})
    with app.lock:
        runtime.step(1)
    deadline = time.monotonic() + 20
    job = started
    while time.monotonic() < deadline and job["status"] not in ("completed", "failed"):
        time.sleep(0.02)
        job = app.get(f"/api/export/status/{started['id']}")

    assert job["status"] == "completed", job
    result = job["result"]
    assert result["simulation_step"] in (2, 3)
    assert result["replay_frames"] == result["simulation_step"]
    assert runtime.total_steps >= result["simulation_step"]
    exported = app.portable_export_path(result["filename"])
    html = exported.read_text(encoding="utf-8")
    embedded = html.split('<script id="checkpoint-bytes" type="application/octet-stream">', 1)[1]
    embedded = embedded.split("</script>", 1)[0]
    checkpoint = tmp_path / "recovered.pt"
    checkpoint.write_bytes(base64.b64decode(embedded))
    assert load_state(checkpoint)["steps"] == result["simulation_step"]
    assert "no browser simulation engine" in html

    server = create_server(port=0, workbench=app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}{result['download_url']}")
        assert response.headers["Content-Disposition"].endswith(result["filename"] + '"')
        assert response.read() == exported.read_bytes()
    finally:
        server.shutdown()
        server.server_close()


def test_small_live_lab_export_contains_loadable_complete_runtime_state(tmp_path):
    from synapse.config import SynapseConfig
    from synapse.runtime import LabRuntime

    runtime = LabRuntime(SynapseConfig(nodes_per_source=16, seed=27), tmp_path)
    app = Workbench(runtime=runtime, output_root=tmp_path)
    with app.lock:
        runtime.step(1, learn=True)
    started = app.start_portable_export()
    deadline = time.monotonic() + 30
    job = started
    while time.monotonic() < deadline and job["status"] not in ("completed", "failed"):
        time.sleep(0.02)
        job = app.get(f"/api/export/status/{started['id']}")

    assert job["status"] == "completed", job
    filename = job["result"]["filename"]
    html = app.portable_export_path(filename).read_text(encoding="utf-8")
    embedded = html.split('<script id="checkpoint-bytes" type="application/octet-stream">', 1)[1]
    embedded = embedded.split("</script>", 1)[0]
    checkpoint = tmp_path / "actual-runtime.pt"
    checkpoint.write_bytes(base64.b64decode(embedded))
    restored = load_state(checkpoint)
    assert restored["steps"] == 1
    assert restored["config"]["seed"] == 27
    assert "modules" in restored["agent"]
    assert "optimizer" in restored["agent"]
    assert "world" in restored and "monitors" in restored


def test_replay_recorder_caps_frames_and_marks_truncation():
    runtime = _PortableRuntime()
    recorder = PortableReplayRecorder(runtime, max_frames=2, max_neurons=2)
    runtime.transition_taps.append(recorder)
    runtime.step(3)

    history = recorder.export_state()

    assert [frame["step"] for frame in history["frames"]] == [2, 3]
    assert history["truncated"] is True
    assert len(history["sampled_neuron_ids"]) == 2


def test_replay_recorder_deepcopy_isolated_from_candidate_runtime():
    recorder = PortableReplayRecorder(_PortableRuntime(), max_frames=8, max_neurons=2)

    cloned = deepcopy(recorder)

    assert cloned is not recorder
    assert cloned.runtime is None
    assert cloned.frames.maxlen == 8
