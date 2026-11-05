"""Status remains useful when one optional component or local report is broken."""
import json
import hashlib

import pytest

from poseidon.runtime import Poseidon


def write_report(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_optional_component_failure_is_isolated(tmp_path, monkeypatch):
    runtime = Poseidon(tmp_path)

    def unavailable():
        raise OSError("optional component unavailable")

    monkeypatch.setattr(runtime, "tessera_status", unavailable)
    status = runtime.status()
    assert status["core_ready"] is False
    assert status["tessera"]["ready"] is False
    assert "optional component unavailable" in status["status_errors"]["tessera"]
    assert "noether" in status
    json.dumps(status, allow_nan=False)


@pytest.mark.parametrize("content", ["{broken", '{"loss":NaN}', '[]'])
def test_bad_reports_are_visible_without_breaking_status(tmp_path, content):
    write_report(tmp_path, "runs/language/report.json", content)
    status = Poseidon(tmp_path).status()
    assert "language" not in status["reports"]
    assert "report:language" in status["status_errors"]
    json.dumps(status, allow_nan=False)


def test_recent_metrics_recover_good_rows_around_partial_writes(tmp_path):
    write_report(tmp_path, "runs/language/metrics.jsonl",
                 '{"step":1}\n{incomplete\n{"step":2}\n')
    status = Poseidon(tmp_path).status()
    assert status["reports"]["language_recent"] == [{"step": 1}, {"step": 2}]
    assert "report:language_recent" in status["status_errors"]


def test_status_reads_only_bounded_recent_metrics(tmp_path, monkeypatch):
    path = write_report(tmp_path, "runs/tidal/training.jsonl",
                        '{"old":"' + 'x' * (300 * 1024) + '"}\n' +
                        ''.join(json.dumps({"step": step}) + '\n' for step in range(20)))
    original = type(path).read_text

    def no_full_log_read(self, *args, **kwargs):
        assert self != path, "status must not read an entire training log"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(path), "read_text", no_full_log_read)
    status = Poseidon(tmp_path).status()
    assert status["reports"]["core_recent"] == [{"step": step} for step in range(8, 20)]


def test_component_nonfinite_payload_is_isolated(tmp_path, monkeypatch):
    runtime = Poseidon(tmp_path)
    monkeypatch.setattr(runtime, "noether_status", lambda: {"ready": True, "value": float("inf")})
    status = runtime.status()
    assert status["noether"]["ready"] is False
    assert "noether" in status["status_errors"]
    json.dumps(status, allow_nan=False)


@pytest.mark.parametrize("controller,schema", [
    ("trident", "poseidon-trident-experiment-v1"),
    ("titan", "poseidon-titan-experiment-v2"),
])
def test_receipt_status_checks_integrity_without_claiming_replay(tmp_path, controller, schema):
    # A sealed metadata example is enough for integrity, never for semantic replay.
    receipt = {"schema": schema, "status": "completed-observational-evaluation",
               "promotion": False, "fitted_artifact_created": False,
               controller + "_telemetry": {"override_count": 2}}
    receipt["sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    path = write_report(tmp_path, f"outputs/{controller}_experiments/RECEIPT.json", json.dumps(receipt))
    runtime = Poseidon(tmp_path)
    status = getattr(runtime, controller + "_status")()
    assert status["receipt_integrity"] is True
    assert status["receipt_verified"] is False
    assert status["replay_verified"] is False
    assert status["verification_scope"] == "checksum_only"
    assert runtime._core is None
    receipt[controller + "_telemetry"]["override_count"] = 99
    path.write_text(json.dumps(receipt), encoding="utf-8")
    invalid = getattr(runtime, controller + "_status")()
    assert invalid["receipt_ready"] is False
    assert invalid["receipt_integrity"] is False
    assert "checksum" in invalid["error"].lower()


@pytest.mark.parametrize("controller", ["chimera", "hyperion", "mco"])
def test_unsealed_receipt_is_not_presented_as_evidence(tmp_path, controller):
    write_report(tmp_path, f"outputs/{controller}_experiments/RECEIPT.json",
                 '{"status":"completed","receipt_sha256":"made-up"}')
    status = getattr(Poseidon(tmp_path), controller + "_status")()
    assert status.get("receipt_integrity") is False
    assert status.get("receipt_ready") is False


def test_oversized_status_report_is_not_loaded(tmp_path):
    write_report(tmp_path, "runs/language/report.json", '{"large":"' + 'x' * (4 * 1024 * 1024) + '"}')
    status = Poseidon(tmp_path).status()
    assert "budget" in status["status_errors"]["report:language"]
    assert "language" not in status["reports"]
