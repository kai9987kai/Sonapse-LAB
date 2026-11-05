import pytest

import poseidon
from poseidon import provenance


def test_source_snapshot_prevents_claiming_new_disk_code_as_loaded(monkeypatch):
    baseline = dict(poseidon._SOURCE_SNAPSHOT)
    changed = dict(baseline, **{"core.py": "0" * 64})
    monkeypatch.setattr(provenance, "capture_sources", lambda: changed)
    state = provenance.source_status()
    assert state["restart_required"] and not state["source_current"]
    assert state["changed_source_files"] == ["core.py"]
    with pytest.raises(provenance.StaleSourceError, match="Restart"):
        provenance.assert_source_current()


def test_missing_import_snapshot_fails_closed(monkeypatch):
    monkeypatch.delattr(poseidon, "_SOURCE_SNAPSHOT")
    with pytest.raises(provenance.StaleSourceError):
        provenance.assert_source_current()


def test_capture_and_return_are_detached(monkeypatch, tmp_path):
    (tmp_path / "one.py").write_text("x=1\n", encoding="utf-8")
    first = provenance.capture_sources(tmp_path)
    (tmp_path / "one.py").write_text("x=2\n", encoding="utf-8")
    assert provenance.capture_sources(tmp_path) != first
    monkeypatch.setattr(provenance, "capture_sources", lambda: dict(poseidon._SOURCE_SNAPSHOT))
    returned = provenance.assert_source_current()
    returned["core.py"] = "changed"
    assert poseidon._SOURCE_SNAPSHOT["core.py"] != "changed"


def test_nested_sources_are_captured_with_collision_free_relative_paths(tmp_path):
    (tmp_path / "research").mkdir()
    (tmp_path / "nexus_omega").mkdir()
    (tmp_path / "research/model.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / "nexus_omega/model.py").write_text("x=2\n", encoding="utf-8")
    captured = provenance.capture_sources(tmp_path)
    assert set(captured) == {"research/model.py", "nexus_omega/model.py"}
    assert captured["research/model.py"] != captured["nexus_omega/model.py"]


@pytest.mark.parametrize("operation", ["edit", "add", "remove"])
def test_nested_source_changes_require_restart(tmp_path, monkeypatch, operation):
    nested = tmp_path / "research"
    nested.mkdir()
    source = nested / "predictor.py"
    source.write_text("version=1\n", encoding="utf-8")
    monkeypatch.setattr(poseidon, "_SOURCE_SNAPSHOT", provenance.capture_sources(tmp_path))
    if operation == "edit":
        source.write_text("version=2\n", encoding="utf-8")
    elif operation == "add":
        (nested / "new.py").write_text("version=1\n", encoding="utf-8")
    else:
        source.unlink()
    capture = provenance.capture_sources
    monkeypatch.setattr(provenance, "capture_sources", lambda: capture(tmp_path))
    with pytest.raises(provenance.StaleSourceError, match="Restart"):
        provenance.assert_source_current()
