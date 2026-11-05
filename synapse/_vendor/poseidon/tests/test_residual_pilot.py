"""A prediction failure may complete; invalid evidence must not."""
import hashlib
import json

import pytest

from tools import run_residual_pilot as pilot


@pytest.mark.parametrize("valid", [False, True])
def test_pilot_preserves_invalid_evidence_and_freezes_both_fits_first(tmp_path, monkeypatch, valid):
    calls = []

    def fit(path, *, architecture, **settings):
        calls.append(("fit", architecture))
        path.mkdir(parents=True)
        (path / "fit_receipt.json").write_text("{}")
        return {"artifact_path": str(path / "oracle.pt"), "receipt_sha256": "a" * 64, "training": {}}

    def evaluate(path, directory, **settings):
        calls.append(("evaluate", path))
        directory.mkdir()
        return {"receipt_sha256": "b" * 64, "artifact_sha256": "c" * 64,
                "model_sha256": "d" * 64, "results": {}}

    monkeypatch.setattr(pilot, "fit_oracle", fit)
    monkeypatch.setattr(pilot, "verify_fit_receipt", lambda *args, **kwargs: {"valid": True})
    monkeypatch.setattr(pilot, "evaluate_oracle", evaluate)
    monkeypatch.setattr(pilot, "assess_prediction_gate", lambda *args: {
        "state": "rejected", "promoted": False, "verification": {"valid": valid, "replayed": valid}})
    output = tmp_path / "pilot"
    if valid:
        result = pilot.run(output)
        assert result["complete"] is True and result["promoted"] is False
        assert len(result["evaluations"]) == 2
    else:
        with pytest.raises(ValueError, match="pilot incomplete"):
            pilot.run(output)
        assert not (output / "summary.json").exists()
        rejected = json.loads((output / "mlp-v1/prediction/prediction_gate.json").read_text())
        assert rejected["verification"]["valid"] is False
    assert [call[0] for call in calls[:3]] == ["fit", "fit", "evaluate"]
    digest = hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        pilot.run(output)
    assert hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest() == digest
