"""Research CLI choices must reach the fit or retain mandatory replay."""
import json
import sys

import pytest

from poseidon import cli


def test_fit_cli_forwards_explicit_architecture(tmp_path, monkeypatch, capsys):
    from poseidon.research import oracle_training
    calls = []

    def fake_fit(output, **kwargs):
        calls.append((output, kwargs))
        return {"activated": False}

    monkeypatch.setattr(oracle_training, "fit_oracle", fake_fit)
    monkeypatch.setattr(sys, "argv", ["poseidon", "oracle-fit", "--root", str(tmp_path),
                                     "--oracle-architecture", "linear-residual-v1"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["activated"] is False
    assert calls[0][1]["architecture"] == "linear-residual-v1"
    assert calls[0][1]["data_protocol"] == "episode-mixture-v2"


@pytest.mark.parametrize("arguments", [
    ["world", "--oracle-architecture", "linear-residual-v1"],
    ["oracle-screen", "missing.json", "--skip-replay"],
])
def test_cli_rejects_ignored_architecture_and_screen_without_replay(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["poseidon", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_screen_cli_resolves_receipt_under_root(tmp_path, monkeypatch, capsys):
    from poseidon.research import prediction_gate
    calls = []

    def fake_gate(path):
        calls.append(path)
        return {"state": "rejected", "promoted": False}

    monkeypatch.setattr(prediction_gate, "assess_prediction_gate", fake_gate)
    monkeypatch.setattr(sys, "argv", ["poseidon", "oracle-screen", "eval/receipt.json",
                                     "--root", str(tmp_path)])
    cli.main()
    assert calls == [tmp_path / "eval" / "receipt.json"]
    assert json.loads(capsys.readouterr().out)["state"] == "rejected"
