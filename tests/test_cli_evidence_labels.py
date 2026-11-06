import json


def test_unified_demo_cannot_be_mistaken_for_a_measured_run(capsys):
    from synapse.cli import main
    assert main(['--nodes', '16', 'unified-eval']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['measured'] is False
    assert result['evidence_kind'] == 'illustrative'


def test_cem_demo_does_not_fabricate_a_realized_outcome(capsys):
    from synapse.cli import main
    assert main(['--nodes', '16', 'world-model-cem', '--cem-horizon', '1']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['measured'] is False
    assert result['exploitation'] is None
    assert result['realized_outcomes'] == 0
