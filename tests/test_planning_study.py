import copy
import pytest


def test_measured_planning_study_pairs_worlds_freezes_queries_and_checks_tampering(tmp_path):
    from synapse.planning_study import run_planning_study, verify_planning_study
    config = {'seeds': [13], 'tasks': ['forage'], 'support_steps': 8, 'query_steps': 8,
              'nodes_per_source': 16, 'horizons': [1, 3]}
    receipt = run_planning_study(tmp_path/'study', config)
    assert receipt['status'] == 'complete'
    assert {r['arm'] for r in receipt['records']} == {'actor', 'horizon_1', 'horizon_3'}
    assert len({r['query_seed'] for r in receipt['records']}) == 1
    assert len({r['support_digest'] for r in receipt['records']}) == 1
    assert all(r['learned_before'] == r['learned_after'] for r in receipt['records'])
    assert all(len(r['trace']) == 8 for r in receipt['records'])
    assert receipt['costs']['query_update_calls'] == 0
    assert verify_planning_study(tmp_path/'study')['ok']
    repeated = run_planning_study(config=config)
    assert repeated['records'] == receipt['records']
    assert repeated['summary'] == receipt['summary']
    bad = copy.deepcopy(receipt)
    bad['records'][0]['trace'][0]['reward'] += 1
    assert not verify_planning_study(bad)['ok']
    from synapse.adaptation_study import _json_hash
    bad = copy.deepcopy(receipt)
    bad['summary']['arms']['actor']['mean_return'] += 1
    bad.pop('integrity')
    bad['integrity'] = _json_hash(bad)
    assert not verify_planning_study(bad)['ok']
    bad = copy.deepcopy(receipt)
    bad['records'][0]['steps'] = 0
    assert not verify_planning_study(bad)['ok']


@pytest.mark.parametrize('value', [[], [0], [6], [True], [1, 1]])
def test_planning_study_rejects_bad_horizons(value):
    from synapse.planning_study import validate_config
    with pytest.raises(ValueError):
        validate_config({'horizons': value})


def test_cli_planning_verify_uses_failure_exit_code(monkeypatch, capsys):
    import synapse.planning_study as study
    from synapse.cli import main
    monkeypatch.setattr(study, 'verify_planning_study', lambda path: {'ok': False, 'errors': ['altered score']})
    assert main(['planning-verify', 'missing']) != 0
    assert 'altered score' in capsys.readouterr().out
