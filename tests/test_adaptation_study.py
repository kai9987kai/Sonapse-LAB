import copy
import json

import pytest


def test_configuration_caps_and_excludes_query_training():
    from synapse.adaptation_study import validate_config
    for config in ({'seeds': []}, {'tasks': ['invented']}, {'support_steps': 10001}, {'query_learning': True}, {'seeds': [7, 7]}):
        with pytest.raises(ValueError):
            validate_config(config)


def test_native_study_records_paired_frozen_arms_and_replays_aggregates(tmp_path, monkeypatch):
    from synapse.adaptation_study import run_adaptation_study, verify_adaptation_study, _json_hash
    config = {'seeds': [13], 'tasks': ['forage'], 'support_steps': 8, 'query_steps': 8, 'nodes_per_source': 16}
    result = run_adaptation_study(tmp_path/'study', config)
    assert result['status'] == 'complete'
    assert {row['arm'] for row in result['records']} == {'off', 'model', 'memory', 'shuffled'}
    assert len({row['query_seed'] for row in result['records']}) == 1
    assert len({row['support_digest'] for row in result['records']}) == 1
    assert all(row['learned_before'] == row['learned_after'] for row in result['records'])
    assert all(len(row['trace']) == 8 for row in result['records'])
    assert verify_adaptation_study(tmp_path/'study')['ok']
    tampered = copy.deepcopy(result)
    tampered['records'][0]['return'] += 1.
    assert not verify_adaptation_study(tampered)['ok']
    tampered = copy.deepcopy(result)
    tampered['summary']['arms']['off']['mean_return'] += 1.
    tampered.pop('integrity')
    tampered['integrity'] = _json_hash(tampered)
    checked = verify_adaptation_study(tampered)
    assert not checked['ok'] and 'summary recomputation' in checked['errors']
    repeated = run_adaptation_study(config=config)
    assert repeated['records'] == result['records']
    assert repeated['summary'] == result['summary']
    import synapse.adaptation_study as study
    changed_sources = dict(result['source_fingerprints'])
    changed_sources['agent.py'] = '0'*64
    monkeypatch.setattr(study, '_source_fingerprints', lambda: changed_sources)
    checked = verify_adaptation_study(result)
    assert checked['ok']
    assert 'current source differs: agent.py' in checked['warnings']
    assert {'python', 'platform', 'numpy', 'torch', 'scipy'} <= set(result['environment'])
    receipt = tmp_path/'study'/'receipt.json'
    stored = json.loads(receipt.read_text())
    stored['records'][0]['trace'][0]['reward'] += 1
    receipt.write_text(json.dumps(stored))
    assert not verify_adaptation_study(tmp_path/'study')['ok']


def test_cli_verification_failure_is_nonzero(monkeypatch, capsys):
    import synapse.adaptation_study as study
    from synapse.cli import main
    monkeypatch.setattr(study, 'verify_adaptation_study', lambda path: {'ok': False, 'errors': ['changed measurement']})
    assert main(['adaptation-verify', 'missing']) != 0
    assert 'changed measurement' in capsys.readouterr().out
