"""Public research interfaces preserve audit scope and failure states."""
import json
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from poseidon.runtime import Poseidon
from poseidon.server import make_handler


def test_runtime_exposes_metamorphic_audit_without_models(tmp_path):
    report = Poseidon(tmp_path).metamorphic_audit(seed=101, samples=1)
    assert report['metrics']['total'] == 10
    assert report['metrics']['joint_correct_groups'] == 5
    assert report['calibrated'] is False


@pytest.fixture
def endpoint(tmp_path):
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(Poseidon(tmp_path), 8787))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()
    worker.join(timeout=3)


def _post(endpoint, path, payload):
    return urlopen(Request(endpoint + path, json.dumps(payload).encode(),
                           {'Content-Type': 'application/json'}))


def test_metamorphic_http_rejects_invalid_settings(endpoint):
    for payload in ({'samples': True}, {'samples': 33}, {'seed': '101'}, {'unknown': 1}):
        with pytest.raises(HTTPError) as error:
            _post(endpoint, '/api/reasoning/metamorphic-audit', payload)
        assert error.value.code == 400
    report = json.load(_post(endpoint, '/api/reasoning/metamorphic-audit', {'seed': 101, 'samples': 1}))
    assert report['metrics']['joint_correct_groups'] == 5


def test_episode_http_rejects_incomplete_evidence(endpoint):
    for payload in ({}, {'protocol': {}, 'outcomes_by_condition': {}}, {'unknown': 1}):
        with pytest.raises(HTTPError) as error:
            _post(endpoint, '/api/research/episode-risk', payload)
        assert error.value.code == 400


def _episode_input():
    from poseidon.research.episode_risk import make_episode_risk_protocol
    return {'protocol': make_episode_risk_protocol(
        policy_sha256='a' * 64, predictor_sha256='b' * 64, panel_sha256='c' * 64,
        thresholds={'margin': 0.1}, required_conditions=['id'],
        fit_episode_ids=[1], selection_episode_ids=[2], calibration_episode_ids=list(range(10, 20)),
        max_episode_risk=0.5), 'outcomes_by_condition': {
            'id': [{'episode_id': i, 'adverse_event': False} for i in range(10, 20)]}}


def test_episode_http_and_cli_recompute_supplied_outcomes(endpoint, tmp_path):
    data = _episode_input()
    report = json.load(_post(endpoint, '/api/research/episode-risk', data))
    assert report['eligible'] is True
    assert report['conditional_action_risk_established'] is False
    source, target = tmp_path / 'study.json', tmp_path / 'report.json'
    source.write_text(json.dumps(data), encoding='utf-8')
    screened = _cli('episode-risk', str(source), '--output', str(target))
    assert screened.returncode == 0, screened.stderr
    checked = _cli('verify-episode-risk', str(target))
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)['recomputed'] is True
    target.write_text('{}', encoding='utf-8')
    invalid = _cli('verify-episode-risk', str(target))
    assert invalid.returncode == 1
    assert json.loads(invalid.stdout)['valid'] is False


def test_episode_http_rejects_duplicate_json_keys(endpoint):
    data = _episode_input()
    encoded = json.dumps(data)
    encoded = encoded[:-1] + ',"protocol":' + json.dumps(data['protocol']) + '}'
    with pytest.raises(HTTPError) as error:
        urlopen(Request(endpoint + '/api/research/episode-risk', encoded.encode(),
                        {'Content-Type': 'application/json'}))
    assert error.value.code == 400
    assert 'duplicate' in json.load(error.value)['error'].lower()


def _cli(*arguments):
    return subprocess.run([sys.executable, '-m', 'poseidon', *arguments],
                          capture_output=True, text=True, timeout=45)


def test_metamorphic_cli_saves_and_replays_report(tmp_path):
    target = tmp_path / 'metamorphic.json'
    result = _cli('metamorphic-audit', '--seed', '101', '--samples', '1', '--output', str(target))
    assert result.returncode == 0, result.stderr
    assert json.loads(target.read_text(encoding='utf-8'))['metrics']['joint_correct_groups'] == 5
    checked = _cli('verify-metamorphic-audit', str(target))
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)['replay_verified'] is True


def test_episode_cli_rejects_nonfinite_input(tmp_path):
    target = tmp_path / 'episode.json'
    target.write_text('{"protocol":NaN,"outcomes_by_condition":{}}', encoding='utf-8')
    result = _cli('episode-risk', str(target))
    assert result.returncode == 2


def test_metamorphic_cli_reports_failed_checks_with_nonzero_exit(monkeypatch, capsys):
    from poseidon import cli
    from poseidon.research import metamorphic_audit
    monkeypatch.setattr(metamorphic_audit, 'run_audit', lambda **kwargs: {
        'metrics': {'wrong': 1, 'errors': 0, 'refusals': 0,
                    'relation_violations': 0, 'joint_correct_groups': 0, 'groups': 1}})
    monkeypatch.setattr(sys, 'argv', ['poseidon', 'metamorphic-audit'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)['metrics']['wrong'] == 1
