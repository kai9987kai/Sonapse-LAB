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


def test_runtime_audit_does_not_require_models(tmp_path):
    result = Poseidon(tmp_path).reasoning_audit(seed=101, samples=1)
    assert result['metrics']['correct'] == 6
    assert result['metrics']['wrong'] == 0
    assert result['calibrated'] is False


def test_poincare_status_reports_diagnostic_scope(tmp_path):
    result = Poseidon(tmp_path).poincare_status()
    assert result['total_verified_domains'] is None
    assert result['evidence_scope'] == 'five_synthetic_diagnostics'
    assert result['live_policy_activated'] is False


def test_http_audit_bounds_and_results(tmp_path):
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(Poseidon(tmp_path), 8787))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}/api/reasoning/audit'
    def post(payload):
        return urlopen(Request(url, json.dumps(payload).encode(), {'Content-Type': 'application/json'}))
    try:
        result = json.load(post({'seed': 101, 'samples': 1}))
        assert result['metrics']['total'] == 14
        assert result['metrics']['wrong'] == 0
        for payload in ({'samples': True}, {'samples': 33}, {'seed': '101'}, {'unknown': 1}):
            with pytest.raises(HTTPError) as exc:
                post(payload)
            assert exc.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_cli_audit_saves_and_replays_report(tmp_path):
    target = tmp_path / 'audit.json'
    result = subprocess.run([sys.executable, '-m', 'poseidon', 'reasoning-audit',
                             '--seed', '101', '--samples', '1', '--output', str(target)],
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    report = json.loads(target.read_text(encoding='utf-8'))
    assert report['metrics']['wrong'] == 0
    checked = subprocess.run([sys.executable, '-m', 'poseidon', 'verify-reasoning-audit', str(target)],
                             capture_output=True, text=True, timeout=40)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)['replay_verified'] is True
