"""The historical display smoke check must not mask regressions in automation."""
import sys

import pytest
import reasoning_bench


@pytest.mark.parametrize('flag', [[], ['--json']])
def test_display_smoke_failure_has_nonzero_exit(monkeypatch, capsys, flag):
    monkeypatch.setattr(sys, 'argv', ['reasoning_bench.py', *flag])
    monkeypatch.setattr(reasoning_bench, 'run', lambda: {
        'total': 1, 'correct': 0, 'accuracy': 0, 'wrong': 1, 'unanswered': 0,
        'by_category': {}, 'rows': []})
    assert reasoning_bench.main() == 1
    capsys.readouterr()
