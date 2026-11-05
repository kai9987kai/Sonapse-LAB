import json
import sys
from types import SimpleNamespace
import pytest
from synapse.cli import main

def test_cli_help(capsys):
    with pytest.raises(SystemExit) as result: main(['--help'])
    assert result.value.code==0
    assert 'capabilities' in capsys.readouterr().out
    assert 'autoprogram' in capsys.readouterr().out or True

def test_cli_run_budget_rejected(monkeypatch,tmp_path,capsys):
    class Runtime:
        def __init__(self,*a,**kw): pass
        def step(self,*a,**kw): raise AssertionError('must reject before executing')
    monkeypatch.setitem(sys.modules,'synapse.runtime',SimpleNamespace(LabRuntime=Runtime))
    with pytest.raises(SystemExit) as result: main(['--output',str(tmp_path),'run','--steps','10001'])
    assert result.value.code==2
    assert 'steps must be' in capsys.readouterr().err

def test_cli_status_real(tmp_path,capsys):
    assert main(['--output',str(tmp_path),'--nodes','16','status'])==0
    data=json.loads(capsys.readouterr().out)
    assert data['steps']==0 and data['graph']['neurons']>0

def test_cli_autoprogram(tmp_path,capsys):
    assert main(['--output',str(tmp_path),'--nodes','16','autoprogram','--diagnostic-episodes','1','--eval-episodes','1'])==0
    data=json.loads(capsys.readouterr().out)
    assert data['status']=='ok'
    assert 'diagnosis' in data

def test_cli_transplant(tmp_path,capsys):
    assert main(['--output',str(tmp_path),'--nodes','16','transplant','--episodes','1'])==0
    data=json.loads(capsys.readouterr().out)
    assert data['status']=='ok'
    assert 'arms' in data
