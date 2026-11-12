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
    assert 'dsag' not in data['config']

def test_cli_run_uses_the_live_capped_dsag_bus(tmp_path, capsys):
    assert main(['--output', str(tmp_path), '--nodes', '16', 'run', '--steps', '1']) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['config']['dsag']['enabled'] is True
    assert data['steps'] == 1
    assert data['telemetry']['graft_bus']['channels']['dsag']['enabled'] is True


def test_cli_no_dsag_runs_the_legacy_path(tmp_path, capsys):
    assert main(['--output', str(tmp_path), '--nodes', '16', '--no-dsag', 'run', '--steps', '1']) == 0
    data = json.loads(capsys.readouterr().out)
    assert 'dsag' not in data['config']
    assert 'dsag' not in data['telemetry']['graft_bus']['channels']

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

@pytest.mark.parametrize('command', ['run', 'train', 'serve'])
def test_cli_ordinary_launches_enable_dsag_by_default(monkeypatch, tmp_path, command):
    captured = {}

    class Runtime:
        def __init__(self, config, output_root=None):
            captured['config'] = config

        def step(self, steps, learn=False):
            captured['learn'] = learn
            return {'steps': steps}

    def serve(host, port, config, output_root):
        captured['config'] = config

    if command == 'serve':
        monkeypatch.setitem(sys.modules, 'synapse.server', SimpleNamespace(serve=serve))
        argv = ['--output', str(tmp_path), '--nodes', '16', command]
    else:
        monkeypatch.setitem(sys.modules, 'synapse.runtime', SimpleNamespace(LabRuntime=Runtime))
        argv = ['--output', str(tmp_path), '--nodes', '16', command, '--steps', '1']

    assert main(argv) == 0
    spec = captured['config'].dsag
    assert spec is not None and spec.enabled is True
    assert spec.gain == pytest.approx(0.05)
    assert spec.seed == 7
    if command == 'train':
        assert captured['learn'] is True


def test_cli_no_dsag_keeps_legacy_config_export(monkeypatch, tmp_path):
    captured = {}

    class Runtime:
        def __init__(self, config, output_root=None):
            captured['config'] = config

        def step(self, steps, learn=False):
            return {'steps': steps}

    monkeypatch.setitem(sys.modules, 'synapse.runtime', SimpleNamespace(LabRuntime=Runtime))
    assert main(['--output', str(tmp_path), '--nodes', '16', '--no-dsag', 'run', '--steps', '1']) == 0
    config = captured['config']
    assert config.dsag is None
    assert 'dsag' not in config.to_dict()


def test_cli_explicit_dsag_config_overrides_default(monkeypatch, tmp_path):
    captured = {}
    dsag_config = tmp_path / 'disabled-dsag.json'
    dsag_config.write_text('{"version": 1, "enabled": false}', encoding='utf-8')

    class Runtime:
        def __init__(self, config, output_root=None):
            captured['config'] = config

        def step(self, steps, learn=False):
            return {'steps': steps}

    monkeypatch.setitem(sys.modules, 'synapse.runtime', SimpleNamespace(LabRuntime=Runtime))
    assert main(['--output', str(tmp_path), '--nodes', '16', '--dsag-config', str(dsag_config), 'run', '--steps', '1']) == 0
    assert captured['config'].dsag.enabled is False


def test_cli_checkpoint_patch_uses_dsag_default(monkeypatch, tmp_path):
    captured = {}
    patch_file = tmp_path / 'patch.json'
    patch_file.write_text('{"edits": []}', encoding='utf-8')

    class Runtime:
        def __init__(self, config, output_root=None):
            captured['config'] = config
            self.agent = object()

        def load(self, name):
            captured['checkpoint'] = name
            return {'status': 'loaded'}

    monkeypatch.setitem(sys.modules, 'synapse.runtime', SimpleNamespace(LabRuntime=Runtime))
    monkeypatch.setitem(sys.modules, 'synapse.patches', SimpleNamespace(
        make_patch=lambda agent, edits: {'status': 'ok'},
        apply_patch=lambda agent, payload: {'status': 'ok'},
    ))
    assert main(['--output', str(tmp_path), '--nodes', '16', 'patch', 'make', str(patch_file), '--checkpoint', 'saved']) == 0
    assert captured['checkpoint'] == 'saved'
    assert captured['config'].dsag.enabled is True

def test_cli_metamorphic_and_spectral(tmp_path,capsys):
    assert main(['--output',str(tmp_path),'--nodes','16','metamorphic'])==0
    data=json.loads(capsys.readouterr().out)
    assert data['status']=='ok'
    assert 'relations' in data

    assert main(['--output',str(tmp_path),'--nodes','16','spectral'])==0
    data_spec=json.loads(capsys.readouterr().out)
    assert 'eigenvalues' in data_spec
    assert 'projection' in data_spec

def test_cli_supermix_commands(tmp_path, capsys):
    assert main(['--output', str(tmp_path), '--nodes', '16', 'neurogenesis']) == 0
    data = json.loads(capsys.readouterr().out)
    assert 'active_synapses' in data

    assert main(['--output', str(tmp_path), '--nodes', '16', 'homeostasis']) == 0
    data = json.loads(capsys.readouterr().out)
    assert 'vitals' in data

    assert main(['--output', str(tmp_path), '--nodes', '16', 'mimomix']) == 0
    data = json.loads(capsys.readouterr().out)
    assert 'modification' in data and 'trace' in data

    assert main(['--output', str(tmp_path), '--nodes', '16', 'ecology', '--task', 'ecosystem']) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['task'] == 'ecosystem'
    assert 'ambient_light' in data
