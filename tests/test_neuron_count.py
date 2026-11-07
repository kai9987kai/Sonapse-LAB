import json
import numpy as np
import pytest

from synapse.config import SynapseConfig
from synapse.runtime import LabRuntime


def test_exact_total_builds_1000_neurons_and_replays(tmp_path):
    cfg = SynapseConfig(total_neurons=1000)
    assert cfg.source_node_counts() == (334, 333, 333)
    runtime = LabRuntime(cfg, tmp_path)
    assert runtime.agent.brain.n == runtime.agent.graph.n == 1000
    assert len(runtime.snapshot()['graph']['nodes']) == 1000
    assert np.bincount(runtime.agent.graph.origin).tolist() == [334, 333, 333]
    runtime.step(1, learn=True)
    runtime.save('1000-neurons')
    expected = runtime.step(1)
    runtime.load('1000-neurons')
    assert runtime.step(1) == expected


def test_explicit_legacy_per_source_count_remains_48():
    cfg = SynapseConfig(nodes_per_source=16)
    assert cfg.source_node_counts() == (16, 16, 16)
    saved = cfg.to_dict()
    saved.pop('total_neurons')
    assert cfg.matches_saved(saved)


@pytest.mark.parametrize('value', [True, 0, 47, 30001, 1000.5])
def test_total_count_validation(value):
    with pytest.raises(ValueError):
        SynapseConfig(total_neurons=value)


def test_cli_default_is_exactly_4000(capsys):
    from synapse.cli import main
    assert main(['status']) == 0
    state = json.loads(capsys.readouterr().out)
    assert state['graph']['neurons'] == 4000
    assert state['config']['total_neurons'] == 4000


def test_api_total_neuron_reset(tmp_path):
    from synapse.server import Workbench
    app = Workbench(LabRuntime(SynapseConfig(nodes_per_source=16), tmp_path), output_root=tmp_path)
    config = app.runtime.config.to_dict()
    config['total_neurons'] = 4000
    state = app.post('/api/reset', {'config': config})
    assert state['graph']['neurons'] == 4000


def test_default_spiking_command_still_has_a_numeric_size(capsys):
    from synapse.cli import main
    assert main(['spiking']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result


def test_exact_total_builds_2000_neurons_and_replays(tmp_path):
    cfg = SynapseConfig(total_neurons=2000)
    assert cfg.source_node_counts() == (667, 667, 666)
    runtime = LabRuntime(cfg, tmp_path)
    assert runtime.agent.brain.n == runtime.agent.graph.n == 2000
    assert len(runtime.snapshot()['graph']['nodes']) == 2000
    assert np.bincount(runtime.agent.graph.origin).tolist() == [667, 667, 666]
    runtime.step(1, learn=True)
    runtime.save('2000-neurons')
    expected = runtime.step(1)
    runtime.load('2000-neurons')
    assert runtime.step(1) == expected


def test_exact_total_builds_4000_neurons_and_replays(tmp_path):
    cfg = SynapseConfig(total_neurons=4000)
    assert cfg.source_node_counts() == (1334, 1333, 1333)
    runtime = LabRuntime(cfg, tmp_path)
    assert runtime.agent.brain.n == runtime.agent.graph.n == 4000
    assert len(runtime.snapshot()['graph']['nodes']) == 4000
    assert np.bincount(runtime.agent.graph.origin).tolist() == [1334, 1333, 1333]
    runtime.step(1, learn=True)
    runtime.save('4000-neurons')
    expected = runtime.step(1)
    runtime.load('4000-neurons')
    assert runtime.step(1) == expected
