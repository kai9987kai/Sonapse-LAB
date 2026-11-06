from synapse.capabilities import catalog, invoke, coverage_sweep

def test_inventory_covers_sources_and_typed_operations():
    c = catalog()
    assert c['source_module_counts']['poseidon'] >= 190
    assert {x['id'] for x in c['capabilities']} >= {
        'reasoning.solve', 'memory.retrieve', 'core.scene', 'graph.successor',
        'expanse.male_cns', 'core.tidal', 'graph.connectomes', 'rsi.meta_evaluate',
        'neuromodulation.state', 'spectral.harmonics', 'hypernetwork.synthesize',
        'metamorphic.audit', 'structural.topology',
    }
    assert all('scope' in x and 'input_schema' in x for x in c['capabilities'])
    assert len(c['capabilities']) >= 65

def test_real_tool_result_reaches_memory_consumer():
    r = invoke('reasoning.solve', {'expression': '12 + 30'})
    assert r['status'] == 'ok'
    assert r['consumer']['stored']
    m = invoke('memory.retrieve', {'query': '12 + 30'})
    assert m['status'] == 'ok'
    assert '42' in str(r['result'])

def test_unknown_and_invalid_arguments_are_structured():
    assert invoke('os.system', {'command': 'bad'})['status'] == 'unavailable'
    assert invoke('graph.successor', {'num_states': 10000000})['status'] == 'error'

def test_real_connectome_male_cns():
    res = invoke('expanse.male_cns', {'query_type': 'MBON01'})
    assert res['status'] == 'ok'
    assert res['total_cell_types'] == 11751
    assert 'MBON01' in res['cell_type']

def test_graph_connectomes():
    res = invoke('graph.connectomes', {})
    assert res['status'] == 'ok'
    assert len(res['connectomes']) == 3
    for c in res['connectomes']:
        assert 'name' in c and 'neurons' in c

def test_frontier_capabilities():
    nm = invoke('neuromodulation.state', {'reward': 1.0})
    assert nm['status'] == 'ok'
    assert 'dopamine' in nm['neuromodulators']

    spec = invoke('spectral.harmonics', {})
    assert spec['status'] == 'ok'
    assert spec['num_modes'] == 8

    hyp = invoke('hypernetwork.synthesize', {'return': 0.4})
    assert hyp['status'] == 'ok'
    assert hyp['writes'] > 0

    meta = invoke('metamorphic.audit', {})
    assert meta['status'] == 'ok'
    assert 'relations' in meta

    struct = invoke('structural.topology', {})
    assert struct['status'] == 'ok'
    assert struct['neurons'] > 0



def test_supermix_capabilities():
    neuro = invoke('structural.neurogenesis', {'neurons': 16, 'steps': 5})
    assert neuro['status'] == 'ok'
    assert 'summary' in neuro
    assert 'mean_covariance' in neuro

    homeo = invoke('motivation.homeostasis', {'action': [0.8, 0.2, 0.1, 0.0, 0.0], 'reward': 0.1})
    assert homeo['status'] == 'ok'
    assert 'vitals' in homeo
    assert 'max_allowed_thrust' in homeo
    assert 'modulated_neuromodulators' in homeo

    mimo = invoke('reasoning.mimomix', {'max_ponder': 3})
    assert mimo['status'] == 'ok'
    assert 'action_modification' in mimo
    assert 'trace' in mimo

    eco = invoke('world.complex_ecology', {'task': 'ecosystem'})
    assert eco['status'] == 'ok'
    assert 'ambient_light' in eco
    assert 'predator_count' in eco

def test_coverage_sweep():
    sweep = coverage_sweep()
    assert sweep['total'] >= 70
    assert sweep['ok'] >= 70
    assert sweep['error'] == 0
