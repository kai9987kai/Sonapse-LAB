from synapse.capabilities import catalog, invoke, coverage_sweep

def test_inventory_covers_sources_and_typed_operations():
    c = catalog()
    assert c['source_module_counts']['poseidon'] >= 190
    assert {x['id'] for x in c['capabilities']} >= {
        'reasoning.solve', 'memory.retrieve', 'core.scene', 'graph.successor',
        'expanse.male_cns', 'core.tidal', 'graph.connectomes', 'rsi.meta_evaluate'
    }
    assert all('scope' in x and 'input_schema' in x for x in c['capabilities'])
    assert len(c['capabilities']) >= 45

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

def test_coverage_sweep():
    sweep = coverage_sweep()
    assert sweep['total'] >= 45
    assert sweep['ok'] >= 44
    assert sweep['error'] == 0
