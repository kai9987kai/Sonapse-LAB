from synapse.capabilities import catalog, invoke, coverage_sweep

def test_inventory_covers_sources_and_typed_operations():
    c=catalog()
    assert c['source_module_counts']['poseidon']>=190
    assert {x['id'] for x in c['capabilities']} >= {'reasoning.solve','memory.retrieve','core.scene','graph.successor'}
    assert all('scope' in x and 'input_schema' in x for x in c['capabilities'])

def test_real_tool_result_reaches_memory_consumer():
    r=invoke('reasoning.solve',{'expression':'12 + 30'})
    assert r['status']=='ok'
    assert r['consumer']['stored']
    m=invoke('memory.retrieve',{'query':'12 + 30'})
    assert m['status']=='ok'
    assert '42' in str(r['result'])

def test_unknown_and_invalid_arguments_are_structured():
    assert invoke('os.system',{'command':'bad'})['status']=='unavailable'
    assert invoke('graph.successor',{'num_states':10000000})['status']=='error'

def test_coverage_sweep():
    sweep = coverage_sweep()
    assert sweep['total'] >= 20
    assert sweep['ok'] >= 18
    assert sweep['error'] == 0
