import json, threading, urllib.request, urllib.error
import pytest
from synapse.server import create_server

class Runtime:
    def snapshot(self): return {'step':0}
    def step(self, steps=1, learn=False): return {'step':steps}

def test_server_boundaries(tmp_path):
    server=create_server(port=0,runtime=Runtime(),output_root=tmp_path)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        assert json.load(urllib.request.urlopen(base+'/api/state'))['step']==0
        req=urllib.request.Request(base+'/api/step',data=b'{"steps":10001}',headers={'Content-Type':'application/json'})
        with pytest.raises(urllib.error.HTTPError) as error: urllib.request.urlopen(req)
        assert error.value.code==400
        req=urllib.request.Request(base+'/api/step',data=b'{}',headers={'Content-Type':'application/json','Origin':'https://evil.example'})
        with pytest.raises(urllib.error.HTTPError) as error: urllib.request.urlopen(req)
        assert error.value.code==403
        req=urllib.request.Request(base+'/api/step',data=b'{"steps":2}',headers={'Content-Type':'application/json'})
        assert json.load(urllib.request.urlopen(req))['step']==2
    finally: server.shutdown();server.server_close()

def test_body_host_and_checkpoint_validation(tmp_path):
    server=create_server(port=0,runtime=Runtime(),output_root=tmp_path)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f'http://127.0.0.1:{server.server_port}'
    def reject(path,body,headers,expected):
        req=urllib.request.Request(base+path,data=body,headers=headers)
        with pytest.raises(urllib.error.HTTPError) as e: urllib.request.urlopen(req)
        assert e.value.code==expected
    try:
        reject('/api/step',b'{}',{'Content-Type':'text/plain'},415)
        reject('/api/step',b'[]',{'Content-Type':'application/json'},400)
        reject('/api/step',b'{"steps":true}',{'Content-Type':'application/json'},400)
        reject('/api/step',b'{"steps":NaN}',{'Content-Type':'application/json'},400)
        reject('/api/save',b'{"name":"../escape"}',{'Content-Type':'application/json'},400)
        reject('/api/step',b'{}',{'Content-Type':'application/json','Host':'evil.example'},403)
        reject('/api/step',b' '*1048577,{'Content-Type':'application/json'},413)
        assert urllib.request.urlopen(base+'/').headers['Content-Security-Policy']
    finally: server.shutdown();server.server_close()

def test_jobs_failure_cancel_and_restart(monkeypatch,tmp_path):
    import sys,time
    from types import SimpleNamespace
    from synapse.server import Workbench
    app=Workbench(Runtime(),output_root=tmp_path)
    def fail(*a,**kw): raise ValueError('expected study failure')
    monkeypatch.setitem(sys.modules,'synapse.experiments',SimpleNamespace(validate_config=lambda c:c,run_study=fail))
    job=app.start_study({})
    for _ in range(100):
        if app.jobs[job['id']]['status']=='failed': break
        time.sleep(.01)
    assert app.jobs[job['id']]['status']=='failed'
    assert 'expected study failure' in app.jobs[job['id']]['error']
    def wait_cancel(path,config,progress,cancelled):
        progress({'phase':'waiting'})
        for _ in range(100):
            if cancelled(): return {'status':'cancelled'}
            time.sleep(.01)
        raise AssertionError('cancellation not received')
    monkeypatch.setitem(sys.modules,'synapse.experiments',SimpleNamespace(validate_config=lambda c:c,run_study=wait_cancel))
    job=app.start_study({}); app.post('/api/jobs/cancel',{'id':job['id']})
    for _ in range(100):
        if app.jobs[job['id']]['status']=='cancelled': break
        time.sleep(.01)
    assert app.jobs[job['id']]['status']=='cancelled'
    loaded=Workbench(Runtime(),output_root=tmp_path)
    assert loaded.jobs[job['id']]['status']=='cancelled'
    stale={'id':'stale','status':'running'}; app.persist(stale)
    loaded=Workbench(Runtime(),output_root=tmp_path)
    assert loaded.jobs['stale']['status']=='interrupted'

def test_server_autoprogram_and_transplant(tmp_path):
    from synapse.config import SynapseConfig
    from synapse.runtime import LabRuntime
    cfg = SynapseConfig(seed=42, nodes_per_source=16)
    rt = LabRuntime(cfg, output_root=tmp_path)
    server = create_server(port=0, runtime=rt, output_root=tmp_path)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        req = urllib.request.Request(
            base + '/api/autoprogram',
            data=b'{"diagnostic_episodes":1,"eval_episodes":1}',
            headers={'Content-Type': 'application/json'}
        )
        res = json.load(urllib.request.urlopen(req))
        assert res['status'] == 'ok'
        assert 'diagnosis' in res

        req2 = urllib.request.Request(
            base + '/api/transplant',
            data=b'{"episodes":1}',
            headers={'Content-Type': 'application/json'}
        )
        res2 = json.load(urllib.request.urlopen(req2))
        assert res2['status'] == 'ok'
        assert 'arms' in res2

        req3 = urllib.request.Request(
            base + '/api/metamorphic',
            data=b'{}',
            headers={'Content-Type': 'application/json'}
        )
        res3 = json.load(urllib.request.urlopen(req3))
        assert res3['status'] == 'ok'
        assert 'relations' in res3

        req4 = urllib.request.Request(
            base + '/api/spectral',
            data=b'{}',
            headers={'Content-Type': 'application/json'}
        )
        res4 = json.load(urllib.request.urlopen(req4))
        assert 'harmonics' in res4
        assert 'projection' in res4
    finally:
        server.shutdown()
        server.server_close()
