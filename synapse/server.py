"""Loopback-only workbench and bounded background study jobs."""
from __future__ import annotations
import argparse
import dataclasses
import ipaddress
import json
import mimetypes
from pathlib import Path
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from .config import SynapseConfig, ARTIFACTS

MAX_BODY = 1024 * 1024

def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value

def safe_name(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',value):
        raise ValueError('name must contain 1–64 letters, digits, underscores or hyphens')
    return value

class Workbench:
    def __init__(self,runtime=None,config=None,output_root=None):
        self.root=Path(output_root or ARTIFACTS).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        if runtime is None:
            from .runtime import LabRuntime
            runtime=LabRuntime(config or SynapseConfig(),output_root=self.root)
        self.runtime=runtime; self.lock=threading.RLock(); self.job_lock=threading.RLock()
        self.jobs={}; self.cancel={}; self.receipts={}
        jobsdir=self.root/'jobs'; jobsdir.mkdir(exist_ok=True)
        for path in jobsdir.glob('*.json'):
            try:
                job=json.loads(path.read_text()); self.jobs[job['id']]=job
                if job['status'] in ('queued','running'):
                    job.update(status='interrupted',error='Server restarted before this job completed'); self.persist(job)
            except (ValueError,KeyError,OSError): pass
    def persist(self,job):
        path=self.root/'jobs'/f"{job['id']}.json"; tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(job,allow_nan=False,default=str),encoding='utf-8'); tmp.replace(path)
    def start_study(self,config):
        if not isinstance(config,dict): raise ValueError('config must be an object')
        from .experiments import validate_config
        config=validate_config(config)
        for key,high in [('generations',20),('steps',10000),('episodes',100),('population',32),('seeds',16)]:
            if key in config:
                if key=='seeds' and isinstance(config[key],list):
                    if len(config[key])>high: raise ValueError('too many seeds')
                    for seed in config[key]: integer(seed,'seed',0,2**31-1)
                else: integer(config[key],key,1,high)
        with self.job_lock:
            if any(j['status'] in ('queued','running') for j in self.jobs.values()): raise ValueError('a study is already active')
            ident=uuid.uuid4().hex; event=threading.Event(); self.cancel[ident]=event
            job={'id':ident,'status':'queued','created':time.time(),'config':config,'progress':{},'result':None,'error':None}
            self.jobs[ident]=job; self.persist(job)
        def worker():
            def progress(value):
                with self.job_lock: job['progress']=value; self.persist(job)
            try:
                from .experiments import run_study
                with self.job_lock: job['status']='running'; self.persist(job)
                result=run_study(self.root/'studies'/ident,config,progress=progress,cancelled=event.is_set)
                with self.job_lock: job.update(status='cancelled' if event.is_set() else 'completed',result=result)
            except Exception as exc:
                with self.job_lock: job.update(status='cancelled' if event.is_set() else 'failed',error=f'{type(exc).__name__}: {exc}')
            finally:
                with self.job_lock: job['finished']=time.time(); self.persist(job)
        threading.Thread(target=worker,daemon=True,name=f'study-{ident[:8]}').start()
        return dict(job)
    def get(self,path):
        if path=='/api/jobs':
            with self.job_lock: return {'jobs':json.loads(json.dumps(list(self.jobs.values()),default=str))}
        with self.lock:
            if path=='/api/state': return self.runtime.snapshot()
            if path=='/api/capabilities':
                from .capabilities import catalog
                return catalog()
            if path=='/api/tensors':
                return {'tensors':{k:{'shape':list(v.shape),'count':v.numel(),'min':float(v.detach().min()),'max':float(v.detach().max())} for k,v in self.runtime.agent.editable_tensors().items()}}
            if path=='/api/checkpoints':
                return {'checkpoints':sorted(p.stem for p in self.root.glob('*.pt') if p.is_file())}
        raise KeyError('unknown API endpoint')
    def post(self,path,data):
        if path=='/api/studies': return self.start_study(data.get('config',{}))
        if path=='/api/jobs/cancel':
            with self.job_lock:
                ident=safe_name(data.get('id')); job=self.jobs[ident]
                if ident in self.cancel: self.cancel[ident].set(); job['cancel_requested']=True; self.persist(job)
                return dict(job)
        with self.lock:
            if path=='/api/step':
                learn=data.get('learn',False)
                if type(learn) is not bool: raise ValueError('learn must be boolean')
                return self.runtime.step(integer(data.get('steps',1),'steps',1,1000),learn=learn)
            if path=='/api/reset':
                cfg=data.get('config')
                if cfg is not None:
                    if not isinstance(cfg,dict): raise ValueError('config must be object')
                    from .runtime import LabRuntime
                    self.runtime=LabRuntime(SynapseConfig(**cfg),output_root=self.root)
                    self.receipts.clear()
                    return self.runtime.snapshot()
                return self.runtime.reset(task=data.get('task'),seed=data.get('seed'))
            if path in ('/api/save','/api/load'):
                return getattr(self.runtime,path.rsplit('/',1)[1])(safe_name(data.get('name')))
            if path=='/api/invoke':
                from .capabilities import invoke
                return invoke(data.get('id'),data.get('payload',{}))
            if path=='/api/patch/make':
                from .patches import make_patch
                return make_patch(self.runtime.agent,data.get('edits',[]),data.get('description',''))
            if path=='/api/patch/preview':
                from .patches import preview_patch
                return preview_patch(self.runtime.agent,data.get('packet',{}))
            if path=='/api/patch/apply':
                from .patches import apply_patch
                receipt=apply_patch(self.runtime.agent,data.get('packet',{})); ident=uuid.uuid4().hex; self.receipts[ident]=receipt
                while len(self.receipts)>8: self.receipts.pop(next(iter(self.receipts)))
                return {'receipt_id':ident,'receipt':receipt}
            if path=='/api/patch/rollback':
                from .patches import rollback
                ident=data.get('receipt_id'); receipt=self.receipts[ident]
                rollback(self.runtime.agent,receipt); del self.receipts[ident]
                return {'rolled_back':ident}
            if path=='/api/autoprogram':
                from .auto_programmer import execute_closed_loop_repair
                return execute_closed_loop_repair(
                    self.runtime.agent,
                    task=data.get('task', self.runtime.config.task),
                    num_diagnostic_episodes=integer(data.get('diagnostic_episodes', 3), 'diagnostic_episodes', 1, 10),
                    num_eval_episodes=integer(data.get('eval_episodes', 3), 'eval_episodes', 1, 10),
                    min_gain=float(data.get('min_gain', -0.05)),
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1),
                )
            if path=='/api/transplant':
                from .transplantation import extract_donor_subcircuit, run_transplantation_experiment
                donor = extract_donor_subcircuit(self.runtime.agent, source_task=data.get('donor_task', 'forage'))
                return run_transplantation_experiment(
                    self.runtime.agent,
                    donor=donor,
                    test_task=data.get('test_task', 'delayed_match'),
                    episodes=integer(data.get('episodes', 3), 'episodes', 1, 10),
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1),
                )
        raise KeyError('unknown API endpoint')

def create_server(host='127.0.0.1',port=8765,runtime=None,config=None,output_root=None):
    if host not in ('127.0.0.1','localhost'): raise ValueError('server must bind to IPv4 loopback')
    app=Workbench(runtime,config,output_root); web=Path(__file__).parent/'web'
    class Handler(BaseHTTPRequestHandler):
        timeout=15
        def log_message(self,*args): pass
        def send(self,status,data,content='application/json'):
            raw=json.dumps(data,allow_nan=False,default=str).encode() if content=='application/json' else data
            self.send_response(status); self.send_header('Content-Type',content); self.send_header('Content-Length',str(len(raw)))
            self.send_header('X-Content-Type-Options','nosniff'); self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'")
            self.end_headers(); self.wfile.write(raw)
        def local(self):
            host=self.headers.get('Host',''); parsed=urlparse('http://'+host)
            if parsed.hostname not in ('127.0.0.1','localhost') or parsed.port != self.server.server_port: return False
            origin=self.headers.get('Origin')
            if origin:
                parsed=urlparse(origin)
                if parsed.scheme!='http' or parsed.netloc!=host: return False
            return True
        def do_GET(self):
            try:
                if not self.local(): return self.send(403,{'error':'loopback host and same origin required'})
                path=urlparse(self.path).path
                if path.startswith('/api/'): return self.send(200,app.get(path))
                names={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/style.css':'style.css'}
                if path not in names: return self.send(404,{'error':'not found'})
                file=web/names[path]; self.send(200,file.read_bytes(),mimetypes.guess_type(file.name)[0] or 'text/plain')
            except KeyError as exc: self.send(404,{'error':str(exc)})
            except Exception as exc: self.send(400,{'error':f'{type(exc).__name__}: {exc}'})
        def do_POST(self):
            try:
                if not self.local(): return self.send(403,{'error':'loopback host and same origin required'})
                if self.headers.get('Content-Type','').split(';')[0]!='application/json': return self.send(415,{'error':'application/json required'})
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=MAX_BODY: return self.send(413,{'error':'body must be between 1 byte and 1 MiB'})
                data=json.loads(self.rfile.read(size),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
                if not isinstance(data,dict): raise ValueError('JSON body must be an object')
                self.send(200,app.post(urlparse(self.path).path,data))
            except KeyError as exc: self.send(404,{'error':str(exc)})
            except Exception as exc: self.send(400,{'error':f'{type(exc).__name__}: {exc}'})
    server=ThreadingHTTPServer((host,port),Handler); server.workbench=app; server.daemon_threads=True
    return server

def serve(host='127.0.0.1',port=8765,**kwargs):
    server=create_server(host,port,**kwargs)
    print(f'SYNAPSE workbench: http://{host}:{server.server_port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
