"""SYNAPSE command line entry point."""
import argparse
import json
from pathlib import Path
from .config import SynapseConfig, ARTIFACTS

def main(argv=None):
    parser=argparse.ArgumentParser(description='SYNAPSE integrated connectome research lab')
    parser.add_argument('--output',type=Path,default=ARTIFACTS)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--nodes',type=int,default=100)
    parser.add_argument('--task',default='forage')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('status')
    for name in ('run','train'):
        p=sub.add_parser(name); p.add_argument('--steps',type=int,default=100); p.add_argument('--save')
    p=sub.add_parser('serve'); p.add_argument('--port',type=int,default=8765); p.add_argument('--host',default='127.0.0.1')
    p=sub.add_parser('study'); p.add_argument('--config',type=Path); p.add_argument('--name',default='cli-study')
    p=sub.add_parser('verify'); p.add_argument('path',type=Path)
    p=sub.add_parser('capabilities'); p.add_argument('--id'); p.add_argument('--payload',default='{}')
    p=sub.add_parser('patch'); p.add_argument('operation',choices=['make','apply']); p.add_argument('file',type=Path); p.add_argument('--checkpoint'); p.add_argument('--save',default='patched')
    args=parser.parse_args(argv)
    try:
        config=SynapseConfig(seed=args.seed,nodes_per_source=args.nodes,task=args.task)
        if args.command=='serve':
            from .server import serve
            serve(args.host,args.port,config=config,output_root=args.output); return 0
        if args.command=='capabilities':
            from .capabilities import catalog,invoke
            result=invoke(args.id,json.loads(args.payload)) if args.id else catalog()
        elif args.command in ('study','verify'):
            from .experiments import run_study,verify_study
            from .server import safe_name
            result=verify_study(args.path) if args.command=='verify' else run_study(args.output/'studies'/safe_name(args.name),json.loads(args.config.read_text()) if args.config else {})
        else:
            from .runtime import LabRuntime
            from .server import integer,safe_name
            runtime=LabRuntime(config,output_root=args.output)
            if args.command=='status': result=runtime.snapshot()
            elif args.command in ('run','train'):
                result=runtime.step(integer(args.steps,'steps',1,10000),learn=args.command=='train')
                if args.save: result={'state':result,'checkpoint':runtime.save(safe_name(args.save))}
            elif args.command=='patch':
                from .patches import make_patch,apply_patch
                if args.checkpoint: runtime.load(safe_name(args.checkpoint))
                payload=json.loads(args.file.read_text())
                if args.operation=='make': result=make_patch(runtime.agent,payload.get('edits',payload) if isinstance(payload,dict) else payload)
                else: result={'receipt':apply_patch(runtime.agent,payload),'checkpoint':runtime.save(safe_name(args.save))}
        print(json.dumps(result,indent=2,allow_nan=False,default=str)); return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        parser.exit(2,f'Error: {exc}\n')

if __name__=='__main__': raise SystemExit(main())
