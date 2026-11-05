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
    p=sub.add_parser('autoprogram'); p.add_argument('--diagnostic-episodes',type=int,default=3); p.add_argument('--eval-episodes',type=int,default=3); p.add_argument('--min-gain',type=float,default=-0.05); p.add_argument('--save')
    p=sub.add_parser('transplant'); p.add_argument('--donor-task',default='forage'); p.add_argument('--test-task',default='delayed_match'); p.add_argument('--episodes',type=int,default=3)
    sub.add_parser('metamorphic')
    sub.add_parser('spectral')
    p=sub.add_parser('hyperprogram'); p.add_argument('--target',default='motor.weight'); p.add_argument('--edits',type=int,default=16)
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
            elif args.command=='autoprogram':
                from .auto_programmer import execute_closed_loop_repair
                result=execute_closed_loop_repair(runtime.agent,task=args.task,num_diagnostic_episodes=args.diagnostic_episodes,num_eval_episodes=args.eval_episodes,min_gain=args.min_gain,seed=args.seed)
                if args.save: result={'report':result,'checkpoint':runtime.save(safe_name(args.save))}
            elif args.command=='transplant':
                from .transplantation import extract_donor_subcircuit,run_transplantation_experiment
                donor=extract_donor_subcircuit(runtime.agent,source_task=args.donor_task)
                result=run_transplantation_experiment(runtime.agent,donor=donor,test_task=args.test_task,episodes=args.episodes,seed=args.seed)
            elif args.command=='metamorphic':
                from .metamorphic import run_metamorphic_verification_suite
                result=run_metamorphic_verification_suite(runtime.agent)
            elif args.command=='spectral':
                from .spectral import compute_connectome_harmonics,project_spectral_activity
                harmonics=compute_connectome_harmonics(runtime.agent.graph,num_modes=8)
                proj=project_spectral_activity(runtime.agent.brain.r,harmonics)
                result={'eigenvalues':harmonics['eigenvalues'],'projection':proj}
            elif args.command=='hyperprogram':
                from .hypernetwork import WeightSpaceHypernetwork,compile_hypernetwork_patch
                hypernet=WeightSpaceHypernetwork()
                diag={'mean_return':0.2,'total_steps':30,'mean_graph_activity':float(runtime.agent.brain.r.mean()),'mean_approach_valence':0.1,'mean_avoid_valence':-0.1,'action_mean':runtime.last_action.tolist(),'task':runtime.config.task}
                result=compile_hypernetwork_patch(runtime.agent,hypernet,diag,target_surface=args.target,max_edits=args.edits)
        print(json.dumps(result,indent=2,allow_nan=False,default=str)); return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        parser.exit(2,f'Error: {exc}\n')

if __name__=='__main__': raise SystemExit(main())
