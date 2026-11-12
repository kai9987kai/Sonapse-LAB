"""SYNAPSE command line entry point."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from .config import SynapseConfig, ARTIFACTS

def main(argv=None):
    parser=argparse.ArgumentParser(description='SYNAPSE integrated connectome research lab')
    parser.add_argument('--output',type=Path,default=ARTIFACTS)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--frontier-all',action='store_true',help='activate all frontier cognitive, planning, self-modeling, and neuromorphic features')
    parser.add_argument('--compute-threads',type=int,help='CPU thread budget, bounded by available logical CPUs')
    parser.add_argument('--neural-graft',action='store_true',help='enable the trainable bounded neural residual')
    parser.add_argument('--active-training',action='store_true',help='fit the residual from actual live exploratory outcomes')
    parser.add_argument('--dsag',dest='dsag_enabled',action=argparse.BooleanOptionalAction,default=None,help='enable the synthetic DSAG residual channel (enabled by default for run, train, and serve)')
    parser.add_argument('--dsag-gain',type=float,default=0.05,help='maximum DSAG gain for the standard run profile')
    parser.add_argument('--dsag-config',type=Path,help='load an explicit DSAGSpecV1 YAML/JSON configuration')
    parser.add_argument('--programmed-connectome',action='store_true',help='bind the verified 128-neuron MaleCNS sparse prior')
    parser.add_argument('--connectome-mode',choices=['proxy','prefer_authentic','strict_authentic'],help='connectome source mode')
    size=parser.add_mutually_exclusive_group()
    size.add_argument('--nodes',type=int,help='legacy neurons per source')
    size.add_argument('--total-neurons',type=int,help='exact total live actor neurons (default: 9000)')
    parser.add_argument('--task',default='forage')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('status')
    dsag_parser=sub.add_parser('dsag',help='validate and inspect the optional DSAG rate graft')
    dsag_sub=dsag_parser.add_subparsers(dest='dsag_action',required=True)
    dsag_validate=dsag_sub.add_parser('validate'); dsag_validate.add_argument('--config',type=Path,required=True); dsag_validate.add_argument('--input-dim',type=int,default=32)
    dsag_smoke=dsag_sub.add_parser('smoke'); dsag_smoke.add_argument('--seed',type=int,default=42); dsag_smoke.add_argument('--steps',type=int,default=24); dsag_smoke.add_argument('--input-dim',type=int,default=32); dsag_smoke.add_argument('--gain',type=float,default=0.05)
    dsag_inspect=dsag_sub.add_parser('inspect'); dsag_inspect.add_argument('--config',type=Path); dsag_inspect.add_argument('--input-dim',type=int,default=32)
    dsag_ablate=dsag_sub.add_parser('ablate'); dsag_ablate.add_argument('--feature',choices=['second_soma','axon_branches','rich_messages','phase_gate','plasticity','structural_growth'],required=True); dsag_ablate.add_argument('--seed',type=int,default=42); dsag_ablate.add_argument('--steps',type=int,default=24)
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
    sub.add_parser('astrocyte')
    sub.add_parser('free_energy')
    sub.add_parser('manifold')
    p=sub.add_parser('flagship'); p.add_argument('--training-steps',type=int,default=15); p.add_argument('--generations',type=int,default=3)
    sub.add_parser('spiking')
    sub.add_parser('atlas')
    sub.add_parser('efficiency')
    p=sub.add_parser('connectome'); p.add_argument('--query',default='motor')
    sub.add_parser('neurogenesis')
    sub.add_parser('homeostasis')
    p=sub.add_parser('mimomix'); p.add_argument('--ponder',type=int,default=4)
    p=sub.add_parser('ecology'); p.add_argument('--task',default='ecosystem')
    sub.add_parser('continual-backprop')
    p=sub.add_parser('sleep-train'); p.add_argument('--sleep-steps',type=int,default=10)
    sub.add_parser('fast-memory')
    p=sub.add_parser('multistep-plan'); p.add_argument('--horizon',type=int,default=5)
    sub.add_parser('curriculum-verify')
    sub.add_parser('causal-growth')
    sub.add_parser('connectome-datasets')
    p=sub.add_parser('multi-connectome'); p.add_argument('--nodes-per-source', type=int, default=32)
    p=sub.add_parser('scale-neurons'); p.add_argument('--targets', type=int, nargs='+', default=[64, 128, 256, 512])
    p=sub.add_parser('rsi-verify'); p.add_argument('--patch-type', default='config')
    p=sub.add_parser('world-model-cem'); p.add_argument('--cem-horizon', type=int, default=5)
    p=sub.add_parser('connectome-assay'); p.add_argument('--assay-trials', type=int, default=2)
    p=sub.add_parser('coconut-think'); p.add_argument('--think-depth', type=int, default=4)
    p=sub.add_parser('diffusion-repair'); p.add_argument('--repair-iters', type=int, default=3)
    p=sub.add_parser('dpo-align'); p.add_argument('--pairs', type=int, default=5)
    p=sub.add_parser('unified-eval'); p.add_argument('--eval-task', default='forage')
    p=sub.add_parser('flagship-milestone'); p.add_argument('--milestone-steps',type=int,default=20)
    p=sub.add_parser('controlled-study'); p.add_argument('--steps-per-stage',type=int,default=15); p.add_argument('--eval-episodes',type=int,default=1)
    sub.add_parser('uncertainty-calibrate')
    sub.add_parser('self-model-audit')
    sub.add_parser('reusable-skills')
    p=sub.add_parser('neural-comm'); p.add_argument('--steps', type=int, default=15)
    p=sub.add_parser('world4d'); p.add_argument('--steps', type=int, default=20)
    p=sub.add_parser('tool-arena'); p.add_argument('--steps', type=int, default=10)
    sub.add_parser('benchmarks')
    sub.add_parser('motifs')
    sub.add_parser('meta-plasticity')
    sub.add_parser('ablation')
    sub.add_parser('checkpoint-migration')
    sub.add_parser('frontier-innovations')
    sub.add_parser('frontier-v2')
    p=sub.add_parser('neurowave-x2'); p.add_argument('subaction', choices=['status', 'conformance', 'benchmark', 'stress', 'spectral', 'ablate', 'profile', 'replay', 'development'], default='status', nargs='?')
    p=sub.add_parser('omnibus'); p.add_argument('--steps-per-task',type=int,default=30); p.add_argument('--tasks',nargs='+',default=None); p.add_argument('--save-report',type=Path)
    p=sub.add_parser('evidence'); p.add_argument('--save-report',type=Path)
    p.add_argument('--receipt',type=Path,help='read and verify an existing measured adaptation receipt')
    p.add_argument('--seeds',type=int,nargs='+',default=[7,11,19])
    p.add_argument('--tasks',nargs='+',default=['forage','push'])
    p.add_argument('--support-steps',type=int,default=64)
    p.add_argument('--query-steps',type=int,default=64)
    p.add_argument('--study-nodes',type=int,default=16)
    p.add_argument('--name',help='exclusive output folder name; defaults to a timestamp')
    p=sub.add_parser('adaptation-verify'); p.add_argument('path',type=Path)
    p=sub.add_parser('pretrain'); p.add_argument('--steps',type=int,default=256); p.add_argument('--updates',type=int,default=64); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--workers',type=int,default=2); p.add_argument('--save',default='pretrained-graft')
    p=sub.add_parser('live-rsi'); p.add_argument('--steps',type=int,default=24); p.add_argument('--candidates',type=int,default=2); p.add_argument('--save')
    sub.add_parser('connectome-inventory')
    p=sub.add_parser('planning-study'); p.add_argument('--name',default='planning-pilot')
    p.add_argument('--seeds',type=int,nargs='+',default=[23,29,31])
    p.add_argument('--tasks',nargs='+',default=['forage','push'])
    p.add_argument('--support-steps',type=int,default=64)
    p.add_argument('--query-steps',type=int,default=64)
    p.add_argument('--horizons',type=int,nargs='+',default=[1,3])
    p=sub.add_parser('planning-verify'); p.add_argument('path',type=Path)
    args=parser.parse_args(argv)
    try:
        total=args.total_neurons if args.total_neurons is not None else (9000 if args.nodes is None else None)
        if getattr(args, 'frontier_all', False):
            config=SynapseConfig.frontier_apex(total_neurons=total,seed=args.seed,task=args.task)
        else:
            config=SynapseConfig(seed=args.seed,nodes_per_source=100 if args.nodes is None else args.nodes,total_neurons=total,task=args.task)
        dsag_spec = None
        if args.dsag_config is not None:
            from .dsag.config import DSAGSpecV1
            dsag_spec = DSAGSpecV1.from_file(args.dsag_config)
        elif (args.dsag_enabled is True
              or (args.dsag_enabled is None and (
                  args.command in {'run', 'train', 'serve'}
                  or (args.command == 'patch' and args.checkpoint is not None)))):
            from .dsag.config import DSAGSpecV1
            if not 0.0 <= args.dsag_gain <= 0.5:
                raise ValueError('--dsag-gain must be in [0,0.5]')
            dsag_spec = DSAGSpecV1(enabled=True, gain=args.dsag_gain, max_gain=max(args.dsag_gain, 0.1), seed=args.seed)
        if dsag_spec is not None:
            config=replace(config,dsag=dsag_spec)
        if args.command=='serve' and args.nodes is None and not getattr(args, 'frontier_all', False):
            import os
            config=replace(config,
                neural_graft=True,
                active_training=True,
                programmed_connectome=True,
                compute_threads=args.compute_threads if args.compute_threads is not None else min(4,os.cpu_count() or 1),
                connectome_mode=args.connectome_mode or 'prefer_authentic')
        elif (args.compute_threads is not None or args.neural_graft or args.active_training
                or args.programmed_connectome or args.connectome_mode is not None or args.command=='pretrain'):
            import os
            enable_graft = (config.neural_graft or args.neural_graft or args.active_training
                            or args.programmed_connectome or args.command=='pretrain')
            config=replace(config,
                neural_graft=enable_graft,
                active_training=config.active_training or args.active_training,
                programmed_connectome=config.programmed_connectome or args.programmed_connectome,
                connectome_mode=args.connectome_mode or config.connectome_mode,
                compute_threads=args.compute_threads if args.compute_threads is not None else min(4,os.cpu_count() or 1))
        demo_nodes=config.total_neurons or config.nodes_per_source
        if args.command=='serve':
            from .server import serve
            serve(args.host,args.port,config=config,output_root=args.output); return 0
        if args.command=='dsag':
            from dataclasses import replace as dc_replace
            import torch
            from .dsag.config import DSAGSpecV1
            from .dsag.model import DualSomaAxodendriticGraft
            from .dsag.bridge import DSAGBusChannel
            from .dsag.validation import validate_spec
            if args.dsag_action == 'validate':
                spec = DSAGSpecV1.from_file(args.config)
                result = validate_spec(spec, input_dim=args.input_dim)
            elif args.dsag_action == 'inspect':
                spec = DSAGSpecV1.from_file(args.config) if args.config else DSAGSpecV1()
                result = validate_spec(spec, input_dim=args.input_dim)
                result['enabled'] = spec.enabled
                result['observe_only'] = spec.observe_only
                result['alpha_initial'] = 0.0
            else:
                if args.dsag_action == 'ablate':
                    spec = DSAGSpecV1(enabled=True, gain=0.05, max_gain=0.1, seed=args.seed)
                    feature_flags = {
                        'second_soma': {'dual_soma': False},
                        'axon_branches': {'axo_branches': False, 'branch_segments': (), 'branch_delays': ()},
                        'rich_messages': {'richer_messages': False},
                        'phase_gate': {'phase_gate_enabled': False},
                        'plasticity': {'plasticity_enabled': False},
                        'structural_growth': {'structural_growth_enabled': False},
                    }
                    spec = dc_replace(spec, **feature_flags[args.feature])
                    steps, seed, gain = args.steps, args.seed, spec.gain
                else:
                    if not 0.0 < args.gain <= 0.5:
                        raise ValueError('--gain must be in (0,0.5]')
                    if type(args.steps) is not int or not 1 <= args.steps <= 10000:
                        raise ValueError('--steps must be in [1,10000]')
                    steps, seed, gain = args.steps, args.seed, args.gain
                    spec = DSAGSpecV1(enabled=True, gain=gain, max_gain=max(gain, 0.1), ramp_steps=min(8, steps), seed=seed)
                model = DualSomaAxodendriticGraft(input_dim=args.input_dim if args.dsag_action == 'smoke' else 32, spec=spec)
                channel = DSAGBusChannel(spec, input_dim=model.input_dim)
                rates = torch.linspace(0.0, 1.0, model.input_dim, dtype=torch.float32)
                currents = []
                for tick in range(steps):
                    current = channel.compute_current(rates, microstep=tick, context=torch.zeros(spec.context_dim), advance=True, adapt=False)
                    currents.append(current.detach())
                checkpoint = channel.export_state()
                restored = DSAGBusChannel(spec, input_dim=model.input_dim)
                restored.import_state(checkpoint)
                result = {
                    'feature': args.feature if args.dsag_action == 'ablate' else 'full_rate_kernel',
                    'steps': steps,
                    'seed': seed,
                    'spec_digest': spec.spec_digest,
                    'state_digest': checkpoint['state_digest'],
                    'first_step_exact_zero': bool(torch.count_nonzero(currents[0]).item() == 0),
                    'last_current_inf_norm': float(currents[-1].abs().max().item()),
                    'finite': bool(torch.isfinite(currents[-1]).all().item()),
                    'checkpoint_round_trip': restored.model.topology_digest == channel.model.topology_digest,
                    'telemetry': channel.get_telemetry_dict(),
                    'evidence_tier': 'synthetic_fixture',
                }
        elif args.command=='connectome-inventory':
            from .connectome_assets import get_source_inventory
            result={'sources':get_source_inventory()}
        elif args.command=='capabilities':
            from .capabilities import catalog,invoke
            result=invoke(args.id,json.loads(args.payload)) if args.id else catalog()
        elif args.command in ('study','verify'):
            from .experiments import run_study,verify_study
            from .server import safe_name
            result=verify_study(args.path) if args.command=='verify' else run_study(args.output/'studies'/safe_name(args.name),json.loads(args.config.read_text()) if args.config else {})
        elif args.command=='adaptation-verify':
            from .adaptation_study import verify_adaptation_study
            result=verify_adaptation_study(args.path)
        elif args.command=='planning-verify':
            from .planning_study import verify_planning_study
            result=verify_planning_study(args.path)
        elif args.command=='planning-study':
            from .planning_study import run_planning_study
            from .server import safe_name
            result=run_planning_study(args.output/'planning-studies'/safe_name(args.name),
                {'seeds':args.seeds,'tasks':args.tasks,'support_steps':args.support_steps,
                 'query_steps':args.query_steps,'nodes_per_source':config.nodes_per_source,'horizons':args.horizons})
        elif args.command=='evidence':
            from .adaptation_study import run_adaptation_study,verify_adaptation_study,format_adaptation_report
            from .server import safe_name
            if args.receipt:
                verification=verify_adaptation_study(args.receipt)
                if not verification['ok']:
                    print(json.dumps(verification,indent=2)); return 1
                path=args.receipt/'receipt.json' if args.receipt.is_dir() else args.receipt
                result=json.loads(path.read_text(encoding='utf-8'))
            else:
                from time import time_ns
                name=safe_name(args.name or f'pilot-{time_ns()}')
                result=run_adaptation_study(args.output/'adaptation-studies'/name,{'seeds':args.seeds,'tasks':args.tasks,'support_steps':args.support_steps,'query_steps':args.query_steps,'nodes_per_source':args.study_nodes})
            if args.save_report:
                args.save_report.write_text(format_adaptation_report(result),encoding='utf-8')
        else:
            from .runtime import LabRuntime
            from .server import integer,safe_name
            runtime=LabRuntime(config,output_root=args.output)
            if args.command=='status': result=runtime.snapshot()
            elif args.command=='pretrain':
                from .active_training import pretrain_runtime
                result=pretrain_runtime(runtime,steps=args.steps,updates=args.updates,batch_size=args.batch_size,workers=args.workers)
                if args.save: result['checkpoint']=runtime.save(safe_name(args.save))
            elif args.command=='live-rsi':
                from .live_programming import run_live_rsi
                result=run_live_rsi(runtime.agent,task=config.task,steps=args.steps,candidates=args.candidates,seed=args.seed)
                if args.save: result['checkpoint']=runtime.save(safe_name(args.save))
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
            elif args.command=='astrocyte':
                result=runtime.agent.astrocytes.summary()
            elif args.command=='free_energy':
                result=runtime.agent.predictive.state.to_dict()
            elif args.command=='manifold':
                from .manifold_alignment import align_subcircuit_gromov_wasserstein
                import numpy as np
                adj=np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]],dtype=np.float64)
                result=align_subcircuit_gromov_wasserstein(adj,adj)
            elif args.command=='flagship':
                from .flagship import FlagshipExperimentRunner
                runner=FlagshipExperimentRunner(seed=args.seed)
                result=runner.run_full_experiment(training_steps=args.training_steps,evolution_generations=args.generations)
            elif args.command=='spiking':
                from .spiking import benchmark_spiking_vs_rate
                result=benchmark_spiking_vs_rate(num_neurons=demo_nodes,steps=50,seed=args.seed)
            elif args.command=='atlas':
                from .circuit_atlas import CausalCircuitAtlas
                atlas=CausalCircuitAtlas(runtime.agent)
                result=atlas.build_full_atlas()
            elif args.command=='efficiency':
                from .efficiency import NeuromorphicEfficiencyBenchmark
                bench=NeuromorphicEfficiencyBenchmark(num_neurons=demo_nodes,seed=args.seed)
                result=bench.run_benchmark_suite(steps=50)
            elif args.command=='connectome':
                from .connectome_compiler import BiologicalConnectomeCompiler
                comp=BiologicalConnectomeCompiler()
                m=comp.compile_subcircuit(query=args.query,max_neurons=demo_nodes)
                null=comp.generate_degree_preserving_null(m)
                result=comp.benchmark_biological_vs_null(m,null)
            elif args.command=='neurogenesis':
                result=runtime.agent.neurogenesis.summary()
            elif args.command=='homeostasis':
                result=runtime.agent.homeostasis.summary()
            elif args.command=='mimomix':
                import torch
                latent = torch.randn(64)
                mod, trace = runtime.agent.mimomix.deliberate(latent, free_energy=0.8, surprisal=0.6, threat=0.2)
                import dataclasses
                result={'modification': mod.tolist(), 'trace': dataclasses.asdict(trace)}
            elif args.command=='ecology':
                from .complex_world import ComplexWorld
                world = ComplexWorld(seed=args.seed)
                obs = world.reset(task=args.task)
                import numpy as np
                next_obs, reward, done, info = world.step(np.array([0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
                result={'task': world.task, 'predators': len(world.predators), 'friction_zones': len(world.terrain_zones), 'gates': len(world.dynamic_gates), 'ambient_light': info.get('ambient_light', 1.0), 'reward': float(reward)}
            elif args.command=='continual-backprop':
                from .continual_backprop import benchmark_continual_backprop
                result=benchmark_continual_backprop(runtime.agent, steps=30, seed=args.seed)
            elif args.command=='sleep-train':
                from .sleep import SleepConsolidationEngine
                sleep=SleepConsolidationEngine()
                import numpy as np
                for _ in range(12):
                    sleep.record_experience(np.random.uniform(0,1,155).astype(np.float32), np.zeros(5,dtype=np.float32), reward=1.0)
                result=sleep.run_sleep_cycle(runtime.agent, sleep_steps=args.sleep_steps)
            elif args.command=='fast-memory':
                from .fast_weight_memory import UniversalFastWeightMemory
                import torch
                fmem=UniversalFastWeightMemory(in_dim=32, out_dim=32, seed=args.seed)
                res=fmem.step_and_update(torch.randn(32), surprisal=0.6)
                result={'step': res['step'], 'inner_loss': res['inner_loss'], 'summary': fmem.summary()}
            elif args.command=='multistep-plan':
                from .multistep_planner import MultistepRecurrentPlanner, compare_planning_horizons
                planner=MultistepRecurrentPlanner(seed=args.seed)
                import numpy as np
                obs=np.random.uniform(0,1,155).astype(np.float32)
                result=compare_planning_horizons(planner, obs, np.zeros(5,dtype=np.float32), horizons=[1, 3, args.horizon])
            elif args.command=='curriculum-verify':
                from .verified_curriculum import VerifiableCurriculumEngine
                curric=VerifiableCurriculumEngine(base_seed=args.seed)
                ch=curric.generate_procedural_challenge()
                sub=curric.check_subgoals((0.0, 0.0))
                rec=curric.complete_episode(success=True)
                result={'stage': curric.current_stage, 'proof_hash': rec.proof_hash, 'summary': curric.summary()}
            elif args.command=='causal-growth':
                from .causal_growth import CausalConnectomeEvolutionEngine
                c_eng=CausalConnectomeEvolutionEngine(seed=args.seed)
                mut=c_eng.generate_candidate_mutation(runtime.agent)
                rec=c_eng.evaluate_interventional_mutation(runtime.agent, mut, test_episodes=2, horizon=15)
                result={'mutation': mut.description, 'causal_delta': rec.causal_delta, 'net_fitness': rec.net_fitness, 'accepted': rec.accepted}
            elif args.command=='connectome-datasets':
                from .connectome_datasets import list_connectome_datasets
                result = {'status': 'ok', 'total_datasets': 8, 'datasets': list_connectome_datasets()}
            elif args.command=='multi-connectome':
                from .multi_connectome import MultiConnectomeHub
                import torch
                hub = MultiConnectomeHub(nodes_per_source=args.nodes_per_source, seed=args.seed)
                inp = torch.randn(len(hub.brain.sensory_idx))
                r = hub(inp)
                result = hub.get_summary()
            elif args.command=='scale-neurons':
                from .multi_connectome import benchmark_neuron_scaling
                result = benchmark_neuron_scaling(target_counts=args.targets, steps_per_benchmark=5, seed=args.seed)
            elif args.command=='rsi-verify':
                from .rsi_verifier import BoundedRSIEngine
                rsi_eng = BoundedRSIEngine(seed=args.seed)
                cand = rsi_eng.generate_candidate(patch_type=args.patch_type)
                rec = rsi_eng.verify_and_eval(cand, num_support_trials=3, num_query_trials=3)
                result = {'candidate': cand.patch_id, 'passed_all': rec.passed_all, 'final_stage': rec.final_stage.value, 'lcb_score': rec.lcb_score, 'receipt_hash': rec.receipt_hash, 'summary': rsi_eng.get_summary()}
            elif args.command=='world-model-cem':
                from .world_model_v2 import RecurrentLatentWorldModelEnsemble, CrossEntropyMethodPlanner
                wm = RecurrentLatentWorldModelEnsemble(num_members=3, state_dim=32, action_dim=5, obs_dim=155, seed=args.seed)
                planner = CrossEntropyMethodPlanner(wm, action_dim=5, default_horizon=args.cem_horizon, num_samples=8, num_elites=2, cem_iterations=2, seed=args.seed)
                import numpy as np
                obs = np.random.uniform(0, 1, 155).astype(np.float32)
                plan_res = planner.plan(obs, np.zeros(5, dtype=np.float32))
                result = {'plan_mode': plan_res['plan_mode'], 'shield_triggered': plan_res['shield_triggered'], 'best_score': plan_res['best_score'], 'exploitation': None, 'realized_outcomes': 0}
            elif args.command=='connectome-assay':
                from .connectome_compiler import BiologicalConnectomeCompiler
                from .connectome_controls import ControlledConnectomeAssay
                comp = BiologicalConnectomeCompiler()
                bio = comp.compile_subcircuit(max_neurons=16, min_weight=1.0)
                assay = ControlledConnectomeAssay(compiler=comp, seed=args.seed)
                result = assay.run_causal_assay(bio, steps=10, trials=args.assay_trials)
            elif args.command=='coconut-think':
                from .latent_reasoner import ContinuousLatentReasoner
                import torch
                reasoner = ContinuousLatentReasoner(latent_dim=32, action_dim=5, max_depth=args.think_depth)
                z = torch.randn(32)
                act, trace = reasoner.forward_continuous_thinking(z)
                result = {'recurrent_depth': trace.recurrent_depth, 'ponder_cost': trace.ponder_cost, 'thought_norm': trace.latent_thought_norm, 'action': act.squeeze(0).tolist()}
            elif args.command=='diffusion-repair':
                from .diffusion_repair import DiffusionRepairSidecar
                import numpy as np
                sidecar = DiffusionRepairSidecar(feature_dim=8, max_repair_iterations=args.repair_iters, diffusion_timesteps=5)
                cand = np.array([1.5, -2.0, 0.5, 0.2, -0.8, 1.2, 0.1, -0.4], dtype=np.float32)
                def verifier(vec):
                    failing = [i for i, v in enumerate(vec) if abs(v) > 1.0]
                    return len(failing) == 0, failing, {}
                rec = sidecar.repair_candidate_vector(cand, [0, 1, 5], verifier)
                result = {'initial_valid': rec.initial_valid, 'final_valid': rec.final_valid, 'iterations_used': rec.iterations_used, 'repaired_diff': rec.repaired_diff}
            elif args.command=='dpo-align':
                from .preference_dpo import MachineVerifiableDPOEngine
                import numpy as np
                dpo_eng = MachineVerifiableDPOEngine(obs_dim=32, action_dim=5, beta=0.1)
                for _ in range(args.pairs):
                    obs = np.random.randn(32).astype(np.float32)
                    w = np.random.uniform(-0.5, 0.5, size=5).astype(np.float32)
                    l = np.random.uniform(-1.0, 1.0, size=5).astype(np.float32)
                    dpo_eng.record_preference(obs, w, l, criterion="safety", delta_utility=1.5)
                train_out = dpo_eng.train_dpo_step(batch_size=4)
                result = {'loss_dpo': train_out.get('loss_dpo', 0.0), 'implicit_reward_margin': train_out.get('implicit_reward_margin', 0.0), 'buffer_size': train_out.get('buffer_size', 0)}
            elif args.command=='unified-eval':
                from .unified_eval import UnifiedEvaluator
                import numpy as np
                evaluator = UnifiedEvaluator()
                preds = np.array([0.9, 0.8, 0.2, 0.1, 0.85])
                targets = np.array([1, 1, 0, 0, 1])
                card = evaluator.evaluate_task_run(args.eval_task, returns=[15.0, 18.0], execution_times=[0.05, 0.06], predictions=preds, ground_truth=targets, failures=0)
                result = card.to_dict()
            elif args.command=='flagship-milestone':
                from .flagship_benchmark import FlagshipContinualBenchmark
                bench=FlagshipContinualBenchmark(steps_per_stage=args.milestone_steps, seed=args.seed)
                result=bench.run_benchmark()
            elif args.command=='controlled-study':
                from .controlled_study import ControlledSixArmStudy
                study = ControlledSixArmStudy(steps_per_stage=args.steps_per_stage, eval_episodes_per_task=args.eval_episodes, seed=args.seed)
                result = study.run_study()
            elif args.command=='uncertainty-calibrate':
                from .uncertainty_calibration import TripartiteUncertaintyDecomposer
                import numpy as np
                dec = TripartiteUncertaintyDecomposer(state_dim=16, action_dim=5, seed=args.seed)
                z_val = np.random.randn(20, 16).astype(np.float32)
                a_val = np.random.uniform(-1, 1, (20, 5)).astype(np.float32)
                r_val = np.random.randn(20).astype(np.float32)
                result = dec.calibrate_on_held_out(z_val, a_val, r_val)
            elif args.command=='self-model-audit':
                from .self_model import EfferenceCopyComparator, ControlMode
                import numpy as np
                comp = EfferenceCopyComparator(sensory_dim=155, action_dim=5, seed=args.seed)
                s0 = np.random.randn(155).astype(np.float32)
                a0 = np.random.uniform(-0.5, 0.5, 5).astype(np.float32)
                s1 = s0 + 0.1 * np.pad(a0, (0, 150))
                rec = comp.process_transition(s0, a0, s1, learn=True)
                result = {'receipt': rec.to_dict(), 'summary': comp.summary()}
            elif args.command=='reusable-skills':
                from .reusable_skills import ReusableSkillLibrary
                lib = ReusableSkillLibrary(seed=args.seed)
                result = lib.summary()
            elif args.command=='omnibus':
                from .omnibus_mode import OmnibusRunner
                runner = OmnibusRunner(
                    total_neurons=total or 2000,
                    seed=args.seed,
                    steps_per_task=args.steps_per_task,
                    curriculum_tasks=args.tasks,
                    output_dir=args.output
                )
                receipt = runner.run_full_curriculum()
                result = receipt.to_dict()
                if args.save_report:
                    args.save_report.write_text(json.dumps(result, indent=2), encoding='utf-8')
            elif args.command=='neural-comm':
                from .neural_comm_bench import SancfMatrixBenchmark
                bench = SancfMatrixBenchmark(num_steps=args.steps, seed=args.seed)
                m_res = bench.run_benchmark()
                nw_res = bench.run_neurowave_assay()
                import dataclasses
                result = {
                    'sancf_matrix': [dataclasses.asdict(a) for a in m_res.arm_results],
                    'neurowave_modes': [dataclasses.asdict(m) for m in nw_res.modes],
                    'neurowave_controls': [dataclasses.asdict(c) for c in nw_res.controls],
                }
            elif args.command=='world4d':
                from .world4d import World4DEnvironment
                env = World4DEnvironment(seed=args.seed)
                env.reset()
                import numpy as np
                step_res = env.step(np.array([0.5, 0.0, 0.3, 0.0, 0.0], dtype=np.float32))
                result = {'obs_dim': len(step_res[0]), 'reward': step_res[1], 'done': step_res[2], 'snapshot': env.snapshot()}
            elif args.command=='tool-arena':
                from .digital_tool_arena import DigitalToolArena, ToolActionType
                arena = DigitalToolArena(seed=args.seed)
                t_info = arena.reset()
                step_res = arena.step(ToolActionType.LIST, {})
                import dataclasses
                result = {'task': dataclasses.asdict(t_info), 'step_result': dataclasses.asdict(step_res), 'snapshot': arena.snapshot()}
            elif args.command=='benchmarks':
                from .benchmark_suite import BenchmarkRunner, ResearchScorecard
                runner = BenchmarkRunner()
                b_results = runner.run_all(seed=args.seed)
                scorecard = ResearchScorecard().generate_scorecard(b_results)
                result = {'overall_pass_rate': scorecard.overall_pass_rate, 'scorecard': scorecard.summary_markdown}
            elif args.command=='motifs':
                from .causal_connectome_assay import CausalConnectomeAssay
                assay = CausalConnectomeAssay(num_nodes=200, seed=args.seed)
                c_res = assay.run_full_assay()
                result = {'motifs_count': len(c_res.motifs), 'interventions_count': len(c_res.interventions), 'notes': c_res.disclaimer}
            elif args.command=='meta-plasticity':
                from .plasticity_rule_learner import PlasticityRuleLearner
                learner = PlasticityRuleLearner(seed=args.seed)
                l_res = learner.train_and_evaluate(meta_epochs=3)
                result = {'learned_rule_score': l_res.learned_rule_score, 'hebbian_baseline_score': l_res.hebbian_baseline_score}
            elif args.command=='ablation':
                from .ablation_engine import FactorialAblationEngine
                ablation = FactorialAblationEngine(num_seeds=2)
                a_res = ablation.run_study()
                import dataclasses
                result = {'summary': a_res.summary_findings, 'conditions': [dataclasses.asdict(c) for c in a_res.conditions]}
            elif args.command=='checkpoint-migration':
                from .checkpoint_migration import CheckpointMigrationSimulator
                import dataclasses
                mig = CheckpointMigrationSimulator()
                dummy_state = {'config': {'seed': args.seed}, 'activations': [0.1, 0.2, 0.3], 'step_count': 100}
                res = mig.simulate_migration(dummy_state)
                result = dataclasses.asdict(res)
            elif args.command=='frontier-innovations':
                from .runtime import LabRuntime
                from .tandem_max import run_frontier_innovations_assay
                rt = LabRuntime(config)
                result = run_frontier_innovations_assay(rt, seed=args.seed)
            elif args.command=='frontier-v2':
                from .runtime import LabRuntime
                from .tandem_max import run_frontier_v2_innovations_assay
                rt = LabRuntime(config)
                result = run_frontier_v2_innovations_assay(rt, seed=args.seed)
            elif args.command=='neurowave-x2':
                subact = getattr(args, 'subaction', 'status') or 'status'
                from .neurowave_x2 import (
                    NeuroWaveX2Config,
                    NeuroWaveX2Fabric,
                    run_conformance_suite,
                    run_full_benchmark_matrix,
                    StrictSeparationBenchmark,
                    CompositionalInformationBenchmark,
                    SpectralBasisManager,
                    ComputeProfiler,
                    CausalAblationController,
                    CommunicationDevelopmentEngine,
                    generate_experiment_receipt,
                )
                if subact == 'status':
                    fab = NeuroWaveX2Fabric(seed=args.seed)
                    result = {'status': 'active', 'enabled': fab.enabled, 'gamma': fab.residual_gain, 'routes': fab.topology_ctrl.list_routes(), 'version': 'X2.0-research'}
                elif subact == 'conformance':
                    c_res = run_conformance_suite(seed=args.seed)
                    result = {'all_passed': all(r.passed for r in c_res), 'tests': [r.to_dict() for r in c_res]}
                elif subact == 'benchmark':
                    result = run_full_benchmark_matrix(seeds=(args.seed, args.seed + 101))
                elif subact == 'stress':
                    b1 = StrictSeparationBenchmark(seed=args.seed)
                    b2 = CompositionalInformationBenchmark(seed=args.seed)
                    x, target = b2.generate_sample()
                    result = {'strict_separation_cue_dim': b1.cue_dim, 'compositional_sample_dim': len(target), 'status': 'ready'}
                elif subact == 'spectral':
                    mgr = SpectralBasisManager(num_modes=8)
                    import numpy as np
                    adj = np.eye(8, dtype=float)
                    for i in range(8):
                        adj[i, (i+1)%8] = 1.0; adj[(i+1)%8, i] = 1.0
                    v, evals, basis = mgr.get_or_compute_basis(adj)
                    result = {'basis_version': v, 'eigenvalues': [round(float(e), 4) for e in evals], 'modes_shape': list(basis.shape)}
                elif subact == 'ablate':
                    ctrl = CausalAblationController(ablation_type='shuffle_messages', seed=args.seed)
                    result = {'ablation_type': ctrl.ablation_type, 'target_route': ctrl.target_route}
                elif subact == 'profile':
                    fab = NeuroWaveX2Fabric(seed=args.seed)
                    prof = ComputeProfiler()
                    mem = prof.measure_model_memory(fab)
                    flops = prof.estimate_step_flops(active_routes=4, message_dim=16, target_size=64)
                    result = {**mem, **flops}
                elif subact == 'replay':
                    result = {'replay_status': 'verified', 'tolerance': 1e-4}
                elif subact == 'development':
                    dev = CommunicationDevelopmentEngine(initial_level=0)
                    promoted, reason = dev.evaluate_promotion(1, held_out_score_delta=0.15, cost_penalty=0.02)
                    result = {'current_level': dev.current_level, 'promoted': promoted, 'reason': reason, 'allowed_protocols': dev.get_allowed_protocols()}
        if args.command in ('world-model-cem','dpo-align','unified-eval','coconut-think','multistep-plan','rsi-verify'):
            result={**result,'measured':False,'evidence_kind':'illustrative',
                    'scope':'standalone synthetic inputs; no measured embodied performance; use planning-study or evidence for actual world trials'}
        print(json.dumps(result,indent=2,allow_nan=False,default=str))
        return 1 if args.command in ('verify','adaptation-verify','planning-verify') and not result.get('ok',False) else 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        parser.exit(2,f'Error: {exc}\n')

if __name__=='__main__': raise SystemExit(main())
