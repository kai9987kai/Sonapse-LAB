import argparse
import json
from .runtime import Poseidon

def main():
    p = argparse.ArgumentParser(description="Supermix Poseidon & Beyond local research system")
    p.add_argument("command", choices=[
        "serve", "chat", "math", "image", "video", "mesh", "world", "status",
        "beyond-benchmark", "beyond-circuits", "beyond-memory", "beyond-moe", "beyond-scene",
        "atlas-fit", "experiment", "verify-experiment", "contrast-fit", "contrast-experiment", "verify-contrast",
        "horizon-fit", "horizon-experiment", "verify-horizon",
        "odyssey-fit", "odyssey-experiment", "verify-odyssey",
        "verify-evidence", "audit-trajectory", "verify-trajectory",
        "fit-helm", "helm-experiment", "verify-helm",
        "fit-odysseus", "odysseus-experiment", "verify-odysseus",
        "mco-experiment", "verify-mco",
        "aura-experiment", "verify-aura",
        "tessera-experiment", "verify-tessera",
        "mnemorph-experiment", "verify-mnemorph",
        "metamorph-experiment", "verify-metamorph",
        "chimera-experiment", "verify-chimera",
        "hyperion-experiment", "verify-hyperion",
        "titan-experiment", "verify-titan",
        "trident-experiment", "verify-trident",
        "cognitive-experiment", "verify-cognitive",
        "oracle-fit", "oracle-evaluate", "oracle-screen", "verify-oracle-fit", "verify-oracle-evaluation",
        "hyper-experiment", "world-3d",
        "ecosystem-experiment", "verify-ecosystem", "world-ecosystem",
        "poseidon-x-experiment", "verify-poseidon-x", "world-multivector",
        "nexus-omega-assay", "verify-nexus-omega",
        "nexus-omega-discovery", "verify-nexus-omega-discovery",
        "verify-3d-replay",
        "discovery-lab",
        "reason", "reasoning-audit", "verify-reasoning-audit",
        "metamorphic-audit", "verify-metamorphic-audit", "episode-risk", "verify-episode-risk",
        "problem",
        "pantheon",
        "creative",
        "hypothesis",
        "proof",
        "invent",
        "athenaeum",
        "hephaestus",
        "hyperborea",
        "neurograft",
        "aether",
        "chronos",
        "noether",
        "cartan",
        "tether",
        "faultline",
        "crosswire",
        "poincare",
    ])
    p.add_argument("prompt", nargs="?", default="")
    p.add_argument("--root", default=".")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--scarcity", type=float, default=None)
    p.add_argument("--adapter")
    p.add_argument("--oracle-checkpoint", help="explicit trained research ORACLE for cognitive/hyper TidePool control")
    p.add_argument("--planner", choices=["policy", "mpc", "hybrid", "risk_aware", "uncertainty", "atlas", "contrast", "horizon", "odyssey", "helm", "odysseus", "aura", "metamorph", "chimera", "hyperion", "titan", "trident", "cognitive", "hyper", "poseidon_x", "nexus_omega", "pantheon"], default="policy", help="world simulation planner")
    p.add_argument("--use-pomdp", action="store_true", help="use partially observable POMDP TidePool environment")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--anchors", type=int, default=24)
    p.add_argument("--train-episodes", type=int, default=12)
    p.add_argument("--selection-episodes", type=int, default=8)
    p.add_argument("--calibration-episodes", type=int, default=None)
    p.add_argument("--parent", help="parent experiment receipt for verify-trajectory")
    p.add_argument("--stride", type=int, default=32, help="regular trajectory-audit anchor interval")
    p.add_argument("--horizon", type=int, default=16, help="trajectory-audit return horizon")
    p.add_argument("--max-anchors", type=int, default=256, help="trajectory-audit anchor budget")
    p.add_argument("--output", help="optional output directory for ORACLE fit/evaluation")
    p.add_argument("--samples", type=int, default=8, help="generated checks per reasoning family (1-32)")
    p.add_argument("--oracle-train-episodes", type=int, default=32)
    p.add_argument("--oracle-selection-episodes", type=int, default=8)
    p.add_argument("--oracle-calibration-episodes", type=int, default=12)
    p.add_argument("--oracle-updates", type=int, default=256)
    p.add_argument("--oracle-batch-size", type=int, default=128)
    p.add_argument("--oracle-hidden-size", type=int, default=128)
    p.add_argument("--oracle-ensemble-size", type=int, default=5)
    p.add_argument("--oracle-coverage", type=float, default=0.90)
    p.add_argument("--oracle-data-protocol", choices=["legacy-v1", "episode-mixture-v2"], default="episode-mixture-v2")
    p.add_argument("--oracle-architecture", choices=["mlp-v1", "linear-residual-v1"], default="mlp-v1",
                   help="opt-in trained state predictor architecture for oracle-fit")
    p.add_argument("--eval-episodes", type=int, default=32, help="fresh episodes per ORACLE evaluation condition")
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--skip-replay", action="store_true", help="verify receipt hashes without regenerating episodes")
    p.add_argument("--max-new-tokens", type=int, default=None, help="maximum new generation tokens for language model")
    args = p.parse_args()
    if args.command in ("episode-risk", "verify-episode-risk"):
        from .research.episode_risk import (
            audit_episode_risk, read_episode_risk_json, verify_episode_risk_report,
            write_episode_risk_report,
        )
        try:
            if not args.prompt:
                p.error(f"{args.command} requires a JSON path")
            if args.skip_replay:
                p.error("Episode risk reports always recompute bounds; --skip-replay is unsupported")
            data = read_episode_risk_json(args.prompt)
            if args.command == "verify-episode-risk":
                report = verify_episode_risk_report(data)
            else:
                if not isinstance(data, dict) or set(data) != {"protocol", "outcomes_by_condition"}:
                    p.error("Provide protocol and outcomes_by_condition only")
                report = audit_episode_risk(**data)
                if args.output:
                    write_episode_risk_report(args.output, report)
            print(json.dumps(report, indent=2, allow_nan=False))
            if (args.command == "verify-episode-risk" and not report["valid"]) or (
                    args.command == "episode-risk" and not report["eligible"]):
                raise SystemExit(1)
        except (ValueError, OSError, KeyError, TypeError) as error:
            p.error(str(error))
        return
    if args.command in ("reasoning-audit", "verify-reasoning-audit", "metamorphic-audit", "verify-metamorphic-audit"):
        from pathlib import Path
        metamorphic = "metamorphic" in args.command
        if metamorphic:
            from .research.metamorphic_audit import run_audit, verify_audit
        else:
            from .research.reasoning_audit import run_audit, verify_audit
        try:
            if args.command.startswith("verify-"):
                if not args.prompt:
                    p.error(f"{args.command} requires a report JSON path")
                target = Path(args.prompt)
                if target.stat().st_size > 2_000_000:
                    p.error("reasoning audit report exceeds 2 MB")
                report = json.loads(target.read_text(encoding="utf-8"))
                print(json.dumps(verify_audit(report, replay=not args.skip_replay), indent=2))
            else:
                report = run_audit(seed=args.seed if args.seed is not None else (20261031 if metamorphic else 20261030), samples=args.samples)
                if args.output:
                    target = Path(args.output)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                print(json.dumps(report, indent=2, allow_nan=False))
                metrics = report["metrics"]
                failed = (metrics["wrong"] or metrics["errors"] or
                          (metrics["refusals"] or metrics["relation_violations"] if metamorphic
                           else metrics["missed_answerable"]))
                if failed:
                    raise SystemExit(1)
        except (ValueError, OSError, KeyError, TypeError) as error:
            p.error(str(error))
        return
    if args.oracle_architecture != "mlp-v1" and args.command != "oracle-fit":
        p.error("--oracle-architecture is only supported by oracle-fit")
    if args.command == "oracle-screen" and args.skip_replay:
        p.error("oracle-screen requires full replay; --skip-replay is not supported")
    if args.oracle_checkpoint and args.command not in ("world", "status", "cognitive-experiment", "hyper-experiment", "world-3d"):
        p.error("--oracle-checkpoint is supported by world, status, and cognitive-experiment (TidePool only)")
    if args.oracle_checkpoint and args.command == "world" and args.planner not in ("cognitive", "hyper"):
        p.error("--oracle-checkpoint requires --planner cognitive or hyper")
    default_seeds = {"atlas-fit": 81000001, "experiment": 93000001,
                     "contrast-fit": 101000001, "contrast-experiment": 104000001,
                     "horizon-fit": 107000001, "horizon-experiment": 108000001,
                     "odyssey-fit": 109000001, "odyssey-experiment": 110000001,
                     "fit-helm": 120000001, "helm-experiment": 123000001,
                     "fit-odysseus": 131000001, "odysseus-experiment": 133000001,
                     "mco-experiment": 140000001, "aura-experiment": 144000001,
                     "tessera-experiment": 155000001, "mnemorph-experiment": 160000001,
                     "metamorph-experiment": 199000001, "chimera-experiment": 202000001,
                     "hyperion-experiment": 303000001, "titan-experiment": 99000000,
                     "trident-experiment": 88000000, "cognitive-experiment": 90000000,
                     "oracle-fit": 442000000, "oracle-evaluate": 442000000,
                     "hyper-experiment": 90000001, "world-3d": 42}
    args.seed = args.seed if args.seed is not None else default_seeds.get(args.command, 42)
    args.episodes = args.episodes if args.episodes is not None else (4 if args.command in ("experiment", "contrast-experiment", "horizon-experiment", "odyssey-experiment", "helm-experiment", "odysseus-experiment", "aura-experiment", "metamorph-experiment", "chimera-experiment", "hyperion-experiment", "titan-experiment", "trident-experiment", "cognitive-experiment", "hyper-experiment") else 100)
    args.scarcity = args.scarcity if args.scarcity is not None else (1.0 if args.command == "world" else 2.5)
    args.calibration_episodes = args.calibration_episodes if args.calibration_episodes is not None else (8 if args.command in ("contrast-fit", "horizon-fit", "odyssey-fit", "fit-helm", "fit-odysseus") else 6)

    if args.command == "oracle-fit":
        from pathlib import Path
        from .research.oracle_training import fit_oracle
        root = Path(args.root).resolve()
        output = Path(args.output) if args.output else root / "outputs" / f"oracle-fit-{args.seed}"
        if not output.is_absolute():
            output = root / output
        def progress(event):
            import sys
            print(json.dumps(event, allow_nan=False), file=sys.stderr, flush=True)
        receipt = fit_oracle(
            output,
            base_seed=args.seed,
            train_episodes=args.oracle_train_episodes,
            selection_episodes=args.oracle_selection_episodes,
            calibration_episodes=args.oracle_calibration_episodes,
            max_steps=args.max_steps if args.max_steps is not None else 96,
            training_updates=args.oracle_updates,
            batch_size=args.oracle_batch_size,
            hidden_size=args.oracle_hidden_size,
            ensemble_size=args.oracle_ensemble_size,
            coverage=args.oracle_coverage,
            data_protocol=args.oracle_data_protocol,
            architecture=args.oracle_architecture,
            progress=progress,
        )
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return

    if args.command == "oracle-evaluate":
        from pathlib import Path
        from .research.candidate_evaluation import evaluate_oracle
        root = Path(args.root).resolve()
        if not args.prompt:
            p.error("oracle-evaluate requires the ORACLE artifact path")
        artifact = Path(args.prompt)
        if not artifact.is_absolute():
            artifact = root / artifact
        output = Path(args.output) if args.output else root / "outputs" / f"oracle-eval-{args.seed}"
        if not output.is_absolute():
            output = root / output
        receipt = evaluate_oracle(artifact, output, base_seed=args.seed,
                                  episodes_per_condition=args.eval_episodes,
                                  bootstrap_reps=args.bootstrap_reps)
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return

    if args.command == "oracle-screen":
        from pathlib import Path
        from .research.prediction_gate import assess_prediction_gate
        if not args.prompt:
            p.error("oracle-screen requires an evaluation receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        print(json.dumps(assess_prediction_gate(path), indent=2, ensure_ascii=False))
        return

    if args.command == "verify-oracle-fit":
        from pathlib import Path
        from .research.oracle_training import verify_fit_receipt
        if not args.prompt:
            p.error("verify-oracle-fit requires a fit receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        print(json.dumps(verify_fit_receipt(path, replay_data=not args.skip_replay), indent=2))
        return

    if args.command == "verify-oracle-evaluation":
        from pathlib import Path
        from .research.candidate_evaluation import verify_evaluation_receipt
        if not args.prompt:
            p.error("verify-oracle-evaluation requires an evaluation receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        print(json.dumps(verify_evaluation_receipt(path, replay=not args.skip_replay), indent=2))
        return

    if args.command == "verify-helm":
        from pathlib import Path
        from .helm import stable_json
        from .helm_experiment import verify_helm_receipt
        if not args.prompt:
            p.error("verify-helm requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        print(json.dumps(verify_helm_receipt(stable_json(path, max_bytes=128 * 1024 * 1024)), indent=2))
        return

    if args.command == "fit-helm":
        from .helm import HelmCritic
        from pathlib import Path
        from .provenance import assert_source_current
        import os
        import tempfile
        if any(not 2 <= value <= 64 for value in (args.train_episodes, args.selection_episodes, args.calibration_episodes)) or not 0 <= args.seed <= 2**63 - 2000100:
            p.error("fit-helm requires 2–64 episodes per partition and disjoint seed namespaces")
        runtime = Poseidon(args.root)
        sources = assert_source_current()
        def progress(event):
            assert_source_current()
            if event.get("phase") and event.get("episode") is not None:
                import sys
                print(json.dumps(event), file=sys.stderr, flush=True)
        critic, receipt = HelmCritic.fit(runtime.core(),
            train_seeds=list(range(args.seed, args.seed + args.train_episodes)),
            selection_seeds=list(range(args.seed + 1000000, args.seed + 1000000 + args.selection_episodes)),
            calibration_seeds=list(range(args.seed + 2000000, args.seed + 2000000 + args.calibration_episodes)),
            anchors_per_episode=args.anchors, max_steps=args.max_steps if args.max_steps is not None else 96, progress=progress)
        assert_source_current()
        directory = runtime.root / "outputs/helm"
        critic.save(directory / "atlas.json")
        descriptor, temporary = tempfile.mkstemp(prefix="fit-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2, allow_nan=False)
            os.replace(temporary, directory / "fit_receipt.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(json.dumps(receipt, indent=2))
        return

    if args.command in ("verify-evidence", "audit-trajectory", "verify-trajectory"):
        from pathlib import Path
        from .trajectory_audit import load_json, load_artifacts, verify_parent, collect, verify_audit, save_audit
        if not args.prompt:
            p.error(f"{args.command} requires an evidence path")
        root = Path(args.root)
        def resolve(value):
            path = Path(value)
            return path if path.is_absolute() else root / path
        evidence = load_json(resolve(args.prompt))
        artifacts = load_artifacts(root)
        if args.command == "verify-evidence":
            result = verify_parent(evidence, artifacts)
        elif args.command == "verify-trajectory":
            if not args.parent:
                p.error("verify-trajectory requires --parent with its experiment receipt")
            result = verify_audit(evidence, load_json(resolve(args.parent)), artifacts)
        else:
            bundle = collect(Poseidon(root).core(), evidence, artifacts,
                             stride=args.stride, horizon=args.horizon, max_anchors=args.max_anchors)
            verification = verify_audit(bundle, evidence, artifacts)
            path = save_audit(bundle, root / "outputs/trajectory_audits")
            result = {"path": str(path), "experiment_id": bundle["experiment_id"],
                      "summary": bundle["summary"], "parent_verification": bundle["parent_verification"],
                      "verification": verification, "limits": bundle["limits"]}
        print(json.dumps(result, indent=2))
        return

    if args.command in ("verify-experiment", "verify-contrast", "verify-horizon", "verify-odyssey"):
        from pathlib import Path
        if args.command == "verify-odyssey":
            from .odyssey_experiments import load_and_verify
        elif args.command == "verify-horizon":
            from .horizon_experiments import load_and_verify
        elif args.command == "verify-contrast":
            from .contrast_experiments import load_and_verify
        else:
            from .experiments import load_and_verify
        if not args.prompt:
            p.error(f"{args.command} requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        print(json.dumps(load_and_verify(path), indent=2))
        return

    if args.command == "contrast-fit":
        from .contrast import ContrastAtlas
        from pathlib import Path
        import os
        import tempfile
        if any(not 1 <= value <= 64 for value in (args.train_episodes, args.selection_episodes, args.calibration_episodes)) or not 0 <= args.seed <= 2**63 - 2000100:
            p.error("contrast-fit needs 1–64 episodes per partition and room for disjoint selection/calibration seeds")
        runtime = Poseidon(args.root)
        atlas, receipt = ContrastAtlas.fit(runtime.core(),
            train_seeds=list(range(args.seed, args.seed + args.train_episodes)),
            selection_seeds=list(range(args.seed + 1000000, args.seed + 1000000 + args.selection_episodes)),
            calibration_seeds=list(range(args.seed + 2000000, args.seed + 2000000 + args.calibration_episodes)),
            anchors_per_episode=args.anchors, max_steps=args.max_steps if args.max_steps is not None else 128)
        directory = Path(args.root) / "outputs/contrast"
        directory.mkdir(parents=True, exist_ok=True)
        atlas.save(directory / "atlas.json")
        descriptor, temporary = tempfile.mkstemp(prefix="fit-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2, allow_nan=False)
            os.replace(temporary, directory / "fit_receipt.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(json.dumps({key: value for key, value in receipt.items() if key != "rows" and not key.endswith("_rows")}, indent=2))
        return

    if args.command == "atlas-fit":
        from .atlas import CounterfactualAtlas
        from pathlib import Path
        import os
        import tempfile
        if not 1 <= args.train_episodes <= 64 or not 1 <= args.calibration_episodes <= 64 or not 0 <= args.seed <= 2**63 - 1000100:
            p.error("atlas-fit needs 1–64 episodes per partition and a seed with room for a disjoint calibration namespace")
        runtime = Poseidon(args.root)
        atlas, receipt = CounterfactualAtlas.fit(runtime.core(), train_seeds=list(range(args.seed, args.seed + args.train_episodes)),
            calibration_seeds=list(range(args.seed + 1000000, args.seed + 1000000 + args.calibration_episodes)),
            anchors_per_episode=args.anchors, max_steps=args.max_steps if args.max_steps is not None else 128)
        directory = runtime.root / "outputs/atlas"
        atlas.save(directory / "atlas.json")
        fd, temporary = tempfile.mkstemp(prefix="fit-", suffix=".tmp", dir=directory)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
        os.replace(temporary, directory / "fit_receipt.json")
        print(json.dumps({key: value for key, value in receipt.items() if key != "calibration_rows"}, indent=2))
        return

    if args.command == "horizon-fit":
        from .horizon import HorizonAtlas
        from pathlib import Path
        import os
        import tempfile
        if any(not 1 <= value <= 64 for value in (args.train_episodes, args.calibration_episodes)) or not 0 <= args.seed <= 2**63 - 2000100:
            p.error("horizon-fit needs 1–64 episodes per partition and room for disjoint calibration seeds")
        runtime = Poseidon(args.root)
        atlas, receipt = HorizonAtlas.fit(runtime.core(),
            train_seeds=list(range(args.seed, args.seed + args.train_episodes)),
            calibration_seeds=list(range(args.seed + 1000000, args.seed + 1000000 + args.calibration_episodes)),
            anchors_per_episode=args.anchors, max_steps=args.max_steps if args.max_steps is not None else 128)
        directory = Path(args.root) / "outputs/horizon"
        directory.mkdir(parents=True, exist_ok=True)
        atlas.save(directory / "atlas.json")
        descriptor, temporary = tempfile.mkstemp(prefix="fit-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2, allow_nan=False)
            os.replace(temporary, directory / "fit_receipt.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(json.dumps({key: value for key, value in receipt.items() if key != "calibration_anchor_rows"}, indent=2))
        return

    if args.command == "odyssey-fit":
        from .odyssey import OdysseyAtlas, OdysseyConfig
        from pathlib import Path
        import os
        import tempfile
        if any(not 1 <= value <= 64 for value in (args.train_episodes, args.calibration_episodes)) or not 0 <= args.seed <= 2**63 - 2000100:
            p.error("odyssey-fit needs 1–64 episodes per partition and room for disjoint calibration seeds")
        runtime = Poseidon(args.root)
        config = OdysseyConfig(scarcity=args.scarcity)
        atlas, receipt = OdysseyAtlas.fit(runtime.core(),
            train_seeds=list(range(args.seed, args.seed + args.train_episodes)),
            calibration_seeds=list(range(args.seed + 1000000, args.seed + 1000000 + args.calibration_episodes)),
            anchors_per_episode=args.anchors, max_steps=args.max_steps if args.max_steps is not None else 128,
            config=config)
        directory = Path(args.root) / "outputs/odyssey"
        directory.mkdir(parents=True, exist_ok=True)
        atlas.save(directory / "atlas.json")
        descriptor, temporary = tempfile.mkstemp(prefix="fit-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2, allow_nan=False)
            os.replace(temporary, directory / "fit_receipt.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(json.dumps({key: value for key, value in receipt.items() if key != "calibration_anchor_rows"}, indent=2))
        return

    if args.command == "verify-mco":
        from pathlib import Path
        from .mco import verify_mco_receipt
        target = Path(args.prompt) if args.prompt else (Path(args.root) / "outputs/mco_experiments/RECEIPT.json")
        if not target.is_absolute():
            target = Path(args.root) / target
        data = json.loads(target.read_text(encoding="utf-8"))
        print(json.dumps(verify_mco_receipt(data), indent=2))
        return

    if args.command == "mco-experiment":
        runtime = Poseidon(args.root)
        result = runtime.mco_experiment()
        print(json.dumps(result, indent=2))
        return

    if args.command == "verify-odysseus":
        from pathlib import Path
        from .odysseus_experiment import verify_odysseus_receipt
        if not args.prompt:
            p.error("verify-odysseus requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(verify_odysseus_receipt(data), indent=2))
        return

    if args.command == "fit-odysseus":
        from .odysseus import fit_odysseus
        from pathlib import Path
        if any(not 1 <= value <= 64 for value in (args.train_episodes, args.calibration_episodes)) or not 0 <= args.seed <= 2**63 - 2000100:
            p.error("fit-odysseus needs 1–64 episodes per partition and room for disjoint calibration seeds")
        runtime = Poseidon(args.root)
        train_seeds = list(range(args.seed, args.seed + args.train_episodes))
        cal_seeds = list(range(args.seed + 1000000, args.seed + 1000000 + args.calibration_episodes))
        atlas, receipt = fit_odysseus(
            runtime.core(),
            train_seeds=train_seeds,
            calibration_seeds=cal_seeds,
            max_steps=args.max_steps if args.max_steps is not None else 64,
            scarcity=args.scarcity,
        )
        directory = Path(args.root) / "outputs/odysseus"
        directory.mkdir(parents=True, exist_ok=True)
        atlas.save(directory / "atlas.json")
        (directory / "fit_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return

    if args.command == "verify-aura":
        from pathlib import Path
        from .aura_experiment import verify_aura_receipt
        if not args.prompt:
            p.error("verify-aura requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(verify_aura_receipt(data), indent=2))
        return

    if args.command == "tessera-experiment":
        runtime = Poseidon(args.root)
        result = runtime.tessera_experiment()
        print(json.dumps(result, indent=2))
        return

    if args.command == "verify-tessera":
        from pathlib import Path
        from .tessera_experiment import verify_tessera_receipt
        if not args.prompt:
            p.error("verify-tessera requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(verify_tessera_receipt(data), indent=2))
        return

    if args.command == "mnemorph-experiment":
        runtime = Poseidon(args.root)
        result = runtime.mnemorph_experiment()
        print(json.dumps(result, indent=2))
        return

    if args.command == "verify-mnemorph":
        from pathlib import Path
        from .mnemorph import verify_mnemorph_receipt
        if not args.prompt:
            p.error("verify-mnemorph requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(verify_mnemorph_receipt(data), indent=2))
        return

    if args.command == "verify-metamorph":
        from pathlib import Path
        from .metamorph_experiment import verify_metamorph_receipt
        if not args.prompt:
            p.error("verify-metamorph requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(verify_metamorph_receipt(data), indent=2))
        return

    if args.command == "verify-chimera":
        from pathlib import Path
        from .chimera_experiment import verify_chimera_receipt
        if not args.prompt:
            p.error("verify-chimera requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(verify_chimera_receipt(data), indent=2))
        return

    if args.command == "verify-hyperion":
        from pathlib import Path
        from .hyperion_experiment import verify_hyperion_receipt
        if not args.prompt:
            p.error("verify-hyperion requires a receipt path")
        path = Path(args.prompt)
        if not path.is_absolute():
            path = Path(args.root) / path
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(verify_hyperion_receipt(data), indent=2))
        return

    if args.command == "verify-titan":
        from pathlib import Path
        from .titan_experiment import verify_titan_receipt
        target = Path(args.prompt) if args.prompt else (Path(args.root) / "outputs/titan_experiments/RECEIPT.json")
        if not target.is_absolute():
            target = Path(args.root) / target
        data = json.loads(target.read_text(encoding="utf-8"))
        print(json.dumps(verify_titan_receipt(data), indent=2))
        return

    if args.command == "verify-trident":
        from pathlib import Path
        from .trident_experiment import verify_trident_receipt
        target = Path(args.prompt) if args.prompt else (Path(args.root) / "outputs/trident_experiments/RECEIPT.json")
        if not target.is_absolute():
            target = Path(args.root) / target
        data = json.loads(target.read_text(encoding="utf-8"))
        print(json.dumps(verify_trident_receipt(data), indent=2))
        return

    if args.command == "verify-cognitive":
        from pathlib import Path
        from .cognitive_experiment import verify_cognitive_receipt
        target = Path(args.prompt) if args.prompt else (Path(args.root) / "outputs/cognitive_experiments/RECEIPT.json")
        if not target.is_absolute():
            target = Path(args.root) / target
        data = json.loads(target.read_text(encoding="utf-8"))
        print(json.dumps({"verified": verify_cognitive_receipt(data), "sha256": data.get("sha256")}, indent=2))
        return

    if args.command == "world-3d":
        from .genesis_3d import ContinuousTidePool3D
        runtime = Poseidon(args.root, oracle_checkpoint=args.oracle_checkpoint)
        agent = runtime.hyper_agent()
        env = ContinuousTidePool3D(seed=args.seed)
        result = agent.rollout_3d(env, max_steps=args.max_steps if args.max_steps is not None else 64)
        print(json.dumps({k: v for k, v in result.items() if k != "decisions"}, indent=2))
        return

    if args.command == "world-ecosystem":
        from .ecosystem_3d import ContinuousEcosystem3D
        runtime = Poseidon(args.root)
        eco = runtime.ecosystem(seed=args.seed or 42)
        eco.spawn_default_colony(n_grazers=2, n_predators=1, n_symbionts=1)
        actions = {"grazer_0": 4, "grazer_1": 1, "predator_0": 4, "symbiont_0": 0}
        max_steps = args.max_steps if args.max_steps is not None else 32
        for _ in range(max_steps):
            obs_dict, rewards, dones, telemetry = eco.step(actions)
        print(json.dumps({
            "status": "success",
            "telemetry": telemetry,
            "population_alive": {aid: a.alive for aid, a in eco.agents.items()},
        }, indent=2))
        return

    if args.command == "ecosystem-experiment":
        runtime = Poseidon(args.root)
        episodes = args.episodes if args.episodes is not None else 2
        max_steps = args.max_steps if args.max_steps is not None else 32
        result = runtime.ecosystem_experiment(seed=args.seed or 42, episodes=episodes, max_steps=max_steps)
        print(json.dumps(result, indent=2))
        return

    if args.command == "verify-ecosystem":
        from .ecosystem_experiment import verify_ecosystem_receipt
        from pathlib import Path
        if not args.prompt:
            p.error("verify-ecosystem requires a receipt path")
        target = Path(args.prompt)
        data = json.loads(target.read_text(encoding="utf-8"))
        print(json.dumps({"verified": verify_ecosystem_receipt(data), "sha256": data.get("receipt_sha256")}, indent=2))
        return

    if args.command == "world-multivector":
        from .geometric_algebra import GeometricKinematicsEngine, Multivector3D
        engine = GeometricKinematicsEngine(dt=0.2)
        vorticity = Multivector3D.from_bivector(b12=0.4, b23=0.1, b31=0.2)
        steps = args.max_steps if args.max_steps is not None else 20
        for _ in range(steps):
            engine.apply_thrust_and_torque(thrust_forward=6.0, torque_yaw_b12=0.1, fluid_vorticity=vorticity)
        print(json.dumps({
            "status": "success",
            "position": engine.pos.to_vector_tuple(),
            "velocity": engine.vel.to_vector_tuple(),
            "rotor": [engine.rotor.mv.s, engine.rotor.mv.b12, engine.rotor.mv.b23, engine.rotor.mv.b31],
            "vorticity": [vorticity.b12, vorticity.b23, vorticity.b31],
        }, indent=2))
        return

    if args.command == "poseidon-x-experiment":
        runtime = Poseidon(args.root)
        episodes = args.episodes if args.episodes is not None else 2
        max_steps = args.max_steps if args.max_steps is not None else 32
        result = runtime.poseidon_x_experiment(seed=args.seed or 101, episodes=episodes, max_steps=max_steps)
        print(json.dumps(result, indent=2))
        return

    if args.command == "verify-poseidon-x":
        from .poseidon_x_experiment import verify_poseidon_x_receipt
        from pathlib import Path
        if not args.prompt:
            p.error("verify-poseidon-x requires a receipt path")
        target = Path(args.prompt)
        data = json.loads(target.read_text(encoding="utf-8"))
        print(json.dumps({"verified": verify_poseidon_x_receipt(data), "sha256": data.get("receipt_sha256")}, indent=2))
        return

    if args.command in ("experiment", "contrast-experiment", "horizon-experiment", "odyssey-experiment", "helm-experiment", "odysseus-experiment", "aura-experiment", "metamorph-experiment", "chimera-experiment", "hyperion-experiment", "titan-experiment", "trident-experiment", "cognitive-experiment", "hyper-experiment"):
        runtime = Poseidon(args.root, oracle_checkpoint=args.oracle_checkpoint)
        if args.command == "hyper-experiment":
            result = runtime.hyper_experiment(args.seed, args.episodes, args.max_steps if args.max_steps is not None else 64, args.scarcity)
            print(json.dumps(result, indent=2))
            return
        elif args.command == "cognitive-experiment":
            result = runtime.cognitive_experiment(args.seed, args.episodes, args.max_steps if args.max_steps is not None else 32, args.scarcity, use_pomdp=args.use_pomdp)
            print(json.dumps(result, indent=2))
            return
        elif args.command == "hyperion-experiment":
            experiment = runtime.hyperion_experiment
        elif args.command == "trident-experiment":
            experiment = runtime.trident_experiment
        elif args.command == "titan-experiment":
            experiment = runtime.titan_experiment
        elif args.command == "chimera-experiment":
            experiment = runtime.chimera_experiment
        elif args.command == "metamorph-experiment":
            experiment = runtime.metamorph_experiment
        elif args.command == "aura-experiment":
            experiment = runtime.aura_experiment
        elif args.command == "odysseus-experiment":
            experiment = runtime.odysseus_experiment
        elif args.command == "helm-experiment":
            experiment = runtime.helm_experiment
        elif args.command == "odyssey-experiment":
            experiment = runtime.odyssey_experiment
        elif args.command == "horizon-experiment":
            experiment = runtime.horizon_experiment
        elif args.command == "contrast-experiment":
            experiment = runtime.contrast_experiment
        else:
            experiment = runtime.experiment
        result = experiment(args.seed, args.episodes, args.max_steps if args.max_steps is not None else 64, args.scarcity)
        print(json.dumps({key: value for key, value in result.items() if key != "replay"}, indent=2))
        return

    if args.command == "serve":
        from .server import serve
        serve(args.root, args.port, args.adapter)
        return

    if args.command == "beyond-benchmark":
        from .beyond_benchmark import run_paired_benchmark
        results = run_paired_benchmark(
            core_path="runs/tidal_dagger/core.pt",
            n_episodes=args.episodes,
            base_seed=args.seed,
            scarcity=args.scarcity
        )
        output = {name: res.__dict__ for name, res in results.items()}
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if args.command == "beyond-circuits":
        from .connectome import compare_circuit_topologies
        results = compare_circuit_topologies(seed=args.seed)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if args.command == "beyond-memory":
        from .persistent_memory import run_delayed_recall_benchmark
        results = run_delayed_recall_benchmark(seed=args.seed, include_receipts=True)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if args.command == "beyond-moe":
        from .sparse_moe import benchmark_sparse_vs_dense
        results = benchmark_sparse_vs_dense()
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if args.command == "beyond-scene":
        from .scene_graph import create_procedural_beyond_world
        world = create_procedural_beyond_world()
        obj_file = world.export_wavefront_obj("outputs/media/beyond_world.obj")
        world.step_simulation(dt=0.2)
        world.apply_agent_impulse("agent", "explore")
        world.step_simulation(dt=0.2)
        print(json.dumps({
            "exported_obj": obj_file,
            "scene": world.to_dict()
        }, indent=2, ensure_ascii=False))
        return

    if args.command == "nexus-omega-assay":
        from .nexus_omega import IndependentScientificAuditor, HypothesisSpec
        auditor = IndependentScientificAuditor(max_steps=args.max_steps or 32)
        hypothesis = HypothesisSpec(
            hypothesis_id="hyp_nexus_omega_persistence",
            statement="Causal memory fabric significantly improves POMDP survival over ablated memory",
            null_hypothesis="There is no difference in reward between intact and ablated memory",
            independent_variable="Working memory presence (intact vs ablated)",
            dependent_variable="Episode total reward",
            alpha=0.05,
            minimum_effect_size=0.01,
        )
        receipt = auditor.run_causal_memory_assay(hypothesis)
        four_fold = auditor.run_four_fold_memory_test(seed=args.seed or 42)
        output = {
            "receipt": receipt.__dict__,
            "four_fold_verification": four_fold,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if args.command == "verify-nexus-omega":
        from .nexus_omega import IndependentScientificAuditor, HypothesisSpec
        auditor = IndependentScientificAuditor(max_steps=16)
        hypothesis = HypothesisSpec(
            hypothesis_id="hyp_verification_check",
            statement="Memory verification check",
            null_hypothesis="No difference",
            independent_variable="Ablation",
            dependent_variable="Reward",
        )
        receipt = auditor.run_causal_memory_assay(hypothesis)
        print(json.dumps({
            "status": "verified",
            "receipt_hash": receipt.receipt_hash,
            "verdict": receipt.verdict,
            "retained": receipt.retained_in_library,
        }, indent=2, ensure_ascii=False))
        return

    if args.command == "nexus-omega-discovery":
        from .nexus_omega import AutonomousDiscoveryEngine
        engine = AutonomousDiscoveryEngine(max_steps=args.max_steps or 32)
        ledger = engine.run_discovery_cycle(
            seed_base=args.seed or 50000,
            num_seeds=args.episodes or 8,
        )
        output = {
            "ledger_id": ledger.ledger_id,
            "timestamp": ledger.timestamp,
            "ratified_laws": ledger.ratified_laws,
            "refutation_graveyard": ledger.refutation_graveyard,
            "ledger_sha256": ledger.ledger_sha256,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if args.command == "verify-nexus-omega-discovery":
        from pathlib import Path
        from .nexus_omega import AutonomousDiscoveryEngine, verify_discovery_receipt
        if args.prompt:
            p_path = Path(args.prompt)
            data = json.loads(p_path.read_text(encoding="utf-8"))
        else:
            engine = AutonomousDiscoveryEngine(max_steps=16)
            ledger = engine.run_discovery_cycle(seed_base=args.seed or 50000, num_seeds=args.episodes or 4)
            data = {
                "ledger_id": ledger.ledger_id,
                "timestamp": ledger.timestamp,
                "ratified_laws": ledger.ratified_laws,
                "refutation_graveyard": ledger.refutation_graveyard,
                "ledger_sha256": ledger.ledger_sha256,
            }
        is_valid = verify_discovery_receipt(data)
        print(json.dumps({
            "status": "verified" if is_valid else "rejected",
            "ledger_sha256": data["ledger_sha256"],
            "ratified_laws_count": len(data["ratified_laws"]),
            "refutations_count": len(data["refutation_graveyard"]),
        }, indent=2, ensure_ascii=False))
        return

    if args.command == "verify-3d-replay":
        from pathlib import Path
        from .external_adapter import Compact3DReplayVerifier
        if not args.prompt:
            p.error("verify-3d-replay requires path to replay json file")
        p_path = Path(args.prompt).resolve()
        receipt = Compact3DReplayVerifier.verify_replay_file(p_path)
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return

    if args.command == "discovery-lab":
        from .discovery_lab import (
            RiverWorld,
            DiscoveryCurriculumRunner,
            MultimodalEnzymeGenerator,
            DiscoveryEvaluator,
        )
        mode = args.prompt or "curriculum"
        seed = args.seed if args.seed is not None else 42
        if mode == "curriculum":
            runner = DiscoveryCurriculumRunner(base_seed=seed)
            print(json.dumps(runner.run_full_curriculum(), indent=2, ensure_ascii=False))
        elif mode == "enzyme":
            gen = MultimodalEnzymeGenerator(seed=seed)
            enzyme = gen.generate_catalyst_for_substrate(substrate_type="lignocellulose")
            print(json.dumps(enzyme.to_dict(), indent=2, ensure_ascii=False))
        elif mode == "eval":
            evaluator = DiscoveryEvaluator(bootstrap_reps=1000)
            report = evaluator.evaluate_crossing_benchmark(seeds=(2001, 2002, 2003, 2004, 2005), river_width=3)
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            env = RiverWorld(seed=seed, river_width=3)
            obs, r, done, info = env.step(0)
            print(json.dumps({"reward": r, "done": done, "info": info, "alive": env.agent.alive}, indent=2, ensure_ascii=False))
        return

    runtime = Poseidon(args.root, args.adapter, oracle_checkpoint=args.oracle_checkpoint)
    if args.command == "status":
        result = runtime.status()
    else:
        result = runtime.respond(args.prompt or "Survive", args.command, seed=args.seed, planner=args.planner,
                                 scarcity=args.scarcity, max_steps=args.max_steps if args.max_steps is not None else 256,
                                 max_new_tokens=args.max_new_tokens)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
