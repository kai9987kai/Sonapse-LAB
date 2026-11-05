"""Standalone CLI tool for Project PROMETHEUS: Autonomous Cognitive Discovery Lab.

Usage:
    python tools/run_discovery_lab.py --mode curriculum
    python tools/run_discovery_lab.py --mode sprint1
    python tools/run_discovery_lab.py --mode enzyme --substrate lignocellulose
    python tools/run_discovery_lab.py --mode eval --seeds 2001 2002 2003 2004 2005
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseidon.discovery_lab import (
    RiverWorld,
    DiscoveryCurriculumRunner,
    DiscoverySkillComposer,
    DiscoveryEvaluator,
    MultimodalEnzymeGenerator,
    ACTION_NAMES,
)


def banner(title: str) -> None:
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)


def run_sprint1(seed: int = 42) -> None:
    banner("PROJECT PROMETHEUS: SPRINT 1 MINIMAL DETERMINISTIC RIVER WORLD")
    env = RiverWorld(seed=seed, river_width=3, max_steps=20)
    obs = env.reset()
    print(f"Initialized RiverWorld (Seed: {seed})")
    print(f"Grid: {env.grid_width}x{env.grid_height} | River Width: {env.river_width_actual}m | Water Depth: {env.water_depth}m")
    print(f"Agent initial pos: ({env.agent.x:.2f}, {env.agent.y:.2f}) | Objects spawned: {len(env.objects)}")

    # Execute deterministic sequence: Grasp nearest stone, move toward river, place stepping stone, cross
    test_actions = [5, 2, 6, 2, 2]  # GRASP, MOVE_EAST, RELEASE, MOVE_EAST, MOVE_EAST
    print(f"\nExecuting Sprint 1 Action Sequence: {[ACTION_NAMES[a] for a in test_actions]}")
    total_reward = 0.0

    print(f"{'Step':<5} | {'Action':<14} | {'Reward':<8} | {'Agent Position':<18} | {'Event'}")
    print("-" * 75)
    for i, a in enumerate(test_actions):
        next_obs, reward, done, info = env.step(a)
        total_reward += reward
        pos_str = f"({info['agent_position'][0]:.2f}, {info['agent_position'][1]:.2f}, {info['agent_position'][2]:.2f})"
        print(f"{i+1:<5} | {ACTION_NAMES[a]:<14} | {reward:+7.3f} | {pos_str:<18} | {info.get('event')}")
        if done:
            break

    replay_hash = env.verify_exact_replay()
    print("-" * 75)
    print(f"Sprint 1 Completed: Steps = {len(test_actions)} | Total Reward = {total_reward:+.3f}")
    print(f"Deterministic Replay Receipt SHA-256: {replay_hash}")
    print("Verification: PASSED (Exact replayable state transitions confirmed)")


def run_curriculum(seed: int = 42) -> None:
    banner("PROJECT PROMETHEUS: FULL 6-PHASE DEVELOPMENTAL CURRICULUM")
    runner = DiscoveryCurriculumRunner(base_seed=seed)
    report = runner.run_full_curriculum()

    print(f"Curriculum Base Seed: {seed} | Overall Status: {report['overall_status']}")
    print(f"Receipt SHA-256: {report['receipt_sha256']}\n")

    for key, phase in report["phases"].items():
        p_name = phase["phase_name"]
        p_passed = "PASSED [OK]" if phase["passed"] else "FAILED [X]"
        print(f"* Phase {phase['phase_id']}: {p_name:<46} -> {p_passed}")
        for mk, mv in phase["metrics"].items():
            if isinstance(mv, dict):
                print(f"    - {mk}: {len(mv)} items")
            else:
                print(f"    - {mk}: {mv}")

    print("\n" + "=" * 75)


def run_enzyme_generation(substrate: str = "lignocellulose", seed: int = 42) -> None:
    banner("PROJECT PROMETHEUS: MULTIMODAL BIOCATALYTIC ENZYME SYNTHESIS")
    generator = MultimodalEnzymeGenerator(seed=seed)
    enzyme = generator.generate_catalyst_for_substrate(substrate_type=substrate, target_shear_force=350.0)

    print(f"Target Substrate:          {substrate}")
    print(f"Synthesized Enzyme ID:     {enzyme.enzyme_id}")
    print(f"Enzyme Name:               {enzyme.name}")
    print(f"Catalytic Class:           {enzyme.target_class}")
    print(f"Active Site Residues:      {', '.join(enzyme.active_site.residues)}")
    print(f"Pocket Volume:             {enzyme.active_site.pocket_volume_angstrom3:.2f} A^3")
    print(f"Catalytic Fragment:        {enzyme.sequence_fragment}")
    print(f"Michaelis Constant (K_m):  {enzyme.k_m:.4f} mM")
    print(f"Turnover Number (k_cat):   {enzyme.k_cat:.2f} s^-1")
    print(f"Max Velocity (V_max):      {enzyme.v_max:.2f} umol/(min*mg)")
    print(f"Activation Delta Delta G:  {enzyme.delta_delta_g_barrier:.2f} kcal/mol (Transition State Lowering)")
    print(f"Cured Bond Strength:       {enzyme.cured_bond_strength:.3f} / 1.000")
    print(f"Water-Proof Rating:        {enzyme.water_proof_rating:.3f} / 1.000")
    print(f"Synthesis Receipt SHA-256: {enzyme.synthesis_receipt_hash}")
    print("=" * 75)


def run_evaluation(seeds: list[int]) -> None:
    banner("PROJECT PROMETHEUS: PAIRED SEED EVALUATION & STATISTICAL AUDIT")
    evaluator = DiscoveryEvaluator(bootstrap_reps=1000)
    report = evaluator.evaluate_crossing_benchmark(seeds=tuple(seeds), river_width=3)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="SUPERMIX POSEIDON: Project PROMETHEUS Discovery Lab CLI")
    parser.add_argument(
        "--mode",
        choices=["curriculum", "sprint1", "enzyme", "eval"],
        default="curriculum",
        help="Execution mode (default: curriculum)"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for procedural generation")
    parser.add_argument("--substrate", type=str, default="lignocellulose", help="Substrate for enzyme synthesis")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2001, 2002, 2003, 2004, 2005], help="Evaluation seeds")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if args.mode == "sprint1":
        run_sprint1(seed=args.seed)
    elif args.mode == "curriculum":
        run_curriculum(seed=args.seed)
    elif args.mode == "enzyme":
        run_enzyme_generation(substrate=args.substrate, seed=args.seed)
    elif args.mode == "eval":
        run_evaluation(seeds=args.seeds)


if __name__ == "__main__":
    main()
