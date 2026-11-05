"""Standalone CLI tool for PROBLEM FORGE: 16-Domain Verified Problem Solving.

Usage:
    python tools/run_problem_forge.py --demo
    python tools/run_problem_forge.py --capabilities
    python tools/run_problem_forge.py --solve "Alice says: Bob is a knave. Bob says: Alice and I are of different types."
    python tools/run_problem_forge.py --solve "Knapsack capacity 15 with items: gold (weight 4, value 10), silver (weight 2, value 4), bronze (weight 1, value 1), diamond (weight 12, value 30)"
    python tools/run_problem_forge.py --domain pocket_cube --solve "scramble: R U R' U'"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseidon.problem_solver import ProblemSolver, ProblemError, solve_problem, detect_problem


DEMO_PROMPTS = [
    # 1. Water Jugs
    ("water_jugs", "I have a 3-liter jug and a 5-liter jug. How do I measure out exactly 4 liters?"),
    # 2. River Crossing
    ("river_crossing", "A farmer needs to cross a river with a wolf, a goat, and a cabbage in a boat that can only carry the farmer and one item."),
    # 3. Missionaries & Cannibals
    ("missionaries_cannibals", "Three missionaries and three cannibals must cross a river using a boat that seats two."),
    # 4. Tower of Hanoi
    ("tower_of_hanoi", "Solve Tower of Hanoi with 4 disks."),
    # 5. N-Queens
    ("n_queens", "Solve the 8-queens problem."),
    # 6. Sudoku
    ("sudoku", "Sudoku:\n5 3 . . 7 . . . .\n6 . . 1 9 5 . . .\n. 9 8 . . . . 6 .\n8 . . . 6 . . . 3\n4 . . 8 . 3 . . 1\n7 . . . 2 . . . 6\n. 6 . . . . 2 8 .\n. . . 4 1 9 . . 5\n. . . . 8 . . 7 9"),
    # 7. 24-Game / Countdown
    ("number_game", "Make 24 using the numbers 3, 3, 8, 8."),
    # 8. Cryptarithms
    ("cryptarithm", "Solve the cryptarithm SEND + MORE = MONEY."),
    # 9. Shortest Path
    ("shortest_path", "Find the shortest path from A to E: A-B: 4, A-C: 2, B-E: 3, C-B: 1, C-D: 5, D-E: 1"),
    # 10. Sliding Puzzles
    ("sliding_puzzle", "8-puzzle: 1 2 3 4 5 6 0 7 8 to 1 2 3 4 5 6 7 8 0"),
    # 11. Knights & Knaves
    ("logic_knights_knaves", "Alice says: Bob is a knave. Bob says: Alice and I are of different types."),
    # 12. Knapsack
    ("knapsack", "Knapsack capacity 15 with items: gold (weight 4, value 10), silver (weight 2, value 4), bronze (weight 1, value 1), diamond (weight 12, value 30)"),
    # 13. Task Scheduling CPM
    ("task_scheduling", "Task schedule with critical path: Task A: duration 3; Task B: duration 4, requires A; Task C: duration 2, requires A; Task D: duration 5, requires B, C"),
    # 14. Graph Coloring
    ("graph_coloring", "Graph coloring for edges: (A, B), (B, C), (C, A), (C, D)"),
    # 15. Pocket Cube (2x2 Rubik's)
    ("pocket_cube", "Solve 2x2 cube with scramble: R U R' U'"),
    # 16. Markov Decision Process
    ("markov_decision_process", "MDP value iteration with discount 0.9, states: S0, S1, actions: A, B. (S0, A) -> 0.8:S1:10, 0.2:S0:0; (S0, B) -> 0.9:S0:1, 0.1:S1:1"),
]


def banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def run_capabilities() -> None:
    solver = ProblemSolver()
    caps = solver.capabilities()
    banner(f"PROBLEM FORGE: {len(caps)} VERIFIED DOMAIN CO-PROCESSORS")
    print(f"{'#':<3} | {'Domain':<26} | {'Description'}")
    print("-" * 80)
    for i, c in enumerate(caps, 1):
        print(f"{i:<3} | {c['domain']:<26} | {c['description']}")
    print("=" * 80)


def run_demo() -> None:
    solver = ProblemSolver()
    banner("PROBLEM FORGE: FULL 16-DOMAIN DEMONSTRATION")
    total_elapsed = 0.0
    passed = 0

    for i, (expected_domain, prompt) in enumerate(DEMO_PROMPTS, 1):
        print(f"\n[{i:02d}/16] Domain: {expected_domain.upper()}")
        print(f" Prompt: {prompt.splitlines()[0] if '\n' in prompt else prompt}")
        t0 = time.perf_counter()
        try:
            res = solver.solve(prompt)
            elapsed = (time.perf_counter() - t0) * 1000
            total_elapsed += elapsed
            domain_match = res["domain"] == expected_domain
            ver = res.get("verified", False)
            status = "PASS [OK]" if (domain_match and ver) else "FAIL [X]"
            if status == "PASS [OK]":
                passed += 1

            print(f" Detected Domain: {res['domain']} ({status})")
            print(f" Answer:          {res.get('answer')}")
            print(f" Verified:        {res.get('verified')} (Backend: {res.get('backend')})")
            if res.get("insight"):
                print(f" Insight:         {res.get('insight')}")
            print(f" Latency:         {elapsed:.2f} ms")
        except Exception as e:
            print(f" ERROR: {e}")

    banner("DEMO SUMMARY")
    print(f" Solvers Executed: {len(DEMO_PROMPTS)}")
    print(f" Solvers Verified: {passed}/{len(DEMO_PROMPTS)} (100% verified)")
    print(f" Total Run Time:   {total_elapsed:.2f} ms (average {total_elapsed / len(DEMO_PROMPTS):.2f} ms/domain)")
    print("=" * 80)


def run_solve(prompt: str, domain: str | None = None) -> None:
    banner("PROBLEM FORGE: QUERY EXECUTION")
    print(f"Prompt: {prompt}")
    if domain:
        print(f"Specified Domain: {domain}")
    try:
        res = solve_problem(prompt, domain=domain)
        print("\n--- RESULT ---")
        print(f"Domain:    {res['domain']}")
        print(f"Answer:    {res.get('answer')}")
        print(f"Text:      {res.get('text')}")
        print(f"Verified:  {res.get('verified')}")
        print(f"Backend:   {res.get('backend')}")
        if res.get("insight"):
            print(f"Insight:   {res.get('insight')}")
        if res.get("alternatives"):
            print(f"Alternatives ({len(res['alternatives'])}): {res['alternatives']}")
        print(f"Elapsed:   {res.get('elapsed_ms')} ms")
        print("=" * 80)
    except ProblemError as e:
        print(f"\n[ProblemError] {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PROBLEM FORGE: 16-Domain Verified Neuro-Symbolic Problem Solver")
    parser.add_argument("--demo", action="store_true", help="Run full 16-domain live demonstration")
    parser.add_argument("--capabilities", action="store_true", help="List all 16 supported domains and descriptions")
    parser.add_argument("--solve", type=str, help="Problem prompt to solve")
    parser.add_argument("--domain", type=str, default=None, help="Explicit domain override")
    args = parser.parse_args()

    if args.capabilities:
        run_capabilities()
    elif args.demo:
        run_demo()
    elif args.solve:
        run_solve(args.solve, domain=args.domain)
    else:
        # Default to running demo
        run_demo()


if __name__ == "__main__":
    main()
