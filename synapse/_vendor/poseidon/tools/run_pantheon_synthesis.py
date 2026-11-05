"""Project PANTHEON (24-Pillar Synthesis), Project NOOSPHERE (Creative Bisociation &
4-Agent Dialectical Tribunal), 25-Domain Problem Forge, and Symbolic Calculus/Matrix/Number-Theory
Verification & Benchmark Suite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseidon.creative_synthesis import ARCHETYPE_LIBRARY, DialecticalTribunal, verify_creative_receipt
from poseidon.pantheon import PANTHEON_24_PILLARS, PantheonEngine, verify_pantheon_receipt
from poseidon.problem_solver import ProblemSolver
from poseidon.reasoning import solve as solve_math


BENCHMARK_PROMPTS = [
    (
        "QuantumBot / Entanglement",
        "Calculate quantum entanglement entropy, Wootters concurrence, and CHSH Bell violation for Bell state amplitudes [1, 0, 0, 1]",
    ),
    (
        "QuantumBot / Grover Search",
        "Run Grover's algorithm unstructured quantum search for N=1024 items with M=1 marked state",
    ),
    (
        "Causeway / Pearl Do-Calculus",
        "Compute Pearl backdoor adjustment average treatment effect (ATE) and detect Simpson's paradox for confounder Z",
    ),
    (
        "GenesisEngine / Lotka-Volterra",
        "Solve Lotka-Volterra predator-prey trophic equilibrium and symplectic Hamiltonian for alpha=1.2, beta=0.4, gamma=0.8, delta=0.2",
    ),
    (
        "svideo & Flipper / Shannon-Hartley",
        "Calculate Shannon-Hartley channel capacity and Nyquist rate for bandwidth B = 2 MHz with SNR = 30 dB",
    ),
    (
        "DPLL Formal SAT / Boolean Satisfiability",
        "Solve Boolean SAT for CNF formula (A or B) and (not A or C) and (not B or C)",
    ),
    (
        "Game Theory / Bimatrix Nash Equilibrium",
        "Find all pure and mixed Nash equilibria for the Battle of the Sexes game",
    ),
    (
        "Bayesian Network / Diagnostic Posterior",
        "Calculate Bayesian posterior probability for prior = 0.01, sensitivity = 0.99, false positive rate = 0.05",
    ),
    (
        "Control Theory / Kalman & LQR Riccati",
        "Solve steady-state Kalman filter gain and Riccati covariance for A=1.0, H=1.0, Q=0.1, R=0.5",
    ),
    (
        "Bioinformatics / Sequence Alignment & LCS",
        "Compute Levenshtein edit distance and LCS between 'KITTEN' and 'SITTING'",
    ),
    (
        "Symbolic Calculus / Definite Integral",
        "integrate(3*x^2 - 4*x + 5, x, 0, 2)",
    ),
    (
        "Symbolic Linear Algebra / Matrix Eigenvalues",
        "eigenvalues([[4, 1], [2, 3]])",
    ),
]


def main() -> int:
    solver = ProblemSolver()
    caps = solver.capabilities()
    engine = PantheonEngine(core=None, seed=42)
    tribunal = DialecticalTribunal()

    results = []
    for label, prompt in BENCHMARK_PROMPTS:
        receipt = engine.synthesize(prompt, seed=42)
        ver = verify_pantheon_receipt(receipt)
        assert ver["verified"], f"Pantheon verification failed for {label}: {ver}"

        creative = tribunal.deliberate(prompt, seed=42)
        c_ver = verify_creative_receipt(creative)
        assert c_ver["verified"], f"Creative verification failed for {label}: {c_ver}"

        results.append(
            {
                "label": label,
                "domain": receipt["domain"],
                "answer": receipt["answer"],
                "dream_prompt": receipt["counterfactual_dream"]["dream_prompt"],
                "dream_answer": receipt["counterfactual_dream"]["dream_answer"],
                "noosphere_hybrid_operator": creative["blended_concept"]["hybrid_name"],
                "koestler_bisociation_score": creative["primary_bisociation"]["koestler_bisociation_score"],
                "top_hcri": creative["creative_portfolio"][0]["hcri_score"],
                "pillars_verified": receipt["pillars_verified"],
                "quibble_merkle_root": receipt["quibble_merkle_root"],
                "ntag215_crc16": receipt["ntag215_crc16"],
                "receipt_verified": ver["verified"] and c_ver["verified"],
            }
        )

    # Also verify standalone symbolic math extensions
    math_checks = {
        "calculus_diff": solve_math("diff(x^3 - 4*x^2 + 5*x - 2, x)")["answer"],
        "calculus_integrate": solve_math("integrate(3*x^2 - 8*x + 5, x, 0, 3)")["answer"],
        "matrix_inv": solve_math("inv([[4, 7], [2, 6]])")["answer"],
        "number_theory_crt": solve_math("crt([(2, 3), (3, 5), (2, 7)])")["answer"],
    }

    summary = {
        "benchmark": "Stage 17 Project NOOSPHERE, Project PANTHEON (24 Pillars), 25-Domain Problem Forge & Symbolic Math",
        "total_problem_forge_domains": len(caps),
        "total_pantheon_pillars": len(PANTHEON_24_PILLARS),
        "total_noosphere_archetypes": len(ARCHETYPE_LIBRARY),
        "prompts_evaluated": len(results),
        "all_receipts_verified": all(r["receipt_verified"] for r in results),
        "nexus_search_indexed_theorems": len(engine.nexus_search.entries),
        "symbolic_math_checks": math_checks,
        "results": results,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
