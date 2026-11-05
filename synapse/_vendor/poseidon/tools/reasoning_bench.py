"""Historical display-token smoke checks for Poseidon's symbolic tool stack.

Each case is a natural-language (or semi-formal) prompt with a known correct answer.
The router mirrors ``Poseidon.respond(mode="chat")``: exact maths first, then the
Problem Forge. A case scores only if a tool fires AND its answer contains every
expected token (whitespace/case-insensitive). Unanswered cases are reported
separately from wrong answers so routing gaps and solver errors stay distinguishable.
Token matching can accept incorrect answers. Use ``poseidon reasoning-audit``
for generated, independent numeric/structural checks; these smoke checks are not
an accuracy benchmark or proof that every scientific domain is verified.

Usage: python tools/reasoning_bench.py [--json] [--failures]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (id, category, prompt, expected tokens that must all appear in the normalized answer)
CASES = [
    # --- Formal maths (already-supported syntax) ---
    ("m01", "formal_math", "(17 + 5) * 3", ["66"]),
    ("m02", "formal_math", "3*x + 7 = 22", ["5"]),
    ("m03", "formal_math", "x^2 - 5*x + 6 = 0", ["2", "3"]),
    ("m04", "formal_math", "x^3 - 6*x^2 + 11*x - 6 = 0", ["1", "2", "3"]),
    ("m05", "formal_math", "det([[1, 2], [3, 4]])", ["-2"]),
    ("m06", "formal_math", "factor(360)", ["2^3", "3^2", "5"]),
    ("m07", "formal_math", "totient(36)", ["12"]),
    ("m08", "formal_math", "modinv(3, 11)", ["4"]),
    ("m09", "formal_math", "integrate(3*x^2 + 2*x, x, 0, 2)", ["12"]),
    ("m10", "formal_math", "diff(x^3 - 4*x, x, at=2)", ["8"]),
    # --- Natural-language maths (how people actually ask) ---
    ("n01", "nl_math", "If 2x + 3 = 11, what is x?", ["4"]),
    ("n02", "nl_math", "What is 15% of 240?", ["36"]),
    ("n03", "nl_math", "What is the derivative of x^3 - 4x?", ["3*x^2", "-4"]),
    ("n04", "nl_math", "Find the integral of 2x from 0 to 3", ["9"]),
    ("n05", "nl_math", "Solve x + y = 10 and x - y = 2", ["6", "4"]),
    ("n06", "nl_math", "What is the greatest common divisor of 84 and 36?", ["12"]),
    ("n07", "nl_math", "Factorize 1001", ["7", "11", "13"]),
    ("n08", "nl_math", "What is the determinant of [[2, 1], [5, 3]]?", ["1"]),
    ("n09", "nl_math", "What is 2 to the power of 10?", ["1024"]),
    ("n10", "nl_math", "What is the square root of 144?", ["12"]),
    ("n11", "nl_math", "Calculate 7 times 8 plus 6", ["62"]),
    ("n12", "nl_math", "What is the least common multiple of 4, 6 and 10?", ["60"]),
    # --- Problem Forge domains ---
    ("p01", "problem", "Tower of Hanoi with 3 disks", ["7"]),
    ("p02", "problem", "Solve the cryptarithm SEND + MORE = MONEY", ["9567", "1085", "10652"]),
    ("p03", "problem", "What is the edit distance between 'kitten' and 'sitting'?", ["3"]),
    ("p04", "problem", "Find the Nash equilibrium of the prisoner's dilemma", ["defect"]),
    ("p05", "problem", "A disease has prevalence 1%, test sensitivity 99% and false positive rate 5%. Using Bayes' theorem, what is the posterior probability of disease after a positive test?", ["1/6"]),
    ("p06", "problem", "Run Grover's algorithm for N=1024 items with M=1 marked state", ["25"]),
    ("p07", "problem", "Shannon-Hartley channel capacity for bandwidth B = 1 MHz and SNR = 20 dB", ["6.65"]),
    ("p08", "problem", "Hawk-Dove evolutionarily stable strategy with V=4 and C=6", ["2/3"]),
    ("p09", "problem", "Solve the 6-queens problem", ["["]),
    ("p10", "problem", "Kalman filter gain for A=1.0, H=1.0, Q=0.1, R=0.5", ["0.358"]),
    ("p11", "problem", "Is the CNF formula (A or B) and (not A) and (not B) satisfiable?", ["unsatisfiable"]),
    ("p12", "problem", "Make 24 from the numbers 4, 7, 8, 8", ["24"]),
    ("p13", "problem", "Fisher information metric and Cramer-Rao lower bound for Bernoulli coin flip with p=0.5", ["4.0000", "0.2500"]),
    ("p14", "problem", "Turing pattern morphogenesis activator-inhibitor critical wavenumber with a11=1.0, a12=-1.5, a21=1.5, a22=-2.0, Du=1.0, Dv=10.0", ["0.6325", "9.9346"]),
    ("p15", "problem", "Carnot heat engine thermal efficiency and work with Th=600K, Tc=300K, Qh=1000J", ["0.5000", "500.00"]),
    ("p16", "problem", "Spectral graph partition Fiedler algebraic connectivity and Cheeger conductance for dumbbell graph", ["0.4384", "0.1429"]),
    ("p17", "problem", "Kendall M/M/1 queuing system waiting time and queue length with arrival rate lambda=4.0 and service rate mu=5.0", ["4.000", "1.0000"]),
    ("p18", "problem", "Vietoris-Rips persistent homology Betti numbers with points: (0,0), (2,0), (2,2), (0,2) at filtration epsilon=2.1", ["beta_0 = 1", "beta_1 = 1"]),
    ("p19", "problem", "CRISPR-Cas9 sgRNA protospacer design and PAM sequence for target DNA sequence ATGCGATCGATCGATCGATCGATCGG", ["CGATCGATCGATCGATCGAT", "CGG"]),
    ("p20", "problem", "Monge-Kantorovich optimal transport and exact 1D Wasserstein distance with regularization=0.1", ["0.3000"]),
    ("p21", "problem", "Hohmann transfer orbital maneuver delta-v from r1=7000 km to r2=42164 km", ["3.77", "km/s"]),
    ("p22", "problem", "Hebbian Hopfield network associative memory and Ising model Lyapunov energy", ["attractor", "-5.000"]),
    ("p23", "problem", "Use E-Graph equality saturation to simplify expression: (x + 0) * 1 + (x - x)", ["x", "mdl"]),
    ("p24", "problem", "Run SINDy sparse identification symbolic regression on data points (-2, 15), (-1, 6), (0, 1), (1, 0), (2, 3), (3, 10)", ["2*x^2", "3*x"]),
    ("p25", "problem", "Analyze Kermack-McKendrick SIR model epidemiological dynamics with beta=0.5, gamma=0.1, s0=0.99", ["5.00", "0.8000"]),
    ("p26", "problem", "Design a closed-loop PID controller and check Routh-Hurwitz stability with kp=6.0, ki=4.0, kd=3.0", ["hurwitz", "6.2000"]),
    ("p27", "problem", "Verify Applied Category Theory commutative diagram and Symmetric Monoidal Category (SMC) interchange law", ["commutative", "monoidal"]),
    ("p28", "problem", "Verify GFlowNet trajectory balance and detailed balance on compositional DAG with rewards 4, 12, 8, 16", ["40.00", "m4"]),
    ("p29", "problem", "Compute Koopman operator EDMD polynomial observable linearization and eigenfunctions with lambda=0.8, mu=0.5, gamma=-0.6", ["0.8", "eigenfunction"]),
    ("p30", "problem", "Compute Buchberger Groebner basis and check polynomial ideal membership in Q[x, y]", ["x - 2*y", "y^2 - 1"]),
    ("p31", "problem", "Solve Control Barrier Function CBF-QP safety filter with x=1.8, x_max=2.0, u_des=3.0, alpha=2.0", ["1.3222", "cbf"]),
    ("p32", "problem", "Compute split conformal prediction quantile and finite-sample coverage guarantee at alpha=0.10 with y_hat=10.0", ["8.6000", "11.4000", "1.4000"]),
    ("p33", "problem", "Compute cellular sheaf cohomology H^0 and Sheaf Laplacian spectrum with stalk dimension 2", ["dim(h^0) = 2", "dim(h^1) = 2"]),
    ("p34", "problem", "Verify Port-Hamiltonian Dirac structure power conservation and IDA-PBC passivity-based control", ["passivity certified", "h_d"]),
    ("p35", "problem", "Compute Migdal-Kadanoff renormalization group critical coupling and correlation length critical exponent nu", ["0.609378", "1.338266"]),
    ("p36", "problem", "Verify Fibonacci anyon braiding Pentagon identity and Hexagon identity for 4 tau-anyons with braid word sigma_1 sigma_2 sigma_1", ["1.618034", "dim = 3"]),
    ("p37", "problem", "Compute Solomonoff universal prior, Minimum Description Length (MDL) posterior, and Levin universal search Kt complexity", ["h_1", "32/35", "9.0000"]),
    ("p38", "problem", "Analyze connectome network graph nodes=12 edges=18 for rich-club threshold degree k=3 and reciprocal dyads", ["rich-club", "0.6667", "dyads = 2"]),
    ("p39", "problem", "Calculate closed-form continuous-depth liquid time-constant neural dynamics with mean_act=0.5, gate=0.7, and dt=0.1", ["1.1423", "0.9162", "stable"]),
    ("p40", "problem", "Evaluate 3-factor monoaminergic neuromodulation with chemo=0.8, reward=0.6, wall=0.3, and choke=false", ["0.7400", "0.8400", "goal-exploitation"]),
    ("p41", "problem", "Execute differentiable neural graft routing across num_sources=4 with top_k=2 senders and null-route logit", ["0.6240", "top-2", "zero self-routing"]),
    ("p42", "problem", "Perform in silico virtual lesion audit and connectome ablation with threshold=0.15", ["human", "0.4720", "62.6%"]),
    ("p43", "problem", "Compute lattice gauge field connection Wilson loop holonomy on 4x4 grid with coupling g=1.0 at plaquette x=0, y=0", ["u(1) holonomy", "action s ="]),
    ("p44", "problem", "Calculate Jarzynski non-equilibrium thermodynamics free energy difference and Crooks fluctuation with beta=1.0, trajectories=150, shift=2.0", ["delta f", "w_diss", "sigma"]),
    ("p45", "problem", "Simulate symplectic Störmer-Verlet leapfrog integrator contact mechanics with q0=1.0, p0=0.0, dt=0.05, steps=100", ["e_0", "e_f", "symplectic conserved = true"]),
    ("p46", "problem", "Compose categorical bidirectional lens in sequential pipeline with state=[1.0, 2.0, 3.0] and feedback=[0.5, -0.2]", ["composed view", "composed update", "verified = true"]),
    ("p47", "problem", "Solve 2-player monoidal open game categorical optics Nash equilibrium with P1=[[3, 0], [1, 2]] and P2=[[3, 1], [0, 2]]", ["pure equilibria", "payoffs"]),
    ("p48", "problem", "Compute Causal Dynamical Triangulations CDT quantum spacetime foliation with time_slices=4, spatial_nodes=4, lambda=0.5", ["s_regge", "euler characteristic"]),
    ("p49", "problem", "Analyze Alain Connes spectral triple noncommutative geometry with nodes=3 and coupling c=1.5", ["tr(d²)", "connes spectral distances"]),
    ("p50", "problem", "Evaluate tropical algebra max-plus polynomial 0 + 2*x and 3 + 1*x and 5 + 0*x at x=2.0 with corner locus roots", ["tropical polynomial", "corner locus"]),
    ("p51", "problem", "Run Riemannian optimization on Stiefel manifold St(2, 3) with ambient n=3 and rank p=2 for iterations=5", ["manifold: st(2, 3)", "orthogonality residual"]),
    ("p52", "problem", "Compute coalgebraic bisimulation metric quantitative behavioral pseudometric with gamma=0.8 and iterations=8", ["behavioral distance", "triangle inequality preservation"]),
    ("p53", "problem", "Compute Noether symmetry and conserved charge for continuous SO(2) rotation with mass=1.0, r0=1.5, omega=2.0", ["so(2)", "dq/dt = 0"]),
    ("p54", "problem", "Evaluate Fibonacci modular tensor category MTC with Verlinde formula and quantum dimension d_tau", ["mtc", "verlinde = true"]),
    ("p55", "problem", "Compute Clifford spacetime algebra Cl(1,3) STA Faraday bivector with beta=0.6, e_field=[3.0, 4.0, 0.0], b_field=[0.0, 2.0, 1.0]", ["faraday bivector", "i1 ="]),
    ("p56", "problem", "Evaluate Batalin-Vilkovisky BV quantization with Classical Master Equation CME and gauge_param xi=1.0", ["(s,s)_bv = 0", "anomaly free"]),
    ("p57", "problem", "Analyze Active Brownian Particle ABP active matter nonequilibrium dynamics with swim_speed=10.0 and rotational_diffusion=1.0", ["persistence length lp", "deff ="]),
    ("p58", "problem", "Analyze Cartan exterior differential system and moving frames for dimension=3", ["cartan", "characters", "involutive"]),
    ("p59", "problem", "Evaluate non-hermitian PT-symmetric quantum Hamiltonian with delta=0.0, gamma=1.0, and coupling_g=1.5", ["unbroken", "phase"]),
    ("p60", "problem", "Compute exceptional Jordan algebra Albert algebra h3(O) with diagonal=[2.0, 3.0, 1.0]", ["albert algebra", "det(a) ="]),
    ("p61", "problem", "Compute holographic Ryu-Takayanagi entanglement entropy with central_charge=12.0, length=100.0, and subsystem=25.0", ["ryu-takayanagi", "nats"]),
    ("p62", "problem", "Analyze morphogenetic Turing reaction-diffusion instability with Du=0.01, Dv=0.50, fu=0.6, and gv=-1.1", ["turing active", "kc ="]),
    ("p63", "problem", "Evaluate tether reliability and expert weighting with ensemble_size=5, temperature=1.0, and outlier_error=0.8", ["tether weights", "epistemic trust tau"]),
    ("p64", "problem", "Evaluate counterfactual covariation and four-view causal objective for arm=tether, effect_weight=1.0, and invariance_weight=1.0", ["arm = tether", "total loss ="]),
    ("p65", "problem", "Compute causal attribution and pathway saliency with members=5 and sensitivity=1.0", ["pathway contributions", "gini ="]),
    ("p66", "problem", "Run zero-shot corruption detection and detect corrupted specialists with members=5 and corrupted_count=1", ["corrupted detected", "consensus preserved = true"]),
    ("p67", "problem", "Evaluate evidence-calibrated fallback gate with trust=0.85, margin=0.25, and risk=0.10", ["override admitted = true", "decision = qualified_override_admitted"]),
    ("p68", "problem", "Run faultline joint failure selection with panel size=3, policy=joint, and seed=42", ["selected active panel", "joint failure advantage"]),
    ("p69", "problem", "Verify higher order failure hyperedge and 3-bit parity benchmark for even parity", ["pairwise second-moment bound", "actual majority error rate"]),
    ("p70", "problem", "Compute condition stratified risk matrix and co-failure signature with seed=42", ["condition-stratified risk matrix", "dominant joint failure patterns"]),
    ("p71", "problem", "Compute continuous joint error covariance matrix and ensemble diversity gain with models=5, samples=64, and seed=42", ["continuous joint error covariance", "ensemble diversity gain"]),
    ("p72", "problem", "Perform ensemble panel minimax triage to find worst regime error minimization with panel size=3 and seed=42", ["minimax continuous triage", "minimax optimal panel"]),
    ("p73", "problem", "Prune Crosswire version space with actions=4 and traces=3", ["surviving version space", "final entropy"]),
    ("p74", "problem", "Run active diagnostic probing with actions=4 and budget=8", ["probing steps executed", "final surviving"]),
    ("p75", "problem", "Verify interface contract all-hypothesis verification for plan=[1, 1] with surviving hypotheses=6 and hazard=4", ["plan admitted", "satisfying"]),
    ("p76", "problem", "Execute closed-loop runtime shield with intended action=2, surviving count=4, and safe fallback=0", ["decision", "dispatched action"]),
    ("p77", "problem", "Synthesize unfamiliar protocol interoperability with actions=4, probing_budget=12, and seed=42", ["interoperability certified", "probing steps"]),
    ("p78", "problem", "Update successor representation TD learning with state=2, next_state=3, discount=0.9, and learning_rate=0.2", ["updated m", "td error norm"]),
    ("p79", "problem", "Compute successor reward revaluation with num_states=4 and reward vector: [0.0, 1.0, 5.0, -2.0]", ["best state", "revalued values"]),
    ("p80", "problem", "Run sharp-wave ripple reverse replay with sweeps=4, decay=0.8, and threshold=0.1", ["propagated value s0", "sweeps executed"]),
    ("p81", "problem", "Compute graph Laplacian spatial eigenvectors with num_states=5, tau=1.0, and seed=42", ["fiedler eigenvalue", "mean diffusion distance"]),
    ("p82", "problem", "Run trainable neural graft routing with num_sources=4, null_route_bias=0.5, and top_k=2", ["routing entropy", "mean null weight"]),
    ("p83", "problem", "Compute Poincare return map Chirikov standard map with stochastic_k=0.85 and iterations=400", ["lambda_1", "lambda_2", "regime"]),
    ("p84", "problem", "Compute hyperbolic Poincare disk geodesic distance with curvature c=1.0 for point u=[0.2, 0.3] and point v=[-0.4, 0.1]", ["geodesic distance", "mobius sum", "gromov delta"]),
    ("p85", "problem", "Verify Poincare duality and Morse chain complex for surface=torus", ["betti numbers", "euler chi = 0", "poincare duality = true"]),
    ("p86", "problem", "Run Hamilton-Perelman normalized Ricci flow with vertices=6, steps=30, and euler chi=2", ["k_bar", "variance reduction", "gauss-bonnet conserved = true"]),
    ("p87", "problem", "Synthesize unified quartet cognitive synthesis with base_action=0, intended_action=1, and hazard_action=5", ["final action", "override admitted", "shield", "panel"]),
]


def _norm(text: str) -> str:
    return "".join(str(text).lower().split())


def route(prompt: str):
    """Same precedence as runtime chat routing: exact maths, then Problem Forge."""
    from poseidon.reasoning import extract_math, solve
    from poseidon.problem_solver import detect_problem, solve_problem

    expr = extract_math(prompt)
    if expr:
        try:
            return "math", solve(expr)
        except Exception as error:  # solver refused: report, don't crash the bench
            return "math_error", {"answer": "", "text": f"{type(error).__name__}: {error}"}
    domain = detect_problem(prompt)
    if domain is not None:
        try:
            return f"problem:{domain}", solve_problem(prompt, domain=domain)
        except Exception as error:
            return "problem_error", {"answer": "", "text": f"{type(error).__name__}: {error}"}
    return None, None


def run(cases=CASES):
    rows = []
    for case_id, category, prompt, expected in cases:
        t0 = time.perf_counter()
        tool, result = route(prompt)
        ms = (time.perf_counter() - t0) * 1000.0
        answer = "" if result is None else f"{result.get('answer', '')} {result.get('text', '')}"
        if tool is None:
            status = "unanswered"
        elif all(_norm(tok) in _norm(answer) for tok in expected):
            status = "correct"
        else:
            status = "wrong"
        rows.append({
            "id": case_id, "category": category, "prompt": prompt, "tool": tool,
            "status": status, "expected": expected,
            "answer": (result or {}).get("answer"), "ms": round(ms, 1),
        })
    by_cat = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"total": 0, "correct": 0, "wrong": 0, "unanswered": 0})
        c["total"] += 1
        c[r["status"]] += 1
    total = len(rows)
    correct = sum(r["status"] == "correct" for r in rows)
    return {
        "scoring_scope": "legacy_display_token_smoke",
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "wrong": sum(r["status"] == "wrong" for r in rows),
        "unanswered": sum(r["status"] == "unanswered" for r in rows),
        "by_category": by_cat,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--failures", action="store_true")
    args = ap.parse_args()
    report = run()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return int(bool(report["wrong"] or report["unanswered"]))
    print(f"Display checks: {report['correct']}/{report['total']} = {report['accuracy']:.1%}  "
          f"(wrong {report['wrong']}, unanswered {report['unanswered']})")
    for cat, c in report["by_category"].items():
        print(f"  {cat:12s} {c['correct']}/{c['total']}  wrong={c['wrong']} unanswered={c['unanswered']}")
    if args.failures:
        for r in report["rows"]:
            if r["status"] != "correct":
                print(f"[{r['status'].upper():10s}] {r['id']} tool={r['tool']} expected={r['expected']} "
                      f"got={str(r['answer'])[:90]!r}\n             prompt={r['prompt']!r}")
    return int(bool(report["wrong"] or report["unanswered"]))


if __name__ == "__main__":
    raise SystemExit(main())
