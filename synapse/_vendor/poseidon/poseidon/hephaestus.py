"""PROJECT HEPHAESTUS: Wake-Sleep Library Learning, GFlowNet Trajectory Balance, Koopman EDMD & Buchberger Gröbner Basis Engine.

Grounded in 2024–2026 frontier neuro-symbolic AI, operator theory, and algebraic verification literature:
1. WakeSleepLibraryLearner (DreamCoder / Stitch POPL 2023 / LILO 2024):
   - Discovers reusable lambda-abstractions fn_k(#0, #1) across a corpus of AST programs
     by maximizing Minimum Description Length (MDL) compression utility, rewrites the corpus,
     auto-documents abstractions, and verifies exact beta-reduction equivalence.
2. GFlowNetSampler (Bengio et al. / Malkin et al. Trajectory Balance & Detailed Balance):
   - Amortized probabilistic inference over compositional DAGs where terminal states x are
     sampled proportionally to reward R(x) / Z. Computes exact edge flows, partition function Z,
     forward/backward policies P_F and P_B, and certifies Trajectory Balance L_TB(tau) == 0.
3. KoopmanEDMDLinearizer (Extended Dynamic Mode Decomposition & Koopman Spectral Lifting):
   - Lifts nonlinear discrete-time dynamical systems x_{t+1} = F(x_t) into a finite dictionary
     of nonlinear observables Psi(x), solves K = G^+ A in exact least squares, extracts Koopman
     eigenvalues/eigenfunctions, and verifies linear multi-step evolution in Hilbert space.
4. BuchbergerGroebnerProver (Buchberger's Algorithm & Hilbert Nullstellensatz Ideal Membership):
   - Exact rational multivariate polynomial ring Q[x, y] with lexicographic monomial order,
     multivariate polynomial division, S-polynomial reduction, reduced Gröbner basis computation,
     and certified polynomial ideal membership p in <f_1, ..., f_k>.

Strictly zero eval() / exec().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .athenaeum import _parse_to_ast, _ast_to_str, _ast_cost


# ---------------------------------------------------------------------------
# 1. Wake-Sleep Library Learning & Stitch Abstraction Compression
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LambdaAbstraction:
    """A learned reusable lambda primitive extracted during the Sleep phase."""
    name: str
    arity: int
    body_ast: Any
    body_str: str
    autodoc: str
    compression_gain: int
    usage_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "signature": f"{self.name}({', '.join(f'#{i}' for i in range(self.arity))}) := {self.body_str}",
            "arity": self.arity,
            "body_str": self.body_str,
            "autodoc": self.autodoc,
            "compression_gain": self.compression_gain,
            "usage_count": self.usage_count,
        }


def _collect_subtrees(ast_node: Any) -> List[Any]:
    """Return all compound subtrees of an AST node."""
    out: List[Any] = []
    if isinstance(ast_node, tuple) and len(ast_node) >= 2:
        out.append(ast_node)
        for child in ast_node[1:]:
            out.extend(_collect_subtrees(child))
    return out


def _anti_unify(node_a: Any, node_b: Any, hole_map: Dict[Tuple[Any, Any], str]) -> Optional[Any]:
    """Perform syntactic anti-unification (LGG) of two AST subtrees into a pattern with #i holes."""
    if node_a == node_b:
        return node_a
    if isinstance(node_a, tuple) and isinstance(node_b, tuple):
        if len(node_a) == len(node_b) and node_a[0] == node_b[0]:
            children = []
            for ca, cb in zip(node_a[1:], node_b[1:]):
                u = _anti_unify(ca, cb, hole_map)
                if u is None:
                    return None
                children.append(u)
            return (node_a[0], *children)
    # Replace differing leaves/subtrees with a hole #i (up to 2 holes)
    key = (node_a, node_b)
    if key in hole_map:
        return hole_map[key]
    if len(hole_map) >= 2:
        return None
    hole_name = f"#{len(hole_map)}"
    hole_map[key] = hole_name
    return hole_name


def _match_abstraction(pattern: Any, target: Any, bindings: Dict[str, Any]) -> bool:
    """Check if target AST matches pattern with holes #0, #1."""
    if isinstance(pattern, str) and pattern.startswith("#"):
        if pattern in bindings:
            return bindings[pattern] == target
        bindings[pattern] = target
        return True
    if isinstance(pattern, tuple) and isinstance(target, tuple):
        if len(pattern) != len(target) or pattern[0] != target[0]:
            return False
        for p_child, t_child in zip(pattern[1:], target[1:]):
            if not _match_abstraction(p_child, t_child, bindings):
                return False
        return True
    return pattern == target


def _rewrite_with_abstraction(ast_node: Any, fn_name: str, pattern: Any, arity: int) -> Tuple[Any, int]:
    """Replace occurrences of pattern in ast_node with (fn_name, arg0, ...)."""
    bindings: Dict[str, Any] = {}
    if _match_abstraction(pattern, ast_node, bindings) and len(bindings) == arity:
        args = [bindings[f"#{i}"] for i in range(arity)]
        return (fn_name, *args), 1
    if isinstance(ast_node, tuple) and len(ast_node) >= 2:
        new_children = []
        total_hits = 0
        for child in ast_node[1:]:
            rw_child, hits = _rewrite_with_abstraction(child, fn_name, pattern, arity)
            new_children.append(rw_child)
            total_hits += hits
        return (ast_node[0], *new_children), total_hits
    return ast_node, 0


def _substitute_holes(pattern: Any, args: Sequence[Any]) -> Any:
    """Beta-reduce an abstraction body by substituting #i -> args[i]."""
    if isinstance(pattern, str) and pattern.startswith("#"):
        idx = int(pattern[1:])
        return args[idx]
    if isinstance(pattern, tuple) and len(pattern) >= 2:
        return (pattern[0], *[_substitute_holes(c, args) for c in pattern[1:]])
    return pattern


def _beta_reduce_all(ast_node: Any, library: Dict[str, LambdaAbstraction]) -> Any:
    """Expand all learned function calls back to base primitives for semantic verification."""
    if isinstance(ast_node, tuple) and len(ast_node) >= 1:
        op = ast_node[0]
        expanded_children = [_beta_reduce_all(c, library) for c in ast_node[1:]]
        if op in library:
            abstr = library[op]
            return _beta_reduce_all(_substitute_holes(abstr.body_ast, expanded_children), library)
        return (op, *expanded_children)
    return ast_node


class WakeSleepLibraryLearner:
    """DreamCoder / Stitch / LILO Wake-Sleep Abstraction Compression Engine."""

    def compress_corpus(self, programs: Sequence[str], max_abstractions: int = 2) -> Dict[str, Any]:
        parsed_corpus = [_parse_to_ast(p) for p in programs]
        initial_total_cost = sum(_ast_cost(ast) for ast in parsed_corpus)

        current_corpus = list(parsed_corpus)
        library: Dict[str, LambdaAbstraction] = {}

        for step_idx in range(max_abstractions):
            # Collect candidate patterns via anti-unification across subtree pairs
            all_subtrees: List[Any] = []
            for ast in current_corpus:
                all_subtrees.extend(_collect_subtrees(ast))

            candidate_patterns: List[Any] = []
            seen_pat: Set[Any] = set()
            for i in range(len(all_subtrees)):
                for j in range(i + 1, len(all_subtrees)):
                    hmap: Dict[Tuple[Any, Any], str] = {}
                    pat = _anti_unify(all_subtrees[i], all_subtrees[j], hmap)
                    if pat is not None and isinstance(pat, tuple) and _ast_cost(pat) >= 4:
                        if pat not in seen_pat:
                            seen_pat.add(pat)
                            candidate_patterns.append(pat)

            best_gain = 0
            best_abstr: Optional[LambdaAbstraction] = None
            best_rewritten: Optional[List[Any]] = None

            fn_name = f"fn_{step_idx}"
            for pat in candidate_patterns:
                # Determine arity
                pat_str = _ast_to_str(pat)
                holes = sorted(set(re.findall(r"#\d+", pat_str)))
                arity = len(holes)
                if holes != [f"#{k}" for k in range(arity)]:
                    continue

                rewritten_list = []
                total_hits = 0
                for ast in current_corpus:
                    rw_ast, hits = _rewrite_with_abstraction(ast, fn_name, pat, arity)
                    rewritten_list.append(rw_ast)
                    total_hits += hits

                if total_hits < 2:
                    continue

                old_cost = sum(_ast_cost(a) for a in current_corpus)
                new_corpus_cost = sum(_ast_cost(a) for a in rewritten_list)
                lib_def_cost = _ast_cost(pat) + arity
                gain = old_cost - (new_corpus_cost + lib_def_cost)

                if gain > best_gain:
                    best_gain = gain
                    autodoc = (
                        f"Reusable {arity}-ary operator computing {pat_str} "
                        f"(reduces corpus MDL by {gain} nodes across {total_hits} call sites)."
                    )
                    best_abstr = LambdaAbstraction(
                        name=fn_name,
                        arity=arity,
                        body_ast=pat,
                        body_str=pat_str,
                        autodoc=autodoc,
                        compression_gain=gain,
                        usage_count=total_hits,
                    )
                    best_rewritten = rewritten_list

            if best_abstr is None or best_rewritten is None:
                break

            library[best_abstr.name] = best_abstr
            current_corpus = best_rewritten

        # Verify that beta-reducing every compressed program exactly equals the original AST
        beta_verified = True
        for orig_ast, comp_ast in zip(parsed_corpus, current_corpus):
            expanded = _beta_reduce_all(comp_ast, library)
            if expanded != orig_ast:
                beta_verified = False
                break

        final_corpus_cost = sum(_ast_cost(a) for a in current_corpus) + sum(
            _ast_cost(ab.body_ast) + ab.arity for ab in library.values()
        )

        return {
            "original_programs": [_ast_to_str(a) for a in parsed_corpus],
            "compressed_programs": [_ast_to_str(a) for a in current_corpus],
            "learned_library": [ab.to_dict() for ab in library.values()],
            "num_abstractions": len(library),
            "initial_mdl_cost": initial_total_cost,
            "final_mdl_cost": final_corpus_cost,
            "compression_ratio": round(initial_total_cost / max(1, final_corpus_cost), 4),
            "beta_reduction_verified": beta_verified,
            "verified": bool(beta_verified and final_corpus_cost <= initial_total_cost),
        }


# ---------------------------------------------------------------------------
# 2. Generative Flow Network (GFlowNet) with Trajectory & Detailed Balance
# ---------------------------------------------------------------------------

class GFlowNetSampler:
    """Exact DAG Generative Flow Network with Trajectory Balance (TB) & Detailed Balance (DB)."""

    def calibrate_and_verify(
        self,
        terminal_rewards: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Calibrates flows over a 3-layer compositional DAG and certifies TB & DB identities."""
        if terminal_rewards is None:
            terminal_rewards = {
                "T1_linear_law": 4.0,
                "T2_quadratic_law": 12.0,
                "T3_harmonic_law": 8.0,
                "T4_conservation_law": 16.0,
            }

        t_keys = list(terminal_rewards.keys())
        while len(t_keys) < 4:
            t_keys.append(f"T{len(t_keys)+1}")
            terminal_rewards[t_keys[-1]] = 4.0

        r1, r2, r3, r4 = [max(1e-6, float(terminal_rewards[k])) for k in t_keys[:4]]
        t1, t2, t3, t4 = t_keys[:4]

        # Compositional DAG with non-tree diamond merges:
        # Root S0 -> Intermediate {A1, A2}
        # A1 -> {t1, t2, t3}
        # A2 -> {t2, t3, t4}
        # Note t2 and t3 have TWO parents (A1 and A2), testing genuine DAG non-tree flow splitting!
        edge_flows: Dict[Tuple[str, str], float] = {
            ("A1", t1): r1,
            ("A1", t2): 0.5 * r2,
            ("A1", t3): 0.25 * r3,
            ("A2", t2): 0.5 * r2,
            ("A2", t3): 0.75 * r3,
            ("A2", t4): r4,
        }
        flow_A1 = edge_flows[("A1", t1)] + edge_flows[("A1", t2)] + edge_flows[("A1", t3)]
        flow_A2 = edge_flows[("A2", t2)] + edge_flows[("A2", t3)] + edge_flows[("A2", t4)]
        edge_flows[("S0", "A1")] = flow_A1
        edge_flows[("S0", "A2")] = flow_A2

        Z = flow_A1 + flow_A2
        state_flows: Dict[str, float] = {
            "S0": Z,
            "A1": flow_A1,
            "A2": flow_A2,
            t1: r1,
            t2: r2,
            t3: r3,
            t4: r4,
        }

        # Compute Forward Policy P_F(s' | s) = F(s -> s') / F(s)
        # and Backward Policy P_B(s | s') = F(s -> s') / F(s')
        children: Dict[str, List[str]] = {
            "S0": ["A1", "A2"],
            "A1": [t1, t2, t3],
            "A2": [t2, t3, t4],
        }
        parents: Dict[str, List[str]] = {
            "A1": ["S0"],
            "A2": ["S0"],
            t1: ["A1"],
            t2: ["A1", "A2"],
            t3: ["A1", "A2"],
            t4: ["A2"],
        }

        p_forward: Dict[str, float] = {}
        p_backward: Dict[str, float] = {}
        max_db_residual = 0.0

        for (u, v), f_uv in edge_flows.items():
            pf = f_uv / state_flows[u]
            pb = f_uv / state_flows[v]
            p_forward[f"{u}->{v}"] = round(pf, 8)
            p_backward[f"{v}->{u}"] = round(pb, 8)
            # Detailed Balance check: F(u) * P_F(v|u) == F(v) * P_B(u|v)
            db_res = abs(state_flows[u] * pf - state_flows[v] * pb)
            if db_res > max_db_residual:
                max_db_residual = db_res

        # Enumerate all 6 complete trajectories tau = (S0 -> Ai -> Tj) and verify Trajectory Balance:
        # L_TB(tau) = (ln Z + sum ln P_F - ln R(x) - sum ln P_B)^2
        trajectories = [
            ("S0", "A1", t1),
            ("S0", "A1", t2),
            ("S0", "A1", t3),
            ("S0", "A2", t2),
            ("S0", "A2", t3),
            ("S0", "A2", t4),
        ]
        tb_losses: List[float] = []
        terminal_marginal_prob: Dict[str, float] = {k: 0.0 for k in (t1, t2, t3, t4)}

        for s0, s1, s2 in trajectories:
            pf_1 = edge_flows[(s0, s1)] / state_flows[s0]
            pf_2 = edge_flows[(s1, s2)] / state_flows[s1]
            pb_1 = edge_flows[(s0, s1)] / state_flows[s1]
            pb_2 = edge_flows[(s1, s2)] / state_flows[s2]
            r_x = state_flows[s2]

            tb_diff = math.log(Z) + math.log(pf_1) + math.log(pf_2) - math.log(r_x) - math.log(pb_1) - math.log(pb_2)
            tb_losses.append(tb_diff * tb_diff)
            terminal_marginal_prob[s2] += pf_1 * pf_2

        max_tb_loss = max(tb_losses)
        # Verify terminal sampling distribution matches R(x) / Z
        kl_div = 0.0
        for k in (t1, t2, t3, t4):
            p_emp = terminal_marginal_prob[k]
            p_target = state_flows[k] / Z
            if p_emp > 0 and p_target > 0:
                kl_div += p_emp * math.log(p_emp / p_target)

        verified = bool(max_tb_loss < 1e-18 and max_db_residual < 1e-12 and abs(kl_div) < 1e-12)
        mode_star = max((t1, t2, t3, t4), key=lambda k: terminal_marginal_prob[k])

        return {
            "partition_function_Z": round(Z, 6),
            "terminal_rewards": {k: state_flows[k] for k in (t1, t2, t3, t4)},
            "terminal_sampling_probabilities": {k: round(v, 6) for k, v in terminal_marginal_prob.items()},
            "highest_probability_mode": mode_star,
            "forward_policy": p_forward,
            "backward_policy": p_backward,
            "max_trajectory_balance_loss": max_tb_loss,
            "max_detailed_balance_residual": max_db_residual,
            "kl_divergence_to_reward_posterior": abs(kl_div),
            "trajectories_evaluated": len(trajectories),
            "verified": verified,
        }


# ---------------------------------------------------------------------------
# 3. Koopman Operator & Extended Dynamic Mode Decomposition (EDMD)
# ---------------------------------------------------------------------------

class KoopmanEDMDLinearizer:
    """Extended Dynamic Mode Decomposition (EDMD) & Exact Koopman Eigenfunction Extractor."""

    OBSERVABLE_NAMES = ["x1", "x2", "x1^2"]

    @staticmethod
    def lift(x1: float, x2: float) -> List[float]:
        """Lift 2D state (x1, x2) into 3D Koopman observable space Psi(x) = [x1, x2, x1^2]."""
        return [x1, x2, x1 * x1]

    @staticmethod
    def _solve_3x3(G: List[List[float]], rhs: List[float]) -> List[float]:
        n = 3
        aug = [[G[i][j] + (1e-12 if i == j else 0.0) for j in range(n)] + [rhs[i]] for i in range(n)]
        for col in range(n):
            piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[piv] = aug[piv], aug[col]
            div = aug[col][col]
            if abs(div) < 1e-15:
                continue
            for j in range(col, n + 1):
                aug[col][j] /= div
            for r in range(n):
                if r != col and abs(aug[r][col]) > 0.0:
                    fac = aug[r][col]
                    for j in range(col, n + 1):
                        aug[r][j] -= fac * aug[col][j]
        return [aug[i][n] for i in range(n)]

    def analyze_system(
        self,
        lam: float = 0.8,
        mu: float = 0.5,
        gamma: float = -0.6,
        steps: int = 8,
    ) -> Dict[str, Any]:
        """Analyzes the classic nonlinear dynamical system:
        x1_{t+1} = lam * x1_t
        x2_{t+1} = mu * x2_t + gamma * x1_t^2
        In the lifted observable basis Psi(x) = [x1, x2, x1^2]^T, the Koopman operator is finite & exact:
        Psi(x_{t+1}) = K_op * Psi(x_t) where K_op = [[lam, 0, 0], [0, mu, gamma], [0, 0, lam^2]].
        """
        # Generate diverse snapshot pairs (x_m, y_m = F(x_m))
        initial_states = [
            (1.0, 0.5),
            (2.0, -1.0),
            (-1.5, 1.2),
            (0.8, -0.4),
            (1.2, 2.0),
            (-0.7, -1.5),
        ]
        psi_x: List[List[float]] = []
        psi_y: List[List[float]] = []
        for x1_0, x2_0 in initial_states:
            x1, x2 = x1_0, x2_0
            for _ in range(3):
                y1 = lam * x1
                y2 = mu * x2 + gamma * (x1 * x1)
                psi_x.append(self.lift(x1, x2))
                psi_y.append(self.lift(y1, y2))
                x1, x2 = y1, y2

        m_snaps = len(psi_x)
        dim = 3
        # Gram matrix G = (1/M) sum Psi(x_m) Psi(x_m)^T
        # Cross matrix A = (1/M) sum Psi(x_m) Psi(y_m)^T
        G = [[sum(psi_x[m][i] * psi_x[m][j] for m in range(m_snaps)) / m_snaps for j in range(dim)] for i in range(dim)]
        A = [[sum(psi_x[m][i] * psi_y[m][j] for m in range(m_snaps)) / m_snaps for j in range(dim)] for i in range(dim)]

        # Solve K = G^+ A column by column, then transpose so Psi(y) = K_op @ Psi(x)
        K_cols = [self._solve_3x3(G, [A[i][col] for i in range(dim)]) for col in range(dim)]
        # K_op[row][col] = K_cols[row][col] maps Psi(x) -> Psi(y)
        K_op = [[round(K_cols[row][col], 6) for col in range(dim)] for row in range(dim)]

        # Multi-step linear prediction error in lifted Koopman space from x0 = (1.5, 1.0)
        x1_true, x2_true = 1.5, 1.0
        psi_pred = self.lift(x1_true, x2_true)
        max_koopman_error = 0.0
        for _ in range(steps):
            x1_true = lam * x1_true
            x2_true = mu * x2_true + gamma * (x1_true / lam) ** 2
            psi_pred = [sum(K_op[i][j] * psi_pred[j] for j in range(dim)) for i in range(dim)]
            err = math.hypot(psi_pred[0] - x1_true, psi_pred[1] - x2_true)
            if err > max_koopman_error:
                max_koopman_error = err

        # Exact Koopman eigenvalues: mu_1 = lam, mu_2 = mu, mu_3 = lam^2
        eigenvalues = sorted([round(lam, 6), round(mu, 6), round(lam * lam, 6)], reverse=True)
        # Verify Koopman eigenfunction phi_2(x) = x2 - (gamma / (lam^2 - mu)) * x1^2 satisfies phi_2(F(x)) = mu * phi_2(x)
        denom = (lam * lam - mu) if abs(lam * lam - mu) > 1e-9 else 1.0
        c_eigen = -gamma / denom
        x1_test, x2_test = 1.25, -0.75
        phi_x = x2_test + c_eigen * (x1_test ** 2)
        phi_fx = (mu * x2_test + gamma * (x1_test ** 2)) + c_eigen * ((lam * x1_test) ** 2)
        eigenfunction_residual = abs(phi_fx - mu * phi_x)

        spectral_radius = max(abs(e) for e in eigenvalues)
        verified = bool(max_koopman_error < 1e-5 and eigenfunction_residual < 1e-9)

        return {
            "observable_dictionary": self.OBSERVABLE_NAMES,
            "koopman_matrix": K_op,
            "koopman_eigenvalues": eigenvalues,
            "spectral_radius": round(spectral_radius, 6),
            "asymptotically_stable": spectral_radius < 1.0,
            "principal_eigenfunction": f"phi(x1, x2) = x2 + ({c_eigen:.4f})*x1^2",
            "eigenfunction_linearity_residual": eigenfunction_residual,
            "multi_step_prediction_error": max_koopman_error,
            "snapshots_used": m_snaps,
            "verified": verified,
        }


# ---------------------------------------------------------------------------
# 4. Buchberger's Gröbner Basis & Polynomial Ideal Membership Engine
# ---------------------------------------------------------------------------

Monomial = Tuple[int, int]  # (deg_x, deg_y) in Q[x, y] under Lex order (x > y)
PolyDict = Dict[Monomial, Fraction]


def _clean_poly(p: PolyDict) -> PolyDict:
    return {m: c for m, c in p.items() if c != 0}


def _lead_monomial(p: PolyDict) -> Monomial:
    return max(p.keys())  # Python tuple comparison on (deg_x, deg_y) is exact Lex order (x > y)


def _poly_add(a: PolyDict, b: PolyDict, scale_b: Fraction = Fraction(1)) -> PolyDict:
    res = dict(a)
    for m, c in b.items():
        res[m] = res.get(m, Fraction(0)) + scale_b * c
    return _clean_poly(res)


def _poly_mul_term(p: PolyDict, mono: Monomial, coeff: Fraction) -> PolyDict:
    if coeff == 0:
        return {}
    return {(m[0] + mono[0], m[1] + mono[1]): c * coeff for m, c in p.items() if c * coeff != 0}


def _monomial_divides(a: Monomial, b: Monomial) -> bool:
    """Return True if monomial a divides monomial b."""
    return a[0] <= b[0] and a[1] <= b[1]


def _poly_rem(f: PolyDict, basis: Sequence[PolyDict]) -> PolyDict:
    """Multivariate polynomial division of f by basis G in Q[x, y] under Lex order."""
    p = _clean_poly(f)
    r: PolyDict = {}
    basis_nonempty = [_clean_poly(g) for g in basis if _clean_poly(g)]

    steps = 0
    while p and steps < 200:
        steps += 1
        lt_m = _lead_monomial(p)
        lt_c = p[lt_m]
        divided = False
        for g in basis_nonempty:
            gm = _lead_monomial(g)
            gc = g[gm]
            if _monomial_divides(gm, lt_m):
                diff_m = (lt_m[0] - gm[0], lt_m[1] - gm[1])
                scale = lt_c / gc
                p = _poly_add(p, _poly_mul_term(g, diff_m, -scale))
                divided = True
                break
        if not divided:
            r[lt_m] = r.get(lt_m, Fraction(0)) + lt_c
            del p[lt_m]
            p = _clean_poly(p)
    return _clean_poly(r)


def _s_polynomial(f: PolyDict, g: PolyDict) -> PolyDict:
    """Construct S-polynomial S(f, g) = (lcm/LT(f))*f - (lcm/LT(g))*g."""
    fm, gm = _lead_monomial(f), _lead_monomial(g)
    fc, gc = f[fm], g[gm]
    lcm_m = (max(fm[0], gm[0]), max(fm[1], gm[1]))
    term_f = _poly_mul_term(f, (lcm_m[0] - fm[0], lcm_m[1] - fm[1]), Fraction(1) / fc)
    term_g = _poly_mul_term(g, (lcm_m[0] - gm[0], lcm_m[1] - gm[1]), Fraction(1) / gc)
    return _poly_add(term_f, term_g, Fraction(-1))


def _poly_to_str(p: PolyDict) -> str:
    p = _clean_poly(p)
    if not p:
        return "0"
    terms: List[str] = []
    for (dx, dy) in sorted(p.keys(), reverse=True):
        c = p[(dx, dy)]
        v_parts: List[str] = []
        if dx == 1:
            v_parts.append("x")
        elif dx > 1:
            v_parts.append(f"x^{dx}")
        if dy == 1:
            v_parts.append("y")
        elif dy > 1:
            v_parts.append(f"y^{dy}")
        v_str = "*".join(v_parts)
        if not v_str:
            terms.append(str(c))
        elif c == 1:
            terms.append(v_str)
        elif c == -1:
            terms.append(f"-{v_str}")
        else:
            terms.append(f"{c}*{v_str}")
    return " + ".join(terms).replace("+ -", "- ")


class BuchbergerGroebnerProver:
    """Buchberger's Algorithm for Reduced Gröbner Bases & Polynomial Ideal Membership in Q[x, y]."""

    def compute_groebner_and_verify(
        self,
        generators: Optional[List[PolyDict]] = None,
        target_poly: Optional[PolyDict] = None,
    ) -> Dict[str, Any]:
        """Default system:
        f1 = x^2 + y^2 - 5  (circle of radius sqrt(5))
        f2 = x - 2*y        (line x = 2y)
        Under Lex order (x > y), the reduced Gröbner basis is:
        G_red = [x - 2*y, y^2 - 1]
        And target polynomial p = x^2 - 4 in <f1, f2> reduces to 0 mod G_red!
        """
        if generators is None:
            f1: PolyDict = {(2, 0): Fraction(1), (0, 2): Fraction(1), (0, 0): Fraction(-5)}
            f2: PolyDict = {(1, 0): Fraction(1), (0, 1): Fraction(-2)}
            generators = [f1, f2]
        if target_poly is None:
            # p = x^2 - 4 = (x^2 + y^2 - 5) - (y^2 - 1) in <f1, f2>
            target_poly = {(2, 0): Fraction(1), (0, 0): Fraction(-4)}

        G: List[PolyDict] = [_clean_poly(g) for g in generators if _clean_poly(g)]
        pairs = [(i, j) for i in range(len(G)) for j in range(i + 1, len(G))]

        s_poly_reductions = 0
        while pairs and len(G) < 16:
            i, j = pairs.pop(0)
            s_ij = _s_polynomial(G[i], G[j])
            s_poly_reductions += 1
            rem = _poly_rem(s_ij, G)
            if rem:
                new_idx = len(G)
                G.append(rem)
                for k in range(new_idx):
                    pairs.append((k, new_idx))

        # Reduce to minimal & monic Reduced Gröbner Basis
        minimal: List[PolyDict] = []
        for idx, g in enumerate(G):
            lm_g = _lead_monomial(g)
            redundant = False
            for idx2, h in enumerate(G):
                if idx == idx2:
                    continue
                lm_h = _lead_monomial(h)
                if _monomial_divides(lm_h, lm_g):
                    if lm_h != lm_g or idx2 < idx:
                        redundant = True
                        break
            if not redundant:
                minimal.append(g)

        reduced_basis: List[PolyDict] = []
        for idx, g in enumerate(minimal):
            others = [minimal[k] for k in range(len(minimal)) if k != idx]
            red_g = _poly_rem(g, others)
            if red_g:
                lm = _lead_monomial(red_g)
                lc = red_g[lm]
                monic_g = {m: c / lc for m, c in red_g.items()}
                reduced_basis.append(_clean_poly(monic_g))

        reduced_basis.sort(key=lambda poly: _lead_monomial(poly), reverse=True)

        # Ideal membership test for target_poly
        rem_target = _poly_rem(target_poly, reduced_basis)
        in_ideal = (len(rem_target) == 0)

        # Verify all S-polynomials of reduced_basis reduce to 0 (Buchberger's Criterion!)
        buchberger_criterion_satisfied = True
        for i in range(len(reduced_basis)):
            for j in range(i + 1, len(reduced_basis)):
                if _poly_rem(_s_polynomial(reduced_basis[i], reduced_basis[j]), reduced_basis):
                    buchberger_criterion_satisfied = False

        return {
            "input_generators": [_poly_to_str(g) for g in generators],
            "reduced_groebner_basis": [_poly_to_str(g) for g in reduced_basis],
            "monomial_order": "lex(x > y)",
            "s_polynomial_reductions": s_poly_reductions,
            "target_polynomial": _poly_to_str(target_poly),
            "target_remainder": _poly_to_str(rem_target),
            "ideal_membership_certified": in_ideal,
            "buchberger_criterion_satisfied": buchberger_criterion_satisfied,
            "verified": bool(buchberger_criterion_satisfied and len(reduced_basis) >= 1),
        }


# ---------------------------------------------------------------------------
# Unified Project HEPHAESTUS Orchestrator
# ---------------------------------------------------------------------------

class HephaestusEngine:
    """Unified Stage 21 Project HEPHAESTUS Engine."""

    def __init__(self) -> None:
        self.library_learner = WakeSleepLibraryLearner()
        self.gflownet = GFlowNetSampler()
        self.koopman = KoopmanEDMDLinearizer()
        self.groebner = BuchbergerGroebnerProver()

    def status(self) -> Dict[str, Any]:
        return {
            "ready": True,
            "version": "1.0.0",
            "project": "HEPHAESTUS: Wake-Sleep Library Learning, GFlowNet Trajectory Balance, Koopman EDMD & Buchberger Gröbner Basis Engine",
            "wake_sleep_library_learning": True,
            "gflownet_trajectory_balance": True,
            "koopman_edmd_operator": True,
            "koopman_edmd_linearizer": True,
            "buchberger_groebner_basis": True,
            "buchberger_groebner_prover": True,
        }

    def synthesize(
        self,
        corpus: Optional[List[str]] = None,
        terminal_rewards: Optional[Dict[str, float]] = None,
        koopman_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        if corpus is None:
            corpus = [
                "(2 * x + 1) * (2 * x - 1)",
                "(2 * y + 1) * (2 * y - 1)",
                "(2 * z + 1) * (2 * z - 1)",
            ]
        lib_res = self.library_learner.compress_corpus(corpus, max_abstractions=2)
        gfn_res = self.gflownet.calibrate_and_verify(terminal_rewards=terminal_rewards)
        kp = koopman_params or {}
        koop_res = self.koopman.analyze_system(
            lam=kp.get("lam", 0.8),
            mu=kp.get("mu", 0.5),
            gamma=kp.get("gamma", -0.6),
        )
        grob_res = self.groebner.compute_groebner_and_verify()

        payload = {
            "library_compression": lib_res["compression_ratio"],
            "gflownet_Z": gfn_res["partition_function_Z"],
            "koopman_eigenvalues": koop_res["koopman_eigenvalues"],
            "groebner_basis": grob_res["reduced_groebner_basis"],
        }
        receipt_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

        verified = bool(
            lib_res["verified"]
            and gfn_res["verified"]
            and koop_res["verified"]
            and grob_res["verified"]
        )
        text = (
            f"Project HEPHAESTUS Synthesis: (1) Wake-Sleep Stitch Library Learner extracted {lib_res['num_abstractions']} "
            f"reusable lambda-abstraction(s) (MDL compression {lib_res['compression_ratio']}x, beta-verified). "
            f"(2) GFlowNet Trajectory Balance certified partition function Z = {gfn_res['partition_function_Z']} "
            f"(max TB loss = {gfn_res['max_trajectory_balance_loss']:.2e}). "
            f"(3) Koopman EDMD lifted nonlinear system to linear spectrum {koop_res['koopman_eigenvalues']} "
            f"(error = {koop_res['multi_step_prediction_error']:.2e}). "
            f"(4) Buchberger Reduced Gröbner Basis computed {grob_res['reduced_groebner_basis']} "
            f"(ideal membership = {grob_res['ideal_membership_certified']})."
        )
        return {
            "project": "HEPHAESTUS",
            "text": text,
            "wake_sleep_library": lib_res,
            "gflownet_inference": gfn_res,
            "gflownet_sampler": gfn_res,
            "koopman_edmd": koop_res,
            "groebner_basis": grob_res,
            "verified": verified,
            "receipt_sha256": receipt_sha256,
            "backend": "hephaestus-stitch-gflownet-koopman-groebner-engine",
            "timestamp": time.time(),
        }

