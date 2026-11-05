"""PROJECT OLYMPUS (Stage 18): Scientific Hypothesis & Falsification Engine.

Operationalizes Karl Popper's empirical falsifiability demarcation and Imre Lakatos's
Methodology of Scientific Research Programmes (MSRP) with:
1. Popperian Falsifiable Conjectures: explicit refutation conditions and deductive consequences.
2. Lakatosian Protective Belt vs Hard Core: auxiliary hypothesis tracking and progressive
   vs degenerating theoretical problem-shift auditing.
3. Baconian Crucial Experiments: discriminative testing between rival scientific hypotheses
   with expected Shannon information gain (ebits).
4. Immutable Cryptographic Falsification Receipts: tamper-evident SHA-256 receipts certifying
   refutation or severe corroboration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ScientificHypothesis:
    """A formal scientific conjecture equipped with hard core and protective belt."""
    id: str
    conjecture: str
    domain: str
    hard_core_assumptions: List[str]
    protective_belt: List[str]
    falsification_criteria: List[str]
    novel_predictions: List[str]
    epistemic_confidence: float = 0.5  # Prior P(H) in (0, 1)
    status: str = "PROPOSED"  # PROPOSED, CORROBORATED, FALSIFIED, ANOMALOUS_PROTECTED
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CrucialExperiment:
    """A Baconian/Popperian decisive experiment designed to discriminate rival hypotheses."""
    id: str
    name: str
    description: str
    hypothesis_a_id: str
    hypothesis_b_id: str
    discriminator_variable: str
    prediction_a: Dict[str, Any]  # e.g. {"range": [a_min, a_max], "point": val}
    prediction_b: Dict[str, Any]
    severity_score: float = 0.85  # Popper severity S in [0, 1]
    expected_shannon_information_bits: float = 1.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HypothesisFalsificationReceipt:
    """Cryptographic, tamper-evident receipt certifying a falsification or corroboration event."""
    receipt_id: str
    hypothesis_id: str
    experiment_id: str
    outcome_status: str  # CORROBORATED, FALSIFIED, ANOMALOUS
    observed_metric: str
    observed_value: Any
    bayes_factor: float
    hash_sha256: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def issue(
        cls,
        hypothesis_id: str,
        experiment_id: str,
        outcome_status: str,
        observed_metric: str,
        observed_value: Any,
        bayes_factor: float,
    ) -> HypothesisFalsificationReceipt:
        now = time.time()
        receipt_id = f"rcpt-{hashlib.sha256(f'{hypothesis_id}:{experiment_id}:{now}'.encode()).hexdigest()[:12]}"
        payload = {
            "receipt_id": receipt_id,
            "hypothesis_id": hypothesis_id,
            "experiment_id": experiment_id,
            "outcome_status": outcome_status,
            "observed_metric": observed_metric,
            "observed_value": str(observed_value),
            "bayes_factor": round(float(bayes_factor), 6),
            "timestamp": now,
        }
        canonical = json.dumps(payload, sort_keys=True)
        sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(
            receipt_id=receipt_id,
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
            outcome_status=outcome_status,
            observed_metric=observed_metric,
            observed_value=observed_value,
            bayes_factor=round(float(bayes_factor), 6),
            hash_sha256=sha,
            timestamp=now,
        )

    def verify(self) -> bool:
        """Verifies integrity of the cryptographic receipt."""
        payload = {
            "receipt_id": self.receipt_id,
            "hypothesis_id": self.hypothesis_id,
            "experiment_id": self.experiment_id,
            "outcome_status": self.outcome_status,
            "observed_metric": self.observed_metric,
            "observed_value": str(self.observed_value),
            "bayes_factor": round(float(self.bayes_factor), 6),
            "timestamp": self.timestamp,
        }
        canonical = json.dumps(payload, sort_keys=True)
        recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return recomputed == self.hash_sha256


@dataclass
class FalsificationOutcome:
    """The complete result of evaluating a crucial experiment against a hypothesis."""
    outcome_id: str
    hypothesis_id: str
    experiment_id: str
    observed_metric: str
    observed_value: Any
    falsification_triggered: bool
    decisive: bool
    bayes_factor: float
    posterior_confidence: float
    lakatos_assessment: str  # PROGRESSIVE, DEGENERATING, or REFUTED
    explanation: str
    receipt: HypothesisFalsificationReceipt

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["receipt"] = self.receipt.to_dict()
        return d


# ---------------------------------------------------------------------------
# Project OLYMPUS Hypothesis & Falsification Engine
# ---------------------------------------------------------------------------

class HypothesisEngine:
    """Autonomous Scientific Hypothesis Formulation, Crucial Experimentation, and Falsification Engine."""

    def __init__(self) -> None:
        self.hypotheses: Dict[str, ScientificHypothesis] = {}
        self.experiments: Dict[str, CrucialExperiment] = {}
        self.outcomes: List[FalsificationOutcome] = []

    def formulate_hypothesis(
        self,
        conjecture: str,
        domain: str,
        hard_core_assumptions: Sequence[str],
        protective_belt: Sequence[str],
        falsification_criteria: Sequence[str],
        novel_predictions: Sequence[str],
        prior: float = 0.5,
    ) -> ScientificHypothesis:
        """Formulates a new falsifiable scientific hypothesis with Lakatosian research programme structure."""
        if not conjecture or not conjecture.strip():
            raise ValueError("Conjecture cannot be empty.")
        if not falsification_criteria:
            raise ValueError("Popperian demarcation requires at least one explicit falsification criterion.")

        content_seed = f"{domain}:{conjecture}:{time.time()}"
        hyp_id = f"hyp-{hashlib.sha256(content_seed.encode()).hexdigest()[:10]}"
        hyp = ScientificHypothesis(
            id=hyp_id,
            conjecture=conjecture.strip(),
            domain=domain.strip().lower(),
            hard_core_assumptions=list(hard_core_assumptions),
            protective_belt=list(protective_belt),
            falsification_criteria=list(falsification_criteria),
            novel_predictions=list(novel_predictions),
            epistemic_confidence=max(0.01, min(0.99, float(prior))),
            status="PROPOSED",
        )
        self.hypotheses[hyp_id] = hyp
        return hyp

    def design_crucial_experiment(
        self,
        hypothesis_a: ScientificHypothesis,
        hypothesis_b: ScientificHypothesis,
        name: str,
        description: str,
        discriminator_variable: str,
        prediction_a: Dict[str, Any],
        prediction_b: Dict[str, Any],
        severity_score: float = 0.85,
    ) -> CrucialExperiment:
        """Designs a crucial experiment that differentiates two competing hypotheses."""
        # Calculate expected Shannon mutual information gain in bits:
        # I(E; H) = H(H) - H(H|E)
        p = hypothesis_a.epistemic_confidence
        entropy_before = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)) if 0.0 < p < 1.0 else 0.0
        # Under severe discriminator, expected posterior certainty increases proportionally to severity
        severity = max(0.1, min(0.99, float(severity_score)))
        expected_gain = entropy_before * severity

        exp_id = f"exp-{hashlib.sha256(f'{hypothesis_a.id}:{hypothesis_b.id}:{name}'.encode()).hexdigest()[:10]}"
        exp = CrucialExperiment(
            id=exp_id,
            name=name.strip(),
            description=description.strip(),
            hypothesis_a_id=hypothesis_a.id,
            hypothesis_b_id=hypothesis_b.id,
            discriminator_variable=discriminator_variable.strip(),
            prediction_a=prediction_a,
            prediction_b=prediction_b,
            severity_score=severity,
            expected_shannon_information_bits=round(expected_gain, 4),
        )
        self.experiments[exp_id] = exp
        return exp

    def conduct_test(
        self,
        experiment: CrucialExperiment,
        observed_data: Dict[str, Any],
        target_hypothesis: Optional[ScientificHypothesis] = None,
    ) -> FalsificationOutcome:
        """Executes the decisive empirical evaluation against the hypothesis."""
        hyp = target_hypothesis or self.hypotheses.get(experiment.hypothesis_a_id)
        if not hyp:
            raise ValueError(f"Target hypothesis not found for experiment {experiment.id}.")

        var = experiment.discriminator_variable
        if var not in observed_data:
            raise KeyError(f"Observed data missing discriminator variable '{var}'.")

        observed_val = observed_data[var]
        pred_a = experiment.prediction_a
        pred_b = experiment.prediction_b

        # Evaluation logic: Check if observation matches prediction_a vs prediction_b
        matches_a = self._check_prediction_match(observed_val, pred_a)
        matches_b = self._check_prediction_match(observed_val, pred_b)

        # Falsification check
        if matches_a and not matches_b:
            falsified = False
            decisive = True
            bayes_factor = 1.0 / (1.0 - experiment.severity_score + 1e-6)
            outcome_status = "CORROBORATED"
            lakatos = "PROGRESSIVE"
            explanation = (
                f"Observation {var}={observed_val} strictly corroborated {hyp.id}'s prediction "
                f"while decisively refuting rival {experiment.hypothesis_b_id}."
            )
            hyp.status = "CORROBORATED"
        elif not matches_a and matches_b:
            falsified = True
            decisive = True
            bayes_factor = (1.0 - experiment.severity_score + 1e-6)
            outcome_status = "FALSIFIED"
            lakatos = "REFUTED"
            explanation = (
                f"Observation {var}={observed_val} strictly refutes {hyp.id}'s prediction "
                f"and confirms rival {experiment.hypothesis_b_id}."
            )
            hyp.status = "FALSIFIED"
        elif not matches_a and not matches_b:
            # Anomalous to both: check if protective belt can absorb it
            falsified = True
            decisive = False
            bayes_factor = 0.5
            outcome_status = "ANOMALOUS"
            lakatos = "DEGENERATING" if not hyp.protective_belt else "ANOMALOUS_PROTECTED"
            explanation = (
                f"Observation {var}={observed_val} is anomalous to both rival hypotheses. "
                f"Requires protective belt revision."
            )
            hyp.status = "ANOMALOUS_PROTECTED" if hyp.protective_belt else "FALSIFIED"
        else:
            # Matches both: inconclusive discriminator
            falsified = False
            decisive = False
            bayes_factor = 1.0
            outcome_status = "INCONCLUSIVE"
            lakatos = "PROGRESSIVE"
            explanation = f"Observation {var}={observed_val} satisfies both predictions; test is non-discriminative."

        # Bayesian update on epistemic confidence:
        prior = hyp.epistemic_confidence
        prior_odds = prior / (1.0 - prior)
        posterior_odds = prior_odds * bayes_factor
        posterior_conf = posterior_odds / (1.0 + posterior_odds)
        posterior_conf = max(0.001, min(0.999, posterior_conf))
        hyp.epistemic_confidence = round(posterior_conf, 4)

        receipt = HypothesisFalsificationReceipt.issue(
            hypothesis_id=hyp.id,
            experiment_id=experiment.id,
            outcome_status=outcome_status,
            observed_metric=var,
            observed_value=observed_val,
            bayes_factor=bayes_factor,
        )

        outcome_id = f"out-{hashlib.sha256(f'{hyp.id}:{experiment.id}:{time.time()}'.encode()).hexdigest()[:10]}"
        outcome = FalsificationOutcome(
            outcome_id=outcome_id,
            hypothesis_id=hyp.id,
            experiment_id=experiment.id,
            observed_metric=var,
            observed_value=observed_val,
            falsification_triggered=falsified,
            decisive=decisive,
            bayes_factor=round(bayes_factor, 6),
            posterior_confidence=round(posterior_conf, 4),
            lakatos_assessment=lakatos,
            explanation=explanation,
            receipt=receipt,
        )
        self.outcomes.append(outcome)
        return outcome

    def _check_prediction_match(self, observed: Any, prediction: Dict[str, Any]) -> bool:
        """Determines if an observed value satisfies a prediction specification."""
        if "point" in prediction:
            pt = prediction["point"]
            tol = prediction.get("tolerance", 1e-4)
            if isinstance(observed, (int, float)) and isinstance(pt, (int, float)):
                return abs(float(observed) - float(pt)) <= tol
            return bool(observed == pt)
        if "range" in prediction:
            low, high = prediction["range"]
            if isinstance(observed, (int, float)):
                return low <= float(observed) <= high
        if "exact" in prediction:
            return observed == prediction["exact"]
        if "predicate" in prediction:
            pred_type = prediction["predicate"]
            if pred_type == "positive":
                return float(observed) > 0
            if pred_type == "negative":
                return float(observed) < 0
            if pred_type == "zero":
                return abs(float(observed)) < 1e-6
        return False

    def evaluate_research_programme(
        self,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluates Lakatosian research programme health across hypotheses and empirical outcomes."""
        relevant_hyps = [
            h for h in self.hypotheses.values()
            if domain is None or h.domain == domain.strip().lower()
        ]
        relevant_outcomes = [
            o for o in self.outcomes
            if o.hypothesis_id in {h.id for h in relevant_hyps}
        ]

        total_tested = len(relevant_outcomes)
        corroborated_count = sum(1 for o in relevant_outcomes if o.receipt.outcome_status == "CORROBORATED")
        falsified_count = sum(1 for o in relevant_outcomes if o.receipt.outcome_status == "FALSIFIED")
        anomalous_count = sum(1 for o in relevant_outcomes if o.receipt.outcome_status == "ANOMALOUS")

        progressive_shifts = sum(1 for o in relevant_outcomes if o.lakatos_assessment == "PROGRESSIVE")
        degenerating_shifts = sum(1 for o in relevant_outcomes if o.lakatos_assessment in {"DEGENERATING", "REFUTED"})

        is_progressive = progressive_shifts >= degenerating_shifts and corroborated_count > 0

        novel_prediction_count = sum(len(h.novel_predictions) for h in relevant_hyps)
        average_confidence = (
            sum(h.epistemic_confidence for h in relevant_hyps) / len(relevant_hyps)
            if relevant_hyps else 0.5
        )

        return {
            "domain": domain or "all",
            "total_hypotheses": len(relevant_hyps),
            "total_experiments_conducted": total_tested,
            "corroborations": corroborated_count,
            "falsifications": falsified_count,
            "anomalies": anomalous_count,
            "programme_status": "THEORETICALLY_AND_EMPIRICALLY_PROGRESSIVE" if is_progressive else "DEGENERATING_PROBLEM_SHIFT",
            "novel_predictions_generated": novel_prediction_count,
            "average_epistemic_confidence": round(average_confidence, 4),
            "progressive_ratio": round(progressive_shifts / max(1, total_tested), 4),
        }

    def generate_paradigm_synthesis(self, domain: str) -> Dict[str, Any]:
        """Synthesizes verified knowledge into a scientific paradigm report."""
        eval_metrics = self.evaluate_research_programme(domain)
        domain_hyps = [h for h in self.hypotheses.values() if h.domain == domain.strip().lower()]

        corroborated_principles = [
            {"id": h.id, "conjecture": h.conjecture, "confidence": h.epistemic_confidence}
            for h in domain_hyps if h.status == "CORROBORATED"
        ]
        refuted_conjectures = [
            {"id": h.id, "conjecture": h.conjecture, "confidence": h.epistemic_confidence}
            for h in domain_hyps if h.status == "FALSIFIED"
        ]
        open_frontiers = [
            {"id": h.id, "conjecture": h.conjecture, "novel_predictions": h.novel_predictions}
            for h in domain_hyps if h.status in {"PROPOSED", "ANOMALOUS_PROTECTED"}
        ]

        return {
            "domain": domain,
            "research_programme": eval_metrics["programme_status"],
            "corroborated_principles": corroborated_principles,
            "refuted_conjectures": refuted_conjectures,
            "open_frontiers": open_frontiers,
            "epistemic_integrity_receipts_verified": all(o.receipt.verify() for o in self.outcomes),
            "summary": (
                f"Scientific Paradigm for '{domain}': {len(corroborated_principles)} corroborated core principles, "
                f"{len(refuted_conjectures)} conclusively refuted conjectures, "
                f"programme health: {eval_metrics['programme_status']}."
            ),
        }
