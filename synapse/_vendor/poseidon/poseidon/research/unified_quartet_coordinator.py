"""Opt-in evidence receiver for FAULTLINE, TETHER, and CROSSWIRE.

This module does not collect calibration data, certify its provenance, or activate
a controller. Callers fit a panel on historical nonnegative member errors, then
provide candidate-action predictions, a margin, a calibrated risk estimate and
its identity, and an actual discrete interface state. Missing evidence preserves
the baseline. The TETHER trust score is a heuristic, not a calibrated probability.
No successor-value or Lyapunov-stability evidence is inferred from an observation.

Policy actions remain internal codes. An admitted candidate's external interface
code is reported separately; an abstained baseline has no certified dispatch.
SHA-256 receipts detect changed contents, not fabricated evidence or unsafe models.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Real
from typing import Any, Mapping, Sequence

from ..crosswire_engine import (
    CandidateMapping, CandidateSpace, ClosedLoopRuntimeShield,
    CrosswireEngine, VersionSpaceTracker,
)
from ..faultline_engine import ContinuousJointErrorEngine
from ..tether_engine import EvidenceCalibratedFallbackGate, TetherContinuousReliabilityEngine


def _digest(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _integer(value: Any, label: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    # Keep downstream sum-of-squares arithmetic finite as well as the inputs.
    if not math.isfinite(result) or abs(result) > 1e100:
        raise ValueError(f"{label} must be finite with magnitude <= 1e100")
    return result


def _rows(value: Any, columns: int, label: str) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 100000:
        raise ValueError(f"{label} must contain 1..100000 rows")
    result = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != columns:
            raise ValueError(f"{label} rows must have {columns} columns")
        result.append([_number(item, label) for item in row])
    return result


@dataclass
class QuartetCognitiveDecision:
    """Policy decision plus a content fingerprint; unknown quantities are None."""
    action: int
    base_action: int
    intended_action: int
    override_admitted: bool
    epistemic_trust: float | None
    active_panel: list[int]
    shield_verdict: str
    successor_value: float | None
    lyapunov_stable: bool | None
    certificate_sha256: str
    telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UnifiedQuartetCoordinator:
    """Compose supplied evidence without changing the frozen/default policy.

    ``fit_panel`` is a separate selection operation; evaluating decisions never
    refits the panel. Its error matrices must precede evaluation and contain one
    nonnegative error magnitude per member, row, and condition. This contract is
    caller-declared; this receiver cannot prove held-out data or risk calibration.
    The default interface is the four-action synthetic CROSSWIRE model, even when
    the policy has six actions. Unsupported actions abstain instead of aliasing.
    """

    def __init__(
        self, num_actions: int = 6, num_oracle_members: int = 5,
        panel_size: int = 3, min_trust_threshold: float = 0.60, seed: int = 42,
        *, transition_model: Mapping[tuple[int, int], int] | None = None,
        hypotheses: Sequence[CandidateMapping] | None = None,
        hazard_states: Sequence[int] | None = None,
        margin_threshold: float = 0.15, max_admissible_risk: float = 0.40,
    ):
        self.num_actions = _integer(num_actions, "num_actions", 2, 7)
        self.num_oracle_members = _integer(num_oracle_members, "num_oracle_members", 2, 12)
        self.panel_size = _integer(panel_size, "panel_size", 2, self.num_oracle_members)
        self.min_trust = _number(min_trust_threshold, "min_trust_threshold")
        self.margin_threshold = _number(margin_threshold, "margin_threshold")
        self.max_risk = _number(max_admissible_risk, "max_admissible_risk")
        if not 0 <= self.min_trust <= 1 or not 0 <= self.max_risk <= 1 or self.margin_threshold < 0:
            raise ValueError("trust/risk thresholds must be in [0, 1]; margin threshold must be nonnegative")
        self.seed = _integer(seed, "seed", 0, 2**63 - 1)
        if hypotheses is not None and (not isinstance(hypotheses, (list, tuple)) or
                any(not isinstance(hypothesis, CandidateMapping) for hypothesis in hypotheses)):
            raise ValueError("hypotheses must be a sequence of CandidateMapping objects")
        self.crosswire_k = (len(hypotheses[0].permutation) if hypotheses else
                            self.num_actions if transition_model is not None else min(self.num_actions, 4))
        _integer(self.crosswire_k, "interface action count", 2, self.num_actions)
        self.transition_model = dict(transition_model) if transition_model is not None else {
            key: value for key, value in CrosswireEngine._create_canonical_transition_model(self.crosswire_k).items()
            if key[1] < self.crosswire_k
        }
        for key, value in self.transition_model.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("transition keys must be (state, internal_action) pairs")
            _integer(key[0], "transition state", 0, 2**31 - 1)
            _integer(key[1], "transition action", 0, self.crosswire_k - 1)
            _integer(value, "transition successor", 0, 2**31 - 1)
        hazards = {4} if hazard_states is None else set(hazard_states)
        for state in hazards:
            _integer(state, "hazard state", 0, 2**31 - 1)
        self.hazard_states = hazards
        self.all_hypotheses = (list(hypotheses) if hypotheses is not None else
                               CandidateSpace.generate_permutations(self.crosswire_k))
        self._validate_hypotheses(self.all_hypotheses)
        self.tracker = VersionSpaceTracker(self.all_hypotheses, transition_model=self.transition_model)
        self.shield = ClosedLoopRuntimeShield(fallback_action_int=0, hazard_states=hazards,
                                             transition_model=self.transition_model)
        self.tether = TetherContinuousReliabilityEngine()
        self.tether_gate = EvidenceCalibratedFallbackGate(
            epistemic_threshold=self.min_trust, margin_threshold=self.margin_threshold,
            max_admissible_risk=self.max_risk)
        self._panel_fit: dict[str, Any] | None = None

    @property
    def active_panel(self) -> list[int]:
        return [] if self._panel_fit is None else list(self._panel_fit["selected_panel"])

    def _validate_hypotheses(self, hypotheses: Sequence[CandidateMapping]) -> None:
        if len(hypotheses) > 5040:
            raise ValueError("too many interface hypotheses")
        ids = set()
        for hypothesis in hypotheses:
            if (not isinstance(hypothesis, CandidateMapping) or
                    not isinstance(hypothesis.mapping_id, str) or not hypothesis.mapping_id or
                    hypothesis.mapping_id in ids or
                    len(hypothesis.permutation) != self.crosswire_k or
                    any(type(action) is not int for action in hypothesis.permutation) or
                    set(hypothesis.permutation) != set(range(self.crosswire_k))):
                raise ValueError("hypotheses require unique identities and complete action permutations")
            ids.add(hypothesis.mapping_id)

    def fit_panel(self, member_errors_by_condition: Mapping[str, Sequence[Sequence[float]]]) -> dict[str, Any]:
        """Freeze minimax panel selection from caller-supplied historical errors.

        No demonstration errors are substituted. Explicit refitting replaces the
        fit only after all validation and selection succeed. The full fit data is
        fingerprinted for audit; it does not establish out-of-sample performance.
        """
        if not isinstance(member_errors_by_condition, Mapping) or not 1 <= len(member_errors_by_condition) <= 64:
            raise ValueError("provide historical errors for 1..64 conditions")
        errors = {}
        for condition, rows in member_errors_by_condition.items():
            if not isinstance(condition, str) or not condition:
                raise ValueError("condition names must be nonempty strings")
            matrix = _rows(rows, self.num_oracle_members, "historical member errors")
            if any(value < 0 for row in matrix for value in row):
                raise ValueError("historical errors must be nonnegative magnitudes")
            errors[condition] = matrix
        errors = dict(sorted(errors.items()))
        selection = ContinuousJointErrorEngine.select_minimax_continuous_panel(errors, size=self.panel_size)
        fit = {"schema": "poseidon-quartet-panel-fit-v1", "member_errors_by_condition": errors,
               "selected_panel": selection["selected_panel"],
               "worst_condition_error": selection["worst_condition_error"],
               "pooled_error": selection["pooled_error"], "scope": "historical_selection_only"}
        fit["fit_sha256"] = _digest(fit)
        self._panel_fit = fit
        return deepcopy(fit)

    def evaluate_step(
        self, observation_16d: Sequence[float], base_action: int, intended_action: int,
        hazard_action: int = 5, member_errors: Sequence[float] | None = None,
        *, current_state: int | None = None,
        member_predictions: Sequence[Sequence[float]] | None = None,
        candidate_margin: float | None = None, calibrated_risk: float | None = None,
        risk_calibration_id: str | None = None,
    ) -> QuartetCognitiveDecision:
        """Evaluate supplied candidate evidence, retaining baseline on abstention.

        Predictions must be the candidate action's 16-dimensional forecasts in
        member order. ``calibrated_risk`` and its identity are caller declarations;
        this module validates numbers and fingerprints, not the calibration study.
        Legacy ``member_errors`` cannot fit on evaluation outcomes; use fit_panel.
        Legacy ``hazard_action`` is recorded but is not a substitute for a model.
        """
        obs = _rows([list(observation_16d)], 16, "observation")[0]
        if any(not 0 <= value <= 1 for value in obs):
            raise ValueError("observation values must be in [0, 1]")
        _integer(base_action, "base_action", 0, self.num_actions - 1)
        _integer(intended_action, "intended_action", 0, self.num_actions - 1)
        _integer(hazard_action, "legacy hazard_action", 0, 2**31 - 1)
        if member_errors is not None:
            raise ValueError("per-decision member_errors cannot fit a panel; use fit_panel on historical data")
        if current_state is not None:
            _integer(current_state, "current_state", 0, 2**31 - 1)
        if candidate_margin is not None:
            candidate_margin = _number(candidate_margin, "candidate_margin")
        if calibrated_risk is not None:
            calibrated_risk = _number(calibrated_risk, "calibrated_risk")
            if not 0 <= calibrated_risk <= 1:
                raise ValueError("calibrated_risk must be in [0, 1]")
        if risk_calibration_id is not None and (not isinstance(risk_calibration_id, str) or not risk_calibration_id.strip()):
            raise ValueError("risk_calibration_id must be a nonempty string")
        predictions = None
        if member_predictions is not None:
            predictions = _rows(member_predictions, 16, "member predictions")
            if len(predictions) != self.num_oracle_members:
                raise ValueError("one prediction is required for every configured member")
        surviving = list(self.tracker.surviving)
        self._validate_hypotheses(surviving)
        reasons = []
        for value, label in ((current_state, "current_state"), (predictions, "member_predictions"),
                             (candidate_margin, "candidate_margin"), (calibrated_risk, "calibrated_risk"),
                             (risk_calibration_id, "risk_calibration_id")):
            if value is None:
                reasons.append(f"missing_{label}")
        if self._panel_fit is None:
            reasons.append("panel_not_fitted")

        trust = None
        reliability = None
        if predictions is not None and self._panel_fit is not None:
            reliability = self.tether.evaluate_reliability([predictions[index] for index in self.active_panel])
            trust = reliability["epistemic_trust"]
            _digest(reliability)  # No nonfinite computed evidence may reach the gate.
            if trust < self.min_trust:
                reasons.append("insufficient_epistemic_trust")
        if candidate_margin is not None and candidate_margin <= self.margin_threshold:
            reasons.append("insufficient_candidate_margin")
        if calibrated_risk is not None and calibrated_risk > self.max_risk:
            reasons.append("risk_threshold_exceeded")

        shield_result = None
        shield_verdict = "not_evaluated"
        if intended_action >= self.crosswire_k:
            reasons.append("unsupported_interface_action")
        elif not surviving:
            reasons.append("empty_version_space")
        elif current_state is not None:
            # CROSSWIRE treats unknown transitions as self-loops. Require complete
            # action support at the supplied state before consulting that shield.
            if any((current_state, action) not in self.transition_model for action in range(self.crosswire_k)):
                reasons.append("incomplete_transition_model")
            elif current_state in self.hazard_states:
                reasons.append("current_state_is_hazard")
            else:
                shield_result = self.shield.evaluate_and_filter(
                    current_state=current_state, intended_action_int=intended_action,
                    version_space=surviving)
                shield_verdict = shield_result["decision"]
                if shield_verdict != "certified_admit":
                    reasons.append("shield_rejected_candidate")

        gate_result = None
        admitted = False
        if not reasons:
            gate_result = self.tether_gate.evaluate_override(
                base_action=base_action, candidate_action=intended_action,
                candidate_margin=candidate_margin, epistemic_trust=trust,
                calibrated_risk=calibrated_risk)
            admitted = gate_result["admit_override"] and intended_action != base_action
        final_action = intended_action if admitted else base_action
        telemetry = {
            "schema": "poseidon-quartet-decision-v2", "receipt_scope": "fingerprint_only",
            "action_space": "internal_policy_codes", "observation_16d": obs,
            "current_state": current_state, "member_predictions": predictions,
            "candidate_margin": candidate_margin, "calibrated_risk": calibrated_risk,
            "risk_calibration_id": risk_calibration_id,
            "risk_calibration_verified": False, "trust_scope": "heuristic_member_agreement",
            "legacy_hazard_action": hazard_action, "legacy_hazard_action_used": False,
            "panel_fit": deepcopy(self._panel_fit),
            "panel_fit_sha256": self._panel_fit["fit_sha256"] if self._panel_fit else None,
            "reliability": reliability, "shield": shield_result, "gate": gate_result,
            "abstention_reasons": reasons,
            "dispatched_action_ext": shield_result["dispatched_action_ext"] if admitted else None,
            "config": {"num_actions": self.num_actions, "num_oracle_members": self.num_oracle_members,
                       "panel_size": self.panel_size, "min_trust_threshold": self.min_trust,
                       "margin_threshold": self.margin_threshold, "max_admissible_risk": self.max_risk,
                       "seed": self.seed, "hazard_states": sorted(self.hazard_states),
                       "transition_model": [[s, a, nxt] for (s, a), nxt in sorted(self.transition_model.items())],
                       "surviving_hypotheses": [hypothesis.to_dict() for hypothesis in surviving]},
        }
        decision = QuartetCognitiveDecision(
            action=final_action, base_action=base_action, intended_action=intended_action,
            override_admitted=admitted, epistemic_trust=trust, active_panel=self.active_panel,
            shield_verdict=shield_verdict, successor_value=None, lyapunov_stable=None,
            certificate_sha256="", telemetry=telemetry)
        payload = decision.to_dict()
        payload.pop("certificate_sha256")
        decision.certificate_sha256 = _digest(payload)
        return decision

    @staticmethod
    def verify_decision_integrity(decision: QuartetCognitiveDecision | Mapping[str, Any]) -> bool:
        """Check content integrity only; this is neither replay nor authentication."""
        try:
            payload = decision.to_dict() if isinstance(decision, QuartetCognitiveDecision) else dict(decision)
            expected = payload.pop("certificate_sha256")
            return isinstance(expected, str) and _digest(payload) == expected
        except (ValueError, TypeError, KeyError, OverflowError):
            return False
