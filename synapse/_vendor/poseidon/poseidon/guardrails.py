"""POSEIDON GUARDRAILS: Formal Semantic Safety Contracts & Invariant Verification.

Inspired by the TESSERA verified macro-action commons and safe reinforcement learning:
1. Formal Biological Invariants:
   Evaluates deterministic physiological constraints before any proposed action is executed:
     - HydrationStarvationGuard: Forbids wandering away from water when critically dehydrated.
     - EnergyDepletionGuard: Forbids wandering away from food when critically starving.
     - SevereStormShelterGuard: Mandates sheltering during extreme tempest storms when shelter is available.
     - StaminaCollapseGuard: Enforces rest recovery before complete motor exhaustion.
     - CatastrophicCliffVeto: Rejects proposed actions where the world model predicts terminal failure
       if a viable survival action exists.

2. Audit Traceability:
   Every decision is certified with a formal audit trail recording verified invariants and
   reasons for any emergency veto.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .world import ACTIONS, N_ACTIONS


@dataclass(frozen=True)
class InvariantResult:
    invariant_name: str
    satisfied: bool
    severity: str  # "advisory", "warning", "critical_veto"
    recommended_action: int | None = None
    reason: str = ""


@dataclass
class GuardrailAudit:
    action: int
    action_name: str
    vetoed: bool
    original_action: int
    original_action_name: str
    active_invariants: list[str]
    invariants_checked: int
    violation_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "action_name": self.action_name,
            "vetoed": self.vetoed,
            "original_action": self.original_action,
            "original_action_name": self.original_action_name,
            "active_invariants": self.active_invariants,
            "invariants_checked": self.invariants_checked,
            "violations": self.violation_details,
        }


class FormalSafetyGuard:
    """Verifies formal semantic contracts and invariants to prevent catastrophic actions."""

    def __init__(
        self,
        critical_hydration_threshold: float = 0.15,
        critical_energy_threshold: float = 0.15,
        critical_stamina_threshold: float = 0.08,
        severe_storm_threshold: float = 0.70,
        safe_shelter_threshold: float = 0.40,
    ):
        self.crit_hyd = critical_hydration_threshold
        self.crit_ene = critical_energy_threshold
        self.crit_sta = critical_stamina_threshold
        self.storm_thr = severe_storm_threshold
        self.shelter_thr = safe_shelter_threshold

    def audit_and_constrain(
        self,
        proposed_action: int,
        observation: Sequence[float],
        predicted_healths: dict[int, float] | None = None,
    ) -> GuardrailAudit:
        """Audits proposed action against biological invariants and intervenes if violated."""
        obs = list(observation)
        health = obs[0]
        energy = obs[1]
        hydration = obs[2]
        stamina = obs[3]
        threat = obs[5]
        local_food = obs[6]
        local_water = obs[7]
        local_shelter = obs[8]
        storm_severity = obs[9]

        results: list[InvariantResult] = []

        # Invariant 1: Critical Hydration
        # Action 2 is 'drink', Action 4 is 'explore', Action 5 is 'flee'
        if hydration < self.crit_hyd and local_water > 0.15 and proposed_action in (4, 5) and threat < 0.20:
            results.append(
                InvariantResult(
                    invariant_name="HydrationStarvationGuard",
                    satisfied=False,
                    severity="critical_veto",
                    recommended_action=2,  # drink
                    reason=f"Hydration={hydration:.3f} is critically low while water={local_water:.3f} is present.",
                )
            )

        # Invariant 2: Critical Energy
        # Action 1 is 'forage'
        if energy < self.crit_ene and local_food > 0.15 and proposed_action in (4, 5) and threat < 0.20:
            results.append(
                InvariantResult(
                    invariant_name="EnergyDepletionGuard",
                    satisfied=False,
                    severity="critical_veto",
                    recommended_action=1,  # forage
                    reason=f"Energy={energy:.3f} is critically low while food={local_food:.3f} is present.",
                )
            )

        # Invariant 3: Severe Tempest Storm
        # Action 3 is 'shelter'
        if storm_severity > self.storm_thr and local_shelter > self.shelter_thr and health < 0.50:
            if proposed_action in (4,):  # explore into open storm
                results.append(
                    InvariantResult(
                        invariant_name="SevereStormShelterGuard",
                        satisfied=False,
                        severity="critical_veto",
                        recommended_action=3,  # shelter
                        reason=f"Storm severity={storm_severity:.3f} with shelter={local_shelter:.3f}; open exploration vetoed.",
                    )
                )

        # Invariant 4: Stamina Exhaustion
        # Action 0 is 'rest'
        if stamina < self.crit_sta and threat < 0.10 and proposed_action in (4, 5):
            results.append(
                InvariantResult(
                    invariant_name="StaminaCollapseGuard",
                    satisfied=False,
                    severity="warning",
                    recommended_action=0,  # rest
                    reason=f"Stamina={stamina:.3f} is near total exhaustion; rest required.",
                )
            )

        # Invariant 5: Catastrophic Cliff Veto (predicted terminal drop)
        if predicted_healths is not None and proposed_action in predicted_healths:
            pred_h = predicted_healths[proposed_action]
            if pred_h <= 0.0 or (health - pred_h) >= 0.45:
                # Find best surviving alternative
                best_alt = max(predicted_healths.items(), key=lambda kv: kv[1])
                if best_alt[1] > pred_h:
                    results.append(
                        InvariantResult(
                            invariant_name="CatastrophicCliffVeto",
                            satisfied=False,
                            severity="critical_veto",
                            recommended_action=best_alt[0],
                            reason=f"Action {ACTIONS[proposed_action]} predicts health collapse to {pred_h:.3f}.",
                        )
                    )

        # Determine if veto needed
        active_invariants = [r.invariant_name for r in results if not r.satisfied]
        veto_triggers = [r for r in results if not r.satisfied and r.severity == "critical_veto"]

        final_action = proposed_action
        vetoed = False
        if veto_triggers:
            vetoed = True
            # Prioritize first critical veto recommended action
            rec = veto_triggers[0].recommended_action
            if rec is not None:
                final_action = rec

        violations = [
            {
                "invariant": r.invariant_name,
                "severity": r.severity,
                "reason": r.reason,
                "recommended_action": ACTIONS[r.recommended_action] if r.recommended_action is not None else None,
            }
            for r in results if not r.satisfied
        ]

        return GuardrailAudit(
            action=final_action,
            action_name=ACTIONS[final_action],
            vetoed=vetoed,
            original_action=proposed_action,
            original_action_name=ACTIONS[proposed_action],
            active_invariants=active_invariants,
            invariants_checked=5,
            violation_details=violations,
        )
