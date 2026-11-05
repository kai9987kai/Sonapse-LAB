"""Universal Modder Dynamic Bytecode & Hook Interceptor Engine.

Synthesizes dynamic function interception, hot-patchable trampoline hooks,
and runtime safety adapters from universal-modder and rea:
1. Non-destructive hook registry intercepting decision lifecycle events
   (PRE_OBSERVATION, POST_SUPERPOSITION, PRE_ACTION, POST_STEP).
2. Trampoline dispatch pattern allowing emergency heuristic overrides without
   modifying or retraining the frozen Tidal core weights.
3. Built-in ratified safety trampolines:
   - EmergencyStaminaGuard: overrides exhaustion-induced actions to restorative rest.
   - SevereStormShelterGuard: forces shelter seeking when storm exposure is lethal.
4. Cryptographic hook manifest and telemetry tracking overrides and execution integrity.
"""
from __future__ import annotations

import enum
import hashlib
from typing import Callable, Dict, List, Optional, Tuple


class HookPoint(enum.Enum):
    PRE_OBSERVATION = "pre_observation"
    POST_SUPERPOSITION = "post_superposition"
    PRE_ACTION = "pre_action"
    POST_STEP = "post_step"


class HookRecord:
    """Descriptor for a registered dynamic hook."""

    def __init__(
        self,
        hook_id: str,
        point: HookPoint,
        callback: Callable,
        priority: int = 100,
        description: str = "",
    ) -> None:
        self.hook_id = hook_id
        self.point = point
        self.callback = callback
        self.priority = priority
        self.description = description
        self.invocations: int = 0
        self.overrides: int = 0


class UniversalModderEngine:
    """Dynamic trampoline hook and bytecode interceptor."""

    def __init__(self) -> None:
        self.hooks: Dict[HookPoint, List[HookRecord]] = {p: [] for p in HookPoint}
        self.total_overrides: int = 0
        self.total_interceptions: int = 0
        self._install_default_guards()

    def reset(self) -> None:
        """Reset execution counters while preserving registered hooks."""
        self.total_overrides = 0
        self.total_interceptions = 0
        for hook_list in self.hooks.values():
            for h in hook_list:
                h.invocations = 0
                h.overrides = 0

    def register_hook(
        self,
        point: HookPoint,
        name: str,
        callback: Callable,
        priority: int = 100,
        description: str = "",
    ) -> str:
        """Register a new dynamic hook at the specified lifecycle point."""
        hook_id = hashlib.sha256(f"{point.value}:{name}:{priority}".encode()).hexdigest()[:12]
        record = HookRecord(hook_id, point, callback, priority, description)
        self.hooks[point].append(record)
        # Sort by priority ascending (lower integer = executed earlier)
        self.hooks[point].sort(key=lambda r: r.priority)
        return hook_id

    def unregister_hook(self, hook_id: str) -> bool:
        """Remove a hook by its unique identifier."""
        found = False
        for p in HookPoint:
            initial_len = len(self.hooks[p])
            self.hooks[p] = [h for h in self.hooks[p] if h.hook_id != hook_id]
            if len(self.hooks[p]) < initial_len:
                found = True
        return found

    def apply_pre_action_trampolines(
        self,
        proposed_action: int,
        observation: List[float],
        context: Dict[str, object],
    ) -> Tuple[int, Optional[str]]:
        """Execute registered PRE_ACTION hooks in priority order.

        Returns:
            (final_action, override_reason_or_None)
        """
        current_action = proposed_action
        applied_override: Optional[str] = None

        for record in self.hooks[HookPoint.PRE_ACTION]:
            self.total_interceptions += 1
            record.invocations += 1
            try:
                mod_action, reason = record.callback(current_action, observation, context)
                if mod_action != current_action:
                    current_action = mod_action
                    applied_override = reason or record.description
                    record.overrides += 1
                    self.total_overrides += 1
            except Exception:
                # Hooks must never crash the primary decision loop
                continue

        return (current_action, applied_override)

    def _install_default_guards(self) -> None:
        """Install ratified baseline safety trampolines."""
        # 1. Emergency Stamina Guard
        def emergency_stamina_guard(action: int, obs: List[float], ctx: Dict[str, object]):
            stamina = obs[3] if len(obs) > 3 else 1.0
            # If stamina is depleted (< 0.10) and action is costly movement, trampoline to rest
            if stamina < 0.10 and action in (1, 2, 4, 5):
                return (0, "modder:emergency_stamina_depletion_override")
            return (action, None)

        self.register_hook(
            HookPoint.PRE_ACTION,
            "emergency_stamina_guard",
            emergency_stamina_guard,
            priority=10,
            description="Forces restorative rest when stamina is near zero",
        )

        # 2. Severe Storm Shelter Guard
        def severe_storm_shelter_guard(action: int, obs: List[float], ctx: Dict[str, object]):
            severity = obs[9] if len(obs) > 9 else 0.0
            exposure = obs[4] if len(obs) > 4 else 0.0
            shelter_avail = obs[8] if len(obs) > 8 else 0.0
            # If impending lethal storm and exposure is high, and shelter is present
            if severity > 0.85 and exposure > 0.70 and shelter_avail > 0.40 and action != 3:
                return (3, "modder:severe_storm_shelter_override")
            return (action, None)

        self.register_hook(
            HookPoint.PRE_ACTION,
            "severe_storm_shelter_guard",
            severe_storm_shelter_guard,
            priority=20,
            description="Forces shelter seeking during lethal storm conditions",
        )

    def manifest(self) -> Dict[str, object]:
        """Return cryptographic inventory of all active modder hooks."""
        entries = []
        for p, hook_list in self.hooks.items():
            for h in hook_list:
                entries.append({
                    "hook_id": h.hook_id,
                    "point": p.value,
                    "priority": h.priority,
                    "description": h.description,
                    "invocations": h.invocations,
                    "overrides": h.overrides,
                })
        manifest_digest = hashlib.sha256(str(entries).encode()).hexdigest()[:16]
        return {
            "manifest_digest": manifest_digest,
            "total_hooks": len(entries),
            "total_interceptions": self.total_interceptions,
            "total_overrides": self.total_overrides,
            "hooks": entries,
        }
