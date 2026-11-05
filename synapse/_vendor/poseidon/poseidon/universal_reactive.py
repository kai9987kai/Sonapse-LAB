"""POSEIDON UNIVERSAL-REACTIVE: Runtime Bytecode Hot-Patching & Decoupled Actor Pipeline.

Synthesizes:
1. Universal-Modder: Non-destructive dynamic bytecode hooking, method interception,
   hot-patching of weights and scoring parameters, and transactional rollback semantics.
2. Rea (Reactive Actor Framework): Asynchronous message-passing actor pipeline separating
   high-frequency sensory-motor reflexes from compute-heavy causal and quantum tree exploration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import functools
import inspect
import queue
import threading
import time
from typing import Any, Callable


# =====================================================================
# 1. UNIVERSAL MODDER: Dynamic Bytecode & Method Interception Engine
# =====================================================================

@dataclass
class HookRecord:
    target: Any
    attribute_name: str
    original_callable: Callable[..., Any]
    active_wrapper: Callable[..., Any]


class UniversalModder:
    """Non-destructive runtime interceptor for hot-patching functions, methods, and parameters."""

    def __init__(self):
        self._hooks: dict[tuple[int, str], HookRecord] = {}
        self._value_backups: dict[tuple[int, str], Any] = {}
        self._lock = threading.RLock()

    def intercept(
        self,
        target: Any,
        attr_name: str,
        interceptor_fn: Callable[[Callable[..., Any], Any, tuple, dict], Any],
    ) -> HookRecord:
        """Hooks a method with an interceptor: interceptor_fn(original_fn, self_or_cls, args, kwargs)."""
        with self._lock:
            key = (id(target), attr_name)
            if not hasattr(target, attr_name):
                raise AttributeError(f"Target {target} does not have attribute '{attr_name}'")

            current_val = getattr(target, attr_name)
            if not callable(current_val):
                raise TypeError(f"Attribute '{attr_name}' on {target} is not callable")

            original_fn = self._hooks[key].original_callable if key in self._hooks else current_val

            @functools.wraps(original_fn)
            def wrapper(*args, **kwargs):
                return interceptor_fn(original_fn, target, args, kwargs)

            setattr(target, attr_name, wrapper)
            record = HookRecord(
                target=target,
                attribute_name=attr_name,
                original_callable=original_fn,
                active_wrapper=wrapper,
            )
            self._hooks[key] = record
            return record

    def revert(self, target: Any, attr_name: str) -> bool:
        """Restores a hooked callable back to its original definition."""
        with self._lock:
            key = (id(target), attr_name)
            if key in self._hooks:
                record = self._hooks.pop(key)
                setattr(target, attr_name, record.original_callable)
                return True
            return False

    def hotpatch_value(self, target: Any, attr_name: str, new_value: Any) -> Any:
        """Updates an attribute value while saving backup for transactional rollback."""
        with self._lock:
            key = (id(target), attr_name)
            if key not in self._value_backups:
                self._value_backups[key] = getattr(target, attr_name)
            setattr(target, attr_name, new_value)
            return new_value

    def restore_value(self, target: Any, attr_name: str) -> bool:
        """Restores a patched attribute value to its original pre-patch state."""
        with self._lock:
            key = (id(target), attr_name)
            if key in self._value_backups:
                original_val = self._value_backups.pop(key)
                setattr(target, attr_name, original_val)
                return True
            return False

    def revert_all(self) -> None:
        """Restores all hooked callables and hotpatched attributes across the system."""
        with self._lock:
            for (obj_id, attr_name), record in list(self._hooks.items()):
                setattr(record.target, attr_name, record.original_callable)
            self._hooks.clear()

            # Note: value backups cannot all resolve target if object was collected, but we clear
            self._value_backups.clear()


class ModderTransaction:
    """Context manager guaranteeing atomic hotpatches with automatic rollback on error."""

    def __init__(self, modder: UniversalModder):
        self.modder = modder
        self._staged_hooks: list[tuple[Any, str]] = []
        self._staged_values: list[tuple[Any, str]] = []

    def intercept(self, target: Any, attr_name: str, interceptor_fn: Callable) -> HookRecord:
        record = self.modder.intercept(target, attr_name, interceptor_fn)
        self._staged_hooks.append((target, attr_name))
        return record

    def hotpatch_value(self, target: Any, attr_name: str, new_value: Any) -> Any:
        res = self.modder.hotpatch_value(target, attr_name, new_value)
        self._staged_values.append((target, attr_name))
        return res

    def __enter__(self) -> ModderTransaction:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Error occurred: rollback all changes made in this transaction
            for target, attr_name in reversed(self._staged_hooks):
                self.modder.revert(target, attr_name)
            for target, attr_name in reversed(self._staged_values):
                self.modder.restore_value(target, attr_name)
        return False


# =====================================================================
# 2. REA REACTIVE ACTOR ENGINE: Asynchronous Decoupled Message Passing
# =====================================================================

@dataclass
class ActorMessage:
    """Envelope for typed reactive message passing."""
    sender: str
    recipient: str
    topic: str
    payload: Any
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = ""


class ReactiveActor:
    """Base asynchronous message-passing actor."""

    def __init__(self, actor_id: str, system: ReactiveActorSystem):
        self.actor_id = actor_id
        self.system = system
        self.inbox: queue.Queue[ActorMessage] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def post(self, msg: ActorMessage) -> None:
        self.inbox.put(msg)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name=f"Actor-{self.actor_id}")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self.inbox.put(ActorMessage(sender="system", recipient=self.actor_id, topic="sys.stop", payload=None))
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run_loop(self) -> None:
        while self._running:
            try:
                msg = self.inbox.get(timeout=0.05)
                if msg.topic == "sys.stop":
                    break
                self.handle_message(msg)
                self.inbox.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Log or handle error without crashing actor thread
                pass

    def handle_message(self, msg: ActorMessage) -> None:
        raise NotImplementedError


class PerceptionActor(ReactiveActor):
    """Processes sensory observations and incoming demodulated acoustic bio-pulses."""

    def __init__(self, actor_id: str, system: ReactiveActorSystem):
        super().__init__(actor_id, system)
        self.last_somatic_state: list[float] = []

    def handle_message(self, msg: ActorMessage) -> None:
        if msg.topic == "sensory.raw_obs":
            obs = msg.payload.get("obs", [])
            self.last_somatic_state = obs
            # Compute immediate threat saliency
            threat = obs[5] if len(obs) > 5 else 0.0
            # Broadcast normalized percept
            self.system.dispatch(ActorMessage(
                sender=self.actor_id,
                recipient="motor_shield",
                topic="percept.normalized",
                payload={"obs": obs, "threat": threat},
                correlation_id=msg.correlation_id,
            ))


class CausalActor(ReactiveActor):
    """Performs background Pearl SCM counterfactual interventional predictions."""

    def __init__(self, actor_id: str, system: ReactiveActorSystem, scm_model: Any = None):
        super().__init__(actor_id, system)
        self.scm = scm_model
        self.last_intervention_effect: float = 0.0

    def handle_message(self, msg: ActorMessage) -> None:
        if msg.topic == "causal.evaluate":
            obs = msg.payload.get("obs", [])
            action = msg.payload.get("action", 0)
            effect = 0.0
            if self.scm is not None and hasattr(self.scm, "intervene"):
                pred = self.scm.intervene(obs, action)
                effect = pred.expected_vital_delta
            else:
                effect = 0.05 if action in (1, 2) else 0.01

            self.last_intervention_effect = effect
            self.system.dispatch(ActorMessage(
                sender=self.actor_id,
                recipient="motor_shield",
                topic="causal.projection",
                payload={"action": action, "effect": effect},
                correlation_id=msg.correlation_id,
            ))


class QuantumSearchActor(ReactiveActor):
    """Explores complex probability amplitude tree superposition in background thread."""

    def __init__(self, actor_id: str, system: ReactiveActorSystem, planner: Any = None):
        super().__init__(actor_id, system)
        self.planner = planner
        self.last_coherence: float = 1.0

    def handle_message(self, msg: ActorMessage) -> None:
        if msg.topic == "quantum.search":
            obs = msg.payload.get("obs", [])
            coherence = 0.95
            gain = 0.12
            if self.planner is not None and hasattr(self.planner, "plan_superposition"):
                plan = self.planner.plan_superposition(obs)
                coherence = plan.quantum_coherence
                gain = plan.interference_gain

            self.last_coherence = coherence
            self.system.dispatch(ActorMessage(
                sender=self.actor_id,
                recipient="motor_shield",
                topic="quantum.plan_ready",
                payload={"coherence": coherence, "interference_gain": gain},
                correlation_id=msg.correlation_id,
            ))


class MotorShieldActor(ReactiveActor):
    """Shields candidate motor commands against formal invariants before physical actuation."""

    def __init__(self, actor_id: str, system: ReactiveActorSystem):
        super().__init__(actor_id, system)
        self.latest_percept: dict[str, Any] = {}
        self.latest_causal: dict[str, Any] = {}
        self.latest_quantum: dict[str, Any] = {}
        self.certified_action: int = 0
        self.intervention_count: int = 0

    def handle_message(self, msg: ActorMessage) -> None:
        if msg.topic == "percept.normalized":
            self.latest_percept = msg.payload
        elif msg.topic == "causal.projection":
            self.latest_causal = msg.payload
        elif msg.topic == "quantum.plan_ready":
            self.latest_quantum = msg.payload
        elif msg.topic == "motor.candidate":
            cand = msg.payload.get("candidate_action", 0)
            obs = msg.payload.get("obs") or self.latest_percept.get("obs", [])

            # Check invariant vital shield
            # e.g., if energy is critically low (< 0.12) and candidate is flee/explore, shield to forage
            shielded = cand
            if obs and len(obs) > 2:
                energy = obs[1]
                hydration = obs[2]
                if energy < 0.12 and cand not in (0, 1):
                    shielded = 1  # override to forage
                    self.intervention_count += 1
                elif hydration < 0.12 and cand not in (0, 2):
                    shielded = 2  # override to drink
                    self.intervention_count += 1

            self.certified_action = shielded


class ReactiveActorSystem:
    """Decoupled reactive actor bus coordinating concurrent actors."""

    def __init__(self):
        self.actors: dict[str, ReactiveActor] = {}
        self.perception = PerceptionActor("perception", self)
        self.causal = CausalActor("causal", self)
        self.quantum = QuantumSearchActor("quantum", self)
        self.motor_shield = MotorShieldActor("motor_shield", self)

        self.register_actor(self.perception)
        self.register_actor(self.causal)
        self.register_actor(self.quantum)
        self.register_actor(self.motor_shield)

    def register_actor(self, actor: ReactiveActor) -> None:
        self.actors[actor.actor_id] = actor

    def dispatch(self, msg: ActorMessage) -> None:
        if msg.recipient in self.actors:
            self.actors[msg.recipient].post(msg)

    def start_all(self) -> None:
        for actor in self.actors.values():
            actor.start()

    def stop_all(self) -> None:
        for actor in self.actors.values():
            actor.stop()

    def step_cognitive_pipeline(self, obs: list[float], candidate_action: int) -> int:
        """Synchronously pushes sensory input and queries certified motor output."""
        corr = f"step-{time.time()}"
        self.dispatch(ActorMessage("sys", "perception", "sensory.raw_obs", {"obs": obs}, correlation_id=corr))
        self.dispatch(ActorMessage("sys", "causal", "causal.evaluate", {"obs": obs, "action": candidate_action}, correlation_id=corr))
        self.dispatch(ActorMessage("sys", "quantum", "quantum.search", {"obs": obs}, correlation_id=corr))
        self.dispatch(ActorMessage("sys", "motor_shield", "motor.candidate", {"candidate_action": candidate_action, "obs": obs}, correlation_id=corr))

        # Yield tiny slice for actors
        time.sleep(0.01)
        return self.motor_shield.certified_action
