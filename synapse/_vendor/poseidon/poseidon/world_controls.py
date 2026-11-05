"""Retain planner evidence around the unchanged, replayable TidePool engine."""
from __future__ import annotations

import copy
import math
from typing import Callable

from .world import ACTIONS, WORLD_VERSION, TidePool, rollout


class RecordingController:
    """Call a planner once per action and keep detached decision evidence."""

    def __init__(self, controller):
        self.controller = controller
        self.decisions = []
        self.plan = getattr(controller, "plan", None)
        self.choose = controller.act if hasattr(controller, "act") else controller
        if not callable(self.choose) and not callable(self.plan):
            raise TypeError("policy must be callable or expose act(observation) or plan(observation)")

    def act(self, observation):
        if callable(self.plan):
            decision = self.plan(observation)
            if not isinstance(decision, dict) or "action" not in decision:
                raise ValueError("Planner must return a decision containing an action")
            action = decision["action"]
            self.decisions.append(copy.deepcopy(decision))
            return action
        self.decisions.append(None)
        return self.choose(observation)


def rollout_with_decisions(controller, seed=0, max_steps=256, scarcity=1.0):
    recorder = RecordingController(controller)
    episode = rollout(recorder, seed=seed, max_steps=max_steps, scarcity=scarcity)
    if len(recorder.decisions) != len(episode["trajectory"]):
        raise RuntimeError("Planner evidence and executed transitions disagree")
    for event, decision in zip(episode["trajectory"], recorder.decisions):
        if decision is not None:
            event["decision"] = decision
    return episode


def rollout_with_observed_transitions(controller, seed=0, max_steps=256, scarcity=1.0,
                                      on_transition: Callable[[dict], None] | None = None):
    """Run a replayable rollout and deliver real outcomes to transition-aware controllers."""
    env = TidePool(seed, scarcity, max_steps)
    initial = env.snapshot()
    trajectory = []
    action_counts = {name: 0 for name in ACTIONS}
    planner = getattr(controller, "plan", None)
    choose = controller.act if hasattr(controller, "act") else controller
    if not callable(choose) and not callable(planner):
        raise TypeError("policy must be callable or expose act(observation) or plan(observation)")
    obs = env.observe()
    info = {}
    while not env.done:
        before = list(obs)
        decision = planner(list(before)) if callable(planner) else None
        raw_action = decision["action"] if decision is not None else choose(list(before))
        if isinstance(raw_action, bool) or not hasattr(raw_action, "__index__"):
            raise ValueError("Policy must return an integer action")
        action = int(raw_action)
        obs, reward, _, info = env.step(action)
        if hasattr(controller, "observe_transition"):
            controller.observe_transition(before, action, reward, obs, info)
        action_counts[ACTIONS[action]] += 1
        event = {"observation": before, "action": action, "action_name": ACTIONS[action],
                 "reward": reward, "next_observation": obs, "state": env.state(), "info": info}
        if decision is not None:
            event["decision"] = copy.deepcopy(decision)
        trajectory.append(event)
        if on_transition is not None:
            on_transition({"seed": seed, "tick": env.tick, "max_steps": max_steps})
    return {"schema": "poseidon-episode-v1", "world_version": WORLD_VERSION,
            "seed": seed, "scarcity": scarcity, "max_steps": max_steps,
            "steps": env.tick, "survived": env.alive, "alive": env.alive,
            "death": not env.alive, "death_reason": env.death_reason,
            "truncated": info["truncated"], "terminal_reason": info["terminal_reason"],
            "reward": env.total_reward, "final_health": env.health,
            "visited": len(env.visited), "action_counts": action_counts,
            "initial_snapshot": initial, "final_snapshot": env.snapshot(),
            "trajectory": trajectory}
