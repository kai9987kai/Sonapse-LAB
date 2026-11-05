"""Deterministic synthetic survival task; no neural policy or teacher fallback.

Actions are bounded macro actions, not a robotics controller. Event-keyed weather
and hazards align exogenous events between policies even after paths diverge.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from statistics import fmean
from typing import Callable, Iterable

WORLD_VERSION = "tidepool-v1"
ACTIONS = ("rest", "forage", "drink", "shelter", "explore", "flee")
OBSERVATION_NAMES = (
    "health", "energy", "hydration", "stamina", "exposure", "threat", "food",
    "water", "shelter_quality", "weather_severity", "temperature", "daylight",
    "terrain_difficulty", "resource_scent", "last_action_scaled", "episode_progress",
)
OBS_SIZE = 16
N_ACTIONS = 6
GRID_SIZE = 6


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def event_uniform(seed: int, tick: int, actor: object, channel: str) -> float:
    """Stable named-event draw; no global or mutable random generator is used."""
    key = json.dumps([int(seed), int(tick), actor, channel], separators=(",", ":"))
    value = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
    return (value >> 11) / float(1 << 53)


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return float(value)


class TidePool:
    """A small resource world with one agent and six observable-action choices.

    ``scarcity=1`` is the reference environment; larger values reduce resource
    capacity and renewal. ``terminated`` from step means the episode ended;
    inspect ``info['death']`` versus ``info['truncated']`` to distinguish causes.
    """

    def __init__(self, seed: int = 0, scarcity: float = 1.0, max_steps: int = 256):
        self.scarcity = _number(scarcity, "scarcity", 0.5, 4.0)
        self.max_steps = _integer(max_steps, "max_steps", 1, 10000)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is None:
            seed = getattr(self, "seed", 0)
        self.seed = _integer(seed, "seed", 0, 2**63 - 1)
        self.tick = 0
        self.x = int(event_uniform(seed, 0, "spawn", "x") * GRID_SIZE)
        self.y = int(event_uniform(seed, 0, "spawn", "y") * GRID_SIZE)
        self.health = 1.0
        self.energy = 0.4 + 0.55 * event_uniform(seed, 0, "agent", "energy")
        self.hydration = 0.4 + 0.55 * event_uniform(seed, 0, "agent", "hydration")
        self.stamina = 0.35 + 0.6 * event_uniform(seed, 0, "agent", "stamina")
        self.exposure = 0.05 + 0.55 * event_uniform(seed, 0, "agent", "exposure")
        self.last_action = 0
        self.total_reward = 0.0
        self.death_reason = None
        self.done = False
        self.visited = {self.y * GRID_SIZE + self.x}
        self.patches = []
        for index in range(GRID_SIZE * GRID_SIZE):
            food_cap = min(1.0, (0.55 + 0.45 * event_uniform(seed, 0, index, "food-cap")) / self.scarcity)
            water_cap = min(1.0, (0.5 + 0.5 * event_uniform(seed, 0, index, "water-cap")) / self.scarcity)
            self.patches.append({
                "food": food_cap, "water": water_cap,
                "food_cap": food_cap, "water_cap": water_cap,
                "shelter": 0.25 + 0.75 * event_uniform(seed, 0, index, "shelter"),
                "terrain": event_uniform(seed, 0, index, "terrain"), "updated": 0,
            })
        return self.observe()

    @property
    def alive(self) -> bool:
        return self.health > 0

    def _index(self) -> int:
        return self.y * GRID_SIZE + self.x

    def _patch(self, index: int | None = None) -> dict:
        patch = self.patches[self._index() if index is None else index]
        elapsed = self.tick - patch["updated"]
        if elapsed:
            patch["food"] = min(patch["food_cap"], patch["food"] + elapsed * 0.007 / self.scarcity)
            patch["water"] = min(patch["water_cap"], patch["water"] + elapsed * 0.014 / self.scarcity)
            patch["updated"] = self.tick
        return patch

    def _neighbors(self) -> list[int]:
        return [self.y * GRID_SIZE + (self.x + 1) % GRID_SIZE,
                ((self.y + 1) % GRID_SIZE) * GRID_SIZE + self.x,
                self.y * GRID_SIZE + (self.x - 1) % GRID_SIZE,
                ((self.y - 1) % GRID_SIZE) * GRID_SIZE + self.x]

    def _weather(self) -> tuple[float, float, float]:
        severity = event_uniform(self.seed, self.tick // 12, "world", "storm")
        temperature = 0.35 + 0.45 * event_uniform(self.seed, self.tick // 18, "world", "heat")
        daylight = (1.0 + math.sin((self.tick + self.seed % 64) * math.tau / 64)) / 2
        return severity, temperature, daylight

    def _threat(self, index: int | None = None) -> float:
        index = self._index() if index is None else index
        draw = event_uniform(self.seed, self.tick // 4, index, "encounter")
        return (0.65 + 0.35 * draw) if draw > 0.87 else draw * 0.3

    def observe(self) -> list[float]:
        patch = self._patch()
        severity, temperature, daylight = self._weather()
        # The scent is a local sensor aggregate, not a future outcome or action label.
        scent = max((self._patch(i)["food"] + self._patch(i)["water"]) / 2 for i in self._neighbors())
        return [_unit(x) for x in (
            self.health, self.energy, self.hydration, self.stamina, self.exposure,
            self._threat(), patch["food"], patch["water"], patch["shelter"],
            severity, temperature, daylight, patch["terrain"], scent,
            self.last_action / 5, self.tick / self.max_steps,
        )]

    def _move(self, fleeing: bool) -> None:
        neighbors = self._neighbors()
        if fleeing:
            target = max(neighbors, key=lambda i: self._patch(i)["shelter"] - 2 * self._threat(i))
        else:
            # Exploring follows local food/water scent, an explicit macro action.
            target = max(neighbors, key=lambda i: (1 - self.energy) * self._patch(i)["food"]
                         + (1 - self.hydration) * self._patch(i)["water"])
        self.y, self.x = divmod(target, GRID_SIZE)
        self.visited.add(target)

    def step(self, action: int) -> tuple[list[float], float, bool, dict]:
        action = _integer(action, "action", 0, len(ACTIONS) - 1)
        if self.done:
            raise RuntimeError("Episode ended; call reset before stepping again")
        before = self.observe()
        patch = self._patch()
        severity, temperature, _ = self._weather()
        food_taken = water_taken = 0.0
        if action == 0:
            self.stamina += 0.25
            self.exposure -= 0.045 * patch["shelter"]
        elif action == 1:
            food_taken = min(patch["food"], 0.30)
            patch["food"] -= food_taken
            self.energy += food_taken
            self.stamina -= 0.06
        elif action == 2:
            water_taken = min(patch["water"], 0.38)
            patch["water"] -= water_taken
            self.hydration += water_taken
            self.stamina -= 0.015
        elif action == 3:
            self.exposure -= 0.28 + 0.30 * patch["shelter"]
            self.stamina += 0.10
        elif action in (4, 5):
            self._move(action == 5)
            self.stamina -= 0.10 if action == 4 else 0.13
            self.energy -= 0.02
            patch = self._patch()

        self.energy -= 0.018 + 0.008 * severity
        self.hydration -= 0.023 + 0.014 * temperature
        self.stamina -= 0.012
        self.exposure += 0.016 + 0.04 * severity * (1 - patch["shelter"])
        damage = 0.0
        causes = []
        for condition, cost, cause in (
            (self.energy < 0.08, 0.07, "starvation"),
            (self.hydration < 0.08, 0.09, "dehydration"),
            (self.stamina < 0.04, 0.035, "exhaustion"),
            (self.exposure > 0.85, 0.065, "exposure"),
        ):
            if condition:
                damage += cost
                causes.append(cause)
        threat = before[5]
        if threat > 0.68 and action not in (3, 5):
            damage += 0.075 + 0.035 * threat
            causes.append("hazard")
        self.health -= damage
        if damage == 0 and min(self.energy, self.hydration) > 0.4 and self.exposure < 0.5:
            self.health += 0.014
        for field in ("health", "energy", "hydration", "stamina", "exposure"):
            setattr(self, field, _unit(getattr(self, field)))
        self.tick += 1
        self.last_action = action
        if not self.alive:
            self.death_reason = "+".join(causes) or "depleted-health"
        self.done = not self.alive or self.tick >= self.max_steps
        reward = 0.1 + 0.3 * (self.energy - before[1] + self.hydration - before[2]) - 3 * damage
        if not self.alive:
            reward -= 2.0
        self.total_reward += reward
        obs = self.observe()
        return obs, float(reward), self.done, {
            "tick": self.tick, "action": ACTIONS[action], "alive": self.alive,
            "death": not self.alive, "death_reason": self.death_reason,
            "truncated": self.alive and self.tick >= self.max_steps,
            "terminal_reason": "death" if not self.alive else "horizon" if self.done else None,
            "food_taken": food_taken, "water_taken": water_taken,
            "position": [self.x, self.y], "visited": len(self.visited),
        }

    def snapshot(self) -> dict:
        return copy.deepcopy({
            "schema": WORLD_VERSION, "seed": self.seed, "tick": self.tick,
            "scarcity": self.scarcity, "max_steps": self.max_steps,
            "x": self.x, "y": self.y, "health": self.health, "energy": self.energy,
            "hydration": self.hydration, "stamina": self.stamina, "exposure": self.exposure,
            "last_action": self.last_action, "total_reward": self.total_reward,
            "death_reason": self.death_reason, "done": self.done,
            "visited": sorted(self.visited), "patches": self.patches,
            "randomization": "blake2b-named-events-v1",
        })

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "TidePool":
        if not isinstance(snapshot, dict) or snapshot.get("schema") != WORLD_VERSION:
            raise ValueError("Unsupported TidePool snapshot schema")
        data = copy.deepcopy(snapshot)
        if data.get("randomization") != "blake2b-named-events-v1":
            raise ValueError("Unsupported randomization version")
        seed = _integer(data.get("seed"), "seed", 0, 2**63 - 1)
        horizon = _integer(data.get("max_steps"), "max_steps", 1, 10000)
        scarcity = _number(data.get("scarcity"), "scarcity", 0.5, 4.0)
        tick = _integer(data.get("tick"), "tick", 0, horizon)
        for field in ("x", "y"):
            _integer(data.get(field), field, 0, GRID_SIZE - 1)
        for field in ("health", "energy", "hydration", "stamina", "exposure"):
            _number(data.get(field), field, 0, 1)
        _integer(data.get("last_action"), "last_action", 0, 5)
        _number(data.get("total_reward"), "total_reward", -100000, 100000)
        if type(data.get("done")) is not bool or data["done"] != (data["health"] == 0 or tick >= horizon):
            raise ValueError("Snapshot terminal state is inconsistent")
        if data.get("death_reason") is not None and (not isinstance(data["death_reason"], str) or len(data["death_reason"]) > 120):
            raise ValueError("Invalid death reason")
        if (data["health"] == 0) != bool(data.get("death_reason")):
            raise ValueError("Death reason and health disagree")
        patches = data.get("patches")
        if not isinstance(patches, list) or len(patches) != GRID_SIZE * GRID_SIZE:
            raise ValueError("Snapshot must contain every patch")
        for patch in patches:
            if not isinstance(patch, dict) or set(patch) != {"food", "water", "food_cap", "water_cap", "shelter", "terrain", "updated"}:
                raise ValueError("Invalid patch fields")
            for field in ("food", "water", "food_cap", "water_cap", "shelter", "terrain"):
                _number(patch[field], "patch." + field, 0, 1)
            if patch["food"] > patch["food_cap"] or patch["water"] > patch["water_cap"]:
                raise ValueError("Resources exceed capacity")
            _integer(patch["updated"], "patch.updated", 0, tick)
        visited = data.get("visited")
        if not isinstance(visited, list) or not visited or len(visited) > GRID_SIZE * GRID_SIZE:
            raise ValueError("Invalid visited patches")
        for value in visited:
            _integer(value, "visited", 0, GRID_SIZE * GRID_SIZE - 1)
        if len(set(visited)) != len(visited) or data["y"] * GRID_SIZE + data["x"] not in visited:
            raise ValueError("Inconsistent visited patches")
        world = cls(seed, scarcity, horizon)
        for field in ("seed", "tick", "scarcity", "max_steps", "x", "y", "health", "energy", "hydration",
                      "stamina", "exposure", "last_action", "total_reward", "death_reason", "done", "patches"):
            setattr(world, field, data[field])
        world.visited = set(visited)
        return world

    def load_snapshot(self, snapshot: dict) -> list[float]:
        candidate = self.from_snapshot(snapshot)
        self.__dict__.update(candidate.__dict__)
        return self.observe()

    def state(self) -> dict:
        """Compact rendering state; snapshot() is the complete replay artifact."""
        return {"tick": self.tick, "position": [self.x, self.y], "x": self.x, "y": self.y,
                "health": self.health, "energy": self.energy, "hydration": self.hydration,
                "stamina": self.stamina, "exposure": self.exposure, "alive": self.alive,
                "visited": len(self.visited), "observation": self.observe()}


def teacher_action(obs: Iterable[float]) -> int:
    """Visible-state scripted baseline. Never invoked by learned policy evaluation."""
    values = list(obs)
    if len(values) != OBS_SIZE or any(not math.isfinite(float(x)) for x in values):
        raise ValueError("teacher_action requires 16 finite observable values")
    _, energy, hydration, stamina, exposure, threat, food, water, *_ = values
    if threat > 0.68:
        return 5
    if hydration < 0.42:
        return 2 if water > 0.12 else 4
    if energy < 0.40:
        return 1 if food > 0.10 else 4
    if stamina < 0.28:
        return 0
    if exposure > 0.60:
        return 3
    if hydration < 0.76 and water > 0.18:
        return 2
    if energy < 0.72 and food > 0.14:
        return 1
    if min(food, water) < 0.10 and min(energy, hydration) < 0.72:
        return 4
    if exposure > 0.35:
        return 3
    return 0


def generate_transitions(count: int, seed: int = 0) -> list[dict]:
    """Collect actual dynamics with teacher labels and 12% exploration.

    action is the EXECUTED action for dynamics; teacher_action is the imitation
    label. Entire episodes have unique seeds. Callers must split by episode_seed.
    """
    _integer(count, "count", 0, 2000000)
    _integer(seed, "seed", 0, 2**40 - 1)
    rng = random.Random(seed)
    records = []
    episode = 0
    while len(records) < count:
        # A distinct namespace for each generator seed prevents train/dev overlap.
        episode_seed = (seed << 23) | episode
        scarcity = (0.8, 1.0, 1.25, 1.5)[episode % 4]
        env = TidePool(episode_seed, scarcity=scarcity, max_steps=192)
        obs = env.observe()
        while not env.done and len(records) < count:
            label = teacher_action(obs)
            action = rng.randrange(N_ACTIONS) if rng.random() < 0.12 else label
            nxt, reward, done, info = env.step(action)
            records.append({"obs": obs, "action": action, "teacher_action": label,
                            "next_obs": nxt, "reward": reward, "terminated": done,
                            "death": info["death"], "episode_seed": episode_seed,
                            "scarcity": scarcity, "tick": env.tick - 1})
            obs = nxt
        episode += 1
    return records


def training_transitions(n: int, seed: int = 0):
    """Tuple compatibility view; includes executed action, not teacher label."""
    return [(r["obs"], r["action"], r["next_obs"]) for r in generate_transitions(n, seed)]


def rollout(policy: Callable | object, seed: int = 0, max_steps: int = 256,
            scarcity: float = 1.0, include_trajectory: bool = True) -> dict:
    env = TidePool(seed, scarcity, max_steps)
    initial = env.snapshot()
    trajectory = []
    action_counts = {name: 0 for name in ACTIONS}
    choose = policy.act if hasattr(policy, "act") else policy
    if not callable(choose):
        raise TypeError("policy must be callable or expose act(observation)")
    obs = env.observe()
    info = {}
    while not env.done:
        raw_action = choose(list(obs))
        # NumPy integer outputs are permitted; floats/booleans are not.
        if isinstance(raw_action, bool) or not hasattr(raw_action, "__index__"):
            raise ValueError("Policy must return an integer action")
        action = int(raw_action)
        before = obs
        obs, reward, _, info = env.step(action)
        action_counts[ACTIONS[action]] += 1
        if include_trajectory:
            trajectory.append({"observation": before, "action": action,
                               "action_name": ACTIONS[action], "reward": reward,
                               "next_observation": obs, "state": env.state(), "info": info})
    return {"schema": "poseidon-episode-v1", "world_version": WORLD_VERSION,
            "seed": seed, "scarcity": scarcity, "max_steps": max_steps,
            "steps": env.tick, "survived": env.alive, "alive": env.alive,
            "death": not env.alive, "death_reason": env.death_reason,
            "truncated": info["truncated"], "terminal_reason": info["terminal_reason"],
            "reward": env.total_reward, "final_health": env.health,
            "visited": len(env.visited), "action_counts": action_counts,
            "initial_snapshot": initial, "final_snapshot": env.snapshot(),
            "trajectory": trajectory}


def benchmark(policy: Callable | object, seeds: Iterable[int] = range(900001, 900017),
              max_steps: int = 256, scarcity: float = 1.0) -> dict:
    seed_list = list(seeds)
    if not seed_list or len(set(seed_list)) != len(seed_list):
        raise ValueError("Benchmark seeds must be nonempty and distinct")
    rows = []
    for seed in seed_list:
        rng = random.Random(seed ^ 0x5A17)
        arms = {"learned": policy, "heuristic": teacher_action,
                "random": lambda obs, generator=rng: generator.randrange(N_ACTIONS)}
        for name, controller in arms.items():
            result = rollout(controller, seed, max_steps, scarcity, include_trajectory=False)
            rows.append({"arm": name, **{k: result[k] for k in
                         ("seed", "steps", "survived", "death_reason", "truncated", "reward", "final_health")}})
    summaries = {}
    for arm in ("learned", "heuristic", "random"):
        group = [r for r in rows if r["arm"] == arm]
        summaries[arm] = {"episodes": len(group), "survival_rate": fmean(r["survived"] for r in group),
                          "mean_steps": fmean(r["steps"] for r in group),
                          "mean_reward": fmean(r["reward"] for r in group)}
    paired = []
    for seed in seed_list:
        group = {r["arm"]: r for r in rows if r["seed"] == seed}
        paired.append({"seed": seed,
                       "survival_vs_random": int(group["learned"]["survived"]) - int(group["random"]["survived"]),
                       "steps_vs_random": group["learned"]["steps"] - group["random"]["steps"],
                       "steps_vs_heuristic": group["learned"]["steps"] - group["heuristic"]["steps"]})
    return {"schema": "poseidon-survival-benchmark-v1", "world_version": WORLD_VERSION,
            "scarcity": scarcity, "max_steps": max_steps, "seeds": seed_list,
            "summary": summaries, "rows": rows, "paired_differences": paired,
            "scope": "Synthetic macro-action task; no real-world survival or robotics validation.",
            "randomization": "Exogenous named events share seed/tick/location keys; policy RNG is separate."}
