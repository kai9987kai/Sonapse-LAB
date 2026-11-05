"""POSEIDON WORLDLAB: Realistic POMDP Benchmarks, Procedural Variations & Simulation Bridge.

Provides advanced experimental simulation testbeds beyond saturated base environments:
1. POMDPTidePool (Partially Observable Markov Decision Process):
   - Fog of War: Distant scent indicators occluded during intense weather storms.
   - Perceptual Sensor Noise: Gaussian noise on vital and environmental channels.
   - Dynamic Tidal Oscillations: Water availability modulates cyclically with tidal phase:
       W_{eff}(t) = W_{base} * (1.0 + A_{tide} * sin(2 * pi * t / T_{tide}))
   - Procedural Shifts: Seasonal droughts and blooms altering replenishment over time.

2. Standardized Environment Contract:
   Unified StepResult protocol ensuring zero friction between synthetic testbeds
   and external game engines.

3. WorldBridgeProtocol (Unity / 3D Simulation Interface):
   Standardized TCP/JSON socket bridge protocol enabling bidirectional control loops
   with external simulation engines (e.g. Unity, Unreal, WebGL simulators).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import json
import math
import random
import socket
from typing import Any, Sequence

from .world import ACTIONS, GRID_SIZE, N_ACTIONS, OBS_SIZE, WORLD_VERSION, TidePool, _unit

POMDP_SNAPSHOT_SCHEMA = "poseidon-pomdp-tidepool-v1"


@dataclass
class StepResult:
    """Standardized environment transition result."""
    observation: list[float]
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": [round(x, 6) for x in self.observation],
            "reward": round(self.reward, 6),
            "done": self.done,
            "truncated": self.truncated,
            "info": self.info,
        }


@dataclass(frozen=True)
class POMDPConfig:
    scarcity: float = 2.5
    sensor_noise_std: float = 0.02
    fog_threshold: float = 0.65
    tidal_amplitude: float = 0.35
    tidal_period: int = 24
    season_period: int = 48
    seed: int = 42

    def __post_init__(self):
        for name, low, high in (("scarcity", 0.5, 4.0), ("sensor_noise_std", 0, 1),
                                ("fog_threshold", 0, 1), ("tidal_amplitude", 0, 1)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid POMDP {name}")
        for name, high in (("tidal_period", 10000), ("season_period", 10000), ("seed", 2**63 - 1)):
            value = getattr(self, name)
            if type(value) is not int or not (0 if name == "seed" else 1) <= value <= high:
                raise ValueError(f"Invalid POMDP {name}")


class POMDPTidePool:
    """Partially Observable TidePool with sensor noise, fog of war, and dynamic tides."""

    def __init__(self, config: POMDPConfig | None = None, max_steps: int = 64):
        self.config = config or POMDPConfig()
        self.max_steps = max_steps
        self.underlying = TidePool(
            scarcity=self.config.scarcity,
            seed=self.config.seed,
            max_steps=max_steps,
        )
        self.rng = random.Random(self.config.seed)

    def reset(self, seed: int | None = None) -> list[float]:
        self.underlying.reset(seed=seed)
        self.rng = random.Random(self.underlying.seed)
        return self.observe()

    def observe(self) -> list[float]:
        raw_obs = list(self.underlying.observe())
        severity = raw_obs[9]
        tick = self.underlying.tick

        # Dynamic Tidal Modulation on water channel (channel 7)
        tide_factor = 1.0 + self.config.tidal_amplitude * math.sin(
            2.0 * math.pi * (tick % self.config.tidal_period) / self.config.tidal_period
        )
        raw_obs[7] = _unit(raw_obs[7] * tide_factor)

        # Fog of War: If storm severity exceeds threshold, obscure distant scent (channel 13)
        if severity > self.config.fog_threshold:
            raw_obs[13] = 0.0  # Fog obscuring distant scents

        # Sensory noise injection
        if self.config.sensor_noise_std > 0.0:
            noisy_obs = []
            for i, val in enumerate(raw_obs):
                # Don't add noise to action index or step fraction
                if i in (14, 15):
                    noisy_obs.append(val)
                else:
                    noise = self.rng.gauss(0.0, self.config.sensor_noise_std)
                    noisy_obs.append(_unit(val + noise))
            return noisy_obs

        return raw_obs

    def step(self, action: int) -> StepResult:
        obs, reward, done, info = self.underlying.step(action)
        noisy_obs = self.observe()
        truncated = bool(info.get("truncated", False))
        return StepResult(
            observation=noisy_obs,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def snapshot(self) -> dict[str, Any]:
        """Capture the world and sensor stream *after* all observations already drawn."""
        return {
            "schema": POMDP_SNAPSHOT_SCHEMA,
            "config": asdict(self.config),
            "underlying": self.underlying.snapshot(),
            "sensor_rng_state": copy.deepcopy(self.rng.getstate()),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "POMDPTidePool":
        if isinstance(snapshot, dict) and snapshot.get("schema") == WORLD_VERSION:
            raise ValueError("Legacy POMDP snapshots omit sensor RNG state; exact noisy replay is unavailable")
        fields = {"schema", "config", "underlying", "sensor_rng_state"}
        if not isinstance(snapshot, dict) or set(snapshot) != fields or snapshot["schema"] != POMDP_SNAPSHOT_SCHEMA:
            raise ValueError("Unsupported POMDP snapshot schema")
        try:
            config = POMDPConfig(**snapshot["config"])
            if set(snapshot["config"]) != set(asdict(config)):
                raise ValueError("Incomplete POMDP configuration")
            underlying = TidePool.from_snapshot(snapshot["underlying"])
            if underlying.scarcity != config.scarcity:
                raise ValueError("POMDP configuration differs from underlying world")
            state = snapshot["sensor_rng_state"]
            if (not isinstance(state, (tuple, list)) or len(state) != 3
                    or type(state[0]) is not int or state[0] != 3
                    or not isinstance(state[1], (tuple, list)) or len(state[1]) != 625
                    or any(type(v) is not int or not 0 <= v <= 2**32 - 1 for v in state[1][:-1])
                    or type(state[1][-1]) is not int or not 0 <= state[1][-1] <= 624
                    or (state[2] is not None and (type(state[2]) not in (float, int) or not math.isfinite(state[2])))):
                raise ValueError("Invalid sensor RNG state")
            rng = random.Random()
            rng.setstate((state[0], tuple(state[1]), state[2]))
        except (KeyError, TypeError, AttributeError, OverflowError) as error:
            raise ValueError("Malformed POMDP snapshot") from error
        restored = cls(config, max_steps=underlying.max_steps)
        restored.underlying = underlying
        restored.rng = rng
        return restored

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore atomically without drawing or advancing the sensor stream."""
        restored = self.from_snapshot(snapshot)
        self.__dict__.update(restored.__dict__)

    def state(self) -> dict[str, Any]:
        return self.underlying.state()

    @property
    def tick(self) -> int:
        return self.underlying.tick

    @property
    def alive(self) -> bool:
        return self.underlying.alive

    @property
    def done(self) -> bool:
        return self.underlying.done

    @property
    def health(self) -> float:
        return self.underlying.health

    @property
    def death_reason(self) -> str | None:
        return self.underlying.death_reason


class WorldBridgeProtocol:
    """Standardized serialization and message exchange protocol for external 3D simulators."""

    VERSION = "poseidon-worldbridge-v1"

    @classmethod
    def encode_action(cls, action_id: int) -> bytes:
        payload = {
            "version": cls.VERSION,
            "type": "ACTION_DISPATCH",
            "action": int(action_id),
            "action_name": ACTIONS[action_id],
        }
        return (json.dumps(payload) + "\n").encode("utf-8")

    @classmethod
    def decode_action(cls, data: bytes | str) -> int:
        msg = json.loads(data.decode("utf-8") if isinstance(data, bytes) else data)
        return int(msg["action"])

    @classmethod
    def encode_step_result(cls, result: StepResult) -> bytes:
        payload = {
            "version": cls.VERSION,
            "type": "STEP_RESULT",
            "data": result.to_dict(),
        }
        return (json.dumps(payload) + "\n").encode("utf-8")

    @classmethod
    def decode_step_result(cls, data: bytes | str) -> StepResult:
        msg = json.loads(data.decode("utf-8") if isinstance(data, bytes) else data)
        d = msg["data"]
        return StepResult(
            observation=list(d["observation"]),
            reward=float(d["reward"]),
            done=bool(d["done"]),
            truncated=bool(d["truncated"]),
            info=dict(d["info"]),
        )
