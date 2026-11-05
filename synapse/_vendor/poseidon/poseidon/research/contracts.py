"""Versioned contracts for transition datasets and episode-level splits.

The contract makes episode identity explicit so transitions from one simulated
world can never be scattered across train, calibration, and test partitions.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from ..world import ACTIONS, OBSERVATION_NAMES, OBS_SIZE, WORLD_VERSION

CONTRACT_VERSION = "poseidon-transition-contract-v1"
ALIAS_WORLD_VERSION = "alias-tidepool-v1"
ALIAS_WORLD_V2_VERSION = "alias-tidepool-v2-resource-maintenance"
SUPPORTED_WORLD_VERSIONS = (WORLD_VERSION, ALIAS_WORLD_VERSION, ALIAS_WORLD_V2_VERSION)


@dataclass(frozen=True)
class EnvironmentContract:
    version: str
    world_version: str
    observation_names: tuple[str, ...]
    actions: tuple[str, ...]

    def validate(self) -> None:
        if self.version != CONTRACT_VERSION:
            raise ValueError("unsupported environment-contract version")
        if self.world_version not in SUPPORTED_WORLD_VERSIONS:
            raise ValueError("environment contract targets an unsupported world version")
        if len(self.observation_names) != OBS_SIZE or len(set(self.observation_names)) != OBS_SIZE:
            raise ValueError("observation schema is invalid")
        if self.observation_names != OBSERVATION_NAMES or self.actions != ACTIONS:
            raise ValueError("environment contract does not match TidePool")


def tidepool_contract() -> EnvironmentContract:
    contract = EnvironmentContract(CONTRACT_VERSION, WORLD_VERSION, OBSERVATION_NAMES, ACTIONS)
    contract.validate()
    return contract


def alias_tidepool_contract() -> EnvironmentContract:
    contract = EnvironmentContract(CONTRACT_VERSION, ALIAS_WORLD_VERSION, OBSERVATION_NAMES, ACTIONS)
    contract.validate()
    return contract


def alias_tidepool_v2_contract() -> EnvironmentContract:
    contract = EnvironmentContract(CONTRACT_VERSION, ALIAS_WORLD_V2_VERSION, OBSERVATION_NAMES, ACTIONS)
    contract.validate()
    return contract


@dataclass(frozen=True)
class TransitionRecord:
    episode_seed: int
    tick: int
    scarcity: float
    observation: tuple[float, ...]
    action: int
    next_observation: tuple[float, ...]
    reward: float
    death: bool
    truncated: bool

    @property
    def terminal(self) -> bool:
        return self.death or self.truncated

    def validate(self) -> None:
        if type(self.episode_seed) is not int or not 0 <= self.episode_seed < 2**63:
            raise ValueError("episode_seed must be an integer in [0, 2**63)")
        if type(self.tick) is not int or self.tick < 0:
            raise ValueError("tick must be a nonnegative integer")
        if isinstance(self.scarcity, bool) or not isinstance(self.scarcity, (int, float)) or not math.isfinite(self.scarcity) or not 0.5 <= self.scarcity <= 4:
            raise ValueError("scarcity must be finite and in [0.5, 4]")
        if type(self.action) is not int or not 0 <= self.action < len(ACTIONS):
            raise ValueError("action is outside the TidePool action set")
        if type(self.death) is not bool or type(self.truncated) is not bool:
            raise ValueError("death and truncated must be booleans")
        if self.death and self.truncated:
            raise ValueError("a transition cannot be both a death and a horizon truncation")
        if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)) or not math.isfinite(self.reward):
            raise ValueError("reward must be finite")
        for name, values in (("observation", self.observation), ("next_observation", self.next_observation)):
            if not isinstance(values, (tuple, list)) or len(values) != OBS_SIZE:
                raise ValueError(f"{name} must contain {OBS_SIZE} values")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError(f"{name} values must be finite and in [0, 1]")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "episode_seed": self.episode_seed,
            "tick": self.tick,
            "scarcity": float(self.scarcity),
            "observation": list(self.observation),
            "action": self.action,
            "next_observation": list(self.next_observation),
            "reward": float(self.reward),
            "death": self.death,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class Episode:
    seed: int
    scarcity: float
    transitions: tuple[TransitionRecord, ...]

    def validate(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("episode seed is invalid")
        if not self.transitions:
            raise ValueError("episode must contain at least one transition")
        for index, row in enumerate(self.transitions):
            row.validate()
            if row.episode_seed != self.seed or row.tick != index or row.scarcity != self.scarcity:
                raise ValueError("episode transition identity, tick, or scarcity is inconsistent")
            if index and self.transitions[index - 1].next_observation != row.observation:
                raise ValueError("episode transition observations are not contiguous")
            if index < len(self.transitions) - 1 and row.terminal:
                raise ValueError("episode contains transitions after its terminal state")
        if not self.transitions[-1].terminal:
            raise ValueError("episode must end in death or horizon truncation")


@dataclass(frozen=True)
class EpisodeSplit:
    name: str
    episodes: tuple[Episode, ...]

    def validate(self, minimum_episodes: int = 1) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("split name is required")
        if len(self.episodes) < minimum_episodes:
            raise ValueError(f"{self.name} requires at least {minimum_episodes} episodes")
        seeds = [episode.seed for episode in self.episodes]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{self.name} contains duplicate episode seeds")
        for episode in self.episodes:
            episode.validate()


def assert_disjoint_splits(splits: Mapping[str, EpisodeSplit]) -> dict[str, list[int]]:
    if not splits:
        raise ValueError("episode partitions cannot be empty")
    seen: dict[int, str] = {}
    manifest: dict[str, list[int]] = {}
    for name, split in splits.items():
        if name != split.name:
            raise ValueError("split mapping key must match its name")
        split.validate()
        manifest[name] = [episode.seed for episode in split.episodes]
        for seed in manifest[name]:
            if seed in seen:
                raise ValueError(f"episode seed {seed} overlaps {seen[seed]} and {name}")
            seen[seed] = name
    return manifest


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def split_fingerprint(split: EpisodeSplit) -> str:
    split.validate()
    digest = hashlib.sha256()
    for episode in split.episodes:
        digest.update(canonical_bytes({"seed": episode.seed, "scarcity": episode.scarcity,
                                       "transitions": [row.to_dict() for row in episode.transitions]}))
        digest.update(b"\n")
    return digest.hexdigest()


def split_summary(split: EpisodeSplit, *, include_exposure: bool = False) -> dict:
    split.validate()
    rows = [row for episode in split.episodes for row in episode.transitions]
    result = {
        "name": split.name,
        "episode_seeds": [episode.seed for episode in split.episodes],
        "episodes": len(split.episodes),
        "transitions": len(rows),
        "deaths": sum(row.death for row in rows),
        "scarcity_values": sorted({float(episode.scarcity) for episode in split.episodes}),
        "sha256": split_fingerprint(split),
    }
    # Legacy fingerprints and summaries remain byte-for-byte interpretable.
    if include_exposure:
        result.update({
            "action_counts": [sum(row.action == action for row in rows) for action in range(len(ACTIONS))],
            "death_count_by_action": [sum(row.death and row.action == action for row in rows) for action in range(len(ACTIONS))],
            "truncations": sum(row.truncated for row in rows),
            "per_episode": [{"seed": episode.seed, "steps": len(episode.transitions),
                             "death": episode.transitions[-1].death,
                             "action_counts": [sum(row.action == action for row in episode.transitions)
                                               for action in range(len(ACTIONS))]}
                            for episode in split.episodes],
        })
    return result
