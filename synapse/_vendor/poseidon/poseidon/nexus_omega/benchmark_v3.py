"""Nonstationary phase-world benchmark with one hidden storm-schedule shift."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np

from poseidon.nexus_omega.benchmark_v2 import POMDPPhaseWorldV2


class POMDPPhaseWorldV3(POMDPPhaseWorldV2):
    """Versioned v2 world that changes its pulse schedule once mid-episode.

    The schedule change is hidden. The next observation is generated under the
    new schedule, while the transition that crosses the boundary is evaluated
    under the old schedule. Phase is mapped proportionally into the new cycle,
    avoiding a free reset to a cue or to a safe state.
    """

    VERSION = "nexus-phaseworld-v3-hidden-schedule-shift"

    def __init__(
        self,
        seed: int = 12345,
        max_steps: int = 128,
        storm_interval: int = 8,
        storm_duration: int = 6,
        schedule: Iterable[Tuple[int, int, int]] = ((64, 10, 6),),
    ) -> None:
        entries = tuple((int(step), int(interval), int(duration)) for step, interval, duration in schedule)
        if any(step < 1 or interval < 2 or duration < 1 or duration >= interval for step, interval, duration in entries):
            raise ValueError("schedule entries must be (positive step, interval >= 2, 1 <= duration < interval)")
        if tuple(sorted(entries)) != entries or len({entry[0] for entry in entries}) != len(entries):
            raise ValueError("schedule changes must be sorted and have unique step indices")
        super().__init__(seed, max_steps, storm_interval, storm_duration)
        self.initial_storm_interval = int(storm_interval)
        self.initial_storm_duration = int(storm_duration)
        self.schedule = entries
        self.schedule_index = 0
        self.last_phase_change: Dict[str, Any] | None = None

    def reset(self) -> np.ndarray:
        self.storm_interval = self.initial_storm_interval
        self.storm_duration = self.initial_storm_duration
        self.schedule_index = 0
        self.last_phase_change = None
        return super().reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        next_obs, reward, done, info = super().step(action)
        info = dict(info)
        info["version"] = self.VERSION
        if self.schedule_index >= len(self.schedule):
            return next_obs, reward, done, info

        change_step, next_interval, next_duration = self.schedule[self.schedule_index]
        if self.step_idx != change_step:
            return next_obs, reward, done, info

        old_interval = int(self.storm_interval)
        old_duration = int(self.storm_duration)
        old_phase = int(self.cycle_step)
        phase_fraction = old_phase / float(old_interval)
        self.storm_interval = next_interval
        self.storm_duration = next_duration
        self.cycle_step = min(next_interval - 1, int(np.floor(phase_fraction * next_interval)))
        self.schedule_index += 1
        self.last_phase_change = {
            "step": change_step,
            "old_interval": old_interval,
            "old_duration": old_duration,
            "new_interval": next_interval,
            "new_duration": next_duration,
            "old_phase": old_phase,
            "new_phase": int(self.cycle_step),
        }
        info.update({
            "version": self.VERSION,
            "phase_changed": True,
            "phase_change": dict(self.last_phase_change),
            "storm_interval_after_transition": int(self.storm_interval),
            "storm_duration_after_transition": int(self.storm_duration),
        })
        # Rebuild the returned observation under the new schedule. It remains
        # a deterministic function of the episode seed and transition history.
        return self._get_obs(), reward, done, info
