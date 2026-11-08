"""Sleep, Fatigue, and Dream Replay Subsystem.

Provides:
- Two-process homeostatic sleep pressure and brain state machine (FatigueController, SleepState).
- Counterfactual generative dream simulation with strict synthetic segregation (DreamReplayEngine, DreamEpisode).
- Salience-prioritized experience replay scheduler (ReplayScheduler, ReplayMemoryItem).
"""

from synapse.sleep_replay.fatigue_controller import (
    SleepState,
    FatigueController,
)
from synapse.sleep_replay.generative_dreams import (
    DreamEpisode,
    DreamReplayEngine,
)
from synapse.sleep_replay.replay_scheduler import (
    ReplayMemoryItem,
    ReplayScheduler,
)

__all__ = [
    "SleepState",
    "FatigueController",
    "DreamEpisode",
    "DreamReplayEngine",
    "ReplayMemoryItem",
    "ReplayScheduler",
]
