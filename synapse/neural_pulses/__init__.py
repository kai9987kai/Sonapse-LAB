"""Propagating Neural Pulses and Traveling Graph Waves Subsystem.

Provides:
- Macroscale traveling wave pulse schemas and triggers (PulseEventType, PulseEvent, NeuromodulatoryPulseField).
- Discrete damped graph wave equation solver on region Laplacian (GraphWaveSolver).
- Soft phase reset and Kuramoto coherence protocol (NeuralPhaseResetProtocol).
- Bounded pulse router for connectome injection (BoundedPulseRouter).
"""

from synapse.neural_pulses.pulse_events import (
    PulseEventType,
    PulseEvent,
    NeuromodulatoryPulseField,
)
from synapse.neural_pulses.graph_wave_solver import (
    GraphWaveSolver,
)
from synapse.neural_pulses.phase_reset import (
    NeuralPhaseResetProtocol,
)
from synapse.neural_pulses.pulse_router import (
    BoundedPulseRouter,
)

__all__ = [
    "PulseEventType",
    "PulseEvent",
    "NeuromodulatoryPulseField",
    "GraphWaveSolver",
    "NeuralPhaseResetProtocol",
    "BoundedPulseRouter",
]
