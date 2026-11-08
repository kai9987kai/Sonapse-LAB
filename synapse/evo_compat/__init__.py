"""SONAPSE-EVO Compatibility and Structural Plasticity Subsystem.

Provides:
- Evolutionary circuit candidate representation (GrowthCandidate).
- Causal neurogenesis and two-phase commit transaction manager (EvolutionaryGrowthController, GrowthTransactionReceipt).
"""

from synapse.evo_compat.growth_transactions import (
    GrowthCandidate,
    GrowthTransactionReceipt,
    EvolutionaryGrowthController,
)

__all__ = [
    "GrowthCandidate",
    "GrowthTransactionReceipt",
    "EvolutionaryGrowthController",
]
