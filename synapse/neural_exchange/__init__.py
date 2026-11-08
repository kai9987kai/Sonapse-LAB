"""Cross-Model Neural Exchange and Parameter Transfer Subsystem.

Provides:
- Spontaneous cross-model neural parameter exchange controller (SpontaneousNeuralExchangeController).
- Causal validation receipts and atomic rollback protection (ExchangeReceipt).
"""

from synapse.neural_exchange.exchange_policy import (
    SpontaneousNeuralExchangeController,
    ExchangeReceipt,
)

__all__ = [
    "SpontaneousNeuralExchangeController",
    "ExchangeReceipt",
]
