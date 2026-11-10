"""Optional synthetic Dual-Soma Axodendritic Graft research components."""

from .bridge import DSAGBusChannel
from .config import DSAGSpecV1
from .model import DualSomaAxodendriticGraft
from .packet_adapter import DSAGLocalEnvelope

__all__ = ["DSAGBusChannel", "DSAGSpecV1", "DualSomaAxodendriticGraft", "DSAGLocalEnvelope"]
