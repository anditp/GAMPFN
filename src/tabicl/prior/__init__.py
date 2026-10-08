"""Synthetic prior data generation for TabICL pre-training.

Only :class:`PriorDataset` and :class:`GAMConfig` are public. All other symbols
in this subpackage are internal pre-training utilities and may change without
notice. A CLI entry point is provided via ``python -m tabicl.prior``.
"""

from ._dataset import PriorDataset
from ._gam import GAMConfig

__all__ = ["GAMConfig", "PriorDataset"]
