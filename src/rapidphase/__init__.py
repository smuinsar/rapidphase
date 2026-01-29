"""
RapidPhase: GPU-accelerated phase unwrapping using PyTorch.

This package provides fast phase unwrapping algorithms optimized for
GPU execution (CUDA and Apple Silicon MPS), with CPU fallback.
"""

from rapidphase.api import (
    unwrap,
    unwrap_dct,
    unwrap_irls,
    unwrap_irls_cg,
    get_available_devices,
    goldstein_filter,
)
from rapidphase.device.manager import DeviceManager
from rapidphase.filtering.goldstein import GoldsteinFilter

__version__ = "0.1.0"
__all__ = [
    "unwrap",
    "unwrap_dct",
    "unwrap_irls",
    "unwrap_irls_cg",
    "get_available_devices",
    "goldstein_filter",
    "DeviceManager",
    "GoldsteinFilter",
]
