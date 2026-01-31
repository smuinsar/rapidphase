"""
Device management for GPU-accelerated phase unwrapping.

Handles automatic device selection, tensor placement, and device-specific
dtype constraints (e.g., MPS requires float32).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

DeviceType = Literal["cpu", "cuda", "mps", "auto"] | str  # Also supports "cuda:0", "cuda:1", etc.


@dataclass
class DeviceInfo:
    """Information about available compute devices."""

    cpu: bool = True
    cuda: bool = False
    mps: bool = False
    cuda_devices: list[dict] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "cpu": self.cpu,
            "cuda": self.cuda,
            "mps": self.mps,
            "cuda_devices": self.cuda_devices,
        }


class DeviceManager:
    """
    Manages device selection and tensor operations for GPU acceleration.

    Handles:
    - Automatic detection of best available device (CUDA > MPS > CPU)
    - Device-specific dtype constraints (MPS requires float32)
    - Tensor placement and type conversion
    - Memory management utilities
    """

    def __init__(self, device: DeviceType = "auto"):
        """
        Initialize the device manager.

        Parameters
        ----------
        device : str
            Device to use: "cuda", "mps", "cpu", or "auto" (default).
            "auto" selects the best available device.
        """
        self._requested_device = device
        self._device = self._resolve_device(device)
        self._dtype = self._get_default_dtype()

    def _resolve_device(self, device: DeviceType) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            return self._get_best_device()
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device("cuda")
        elif isinstance(device, str) and device.startswith("cuda:"):
            # Handle specific CUDA device like "cuda:0", "cuda:1"
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            gpu_id = int(device.split(":")[1])
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device {gpu_id} not available (only {torch.cuda.device_count()} GPUs)")
            return torch.device(device)
        elif device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available")
            return torch.device("mps")
        elif device == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError(f"Unknown device: {device}")

    def _get_best_device(self) -> torch.device:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _get_default_dtype(self) -> torch.dtype:
        """Get the default dtype for the current device."""
        # MPS has limited float64 support, use float32
        if self._device.type == "mps":
            return torch.float32
        # CUDA and CPU can use float64 for better precision
        return torch.float64

    @property
    def device(self) -> torch.device:
        """The active torch device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """The default dtype for tensors on this device."""
        return self._dtype

    @property
    def device_type(self) -> str:
        """String representation of device type."""
        return self._device.type

    def to_tensor(
        self,
        data: np.ndarray | torch.Tensor,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Convert data to a tensor on the managed device.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            Input data to convert.
        dtype : torch.dtype, optional
            Desired dtype. If None, uses device default.

        Returns
        -------
        torch.Tensor
            Tensor on the managed device.
        """
        if dtype is None:
            dtype = self._dtype

        if isinstance(data, np.ndarray):
            # Handle complex numpy arrays
            if np.iscomplexobj(data):
                if dtype == torch.float32:
                    tensor = torch.from_numpy(data.astype(np.complex64))
                else:
                    tensor = torch.from_numpy(data.astype(np.complex128))
            else:
                tensor = torch.from_numpy(data.astype(np.float64))
                tensor = tensor.to(dtype)
        else:
            tensor = data.to(dtype)

        return tensor.to(self._device)

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a tensor back to numpy array.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to convert.

        Returns
        -------
        np.ndarray
            NumPy array on CPU.
        """
        return tensor.detach().cpu().numpy()

    def empty(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Create an empty tensor on the managed device."""
        if dtype is None:
            dtype = self._dtype
        return torch.empty(shape, dtype=dtype, device=self._device)

    def zeros(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Create a zero tensor on the managed device."""
        if dtype is None:
            dtype = self._dtype
        return torch.zeros(shape, dtype=dtype, device=self._device)

    def ones(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Create a ones tensor on the managed device."""
        if dtype is None:
            dtype = self._dtype
        return torch.ones(shape, dtype=dtype, device=self._device)

    def synchronize(self) -> None:
        """Synchronize the device (wait for GPU operations to complete)."""
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        elif self._device.type == "mps":
            torch.mps.synchronize()

    def memory_stats(self) -> dict | None:
        """Get memory statistics for GPU devices."""
        if self._device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(self._device),
                "reserved": torch.cuda.memory_reserved(self._device),
                "max_allocated": torch.cuda.max_memory_allocated(self._device),
            }
        elif self._device.type == "mps":
            return {
                "allocated": torch.mps.current_allocated_memory(),
            }
        return None

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps":
            torch.mps.empty_cache()

    @staticmethod
    def get_available_devices() -> DeviceInfo:
        """
        Get information about available compute devices.

        Returns
        -------
        DeviceInfo
            Information about available devices.
        """
        info = DeviceInfo()

        # Check CUDA
        info.cuda = torch.cuda.is_available()
        if info.cuda:
            info.cuda_devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info.cuda_devices.append({
                    "index": i,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "compute_capability": f"{props.major}.{props.minor}",
                })

        # Check MPS (Apple Silicon)
        info.mps = torch.backends.mps.is_available()

        return info

    def __repr__(self) -> str:
        return f"DeviceManager(device={self._device}, dtype={self._dtype})"
