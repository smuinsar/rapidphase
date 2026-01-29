"""
Base class for phase unwrapping algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from rapidphase.device.manager import DeviceManager


class BaseUnwrapper(ABC):
    """
    Abstract base class for phase unwrapping algorithms.

    All unwrapper implementations should inherit from this class
    and implement the `unwrap` method.
    """

    def __init__(self, device_manager: DeviceManager):
        """
        Initialize the unwrapper.

        Parameters
        ----------
        device_manager : DeviceManager
            Device manager for GPU/CPU operations.
        """
        self.dm = device_manager

    @abstractmethod
    def unwrap(
        self,
        phase: torch.Tensor,
        coherence: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Unwrap a 2D phase field.

        Parameters
        ----------
        phase : torch.Tensor
            Wrapped phase of shape (H, W), values in [-pi, pi].
        coherence : torch.Tensor, optional
            Coherence/correlation map of shape (H, W), values in [0, 1].
            Used for weighted unwrapping if supported.
        **kwargs
            Algorithm-specific parameters.

        Returns
        -------
        torch.Tensor
            Unwrapped phase of shape (H, W).
        """
        pass

    def __call__(
        self,
        phase: np.ndarray | torch.Tensor,
        coherence: np.ndarray | torch.Tensor | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Convenience method to unwrap with numpy input/output.

        Parameters
        ----------
        phase : np.ndarray or torch.Tensor
            Wrapped phase.
        coherence : np.ndarray or torch.Tensor, optional
            Coherence map.
        **kwargs
            Passed to unwrap().

        Returns
        -------
        np.ndarray
            Unwrapped phase as numpy array.
        """
        # Convert to tensor
        phase_t = self.dm.to_tensor(phase)
        coh_t = None
        if coherence is not None:
            coh_t = self.dm.to_tensor(coherence)

        # Unwrap
        result = self.unwrap(phase_t, coh_t, **kwargs)

        # Convert back to numpy
        return self.dm.to_numpy(result)

    @property
    def device(self) -> torch.device:
        """The device used for computation."""
        return self.dm.device

    @property
    def dtype(self) -> torch.dtype:
        """The dtype used for computation."""
        return self.dm.dtype
