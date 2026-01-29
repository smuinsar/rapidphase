"""Tests for DeviceManager."""

import numpy as np
import pytest
import torch

from rapidphase.device.manager import DeviceManager, DeviceInfo


class TestDeviceManager:
    """Tests for the DeviceManager class."""

    def test_cpu_device(self):
        """Test CPU device initialization."""
        dm = DeviceManager(device="cpu")
        assert dm.device.type == "cpu"
        assert dm.dtype == torch.float64

    def test_auto_device(self):
        """Test auto device selection."""
        dm = DeviceManager(device="auto")
        # Should select some device without error
        assert dm.device.type in ("cpu", "cuda", "mps")

    def test_invalid_device(self):
        """Test that invalid device raises error."""
        with pytest.raises(ValueError, match="Unknown device"):
            DeviceManager(device="invalid")

    def test_cuda_not_available(self):
        """Test CUDA request when not available."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available")
        with pytest.raises(RuntimeError, match="CUDA requested but not available"):
            DeviceManager(device="cuda")

    def test_mps_not_available(self):
        """Test MPS request when not available."""
        if torch.backends.mps.is_available():
            pytest.skip("MPS is available")
        with pytest.raises(RuntimeError, match="MPS requested but not available"):
            DeviceManager(device="mps")

    def test_to_tensor_numpy(self, device_manager):
        """Test converting numpy array to tensor."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor = device_manager.to_tensor(arr)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.device == device_manager.device
        assert tensor.dtype == device_manager.dtype
        np.testing.assert_array_almost_equal(tensor.numpy(), arr)

    def test_to_tensor_complex(self, device_manager):
        """Test converting complex numpy array to tensor."""
        arr = np.array([[1.0 + 2.0j, 3.0 + 4.0j]])
        tensor = device_manager.to_tensor(arr)

        assert torch.is_complex(tensor)
        assert tensor.device == device_manager.device

    def test_to_numpy(self, device_manager):
        """Test converting tensor back to numpy."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device_manager.device)
        arr = device_manager.to_numpy(tensor)

        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_almost_equal(arr, [[1.0, 2.0], [3.0, 4.0]])

    def test_zeros_ones_empty(self, device_manager):
        """Test tensor creation utilities."""
        shape = (3, 4)

        zeros = device_manager.zeros(shape)
        assert zeros.shape == shape
        assert torch.all(zeros == 0)

        ones = device_manager.ones(shape)
        assert ones.shape == shape
        assert torch.all(ones == 1)

        empty = device_manager.empty(shape)
        assert empty.shape == shape

    def test_get_available_devices(self):
        """Test getting available device information."""
        info = DeviceManager.get_available_devices()

        assert isinstance(info, DeviceInfo)
        assert info.cpu is True  # CPU always available

        # Convert to dict
        info_dict = info.to_dict()
        assert "cpu" in info_dict
        assert "cuda" in info_dict
        assert "mps" in info_dict

    def test_synchronize(self, device_manager):
        """Test device synchronization (no-op on CPU)."""
        # Should not raise
        device_manager.synchronize()

    def test_clear_cache(self, device_manager):
        """Test cache clearing (no-op on CPU)."""
        # Should not raise
        device_manager.clear_cache()

    def test_memory_stats_cpu(self, device_manager):
        """Test memory stats returns None on CPU."""
        stats = device_manager.memory_stats()
        assert stats is None

    def test_repr(self, device_manager):
        """Test string representation."""
        repr_str = repr(device_manager)
        assert "DeviceManager" in repr_str
        assert "device=" in repr_str


class TestDeviceManagerMPS:
    """Tests specific to MPS device (Apple Silicon)."""

    @pytest.fixture
    def mps_manager(self):
        """Create MPS device manager if available."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
        return DeviceManager(device="mps")

    def test_mps_dtype_is_float32(self, mps_manager):
        """MPS should use float32 due to limited float64 support."""
        assert mps_manager.dtype == torch.float32

    def test_mps_tensor_creation(self, mps_manager):
        """Test tensor creation on MPS."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor = mps_manager.to_tensor(arr)

        assert tensor.device.type == "mps"
        assert tensor.dtype == torch.float32


class TestDeviceManagerCUDA:
    """Tests specific to CUDA device."""

    @pytest.fixture
    def cuda_manager(self):
        """Create CUDA device manager if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return DeviceManager(device="cuda")

    def test_cuda_dtype_is_float64(self, cuda_manager):
        """CUDA should use float64 for precision."""
        assert cuda_manager.dtype == torch.float64

    def test_cuda_tensor_creation(self, cuda_manager):
        """Test tensor creation on CUDA."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor = cuda_manager.to_tensor(arr)

        assert tensor.device.type == "cuda"

    def test_cuda_memory_stats(self, cuda_manager):
        """Test memory stats on CUDA."""
        stats = cuda_manager.memory_stats()

        assert stats is not None
        assert "allocated" in stats
        assert "reserved" in stats
