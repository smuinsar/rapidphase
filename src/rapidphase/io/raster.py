"""
Raster I/O utilities using rasterio (optional dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def _check_rasterio() -> None:
    """Check if rasterio is available."""
    try:
        import rasterio  # noqa: F401
    except ImportError:
        raise ImportError(
            "rasterio is required for raster I/O. "
            "Install with: pip install rapidphase[raster]"
        )


def read_phase(
    path: str,
    band: int = 1,
) -> tuple[np.ndarray, dict]:
    """
    Read phase data from a raster file.

    Parameters
    ----------
    path : str
        Path to the raster file (GeoTIFF, etc.).
    band : int
        Band number to read (1-indexed).

    Returns
    -------
    phase : np.ndarray
        Phase data as 2D array.
    metadata : dict
        Raster metadata (CRS, transform, etc.).
    """
    _check_rasterio()
    import rasterio

    with rasterio.open(path) as src:
        phase = src.read(band).astype(np.float64)
        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[band - 1],
            "nodata": src.nodata,
        }

    return phase, metadata


def read_coherence(
    path: str,
    band: int = 1,
) -> tuple[np.ndarray, dict]:
    """
    Read coherence data from a raster file.

    Parameters
    ----------
    path : str
        Path to the raster file.
    band : int
        Band number to read (1-indexed).

    Returns
    -------
    coherence : np.ndarray
        Coherence data as 2D array, values in [0, 1].
    metadata : dict
        Raster metadata.
    """
    _check_rasterio()
    import rasterio

    with rasterio.open(path) as src:
        coherence = src.read(band).astype(np.float64)
        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[band - 1],
            "nodata": src.nodata,
        }

    # Ensure values are in valid range
    coherence = np.clip(coherence, 0.0, 1.0)

    return coherence, metadata


def write_phase(
    path: str,
    phase: np.ndarray,
    metadata: dict | None = None,
    **kwargs,
) -> None:
    """
    Write phase data to a raster file.

    Parameters
    ----------
    path : str
        Output path.
    phase : np.ndarray
        Phase data to write.
    metadata : dict, optional
        Raster metadata (CRS, transform, etc.).
        If None, writes without georeferencing.
    **kwargs
        Additional arguments passed to rasterio.open().
    """
    _check_rasterio()
    import rasterio

    height, width = phase.shape

    # Default profile
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
    }

    # Add metadata if provided
    if metadata is not None:
        if "crs" in metadata:
            profile["crs"] = metadata["crs"]
        if "transform" in metadata:
            profile["transform"] = metadata["transform"]
        if "nodata" in metadata and metadata["nodata"] is not None:
            profile["nodata"] = metadata["nodata"]

    # Override with kwargs
    profile.update(kwargs)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(phase.astype(np.float32), 1)


def read_complex_interferogram(
    path: str,
    real_band: int = 1,
    imag_band: int = 2,
) -> tuple[np.ndarray, dict]:
    """
    Read a complex interferogram from a raster file.

    Parameters
    ----------
    path : str
        Path to the raster file.
    real_band : int
        Band containing real part.
    imag_band : int
        Band containing imaginary part.

    Returns
    -------
    igram : np.ndarray
        Complex interferogram.
    metadata : dict
        Raster metadata.
    """
    _check_rasterio()
    import rasterio

    with rasterio.open(path) as src:
        real = src.read(real_band).astype(np.float64)
        imag = src.read(imag_band).astype(np.float64)
        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }

    igram = real + 1j * imag

    return igram, metadata
