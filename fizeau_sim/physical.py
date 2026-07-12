from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicalOptics:
    """Physical parameters used by the interactive simulation."""

    aperture_diameter_m: float = 80e-3
    focal_length_m: float = 28.0
    detector_pixel_m: float = 6.5e-6
    image_size: int = 256

    @property
    def nyquist_frequency_cpm(self) -> float:
        return 1.0 / (2.0 * self.detector_pixel_m)


def detector_frequency_grid(params: PhysicalOptics) -> tuple[np.ndarray, np.ndarray]:
    """Centered detector spatial-frequency grid in cycles per meter."""
    frequencies = np.fft.fftshift(
        np.fft.fftfreq(params.image_size, d=params.detector_pixel_m)
    )
    return np.meshgrid(frequencies, frequencies, indexing="xy")


def circular_aperture_mtf(radius_cpm: np.ndarray, cutoff_cpm: float) -> np.ndarray:
    """Diffraction-limited incoherent MTF envelope of a circular aperture."""
    normalized_radius = radius_cpm / cutoff_cpm
    mtf = np.zeros_like(radius_cpm, dtype=float)
    inside = normalized_radius <= 1.0
    clipped = np.clip(normalized_radius[inside], 0.0, 1.0)
    mtf[inside] = (2.0 / np.pi) * (
        np.arccos(clipped) - clipped * np.sqrt(np.maximum(0.0, 1.0 - clipped**2))
    )
    return mtf


def otf_for_wavelength(
    centers_m: np.ndarray,
    params: PhysicalOptics,
    wavelength_m: float,
) -> np.ndarray:
    """Compute a sparse-array OTF on the detector frequency grid."""
    fx, fy = detector_frequency_grid(params)
    cutoff_cpm = params.aperture_diameter_m / (wavelength_m * params.focal_length_m)
    otf = np.zeros((params.image_size, params.image_size), dtype=np.complex128)
    for center_i in centers_m:
        for center_j in centers_m:
            shift_x, shift_y = (center_i - center_j) / (
                wavelength_m * params.focal_length_m
            )
            radius = np.sqrt((fx - shift_x) ** 2 + (fy - shift_y) ** 2)
            otf += circular_aperture_mtf(radius, cutoff_cpm)
    center = params.image_size // 2
    return otf / otf[center, center]


def mtf_from_centered_otf(otf: np.ndarray) -> np.ndarray:
    """Return the OTF magnitude normalized to a unit peak."""
    mtf = np.abs(otf)
    peak = float(np.max(mtf))
    return mtf if peak == 0.0 else mtf / peak
