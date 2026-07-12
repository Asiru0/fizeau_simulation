from __future__ import annotations

import numpy as np


def wiener_filter(
    observed: np.ndarray,
    otf: np.ndarray,
    noise_to_signal: float = 2e-3,
    gain: float = 1.0,
    bias: float = 0.0,
) -> np.ndarray:
    """Reconstruct one observation with Wiener regularization."""
    transfer = _effective_otf(otf, gain)
    observed_spectrum = np.fft.fftshift(np.fft.fft2(observed - bias))
    estimate_spectrum = (
        observed_spectrum
        * np.conj(transfer)
        / (np.abs(transfer) ** 2 + noise_to_signal)
    )
    return _real_image(estimate_spectrum)


def normalized_mtf(otf: np.ndarray) -> np.ndarray:
    """Return an OTF magnitude normalized to a unit peak."""
    mtf = np.abs(otf)
    peak = float(mtf.max())
    return np.zeros_like(mtf, dtype=float) if peak == 0.0 else mtf / peak


def soft_frequency_weights(
    otf: np.ndarray,
    mtf_threshold: float = 0.03,
    transition_width: float = 0.04,
) -> np.ndarray:
    """Build smooth frequency weights around an MTF threshold."""
    if not 0.0 <= mtf_threshold <= 1.0:
        raise ValueError("mtf_threshold must be between 0 and 1.")
    if transition_width <= 0.0:
        raise ValueError("transition_width must be positive.")
    lower = max(0.0, mtf_threshold - transition_width / 2.0)
    upper = min(1.0, mtf_threshold + transition_width / 2.0)
    if upper == lower:
        return (normalized_mtf(otf) >= mtf_threshold).astype(float)
    ramp = np.clip((normalized_mtf(otf) - lower) / (upper - lower), 0.0, 1.0)
    return ramp * ramp * (3.0 - 2.0 * ramp)


def adaptive_wiener_regularization(
    otf: np.ndarray,
    base_regularization: float,
    strength: float = 4.0,
    exponent: float = 2.0,
) -> np.ndarray:
    """Increase Wiener regularization smoothly as MTF falls."""
    if base_regularization < 0.0:
        raise ValueError("base_regularization must be non-negative.")
    if strength < 0.0:
        raise ValueError("strength must be non-negative.")
    if exponent <= 0.0:
        raise ValueError("exponent must be positive.")
    return base_regularization * (
        1.0 + strength * (1.0 - normalized_mtf(otf)) ** exponent
    )


def multi_pose_soft_adaptive_wiener(
    observations: list[np.ndarray],
    otfs: list[np.ndarray],
    noise_variances: list[float] | None = None,
    base_regularization: float = 2e-2,
    mtf_threshold: float = 0.03,
    transition_width: float = 0.04,
) -> np.ndarray:
    """Fuse observations with precision weights and adaptive regularization."""
    if len(observations) != len(otfs):
        raise ValueError("observations and otfs must have the same length.")
    if not observations:
        raise ValueError("At least one observation is required.")
    variances = [1.0] * len(observations) if noise_variances is None else noise_variances
    if len(variances) != len(observations):
        raise ValueError("noise_variances must have the same length as observations.")
    if any(variance <= 0.0 for variance in variances):
        raise ValueError("noise_variances must be positive.")

    precisions = 1.0 / np.asarray(variances, dtype=float)
    precisions /= float(np.mean(precisions))
    numerator = np.zeros_like(otfs[0], dtype=np.complex128)
    denominator = np.zeros_like(otfs[0], dtype=float)
    for observed, otf, precision in zip(observations, otfs, precisions):
        observed_spectrum = np.fft.fftshift(np.fft.fft2(observed))
        numerator += precision * np.conj(otf) * observed_spectrum
        denominator += precision * np.abs(otf) ** 2

    composite_mtf = np.sqrt(denominator)
    composite_mtf /= max(float(composite_mtf.max()), 1e-12)
    weights = soft_frequency_weights(composite_mtf, mtf_threshold, transition_width)
    regularization = adaptive_wiener_regularization(
        composite_mtf, base_regularization
    )
    denominator_with_prior = denominator + regularization
    estimate_spectrum = np.divide(
        weights * numerator,
        denominator_with_prior,
        out=np.zeros_like(numerator),
        where=denominator_with_prior > 0.0,
    )
    return _real_image(estimate_spectrum)


def _effective_otf(otf: np.ndarray, gain: float) -> np.ndarray:
    if gain == 0.0:
        raise ValueError("gain must be non-zero.")
    return gain * otf


def _real_image(centered_spectrum: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft2(np.fft.ifftshift(centered_spectrum)))
