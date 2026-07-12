from __future__ import annotations

import numpy as np


def mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.mean((reference - estimate) ** 2))


def psnr(reference: np.ndarray, estimate: np.ndarray, data_range: float = 1.0) -> float:
    error = mse(reference, estimate)
    if error == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range / np.sqrt(error)))


def foreground_balanced_psnr(
    reference: np.ndarray,
    estimate: np.ndarray,
    foreground_threshold: float = 0.05,
    data_range: float = 1.0,
) -> float:
    """PSNR with equal foreground/background influence for sparse targets."""
    foreground = reference >= foreground_threshold * max(float(reference.max()), 1e-12)
    if not np.any(foreground) or np.all(foreground):
        return psnr(reference, estimate, data_range)
    squared_error = (reference - estimate) ** 2
    balanced_error = 0.5 * (
        float(np.mean(squared_error[foreground]))
        + float(np.mean(squared_error[~foreground]))
    )
    if balanced_error == 0.0:
        return float("inf")
    return float(20.0 * np.log10(data_range / np.sqrt(balanced_error)))


def ssim(reference: np.ndarray, estimate: np.ndarray, data_range: float = 1.0) -> float:
    """Global SSIM approximation without extra dependencies."""
    ref = reference.astype(float)
    est = estimate.astype(float)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = float(np.mean(ref))
    mu_y = float(np.mean(est))
    sigma_x = float(np.var(ref))
    sigma_y = float(np.var(est))
    sigma_xy = float(np.mean((ref - mu_x) * (est - mu_y)))

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / denominator)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(image.astype(float))
    return np.sqrt(gx**2 + gy**2)


def gradient_similarity(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Cosine similarity between gradient magnitude maps."""
    ref_g = gradient_magnitude(reference).ravel()
    est_g = gradient_magnitude(estimate).ravel()
    denom = np.linalg.norm(ref_g) * np.linalg.norm(est_g)
    if denom == 0:
        return 0.0
    return float(np.dot(ref_g, est_g) / denom)

