from __future__ import annotations

import numpy as np


def normalize01(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1]."""
    image = np.asarray(image, dtype=float)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value == min_value:
        return np.zeros_like(image)
    return (image - min_value) / (max_value - min_value)


def make_test_target(size: int = 256) -> np.ndarray:
    """Create a synthetic target with bars, point pairs and angular detail."""
    y, x = np.indices((size, size))
    xn = (x - size / 2) / (size / 2)
    yn = (y - size / 2) / (size / 2)
    radius = np.sqrt(xn**2 + yn**2)
    angle = np.arctan2(yn, xn)

    siemens = 0.5 + 0.5 * np.sign(np.sin(36 * angle))
    siemens *= radius < 0.48

    target = 0.08 * np.ones((size, size), dtype=float)
    target += 0.55 * siemens

    for idx, width in enumerate([16, 10, 7, 5, 3, 2]):
        y0 = 18 + idx * 15
        for col in range(18, 110, width * 2):
            target[y0 : y0 + 8, col : col + width] = 0.95
        x0 = 146 + idx * 15
        for row in range(18, 110, width * 2):
            target[row : row + width, x0 : x0 + 8] = 0.95

    points = [(168, 64), (174, 64), (190, 70), (198, 70), (216, 76), (228, 76)]
    for px, py in points:
        spot = np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2.0 * 1.4**2))
        target += 0.85 * spot

    target[178:226, 150:224] = np.maximum(target[178:226, 150:224], 0.28)
    target[185:190, 158:216] = 0.92
    target[199:204, 158:216] = 0.75
    target[213:218, 158:216] = 0.58

    return normalize01(target)


def apply_otf(image: np.ndarray, otf: np.ndarray) -> np.ndarray:
    """Blur an image with a centered OTF while preserving radiometry."""
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    degraded = np.fft.ifft2(np.fft.ifftshift(spectrum * otf))
    return np.real(degraded)


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float = 0.02,
    seed: int | None = 7,
    gain: float = 1.0,
    bias: float = 0.0,
) -> np.ndarray:
    """Apply detector gain/bias and add unclipped zero-mean Gaussian noise."""
    if gain == 0.0:
        raise ValueError("gain must be non-zero.")
    rng = np.random.default_rng(seed)
    return gain * image + bias + rng.normal(0.0, sigma, image.shape)
