import unittest

import numpy as np

from fizeau_sim.metrics import foreground_balanced_psnr, psnr
from fizeau_sim.physical import PhysicalOptics, mtf_from_centered_otf, otf_for_wavelength
from fizeau_sim.reconstruct import (
    adaptive_wiener_regularization,
    multi_pose_soft_adaptive_wiener,
    soft_frequency_weights,
    wiener_filter,
)
from fizeau_sim.simulate import add_gaussian_noise, apply_otf, make_test_target


class RadiometryTests(unittest.TestCase):
    def test_target_is_normalized_and_nonconstant(self) -> None:
        target = make_test_target(64)
        self.assertEqual(target.shape, (64, 64))
        self.assertAlmostEqual(float(target.min()), 0.0)
        self.assertAlmostEqual(float(target.max()), 1.0)
        self.assertGreater(float(np.std(target)), 0.01)

    def test_apply_otf_preserves_linear_scale(self) -> None:
        image = np.arange(16, dtype=float).reshape(4, 4) / 10.0
        identity_otf = np.ones_like(image)
        degraded = apply_otf(image, identity_otf)
        np.testing.assert_allclose(degraded, image, atol=1e-12)
        np.testing.assert_allclose(
            apply_otf(2.5 * image, identity_otf), 2.5 * degraded, atol=1e-12
        )

    def test_noise_is_repeatable_and_unclipped(self) -> None:
        image = np.array([[0.0, 1.0]])
        first = add_gaussian_noise(image, sigma=0.1, seed=12)
        second = add_gaussian_noise(image, sigma=0.1, seed=12)
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(
            add_gaussian_noise(image, sigma=0.0, gain=2.0, bias=-0.25),
            [[-0.25, 1.75]],
        )

    def test_physical_otf_is_center_normalized(self) -> None:
        params = PhysicalOptics(image_size=32)
        centers = np.array([[-0.05, 0.0], [0.05, 0.0]])
        otf = otf_for_wavelength(centers, params, 550e-9)
        self.assertEqual(otf.shape, (32, 32))
        self.assertAlmostEqual(float(np.real(otf[16, 16])), 1.0)
        self.assertAlmostEqual(float(mtf_from_centered_otf(otf).max()), 1.0)

    def test_wiener_preserves_identity_observation(self) -> None:
        image = np.arange(16, dtype=float).reshape(4, 4) / 10.0
        reconstructed = wiener_filter(image, np.ones_like(image), noise_to_signal=0.0)
        np.testing.assert_allclose(reconstructed, image, atol=1e-12)

    def test_multi_pose_soft_wiener_uses_precision_weights(self) -> None:
        image = np.arange(16, dtype=float).reshape(4, 4) / 10.0
        identity_otf = np.ones_like(image)
        reconstructed = multi_pose_soft_adaptive_wiener(
            [image, image + 1.0],
            [identity_otf, identity_otf],
            noise_variances=[1e-6, 1.0],
            base_regularization=0.0,
            mtf_threshold=0.0,
            transition_width=0.1,
        )
        np.testing.assert_allclose(reconstructed, image, atol=1e-5)

    def test_soft_weights_and_regularization_follow_mtf(self) -> None:
        otf = np.array([[0.0, 0.6, 1.0]])
        weights = soft_frequency_weights(otf, mtf_threshold=0.6, transition_width=0.4)
        regularization = adaptive_wiener_regularization(
            otf, base_regularization=0.1, strength=3.0, exponent=2.0
        )
        np.testing.assert_allclose(weights, [[0.0, 0.5, 1.0]], atol=1e-12)
        np.testing.assert_allclose(regularization, [[0.4, 0.148, 0.1]], atol=1e-12)

    def test_foreground_psnr_penalizes_lost_sparse_sources(self) -> None:
        reference = np.zeros((32, 32), dtype=float)
        reference[15:17, 15:17] = 1.0
        estimate = np.zeros_like(reference)
        self.assertLess(foreground_balanced_psnr(reference, estimate), psnr(reference, estimate))


if __name__ == "__main__":
    unittest.main()
