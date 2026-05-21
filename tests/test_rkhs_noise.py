import unittest

import numpy as np

from esd_experiments.core.esd import compute_esd
from esd_experiments.core.rkhs import (
    effective_noise_from_sup_norm,
    estimate_sup_norm_from_cosine_coeffs,
    pathwise_esd_from_eigendecomposition,
)


class RKHSNoiseTests(unittest.TestCase):
    def test_effective_noise_uses_sup_norm(self):
        self.assertAlmostEqual(effective_noise_from_sup_norm(1.0, 4.0, 10), 0.5)

    def test_sup_norm_estimate_positive(self):
        theta = np.array([1.0, 0.5])
        self.assertGreater(estimate_sup_norm_from_cosine_coeffs(theta, num_grid=200), 0.0)

    def test_pathwise_esd_delegates_to_shared_contract(self):
        theta = np.array([3.0, 2.0, 1.0])
        lambdas = np.array([3.0, 2.0, 1.0])
        expected = compute_esd(theta, lambdas, 0.6)
        self.assertEqual(pathwise_esd_from_eigendecomposition(theta, lambdas, 0.6), expected)


if __name__ == "__main__":
    unittest.main()

