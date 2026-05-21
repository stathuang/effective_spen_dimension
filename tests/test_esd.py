import unittest

import numpy as np

from esd_experiments.core.esd import compute_H, compute_esd, pc_oracle_error


class ESDTests(unittest.TestCase):
    def test_hand_computed_H(self):
        theta = np.array([3.0, 2.0, 1.0])
        lambdas = np.array([3.0, 2.0, 1.0])
        np.testing.assert_allclose(compute_H(theta, lambdas), np.array([5.0, 0.5, 0.0]))

    def test_H_d_is_zero(self):
        theta = np.array([1.0, -4.0, 2.0, 7.0])
        lambdas = np.array([0.2, 0.4, 0.1, 0.3])
        self.assertEqual(compute_H(theta, lambdas)[-1], 0.0)

    def test_esd_cutoffs(self):
        theta = np.array([3.0, 2.0, 1.0])
        lambdas = np.array([3.0, 2.0, 1.0])
        self.assertEqual(compute_esd(theta, lambdas, 0.1), 3)
        self.assertEqual(compute_esd(theta, lambdas, 0.6), 2)
        self.assertEqual(compute_esd(theta, lambdas, 6.0), 1)

    def test_pc_oracle_uses_unsquared_coefficients(self):
        theta = np.array([3.0, 2.0, 1.0])
        y = theta.copy()
        lambdas = np.array([3.0, 2.0, 1.0])
        error, d_dagger = pc_oracle_error(y, theta, lambdas, 0.6)
        self.assertEqual(d_dagger, 2)
        self.assertAlmostEqual(error, 1.0)


if __name__ == "__main__":
    unittest.main()

