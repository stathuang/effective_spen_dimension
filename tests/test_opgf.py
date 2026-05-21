import unittest

import numpy as np

from esd_experiments.core.opgf import OPGFTrajectory, run_opgf


class OPGFTests(unittest.TestCase):
    def test_run_opgf_returns_typed_trajectory(self):
        trajectory = run_opgf(
            y=np.array([1.0, 0.5, -0.25]),
            lambda0=np.array([1.0, 0.5, 0.25]),
            n=20,
            D=0,
            learning_rate=1e-2,
            num_iterations=4,
            record_iterations=np.array([0, 2, 4]),
        )
        self.assertIsInstance(trajectory, OPGFTrajectory)
        np.testing.assert_array_equal(trajectory.iterations, np.array([0, 2, 4]))
        self.assertEqual(trajectory.lambda_path.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
