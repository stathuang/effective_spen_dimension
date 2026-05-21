import unittest

import numpy as np

from esd_experiments.core.sequence import (
    generate_sparse_misaligned_theta,
    support_indices,
    validate_zero_off_support,
)


class SequenceTests(unittest.TestCase):
    def test_support_indices_match_sparse_generator_rule(self):
        np.testing.assert_array_equal(
            support_indices(q=2.0, J=4, d=20),
            np.array([0, 3, 8, 15]),
        )

    def test_off_support_zero(self):
        theta = generate_sparse_misaligned_theta(q=2.0, p=2.5, J=4, d=20, C=5.0)
        support = set(support_indices(q=2.0, J=4, d=20).tolist())
        for idx, value in enumerate(theta):
            if idx in support:
                self.assertNotEqual(value, 0.0)
            else:
                self.assertEqual(value, 0.0)
        validate_zero_off_support(theta, np.array(sorted(support)))

    def test_validate_zero_off_support_rejects_nonzero_background(self):
        theta = generate_sparse_misaligned_theta(
            q=2.0, p=2.5, J=4, d=20, C=5.0, background=0.1
        )
        support = support_indices(q=2.0, J=4, d=20)
        with self.assertRaises(ValueError):
            validate_zero_off_support(theta, support)

    def test_out_of_range_indices_rejected(self):
        with self.assertRaises(ValueError):
            generate_sparse_misaligned_theta(q=3.0, p=2.5, J=4, d=20)


if __name__ == "__main__":
    unittest.main()
