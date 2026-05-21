import tempfile
import unittest
from pathlib import Path

import numpy as np

from esd_experiments.core.results import (
    arrays_to_records,
    load_result,
    records_to_arrays,
    save_result,
    stable_parameter_key,
)


class ResultFileTests(unittest.TestCase):
    def test_save_and_load_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.npz"
            arrays = {"x": np.array([1.0, 2.0]), "y": np.array([[3.0]])}
            metadata = {"figure": "test", "seed": 123}
            save_result(path, arrays, metadata)
            loaded_arrays, loaded_metadata = load_result(path)

        np.testing.assert_allclose(loaded_arrays["x"], arrays["x"])
        np.testing.assert_allclose(loaded_arrays["y"], arrays["y"])
        self.assertEqual(loaded_metadata, metadata)

    def test_record_column_roundtrip(self):
        records = [{"alpha": 0.0, "risk": 1.5}, {"alpha": 1.0, "risk": 0.5}]
        arrays = records_to_arrays(records)
        self.assertEqual(arrays["alpha"].shape, (2,))
        self.assertEqual(arrays_to_records(arrays), records)

    def test_stable_parameter_key_changes_with_payload(self):
        base = {"n": 100, "seed": 1, "D_values": [0, 1], "grid": [10, 20]}
        changed = {"n": 101, "seed": 1, "D_values": [0, 1], "grid": [10, 20]}
        self.assertEqual(stable_parameter_key(base), stable_parameter_key(dict(base)))
        self.assertNotEqual(stable_parameter_key(base), stable_parameter_key(changed))


if __name__ == "__main__":
    unittest.main()
