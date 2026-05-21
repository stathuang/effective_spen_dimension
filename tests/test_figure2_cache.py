import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from esd_experiments.core.results import load_result
from esd_experiments.experiments.figure2_opgf_error_esd import (
    build_config,
    log_record_iterations,
    parameter_key,
    run_experiment,
)


class Figure2CacheTests(unittest.TestCase):
    def test_parameter_key_changes_with_experiment_contract(self):
        config = build_config(
            fast=True,
            n=100,
            d=40,
            J=3,
            q=2.0,
            D_values=(0,),
            num_experiments=1,
            num_iterations=20,
            log_start=5,
            log_step=0.5,
        )
        record_iterations = log_record_iterations(
            config.num_iterations, config.log_start, config.log_step
        )
        changed = build_config(
            fast=True,
            seed=config.seed + 1,
            n=100,
            d=40,
            J=3,
            q=2.0,
            D_values=(0,),
            num_experiments=1,
            num_iterations=20,
            log_start=5,
            log_step=0.5,
        )
        self.assertNotEqual(
            parameter_key(config, record_iterations),
            parameter_key(changed, record_iterations),
        )

    def test_use_cache_does_not_reuse_mismatched_data_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path = tmp / "figure2.npz"
            cache_dir = tmp / "cache"
            out1 = tmp / "first.pdf"
            out2 = tmp / "second.pdf"
            env = dict(os.environ)
            env["MPLBACKEND"] = "Agg"
            common = [
                sys.executable,
                "-m",
                "esd_experiments.scripts.figure2_opgf_error_esd",
                "--fast",
                "--data-output",
                str(data_path),
                "--cache-dir",
                str(cache_dir),
                "--D-values",
                "0",
                "--num-experiments",
                "1",
                "--n",
                "80",
                "--d",
                "30",
                "--J",
                "3",
                "--q",
                "2.0",
                "--num-iterations",
                "20",
                "--log-start",
                "5",
                "--log-step",
                "0.5",
            ]
            subprocess.run(common + ["--output", str(out1)], cwd=Path.cwd(), env=env, check=True)
            _, first_metadata = load_result(data_path)
            subprocess.run(
                common + ["--seed", "999", "--use-cache", "--output", str(out2)],
                cwd=Path.cwd(),
                env=env,
                check=True,
            )
            _, second_metadata = load_result(data_path)
            self.assertTrue(out1.exists())
            self.assertTrue(out2.exists())

        self.assertNotEqual(first_metadata["parameter_key"], second_metadata["parameter_key"])
        self.assertEqual(second_metadata["seed"], 999)

    def test_parallel_matches_serial_results(self):
        config = build_config(
            fast=True,
            n=80,
            d=30,
            J=3,
            q=2.0,
            D_values=(0, 1),
            num_experiments=2,
            num_iterations=20,
            log_start=5,
            log_step=0.5,
        )
        record_iterations = log_record_iterations(
            config.num_iterations, config.log_start, config.log_step
        )
        serial_arrays, serial_metadata = run_experiment(
            config, record_iterations, workers=1
        )
        parallel_arrays, parallel_metadata = run_experiment(
            config, record_iterations, workers=2
        )

        self.assertEqual(serial_metadata["parameter_key"], parallel_metadata["parameter_key"])
        for key in ("time_points", "sq_errors", "d_daggers"):
            np.testing.assert_allclose(serial_arrays[key], parallel_arrays[key])


if __name__ == "__main__":
    unittest.main()
