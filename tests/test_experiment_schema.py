import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

from esd_experiments.core.results import save_result
from esd_experiments.experiments.figure1_span_profile import build_config as build_figure1_config
from esd_experiments.experiments.figure1_span_profile import run_experiment as run_figure1
from esd_experiments.experiments.figure2_opgf_error_esd import (
    build_config as build_figure2_config,
    log_record_iterations,
    run_experiment as run_figure2,
)
from esd_experiments.experiments.figure3_linear_model import (
    LinearCaseConfig,
    arrays_for_case,
    metadata_for_case,
    run_case,
)
from esd_experiments.experiments.figure4_rkhs_kpcpe import (
    RKHSConfig,
    arrays_for_records,
    metadata_for_config,
    simulate,
)
from esd_experiments.experiments.figure5_pathwise_linear_net import history_to_arrays
from esd_experiments.plots.figure5_pathwise_linear_net import plot_from_data


class ExperimentSchemaTests(unittest.TestCase):
    def test_fast_experiment_schemas(self):
        f1_config = build_figure1_config(
            fast=True, d=40, J=3, q_values=(1.0, 2.0), num_iterations=20
        )
        f1_arrays, _ = run_figure1(f1_config)
        self.assertIn("profiles", f1_arrays)
        self.assertIn("lambda_paths", f1_arrays)

        f2_config = build_figure2_config(
            fast=True,
            n=80,
            d=30,
            J=3,
            q=2.0,
            D_values=(0,),
            num_experiments=1,
            num_iterations=20,
            log_start=5,
            log_step=0.5,
        )
        record_iterations = log_record_iterations(
            f2_config.num_iterations, f2_config.log_start, f2_config.log_step
        )
        f2_arrays, f2_metadata = run_figure2(f2_config, record_iterations)
        self.assertIn("sq_errors", f2_arrays)
        self.assertIn("parameter_key", f2_metadata)

        f3_config = LinearCaseConfig(
            case_id=1,
            n=30,
            p=40,
            sigma0_sq=1.0,
            replicates=2,
            design_decay=0.95,
            beta_decay="power",
            beta_power=0.2,
            alpha_grid=np.array([0.0, 1.0]),
            output_name="dummy.pdf",
        )
        f3_records, f3_raw = run_case(f3_config, seed=10)
        f3_arrays = arrays_for_case(f3_records, f3_raw)
        self.assertIn("loss_by_alpha_k_replication", f3_arrays)
        self.assertEqual(metadata_for_case(f3_config, 10, True)["figure"], "figure3_linear_model")

        f4_config = RKHSConfig(
            n=30,
            J=20,
            D=5,
            replicates=2,
            sigma0_sq=1.0,
            spectrum_decay=1.1,
            signal_decay=4.0,
            alpha_grid=np.array([0.0, 1.0]),
            sup_norm_grid_size=200,
            output_name="dummy.pdf",
        )
        f4_records, f4_sup_norm_sq, f4_raw = simulate(f4_config, seed=11)
        f4_arrays = arrays_for_records(f4_records, f4_raw)
        self.assertIn("risk_by_alpha_k", f4_arrays)
        self.assertIn("sigma_eff2", metadata_for_config(f4_config, 11, True, f4_sup_norm_sq))

        f5_arrays = history_to_arrays({"epoch": [0, 1], "loss": [1.0, 0.5], "esd": [2, 1], "risk": [0.4, 0.3]})
        self.assertEqual(set(f5_arrays), {"epoch", "loss", "esd", "risk"})

    def test_plot_only_from_saved_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path = tmp / "figure5.npz"
            output_path = tmp / "figure5.pdf"
            save_result(
                data_path,
                {"epoch": [0, 1], "loss": [1.0, 0.5], "esd": [2, 1], "risk": [0.4, 0.3]},
                {"figure": "figure5_pathwise_linear_net"},
            )
            plot_from_data(data_path, output_path)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
