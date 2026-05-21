# Figure Reproduction Contract

This file records the figure/data contract for the public ESD experiments
release.  The contract was checked against the compiled paper on 2026-05-21.
The paper source files and compiled PDF are not included in this code release;
this release contains the code, tests, bundled full-run data, and plot-only
entry points needed to regenerate the numerical figures.

## Shared Rules

- All ESD calculations use `esd_experiments.core.esd.compute_esd`.
- ESD inputs are unsquared signal coefficients.
- `H(k)=k^{-1}\sum_{i=k+1}^d \theta_{\pi_i}^{*2}` and `H(d)=0`.
- Error bars in Figures 2--4 are standard errors.
- Experiment scripts must save numerical data before plotting.
- Plot-only modules under `esd_experiments.plots` must redraw each figure from
  saved `.npz` data without rerunning the experiment.
- Active experiment defaults use random seed `2026` unless `--seed` is supplied.
- Bundled full-run saved data live under `data/figure_data/`.
- Smoke saved data live under `outputs/smoke_figures/data/`.
- Smoke runs write under `outputs/smoke_figures/`.
- Release plot-only outputs should be written under `reproduced_figures/`.
- Manuscript figure paths were checked to avoid spaces.

## Figure 1

- Paper figure: Figure 1, span-profile evolution during over-parameterized
  gradient flow.
- Manuscript asset checked: `span_profile_evolution.pdf`.
- Full-run command:
  `python -m esd_experiments.scripts.figure1_span_profile --output outputs/figures/figure1_span_profile.pdf --data-output outputs/figure_data/figure1_span_profile.npz`.
- Bundled saved data:
  `data/figure_data/figure1_span_profile.npz`.
- Plot-only command:
  `python -m esd_experiments.plots.figure1_span_profile --data data/figure_data/figure1_span_profile.npz --output reproduced_figures/figure1_span_profile.pdf`.
- Smoke:
  `python -m esd_experiments.scripts.figure1_span_profile --fast --output outputs/smoke_figures/figure1.pdf`.
- Parameters: `n=10000`, `sigma0=1`, `d=5000`, `J=15`, `p=2.5`,
  `gamma=1`, `D=0`, `q in {1,1.5,2,3}`, `C=5`.
- Sequence off-support coordinates are zero.

## Figure 2

- Paper figure: Figure 2, averaged PC-estimator error and ESD during OP-GF
  training.
- Manuscript asset checked: `opgf_error_esd_20_runs.pdf`.
- Full-run command:
  `python -m esd_experiments.scripts.figure2_opgf_error_esd --output outputs/figures/figure2_opgf_error_esd.pdf --data-output outputs/figure_data/figure2_opgf_error_esd.npz --num-experiments 20 --workers 6`.
- Bundled saved data:
  `data/figure_data/figure2_opgf_error_esd.npz`.
- CPU parallelism: independent `(D, replication)` trajectories may be run with
  `--workers`; `--workers 0` chooses a conservative automatic default.
- Plot-only command:
  `python -m esd_experiments.plots.figure2_opgf_error_esd --data data/figure_data/figure2_opgf_error_esd.npz --output reproduced_figures/figure2_opgf_error_esd.pdf`.
- Smoke:
  `python -m esd_experiments.scripts.figure2_opgf_error_esd --fast --output outputs/smoke_figures/figure2.pdf`.
- Error bars: standard errors over replications.
- Supports `--cache-dir` and `--use-cache`.

## Figure 3

- Paper figure: Figure 3, linear-model prediction risk and ESD in two alignment
  cases.
- Manuscript assets checked: `risk_vs_alpha_case_1.pdf` and
  `risk_vs_alpha_case_2.pdf`.
- Full-run command:
  `python -m esd_experiments.scripts.figure3_linear_model --case both --output-dir outputs/figures --data-dir outputs/figure_data`.
- Bundled saved data:
  `data/figure_data/figure3_case_1.npz` and
  `data/figure_data/figure3_case_2.npz`.
- Plot-only commands:
  `python -m esd_experiments.plots.figure3_linear_model --data data/figure_data/figure3_case_1.npz --output reproduced_figures/figure3_case_1.pdf`.
  `python -m esd_experiments.plots.figure3_linear_model --data data/figure_data/figure3_case_2.npz --output reproduced_figures/figure3_case_2.pdf`.
- Smoke:
  `python -m esd_experiments.scripts.figure3_linear_model --fast --case both --output-dir outputs/smoke_figures`.
- Error bars: standard errors over Monte Carlo replications.

## Figure 4

- Paper figure: Figure 4, RKHS kernel PC estimator risk and ESD bound.
- Manuscript asset checked: `risk_vs_alpha_rkhs.pdf`.
- Full-run command:
  `python -m esd_experiments.scripts.figure4_rkhs_kpcpe --output outputs/figures/figure4_rkhs_kpcpe.pdf --data-output outputs/figure_data/figure4_rkhs_kpcpe.npz`.
- Bundled saved data:
  `data/figure_data/figure4_rkhs_kpcpe.npz`.
- Plot-only command:
  `python -m esd_experiments.plots.figure4_rkhs_kpcpe --data data/figure_data/figure4_rkhs_kpcpe.npz --output reproduced_figures/figure4_rkhs_kpcpe.pdf`.
- Smoke:
  `python -m esd_experiments.scripts.figure4_rkhs_kpcpe --fast --output outputs/smoke_figures/figure4.pdf`.
- Parameters: `n=400`, `J=800`, `D=80`, `B=10`, `sigma0^2=1`.
- Effective noise: `sigma_eff^2=(sigma0^2 + ||f^*||_infty^2)/n`, with
  `||f^*||_infty` estimated on a grid.
- Error bars: standard errors.

## Figure 5

- Paper figure: Figure 5, pathwise ESD for a learned four-layer linear network.
- Manuscript asset checked: `pathwise_esd_network.pdf`.
- Full-run command:
  `python -m esd_experiments.scripts.figure5_pathwise_linear_net --output outputs/figures/figure5_pathwise_linear_net.pdf --data-output outputs/figure_data/figure5_pathwise_linear_net.npz`.
- Bundled saved data:
  `data/figure_data/figure5_pathwise_linear_net.npz`.
- Plot-only command:
  `python -m esd_experiments.plots.figure5_pathwise_linear_net --data data/figure_data/figure5_pathwise_linear_net.npz --output reproduced_figures/figure5_pathwise_linear_net.pdf`.
- Smoke:
  `python -m esd_experiments.scripts.figure5_pathwise_linear_net --fast --output outputs/smoke_figures/figure5.pdf`.
- Full experiment rerun requires `requirements-torch.txt`.
- Plot-only redraw from bundled data does not require Torch.
- ESD must call the shared helper.
- Plotted risk is function-coefficient error `||A(t)^T w(t)-beta^*||^2`,
  excluding observation noise.
