# Code for Effective Span Dimension for Learned Kernels

This repository contains the Python code and saved numerical data used to
reproduce the numerical figures for the paper [Alignment-Sensitive Minimax Rates for Spectral Algorithms with Learned Kernels](https://arxiv.org/abs/2509.20294).  
The fastest verification path is plot-only: install the
standard dependencies, run the unit tests, and redraw all six active figure PDFs
from the bundled `.npz` files under `data/figure_data/`.

Run all commands from the repository root, the directory that contains
`README.md`, `esd_experiments/`, `data/`, and `docs/`.

## Fastest Verification

Figures 1--4 require NumPy, SciPy, pandas, and Matplotlib:

```bash
python -m pip install -r requirements.txt
python -m compileall esd_experiments
python -m unittest discover -s tests
```

Then redraw the six active figure PDFs from the bundled data.  These commands do
not rerun the experiments:

```bash
mkdir -p reproduced_figures

python -m esd_experiments.plots.figure1_span_profile \
  --data data/figure_data/figure1_span_profile.npz \
  --output reproduced_figures/figure1_span_profile.pdf

python -m esd_experiments.plots.figure2_opgf_error_esd \
  --data data/figure_data/figure2_opgf_error_esd.npz \
  --output reproduced_figures/figure2_opgf_error_esd.pdf

python -m esd_experiments.plots.figure3_linear_model \
  --data data/figure_data/figure3_case_1.npz \
  --output reproduced_figures/figure3_case_1.pdf

python -m esd_experiments.plots.figure3_linear_model \
  --data data/figure_data/figure3_case_2.npz \
  --output reproduced_figures/figure3_case_2.pdf

python -m esd_experiments.plots.figure4_rkhs_kpcpe \
  --data data/figure_data/figure4_rkhs_kpcpe.npz \
  --output reproduced_figures/figure4_rkhs_kpcpe.pdf

python -m esd_experiments.plots.figure5_pathwise_linear_net \
  --data data/figure_data/figure5_pathwise_linear_net.npz \
  --output reproduced_figures/figure5_pathwise_linear_net.pdf
```

Figure 5 plot-only redraw does not require Torch, because the bundled `.npz`
file already contains the saved pathwise arrays.  Rerunning the Figure 5
experiment requires Torch:

```bash
python -m pip install -r requirements-torch.txt
```

## Figures and Data

| Paper figure | Experiment | Bundled data | Plot-only output |
| --- | --- | --- | --- |
| Figure 1 | Span-profile evolution during over-parameterized gradient flow | `data/figure_data/figure1_span_profile.npz` | `reproduced_figures/figure1_span_profile.pdf` |
| Figure 2 | Averaged PC-estimator error and ESD during OP-GF training | `data/figure_data/figure2_opgf_error_esd.npz` | `reproduced_figures/figure2_opgf_error_esd.pdf` |
| Figure 3, case 1 | Linear-model prediction risk and ESD | `data/figure_data/figure3_case_1.npz` | `reproduced_figures/figure3_case_1.pdf` |
| Figure 3, case 2 | Linear-model prediction risk and ESD | `data/figure_data/figure3_case_2.npz` | `reproduced_figures/figure3_case_2.pdf` |
| Figure 4 | RKHS kernel PC estimator risk and ESD bound | `data/figure_data/figure4_rkhs_kpcpe.npz` | `reproduced_figures/figure4_rkhs_kpcpe.pdf` |
| Figure 5 | Pathwise ESD for a learned four-layer linear network | `data/figure_data/figure5_pathwise_linear_net.npz` | `reproduced_figures/figure5_pathwise_linear_net.pdf` |

The release intentionally does not include pre-rendered figure PDFs.  The
commands above recreate them from the saved data.

## Full Rerun Commands

The commands below rerun the experiments and write outputs under `outputs/`.
They are slower than plot-only redraws, especially Figure 2.

```bash
python -m esd_experiments.scripts.figure1_span_profile \
  --output outputs/figures/figure1_span_profile.pdf \
  --data-output outputs/figure_data/figure1_span_profile.npz

python -m esd_experiments.scripts.figure2_opgf_error_esd \
  --output outputs/figures/figure2_opgf_error_esd.pdf \
  --data-output outputs/figure_data/figure2_opgf_error_esd.npz \
  --num-experiments 20 \
  --workers 6

python -m esd_experiments.scripts.figure3_linear_model \
  --case both \
  --output-dir outputs/figures \
  --data-dir outputs/figure_data

python -m esd_experiments.scripts.figure4_rkhs_kpcpe \
  --output outputs/figures/figure4_rkhs_kpcpe.pdf \
  --data-output outputs/figure_data/figure4_rkhs_kpcpe.npz

python -m esd_experiments.scripts.figure5_pathwise_linear_net \
  --output outputs/figures/figure5_pathwise_linear_net.pdf \
  --data-output outputs/figure_data/figure5_pathwise_linear_net.npz
```

Figure 2 supports CPU process-level parallelism with `--workers`, plus
`--cache-dir` and `--use-cache`; run the help command below for those options.

```bash
python -m esd_experiments.scripts.figure2_opgf_error_esd --help
```

## Quick Smoke Runs

Use `--fast` to check imports, data saving, and plotting paths without running
the full experiments.  Smoke outputs are written under `outputs/smoke_figures/`,
which is ignored by `.gitignore`.

```bash
python -m esd_experiments.scripts.figure1_span_profile --fast --output outputs/smoke_figures/figure1.pdf
python -m esd_experiments.scripts.figure2_opgf_error_esd --fast --output outputs/smoke_figures/figure2.pdf
python -m esd_experiments.scripts.figure3_linear_model --fast --case both --output-dir outputs/smoke_figures
python -m esd_experiments.scripts.figure4_rkhs_kpcpe --fast --output outputs/smoke_figures/figure4.pdf
python -m esd_experiments.scripts.figure5_pathwise_linear_net --fast --output outputs/smoke_figures/figure5.pdf
```

## Runtime and Reproducibility Notes

- Commands use the fixed default seed `2026` unless `--seed` is provided.
- Monte Carlo error bars in Figures 2--4 are standard errors over replications.
- Figure 2 full reproduction is computationally expensive; use plot-only or
  smoke commands first.
- Figure 5 full reproduction requires Torch and may show small numerical
  differences across hardware, BLAS libraries, or Torch versions.
- `SHA256SUMS` records hashes for files in this release.  It does not include
  itself.

## Implementation Contract

The detailed figure contract is recorded in
`docs/figure_reproduction_contract.md`.  In particular:

- ESD is computed from unsquared signal coefficients.
- The shared ESD helper implements
  `H(k)=k^{-1} sum_{i>k} theta_{pi_i}^{*2}` with `H(d)=0`.
- In Figures 1 and 2, non-signal coordinates in the sparse sequence model are
  exactly zero.
- Figure 4 uses the effective noise
  `(sigma0^2 + ||f^*||_infty^2) / n`.
- Figure 5 plots the target function error
  `||A(t)^T w(t)-beta^*||^2`, excluding observation noise.

## Release Contents

- `esd_experiments/core/`, `experiments/`, `scripts/`, and `plots/` contain the
  reusable code, experiment entry points, and plot-only redraw entry points.
- `tests/` contains the stdlib `unittest` checks.
- `data/figure_data/` contains the six bundled full-run `.npz` files.
- `docs/figure_reproduction_contract.md` records the figure/data agreement
  checks.
- `RELEASE_MANIFEST.md`, `SHA256SUMS`, `LICENSE`, `CITATION.cff`, and
  `.gitignore` support release inspection and reuse.
