# Emputation (Python)

Emputation is a deep generative framework for learning imputation models. It targets the extrapolation distribution of missing variables given observed variables, and its training is guided by missing-data assumptions that identify this target distribution. The training objective, called the emputation risk, is an energy-score-based risk in which the identification assumption determines how observed entries are masked and which observations contribute to training. After training, the fitted model supports direct conditional sampling for multiple imputation.

This folder contains the Python implementation of emputation and the Python code to reproduce the simulation results in the paper: **Emputation: Identification-Guided Neural Imputation Framework**. It is a translation of the R package (`../r-package`) and of the R simulation code (`../simulation`).

------------------------------------------------------------------------

## Installation

``` bash
pip install "git+https://github.com/yjyang00/emputation.git#subdirectory=python/emputation-python"
```

`emputation` depends on [`PyTorch`](https://pytorch.org), NumPy, pandas, and Matplotlib. The simulation code additionally uses SciPy and scikit-learn (see `requirements.txt`) and runs from this folder without installing the package.

## Quick start

Below we provide a toy example to show how emputation can be used under MCAR.

``` python
import numpy as np
from emputation import emputation

rng = np.random.default_rng(1)

n = 1000
X = rng.normal(size=(n, 5))

# MCAR
X.flat[rng.choice(X.size, int(0.15 * X.size), replace=False)] = np.nan

# Fit emputation under MCAR
fit = emputation(X, mechanism="mcar", num_epochs=500)

print(fit)
fit.summary()
fit.plot()

# Generate 10 completed datasets
imputed = fit.impute(X, m=10)
```

## Supported mechanisms

The Python implementation supports the same identification assumptions as the R package through the `mechanism` argument:

-   `"mcar"` — missing completely at random.
-   `"ccmv"` — complete-case missing value assumption.
-   `"tree"` — tree graph assumption.

For tree graph methodology, see [Suen and Chen (2026)](https://arxiv.org/pdf/2602.16992) and [Chen (2022)](https://projecteuclid.org/journals/annals-of-statistics/volume-50/issue-1/Pattern-graphs-A-graphical-approach-to-nonmonotone-missing-data/10.1214/21-AOS2094.full).

## Folder Structure

-   `emputation-python/` — Source code for the emputation Python package. See `emputation-python/README.md` for details.

-   `simulation/` — Code and instructions for reproducing the simulation figures and tables, including SLURM jobs for a computing cluster. See `simulation/README.md` for details.

The data application is not translated, since the NACC data are not publicly available; its R code is in `../data_application`.
