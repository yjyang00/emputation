# Emputation

Emputation is a deep generative framework for learning imputation models. It targets the extrapolation distribution of missing variables given observed variables, and its training is guided by missing-data assumptions that identify this target distribution. The training objective, called the emputation risk, is an energy-score-based risk in which the identification assumption determines how observed entries are masked and which observations contribute to training. After training, the fitted model supports direct conditional sampling for multiple imputation.

This repository contains the code to reproduce the results in the paper: **Emputation: Identification-Guided Neural Imputation Framework**. A Python package will be made available in this repository in the near future.

------------------------------------------------------------------------

## Installation

``` r
# install.packages("remotes")
remotes::install_github("yjyang00/emputation", subdir = "r-package")
```

`emputation` depends on the [`torch`](https://torch.mlverse.org/) R package. On first use, install the libtorch backend with:

``` r
torch::install_torch()
```

## Quick start

Below we provide a toy example to show how emputation can be used under MCAR.

``` r
library(emputation)

set.seed(1)

n <- 1000
X <- matrix(rnorm(n * 5), nrow = n, ncol = 5)

# MCAR
X[sample(length(X), floor(0.15 * length(X)))] <- NA

# Fit emputation under MCAR
fit <- emputation(X, mechanism = "mcar", num_epochs = 500, silent = FALSE)

print(fit)
summary(fit)
plot(fit)

# Generate 10 completed datasets
imputed <- predict(fit, X, m = 10)
```

## Supported mechanisms

The current R implementation supports the following identification assumptions through the `mechanism` argument:

-   `"mcar"` — missing completely at random.
-   `"ccmv"` — complete-case missing value assumption.
-   `"tree"` — tree graph assumption.

For tree graph methodology, see [Suen and Chen (2026)](https://arxiv.org/pdf/2602.16992) and [Chen (2022)](https://projecteuclid.org/journals/annals-of-statistics/volume-50/issue-1/Pattern-graphs-A-graphical-approach-to-nonmonotone-missing-data/10.1214/21-AOS2094.full).

## Repository Structure

The R package source code, simulation, and data-application analyses are organized in separate folders.

-   `r-package/` — Source code for the emputation R package. See `r-package/README.md` for details.

-   `simulation/` — Code and instructions for reproducing the simulation. See `simulation/README.md` for details.

-   `data_application/` — Code and instructions for reproducing the data application results in the main paper and supplementary material. See `data_application/README.md` for details. The NACC data used in the real-data application are not provided in this repository. Researchers seeking access should submit a data request through [NACC](https://naccdata.org) and follow NACC’s data use policies.
