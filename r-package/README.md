# Emputation

Emputation is a deep generative framework for learning imputation models. It targets the extrapolation distribution of missing variables given observed variables and training is guided by specific missingness assumptions that guarantee identification of the target. The training objective, called the emputation risk, is an energy-score-based risk in which the identification assumption determines how observed entries are masked and which observations contribute to training.

This repository provides an R implementation. A Python implementation is going to be released in the near future.

The current implementation supports three identification assumptions via `mechanism`:

-   `"mcar"` — missing completely at random.

-   `"ccmv"` — complete-case missing variable assumption.

-   `"tree"` — tree graph. See [Suen & Chen (2026)](https://arxiv.org/pdf/2602.16992) and [Chen (2022)](https://projecteuclid.org/journals/annals-of-statistics/volume-50/issue-1/Pattern-graphs-A-graphical-approach-to-nonmonotone-missing-data/10.1214/21-AOS2094.full).

## Installation

``` r
# install.packages("remotes")
remotes::install_github("yjyang00/emputation", subdir = "r-package")
```

`emputation` depends on the [`torch`](https://torch.mlverse.org/) R package. On first use, install the libtorch backend with:

``` r
torch::install_torch()
```

## Usage

``` r
library(emputation)

set.seed(1)
n <- 1000
X <- matrix(rnorm(n * 5), n, 5)
X[sample(length(X), floor(0.15 * length(X)))] <- NA

fit <- emputation(X, mechanism = "mcar", num_epochs = 500, silent = FALSE)

print(fit)
summary(fit)
plot(fit)

imputed <- predict(fit, X, m = 10) # list of 10 completed datasets
```

For `mechanism = "tree"`, supply `tree_edges`, a data frame with columns `child`/`parent` giving binary-string response patterns (`1` = observed), each of length `ncol(dat)` (or `length(tree_vars)` if `tree_vars` is specified).
