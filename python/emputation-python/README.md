# Emputation

Emputation is a deep generative framework for learning imputation models. It targets the extrapolation
distribution of missing variables given observed variables, and training is guided by a missing-data
assumption that identifies this target. The training objective, the emputation risk, is an energy-score loss in
which the identification assumption determines which observed entries are masked and which observations
contribute to training.

This is the Python implementation of the R package in `../../r-package`. It supports three identification
assumptions via `mechanism`:

- `"mcar"` — missing completely at random.
- `"ccmv"` — complete-case missing variable assumption.
- `"tree"` — tree graph; see [Suen & Chen (2026)](https://arxiv.org/pdf/2602.16992) and
  [Chen (2022)](https://projecteuclid.org/journals/annals-of-statistics/volume-50/issue-1/Pattern-graphs-A-graphical-approach-to-nonmonotone-missing-data/10.1214/21-AOS2094.full).

## Installation

Requires torch, numpy, pandas and matplotlib. The scripts of this repository use the package from this folder
without installation; to import it elsewhere, add this folder to `PYTHONPATH` or run `pip install -e .` here.

## Usage

```python
import numpy as np
from emputation import emputation

rng = np.random.default_rng(1)
X = rng.normal(size=(1000, 5))
X[rng.random(X.shape) < 0.15] = np.nan

fit = emputation(X, mechanism="mcar", num_epochs=500)
print(fit)                  # architecture, training settings, final loss
fit.summary()               # adds response patterns, training terms, loss over the last 10% of epochs
fit.plot()                  # training losses
fit.losses                  # per-epoch losses as a DataFrame

imputed = fit.impute(X, m=10)    # list of 10 completed datasets (DataFrames if X is a DataFrame)
draws = fit.sample(X, m=10)      # the same as an array of shape (10, n, d)
```

For `mechanism="tree"`, pass `tree_edges` (child -> parent response patterns as binary strings, `"1"` =
observed; a dict, a list of pairs, or a DataFrame with columns `child`/`parent`) defined over `tree_vars`
(column names or **0-based** column indices; other columns count as always observed):

```python
# patterns over columns a, b, c of df; the other columns count as always observed
fit = emputation(df, "tree", tree_edges={"110": "111", "100": "110", "010": "110"},
                 tree_vars=["a", "b", "c"])
```

Main arguments of `emputation()` / `Emputer` (defaults as in the R package): `B=2` Monte Carlo draws per term,
`hidden_dim=500`, `num_layer=3` affine maps (two hidden layers), `batch_norm=True`, `num_epochs=1000`,
`lr=1e-4`, `beta=1`, `standardize=True`, `batch_size=None` (full batch: one optimizer step per epoch),
`device="cpu"` or `"cuda"`, and `seed`.

`chunk_size` caps memory for very large problems: each optimizer step processes its training terms in random
chunks and accumulates the gradients. Without batch normalization this leaves the step unchanged (tested); with
batch normalization the batch statistics are computed per chunk, a close approximation for large chunks.

## Package layout

| Module | Contents |
|---|---|
| `emputation.py` | `emputation()` and the `Emputer` class (fit, sample/impute, summary, plot) |
| `patterns.py` | response patterns, training terms of the MCAR/CCMV/tree risks, tree-graph validation |
| `loss_func.py` | energy-score terms and the empirical emputation risk |
| `models.py` | the network `EmputationNet` |

## Differences from the R package

| | R package | Python |
|---|---|---|
| Argument for Monte Carlo draws | `B` (`M` in the simulation code) | `B` |
| `tree_vars` indices | 1-based | 0-based |
| `num_layer > 3` | builds `num_layer + 1` affine maps | builds `num_layer` affine maps (identical for `num_layer <= 3`) |
| Zero-variance columns | all scales floored at 1000 (`pmax(sddX, 10^3)`) | the constant column is centered, not scaled |
| Full-batch row permutation | rows permuted each epoch (only reorders terms) | natural order |
| Pairwise energy term | all `B x B` pairs incl. zero diagonal, times `B/(B-1)` | distinct pairs only (same value; finite gradients for `beta < 1`) |

With identical initial weights and noise draws, the R and Python implementations produce the same training
losses and imputations up to float32 rounding.
