"""missForest (Stekhoven and Buehlmann, 2012) with scikit-learn random forests.

Follows R's ``missForest(xmis)`` with default arguments for numeric data:
mean initialization, variables visited by increasing number of missing
values, 100 trees with ``mtry = floor(sqrt(p))`` predictors per split,
bootstrap samples of size n, at most 10 iterations, stopping at the first
increase of ``sum((X_new - X_old)^2) / sum(X_new^2)`` and returning the
previous imputation in that case.
"""

import math

import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Terminal-node rules of the two backends of R's missForest for continuous
# variables: ranger (default since missForest 1.6, min.bucket = 5) and
# randomForest (missForest <= 1.5, nodesize = 5: nodes of <= 5 cases are
# not split).
_BACKENDS = {"ranger": dict(min_samples_leaf=5), "randomForest": dict(min_samples_split=6)}


def missforest_impute(X, m=1, rng=None, maxiter=10, ntree=100, backend="randomForest", n_jobs=None):
    """Single missForest imputation, returned as a one-element list."""
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    miss = np.isnan(X)
    ximp = np.where(miss, np.nanmean(X, axis=0), X)
    mtry = max(int(math.floor(math.sqrt(p))), 1)
    order = [j for j in np.argsort(miss.sum(0), kind="stable") if miss[:, j].any()]
    history, conv_old, conv_new = [], np.inf, 0.0
    while conv_new < conv_old and len(history) < maxiter:
        if history:
            conv_old = conv_new
        previous = ximp.copy()
        for j in order:
            obs = ~miss[:, j]
            rf = RandomForestRegressor(n_estimators=ntree, max_features=mtry, bootstrap=True,
                                       n_jobs=n_jobs, random_state=int(rng.integers(2 ** 31 - 1)),
                                       **_BACKENDS[backend])
            rf.fit(np.delete(ximp[obs], j, axis=1), ximp[obs, j])
            ximp[~obs, j] = rf.predict(np.delete(ximp[~obs], j, axis=1))
        history.append(ximp.copy())
        conv_new = np.sum((ximp - previous) ** 2) / np.sum(ximp ** 2)
    return [history[-1] if len(history) == maxiter else history[-2]]
