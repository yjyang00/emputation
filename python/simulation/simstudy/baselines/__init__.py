"""Benchmark imputation methods of the simulation study.

Every method has the signature ``f(X, m, rng, **params) -> list of arrays``,
returning ``m`` completed datasets for multiple-imputation methods and a single
one otherwise.
"""

import numpy as np

from .em import em_impute, em_mvn
from .gain import gain_impute
from .mice import mice_impute
from .missforest import missforest_impute


def mean_impute(X, m=1, rng=None):
    """Replace missing values by observed column means."""
    X = np.asarray(X, dtype=float)
    return [np.where(np.isnan(X), np.nanmean(X, axis=0), X)]


BASELINES = {"Mean": mean_impute, "EM": em_impute, "MICE": mice_impute, "MF": missforest_impute,
             "GAIN": gain_impute}

__all__ = ["BASELINES", "mean_impute", "em_impute", "em_mvn", "mice_impute", "missforest_impute",
           "gain_impute"]
