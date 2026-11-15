"""Evaluation metrics of the simulation study (Supplement A).

All metrics follow scripts/05_eval.R:

* RMSE / MAE of the averaged imputation on the missing entries, after scaling
  with the column means and standard deviations of the true data.
* MADC: mean absolute difference between the lower-triangular correlations of
  the true data and the average correlation matrix of the imputations.
* Energy: ``energy::eqdist.e`` between the scaled true and imputed data,
  ``n1 n2 / (n1 + n2) * (2 M12 - M11 - M22)`` with ``Mij`` the mean of all
  pairwise Euclidean distances (= n/2 times the energy distance for n1 = n2),
  averaged over imputations.
* MMD2: the unbiased statistic ``kernlab::kmmd(...)@mmdstats[2]`` with the
  Gaussian kernel ``exp(-sigma ||x - y||^2)`` and ``sigma`` the inverse
  median squared distance from ``kernlab::sigest`` (kpar = "automatic"),
  averaged over imputations.
"""

import numpy as np
from scipy.spatial.distance import cdist

METRICS = ("RMSE", "MAE", "MADC", "Energy", "MMD2")
_BLOCK = 2048


def _sq_dists(A, B):
    """Squared Euclidean distances between the rows of A and B (float64)."""
    sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(sq, 0.0, out=sq)


def _pair_sums(A, B=None, fn=np.sqrt):
    """``sum_{i,j} fn(||A_i - B_j||^2)`` computed block-wise (B = A if None)."""
    total = 0.0
    other = A if B is None else B
    for start in range(0, len(A), _BLOCK):
        total += fn(_sq_dists(A[start:start + _BLOCK], other)).sum()
    return total


def _dist_sums(A, B=None):
    """``sum_{i,j} ||A_i - B_j||`` with directly computed differences, as in
    ``energy::eqdist.e`` (identical rows contribute exact zeros)."""
    other = A if B is None else B
    return sum(cdist(A[start:start + _BLOCK], other).sum() for start in range(0, len(A), _BLOCK))


def energy_statistic(X, Y, xx_sum=None):
    """Two-sample energy statistic of ``energy::eqdist.e(rbind(X, Y), c(n1, n2))``.

    Args:
        X, Y: samples (rows are observations).
        xx_sum: optional precomputed sum of all pairwise distances within X.
    """
    n1, n2 = len(X), len(Y)
    xx = _dist_sums(X) if xx_sum is None else xx_sum
    m11, m22, m12 = xx / n1 ** 2, _dist_sums(Y) / n2 ** 2, _dist_sums(X, Y) / (n1 * n2)
    return n1 * n2 / (n1 + n2) * (2.0 * m12 - m11 - m22)


def sigest(X, rng, frac=0.5):
    """``kernlab::sigest(X, frac, scaled = FALSE)``: inverse 0.9, 0.5 and 0.1
    quantiles of squared distances between ``floor(frac * n)`` random pairs."""
    n = len(X)
    k = int(np.floor(frac * n))
    i, j = rng.integers(0, n, k), rng.integers(0, n, k)
    dist = ((X[i] - X[j]) ** 2).sum(1)
    return 1.0 / np.quantile(dist[dist != 0], [0.9, 0.5, 0.1])


def mmd2_unbiased(X, Y, sigma):
    """``kernlab::kmmd(X, Y, kernel = rbfdot(sigma))@mmdstats[2]`` for equal sample sizes:
    ``[sum_{i!=j} k(x_i,x_j) + k(y_i,y_j) - k(x_i,y_j) - k(y_i,x_j)] / (n(n-1))``."""
    n = len(X)
    if len(Y) != n:
        raise ValueError("mmd2_unbiased requires samples of equal size.")
    kern = lambda sq: np.exp(-sigma * sq)  # noqa: E731
    sum_xx = _pair_sums(X, fn=kern) - n
    sum_yy = _pair_sums(Y, fn=kern) - n
    cross = _pair_sums(X, Y, fn=kern) - kern(((X - Y) ** 2).sum(1)).sum()
    return (sum_xx + sum_yy - 2.0 * cross) / (n * (n - 1))


def mmd2(X, Y, rng):
    """Unbiased MMD^2 with ``kpar = "automatic"``, i.e. the median heuristic
    ``sigma = sigest(rbind(X, Y), scaled = FALSE)[2]`` of ``kernlab::kmmd``."""
    return mmd2_unbiased(X, Y, float(sigest(np.vstack([X, Y]), rng)[1]))


def madc(X_true, imputations):
    """Mean absolute difference between true and (imputation-averaged) correlations."""
    il = np.tril_indices(X_true.shape[1], -1)
    c_true = np.corrcoef(X_true, rowvar=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        c_imp = np.mean([np.corrcoef(Xi, rowvar=False) for Xi in imputations], axis=0)
    return float(np.mean(np.abs(c_true[il] - c_imp[il])))


class Evaluator:
    """Evaluates imputations of one incomplete version of a complete dataset.

    Scaling statistics and within-true-data distance sums are computed once
    and shared by all methods.
    """

    def __init__(self, X_true, mask):
        self.X_true = np.asarray(X_true, dtype=float)
        self.mask = np.asarray(mask, dtype=bool)
        self.mu = self.X_true.mean(axis=0)
        self.sd = self.X_true.std(axis=0, ddof=1)
        self.Z_true = (self.X_true - self.mu) / self.sd
        self._xx_sum = None

    def _scale(self, X):
        return (np.asarray(X, dtype=float) - self.mu) / self.sd

    def errors(self, imputations):
        """Scaled RMSE and MAE of the averaged imputation on the missing entries."""
        err = (self.Z_true - self._scale(np.mean(imputations, axis=0)))[self.mask]
        return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))

    def energy(self, imputations):
        if self._xx_sum is None:
            self._xx_sum = _dist_sums(self.Z_true)
        return float(np.mean([energy_statistic(self.Z_true, self._scale(Xi), self._xx_sum)
                              for Xi in imputations]))

    def mmd2(self, imputations, rng):
        return float(np.mean([mmd2(self.Z_true, self._scale(Xi), rng) for Xi in imputations]))

    def evaluate(self, imputations, rng, metrics=METRICS):
        """Dictionary of metrics for a list of completed datasets.

        Invalid imputations (any non-finite value) give NaN metrics, as the R
        evaluation skipped such methods.
        """
        imputations = [np.asarray(Xi, dtype=float) for Xi in imputations]
        if not imputations or any(not np.isfinite(Xi).all() for Xi in imputations):
            return {k: np.nan for k in metrics}
        out = {}
        if "RMSE" in metrics or "MAE" in metrics:
            out["RMSE"], out["MAE"] = self.errors(imputations)
        if "MADC" in metrics:
            out["MADC"] = madc(self.X_true, imputations)
        if "Energy" in metrics:
            out["Energy"] = self.energy(imputations)
        if "MMD2" in metrics:
            out["MMD2"] = self.mmd2(imputations, rng)
        return {k: out[k] for k in metrics}
