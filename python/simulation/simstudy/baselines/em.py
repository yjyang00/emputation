"""EM for a multivariate Gaussian with draws from the fitted conditionals
(src/R/emputation/em_gaussian.R and run_em_gaussian() in scripts/05_eval.R)."""

import warnings

import numpy as np
from scipy.linalg import solve_triangular


def _groups(miss):
    """Rows grouped by missingness pattern: list of (rows, observed idx, missing idx)."""
    patterns, inverse = np.unique(miss, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    return [(np.flatnonzero(inverse == g), np.flatnonzero(~p), np.flatnonzero(p))
            for g, p in enumerate(patterns)]


def _loglik(X, groups, mu, Sigma):
    """Observed-data Gaussian log-likelihood (mvtnorm::dmvnorm, Cholesky based)."""
    total = 0.0
    for rows, o, _ in groups:
        if o.size == 0:
            continue
        try:
            L = np.linalg.cholesky(Sigma[np.ix_(o, o)])
        except np.linalg.LinAlgError:
            return -np.inf
        z = solve_triangular(L, (X[np.ix_(rows, o)] - mu[o]).T, lower=True)
        total += np.sum(-np.log(np.diag(L)).sum() - 0.5 * o.size * np.log(2 * np.pi) - 0.5 * (z ** 2).sum(0))
    return total


def em_mvn(X, tol=1e-4, max_iter=200, reg=1e-6):
    """EM estimates of the mean and covariance of a Gaussian from incomplete data.

    Mirrors em_mvn() of the R code: covariance initialized from complete cases
    (or a diagonal of observed variances), ``reg`` added to the diagonal, and
    iterations stopped once the observed-data log-likelihood increases by less
    than ``tol`` (the parameters of the previous iteration are returned).
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    miss = np.isnan(X)
    mu = np.nanmean(X, axis=0)
    cc = ~miss.any(axis=1)
    if cc.sum() >= D:
        Sigma = np.cov(X[cc], rowvar=False).reshape(D, D)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = np.nanvar(X, axis=0, ddof=1)
        Sigma = np.diag(np.where(np.isfinite(v), v, 1.0))
    Sigma = Sigma + reg * np.eye(D)
    groups = _groups(miss)
    ll_old = -np.inf
    for it in range(1, max_iter + 1):
        sx, sxx = np.zeros(D), np.zeros((D, D))
        for rows, o, m in groups:
            Xg = X[rows]
            if m.size == 0:
                sx += Xg.sum(0)
                sxx += Xg.T @ Xg
                continue
            if o.size == 0:
                sx += rows.size * mu
                sxx += rows.size * (np.outer(mu, mu) + Sigma)
                continue
            A = np.linalg.solve(Sigma[np.ix_(o, o)] + reg * np.eye(o.size), Sigma[np.ix_(o, m)])
            E = Xg.copy()
            E[:, m] = mu[m] + (Xg[:, o] - mu[o]) @ A
            sx += E.sum(0)
            sxx += E.T @ E
            sxx[np.ix_(m, m)] += rows.size * (Sigma[np.ix_(m, m)] - A.T @ Sigma[np.ix_(o, m)])
        mu_new = sx / N
        Sigma_new = sxx / N - np.outer(mu_new, mu_new) + reg * np.eye(D)
        Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
        ll_new = _loglik(X, groups, mu_new, Sigma_new)
        if ll_new - ll_old < tol:
            break
        mu, Sigma, ll_old = mu_new, Sigma_new, ll_new
    if it == max_iter:
        warnings.warn("EM reached the maximum number of iterations without convergence.", stacklevel=2)
    return mu, Sigma


def _mvrnorm(mean, cov, rng):
    """Gaussian draws via the eigendecomposition (MASS::mvrnorm); ``mean`` is (n, p)."""
    ev, V = np.linalg.eigh(cov)
    if np.any(ev < -1e-6 * abs(ev.max())):
        warnings.warn("Conditional covariance is not positive semi-definite; clipping eigenvalues.", stacklevel=3)
    root = V * np.sqrt(np.clip(ev, 0, None))
    return mean + rng.standard_normal(mean.shape) @ root.T


def em_impute(X, m=10, rng=None, reg=1e-4):
    """``m`` imputations drawn from the conditional Gaussians of the EM fit.

    As in run_em_gaussian(): EM with ``reg = 1e-4``, conditionals computed with
    ``reg`` added to the observed block, and non-finite draws replaced by
    observed column means.
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    mu, Sigma = em_mvn(X, reg=reg)
    groups = [g for g in _groups(np.isnan(X)) if g[2].size]
    col_means = np.nanmean(X, axis=0)
    out = []
    for _ in range(m):
        Xi = X.copy()
        for rows, o, mi in groups:
            if o.size == 0:
                cond_mu = np.tile(mu[mi], (rows.size, 1))
                cond_S = Sigma[np.ix_(mi, mi)] + reg * np.eye(mi.size)
            else:
                S_oo = Sigma[np.ix_(o, o)] + reg * np.eye(o.size)
                S_mo = Sigma[np.ix_(mi, o)]
                cond_mu = mu[mi] + np.linalg.solve(S_oo, (X[np.ix_(rows, o)] - mu[o]).T).T @ S_mo.T
                cond_S = Sigma[np.ix_(mi, mi)] - S_mo @ np.linalg.solve(S_oo, Sigma[np.ix_(o, mi)])
            Xi[np.ix_(rows, mi)] = _mvrnorm(cond_mu, cond_S, rng)
        bad = ~np.isfinite(Xi)
        if bad.any():
            Xi[bad] = np.broadcast_to(col_means, Xi.shape)[bad]
        out.append(Xi)
    return out
