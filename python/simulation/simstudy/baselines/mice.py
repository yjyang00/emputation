"""Multiple imputation by chained equations with predictive mean matching.

Re-implements the defaults of R's ``mice(data, m)`` for numeric data, used by
the simulation: every incomplete column is imputed by ``mice.impute.pmm``
(Bayesian linear regression draw, type-1 matching, 5 donors) from all other
columns, columns are visited left to right, ``maxit = 5`` iterations, and
chains start from random draws of the observed values.
"""

import numpy as np


def remove_lindep(x, y, ry, eps=1e-4, maxcor=0.99):
    """Predictors kept by mice's ``remove.lindep``: drop (near-)constant predictors,
    predictors with correlation >= ``maxcor`` with ``y``, and predictors that
    make the predictor correlation matrix (near-)singular."""
    p = x.shape[1]
    if p == 0 or not ry.any():
        return np.ones(p, dtype=bool)
    xo, yo = x[ry], y[ry]
    if yo.size < 2 or np.var(yo, ddof=1) < eps:
        return np.zeros(p, dtype=bool)
    keep = np.var(xo, axis=0, ddof=1) > eps
    with np.errstate(divide="ignore", invalid="ignore"):
        xc, yc = xo - xo.mean(0), yo - yo.mean()
        cor = (xc * yc[:, None]).sum(0) / np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum())
    keep &= np.nan_to_num(cor, nan=np.inf) < maxcor
    if not keep.any():
        return keep
    while keep.sum() > 1:
        idx = np.flatnonzero(keep)
        ev, V = np.linalg.eigh(np.corrcoef(xo[:, idx], rowvar=False))
        ev, V = ev[::-1], V[:, ::-1]
        if ev[-1] / ev[0] >= eps:
            break
        keep[idx[np.argmax(np.abs(V[:, -1]))]] = False
    return keep


def match_index(yhat_obs, yhat_mis, donors, rng):
    """For each target, a random one of the ``donors`` nearest observed
    predictions (mice's ``matchindex``; ties broken at random)."""
    n1 = yhat_obs.size
    k = int(min(max(donors, 1), n1))
    shuffle = rng.permutation(n1)
    order = shuffle[np.argsort(yhat_obs[shuffle], kind="stable")]
    sorted_obs = yhat_obs[order]
    width = min(2 * k, n1)  # the k nearest lie within k positions of the insertion point
    lo = np.clip(np.searchsorted(sorted_obs, yhat_mis) - k, 0, n1 - width)
    window = lo[:, None] + np.arange(width)[None, :]
    dist = np.abs(sorted_obs[window] - yhat_mis[:, None])
    nearest = np.argsort(dist, axis=1, kind="stable")[:, :k]
    pick = nearest[np.arange(yhat_mis.size), rng.integers(0, k, yhat_mis.size)]
    return order[window[np.arange(yhat_mis.size), pick]]


def pmm_draw(y, ry, x, donors=5, ridge=1e-5, rng=None):
    """Imputations of ``y[~ry]`` by ``mice.impute.pmm`` (matchtype 1).

    The regression parameters are drawn from their posterior (``.norm.draw``):
    ``sigma* = sqrt(RSS / chi2_df)``, ``beta* = beta_hat + chol(V) z sigma*``
    with ``V = (X'X)^{-1}`` (ridge-stabilized if singular); observed cases are
    predicted with ``beta_hat`` and missing cases with ``beta*``.
    """
    X1 = np.column_stack([np.ones(len(y)), x])
    xo, yo = X1[ry], y[ry]
    coef = np.linalg.lstsq(xo, yo, rcond=None)[0]
    resid = yo - xo @ coef
    df = max(yo.size - X1.shape[1], 1)
    xtx = xo.T @ xo
    try:
        R = np.linalg.qr(xo, mode="r")
        Rinv = np.linalg.inv(R)
        V = Rinv @ Rinv.T
        if not np.isfinite(V).all():
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        V = np.linalg.inv(xtx + np.diag(np.diag(xtx) * ridge))
    sigma_star = np.sqrt(resid @ resid / rng.chisquare(df))
    beta_star = coef + np.linalg.cholesky(0.5 * (V + V.T)) @ rng.standard_normal(X1.shape[1]) * sigma_star
    idx = match_index(xo @ coef, X1[~ry] @ beta_star, donors, rng)
    return yo[idx]


def mice_impute(X, m=10, rng=None, maxit=5, donors=5):
    """``m`` completed datasets from independent MICE-PMM chains."""
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    miss = np.isnan(X)
    targets = [j for j in range(X.shape[1]) if miss[:, j].any() and not miss[:, j].all()]
    out = []
    for _ in range(m):
        data = X.copy()
        for j in targets:  # initialization: random draws from the observed values
            data[miss[:, j], j] = rng.choice(X[~miss[:, j], j], size=miss[:, j].sum(), replace=True)
        for _ in range(maxit):
            for j in targets:
                ry = ~miss[:, j]
                x = np.delete(data, j, axis=1)
                x = x[:, remove_lindep(x, data[:, j], ry)]
                data[~ry, j] = pmm_draw(data[:, j], ry, x, donors=donors, rng=rng)
        out.append(data)
    return out
