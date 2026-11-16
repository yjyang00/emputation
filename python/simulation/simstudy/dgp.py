"""Missingness-generating processes (Supplement A).

Each generator takes a complete data matrix and returns a boolean mask of the
same shape (``True`` = missing).
"""

import numpy as np

# Incomplete response patterns of the CCMV process, as 1-based indices of the
# missing variables (src/R/specs/*_ccmv_spec.R).
CCMV_PATTERNS = {
    "concrete": [(1, 2, 3, 4), (5, 6, 7, 8), (9, 1, 2, 3), (4, 5, 6, 7), (8, 9, 1, 2), (3, 4, 5, 6),
                 (7, 8, 9, 1), (2, 3, 4, 5), (6, 7, 8, 9), (1, 3, 5, 7), (2, 4, 6, 8, 9)],
    "ccpp": [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3), (2, 4), (3, 5), (4, 1), (5, 2),
             (1,), (2,), (3,), (4,), (5,),
             (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 1), (5, 1, 2)],
    "wine": [(1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12), (1, 3, 5, 7, 9, 11), (2, 4, 6, 8, 10, 12),
             (1, 4, 5, 8, 9, 12), (2, 3, 6, 7, 10, 11), (1, 2, 7, 8, 3, 9), (4, 5, 10, 11, 6, 12),
             (1, 2, 3, 4, 5), (6, 7, 8, 9, 10), (11, 12, 1, 2, 3), (4, 5, 6, 7, 8),
             (9, 10, 11, 12, 2), (3, 4, 5, 6, 7), (8, 9, 10, 11, 1), (12, 2, 3, 4, 7),
             (5, 6, 8, 9, 12), (1, 3, 7, 10, 11), (2, 4, 5, 8, 9), (6, 10, 11, 12, 1)],
}


def ccmv_patterns(dataset, d):
    """Incomplete response patterns (``(K, d)`` 0/1 array, 1 = observed) of the CCMV process."""
    P = np.ones((len(CCMV_PATTERNS[dataset]), d), dtype=np.int8)
    for k, miss in enumerate(CCMV_PATTERNS[dataset]):
        P[k, np.asarray(miss) - 1] = 0
    if len({p.tobytes() for p in P}) != len(P):
        raise ValueError(f"Duplicate CCMV patterns for {dataset}.")
    return P


def generate_mcar(X, rho, cc_prop=0.2, rng=None):
    """MCAR: keep ``round(cc_prop * n)`` random rows complete; in the other
    rows each entry is missing independently with probability ``rho``.
    Rows that would be entirely missing keep one random entry."""
    rng = np.random.default_rng(rng)
    n, d = X.shape
    cc_rows = rng.choice(n, round(cc_prop * n), replace=False)
    rows = np.setdiff1d(np.arange(n), cc_rows)
    sub = rng.random((rows.size, d)) < rho
    full = np.flatnonzero(sub.all(axis=1))
    sub[full, rng.integers(0, d, full.size)] = False
    mask = np.zeros((n, d), dtype=bool)
    mask[rows] = sub
    return mask


def generate_mar(X, target_rate, cc_prop=0.2, rng=None):
    """Sequential MAR process adapted from the GAIN paper (Supplement A).

    After keeping ``round(cc_prop * n)`` random rows complete, variable 1 is
    missing with probability ``p = target_rate / (1 - cc_prop)`` and variable
    ``j > 1`` with probability proportional to
    ``exp(-sum_{k<j} [w_k R_k X_k + b_k (1 - R_k)])`` (min-max scaled data,
    ``w_k, b_k ~ U(0, 1)``), normalized to average ``p``, and clipped to
    [0, 1]. Rows that end up entirely missing get one random entry back.
    """
    rng = np.random.default_rng(rng)
    n, d = X.shape
    cc_idx = rng.choice(n, round(cc_prop * n), replace=False)
    amp = np.setdiff1d(np.arange(n), cc_idx)
    A = X[amp]
    lo, hi = A.min(axis=0), A.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    x = np.where(hi > lo, (A - lo) / span, 0.0)
    w, b = rng.random(d), rng.random(d)
    m = np.ones((amp.size, d))  # 1 = observed
    p_adj = target_rate / (1 - cc_prop)
    score = np.zeros(amp.size)
    for j in range(d):
        if j == 0:
            p_miss = np.full(amp.size, p_adj)
        else:
            score += w[j - 1] * m[:, j - 1] * x[:, j - 1] + b[j - 1] * (1 - m[:, j - 1])
            e = np.exp(-score)
            p_miss = np.clip(p_adj * amp.size * e / e.sum(), 0, 1)
        m[:, j] = rng.binomial(1, 1 - p_miss)
    mask = np.zeros((n, d), dtype=bool)
    mask[amp] = m == 0
    full = np.flatnonzero(mask.all(axis=1))
    for r in full:  # one entry back for rows with nothing observed
        mask[r, rng.integers(d)] = False
    return mask


def generate_ccmv(X, patterns, intercept, slope=1.0, rng=None):
    """Missingness satisfying CCMV via a pattern-graph selection model.

    With standardized data ``Z``, pattern ``r`` has selection odds
    ``O_r = exp(alpha_r + beta_r' Z_r)`` relative to the complete case, so
    ``P(R = 1_d | Z) = 1 / (1 + sum_r O_r)`` and ``P(R = r | Z) = O_r P(R = 1_d | Z)``.

    Args:
        X: (n, d) complete data.
        patterns: (K, d) incomplete patterns (1 = observed).
        intercept: alpha_r, common to all patterns (-4 for ~20% and -0.5
            for ~40% missingness).
        slope: each component of beta_r. Defaults to 1.
        rng: seed or numpy Generator.
    """
    rng = np.random.default_rng(rng)
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    log_odds = intercept + slope * (Z @ patterns.T.astype(float))  # sums over observed coordinates
    odds = np.exp(log_odds)
    p_cc = 1.0 / (1.0 + odds.sum(axis=1))
    probs = np.column_stack([p_cc, p_cc[:, None] * odds])
    cum = np.cumsum(probs, axis=1)
    choice = np.minimum((rng.random(len(X))[:, None] > cum).sum(axis=1), probs.shape[1] - 1)
    table = np.vstack([np.ones((1, X.shape[1]), dtype=np.int8), patterns])
    return table[choice] == 0


def generate_mask(X, dataset, setting, rng=None):
    """Mask for a :class:`~simstudy.config.Setting`."""
    if setting.dgp == "mcar":
        return generate_mcar(X, setting.param, setting.cc_prop, rng)
    if setting.dgp == "mar":
        return generate_mar(X, setting.param, setting.cc_prop, rng)
    if setting.dgp == "ccmv":
        return generate_ccmv(X, ccmv_patterns(dataset, X.shape[1]), setting.param, rng=rng)
    raise ValueError(f"Unknown dgp {setting.dgp!r}.")
