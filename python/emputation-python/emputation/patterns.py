"""Response patterns and the training terms of the emputation risk.

A response pattern ``r`` is a 0/1 vector of length ``d`` with ``r[j] = 1`` when
variable ``j`` is observed. The empirical emputation risk is a sum over
*training terms* ``(i, r)``: the network receives row ``i`` with the
coordinates outside ``r`` replaced by noise, and its output is scored on the
masked pattern ``M_r(R_i)`` with weight ``1 / |M_r(R_i)|``. The identification
assumption only enters through which terms exist and what is masked
(Table 1 of the paper):

=========  ======================  =================================
mechanism  masked pattern M_r(R_i)  terms (selection S_r(R_i) > 0)
=========  ======================  =================================
mcar       R_i - r                  every observed r with r < R_i
ccmv       1_d - r                  R_i = 1_d, every incomplete r
tree       PA(r) - r                R_i = PA(r)
=========  ======================  =================================

Terms are listed row by row and, within a row, in the order in which the
target patterns first appear in the data -- the same order as the R package.
"""

import warnings
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

import numpy as np

TreeEdges = Union[Mapping[str, str], Sequence[Sequence[str]], "pandas.DataFrame"]  # noqa: F821


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------
def response_matrix(x: np.ndarray) -> np.ndarray:
    """Return the ``(n, d)`` 0/1 response matrix of ``x`` (1 = observed)."""
    return (~np.isnan(x)).astype(np.int8)


def pattern_to_str(r) -> str:
    """Encode a 0/1 vector as a binary string, e.g. ``[1, 0, 1] -> "101"``."""
    return "".join("1" if v else "0" for v in np.asarray(r).ravel())


def str_to_pattern(s: str) -> np.ndarray:
    """Decode a binary string into a 0/1 ``int8`` vector."""
    return np.frombuffer(s.encode("ascii"), dtype=np.uint8).astype(np.int8) - ord("0")


def unique_patterns(R: np.ndarray):
    """Unique rows of ``R`` in order of first appearance.

    Returns:
        (patterns, inverse): ``patterns[inverse[i]] == R[i]``.
    """
    uniq, first, inverse = np.unique(R, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return uniq[order], rank[np.asarray(inverse).ravel()]


def strictly_less(patterns: np.ndarray) -> np.ndarray:
    """``out[a, b]`` is True when ``patterns[b] < patterns[a]`` in the partial
    order of the paper (coordinatewise ``<=`` and different)."""
    P = patterns.astype(bool)
    out = np.empty((len(P), len(P)), dtype=bool)
    for start in range(0, len(P), 256):  # blocks bound the (block, K, d) temporary
        blk = P[start:start + 256]
        out[start:start + 256] = ~np.any(P[None, :, :] & ~blk[:, None, :], axis=2)  # P[b] <= P[a]
    np.fill_diagonal(out, False)
    return out


# ---------------------------------------------------------------------------
# Training terms
# ---------------------------------------------------------------------------
@dataclass
class TrainingTerms:
    """Training terms ``(i, r)`` of the empirical emputation risk.

    Attributes:
        rows: ``(T,)`` data row ``i`` of each term.
        patterns: ``(K, d)`` table of target patterns ``r``.
        pattern_index: ``(T,)`` row of ``patterns`` used by each term.
        target_mask: ``(T, d)`` masked pattern ``M_r(R_i)`` (scored coordinates).
        n_rows: number of data rows ``n`` (the risk is an average over rows).
    """

    rows: np.ndarray
    patterns: np.ndarray
    pattern_index: np.ndarray
    target_mask: np.ndarray
    n_rows: int

    @property
    def n_terms(self) -> int:
        return int(self.rows.size)

    @property
    def n_masked(self) -> np.ndarray:
        """``|M_r(R_i)|`` for each term (the inverse selection weight)."""
        return self.target_mask.sum(axis=1)

    @property
    def input_patterns(self) -> np.ndarray:
        """``(T, d)`` pattern ``r`` fed to the network for each term."""
        return self.patterns[self.pattern_index]

    @property
    def n_patterns(self) -> int:
        """Number of distinct target patterns that appear in the risk."""
        return int(np.unique(self.pattern_index).size)


def _row_major_terms(row_patterns: np.ndarray, candidates: Sequence[np.ndarray]):
    """List terms row by row: row ``i`` pairs with ``candidates[row_patterns[i]]``."""
    counts = np.array([candidates[a].size for a in row_patterns], dtype=np.int64)
    rows = np.repeat(np.arange(row_patterns.size), counts)
    if rows.size == 0:
        return rows, np.zeros(0, dtype=np.int64)
    pattern_index = np.concatenate([candidates[a] for a in row_patterns]).astype(np.int64)
    return rows, pattern_index


def mcar_terms(R: np.ndarray) -> TrainingTerms:
    """Terms of the MCAR risk: row ``i`` is paired with every observed pattern
    ``r < R_i`` and scored on ``R_i - r``."""
    patterns, inv = unique_patterns(R)
    less = strictly_less(patterns)
    candidates = [np.flatnonzero(less[a]) for a in range(len(patterns))]
    rows, pidx = _row_major_terms(inv, candidates)
    mask = (R[rows] == 1) & (patterns[pidx] == 0)
    return TrainingTerms(rows, patterns, pidx, mask, R.shape[0])


def ccmv_terms(R: np.ndarray) -> TrainingTerms:
    """Terms of the CCMV risk: each complete case is paired with every
    incomplete pattern ``r`` in the data and scored on ``1_d - r``."""
    patterns, _ = unique_patterns(R)
    incomplete = np.flatnonzero(~patterns.all(axis=1))
    if incomplete.size == 0:
        raise ValueError('mechanism = "ccmv" found no incomplete response patterns in the data.')
    complete_rows = np.flatnonzero(R.all(axis=1))
    if complete_rows.size == 0:
        raise ValueError('mechanism = "ccmv" requires at least one complete case.')
    rows = np.repeat(complete_rows, incomplete.size)
    pidx = np.tile(incomplete, complete_rows.size).astype(np.int64)
    mask = patterns[pidx] == 0
    return TrainingTerms(rows, patterns, pidx, mask, R.shape[0])


def tree_terms(R: np.ndarray, parent_map: Mapping[str, str]) -> TrainingTerms:
    """Terms of the tree-graph risk: each row whose pattern is ``PA(r)`` is
    paired with ``r`` and scored on ``PA(r) - r``.

    Args:
        R: ``(n, d)`` response matrix.
        parent_map: full-length (``d``) child pattern -> parent pattern, as
            returned by :func:`expand_tree_edges`.
    """
    d = R.shape[1]
    root = "1" * d
    patterns, inv = unique_patterns(R)
    pat_str = [pattern_to_str(p) for p in patterns]
    trained = [k for k, s in enumerate(pat_str) if s != root and s in parent_map]
    if not trained:
        raise ValueError("No incomplete response patterns in the data are covered by tree_edges/tree_vars.")
    uncovered = [s for s in pat_str if s != root and s not in parent_map]
    if uncovered:
        warnings.warn("Some incomplete response patterns in the data are not covered by the tree and "
                      "will not be trained on or imputed: " + ", ".join(uncovered), stacklevel=3)

    parent_of = {k: parent_map[pat_str[k]] for k in trained}
    gain = np.stack([str_to_pattern(parent_of[k]) for k in trained]) - patterns[trained]
    if np.any(gain.sum(axis=1) <= 0):
        raise ValueError("Invalid tree graph: every tree edge must add at least one observed coordinate.")
    missing_parent = sorted({p for p in parent_of.values()} - set(pat_str))
    if missing_parent:
        warnings.warn("Some parent patterns have no rows in the data and will contribute no training "
                      "terms: " + ", ".join(missing_parent), stacklevel=3)

    children_of = {}
    for k in trained:  # keep first-appearance order of the children
        children_of.setdefault(parent_of[k], []).append(k)
    candidates = [np.asarray(children_of.get(s, []), dtype=np.int64) for s in pat_str]
    rows, pidx = _row_major_terms(inv, candidates)
    if rows.size == 0:
        raise ValueError("Tree graph created zero training terms. Check whether parent patterns exist in the data.")
    parent_table = np.zeros_like(patterns)
    for k in trained:
        parent_table[k] = str_to_pattern(parent_of[k])
    mask = (parent_table[pidx] == 1) & (patterns[pidx] == 0)
    return TrainingTerms(rows, patterns, pidx, mask, R.shape[0])


# ---------------------------------------------------------------------------
# Tree graphs
# ---------------------------------------------------------------------------
def _edges_to_pairs(tree_edges: TreeEdges):
    """Normalize the accepted tree-edge formats to a list of (child, parent) strings."""
    if tree_edges is None:
        raise ValueError('mechanism = "tree" requires tree_edges (child -> parent response patterns).')
    if hasattr(tree_edges, "columns"):  # pandas DataFrame
        if not {"child", "parent"} <= set(tree_edges.columns):
            raise ValueError("tree_edges must have columns named 'child' and 'parent'.")
        pairs = list(zip(tree_edges["child"], tree_edges["parent"]))
    elif isinstance(tree_edges, Mapping):
        pairs = list(tree_edges.items())
    else:
        pairs = [tuple(e) for e in tree_edges]
    return [(str(c), str(p)) for c, p in pairs]


def validate_tree_edges(tree_edges: TreeEdges, p: int) -> dict:
    """Check that ``tree_edges`` is a forest of response patterns of length
    ``p`` rooted at the complete-case pattern and return it as a dict
    ``{child: parent}``.

    The checks follow the R package: binary strings of length ``p``, the root
    is never a child, one parent per child, each parent strictly dominates its
    child, no cycles, and every pattern has a path to the root.
    """
    pairs = _edges_to_pairs(tree_edges)
    root = "1" * p
    for c, q in pairs:
        if len(c) != p or len(q) != p:
            raise ValueError("All tree graph patterns must have length equal to len(tree_vars) "
                             "(or the number of columns if tree_vars is None).")
        if set(c + q) - {"0", "1"}:
            raise ValueError("All tree graph patterns must be binary strings, e.g. '010'.")
    children = [c for c, _ in pairs]
    if root in children:
        raise ValueError("The complete-case root pattern should not appear as a child.")
    if len(set(children)) != len(children):
        raise ValueError("Each incomplete child pattern must have exactly one parent.")
    for c, q in pairs:
        cv, qv = str_to_pattern(c), str_to_pattern(q)
        if not (np.all(qv >= cv) and np.any(qv > cv)):
            raise ValueError(f"Invalid tree edge {c} -> {q}: parent must strictly dominate child coordinatewise.")
    parent_map = dict(pairs)
    for child in children:
        path_to_root(child, parent_map, root)
    return parent_map


def path_to_root(r: str, parent_map: Mapping[str, str], root: str) -> list:
    """Patterns on the path from ``r`` to ``root`` (inclusive)."""
    path, seen = [r], set()
    cur = r
    while cur != root:
        if cur in seen:
            raise ValueError(f"Cycle detected in tree graph starting from pattern {r}.")
        seen.add(cur)
        if cur not in parent_map:
            raise ValueError(f"Pattern {cur} does not have a path to the root {root}.")
        cur = parent_map[cur]
        path.append(cur)
    return path


def expand_tree_edges(tree_edges: TreeEdges, tree_vars, columns: Optional[Sequence[str]], d: int):
    """Expand tree edges defined over ``tree_vars`` to full-length patterns.

    Variables outside ``tree_vars`` are treated as always observed.

    Args:
        tree_edges: child -> parent patterns over ``tree_vars``.
        tree_vars: ``None`` (all columns), column names, or 0-based column indices.
        columns: column names of the data (or ``None``).
        d: number of columns of the data.

    Returns:
        (parent_map_full, parent_map_small, tree_vars_idx)
    """
    if tree_vars is None:
        idx = np.arange(d)
    elif all(isinstance(v, str) for v in tree_vars):
        if columns is None:
            raise ValueError("tree_vars are column names, but the data has no column names.")
        missing = [v for v in tree_vars if v not in list(columns)]
        if missing:
            raise ValueError("tree_vars not found in the data: " + ", ".join(missing))
        idx = np.array([list(columns).index(v) for v in tree_vars])
    else:
        idx = np.asarray(tree_vars, dtype=int)
        if np.any(idx < 0) or np.any(idx >= d):
            raise ValueError("tree_vars must be column names or valid 0-based column indices.")
    small = validate_tree_edges(tree_edges, p=idx.size)

    def expand(s):
        v = np.ones(d, dtype=np.int8)
        v[idx] = str_to_pattern(s)
        return pattern_to_str(v)

    full = {expand(c): expand(q) for c, q in small.items()}
    return full, small, idx
