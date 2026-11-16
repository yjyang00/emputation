import math
import numbers
import warnings

import numpy as np
import torch

from .loss_func import emputation_risk, energy_score_terms
from .models import EmputationNet
from .patterns import (ccmv_terms, expand_tree_edges, mcar_terms, pattern_to_str, response_matrix,
                       str_to_pattern, tree_terms, unique_patterns)
from .utils import as_float_matrix, resolve_device

MECHANISMS = ("mcar", "ccmv", "tree")
_PREDICT_ROWS = 65536  # rows per forward pass when sampling (exact: eval-mode batch norm is row-wise)


def emputation(x, mechanism, B=2, hidden_dim=500, num_layer=3, batch_norm=True, num_epochs=1000,
               lr=1e-4, beta=1.0, standardize=True, batch_size=None, chunk_size=None,
               tree_edges=None, tree_vars=None, device="cpu", seed=None, verbose=True,
               print_every_nepoch=100):
    """Fit an emputation model.

    Trains a neural imputation model by minimizing the emputation risk, an
    energy-score loss whose masking and selection of observations are set by
    the missing-data identification assumption ``mechanism``. The fitted
    :class:`Emputer` draws multiple imputations with :meth:`Emputer.impute`.

    Args:
        x (array-like or pandas.DataFrame): (n, d) data with missing values as ``NaN``.
        mechanism (str): identification assumption, one of
            ``"mcar"`` (missing completely at random),
            ``"ccmv"`` (complete-case missing variable) or
            ``"tree"`` (tree graph; requires ``tree_edges``).
        B (int, optional): Monte Carlo draws per training term used to
            approximate the energy score. Defaults to 2.
        hidden_dim (int, optional): width of the hidden layers. Defaults to 500.
        num_layer (int, optional): number of affine maps of the network
            (``num_layer - 1`` hidden layers). Defaults to 3.
        batch_norm (bool, optional): batch normalization after each hidden
            layer. Defaults to True.
        num_epochs (int, optional): training epochs. Defaults to 1000.
        lr (float, optional): Adam learning rate. Defaults to 1e-4.
        beta (float, optional): energy-score exponent in (0, 2). Defaults to 1.
        standardize (bool, optional): train on columns standardized with the
            observed means and standard deviations. Defaults to True.
        batch_size (int, optional): ``None`` for full-batch training (one
            optimizer step per epoch, as in the paper) or the number of data
            rows per mini-batch. Defaults to None.
        chunk_size (int, optional): if set, each optimizer step processes its
            training terms in random chunks of at most this many terms and
            accumulates the gradients, which caps memory. Without batch
            normalization the step is unchanged; with it, batch statistics are
            computed per chunk, a close approximation of full-batch statistics
            for large chunks. Defaults to None (no chunking).
        tree_edges (optional): for ``mechanism="tree"``, the edges child ->
            parent of the tree graph as binary-string response patterns
            (``"1"`` = observed) over ``tree_vars``: a dict, a list of
            ``(child, parent)`` pairs, or a DataFrame with columns
            ``child`` and ``parent``.
        tree_vars (optional): ``None`` (patterns over all columns), column
            names, or 0-based column indices over which ``tree_edges`` are
            defined; the other columns count as always observed.
        device (str or torch.device, optional): ``"cpu"`` or ``"cuda"``. Defaults to "cpu".
        seed (int, optional): if given, seeds PyTorch's global random number
            generator before training. Defaults to None.
        verbose (bool, optional): print training progress. Defaults to True.
        print_every_nepoch (int, optional): print the losses every this many
            epochs. Defaults to 100.

    Returns:
        Emputer: the fitted model.
    """
    return Emputer(mechanism, B=B, hidden_dim=hidden_dim, num_layer=num_layer, batch_norm=batch_norm,
                   num_epochs=num_epochs, lr=lr, beta=beta, standardize=standardize,
                   batch_size=batch_size, chunk_size=chunk_size, tree_edges=tree_edges,
                   tree_vars=tree_vars, device=device, seed=seed, verbose=verbose,
                   print_every_nepoch=print_every_nepoch).fit(x)


class Emputer:
    """Emputation model. See :func:`emputation` for the arguments.

    Attributes (after :meth:`fit`):
        model (EmputationNet): the trained network.
        loss_history (numpy.ndarray): (num_epochs, 3) energy loss, term 1 and
            term 2 per epoch, on the training (standardized) scale.
        terms_ (TrainingTerms): the training terms of the emputation risk.
        mean_, scale_ (numpy.ndarray): observed-data column means and
            standard deviations used for standardization.
    """

    def __init__(self, mechanism, B=2, hidden_dim=500, num_layer=3, batch_norm=True, num_epochs=1000,
                 lr=1e-4, beta=1.0, standardize=True, batch_size=None, chunk_size=None,
                 tree_edges=None, tree_vars=None, device="cpu", seed=None, verbose=True,
                 print_every_nepoch=100):
        _check_args(mechanism, B, hidden_dim, num_layer, num_epochs, lr, beta, batch_size, chunk_size,
                    tree_edges)
        self.mechanism = mechanism
        self.B = int(B)
        self.hidden_dim = int(hidden_dim)
        self.num_layer = int(num_layer)
        self.batch_norm = bool(batch_norm)
        self.num_epochs = int(num_epochs)
        self.lr = float(lr)
        self.beta = float(beta)
        self.standardize = bool(standardize)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.tree_edges = tree_edges
        self.tree_vars = tree_vars
        self.device = resolve_device(device)
        self.seed = seed
        self.verbose = verbose
        self.print_every_nepoch = int(print_every_nepoch)
        self.model = None
        self.loss_history = None

    # ------------------------------------------------------------------ fit
    def fit(self, x):
        """Train the model on ``x`` (missing values as ``NaN``) and return ``self``."""
        X, self.columns_, _ = as_float_matrix(x)
        n, d = X.shape
        if d < 1:
            raise ValueError("The data must have at least one column.")
        if self.batch_size is not None and self.batch_size > n:
            warnings.warn("batch_size > number of rows; using the full data as a single batch instead.",
                          stacklevel=2)
        if not np.isnan(X).any():
            warnings.warn("The data have no missing values; emputation will still train, but there is "
                          "nothing to impute.", stacklevel=2)
        R = response_matrix(X)
        empty = np.flatnonzero(R.sum(axis=0) == 0)
        if empty.size:
            raise ValueError(f"Column(s) {empty.tolist()} have no observed values.")

        self.n_, self.d_ = n, d
        self.mean_ = np.nanmean(X, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scale = np.nanstd(X, axis=0, ddof=1)
        bad = ~np.isfinite(scale) | (scale <= 0)
        if bad.any():
            warnings.warn("Some variables have zero variance on the observed data; they are centered but "
                          "not scaled, and results might be unreliable.", stacklevel=2)
            scale[bad] = 1.0
        self.scale_ = scale

        self.terms_ = self._build_terms(R)
        if self.verbose:
            print(f"Training terms prepared ({self.terms_.n_terms} terms from {n} rows). Starting emputation...")
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.model = EmputationNet(d, self.hidden_dim, self.num_layer, self.batch_norm).to(self.device)
        Z = np.nan_to_num(self._standardize(X), nan=0.0)
        self.loss_history = self._train(Z, self.terms_)
        self.model.eval()
        return self

    def _build_terms(self, R):
        if self.mechanism == "mcar":
            return mcar_terms(R)
        if self.mechanism == "ccmv":
            return ccmv_terms(R)
        self.parent_map_, self.tree_edges_small_, self.tree_vars_idx_ = expand_tree_edges(
            self.tree_edges, self.tree_vars, self.columns_, R.shape[1])
        return tree_terms(R, self.parent_map_)

    def _train(self, Z, terms):
        dev = self.device
        n = Z.shape[0]
        self._X = torch.as_tensor(Z, dtype=torch.float32, device=dev)
        self._rows = torch.as_tensor(terms.rows, device=dev)
        self._r = torch.as_tensor(terms.input_patterns, dtype=torch.float32, device=dev)
        self._mask = torch.as_tensor(terms.target_mask, device=dev)
        self._n_masked = torch.as_tensor(terms.n_masked, dtype=torch.float32, device=dev)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        history = np.zeros((self.num_epochs, 3))
        self.model.train()
        if self.batch_size is not None:
            starts = np.searchsorted(terms.rows, np.arange(n))
            counts = np.bincount(terms.rows, minlength=n)
            eff = min(self.batch_size, n)
            num_batches = math.ceil(n / eff)
        try:
            for epoch in range(self.num_epochs):
                if self.batch_size is None:  # full batch: one step on all terms
                    history[epoch] = self._step(optimizer, None, n)
                else:
                    perm = torch.randperm(n).numpy()
                    total = np.zeros(3)
                    for b in range(num_batches):
                        rows_b = perm[b * eff:(b + 1) * eff]
                        idx = _terms_of_rows(rows_b, starts, counts)
                        if idx.size:
                            total += self._step(optimizer, idx, rows_b.size)
                    history[epoch] = total / num_batches
                if self.verbose and (epoch == 0 or (epoch + 1) % self.print_every_nepoch == 0
                                     or epoch + 1 == self.num_epochs):
                    print("Epoch [{:4d}/{:4d}] | loss: {:.4f} | T1: {:.4f} | T2: {:.4f}".format(
                        epoch + 1, self.num_epochs, *history[epoch]), flush=True)
        finally:
            del self._X, self._rows, self._r, self._mask, self._n_masked
        return history

    def _step(self, optimizer, idx, n_rows):
        """One optimizer step on the terms ``idx`` (``None`` = all terms)."""
        optimizer.zero_grad()
        total = np.zeros(3)
        for chunk in self._chunks(idx):
            loss, term1, term2 = self._risk(chunk, n_rows)
            loss.backward()
            total += (loss.item(), term1.item(), term2.item())
        optimizer.step()
        return total

    def _chunks(self, idx):
        n_terms = self._rows.numel() if idx is None else idx.size
        if self.chunk_size is None or n_terms <= self.chunk_size:
            yield idx
            return
        order = torch.randperm(n_terms).numpy()
        if idx is not None:
            order = idx[order]
        for start in range(0, n_terms, self.chunk_size):
            yield order[start:start + self.chunk_size]

    def _risk(self, idx, n_rows):
        """Emputation risk of the terms ``idx`` (``None`` = all terms), normalized by ``n_rows``."""
        if idx is None:
            rows, r, mask, n_masked = self._rows, self._r, self._mask, self._n_masked
        else:
            idx = torch.as_tensor(idx, device=self.device)
            rows, r, mask, n_masked = self._rows[idx], self._r[idx], self._mask[idx], self._n_masked[idx]
        x = self._X[rows]
        n_terms, d = x.shape
        x_rep = x.repeat_interleave(self.B, dim=0)
        r_rep = r.repeat_interleave(self.B, dim=0)
        draws = self.model.draw(x_rep, r_rep, self._noise(x_rep.shape)).view(n_terms, self.B, d)
        s1, s2 = energy_score_terms(x, draws, mask, self.beta)
        return emputation_risk(s1, s2, n_masked, n_rows)

    def _noise(self, shape):
        """Reference noise of the generative model (standard Gaussian)."""
        return torch.randn(shape, device=self.device)

    # --------------------------------------------------------------- impute
    @torch.no_grad()
    def sample(self, x, m=10):
        """Draw ``m`` imputations of the missing entries of ``x``.

        Args:
            x (array-like or pandas.DataFrame): (n, d) data with ``NaN`` for
                missing values, with the columns of the training data.
            m (int, optional): number of imputed datasets. Defaults to 10.

        Returns:
            numpy.ndarray: (m, n, d) completed datasets on the original scale;
            observed entries equal those of ``x``.
        """
        self._check_fitted()
        X, _, _ = as_float_matrix(x)
        if X.shape[1] != self.d_:
            raise ValueError(f"x has {X.shape[1]} columns but the model was fit on {self.d_} columns.")
        if not isinstance(m, numbers.Integral) or m < 1:
            raise ValueError("m must be a positive integer.")
        obs = ~np.isnan(X)
        Z = torch.as_tensor(np.nan_to_num(self._standardize(X), nan=0.0), dtype=torch.float32)
        R = torch.as_tensor(obs, dtype=torch.float32)
        self.model.eval()
        out = np.empty((m,) + X.shape)
        for t in range(m):
            for start in range(0, X.shape[0], _PREDICT_ROWS):
                z = Z[start:start + _PREDICT_ROWS].to(self.device)
                r = R[start:start + _PREDICT_ROWS].to(self.device)
                if self.mechanism == "tree":
                    imp = self._tree_draw(z, r)
                else:
                    imp = torch.where(r.bool(), z, self.model.draw(z, r, self._noise(z.shape)))
                out[t, start:start + _PREDICT_ROWS] = imp.cpu().numpy()
        out = self._unstandardize(out)
        out[:, obs] = X[obs]
        return out

    def impute(self, x, m=10):
        """Multiple imputation: a list of ``m`` completed datasets.

        Each element is a numpy array, or a DataFrame with the index and
        columns of ``x`` when ``x`` is a DataFrame.
        """
        draws = self.sample(x, m)
        if hasattr(x, "columns"):
            import pandas as pd

            return [pd.DataFrame(draw, index=x.index, columns=x.columns) for draw in draws]
        return list(draws)

    def _tree_draw(self, z, r):
        """Sequential imputation along the tree: each step fills ``PA(r) - r``."""
        x, r = z.clone(), r.clone()
        d = self.d_
        for _ in range(2 * d):
            active = torch.nonzero(r.sum(dim=1) < d).flatten()
            if active.numel() == 0:
                break
            r_act = r[active]
            pats, inv = unique_patterns(r_act.cpu().numpy().astype(np.int8))
            keys = [pattern_to_str(p) for p in pats]
            orphan = [k for k in keys if k not in self.parent_map_]
            if orphan:
                raise ValueError("Tree graph has no parent for pattern(s): " + ", ".join(orphan))
            parent = np.stack([str_to_pattern(self.parent_map_[k]) for k in keys])[inv]
            target = (torch.as_tensor(parent, dtype=torch.float32, device=r.device) - r_act) == 1
            draw = self.model.draw(x[active], r_act, self._noise(r_act.shape))
            x[active] = torch.where(target, draw, x[active])
            r[active] = r_act.masked_fill(target, 1.0)
        if (r.sum(dim=1) < d).any():
            raise ValueError("Sequential tree imputation did not reach the complete-case root. Check tree_edges.")
        return x

    # -------------------------------------------------------------- helpers
    def _standardize(self, X):
        return (X - self.mean_) / self.scale_ if self.standardize else X

    def _unstandardize(self, X):
        return X * self.scale_ + self.mean_ if self.standardize else X

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("The model is not fitted yet; call fit() first.")

    # ---------------------------------------------------------- diagnostics
    @property
    def losses(self):
        """Per-epoch training losses as a DataFrame (epoch, energy_loss, term1, term2)."""
        import pandas as pd

        self._check_fitted()
        lv = self.loss_history
        return pd.DataFrame({"epoch": np.arange(1, len(lv) + 1), "energy_loss": lv[:, 0],
                             "term1": lv[:, 1], "term2": lv[:, 2]})

    def _header(self):
        bs = "full" if self.batch_size is None else self.batch_size
        return [f"  mechanism        : {self.mechanism}",
                f"  data             : n = {self.n_}, d = {self.d_}",
                f"  neural network   : {self.num_layer} layers, {max(self.num_layer - 1, 1)} hidden of "
                f"{self.hidden_dim} units (batch_norm = {self.batch_norm})",
                f"  training         : B = {self.B}, lr = {self.lr:.2e}, beta = {self.beta:.2f}, "
                f"epochs = {self.num_epochs}, batch_size = {bs}"]

    def __repr__(self):
        if self.model is None:
            return f"Emputer(mechanism={self.mechanism!r}, unfitted)"
        last = self.loss_history[-1]
        lines = ["<emputation fit>"] + self._header()
        lines.append("  final loss       : energy = {:.4f} (Term1 = {:.4f}, Term2 = {:.4f})".format(*last))
        if self.mechanism == "tree":
            lines.append(f"  tree edges       : {len(self.parent_map_)}")
        return "\n".join(lines)

    def summary(self, tail_frac=0.1):
        """Print and return a summary of the fit, including the loss over the last ``tail_frac`` of epochs."""
        self._check_fitted()
        lv = self.loss_history
        tail = lv[-max(1, round(tail_frac * len(lv))):, 0]
        info = {"mechanism": self.mechanism, "n": self.n_, "d": self.d_,
                "n_patterns": self.terms_.n_patterns, "n_terms": self.terms_.n_terms,
                "final_loss": lv[-1].copy(), "tail_frac": tail_frac,
                "tail_loss_mean": float(tail.mean()), "tail_loss_range": (float(tail.min()), float(tail.max()))}
        lines = ["<emputation fit summary>"] + self._header()
        lines.insert(3, f"  response patterns: {info['n_patterns']} ({info['n_terms']} training terms)")
        lines.append("  final loss       : energy = {:.4f} (Term1 = {:.4f}, Term2 = {:.4f})".format(*lv[-1]))
        lines.append("  last {:.0f}% epochs  : mean energy loss = {:.4f} (range {:.4f} - {:.4f})".format(
            100 * tail_frac, info["tail_loss_mean"], *info["tail_loss_range"]))
        print("\n".join(lines))
        return info

    def plot(self, which="all", log_y=False, ax=None):
        """Plot the training losses.

        Args:
            which (str, optional): ``"all"``, ``"energy"``, ``"term1"`` or ``"term2"``.
            log_y (bool, optional): logarithmic y-axis. Defaults to False.
            ax (matplotlib.axes.Axes, optional): axes to draw on.

        Returns:
            matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        self._check_fitted()
        names = {"energy": (0, "energy-loss", "black"), "term1": (1, "Term 1", "steelblue"),
                 "term2": (2, "Term 2", "firebrick")}
        if which != "all" and which not in names:
            raise ValueError('which must be one of "all", "energy", "term1", "term2".')
        if ax is None:
            _, ax = plt.subplots()
        epochs = np.arange(1, len(self.loss_history) + 1)
        for key in (names if which == "all" else [which]):
            col, label, color = names[key]
            ax.plot(epochs, self.loss_history[:, col], color=color, label=label, lw=1)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss" if which == "all" else names[which][1])
        ax.set_title(f"emputation training loss ({self.mechanism})")
        if which == "all":
            ax.legend(frameon=False)
        return ax


def _terms_of_rows(rows, starts, counts):
    """Indices of the terms of ``rows`` (in that order), given row-major terms."""
    c = counts[rows]
    total = int(c.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    return np.repeat(starts[rows] - np.cumsum(c) + c, c) + np.arange(total)


def _check_args(mechanism, B, hidden_dim, num_layer, num_epochs, lr, beta, batch_size, chunk_size, tree_edges):
    def is_int(v, low=1):
        return isinstance(v, numbers.Real) and float(v).is_integer() and v >= low

    if mechanism not in MECHANISMS:
        raise ValueError('mechanism must be one of "mcar", "ccmv", or "tree".')
    if mechanism == "tree" and tree_edges is None:
        raise ValueError('mechanism = "tree" requires tree_edges (child -> parent response patterns).')
    if not is_int(B):
        raise ValueError("B must be a positive integer.")
    if B == 1:
        warnings.warn("B = 1 disables the pairwise spread term of the energy score; use B >= 2 "
                      "(default B = 2).", stacklevel=3)
    for name, value in (("hidden_dim", hidden_dim), ("num_layer", num_layer), ("num_epochs", num_epochs)):
        if not is_int(value):
            raise ValueError(f"{name} must be a positive integer.")
    if not (isinstance(lr, numbers.Real) and lr > 0):
        raise ValueError("lr must be a positive number.")
    if not (isinstance(beta, numbers.Real) and 0 < beta < 2):
        raise ValueError("beta must be in (0, 2).")
    if batch_size is not None and not is_int(batch_size):
        raise ValueError("batch_size must be None (full batch) or a positive integer.")
    if chunk_size is not None and not is_int(chunk_size):
        raise ValueError("chunk_size must be None or a positive integer.")
