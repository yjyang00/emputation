import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from emputation import Emputer, emputation


@pytest.mark.parametrize("mechanism", ["mcar", "ccmv"])
def test_fit_and_impute(incomplete_data, mechanism):
    X, Xna = incomplete_data
    fit = emputation(Xna, mechanism, hidden_dim=16, num_epochs=5, verbose=False, seed=1)
    assert fit.loss_history.shape == (5, 3) and np.isfinite(fit.loss_history).all()
    draws = fit.sample(Xna, m=4)
    obs = ~np.isnan(Xna)
    assert draws.shape == (4,) + X.shape
    assert np.array_equal(draws[:, obs], np.broadcast_to(Xna[obs], (4, obs.sum())))
    assert np.isfinite(draws).all()
    assert not np.allclose(draws[0][~obs], draws[1][~obs])  # imputations are random


def test_seed_makes_fits_reproducible(incomplete_data):
    _, Xna = incomplete_data
    a = emputation(Xna, "mcar", hidden_dim=8, num_epochs=3, verbose=False, seed=7)
    b = emputation(Xna, "mcar", hidden_dim=8, num_epochs=3, verbose=False, seed=7)
    np.testing.assert_array_equal(a.loss_history, b.loss_history)


def test_chunked_accumulation_equals_full_batch_without_bn(incomplete_data):
    """Without batch norm, accumulating gradients over random chunks of terms
    takes the same optimizer steps as full-batch training (zero noise makes
    both runs deterministic)."""
    _, Xna = incomplete_data
    kw = dict(hidden_dim=8, num_epochs=4, batch_norm=False, verbose=False, seed=3)
    full, chunked = Emputer("mcar", **kw), Emputer("mcar", chunk_size=50, **kw)
    full._noise = chunked._noise = lambda shape: torch.zeros(tuple(shape))
    full.fit(Xna)
    chunked.fit(Xna)
    np.testing.assert_allclose(full.loss_history, chunked.loss_history, rtol=1e-5)


def test_dataframe_round_trip(incomplete_data):
    _, Xna = incomplete_data
    df = pd.DataFrame(Xna, columns=list("abcd"))
    fit = emputation(df, "ccmv", hidden_dim=8, num_epochs=2, verbose=False)
    imps = fit.impute(df, m=2)
    assert len(imps) == 2 and list(imps[0].columns) == list("abcd")
    assert not imps[0].isna().any().any()
    assert list(fit.losses.columns) == ["epoch", "energy_loss", "term1", "term2"]


def test_minibatch_training_runs(incomplete_data):
    _, Xna = incomplete_data
    fit = emputation(Xna, "mcar", hidden_dim=8, num_epochs=2, batch_size=32, verbose=False)
    assert np.isfinite(fit.loss_history).all()


def test_tree_mechanism(incomplete_data):
    X, _ = incomplete_data
    Xt = X.copy()
    Xt[60:90, 3] = np.nan
    Xt[90:, 2:] = np.nan
    fit = emputation(Xt, "tree", tree_edges={"10": "11", "00": "10"}, tree_vars=[2, 3],
                     hidden_dim=8, num_epochs=3, verbose=False)
    out = fit.sample(Xt, m=2)
    assert np.isfinite(out).all()
    with pytest.raises(ValueError, match="no parent"):
        Xbad = Xt.copy()
        Xbad[0, 0] = np.nan  # pattern outside the tree
        fit.sample(Xbad, m=1)


@pytest.mark.parametrize("kwargs, message", [
    (dict(mechanism="mar"), "mechanism"),
    (dict(mechanism="tree"), "tree_edges"),
    (dict(mechanism="mcar", beta=2.0), "beta"),
    (dict(mechanism="mcar", B=0), "B must"),
    (dict(mechanism="mcar", batch_size=1.5), "batch_size"),
])
def test_argument_checks(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Emputer(**kwargs)


def test_constant_column_is_centered_not_scaled(incomplete_data):
    _, Xna = incomplete_data
    Xc = Xna.copy()
    Xc[:, 1] = np.where(np.isnan(Xc[:, 1]), np.nan, 3.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit = emputation(Xc, "mcar", hidden_dim=8, num_epochs=2, verbose=False)
    assert any("zero variance" in str(x.message) for x in w)
    assert fit.scale_[1] == 1.0
