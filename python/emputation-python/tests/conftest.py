import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# the emputation package is used from this folder, without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    torch.set_num_threads(1)


@pytest.fixture
def incomplete_data():
    rng = np.random.default_rng(0)
    n, d = 120, 4
    X = rng.normal(size=(n, d)) @ np.linalg.cholesky(0.5 * np.eye(d) + 0.5).T
    mask = np.zeros_like(X, dtype=bool)
    mask[40:] = rng.random((n - 40, d)) < 0.3
    mask[mask.all(axis=1), 0] = False
    return X, np.where(mask, np.nan, X)
