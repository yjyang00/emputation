import warnings

import numpy as np
import torch


def as_float_matrix(x):
    """Convert data to a float64 matrix with ``NaN`` for missing values.

    Accepts a numpy array, a nested sequence, or a pandas DataFrame. As in the
    R package, categorical columns are mapped to numbers rather than dummy
    coded: categories with numeric labels keep their values and other
    categories (and strings) are coded 1, 2, ... in sorted order.

    Returns:
        tuple: (matrix, column names or None, row index or None)
    """
    if hasattr(x, "columns"):  # pandas DataFrame
        import pandas as pd

        cols = {}
        warned = False
        for name in x.columns:
            col = x[name]
            if isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object or pd.api.types.is_string_dtype(col):
                if not warned:
                    warnings.warn("Data contains categorical variables. Mapping to numeric values; dummy "
                                  "variables would need to be created explicitly by the user.", stacklevel=3)
                    warned = True
                numeric = pd.to_numeric(col.astype(object), errors="coerce")
                if numeric.notna().sum() == col.notna().sum():
                    cols[name] = numeric.to_numpy(dtype=float)
                else:
                    codes = pd.Categorical(col).codes.astype(float)
                    codes[codes < 0] = np.nan
                    cols[name] = codes + 1
            else:
                cols[name] = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
        mat = np.column_stack([cols[c] for c in x.columns]) if cols else np.empty((len(x), 0))
        return mat, [str(c) for c in x.columns], x.index
    mat = np.asarray(x, dtype=float)
    if mat.ndim == 1:
        mat = mat[:, None]
    if mat.ndim != 2:
        raise ValueError("Data must be a matrix (2-D array), a vector, or a pandas DataFrame.")
    return mat, None, None


def resolve_device(device):
    """Return a ``torch.device``; ``"gpu"``/``"cuda"`` fall back to CPU when CUDA is unavailable."""
    if isinstance(device, str):
        device = torch.device("cuda" if device in ("gpu", "cuda") else device)
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA is not available; using the CPU.", stacklevel=3)
        device = torch.device("cpu")
    return device
