"""GAIN (Yoon, Jordon and van der Schaar, 2018) in PyTorch.

A port of the reference implementation (github.com/jsyoon0823/GAIN, gain.py
and utils.py): min-max normalization, generator and discriminator with two
hidden layers of width d, Xavier-normal initialization, uniform(0, 0.01)
noise in missing entries, hint ``H = M * B`` with ``B ~ Bernoulli(hint_rate)``,
Adam with the TensorFlow defaults, and rounding of columns with fewer than 20
distinct observed values.
"""

import numpy as np
import torch
from torch import nn


def _mlp(dim, out_act):
    layers = [nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)]
    for layer in layers[::2]:  # xavier_init of the original code: N(0, 2 / fan_in), zero bias
        nn.init.normal_(layer.weight, std=(2.0 / layer.in_features) ** 0.5)
        nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers, out_act)


def gain_impute(X, m=1, rng=None, batch_size=128, hint_rate=0.9, alpha=100.0, iterations=1000):
    """Single GAIN imputation, returned as a one-element list."""
    rng = np.random.default_rng(rng)
    torch.manual_seed(int(rng.integers(2 ** 31 - 1)))
    X = np.asarray(X, dtype=float)
    n, dim = X.shape
    M = (~np.isnan(X)).astype(np.float32)
    lo = np.nanmin(X, axis=0)
    span = np.nanmax(X - lo, axis=0)
    norm64 = np.nan_to_num((X - lo) / (span + 1e-6), nan=0.0)
    norm = norm64.astype(np.float32)

    G, D = _mlp(dim, nn.Sigmoid()), _mlp(dim, nn.Sigmoid())
    opt_g, opt_d = torch.optim.Adam(G.parameters()), torch.optim.Adam(D.parameters())
    batch_size = min(batch_size, n)
    for _ in range(iterations):
        idx = rng.permutation(n)[:batch_size]
        m_mb = torch.from_numpy(M[idx])
        z_mb = torch.from_numpy(rng.uniform(0, 0.01, (batch_size, dim)).astype(np.float32))
        h_mb = m_mb * torch.from_numpy((rng.uniform(size=(batch_size, dim)) < hint_rate).astype(np.float32))
        x_mb = m_mb * torch.from_numpy(norm[idx]) + (1 - m_mb) * z_mb

        # discriminator step
        g_sample = G(torch.cat([x_mb, m_mb], dim=1))
        x_hat = x_mb * m_mb + g_sample * (1 - m_mb)
        d_prob = D(torch.cat([x_hat.detach(), h_mb], dim=1))
        d_loss = -torch.mean(m_mb * torch.log(d_prob + 1e-8) + (1 - m_mb) * torch.log(1.0 - d_prob + 1e-8))
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # generator step (with the updated discriminator)
        g_sample = G(torch.cat([x_mb, m_mb], dim=1))
        x_hat = x_mb * m_mb + g_sample * (1 - m_mb)
        d_prob = D(torch.cat([x_hat, h_mb], dim=1))
        g_loss = -torch.mean((1 - m_mb) * torch.log(d_prob + 1e-8))
        mse = torch.mean((m_mb * x_mb - m_mb * g_sample) ** 2) / torch.mean(m_mb)
        opt_g.zero_grad()
        (g_loss + alpha * mse).backward()
        opt_g.step()

    with torch.no_grad():
        m_all = torch.from_numpy(M)
        z = torch.from_numpy(rng.uniform(0, 0.01, (n, dim)).astype(np.float32))
        x_all = m_all * torch.from_numpy(norm) + (1 - m_all) * z
        imputed = G(torch.cat([x_all, m_all], dim=1)).numpy().astype(float)
    imputed = M * norm64 + (1 - M) * imputed
    imputed = imputed * (span + 1e-6) + lo
    for j in range(dim):  # rounding of categorical-looking columns
        if np.unique(X[~np.isnan(X[:, j]), j]).size < 20:
            imputed[:, j] = np.round(imputed[:, j])
    return [imputed]
