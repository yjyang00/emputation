import torch
from torch import nn


class EmputationNet(nn.Module):
    """Feed-forward emputation network ``f(x, r)``.

    The input is a noise-filled observation (observed coordinates kept, the
    others replaced by noise) concatenated with its response pattern ``r``;
    the output has the dimension of the observation. ``num_layer`` counts the
    affine maps of eq. (NN) in the paper: ``num_layer - 1`` hidden layers of
    width ``hidden_dim``, each followed by ReLU and, optionally, batch
    normalization, then a linear output layer. This is the architecture of the
    R package for ``num_layer <= 3`` (the paper uses ``num_layer = 3``).

    Args:
        dim (int): number of variables ``d``.
        hidden_dim (int, optional): width of the hidden layers. Defaults to 500.
        num_layer (int, optional): number of affine maps (at least 2). Defaults to 3.
        batch_norm (bool, optional): add batch normalization after each hidden
            ReLU. Defaults to True.
    """

    def __init__(self, dim, hidden_dim=500, num_layer=3, batch_norm=True):
        super().__init__()
        self.dim = dim
        layers, in_dim = [], 2 * dim
        for _ in range(max(num_layer - 1, 1)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, r):
        """Evaluate ``f`` at noise-filled inputs ``x`` (n, d) with patterns ``r`` (n, d)."""
        return self.net(torch.cat([x, r], dim=1))

    def draw(self, x, r, noise):
        """One draw of ``f(x_r, eps, r)``: coordinates with ``r == 0`` are replaced by ``noise``."""
        return self(torch.where(r.bool(), x, noise), r)
