import torch
from torch.linalg import vector_norm


def energy_score_terms(target, samples, mask, beta=1.0):
    """Monte Carlo energy-score components of each training term.

    For term ``(i, r)`` with masked pattern ``M = M_r(R_i)`` and draws
    ``f_b = f(X_{i,r}, eps_b, r)``, ``b = 1, ..., B``:

    * ``s1 = (1/B) sum_b ||X_{i,M} - [f_b]_M||^beta``
    * ``s2 = 1/(B(B-1)) sum_{b != b'} ||[f_b]_M - [f_b']_M||^beta``

    The pair term is computed on distinct pairs only, which equals the
    all-pairs mean times ``B/(B-1)`` of the R package (the diagonal is zero)
    and keeps gradients finite for ``beta < 1``.

    Args:
        target (torch.Tensor): (T, d) data rows; only masked coordinates are used.
        samples (torch.Tensor): (T, B, d) network draws.
        mask (torch.Tensor): (T, d) boolean masked pattern.
        beta (float): energy-score exponent in (0, 2).

    Returns:
        tuple of torch.Tensor: ``s1`` and ``s2``, each of shape (T,).
    """
    pred = samples * mask.unsqueeze(1).to(samples.dtype)
    truth = torch.where(mask, target, torch.zeros_like(target)).unsqueeze(1)
    s1 = vector_norm(truth - pred, dim=2).pow(beta).mean(dim=1)
    B = samples.shape[1]
    if B < 2:
        return s1, torch.zeros_like(s1)
    i, j = torch.triu_indices(B, B, offset=1, device=samples.device)
    pair = vector_norm(pred[:, i] - pred[:, j], dim=2).pow(beta)
    s2 = 2.0 * pair.sum(dim=1) / (B * (B - 1))
    return s1, s2


def emputation_risk(s1, s2, n_masked, n_rows):
    """Empirical emputation risk (eq. (4) of the paper) and its two components.

    ``loss = (1/n) sum_terms (s1 - s2/2) / |M_r(R_i)|``, where ``n`` is the
    number of data rows the terms come from.

    Args:
        s1, s2 (torch.Tensor): (T,) outputs of :func:`energy_score_terms`.
        n_masked (torch.Tensor): (T,) number of masked coordinates of each term.
        n_rows (int): number of data rows.

    Returns:
        tuple of torch.Tensor: energy loss, term 1 and term 2 (scalars).
    """
    loss = torch.sum((s1 - 0.5 * s2) / n_masked) / n_rows
    term1 = torch.sum(s1 / n_masked) / n_rows
    term2 = torch.sum(s2 / n_masked) / n_rows
    return loss, term1, term2
