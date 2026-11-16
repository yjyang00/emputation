import numpy as np
import torch

from emputation.loss_func import emputation_risk, energy_score_terms


def naive(target, samples, mask, beta):
    """Energy-score terms as written in eq. (4), with explicit loops."""
    T, B, _ = samples.shape
    s1, s2 = np.zeros(T), np.zeros(T)
    for t in range(T):
        m = mask[t]
        s1[t] = np.mean([np.linalg.norm(target[t, m] - samples[t, b, m]) ** beta for b in range(B)])
        s2[t] = sum(np.linalg.norm(samples[t, b, m] - samples[t, c, m]) ** beta
                    for b in range(B) for c in range(B)) / (B * (B - 1))
    return s1, s2


def test_energy_score_terms_match_loops():
    rng = np.random.default_rng(1)
    T, B, d = 7, 3, 5
    target, samples = rng.normal(size=(T, d)), rng.normal(size=(T, B, d))
    mask = rng.random((T, d)) < 0.5
    mask[:, 0] = True
    for beta in (1.0, 0.5, 1.7):
        s1, s2 = energy_score_terms(torch.tensor(target), torch.tensor(samples), torch.tensor(mask), beta)
        n1, n2 = naive(target, samples, mask, beta)
        np.testing.assert_allclose(s1.numpy(), n1, rtol=1e-10)
        np.testing.assert_allclose(s2.numpy(), n2, rtol=1e-10)


def test_risk_normalization():
    s1, s2 = torch.tensor([2.0, 4.0]), torch.tensor([1.0, 2.0])
    loss, t1, t2 = emputation_risk(s1, s2, torch.tensor([1.0, 2.0]), n_rows=4)
    assert torch.isclose(loss, torch.tensor(((2 - 0.5) / 1 + (4 - 1) / 2) / 4))
    assert torch.isclose(loss, t1 - 0.5 * t2)


def test_pair_term_gradient_is_finite_for_small_beta():
    samples = torch.randn(4, 2, 3, requires_grad=True)
    s1, s2 = energy_score_terms(torch.randn(4, 3), samples, torch.ones(4, 3, dtype=torch.bool), 0.5)
    (s1 - 0.5 * s2).sum().backward()
    assert torch.isfinite(samples.grad).all()
