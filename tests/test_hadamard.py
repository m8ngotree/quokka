import pytest
import torch

from quokka.reference.hadamard import hadamard_matrix, inverse_rht, random_signs, rht


@pytest.mark.parametrize("n", [1, 2, 16, 32, 128])
def test_hadamard_is_orthonormal(n):
    h = hadamard_matrix(n)
    torch.testing.assert_close(h @ h.T, torch.eye(n), atol=1e-6, rtol=0)
    assert torch.equal(h, h.T)


def test_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        hadamard_matrix(24)


def test_inverse_recovers_input():
    gen = torch.Generator().manual_seed(0)
    signs = random_signs(32, generator=gen)
    x = torch.randn(8, 256, generator=gen)
    torch.testing.assert_close(inverse_rht(rht(x, signs), signs), x, atol=1e-5, rtol=0)


def test_preserves_block_energy():
    gen = torch.Generator().manual_seed(1)
    signs = random_signs(32, generator=gen)
    x = torch.randn(4, 128, generator=gen)
    norms = lambda t: t.reshape(4, -1, 32).norm(dim=-1)  # noqa: E731
    torch.testing.assert_close(norms(rht(x, signs)), norms(x), rtol=1e-5, atol=0)


def test_spreads_an_outlier():
    signs = random_signs(32, generator=torch.Generator().manual_seed(2))
    x = torch.zeros(1, 32)
    x[0, 5] = 100.0
    y = rht(x, signs)
    # One spike of 100 becomes 32 values of magnitude 100 / sqrt(32).
    torch.testing.assert_close(y.abs(), torch.full_like(y, 100 / 32**0.5))
