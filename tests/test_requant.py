import torch

from quokka.reference.nvfp4 import quantize_nvfp4
from quokka.reference.requant import requantize_transposed


def _forward_weights(seed=0):
    w = torch.randn(2, 64, 128, generator=torch.Generator().manual_seed(seed))
    return quantize_nvfp4(w, outer_dims=(-2, -1), four_over_six=True)


def test_shape_and_blocking():
    q = _forward_weights()
    r = requantize_transposed(q, generator=torch.Generator().manual_seed(0))
    assert r.elements.shape == (2, 128, 64)  # transposed
    assert r.scale_exp.shape == (2, 128, 2)  # 32-blocks along the old row axis


def test_backward_matches_forward_in_expectation():
    q = _forward_weights(seed=1)
    target = q.dequantize().transpose(-1, -2)
    gen = torch.Generator().manual_seed(2)
    n = 2000
    mean = sum(requantize_transposed(q, generator=gen).dequantize() for _ in range(n)) / n
    nearest = requantize_transposed(q, stochastic=False).dequantize()
    sr_bias = (mean - target).abs().mean()
    rtn_bias = (nearest - target).abs().mean()
    assert sr_bias < rtn_bias / 10


def test_nearest_is_deterministic_and_close():
    q = _forward_weights(seed=3)
    a = requantize_transposed(q, stochastic=False).dequantize()
    b = requantize_transposed(q, stochastic=False).dequantize()
    assert torch.equal(a, b)
    target = q.dequantize().transpose(-1, -2)
    assert ((a - target) ** 2).mean() / (target**2).mean() < 1e-3
