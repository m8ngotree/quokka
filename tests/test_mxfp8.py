import ml_dtypes
import numpy as np
import pytest
import torch

from quokka.reference.mxfp8 import BLOCK, quantize_mxfp8


def _randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("stochastic", [False, True])
def test_fields_are_representable(stochastic):
    gen = torch.Generator().manual_seed(0)
    q = quantize_mxfp8(_randn(8, 256) * 1e4, stochastic=stochastic, generator=gen)
    e = q.elements.numpy()
    assert np.array_equal(e.astype(ml_dtypes.float8_e4m3fn).astype(np.float32), e)
    s = q.scale.numpy()
    assert np.array_equal(s.astype(ml_dtypes.float8_e8m0fnu).astype(np.float32), s)


def test_scale_is_the_smallest_that_avoids_saturation():
    x = _randn(16, 512) * torch.logspace(-20, 20, 16).unsqueeze(1)
    q = quantize_mxfp8(x)
    amax = x.reshape(16, -1, BLOCK).abs().amax(-1)
    assert (amax / q.scale <= 448).all()  # fits
    assert (amax / (q.scale / 2) > 448).all()  # one step smaller would not


def test_exact_powers_of_two_use_the_full_range():
    x = torch.zeros(1, BLOCK)
    x[0, 0] = 448 * 2.0**5
    q = quantize_mxfp8(x)
    assert q.scale_exp.item() == 5 and q.elements[0, 0].item() == 448


def test_nearest_error_is_half_an_ulp():
    x = _randn(64, 1024)
    q = quantize_mxfp8(x)
    rel = (q.dequantize() - x).abs() / x.abs().clamp(min=1e-30)
    big = x.abs() >= 2**-6 * q.scale.repeat_interleave(BLOCK, -1)  # normal range
    assert (rel[big] <= 2**-4 + 1e-7).all()  # 3 mantissa bits


def test_stochastic_is_unbiased():
    x = _randn(1, 256, seed=4)
    gen = torch.Generator().manual_seed(5)
    n = 4000
    mean = sum(quantize_mxfp8(x, stochastic=True, generator=gen).dequantize() for _ in range(n)) / n
    nearest_err = (quantize_mxfp8(x).dequantize() - x).abs().mean()
    assert (mean - x).abs().mean() < nearest_err / 10


def test_zero_blocks_stay_zero():
    x = _randn(2, 64)
    x[0] = 0
    q = quantize_mxfp8(x)
    assert torch.isfinite(q.dequantize()).all() and (q.dequantize()[0] == 0).all()


def test_rejects_ragged_last_axis():
    with pytest.raises(ValueError):
        quantize_mxfp8(torch.ones(2, 40))
