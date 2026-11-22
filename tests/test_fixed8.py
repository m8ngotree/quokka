import pytest
import torch

from quokka.reference.fixed8 import BLOCK, combine_roundtrip, quantize_fixed8
from quokka.reference.hadamard import random_signs
from quokka.reference.mxfp8 import quantize_mxfp8


def _randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _with_outliers(seed=0):
    """Gaussian rows with a few channels 50x larger, like real activations."""
    x = _randn(64, 1024, seed=seed)
    x[:, ::97] *= 50
    return x


def _rel_mse(a, b):
    return float(((a - b) ** 2).mean() / (b**2).mean())


def test_codes_and_scale():
    q = quantize_fixed8(_randn(8, 256) * 1e3)
    assert q.codes.dtype == torch.int8
    amax = (_randn(8, 256) * 1e3).reshape(8, -1, BLOCK).abs().amax(-1)
    scale = torch.ldexp(torch.ones_like(amax), q.scale_exp)
    assert (amax / scale <= 127 / 64).all()  # never saturates
    assert (amax / (scale / 2) > 127 / 64).all()  # smallest such scale


def test_error_is_half_a_step():
    x = _randn(16, 512)
    q = quantize_fixed8(x)
    ones = torch.ones_like(q.scale_exp, dtype=torch.float32)
    step = torch.ldexp(ones, q.scale_exp - 6).repeat_interleave(BLOCK, -1)
    assert ((q.dequantize() - x).abs() <= step / 2 + 1e-7).all()


def test_zero_blocks_stay_zero():
    x = _randn(2, 64)
    x[1] = 0
    dq = quantize_fixed8(x).dequantize()
    assert torch.isfinite(dq).all() and (dq[1] == 0).all()


def test_hadamard_cuts_error_on_outliers():
    x = _with_outliers()
    signs = random_signs(32, generator=torch.Generator().manual_seed(0))
    plain = _rel_mse(quantize_fixed8(x).dequantize(), x)
    rotated = _rel_mse(combine_roundtrip(x, signs), x)
    assert rotated < plain / 4


def test_rotated_fixed8_beats_mxfp8_on_outliers():
    x = _with_outliers(seed=1)
    signs = random_signs(32, generator=torch.Generator().manual_seed(1))
    assert _rel_mse(combine_roundtrip(x, signs), x) < _rel_mse(quantize_mxfp8(x).dequantize(), x)


def test_rejects_ragged_last_axis():
    with pytest.raises(ValueError):
        quantize_fixed8(torch.ones(2, 40))
