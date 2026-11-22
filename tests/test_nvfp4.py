import ml_dtypes
import numpy as np
import pytest
import torch

from quokka.reference.nvfp4 import BLOCK, quantize_nvfp4


def _randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _on_grid(t, dtype):
    a = t.numpy()
    return np.array_equal(a.astype(dtype).astype(np.float32), a)


def test_fields_are_representable():
    q = quantize_nvfp4(_randn(8, 256) * 30)
    assert _on_grid(q.elements, ml_dtypes.float4_e2m1fn)
    assert _on_grid(q.block_scale, ml_dtypes.float8_e4m3fn)
    assert (q.block_scale <= 448).all()


def test_error_is_bounded_per_block():
    x = _randn(16, 512) * torch.logspace(-3, 3, 16).unsqueeze(1)
    q = quantize_nvfp4(x)
    err = (q.dequantize() - x).abs().reshape(16, -1, BLOCK).amax(-1)
    step = q.block_scale * q.outer_scale  # one E2M1 unit in real terms
    amax_b = x.abs().reshape(16, -1, BLOCK).amax(-1)
    # Half the widest E2M1 gap (2 units), or the block-scale rounding
    # error (E4M3 is good to 2**-4) pushing the maximum past the top.
    bound = torch.maximum(step, amax_b * 2**-4) * (1 + 1e-6)
    assert (err <= bound).all()


def test_relative_error_on_gaussian():
    x = _randn(64, 1024)
    q = quantize_nvfp4(x)
    rel = ((q.dequantize() - x) ** 2).mean() / (x**2).mean()
    assert rel < 0.015


def test_zero_rows_and_blocks_stay_zero():
    x = _randn(4, 64)
    x[1] = 0
    x[2, :BLOCK] = 0
    dq = quantize_nvfp4(x).dequantize()
    assert torch.isfinite(dq).all()
    assert (dq[1] == 0).all() and (dq[2, :BLOCK] == 0).all()


def test_per_row_outer_scale_is_batch_invariant():
    x = _randn(32, 128)
    x[7] *= 1000  # an outlier row must not change the other rows
    full = quantize_nvfp4(x).dequantize()
    for i in (0, 7, 31):
        alone = quantize_nvfp4(x[i : i + 1]).dequantize()
        assert torch.equal(full[i : i + 1], alone)


def test_per_expert_outer_scale():
    w = _randn(4, 32, 64)
    w[2] *= 100
    q = quantize_nvfp4(w, outer_dims=(-2, -1))
    assert q.outer_scale.shape == (4, 1, 1)
    for e in range(4):
        alone = quantize_nvfp4(w[e], outer_dims=(-2, -1)).dequantize()
        assert torch.equal(q.dequantize()[e], alone)


def test_rejects_ragged_last_axis():
    with pytest.raises(ValueError):
        quantize_nvfp4(torch.ones(2, 20))
