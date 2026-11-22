"""The CUDA NVFP4 quantize kernel must match the reference bit for bit."""

import pytest
import torch

from quokka.reference.nvfp4 import quantize_nvfp4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _check(x: torch.Tensor):
    from quokka.cuda import nvfp4_quantize, unpack_e2m1

    packed, scales, outer = nvfp4_quantize(x)
    ref = quantize_nvfp4(x.float().cpu())
    assert torch.equal(outer.cpu(), ref.outer_scale)
    assert torch.equal(scales.float().cpu(), ref.block_scale)
    assert torch.equal(unpack_e2m1(packed).cpu(), ref.elements)


@pytest.mark.parametrize("m,k", [(1, 16), (7, 64), (128, 4096), (33, 7168), (3, 16 * 300)])
def test_matches_reference(m, k):
    gen = torch.Generator().manual_seed(m * 10007 + k)
    x = torch.randn(m, k, generator=gen) * torch.logspace(-4, 4, m).unsqueeze(1)
    _check(x.to(torch.bfloat16).cuda())


def test_outliers_zero_rows_and_zero_blocks():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(16, 512, generator=gen)
    x[3] = 0
    x[5, :16] = 0
    x[7, 100] = 3e4
    x[9] *= 1e-30  # deep in bf16's subnormal-scale range
    _check(x.to(torch.bfloat16).cuda())


def test_every_bf16_magnitude():
    # All finite positive bf16 bit patterns, with random signs, 16 to a block.
    bits = torch.arange(0x0001, 0x7F80, dtype=torch.int32).to(torch.int16)
    vals = bits.view(torch.bfloat16).float()
    vals = vals[: vals.numel() // 256 * 256]
    sign = torch.randint(0, 2, vals.shape, generator=torch.Generator().manual_seed(1)) * 2 - 1
    _check((vals * sign).reshape(-1, 256).to(torch.bfloat16).cuda())


def test_empty_rows():
    from quokka.cuda import nvfp4_quantize

    packed, scales, outer = nvfp4_quantize(torch.empty(0, 64, dtype=torch.bfloat16, device="cuda"))
    assert packed.shape == (0, 32) and scales.shape == (0, 4) and outer.shape == (0, 1)
