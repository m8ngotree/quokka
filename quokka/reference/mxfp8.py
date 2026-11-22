"""MXFP8: FP8 E4M3 elements with a power-of-two (E8M0) scale per 32 values.

A value decodes as  element * 2**exponent, one exponent per 32 consecutive
values along the last axis. E8M0 is just an 8-bit exponent: no sign, no
mantissa, so the scale is exact but coarse. Unlike NVFP4 there is no outer
scale; 8 exponent bits already cover any range training needs.

The scale is chosen as the smallest power of two that keeps the block's
largest value within E4M3's 448, i.e. 2**ceil(log2(amax / 448)). Rounding the
exponent up rather than down means no element ever saturates; the cost is at
most one bit of headroom left unused.

Elements can be rounded to nearest-even or stochastically. The backward pass
uses stochastic rounding: its gradients are summed over many tokens and
steps, so a small bias would accumulate, while zero-mean noise averages away.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from quokka.reference.rounding import E4M3, round_to_grid

BLOCK = 32
E8M0_MIN, E8M0_MAX = -127, 127


@dataclass
class MXFP8:
    """An MXFP8 tensor in fp32 storage: each field holds exactly representable values."""

    elements: torch.Tensor  # [..., K], on the E4M3 grid
    scale_exp: torch.Tensor  # [..., K // 32], int32 exponent of the E8M0 scale

    @property
    def scale(self) -> torch.Tensor:
        return torch.ldexp(torch.ones_like(self.scale_exp, dtype=torch.float32), self.scale_exp)

    def dequantize(self) -> torch.Tensor:
        return self.elements * self.scale.repeat_interleave(BLOCK, dim=-1)


def _ceil_log2(v: torch.Tensor) -> torch.Tensor:
    """Exact ceil(log2(v)) for v > 0: v = m * 2**e with m in [0.5, 1)."""
    m, e = torch.frexp(v)
    return torch.where(m == 0.5, e - 1, e)


def quantize_mxfp8(
    x: torch.Tensor,
    *,
    stochastic: bool = False,
    generator: torch.Generator | None = None,
) -> MXFP8:
    """Quantize `x` along its last axis."""
    x = x.to(torch.float32)
    *lead, k = x.shape
    if k % BLOCK:
        raise ValueError(f"last axis {k} is not a multiple of {BLOCK}")

    xb = x.reshape(*lead, k // BLOCK, BLOCK)
    amax = xb.abs().amax(-1)
    exp = torch.where(amax > 0, _ceil_log2(amax / E4M3.max_val), E8M0_MIN)
    exp = exp.clamp(E8M0_MIN, E8M0_MAX).to(torch.int32)

    scale = torch.ldexp(torch.ones_like(amax), exp).unsqueeze(-1)
    elements = round_to_grid(xb / scale, E4M3, stochastic=stochastic, generator=generator)
    return MXFP8(elements.reshape(x.shape), exp)
