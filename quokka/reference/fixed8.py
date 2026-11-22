"""8-bit fixed point with a power-of-two (E8M0) scale per 32 values.

Each element is a signed 8-bit integer k read as k / 64: 2 integer bits
(including the sign) and 6 fraction bits. That covers [-2, 127/64] in even
steps of 1/64. A value decodes as  (k / 64) * 2**exponent, with one exponent
per 32 consecutive values along the last axis.

Compared with FP8 E4M3, the steps are even rather than relative. FP8 spends
its bits on range: values near the block maximum get only 3 mantissa bits.
Fixed point spends them all on precision across the block. That suits data
that has already been through a Hadamard transform, which flattens each block
so there are no tiny values left that need FP8's extra range.

The exponent is the smallest one that keeps the block maximum within 127/64,
so nothing saturates. Elements round to nearest-even.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from quokka.reference.hadamard import inverse_rht, rht
from quokka.reference.mxfp8 import E8M0_MAX, E8M0_MIN, _ceil_log2

BLOCK = 32
FRAC_BITS = 6
QMIN, QMAX = -128, 127


@dataclass
class Fixed8:
    """A block-scaled fixed-point tensor: int8 codes and E8M0 exponents."""

    codes: torch.Tensor  # [..., K], int8
    scale_exp: torch.Tensor  # [..., K // 32], int32

    def dequantize(self) -> torch.Tensor:
        scale = torch.ldexp(torch.ones_like(self.scale_exp, dtype=torch.float32), self.scale_exp)
        step = scale.repeat_interleave(BLOCK, dim=-1) / 2**FRAC_BITS
        return self.codes.to(torch.float32) * step


def quantize_fixed8(x: torch.Tensor) -> Fixed8:
    """Quantize `x` along its last axis."""
    x = x.to(torch.float32)
    *lead, k = x.shape
    if k % BLOCK:
        raise ValueError(f"last axis {k} is not a multiple of {BLOCK}")

    xb = x.reshape(*lead, k // BLOCK, BLOCK)
    amax = xb.abs().amax(-1)
    top = QMAX / 2**FRAC_BITS
    exp = torch.where(amax > 0, _ceil_log2(amax / top), E8M0_MIN)
    exp = exp.clamp(E8M0_MIN, E8M0_MAX).to(torch.int32)

    step = torch.ldexp(torch.ones_like(amax), exp - FRAC_BITS).unsqueeze(-1)
    codes = torch.round(xb / step).clamp(QMIN, QMAX).to(torch.int8)  # round half to even
    return Fixed8(codes.reshape(x.shape), exp)


def combine_roundtrip(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """What the receiving side of a low-precision combine sees.

    The sender applies the Hadamard transform and quantizes; the receiver
    dequantizes and applies the inverse transform.
    """
    return inverse_rht(quantize_fixed8(rht(x, signs)).dequantize(), signs)
