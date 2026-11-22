"""Rounding onto a small floating-point grid, to nearest or stochastically.

Every low-precision element format here is a sign-magnitude float with a few
mantissa bits. FP4 E2M1 holds 0, 0.5, 1, 1.5, 2, 3, 4, 6. FP8 E4M3 has 3
mantissa bits, a smallest normal exponent of -6, and a largest value of 448.

Between two neighbouring grid points the spacing (one ulp) is
2 ** (max(exponent(x), emin) - mant_bits). Clamping the exponent at emin makes
the subnormal range evenly spaced. Given that spacing:

- Round-to-nearest-even picks the closer neighbour, and on an exact tie the one
  whose last bit is even. A hardware cast (`cvt.rn`) does this, and the tests
  check it bit for bit against ml_dtypes.
- Stochastic rounding picks the upper neighbour with probability
  (x - lower) / ulp, so E[round(x)] == x. Hardware does this with `cvt.rs`,
  which takes the random bits as an operand. It is used wherever bias would
  build up across training steps.

Values past max_val saturate to +-max_val, like the `.satfinite` casts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FloatGrid:
    """A sign-magnitude float format: mantissa bits, smallest exponent, largest value."""

    name: str
    mant_bits: int
    emin: int
    max_val: float


E2M1 = FloatGrid("e2m1", mant_bits=1, emin=0, max_val=6.0)
E4M3 = FloatGrid("e4m3", mant_bits=3, emin=-6, max_val=448.0)


def _ulp(mag: torch.Tensor, grid: FloatGrid) -> torch.Tensor:
    # frexp is exact, unlike log2: mag = m * 2**e with m in [0.5, 1).
    _, e = torch.frexp(mag)
    exp = torch.clamp(e - 1, min=grid.emin)
    return torch.ldexp(torch.ones_like(mag), exp - grid.mant_bits)


def round_to_grid(
    x: torch.Tensor,
    grid: FloatGrid,
    *,
    stochastic: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Round `x` onto `grid`. Returns float32 values that lie exactly on the grid."""
    x = x.to(torch.float32)
    mag = torch.clamp(x.abs(), max=grid.max_val)
    ulp = _ulp(mag, grid)
    lower = torch.floor(mag / ulp) * ulp
    frac = (mag - lower) / ulp  # in [0, 1), exact: ulp is a power of two

    if stochastic:
        u = torch.rand(x.shape, generator=generator, device=x.device)
        up = u < frac
    else:
        lower_is_odd = torch.remainder(lower / ulp, 2.0) == 1.0
        up = (frac > 0.5) | ((frac == 0.5) & lower_is_odd)

    out = torch.where(up, lower + ulp, lower)
    out = torch.clamp(out, max=grid.max_val)
    return torch.copysign(out, x)
