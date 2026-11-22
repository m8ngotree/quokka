"""Random Hadamard transform (RHT) over blocks of the last axis.

H is the n x n Sylvester-Hadamard matrix scaled by 1/sqrt(n), so H @ H = I.
D is a diagonal of random +-1 signs. The transform maps each block of n
values v to (v * signs) @ H, and the inverse maps y back to (y @ H) * signs.

Why it helps quantization: H spreads every input value across all n outputs.
A block with one large outlier and many small values becomes a block of
values of similar size, so its maximum (and therefore its scale) drops, and
the small values stop being rounded to zero. The random signs stop
structured inputs from lining up with H's rows. Because the transform is
orthogonal it loses nothing: after dequantizing, the inverse recovers the
values up to quantization error.
"""

from __future__ import annotations

import torch


def hadamard_matrix(n: int) -> torch.Tensor:
    """The n x n Sylvester-Hadamard matrix, scaled to be orthonormal."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"n must be a power of two, got {n}")
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n**0.5


def random_signs(n: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randint(0, 2, (n,), generator=generator).float() * 2 - 1


def rht(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Apply the transform to each block of len(signs) values along the last axis."""
    n = signs.numel()
    *lead, k = x.shape
    if k % n:
        raise ValueError(f"last axis {k} is not a multiple of {n}")
    blocks = x.to(torch.float32).reshape(*lead, k // n, n)
    return ((blocks * signs) @ hadamard_matrix(n)).reshape(x.shape)


def inverse_rht(y: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    n = signs.numel()
    *lead, k = y.shape
    blocks = y.to(torch.float32).reshape(*lead, k // n, n)
    return ((blocks @ hadamard_matrix(n)) * signs).reshape(y.shape)
