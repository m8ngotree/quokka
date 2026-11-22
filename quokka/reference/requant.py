"""Re-quantize an NVFP4 forward tensor into transposed MXFP8 for the backward pass.

The backward GEMMs contract over a different axis than the forward one:

- DGRAD multiplies by W transposed, contracting over W's output axis N,
  where the forward pass contracted over K.
- WGRAD contracts over tokens, so it needs the activations transposed.

Block scales only run along the contraction axis, so the forward tensor's
1x16 NVFP4 blocks are the wrong shape for the backward pass. Rather than
switching to square 16x16 blocks (which fit both directions but are coarse),
this dequantizes the exact values the forward pass used, transposes them, and
quantizes them again to MXFP8 along the new axis.

That second rounding must be stochastic. The backward pass then sees, in
expectation, exactly the weights the forward pass used:
E[requantize(q)] == dequantize(q).T. With round-to-nearest the backward
weights would differ from the forward ones by a fixed bias every step, and
that bias compounds across layers and steps; stochastic noise averages away.
"""

from __future__ import annotations

import torch

from quokka.reference.mxfp8 import MXFP8, quantize_mxfp8
from quokka.reference.nvfp4 import NVFP4


def requantize_transposed(
    q: NVFP4,
    *,
    stochastic: bool = True,
    generator: torch.Generator | None = None,
) -> MXFP8:
    """NVFP4 [..., M, K] -> MXFP8 [..., K, M], blocked along M."""
    return quantize_mxfp8(
        q.dequantize().transpose(-1, -2), stochastic=stochastic, generator=generator
    )
