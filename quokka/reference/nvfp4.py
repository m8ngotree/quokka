"""NVFP4: FP4 E2M1 elements, an E4M3 scale per 16 values, and an FP32 outer scale.

A value decodes as  element * block_scale * outer_scale.

- element: E2M1, one of +-{0, 0.5, 1, 1.5, 2, 3, 4, 6}.
- block_scale: E4M3, one per 16 consecutive values along the last axis
  (the contraction axis K of a GEMM). It maps the block's largest magnitude
  onto the largest element.
- outer_scale: FP32, shared by a group of blocks. E4M3 scales only reach 448,
  so the outer scale brings each group into that range. Which group shares it
  is a choice: one per row (per token for activations), one per expert for
  weights, or one per tensor. A per-row scale makes each row's encoding
  independent of the rest of the batch.

Elements are rounded to nearest-even, which the forward pass uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from quokka.reference.rounding import E2M1, E4M3, round_to_grid

BLOCK = 16


@dataclass
class NVFP4:
    """An NVFP4 tensor in fp32 storage: each field holds exactly representable values."""

    elements: torch.Tensor  # [..., K], on the E2M1 grid
    block_scale: torch.Tensor  # [..., K // 16], on the E4M3 grid
    outer_scale: torch.Tensor  # broadcastable to [..., 1], fp32

    def dequantize(self) -> torch.Tensor:
        scale = self.block_scale.repeat_interleave(BLOCK, dim=-1) * self.outer_scale
        return self.elements * scale


def _quantize_with_target(x_blocks, amax_blocks, outer, target):
    """Scale each block so its largest magnitude lands on `target`, then round."""
    raw = amax_blocks / (target * outer)
    block_scale = round_to_grid(raw, E4M3)
    denom = (block_scale * outer).unsqueeze(-1)
    safe = torch.where(denom > 0, denom, torch.ones_like(denom))
    elements = torch.where(denom > 0, round_to_grid(x_blocks / safe, E2M1), 0.0)
    return elements, block_scale


def quantize_nvfp4(
    x: torch.Tensor,
    *,
    outer_dims: tuple[int, ...] = (-1,),
) -> NVFP4:
    """Quantize `x` along its last axis.

    outer_dims: the axes that share one outer scale. (-1,) is one per row
    (per token); (-2, -1) on [experts, N, K] weights is one per expert.
    """
    x = x.to(torch.float32)
    *lead, k = x.shape
    if k % BLOCK:
        raise ValueError(f"last axis {k} is not a multiple of {BLOCK}")

    # Size the outer scale so the largest block scale is exactly E4M3's max.
    amax_outer = x.abs().amax(dim=outer_dims, keepdim=True)
    outer = torch.where(amax_outer > 0, amax_outer / (6.0 * E4M3.max_val), 1.0)

    xb = x.reshape(*lead, k // BLOCK, BLOCK)
    amax_b = xb.abs().amax(-1)
    elements, block_scale = _quantize_with_target(xb, amax_b, outer, 6.0)
    return NVFP4(elements.reshape(x.shape), block_scale, outer)
