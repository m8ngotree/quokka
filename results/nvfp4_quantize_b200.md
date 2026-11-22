# NVFP4 quantize on B200: v0

Kernel: `quokka/cuda/csrc/nvfp4_quantize.cu` at v0 (one thread block per row,
two passes, IEEE division per element). Bit-exact with `quokka/reference/nvfp4.py`.

"copy GB/s" is a bf16 `torch.clone` of the same input timed the same way: the
bandwidth this GPU actually reaches. Inputs rotate through 512 MB so L2 can't
serve repeat reads.

| M | K | quantize (us) | GB/s | copy GB/s | % of copy |
|---:|---:|---:|---:|---:|---:|
| 128 | 7168 | 9.2 | 256 | 548 | 47% |
| 1024 | 7168 | 9.5 | 1981 | 4490 | 44% |
| 4096 | 7168 | 25.1 | 2999 | 6845 | 44% |
| 16384 | 7168 | 87.5 | 3439 | 6518 | 53% |
| 65536 | 4096 | 219.9 | 3130 | 6677 | 47% |

Measured 2026-11-22 on Modal (NVIDIA B200, driver 620.70, CUDA 13.5, torch 2.15).

## Why roughly half

Likely causes, to test one at a time:

- **Per-element IEEE division.** `x / denom` costs a multi-instruction
  division for every value; one reciprocal per block plus multiplies would
  be far cheaper. The reference would have to adopt the same rounding to stay
  bit-exact.
- **Two passes over each row.** Pass 2 re-reads the row (from L1/L2, but
  still extra load latency) after a block-wide barrier.
- **Little memory parallelism per thread.** With K = 7168 there are 448
  blocks per row for 256 threads: under two 32-byte loads in flight each.
- **Small M is latency-bound.** At M = 128 there are only 128 thread blocks
  for 148 SMs; the copy is also slow there (548 GB/s).
