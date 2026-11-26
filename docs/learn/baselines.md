# NVFP4 quantize: what to beat

Public NVFP4 quantizers that do what ours does (bf16 → FP4 E2M1 + an E4M3
scale per 16 values + an fp32 outer scale), and how they differ. None of
them has been benchmarked against quokka yet: that's the next step, and
the numbers will go in `results/`.

## The candidates

| Library | Where | Outer scale | Notes |
|---|---|---|---|
| **FlashInfer** | `flashinfer/quantization/kernels/nvfp4_quantize.py` (CuTe DSL) and a CUDA backend from TensorRT-LLM; `nvfp4_quantize(..., backend=..., per_token_activation=...)` | per tensor (passed in), or **per token** computed in-kernel | The closest match. Three scale layouts (linear, swizzled 128×4, 8×4). Optional Four Over Six. Fast-math reciprocal by default (`FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH=1` turns it off). Ships its own benchmark, `benchmarks/bench_nvfp4_quantize_backend_comparison.py`, with a bandwidth mode. |
| **vLLM** | `csrc/quantization/fp4/nvfp4_quant_kernels.cu`, `nvfp4_utils.cuh` | per tensor, **passed in** | 512-thread blocks looping over rows; 16 values per thread on Blackwell; `rcp.approx.ftz` for the scale; writes the swizzled scale layout the GEMM wants. Per-token use needs a separate amax pass. |
| **TensorRT-LLM** | `cpp/tensorrt_llm/kernels/quantization.cu(h)` | per tensor | The ancestor of the vLLM and FlashInfer CUDA kernels. |
| **TransformerEngine** | `transformer_engine/common/cast/nvfp4/` (`quantize_transpose_nvfp4.cuh`, `quantize_4over6_*`, `group_quantize_transpose_nvfp4.cuh`) | per tensor; per row via a **separate** `compute_rowwise_amax_kernel` | The training path: also writes the transposed (column-wise) copy, with optional Hadamard and stochastic rounding, and uses TMA. It does more work than ours, so compare it to the fused kernel, not this one. |
| torchao, fouroversix | `torchao/prototype/moe_training/nvfp4_training/` (Triton and CuTe DSL), `mit-han-lab/fouroversix` `src/fouroversix/csrc/quantize/` | various | Research and training prototypes; worth reading. |

## The thing to beat

**FlashInfer `nvfp4_quantize` with the CuTe DSL backend,
`per_token_activation=True`.** Same input, same per-row outer scale computed
in the same kernel, and it's what serving stacks actually call. vLLM is the
second baseline; it gets a per-tensor scale passed in, so it skips our row
reduction and is a "how fast without the barrier" reference point.

## Making the comparison fair

- **Same scale layout.** GEMMs want scales swizzled (128×4 tiles); we
  write them linearly. Benchmark FlashInfer with `layout_linear` for an
  apples-to-apples comparison, then add a swizzled output to ours, since
  that's what a real pipeline needs.
- **Same numerics, stated.** Theirs uses approximate reciprocals by default.
  Ours uses an exact IEEE reciprocal and is bit-exact with a reference
  (which differs from true division on 0.022% of codes for bf16 inputs;
  journal step 8). Measure theirs both ways,
  and report how often the two disagree.
- **Same timing method.** CUDA graphs replaying every buffer of a
  512 MB rotation, median of 5 replays, on the same GPU at the same shapes.

- **Same ceiling.** Report each against the traffic-ceiling kernel
  (`bench/nvfp4_quantize.py`'s "% of ceiling"), not only against a copy.

## How we could beat it

v2.2 is at 76–85% of its traffic ceiling at training sizes, so a standalone
quantizer has at most ~1.2× left. The journal (steps 9–11) records what
has already been tried: coalesced 32-byte loads, persistent blocks, a TMA
pipeline and paired multiplies were all measured and didn't help, for
reasons the profile explains. What's left:

1. **Small M** (M = 128 is 2.3 µs): latency, not bandwidth. Programmatic
   dependent launch (PDL) lets the kernel start while the previous one
   finishes, which matters more here than any inner-loop work.
2. **The remaining stalls are spread out** (loads 28%, barriers 19%,
   dependencies 12%), so the next standalone gain probably needs a
   different structure, not a tweak, and must stay within 32 registers.
3. **Don't be standalone.** The real win is fusion: quantize inside the
   kernel that produces the activations (RMSNorm, SwiGLU, the w2 epilogue),
   so the bf16 tensor never goes to HBM at all. That saves the whole
   2 bytes/value read, which no standalone kernel can. This is where the
   4-bit training recipe's speedups come from, and where the comparison
   should eventually be made: fused quokka vs. FlashInfer's fused
   `rmsnorm_fp4quant` / `silu_and_mul_nvfp4_quantize`.
