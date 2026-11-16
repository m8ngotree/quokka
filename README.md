# quokka

Low-precision kernels across hardware accelerators: TPU, GPU and Trainium.

## What's here

Kernels where the number format matters to the speed:

- **Quantized GEMMs**: int8, FP8 and block-scaled 4- and 8-bit formats.
- **Grouped GEMMs for MoE**: quantized expert matmuls.
- **Quantize and dequantize**: block scaling, rounding modes and layout transforms, fused into the matmuls where possible.
- **Low-precision attention.**

Each kernel family is written for every backend that can run it well:

| Backend | Language |
|---|---|
| TPU | Pallas (JAX) |
| GPU | CuTe DSL, CUDA, Triton |
| Trainium | NKI |

Hardware differs in which formats its matrix units run natively, and in where scales are applied. So the same idea can end up as a different kernel on each backend.

## How each kernel is held to account

1. **A reference** in plain fp32 code defines the numerics, and every kernel is tested against it.
2. **Baselines**: each kernel is benchmarked against the best existing implementation on the same hardware and shapes.
3. **A roofline** gives the memory and compute bounds, so each result is reported as a fraction of what the hardware can do.

## License

Apache-2.0, see [LICENSE](LICENSE).
