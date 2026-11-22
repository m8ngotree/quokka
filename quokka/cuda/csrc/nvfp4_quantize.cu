// NVFP4 quantize: bf16 [M, K] -> packed FP4 E2M1 [M, K/2], E4M3 scales [M, K/16],
// FP32 outer scale [M, 1] (one per row, i.e. per token).
//
// Matches quokka/reference/nvfp4.py bit for bit. Every float operation below is
// the same IEEE fp32 operation the reference does, in the same order, and the
// two casts round to nearest-even and saturate exactly like the reference:
//
//   outer       = row_amax / (6 * 448)           (1 if the row is all zero)
//   block_scale = e4m3(block_amax / (6 * outer))
//   element     = e2m1(x / (block_scale * outer)) (0 if that product is 0)
//
// One thread block per row, two passes over the row:
//   1. every thread takes the amax of some 16-value blocks; a block-wide
//      reduction gives the row amax, hence the outer scale;
//   2. every thread quantizes its 16-value blocks: block amax, E4M3 scale,
//      then 16 E2M1 elements packed two per byte into one 8-byte store.
// Pass 2 re-reads the row; at these sizes it is still in L1/L2. Fusing the two
// passes is a later optimization.
//
// Packing: byte i of a row holds element 2i in its low nibble and element
// 2i + 1 in its high nibble.

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int kBlock = 16;      // values per E4M3 scale
constexpr int kThreads = 256;   // threads per row
constexpr float kE2M1Max = 6.0f;
constexpr float kE4M3Max = 448.0f;

// Load 16 consecutive bf16 values (32 bytes, two 16-byte loads) as floats.
__device__ __forceinline__ void load16(const __nv_bfloat16* p, float v[kBlock]) {
  const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint4 raw = q[h];
    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 f = __bfloat1622float2(pairs[i]);
      v[h * 8 + 2 * i] = f.x;
      v[h * 8 + 2 * i + 1] = f.y;
    }
  }
}

__device__ __forceinline__ float block_max(float v) {
  __shared__ float warp_max[kThreads / 32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (lane == 0) warp_max[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < kThreads / 32 ? warp_max[lane] : 0.0f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (lane == 0) warp_max[0] = v;
  }
  __syncthreads();
  return warp_max[0];
}

__global__ void __launch_bounds__(kThreads)
nvfp4_quantize_kernel(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ packed,
                      __nv_fp8_e4m3* __restrict__ scales, float* __restrict__ outer_out, int K) {
  const int row = blockIdx.x;
  const int nblocks = K / kBlock;
  const __nv_bfloat16* xr = x + static_cast<size_t>(row) * K;

  // Pass 1: row amax -> outer scale.
  float amax = 0.0f;
  for (int b = threadIdx.x; b < nblocks; b += kThreads) {
    float v[kBlock];
    load16(xr + b * kBlock, v);
#pragma unroll
    for (int i = 0; i < kBlock; ++i) amax = fmaxf(amax, fabsf(v[i]));
  }
  amax = block_max(amax);
  const float outer = amax > 0.0f ? amax / (kE2M1Max * kE4M3Max) : 1.0f;
  if (threadIdx.x == 0) outer_out[row] = outer;

  // Pass 2: per-block E4M3 scale, then 16 E2M1 elements.
  for (int b = threadIdx.x; b < nblocks; b += kThreads) {
    float v[kBlock];
    load16(xr + b * kBlock, v);
    float bmax = 0.0f;
#pragma unroll
    for (int i = 0; i < kBlock; ++i) bmax = fmaxf(bmax, fabsf(v[i]));

    const __nv_fp8_e4m3 scale8(bmax / (kE2M1Max * outer));  // RNE, saturating
    const float denom = static_cast<float>(scale8) * outer;

    uint64_t out = 0;
    if (denom > 0.0f) {
#pragma unroll
      for (int i = 0; i < kBlock; i += 2) {
        const float2 pair = make_float2(v[i] / denom, v[i + 1] / denom);
        const __nv_fp4x2_storage_t two = __nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest);
        out |= static_cast<uint64_t>(two) << (4 * i);
      }
    }
    const size_t blk = static_cast<size_t>(row) * nblocks + b;
    reinterpret_cast<uint64_t*>(packed)[blk] = out;
    scales[blk] = scale8;
  }
}

}  // namespace

std::vector<torch::Tensor> nvfp4_quantize(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16, "x must be a CUDA bf16 tensor");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "x must be a contiguous 2-D tensor");
  const int64_t M = x.size(0), K = x.size(1);
  TORCH_CHECK(K % kBlock == 0, "K must be a multiple of 16");

  auto packed = torch::empty({M, K / 2}, x.options().dtype(torch::kUInt8));
  auto scales = torch::empty({M, K / kBlock}, x.options().dtype(torch::kFloat8_e4m3fn));
  auto outer = torch::empty({M, 1}, x.options().dtype(torch::kFloat32));
  if (M > 0 && K > 0) {
    nvfp4_quantize_kernel<<<M, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), packed.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()), outer.data_ptr<float>(),
        static_cast<int>(K));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {packed, scales, outer};
}
