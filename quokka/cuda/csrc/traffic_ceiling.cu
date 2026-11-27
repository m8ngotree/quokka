// A roofline kernel for the NVFP4 quantizer: the same memory traffic and
// none of the math. Per 16 bf16 values it reads 32 bytes and writes 8 + 1
// bytes, in the same layout and with the same streaming hints, so its
// bandwidth is what a quantizer could reach if its arithmetic, reductions
// and barriers cost nothing.
//
// A plain copy (equal reads and writes) is a different traffic mix, so
// "% of copy" mixes two questions: how good the kernel is, and how fast
// this read/write ratio can go at all. This answers the second.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

__global__ void traffic_kernel(const uint4* __restrict__ x, unsigned long long* __restrict__ packed,
                               unsigned char* __restrict__ scales, int64_t groups) {
  for (int64_t g = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; g < groups;
       g += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const uint4 a = __ldcs(x + 2 * g), b = __ldcs(x + 2 * g + 1);
    // Depend on every input word so the loads can't be dropped.
    const uint32_t lo = a.x ^ a.y ^ a.z ^ a.w, hi = b.x ^ b.y ^ b.z ^ b.w;
    __stcs(packed + g, (static_cast<unsigned long long>(hi) << 32) | lo);
    __stcs(scales + g, static_cast<unsigned char>(lo ^ hi));
  }
}

}  // namespace

std::vector<torch::Tensor> traffic_ceiling(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous());
  const int64_t M = x.size(0), K = x.size(1);
  TORCH_CHECK(K % 16 == 0);
  auto packed = torch::empty({M, K / 2}, x.options().dtype(torch::kUInt8));
  auto scales = torch::empty({M, K / 16}, x.options().dtype(torch::kUInt8));
  const int64_t groups = M * K / 16;
  if (groups > 0) {
    const int threads = 256;
    const int64_t blocks = std::min<int64_t>((groups + threads - 1) / threads, 148 * 32);
    traffic_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4*>(x.data_ptr()),
        reinterpret_cast<unsigned long long*>(packed.data_ptr()), scales.data_ptr<uint8_t>(), groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {packed, scales};
}
