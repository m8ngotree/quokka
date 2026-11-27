// NOT BUILT. The TMA-pipelined persistent variant of the NVFP4 quantizer,
// kept for reading: it was correct but no faster than v2.1 (see
// results/nvfp4_quantize_b200.md, "Tried and reverted: persistent blocks").
// It slots into quokka/cuda/csrc/nvfp4_quantize.cu at v2.1, which provides
// kBlock, kMaxThreads, kE2M1Max, kE4M3Max, Raw16, unpack16, quantize_block
// and a block_max(v, parity) that alternates between two shared slots.

// ---------------------------------------------------------------------------
// Pipelined kernel for rows up to kMaxPipelinedK wide.
//
// The kernel above is limited by bytes in flight: a thread block issues its
// row's loads, then waits through two barriers and the math before it can
// issue more, and the 64-warp limit caps an SM at 7 such rows. Holding a
// second row in registers costs 16 more registers per thread, which drops
// residency to 4 blocks per SM and loses more than it gains.
//
// Here the rows in flight live in shared memory instead. Thread blocks are
// persistent (one wave, each walking rows blockIdx.x, + gridDim.x, ...), and
// each keeps a ring of kStages row buffers. Thread 0 fills a buffer with one
// bulk asynchronous copy (cp.async.bulk, executed by the TMA unit, so no
// thread waits on it or holds registers for it). An mbarrier per buffer
// counts the bytes as they land. The block works on one row while the
// copies of the next kStages - 1 rows are already in flight.
// ---------------------------------------------------------------------------

constexpr int kStages = 3;
constexpr int kMaxPipelinedK = 8192;  // 3 x 16 KB of shared memory per block

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(bar)), "r"(count));
}

// Wait until the barrier's phase `parity` has completed (all bytes landed).
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{ .reg .pred p;\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "  selp.u32 %0, 1, 0, p; }"
        : "=r"(done)
        : "r"(smem_addr(bar)), "r"(parity)
        : "memory");
  }
}

// Copy `bytes` from global to shared memory with the TMA unit; the barrier
// is told to expect them and completes its phase when they have arrived.
__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes,
                                          uint64_t* bar) {
  // Earlier ordinary reads of dst (by every thread) are ordered before this
  // asynchronous write by the caller's __syncthreads and this proxy fence.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(bar)),
               "r"(bytes)
               : "memory");
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(smem_addr(dst)),
      "l"(src), "r"(bytes), "r"(smem_addr(bar))
      : "memory");
}

// (kMaxThreads, 2) caps registers at 65536 / 2048 = 32 per thread, so
// register use never limits residency below what shared memory allows.
__global__ void __launch_bounds__(kMaxThreads, 2)
nvfp4_quantize_pipelined_kernel(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ packed,
                                __nv_fp8_e4m3* __restrict__ scales,
                                float* __restrict__ outer_out, int M, int K) {
  extern __shared__ __align__(128) uint8_t ring[];  // kStages row buffers
  __shared__ __align__(8) uint64_t full[kStages];   // one mbarrier per buffer

  const int nblocks = K / kBlock;
  const uint32_t row_bytes = static_cast<uint32_t>(K) * 2;
  const int step = gridDim.x;
  auto src = [&](int row) { return x + static_cast<size_t>(row) * K; };

  if (threadIdx.x == 0) {
    for (int s = 0; s < kStages; ++s) mbar_init(&full[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // Start the first kStages rows.
    for (int s = 0; s < kStages; ++s) {
      const int row = blockIdx.x + s * step;
      if (row < M) bulk_load(ring + s * row_bytes, src(row), row_bytes, &full[s]);
    }
  }
  __syncthreads();

  int it = 0;
  for (int row = blockIdx.x; row < M; row += step, ++it) {
    const int s = it % kStages;
    mbar_wait(&full[s], (it / kStages) & 1);

    const int b = threadIdx.x;
    const bool live = b < nblocks;
    float v[kBlock];
    float bmax = 0.0f;
    if (live) {
      const uint4* q = reinterpret_cast<const uint4*>(ring + s * row_bytes) + 2 * b;
      Raw16 raw{{q[0], q[1]}};
      unpack16(raw, v);
#pragma unroll
      for (int i = 0; i < kBlock; ++i) bmax = fmaxf(bmax, fabsf(v[i]));
    }

    // block_max's barrier also means every thread has read buffer s, so it
    // can be refilled with the row kStages ahead.
    const float amax = block_max(bmax, it & 1);
    if (threadIdx.x == 0) {
      const int ahead = row + kStages * step;
      if (ahead < M) bulk_load(ring + s * row_bytes, src(ahead), row_bytes, &full[s]);
    }

    const float outer = amax > 0.0f ? amax / (kE2M1Max * kE4M3Max) : 1.0f;
    if (threadIdx.x == 0) outer_out[row] = outer;
    if (live) quantize_block(v, bmax, outer, static_cast<size_t>(row) * nblocks + b, packed, scales);
  }
}

void launch_pipelined(const torch::Tensor& x, torch::Tensor& packed, torch::Tensor& scales,
                      torch::Tensor& outer, int64_t M, int64_t K) {
  const int threads = static_cast<int>((K / kBlock + 31) / 32 * 32);
  const int smem = static_cast<int>(kStages * K * 2);
  // Resident blocks per SM for this block size and buffer size; cached,
  // since it only depends on K.
  static int resident[kMaxPipelinedK / kBlock + 1] = {};
  int& per_sm = resident[K / kBlock];
  if (per_sm == 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(nvfp4_quantize_pipelined_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    // Ask for the largest shared-memory carveout (the rest of the 256 KB
    // goes to L1, which this kernel barely uses).
    C10_CUDA_CHECK(cudaFuncSetAttribute(nvfp4_quantize_pipelined_kernel,
                                        cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, nvfp4_quantize_pipelined_kernel, threads, smem));
  }
  const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = static_cast<int>(std::min<int64_t>(M, per_sm * sms));
  nvfp4_quantize_pipelined_kernel<<<grid, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), packed.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()), outer.data_ptr<float>(),
      static_cast<int>(M), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

