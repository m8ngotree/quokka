# NVFP4 quantize: engineering journal

What was done to `quokka/cuda/csrc/nvfp4_quantize.cu`, in order, and *why*
each step was chosen. The numbers live in `results/nvfp4_quantize_b200.md`;
this file is the reasoning behind them. New entries go at the bottom.

Every entry follows the same loop, which is the whole method:

1. **Measure** where the time goes (benchmark against a bound, then profile).
2. **Form one hypothesis** that explains the gap.
3. **Predict** what a single change will do *before* running it.
4. **Change one thing**, check it's still bit-exact, measure again.
5. **Keep it or revert it**, and write down which, including the failures.

The prediction step is the one people skip, and it's the one that builds
intuition. A wrong prediction teaches more than a right one.

## Context: what the kernel is for

This is the activation quantizer of an NVFP4-forward / MXFP8-backward MoE
training recipe (Thinking Machines' "end-to-end 4-bit training" post, run on
Inkling-shaped layers). Every token entering an expert GEMM in the forward pass goes
through it. Shapes come from Inkling 2 Small: K = 4608 is the hidden size
(input to the first expert GEMM), K = 2304 is the expert FFN width (input to
the second). M is tokens: 128 to 262144.

## Step 0: decide what "fast" means before writing anything

The kernel reads 2 bytes per value and writes 0.5 (packed FP4) + 1/16
(an E4M3 scale per 16 values), so 2.5625 bytes move per value, and it does
only a handful of operations per byte. That makes it **memory-bound**: the
best it can possibly do is move bytes as fast as the GPU's memory system
allows. So the yardstick is a plain copy of the same bytes
(`torch.clone`), timed the same way. "% of copy" is the score. 100% is
roughly the ceiling for a standalone kernel.

Two traps had to be closed before any number meant anything:

- **L2 cache.** B200 has about 126 MB of L2. Re-quantizing the same small
  tensor repeatedly would be served from cache and look faster than HBM can
  go. The benchmark cycles through 512 MB of distinct inputs.
- **Launch overhead.** A launch costs the CPU about 5 µs. At small M the
  kernel runs for 2 µs, so timing back-to-back launches measured the CPU.
  Fixed later with CUDA graphs (step 5): the GPU replays 50 launches with
  no CPU in between.

## Step 1 (v0): the obvious kernel, 34–51% of copy

One thread block (256 threads) per row. Pass 1: every thread strides over
the row computing |x| max, then a block-wide reduction gives the row amax.
Pass 2: stride over the row again, quantizing.

Why start here: it's correct and simple, and the reference test must pass
before speed matters. It's also a good baseline for *learning*, because its
problems are easy to see once you know where to look.

## Step 2: a false result, and the stale-build bug

The first optimization "compiled in 0.9 s" and gave numbers identical to
v0. Identical to three digits is a smell: real changes move things. The
build system (ninja) decides what to rebuild from file timestamps, and
files mounted into Modal containers don't carry trustworthy ones, so the
B200 was running the old .so.

Fix: name the build directory after a hash of the source and flags. Lesson
worth keeping: **when a result is suspiciously unchanged, verify that the
code you think you're running is the code that ran** (compile time, a
printed hash, a deliberate slowdown).

## Step 3 (v1): one thread per 16-value block, 1.25–2.1×

Hypotheses from reading v0:

- At K = 2304 a row has 144 blocks of 16 but got 256 threads, so 112 threads
  idled. Bad for K = 2304 specifically. That matched the data: K = 2304 was
  the worst case (34%).
- Each row was read from memory twice (pass 1 and pass 2). The second read
  mostly hits L2, but it's still extra traffic and latency.

Change: size the thread block to the row (one thread per 16 values; 288
threads at K = 4608, 144 at 2304). Each thread loads its 32 bytes once with
two 16-byte vector loads, keeps them in registers, joins the reduction, and
quantizes from registers.

Prediction: big win at K = 2304 (no idle threads), smaller at 4608.
Result: matched in direction (1.5–2× at 2304, 1.25–1.5× at 4608). But it
was still only ~51–57% of copy at large M. So idle threads and the second
read weren't the main limit. Something else was.

## Step 4 (v2): one reciprocal instead of 16 divisions, another 1.4–1.5×

Reading the source with "how many instructions per byte?" in mind: each
16-value block did 16 IEEE fp32 divisions `x / denom`. An IEEE-correct
division is not one instruction. It's a reciprocal approximation plus
Newton refinement plus a slow-path check, roughly 8–10 instructions plus a
branch. 16 of those per 32 bytes is a lot of issue slots for a kernel that
should be waiting on memory.

Change: `inv = 1 / denom` once per block, then 16 multiplies. This changes
the numerics slightly (`x * (1/d)` isn't always the same float as `x / d`),
so the reference was changed to compute it the same way. Bit-exactness is a
contract between kernel and reference, not with some abstract ideal.
Quality was checked to be unchanged (relative MSE 0.00907 both ways).
(Two problems with this step turned up later; see step 8.)

Result: 73–82% of copy. This was the real bottleneck: the kernel had been
**issue-bound** (limited by how fast warps can issue instructions), not
memory-bound.

## Step 5: profile v2 with Nsight Compute; two failed ideas

Ran `ncu` on 32768 × 4608 (see `profiling.md` for how to run and read it):

- DRAM throughput 64% of peak, SM throughput 68%. Neither is saturated, so
  this is a **latency** problem (nothing is maxed out, things are waiting),
  not a throughput problem.
- No register spills, 85% achieved occupancy, 32 registers/thread.
- The FP4 conversion compiles to one hardware instruction,
  `F2FP.SATFINITE.E2M1...`, so the conversion isn't expensive.

Failed idea 1: compute |x| and max on packed bf16 pairs (`__hmax2`) to cut
instructions. Exact, fewer instructions, **no faster** (81% vs 82%).
Lesson: removing instructions helps only if they're on the critical path.
Once you're latency-bound, instruction count stops mattering.

Benchmark fix: CUDA graphs, so small M measures the GPU, not launch
overhead (M = 128 went from "4.9 µs" to its real 2.1 µs).

Failed idea 2: two 16-value blocks per thread, to halve the reduction cost
per value. **Slower** (60–67%). Fewer threads means fewer warps in flight,
and warps in flight are how a GPU hides memory latency. This is Little's
law: bytes in flight = bandwidth × latency. Cut the bytes in flight and you
cut the bandwidth.

## Step 6: a full profile with source, three facts

Re-profiled v2 with `--set full` and `-lineinfo`, so stalls map to source
lines (`profiling.md` walks through it; the report is in `artifacts/nvfp4_v2/`):

- **27% of stall samples wait on loads, 18% at the two `__syncthreads()`**
  in `block_max`. The loads → barrier → math → store shape is visible.
- **Loads are half-coalesced at L1.** Each thread's two 16-byte loads are
  32 bytes apart across the warp, so each load instruction touches 32 sectors
  and uses half of each. DRAM and L2 still see each byte once (302 MB read
  = the input); only L1 moves 2× the sectors.
- **`ITEMS=8` uses 480 bytes of stack** (`resources.txt`), because
  `__launch_bounds__(1024)` caps registers at 64 and 128 floats don't fit.
  Only K > 16384 uses it.

## Step 7: audit the benchmark before trusting more numbers

A review of the timing code (from another chat) raised two points. Each
was checked against the code rather than assumed:

- *"The event loop measures launch overhead at small M."* True of the
  original loop, already fixed in step 5 by CUDA graphs.
- *"The graph only replays the first 50 buffers, so at M = 128 the inputs
  fit in L2."* **True.** 50 × 1.2 MB = 59 MB of inputs, plus outputs, stays
  in the 126 MB L2 between the warm-up replay and the timed one. Fix: the
  graph replays every buffer (≥ 512 MB per replay), and five replays are
  timed separately (median reported, plus the max/min spread). Only
  M = 128 moved: 2.1 → 2.3 µs.

Lesson: a benchmark is code, and it has bugs. Every claim in its docstring
("rotates through 512 MB") should be checked against what it actually does.

## Step 8: is the reciprocal trick valid?

Two concerns, both measured:

- **Tiny rows produced garbage.** `1 / denom` overflows to inf when
  `denom < ~2.9e-39`, i.e. rows whose amax is below ~1.8e-38 (bf16's
  subnormal range). Then `x * inf` saturates every element to ±6, and
  `0 * inf` is NaN. The reference did the same, so the bit-exact tests
  passed, and `test_every_bf16_magnitude` passed while its first rows
  quantized to all-6s. Fix: when the reciprocal is inf, divide, in both
  the reference and the kernel. The branch is never taken on real data;
  timings unchanged. New tests cover rows at 1e-36 … 1e-40.
- **How often does it differ from true division?** On fp32 Gaussian inputs,
  0 of 18.9 M codes differ. On bf16 inputs, 4131 of 18.9 M (0.022%) differ,
  each by one grid step. bf16 has few mantissa bits, so `x / d` lands
  exactly on a rounding midpoint more often, and `x * (1/d)`, off by one
  ulp, breaks the tie the other way.

Is it still a valid kernel? Yes. NVFP4 fixes how values *decode*
(element × block_scale × outer); how you choose the element is up to the
quantizer. TransformerEngine, vLLM and FlashInfer all use reciprocals, by
default *approximate* ones, so none of them matches true division either.
What makes a kernel valid is a documented reference that it matches
exactly, and quality that's measured (MSE unchanged). What was wrong was
the untested edge, not the choice.

## Step 9: memory-side ideas, measured

- **One 32-byte load per thread** (`ld.global.nc.v8.b32`, new on sm_100;
  SASS `LDG.E.ENL2.256`): fully coalesced, **no change in time**. The 2× L1
  sectors from step 6 weren't on the critical path.
- **Two threads per 16-value block** (16-byte load each, one shuffle to join
  the block max): **2× slower.** A K = 4608 row needs 576 threads, so 3
  rows fit on an SM under the 64-warp limit instead of 7. Fewer rows in
  flight means fewer bytes in flight, and by Little's law
  (bandwidth = bytes in flight / latency) bandwidth drops with them.
- **Streaming hints** (`__ldcs`/`__stcs`, evict-first): every byte is
  touched once, so nothing is worth caching. **Up to 6% faster** at
  mid-sized shapes, nothing slower. Kept (v2.1).

## Step 10: what's the real ceiling? And persistent / TMA blocks

"% of copy" compares against a clone, which reads and writes equal bytes.
The quantizer reads 2 bytes per value and writes 0.56. Is that mix slower
or faster for HBM? A **traffic-ceiling kernel** does exactly the
quantizer's loads and stores with no math. It runs at about copy speed, so
there's real headroom: ~5.4 TB/s against ~6.5.

Then the hypothesis from step 6: blocks can't issue more loads while they
sit in barriers, so make blocks persistent and fetch the next row early.

- **Next row prefetched into registers**: 3–7% *slower*. 48 registers
  instead of 32 means 4 blocks per SM instead of 7.
- **A TMA ring**: three row buffers in shared memory, each filled by one
  `cp.async.bulk` and tracked by an mbarrier. Load stalls nearly vanished
  (27% → 4% of samples), so the pipeline worked. It still lost, because
  37 registers and a 168 KB carveout left 5 blocks per SM. With 32
  registers and the maximum carveout (7 blocks again) it *ties* v2.1 at the
  largest shape and loses at small M.

Why: v2 issues 56.85 M warp-instructions at 32768 × 4608. Across
148 SMs × 4 schedulers that's ~96 K cycles each, **~50 µs of pure
instruction issue** at 1.9 GHz, against 71 µs total and a 59 µs traffic
floor. Hiding memory latency can't help much when issuing instructions
alone takes most of the time. The ncu headline in step 6 had said as much:
*"Compute and Memory are well-balanced"*.

## Step 11: fewer instructions, and the register cliff (v2.2)

The most wasteful SASS sequence was the FP4 packing: each converted byte
returned to C++ as a `uint8` and was masked, shifted and OR-ed, about 15
integer instructions per block.

- **PTX packing** (`cvt.rn.satfinite.e2m1x2.f32` into `.b8` registers, then
  `mov.b32 {b0,b1,b2,b3}`): ptxas chains the conversions through `F2FP`'s
  merge operand, with no masks. **−5.6% instructions**, but *17% slower*,
  because registers went 32 → 37.
- **The register cliff.** At 288 threads per block (K = 4608), 32 registers
  fits 7 blocks per SM (63 warps); 33–40 fits 5. A `__byte_perm` tree in C++
  hit the same cliff. `__launch_bounds__(1024, 2)` holds the one-item kernel
  at 32 registers. It costs one spill (the scale byte, `STL`/`LDL`), and the
  result is **2–3% faster at large M: 85% of the ceiling** at 32768 × 4608.
  Kept (v2.2).
- **Paired multiplies** (`mul.rn.f32x2`, SASS `FMUL2`, exact per lane):
  another −4.3% instructions, no faster. At 85% of the ceiling, issue is no
  longer the only limit. Reverted.

## Where it stands

v2.2: 76–85% of the traffic ceiling at training sizes, 79–84% at small M.
Pretty much every idea has now been tried and measured, so the remaining
gap is shared between barriers, load latency and issue, with no single
dominant cause (v2.2 stalls: long-scoreboard 28%, barrier 19%, wait 12%).

Next, in order of expected value:

- **Fusion**: quantize inside the kernel that produces the activations,
  so the bf16 read disappears. No standalone tuning can match that.
- **Programmatic dependent launch** for small M, where each call is one
  short wave of blocks and the gaps between kernels matter.
- **Four Over Six** in the kernel: more math per byte, which changes which
  of these trade-offs wins. Re-check the register count first.
