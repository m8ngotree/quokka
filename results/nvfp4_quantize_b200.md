# NVFP4 quantize on B200

Kernel: `quokka/cuda/csrc/nvfp4_quantize.cu`, bit-exact with
`quokka/reference/nvfp4.py` at every version. Benchmark: `bench/nvfp4_quantize.py`.

"% of copy" is achieved bandwidth divided by a bf16 `torch.clone` of the same
input timed the same way, which is what this GPU actually reaches (about
6.5 TB/s at large M, against 8 TB/s on paper). Inputs rotate through 512 MB
so L2 can't serve repeat reads. Shapes are the activations of a 4-bit MoE
model: K = 4608 (hidden size) and K = 2304 (expert FFN size).

Measured on Modal: NVIDIA B200, driver 620.70, CUDA 13.5, torch 2.15.

## v0: one thread block of 256 threads per row, two passes

| M | K | us | GB/s | % of copy |
|---:|---:|---:|---:|---:|
| 128 | 4608 | 10.9 | 139 | 43% |
| 1024 | 4608 | 9.8 | 1234 | 51% |
| 8192 | 4608 | 35.7 | 2708 | 42% |
| 32768 | 4608 | 129.8 | 2981 | 45% |
| 1024 | 2304 | 9.7 | 625 | 49% |
| 65536 | 2304 | 173.5 | 2232 | 34% |
| 262144 | 2304 | 683.0 | 2268 | 34% |

K = 2304 is worst: a row has 144 blocks of 16 values but gets 256 threads,
so 112 threads per row do nothing.

## v1: one thread per 16-value block, row held in registers

Each thread loads its 32 bytes once, joins the row max, and quantizes from
registers. The thread block is sized to the row (288 threads at K = 4608, 144
at K = 2304), so no thread idles and the row is read once.

| M | K | us | GB/s | % of copy | vs v0 |
|---:|---:|---:|---:|---:|---:|
| 128 | 4608 | 5.1 | 297 | 48% | 2.1x |
| 1024 | 4608 | 6.5 | 1861 | 43% | 1.5x |
| 8192 | 4608 | 28.6 | 3383 | 52% | 1.25x |
| 32768 | 4608 | 103.8 | 3728 | 57% | 1.25x |
| 1024 | 2304 | 4.9 | 1245 | 51% | 2.0x |
| 65536 | 2304 | 114.8 | 3373 | 51% | 1.5x |
| 262144 | 2304 | 448.9 | 3450 | 51% | 1.5x |

Still about half of copy bandwidth at large M, so the remaining limit isn't
idle threads or the second read. Next suspect: 16 IEEE divisions per block.

## v2: one reciprocal per block instead of 16 divisions

An IEEE fp32 division is a multi-instruction sequence. At 16 per block the
divisions cost more time than moving the data, so the kernel was limited by
arithmetic, not memory. Now each block computes `1 / denom` once and
multiplies. The reference does the same, so the two stay bit-exact.
Reconstruction error is unchanged (relative MSE 0.00907 on Gaussian rows,
before and after).

| M | K | us | GB/s | % of copy | vs v1 |
|---:|---:|---:|---:|---:|---:|
| 128 | 4608 | 4.9 | 307 | 49% | 1.04x |
| 1024 | 4608 | 4.9 | 2446 | 56% | 1.3x |
| 8192 | 4608 | 20.4 | 4736 | 73% | 1.4x |
| 32768 | 4608 | 72.1 | 5366 | 82% | 1.44x |
| 1024 | 2304 | 5.1 | 1181 | 59% | 0.96x |
| 65536 | 2304 | 77.9 | 4969 | 76% | 1.47x |
| 262144 | 2304 | 301.6 | 5135 | 76% | 1.49x |

### Tried and reverted: block max on packed bf16 pairs

Nsight Compute on v2 (32768 x 4608): DRAM 64% busy, SM compute 68%, no
register spills, 85% achieved occupancy, and the FP4 conversion is the
hardware `F2FP.SATFINITE.E2M1` instruction. So it's limited by instruction
count about as much as by memory.

Computing |x| and the block max on bf16 pairs (`__habs2`, `__hmax2`) instead
of converting all 16 values to fp32 first is exact and removes instructions,
but measured no faster (81% vs 82% of copy at 32768 x 4608). Those
instructions weren't on the critical path, so the change was reverted.

## v2 re-measured with CUDA graphs

The tables above time 50 back-to-back launches with events. A launch costs
the CPU about 5 us, longer than the kernel itself at small M, so those rows
partly measured launch overhead, and the copy baseline swung 2x between
runs (620 vs 310 GB/s at M = 128). The benchmark now captures the 50 calls
into a CUDA graph and replays it, so the GPU runs them back to back. Two
runs below agree to within 3%.

| M | K | us | GB/s | % of copy |
|---:|---:|---:|---:|---:|
| 128 | 4608 | 2.1 | 714 | 70% |
| 1024 | 4608 | 4.3 | 2832 | 76% |
| 8192 | 4608 | 19.9 | 4855 | 77% |
| 32768 | 4608 | 71.3 | 5430 | 83% |
| 1024 | 2304 | 3.6 | 1698 | 75% |
| 65536 | 2304 | 77.0 | 5030 | 77% |
| 262144 | 2304 | 298.8 | 5183 | 77% |

From here on, compare against this table.

### Tried and reverted: two 16-value blocks per thread

Halving the threads per row, so each thread pays the row reduction once per
32 values instead of 16, made every large shape slower: 60-67% of copy
against 77-83%. Fewer warps per SM means less memory latency hidden; the
reduction overhead it saved was not the limit.

## Where v2 stands

77-83% of copy bandwidth at training sizes (5.0-5.4 TB/s), 70-76% at small
M. The remaining gap is likely structural: each thread block must finish all
its loads before the row max exists, so its stores wait behind a barrier,
while a copy streams loads and stores continuously. The next step is
asynchronous bulk loads (TMA) that keep memory busy across that barrier.

## v2 re-measured: every buffer replayed, median of 5

The graph above replayed only the first 50 calls. At M = 128 that is
50 x 1.2 MB of inputs, small enough to stay in L2 between the warm-up and
the timed replay, so that row read partly from cache. The benchmark now
replays every rotation buffer (>= 512 MB per replay) and reports the
median of 5 replays; "spread" is slowest / fastest - 1.

| M | K | us | GB/s | copy GB/s | % of copy | spread |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4608 | 2.30 | 657 | 801 | 82% | 0.8% |
| 1024 | 4608 | 4.27 | 2833 | 3719 | 76% | 1.5% |
| 8192 | 4608 | 19.92 | 4857 | 6249 | 78% | 0.4% |
| 32768 | 4608 | 71.45 | 5417 | 6541 | 83% | 0.1% |
| 1024 | 2304 | 3.47 | 1746 | 2292 | 76% | 1.1% |
| 65536 | 2304 | 77.07 | 5024 | 6539 | 77% | 0.1% |
| 262144 | 2304 | 299.36 | 5174 | 6741 | 77% | 0.8% |

Only M = 128 moved: 2.1 -> 2.3 us, and its copy baseline fell more (about
1020 -> 801 GB/s), which is why its "% of copy" rose. Every other shape
already rotated through more than L2 holds. This is the table to beat.

### Tried and reverted: one 32-byte load per thread

Each thread read its 32 bytes with two 16-byte loads, so across a warp each
load instruction used half of every 32-byte sector it touched (ncu: "16.0
of 32 bytes per sector", 2x the L1 sectors of L2). sm_100 has 256-bit
loads (`ld.global.nc.v8.b32`, SASS `LDG.E.ENL2.256.CONSTANT`), so one
instruction per thread reads 1 KB contiguous per warp.

| M | K | us | % of copy |
|---:|---:|---:|---:|
| 128 | 4608 | 2.33 | 81% |
| 1024 | 4608 | 4.32 | 76% |
| 8192 | 4608 | 19.88 | 78% |
| 32768 | 4608 | 71.32 | 83% |
| 1024 | 2304 | 3.47 | 76% |
| 65536 | 2304 | 77.07 | 77% |
| 262144 | 2304 | 301.54 | 76% |

Identical to v2. DRAM already read each byte once; the extra L1 sector
traffic wasn't on the critical path. Reverted to keep plain CUDA.

### Tried and reverted: two threads per 16-value block

Each thread loads 8 values (16 bytes, contiguous across the warp), and the
pair joins its block max with one `__shfl_xor_sync(..., 1)`.

| M | K | us | % of copy |
|---:|---:|---:|---:|
| 128 | 4608 | 2.46 | 78% |
| 1024 | 4608 | 6.51 | 50% |
| 8192 | 4608 | 37.53 | 41% |
| 32768 | 4608 | 142.56 | 41% |
| 1024 | 2304 | 3.91 | 67% |
| 65536 | 2304 | 121.56 | 49% |
| 262144 | 2304 | 477.38 | 48% |

About 2x slower. A row at K = 4608 now needs 576 threads (18 warps), so
only 3 rows fit on an SM under the 64-warp limit, against 7 before. Bytes
in flight per SM fell from 7 to 3 rows' worth, and bandwidth fell with
them (Little's law: bandwidth = bytes in flight / latency). What limits
this kernel is how many bytes each SM has requested and not yet received.

## v2.1: streaming loads and stores

`__ldcs` / `__stcs` mark every load and store evict-first: each byte is
touched once, so there's nothing to keep in cache.

| M | K | us | GB/s | % of copy | vs v2 |
|---:|---:|---:|---:|---:|---:|
| 128 | 4608 | 2.32 | 652 | 82% | 1.0x |
| 1024 | 4608 | 4.26 | 2840 | 77% | 1.0x |
| 8192 | 4608 | 19.57 | 4944 | 79% | 1.02x |
| 32768 | 4608 | 71.24 | 5433 | 83% | 1.0x |
| 1024 | 2304 | 3.28 | 1844 | 81% | 1.06x |
| 65536 | 2304 | 76.99 | 5029 | 77% | 1.0x |
| 262144 | 2304 | 300.13 | 5160 | 77% | 1.0x |

A small, real gain where the working set is mid-sized (spread is about 1%),
and nothing slower. Kept.

## The ceiling for this traffic

`traffic_ceiling` reads and writes exactly what the quantizer does (same
layout, same streaming hints) and computes nothing. "% of ceiling" is the
quantizer's bandwidth over that kernel's. v2.1:

| M | K | us | % of copy | % of ceiling |
|---:|---:|---:|---:|---:|
| 128 | 4608 | 2.32 | 81% | 83% |
| 1024 | 4608 | 4.48 | 73% | 77% |
| 8192 | 4608 | 19.62 | 79% | 68% |
| 32768 | 4608 | 71.17 | 83% | 83% |
| 1024 | 2304 | 3.29 | 81% | 83% |
| 65536 | 2304 | 77.00 | 77% | 77% |
| 262144 | 2304 | 301.53 | 76% | 76% |

The ceiling runs at about copy speed, so the mix of 2 bytes read to 0.56
written isn't what holds the quantizer back: there is ~1.2-1.3x of real
headroom at training sizes. (M = 1024 x 4608 measured 4.26-4.48 us across
separate runs; between-container variation at small shapes is larger than
the within-run spread column.)

### Tried and reverted: persistent blocks

The hypothesis: each thread block loads its row, then waits through two
barriers and the math before memory gets more work, so persistent blocks
that fetch the next row early should keep HBM busier.

| variant | 128x4608 | 1024x4608 | 8192x4608 | 32768x4608 | 1024x2304 | 65536x2304 | 262144x2304 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2.1 (reference) | 2.32 | 4.26 | 19.57 | 71.24 | 3.28 | 76.99 | 300.13 |
| next row prefetched into registers | 2.35 | 4.40 | 20.55 | 73.89 | 3.32 | 82.51 | 320.67 |
| TMA ring of 3 rows in shared memory | 2.46 | 4.89 | 21.14 | 73.86 | 3.71 | 83.65 | 323.82 |
| same, 32 registers, max shared carveout | 2.47 | 4.81 | 20.53 | 72.21 | 3.71 | 79.42 | 307.78 |
| v2.1 with a one-barrier block max | 2.25 | 4.26 | 19.95 | 72.48 | 3.23 | 78.78 | 307.29 |

(microseconds; lower is better)

- **Register prefetch**: 48 registers per thread instead of 32, so 4 blocks
  fit per SM instead of 7, and fewer rows are in flight than before.
- **TMA ring** (`cp.async.bulk` into 3 shared-memory row buffers, one
  mbarrier each; code in `experiments/nvfp4_tma_pipelined.cu`): load stalls
  nearly vanished (long-scoreboard 27% -> 4% of samples), so the pipeline
  works, but 37 registers and a 168 KB carveout left 5 blocks per SM.
  Capping registers at 32 (`__launch_bounds__(1024, 2)`) and asking for the
  full carveout restores 7: then it ties v2.1 at the largest shape and loses
  elsewhere (the single-thread bulk issue and mbarrier waits add latency
  at small M).
- **One barrier per block max** (every warp finishes the reduction itself):
  2-3% faster at small M, 1-2% slower at large M, where the extra
  instructions cost more than the barrier.

Why none of it wins: at 32768 x 4608 v2 issues 56.85 M warp-instructions,
which is 96 K per scheduler, about 50 us at one per cycle. The kernel takes
71 us, and its traffic at the ceiling would take about 59 us. Overlapping
memory doesn't help once instruction issue alone is most of the runtime.
The lever now is fewer instructions per value, not more bytes in flight.

## v2.2: fewer instructions, at 32 registers

Since instruction issue is now most of the runtime, cut instructions. The
FP4 packing was the most wasteful sequence: each converted byte came back
to C++ as a `uint8` and was masked, shifted and OR-ed into place (~15
integer instructions per block). Writing the conversions in PTX and
assembling the word with `mov.b32 {b0, b1, b2, b3}` lets ptxas chain them
through `F2FP`'s merge operand. Executed instructions at 32768 x 4608:
57.74 M -> 54.49 M (-5.6%).

That alone was slower (83.7 us at 32768 x 4608), because registers went
from 32 to 37: at 288 threads per block, 7 blocks fit per SM with 32
registers and 5 with 37. A `__byte_perm` tree in C++ hit the same cliff
(37 registers, 86.9 us) while saving only one instruction. With
`__launch_bounds__(1024, 2)` the one-item kernel is held to 32 registers:

| M | K | us | GB/s | % of copy | % of ceiling | vs v2.1 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4608 | 2.33 | 649 | 81% | 83% | 1.0x |
| 1024 | 4608 | 4.36 | 2772 | 75% | 79% | 1.0x |
| 8192 | 4608 | 19.20 | 5040 | 81% | 69% | 1.02x |
| 32768 | 4608 | 69.53 | 5566 | 85% | 85% | 1.02x |
| 1024 | 2304 | 3.25 | 1859 | 81% | 84% | 1.0x |
| 65536 | 2304 | 74.96 | 5165 | 79% | 79% | 1.03x |
| 262144 | 2304 | 302.18 | 5125 | 76% | 76% | 1.0x |

The cap costs a 24-byte stack frame (cuobjdump: STACK:24) but no measured
time. 262144 x 2304 had a noisy run (6.7% spread); its median is unchanged.
