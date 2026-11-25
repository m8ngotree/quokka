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
