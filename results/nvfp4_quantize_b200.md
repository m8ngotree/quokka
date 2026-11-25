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
