"""Bandwidth of the NVFP4 quantize kernel on the GPU.

Quantizing is memory-bound: per value it reads 2 bytes of bf16 and writes
0.5 bytes of FP4 plus 1/16 of an E4M3 scale byte. So the number that
matters is bytes moved per second, compared with how fast this GPU can
copy memory at all (a bf16 clone of the same size, measured the same way).

Inputs rotate through at least 512 MB of distinct buffers, so the 126 MB L2
cache can't serve repeat reads and every call really goes to HBM.

    modal run scripts/modal_b200.py --cmd "python bench/nvfp4_quantize.py"
"""

from __future__ import annotations

import torch

from quokka.cuda import nvfp4_quantize

# Activation shapes from a public 4-bit MoE model's config (hidden 4608,
# expert FFN 2304, 8 experts per token). M is tokens on one GPU, which the
# model's authors don't publish, so it sweeps from RL-decode sizes (128) to
# large pretraining microbatches.
#   K = 4608: tokens entering the experts (dispatch, first expert GEMM)
#   K = 2304: expert hidden states entering the second expert GEMM; each
#             token visits 8 experts, so M is 8x the token count
SHAPES = [
    (128, 4608),
    (1024, 4608),
    (8192, 4608),
    (32768, 4608),
    (1024, 2304),
    (65536, 2304),
    (262144, 2304),
]
ROTATE_BYTES = 512 << 20


def _time_ms(fn, inputs, iters=50):
    for x in inputs[:3]:
        fn(x)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        fn(inputs[i % len(inputs)])
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    rows = []
    for m, k in SHAPES:
        size = m * k * 2
        n = max(2, -(-ROTATE_BYTES // size))
        inputs = [torch.randn(m, k, device="cuda", dtype=torch.bfloat16) for _ in range(n)]
        moved = m * k * (2 + 0.5 + 1 / 16) + m * 4  # read bf16, write fp4 + scales + outer
        t_q = _time_ms(nvfp4_quantize, inputs)
        t_copy = _time_ms(torch.clone, inputs)
        copy_bw = 2 * size / t_copy / 1e6  # clone reads and writes
        q_bw = moved / t_q / 1e6
        rows.append((m, k, t_q * 1e3, q_bw, copy_bw, q_bw / copy_bw))
        del inputs

    print("\n| M | K | quantize (us) | GB/s | copy GB/s | % of copy |")
    print("|---:|---:|---:|---:|---:|---:|")
    for m, k, us, bw, cbw, frac in rows:
        print(f"| {m} | {k} | {us:.1f} | {bw:.0f} | {cbw:.0f} | {100 * frac:.0f}% |")


if __name__ == "__main__":
    main()
