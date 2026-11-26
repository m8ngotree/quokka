"""A minimal program for Nsight Compute to profile: one shape, a few launches.

    ncu --set full -o profile python bench/profile_nvfp4.py

Launch 0 warms up; profile a later one with --launch-skip 2 --launch-count 1.
"""

import argparse

import torch

from quokka.cuda import nvfp4_quantize

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=32768)
ap.add_argument("--k", type=int, default=4608)
args = ap.parse_args()

x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
for _ in range(4):
    nvfp4_quantize(x)
torch.cuda.synchronize()
