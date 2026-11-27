"""CUDA kernels, built as one PyTorch extension.

`ext()` returns the compiled module. If QUOKKA_PREBUILT names a built .so
(the Modal runner compiles on a CPU container and passes it in), it is
imported directly with no compiler involved. Otherwise the extension is
compiled on first use with torch's JIT builder.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
import pathlib

import torch

CSRC = pathlib.Path(__file__).parent / "csrc"
NAME = "quokka_cuda"
SOURCES = sorted(str(p) for p in CSRC.glob("*.c*"))
# torch adds -std=c++20 itself (its headers require it), so no -std here.
CUDA_FLAGS = ["-O3", "--expt-relaxed-constexpr"]


def _content_hash() -> str:
    """Hash of everything that affects the build: sources, flags, torch version."""
    h = hashlib.sha256()
    for path in SOURCES:
        h.update(pathlib.Path(path).name.encode())
        h.update(pathlib.Path(path).read_bytes())
    h.update(repr((CUDA_FLAGS, torch.__version__, os.environ.get("TORCH_CUDA_ARCH_LIST"))).encode())
    return h.hexdigest()[:16]


def build(build_dir: str | None = None):
    """Compile (or reuse) the extension. Needs nvcc, not a GPU.

    A persistent build_dir is keyed by a hash of the sources. ninja decides
    what to rebuild from file timestamps, and files mounted into a container
    don't reliably carry them, so an edited kernel could otherwise be skipped
    and a stale .so loaded. With the hash, any edit builds in a fresh
    directory; unchanged sources reuse theirs.
    """
    from torch.utils.cpp_extension import load

    if build_dir:
        build_dir = os.path.join(build_dir, _content_hash())
        os.makedirs(build_dir, exist_ok=True)
    return load(
        name=NAME,
        sources=SOURCES,
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=["-O3"],
        build_directory=build_dir,
        verbose=False,
    )


@functools.cache
def ext():
    prebuilt = os.environ.get("QUOKKA_PREBUILT")
    if prebuilt:
        spec = importlib.util.spec_from_file_location(NAME, prebuilt)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return build(os.environ.get("QUOKKA_BUILD_DIR"))


_E2M1_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32
)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] -> float32 [..., K]; low nibble first."""
    lut = _E2M1_VALUES.to(packed.device)
    lo, hi = lut[(packed & 0xF).long()], lut[(packed >> 4).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def nvfp4_quantize(x: torch.Tensor):
    """bf16 [M, K] -> (packed uint8 [M, K/2], e4m3 scales [M, K/16], fp32 outer [M, 1])."""
    return ext().nvfp4_quantize(x)


def traffic_ceiling(x: torch.Tensor):
    """The NVFP4 quantizer's reads and writes with no math: a roofline for benchmarks."""
    return ext().traffic_ceiling(x)
