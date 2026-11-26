"""Dump the PTX and SASS of the CUDA extension, for reading. Needs nvcc, no GPU.

    modal run scripts/modal_b200.py --cpu --cmd "python scripts/dump_asm.py"

Writes, under /build/asm (the Modal volume) or the --out directory:

- <kernel>.ptx: nvcc's virtual-ISA output, before ptxas optimizes it.
- quokka_cuda.sass: the real machine code from the built .so (cuobjdump).
- resources.txt: registers, shared memory and spills per kernel.

The PTX is produced by compiling the source again with the same flags,
since the .so only carries SASS for sm_100a.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sysconfig

import torch.utils.cpp_extension as cpp

from quokka.cuda import CSRC, CUDA_FLAGS


def sh(cmd: list[str], out: pathlib.Path) -> None:
    out.write_text(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/build/asm")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    so = os.environ["QUOKKA_PREBUILT"]
    sh(["cuobjdump", "-sass", so], out / "quokka_cuda.sass")
    sh(["cuobjdump", "-res-usage", so], out / "resources.txt")

    includes = [f"-I{p}" for p in cpp.include_paths(device_type="cuda")]
    includes.append(f"-I{sysconfig.get_paths()['include']}")
    for cu in sorted(CSRC.glob("*.cu")):
        ptx = out / f"{cu.stem}.ptx"
        subprocess.run(
            ["nvcc", "-ptx", "-o", str(ptx), "-arch=sm_100a", "-std=c++20",
             "-DTORCH_EXTENSION_NAME=quokka_cuda", *CUDA_FLAGS, *includes, str(cu)],
            check=True,
        )  # fmt: skip
        print(f"wrote {ptx} ({ptx.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
