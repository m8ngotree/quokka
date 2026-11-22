"""Run one command on a Modal B200, then shut it down.

    modal run scripts/modal_b200.py --cmd "nvidia-smi"
    modal run scripts/modal_b200.py --cmd "pytest -q tests"

Modal bills GPUs per second, so the setup avoids paying for idle time:

- `modal run` starts an ephemeral app. When the command returns, the app
  stops and the container is torn down. Nothing stays up between runs.
- Everything slow to install (CUDA toolkit, PyTorch, CUTLASS) is baked into
  the image. Modal builds images on CPU workers and caches them, so no GPU
  time goes to installing anything. The image rebuilds only when this file's
  image definition changes.
- The repo is mounted at container start rather than copied into the image,
  so editing code never triggers a rebuild.
- The container scales down 2 s after going idle, `max_containers=1` stops
  accidental fan-out, and a 30-minute timeout caps a runaway job.

The GPU seconds actually used are printed at the end of each run.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
REMOTE = "/root/quokka"

CUDA_IMAGE = (
    "nvidia/cuda:13.5.0-devel-ubuntu24.04"  # nvcc for sm_100a; same major as torch's CUDA 13.2
)
CUTLASS_TAG = "v4.8.1"

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("git")
    .uv_pip_install("torch==2.15.0", "numpy>=2.0", "ml_dtypes>=0.5", "pytest>=8.0", "ninja")
    .run_commands(
        f"git clone --depth 1 --branch {CUTLASS_TAG} https://github.com/NVIDIA/cutlass /opt/cutlass"
    )
    .env({"CUTLASS_DIR": "/opt/cutlass", "TORCH_CUDA_ARCH_LIST": "10.0a", "PYTHONPATH": REMOTE})
    .add_local_dir(
        ROOT,
        REMOTE,
        ignore=[".venv", ".git", "**/__pycache__", ".pytest_cache", ".ruff_cache", "results"],
    )
)

app = modal.App("quokka-b200", image=image)


@app.function(gpu="B200", timeout=30 * 60, scaledown_window=2, max_containers=1)
def run(cmd: str) -> int:
    start = time.monotonic()
    code = subprocess.run(cmd, shell=True, cwd=REMOTE).returncode
    print(f"[quokka] exit {code}, {time.monotonic() - start:.1f} s of GPU time in the command")
    return code


@app.local_entrypoint()
def main(cmd: str = "nvidia-smi"):
    code = run.remote(cmd)
    if code:  # exit only on failure: SystemExit(0) makes Modal log a spurious exception
        sys.exit(code)
