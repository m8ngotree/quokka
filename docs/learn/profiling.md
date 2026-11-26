# Reading the machine: PTX, SASS and Nsight Compute

A worked guide using the real NVFP4 quantize kernel
(`quokka/cuda/csrc/nvfp4_quantize.cu`, v2.2). Every listing and number here
comes from the files in `artifacts/`, produced on a Modal B200 with CUDA
13.5 and Nsight Compute 2026.4:

- `artifacts/nvfp4_v2.2/`: the current kernel.
- `artifacts/nvfp4_v2/`: the version before streaming hints, PTX packing
  and the register cap. Diff the two `kernel_items1.sass` files to see
  exactly what each change did to the machine code.

## 1. What the compiler turns your code into

```
nvfp4_quantize.cu ──nvcc (front end)──▶ PTX ──ptxas──▶ SASS ──▶ cubin (ELF) ──▶ fatbin inside quokka_cuda.so
```

- **PTX** is NVIDIA's *virtual* instruction set: documented, stable across
  GPUs, with unlimited virtual registers (`%r12`, `%rd3`). It is what you
  *asked for*. Reference: the PTX ISA manual
  (docs.nvidia.com/cuda/parallel-thread-execution).
- **SASS** is the *real* machine code for one GPU family (`sm_100a` here).
  `ptxas` does register allocation, scheduling and most of the optimization,
  so SASS is what you *got*. NVIDIA only publishes a list of opcode names
  (docs.nvidia.com/cuda/cuda-binary-utilities, "Instruction Set Reference").
  The rest you learn by reading compiler output and running experiments.
- The **cubin** is the ELF file that holds SASS; a **fatbin** bundles cubins
  (and optionally PTX) for several architectures. Our `.so` carries only
  `sm_100a` SASS, which is why PTX is produced by a separate `nvcc -ptx`.

Rule of thumb: check PTX to see whether the compiler understood your intent
(did I get a vector load? the cache hint I asked for?). Check SASS to see
the cost: how many instructions, which slow paths, where registers went.

You can also *write* PTX: `asm("...")` inside CUDA C++. v2.2 does this once,
for the FP4 packing (§3), because C++ couldn't express it cheaply.

### Getting them

```bash
uv run modal run scripts/modal_b200.py --cpu --cmd "python scripts/dump_asm.py"
```

```bash
uv run modal volume get quokka-build asm ./asm-dump
```

That runs on a CPU container (no GPU billed) and writes `quokka_cuda.sass`
(`cuobjdump -sass`), `nvfp4_quantize.ptx` and `resources.txt`
(`cuobjdump -res-usage`: registers, stack and shared memory per kernel).
`artifacts/*/kernel_items1.{sass,ptx}` hold the `ITEMS=1` instantiation,
the one every benchmark shape uses.

## 2. How to read a SASS line

```
/*0170*/ LDG.E.EF.128 R4, desc[UR8][R12.64] ;
/*0210*/ FMNMX3 R17, |R2|, RZ, |R0|, !PT ;
/*0bb0*/ F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C R6, R5, R6, R7 ;
/*0830*/ @!P0 STG.E desc[UR8][R10.64], R22 ;
```

- `/*0170*/` is the byte offset. Every instruction is 16 bytes on this arch.
- Destination comes **first**, then sources.
- `R0..R254` are per-thread registers; `RZ` reads as zero. `UR*` are
  *uniform* registers, one value shared by the whole warp (addresses,
  loop bounds), computed on a separate uniform datapath.
- `P0..P6` are predicate (boolean) registers; `PT` is always true.
  `@!P0 X` means "do X only in threads where P0 is false". That is how short
  `if`s compile, without a branch.
- `c[0x0][0x380]` is constant bank 0, where **kernel arguments** live. Here
  `0x380` is `x`, `0x388` is `packed`, `0x390` is `scales`, `0x398` is
  `outer_out` and `0x3a0` is `K`.
- Suffixes carry the meaning: `.E` means 64-bit address, `.128` means a
  16-byte access, `.EF` means evict-first (our streaming hint), `.reuse` is
  an operand-cache hint, and `|R2|` is a free absolute value.
- `R1` is the stack pointer. `VIADD R1, R1, 0xffffffe8` at the top makes
  a 24-byte frame, and `STL`/`LDL` read and write it: **spills**.

## 3. Walking through the v2.2 kernel's SASS

`artifacts/nvfp4_v2.2/kernel_items1.sass`. The main path is ~185
instructions per warp (54.49 M warp-instructions / 294,912 warps); v2 was
~193.

**Address math (0x0000–0x0160).** `S2R R14, SR_TID.X` reads threadIdx,
`S2R R4, SR_CTAID.X` reads blockIdx. `K / 16` is done with shifts
(`USHF`, `ULEA.HI`) on uniform registers, because K is the same for every
thread. Then `row * K * 2 + tid * 32` builds the address.

**Loads (0x0170–0x0180).** Two `LDG.E.EF.128`, 16 bytes each, evict-first
because the source uses `__ldcs`. In v2 these were `LDG.E.128.CONSTANT`
(the read-only path you get from `const __restrict__`). The hint replaced
the path, so check the SASS when you add hints. Across a warp, each of
these instructions uses half of every 32-byte sector it touches (§5); a
256-bit load (`LDG.E.ENL2.256`) fixes that but measured no faster.

**bf16 → fp32 (0x0190–0x0380).** There is no conversion instruction. A
bf16 is the top 16 bits of an fp32, so `IMAD.U32 R2, R4, 0x10000, RZ`
(multiply by 2^16, i.e. shift left 16) converts the low half of a 32-bit
pair. `PRMT R0, R4, 0x7632, R0` (byte permute) pulls out the high half
first. That's 2 instructions per value, 32 per block.

**Block max.** `FMNMX3 R17, |R2|, RZ, |R0|` is a *three-input* max with
free absolute values, new on Blackwell, so `max(|v0|..|v15|)` takes 8
instructions, not 31.

**Warp and block reduction (0x03c0–0x0720).** `SHFL.BFLY` with
xor-offsets 16, 8, 4, 2, 1 is the butterfly from `block_max()`. Then:
- Lane 0 does `STS` to shared memory, then `BAR.SYNC` (`__syncthreads()`).
- Warp 0 does `LDS` and another butterfly, then a second `BAR.SYNC`.
- `BRA.DIV` jumps to a fallback copy at 0x1eb0 if the warp is
  unexpectedly divergent. That copy never runs here.

**The IEEE divisions (0x0730–0x0940).** `amax / 2688` and
`bmax / (6 * outer)` each compile to:
- `MUFU.RCP`: a fast approximate reciprocal from the special-function unit.
- Two `FFMA` Newton steps.
- `FCHK` + `CALL.REL.NOINC`: FCHK checks whether the fast result might be
  wrong (denormals, huge exponents) and, if so, calls a slow exact
  subroutine.

This is the ~10-instruction sequence v1 paid 16 times per block; v2 pays it
twice.

**Scale conversion (0x0950–0x09a0).**
`F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` is fp32 → E4M3, round to nearest
even, saturating. That's the `__nv_fp8_e4m3(...)` constructor in one
instruction. Converting back for `denom` goes E4M3 → f16
(`F2FP.F16.E4M3.UNPACK_B`) → f32 (`HADD2.F32`). Note `STL [R1+0x14], R11`
here: the scale byte is **spilled** to the stack, and reloaded with
`LDL.LU R0, [R1+0x14]` just before it's stored (0x1dc0). That's the price
of the 32-register cap: one 4-byte store and load per thread, which costs
nothing measurable.

**Reciprocal and the overflow check (0x09f0–0x0af0).** `1 / denom`: an
exponent range check, `MUFU.RCP` + one refinement, a slow call only when
out of range. Then `FSETP.NEU.AND P0, PT, |R11|, +INF` is the `isinf(inv)`
test from the tiny-row fix. The branch to the division path (0x0c90
onward, where most of the other `STL`/`LDL` spills live) is never taken
on real data.

**Quantize and pack (0x0b00–0x0c70).** 16 `FMUL` (x · inv) and 8
`F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`. Look at the last operand: in v2
it was always `RZ`. Now it's the previous conversion's result
(`... R6, R5, R6, R7`), so each conversion **merges** its byte into the
word being built, and no masks, shifts or ORs are left (v2 had ~15). This
came from writing the conversion in PTX (`cvt.rn.satfinite.e2m1x2.f32`
into `.b8` registers, then `mov.b32 %0, {b0, b1, b2, b3}`) instead of C++.

**Stores (0x1dd0–0x1e90).** The compiler re-reads blockIdx
(`S2R R2, SR_CTAID.X`) rather than keep it in a register: that's
*rematerialization*, another way of living within 32 registers. Then
`STG.E.EF.64` (8 bytes of FP4) and `STG.E.EF.U8` (1 byte of scale). Across
a warp these are 256 and 32 contiguous bytes, fully coalesced.

### `resources.txt`, and the register cliff

```
v2:   kernel<1>: REG:32 STACK:0     kernel<8>: REG:32 STACK:480
v2.2: kernel<1>: REG:32 STACK:24    kernel<8>: REG:64 STACK:368
```

Registers decide how many thread blocks fit on an SM (65,536 registers
each). At K = 4608 a block has 288 threads:

| registers / thread | blocks per SM | warps per SM |
|---:|---:|---:|
| 32 | 7 | 63 |
| 33–40 | 5 | 45 |
| 48 | 4 | 36 |

The PTX packing alone used 37 registers and was 17% *slower*, despite
5.6% fewer instructions. `__launch_bounds__(1024, 2)` on the one-item
kernel asks ptxas for at most 65536 / 2048 = 32 registers, and it spills a
little to get there. Always check `REG` after a change, and know where
your cliffs are.

`ITEMS=8` still has a large stack (128 floats in `v[][]` can't fit in 64
registers). Only K > 16384 uses it, which no benchmark shape does.

## 4. Nsight Compute: how to profile

```bash
uv run modal run scripts/modal_b200.py --cmd "mkdir -p /build/profiles && ncu --clock-control none --set full --import-source yes --kernel-name regex:nvfp4 --launch-skip 2 --launch-count 1 -f -o /build/profiles/run python bench/profile_nvfp4.py"
```

- `--set full` collects every section. ncu replays the kernel about 40
  times to read all the counters, so profiled durations are not benchmark
  numbers; the benchmark is.
- `--clock-control none`: Modal doesn't let ncu lock clocks, so expect a few
  percent of noise between profiles.
- `--launch-skip 2 --launch-count 1` skips warm-up launches.
- `--import-source yes` + `-lineinfo` (in `CUDA_FLAGS`) embeds the source so
  the GUI can show per-line costs.
- `bench/profile_nvfp4.py --m 32768 --k 4608` picks the shape.
- For one or two numbers, skip `--set full` and ask for metrics directly.
  That's how each experiment in the journal was checked:
  `--metrics smsp__inst_executed.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active`.

Then read it either way:

- **GUI (best):** download the report
  (`uv run modal volume get quokka-build profiles/run.ncu-repz .`) and open it
  in the Nsight Compute desktop app (it has a macOS host build; it must be
  2026.4 or newer to open these reports). The **Source** page with "Source
  and SASS" side by side is where most of the learning happens. Open the v2
  and v2.2 reports together and use **baselines** to compare them.
- **Text:** `ncu -i run.ncu-repz --page details` (all sections), `--page source
  --csv --print-source sass` (per-instruction stall samples), `--page raw
  --csv` (every raw metric). `ncu` needs a Linux container for these; add
  `--cpu` to the Modal command to run them without a GPU.

## 5. Reading the profile, v2 → v2.2

`artifacts/nvfp4_v2.2/nvfp4_v2.2_32768x4608.details.txt` against
`artifacts/nvfp4_v2/...` (32768 × 4608). Read in this order: it's the
method, not just this kernel.

### Step 1: Speed of Light, "is anything maxed out?"

| Metric | v2 | v2.2 |
|---|---:|---:|
| DRAM throughput | 64.8% | 64.9% |
| Compute (SM) throughput | 68.0% | 66.9% |
| Issue slots busy | 68.0% | 66.9% |

When **nothing** is near 80–90%, the kernel is **latency-bound**: warps
spend their time waiting, and there aren't enough independent ones to
cover the waits. (If DRAM were at 90%, you'd be done. If issue slots were
at 90%, you'd cut instructions.) ncu's headline here is *"Compute and
Memory are well-balanced"*. That turned out to be the key fact (§6).

Profiled DRAM % understates the benchmark because ncu's replays and
unlocked clocks change timing. Use it for *what limits*, and the benchmark
for *how fast*.

### Step 2: Warp State, "what are warps waiting on?"

ncu samples each warp's state periodically (`smsp__pcsamp_warps_issue_stalled_*`
in the raw page):

| Stall reason | v2 | v2.2 | Meaning |
|---|---:|---:|---|
| long_scoreboard | 27% | 28% | waiting on a global-memory load result |
| barrier | 18% | 19% | waiting at `__syncthreads()` for the rest of the block |
| not_selected | 17% | 14% | ready, but another warp issued: *healthy* |
| wait | 11% | 12% | fixed-latency dependency (e.g. FFMA → FFMA) |
| short_scoreboard | 7% | 9% | waiting on shared memory, SHFL or MUFU |
| math_pipe_throttle | 7% | 5% | a pipe (here the integer ALU) is busy |
| selected | 6% | 6% | actually issuing |

The integer pipe throttles less in v2.2: the packing masks are gone.

### Step 3: Source, "where exactly?"

The hottest SASS lines in v2.2 by stall samples (`source_sass.csv`):

- **30.8%**: `PRMT R0, R4, 0x7632, R0`. That's the *first use of the loaded
  register* R4. A stall is charged to the instruction that waits, not to
  the load, so this is the load latency.
- **10.3%** at the `BRA` right after the first `BAR.SYNC`, and **9.5%** at
  the `UMOV` after the second. Those are the two barriers in `block_max`.

### Step 4: Memory, "is traffic wasted?"

| Counter | v2.2 | Read it as |
|---|---|---|
| `dram__bytes_read.sum` | 302.0 MB | exactly 32768 × 4608 × 2 bytes: no wasted reads |
| `lts__t_sectors_srcunit_tex_op_read.sum` | 9.44 M sectors | ×32 B = 302 MB: L2 also sees each byte once |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 18.9 M sectors | **2×** L2: each 16-byte load uses half of each sector |
| `l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum` | 1.18 M sectors | the one spill: 4 bytes × 32 lanes per warp |

`dram__bytes_write.sum` (58.7 MB) is less than the 85 MB written, because
dirty lines still in L2 when the kernel ends are written back later.

ncu flags "only 16.0 of the 32 bytes transmitted per sector are utilized"
with an "est. speedup 21.99%". The 256-bit-load experiment fixed exactly
that and gained nothing: DRAM and L2 never saw the waste. **ncu's
estimated speedups are rule-of-thumb upper bounds, not predictions.**

### Advice to ignore, and why

- *"L2 Compression… est. speedup 45%"*: compressible memory pays off for
  data full of zeros or repeats. Random activations aren't.
- *"Convert non-fused FP32 instructions to fused (FMA)"*: there's nothing
  to fuse `x * inv` with, and fusing rounding steps elsewhere would **break
  bit-exactness** with the reference. Not every speedup is allowed.
- *"Achieved occupancy 85.8% vs theoretical 98.4%"*: that's the tail of
  31.6 waves of blocks plus per-block imbalance. More occupancy wouldn't
  remove the barriers.

## 6. The calculation that explains this kernel

From the v2 profile: 56.85 M warp-instructions, 148 SMs × 4 schedulers,
each issuing at most one instruction per cycle:

```
56.85e6 / (148 × 4) = 96,000 cycles per scheduler  →  at 1.9 GHz ≈ 50 µs
```

The kernel took 71 µs, and its traffic at the measured ceiling would take
~59 µs. So **issuing instructions alone took most of the runtime**, which
is why the memory-side experiments (persistent blocks, a TMA pipeline,
coalesced loads) couldn't win, and why cutting instructions did (until
registers rose). Do this arithmetic on every kernel: it takes a minute and
it tells you which half of the machine to work on.

## 7. Exercises, in order

1. Open `artifacts/nvfp4_v2.2/nvfp4_v2.2_32768x4608.ncu-repz` in the Nsight
   Compute GUI. Find each number in §5 yourself, on the Source page too.
   Then add the v2 report as a baseline and find the differences.
2. Diff `artifacts/nvfp4_v2/kernel_items1.sass` against
   `artifacts/nvfp4_v2.2/kernel_items1.sass`. Find: the load hint change, the
   `MERGE_C` chain, the `isinf` test, the stack frame, and every
   `STL`/`LDL`. Which spills are on the common path?
3. In `artifacts/nvfp4_v2.2/kernel_items1.ptx`, find the
   `cvt.rn.satfinite.e2m1x2.f32` / `mov.b32` block, and the `.loc` line that
   maps it to the source. Then find how ptxas turned it into SASS.
4. Count the instructions between the second `BAR.SYNC` and `EXIT` on the
   common path. Pick the biggest group that remains (the divisions? the
   bf16 unpack?). Predict how many you could remove and whether the time
   would move, given §6. Then try it, keeping REG ≤ 32.
5. Make `ITEMS=8`'s stack go away without growing the others. (Hint:
   launch bounds can differ per template instantiation, and so can the
   number of values a thread holds at once.)
6. The one-barrier `block_max` was 2–3% faster at small M and 1–2% slower
   at large M (results file). Can you get both? For instance, choose by M
   at launch time, or reduce with fewer instructions per warp.
