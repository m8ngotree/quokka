# Learning from quokka

Notes for learning low-level GPU performance work from this repo: what was
done, why, and how to do it yourself.

| File | What's in it |
|---|---|
| [`nvfp4_quantize_journal.md`](nvfp4_quantize_journal.md) | Every optimization step on the NVFP4 quantizer, with the hypothesis, prediction and result, failures included. New work is appended here. |
| [`profiling.md`](profiling.md) | How to get and read PTX, SASS and Nsight Compute profiles, walked through on the real kernel (v2.2, with v2 alongside). Ends with exercises. |
| [`baselines.md`](baselines.md) | The public kernels to beat, how to compare fairly, and where the remaining speed is. |
| `artifacts/nvfp4_v2.2/`, `artifacts/nvfp4_v2/` | The raw material for both versions: SASS, PTX, register usage, and a full `ncu` report you can open in the Nsight Compute GUI. |

Numbers live in `results/`; reasoning lives here.

## How to contribute an optimization

The loop that every journal entry follows:

1. **Profile first** (`profiling.md` §4). Read Speed of Light, then Warp
   State, then Source. Name the bottleneck in one sentence.
2. **Write down a prediction**: which counter moves, by how much, and the
   new time. Put it in the journal *before* running.
3. **Make one change** on a branch. Keep it bit-exact:

   ```bash
   uv run modal run scripts/modal_b200.py --cmd "pytest -q tests/test_cuda_nvfp4.py && python bench/nvfp4_quantize.py"
   ```

   Check that the log says `compiled on CPU in ~70 s` after an edit. If it
   says ~1 s, you're running the old kernel. Also check `REG` in
   `resources.txt` (`profiling.md` §3): at 288 threads per block, going
   past 32 registers costs ~17% before your change does anything.
4. **Compare** against the latest table in `results/nvfp4_quantize_b200.md`,
   then profile again and check whether the counter you predicted moved.
   Differences under ~2% at large M (or ~5% at small M, between separate
   runs) are noise; the "spread" column shows the within-run part.
5. **Keep or revert**, and write the entry either way. A negative result
   with a reason is worth as much as a speedup.

Cost: a test plus benchmark run uses about 10 s of B200 time; a full
profile about 10 s; SASS dumps use no GPU (`--cpu`).

Good first experiments, roughly in order of difficulty:

- `profiling.md` exercises 1–3: read the profile and the SASS, and diff v2
  against v2.2 until every change in the machine code has a reason.
- Exercises 4–6: cut an instruction group within 32 registers, fix the
  `ITEMS=8` stack, and get the one-barrier reduction's small-M win without
  its large-M loss.
- PDL for small M.
- Four Over Six in the kernel (more math per byte, so the trade-offs shift).

Already tried, so check the journal before redoing: 32-byte loads, two
threads per block, persistent blocks with register or TMA prefetch, and
paired multiplies.

## Getting good at this

What separates people who beat existing kernels:

- **Knowing the machine's numbers by heart.** That means latency of an L2 hit vs.
  DRAM, bytes per clock per SM, issue rate per scheduler, and how many bytes
  must be in flight to saturate HBM (Little's law). Get them by writing
  microbenchmarks: a copy kernel tuned until it matches `torch.clone`, then
  variants (load width, threads per block, blocks per SM) to see each
  effect alone.
- **Reading SASS fluently.** Read the SASS of every kernel you change, before
  and after. After a few dozen, you'll predict what the compiler will emit,
  and notice when it didn't do what you meant.
- **Predicting before measuring.** Keep score of your predictions. Being
  wrong and finding out why is how the intuition is built.
- **Reading others' kernels.** FlashInfer, vLLM, TransformerEngine,
  ThunderKittens and CUTLASS all solve these problems; read their SASS too,
  not just their source.

Where you can do better than an AI assistant doing this alone: I can't see
the hardware between tool calls, I know some Blackwell details only from
documentation (some of it out of date), and I make process mistakes (the
stale-build bug in the journal was mine). You get to keep intuition from
hands-on experiments across sessions, and to sit with a profile until
something looks odd. Most real wins start with "huh, that's odd". The best
setup is both: you form hypotheses from the profile, and we argue about them
before running.

### References

- *PTX ISA*: docs.nvidia.com/cuda/parallel-thread-execution
- *CUDA Binary Utilities* (cuobjdump, nvdisasm, SASS opcode list):
  docs.nvidia.com/cuda/cuda-binary-utilities
- *Nsight Compute Profiling Guide* (what every metric and stall reason
  means): docs.nvidia.com/nsight-compute/ProfilingGuide
- *CUDA C++ Best Practices Guide*: coalescing, occupancy, latency hiding.
- Jia et al., *Dissecting the NVIDIA Volta GPU Architecture via
  Microbenchmarking*: the method for measuring what NVIDIA doesn't
  document. Later papers do the same for Hopper and Blackwell.
- GPU MODE lectures: profiling and CUDA kernel talks from practitioners.
