# CompileIQ (ptxas ACF autotuning) on the FA4 kernels — integration + results

Goal: use [NVIDIA CompileIQ](https://nvidia.github.io/CompileIQ/stable/compilers_overview.html)
to tune ptxas controls for our NVFP4+FP8 / NVFP4+BF16 / MXFP8+FP8 kernels —
in particular to let ptxas interleave the softmax warp's different-pipe
instructions (MUFU / FMA / cvt) better.

**Bottom line: CompileIQ does not improve these kernels — verified three
ways with real measurements on this GB300 box (GPU 1), DSL 4.5.2:**

1. **FA4 kernels can't be runtime-wired.** The cutlass-DSL load path is fully
   sealed (7 interception methods tested, all dead — table below) and its
   linked nvPTXCompiler is 12.9/13.1 < the **13.3** that `--apply-controls`
   needs. So an ACF can't be fed into a running FA4 kernel.
2. **Static ACF search on the FA4 PTX** (28 evals, ptxas 13.3): no ACF beats
   the `-O3` default (best 2984 vs 2968 inst; registers launch-bounds-capped
   at 128 → CompileIQ's main lever removed; some ACFs add spills).
3. **Real runtime ACF tuning on Triton proxies** (the only end-to-end-
   tunable path — CompileIQ's first-class integration): no win.
   - bandwidth-bound fused softmax: best **1.003×** (within noise);
   - compute-bound multi-pipe kernel (independent exp+fma chains — the
     faithful softmax-warp interleaving proxy): best **0.930×** (every ACF
     *slower* than `-O3`).

Across static FA4 codegen and real runtime on both a memory- and a
compute-bound softmax-style kernel, the ptxas `-O3` default is at/near the
scheduling optimum and ACF perturbations don't help (often hurt).

## What CompileIQ does

Black-box HPO (evolutionary) over ptxas/nvcc controls, emitting an **Advanced
Controls File (ACF)** — a binary blob of compiler knobs (SASS gen, register
allocation, **instruction scheduling**, memory behavior) applied at PTX→SASS
via `ptxas --apply-controls cand.acf k.ptx` (**ptxas 13.3+**). Objective is a
user function returning measured latency; the ACF changes codegen without
touching source. Confirmed available here: `compileiq` 1.0.1, and
`ptxas 13.3` with `--apply-controls` at
`.venv/.../nvidia/cu13/bin/ptxas`.

## Why a runtime tuning loop is blocked (all paths tested)

CompileIQ's objective must *measure the tuned kernel's latency*. To feed an
ACF-tuned cubin into a running FA4 kernel you must intercept the DSL's
PTX→SASS→load→launch. Every interception point was tested and is dead:

| Hook | Result |
|------|--------|
| Python `load_cubin_module_data` (cute_dsl_utils patch) | never called |
| Python `cuModuleLoadData` / `…Ex` (cuda-python) | 0 calls |
| Python `cuLibraryLoadData` / `cuLaunchKernel` | 0 calls (default **and** `CUTE_DSL_ENABLE_TVM_FFI=1`) |
| `LD_PRELOAD` global symbol of `cuModuleLoadData` / `cuLibrary*` | 0 calls — `libcute_dsl_runtime.so` has no `NEEDED` libcuda; it `dlopen`s the driver, so PLT interposition is bypassed |
| **`dlsym` interposer** (catches dlopen-resolved symbols; `dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.34")`) | the interposer *fires* (the runtime resolves `cuModuleLoadData`/`Ex` through it), but the returned wrapper is **never called** for the kernel load |
| **`cuGetProcAddress`/`_v2` interposer** (modern driver resolution) | fires for all load symbols, redirects the returned pfn — wrapper still **never called** |
| **Global PLT defs of all 5 load fns** (`cuModuleLoadData`/`Ex`, `cuLibraryLoadData`/`FromFile`, `cuModuleLoad`) + dlsym + cuGetProcAddress, simultaneously | none entered — the kernel loads & runs correctly via *none* of them |
| Disk cache swap (`FLASH_ATTENTION_CUTE_DSL_CACHE_*`, `cutlass_python_cache`) | FA cache stores a pickled `.o`; DSL cache empty — no swappable cubin |
| Standalone cubin launcher | kernel `.entry` takes **19 params** incl. several 64-byte-aligned 128-byte packed CuTe tensor/scheduler descriptor blobs — reconstructing that ABI by hand is infeasible (one wrong byte → illegal access) |

The driver-symbol interposers (`dlsym`, `cuGetProcAddress`, global PLT) all
*successfully install* — the runtime resolves the load functions through
them — yet the kernel still loads & runs without ever invoking the wrappers,
so the MLIR ExecutionEngine reaches the driver via a path that bypasses every
userspace-visible module/library load entry point. Effectively unhookable.

Root cause: the DSL JITs/loads/launches entirely inside the native MLIR
ExecutionEngine. Plus the linked **nvPTXCompiler is 12.9** (4.5.2 base) /
**13.1** (4.5.2 `[cu13]`) — both **< 13.3**, so even routing `--apply-controls`
through the DSL's `ptxas_options` can't apply an ACF.

## The achievable integration: CompileIQ search on our PTX (static SASS)

What *can* run: dump the kernel PTX (`CUTE_DSL_KEEP_PTX=1`), then a real
CompileIQ `Search` over `PtxasSearchSpace(version="13.3")` whose objective
compiles that PTX with each candidate ACF (`ptxas-13.3 --apply-controls`) and
scores a deterministic SASS metric. Runtime can't be measured (the wall
above), so the objective is **SASS instruction count** — necessary-not-
sufficient for a speedup, but it reveals whether ptxas-control tuning has any
favorable leverage on our codegen. Scripts: `agent_space/ciq_search.py`
(search), `agent_space/ciq_inject.py` (the would-be runtime injector, kept for
when a hook exists).

Kernel: NVFP4+FP8, (1, 4096, 24, 128), sm_103a. 28 evaluations (pool 8 ×
gen 4), all compiled successfully:

| metric | baseline (`-O3`, no ACF) | CompileIQ search (min … max) |
|--------|--------------------------|------------------------------|
| SASS instructions | **2968** | 2984 … 3696 |
| registers | 128 | 128 … 128 (launch-bounds capped) |
| spill bytes | 0 | 0 … 332 |

**No ACF beat the default.** The minimum the search found (2984) is *above*
the `-O3` baseline (2968); registers never moved off the 128 launch-bounds
cap (so CompileIQ's biggest lever, register allocation, is unavailable here);
and several ACFs *introduced* spills (up to 332 B) — strictly worse. ACFs do
move the SASS (2968→3696, 0→332 B spill), i.e. leverage exists, but only
unfavorably on this static metric.

Caveat: static instruction count is **not** runtime — a different schedule
with equal/more instructions could still hide latency better. So I also ran
CompileIQ end-to-end with a **real runtime objective** on the one path that
*is* wireable: Triton kernels (CompileIQ's first-class integration — set
`TRITON_PTXAS_BLACKWELL_PATH`=ptxas-13.3, `PTXAS_OPTIONS=--apply-controls`,
`TRITON_ALWAYS_COMPILE=1`).

## Real runtime tuning on Triton softmax-style kernels (GB300)

Each CompileIQ eval runs the kernel in a fresh subprocess (avoids fork-after-
CUDA), `do_bench` latency = objective. Scripts: `agent_space/bench_softmax.py`,
`bench_compute.py`, `ciq_softmax_tune.py`, `ciq_compute_tune.py`.

| kernel (the softmax-warp workload) | baseline `-O3` | CompileIQ best | best speedup | worst |
|---|---|---|---|---|
| fused softmax (bandwidth-bound) | 0.0484 ms | 0.0483 ms | **1.003×** (noise) | 0.0556 ms |
| 4× independent exp+fma chains (compute-bound, register-resident — faithful proxy for the FA4 softmax warp's MUFU/FMA cross-pipe interleaving) | 0.4792 ms | 0.5152 ms | **0.930×** | 0.5639 ms |

On the compute-bound kernel — the case where ptxas instruction
scheduling/interleaving *should* matter most — **every** ACF was slower than
the plain `-O3` default (best 0.93×, i.e. −7%). So the runtime objective
confirms the static result: ptxas's default scheduling already interleaves
the MUFU/FMA pipes near-optimally for these kernels, and the ACF search only
perturbs it unfavorably. The softmax-cross-pipe-interleaving idea does not
yield a speedup via CompileIQ here.

## Verdict & path forward

CompileIQ gives **no win** for these kernels today, on two independent
grounds:
- **Mechanism**: the FA4 cute-DSL kernel is un-wireable for runtime tuning
  (sealed native load, nvPTXCompiler < 13.3).
- **Substance**: even where CompileIQ *can* run a real runtime loop (Triton
  proxies, incl. a compute-bound MUFU/FMA cross-pipe kernel) it finds no
  speedup — ptxas `-O3` already schedules these near-optimally and ACFs only
  regress them. The static FA4-PTX search agrees (no improvement, regs
  capped).

So CompileIQ is not the lever for the softmax-warp interleaving here. If we
still wanted to *try* it on the real FA4 kernel (e.g. on a future bigger/
more-scheduling-sensitive kernel), the wireup needs one of:
1. **A DSL hook for the cubin/ptxas step** — cutlass-dsl emits the cubin or
   calls the external `ptxas` binary so an ACF can be applied. Cleanest;
   needs an upstream/DSL change.
2. **nvPTXCompiler ≥13.3 with `--apply-controls` passthrough** via the DSL's
   `ptxas_options` (current linked compiler is 12.9/13.1).
3. **A standalone cubin launcher** reconstructing the 19-param kernel ABI —
   large and fragile.

Given the substance result (no speedup even on the runnable proxies), none of
these is worth building right now — the payoff would very likely be ~0.
