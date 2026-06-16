# CompileIQ (ptxas ACF autotuning) on the FA4 kernels — integration + results

Goal: use [NVIDIA CompileIQ](https://nvidia.github.io/CompileIQ/stable/compilers_overview.html)
to tune ptxas controls for our NVFP4+FP8 / NVFP4+BF16 / MXFP8+FP8 kernels —
in particular to let ptxas interleave the softmax warp's different-pipe
instructions (MUFU / FMA / cvt) better.

**Bottom line: CompileIQ cannot improve these kernels in the current setup.**
A *runtime* tuning loop is structurally blocked (the cutlass-DSL JIT is a
sealed native path with no cubin/launch hook, and its nvPTXCompiler is
< the 13.3 that `--apply-controls` needs). The one integration that *is*
achievable — a real CompileIQ search over ptxas-13.3 ACFs applied to our
dumped PTX, scored on static SASS — found **no ACF that beats the plain
`-O3` default**, and registers are launch-bounds-capped (removing CompileIQ's
main lever). Details below; all run on this GB300 box (GPU 1), DSL 4.5.2.

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
| `LD_PRELOAD` of `cuModuleLoadData` / `cuLibrary*` | not intercepted — `libcute_dsl_runtime.so` has no `NEEDED` libcuda; it `dlopen`s the driver and `dlsym`s from the handle, bypassing interposition |
| Disk cache swap (`FLASH_ATTENTION_CUTE_DSL_CACHE_*`) | cache stores a serialized `.o` (pickled JIT function), not a standalone cubin |
| Standalone cubin launcher | kernel `.entry` takes **19 params** incl. several 64-byte-aligned 128-byte packed CuTe tensor/scheduler descriptor blobs — reconstructing that ABI by hand is infeasible (one wrong byte → illegal access) |

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
with equal/more instructions could still hide latency better (the softmax
cross-pipe interleaving idea). But that can only be decided with a runtime
objective, which is exactly what the wall prevents.

## Verdict & path forward

For these FA4 kernels, today, CompileIQ gives no win: runtime tuning is
un-wireable, and the static-codegen search finds nothing better than `-O3`
with registers capped. To make CompileIQ usable here, one of:

1. **A DSL hook for the cubin/ptxas step** — have cutlass-dsl emit the cubin
   (or call the external `ptxas` binary) so an ACF can be applied and the
   result loaded. Cleanest; needs an upstream/DSL change.
2. **nvPTXCompiler 13.3+ with `--apply-controls` passthrough** via the DSL's
   `ptxas_options` (the current linked compiler is 12.9/13.1).
3. **A standalone cubin launcher** reconstructing the 19-param kernel ABI —
   large and fragile.
4. **A Triton port** of the kernel — CompileIQ has first-class Triton support
   (`TRITON_PTXAS_PATH` + `ptx_options=--apply-controls`), so a Triton
   version could be tuned end-to-end with a real runtime objective.
