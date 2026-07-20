---
name: kernel-bottleneck-tuning
description: Workflow for finding and fixing instruction-throughput bottlenecks in warp-specialized CuTeDSL kernels (FA4-style). Use when a kernel underperforms its roofline and you need to decide between reordering, eliminating, or re-pipelining work.
---

# Kernel Bottleneck Tuning (validated on FA4 FP4/FP8 PV, GB300)

Methodology that produced +15% (FP4 PV log-domain quant) and +24% (FP8 PV
P-split) on this repo. Follow the order — each step rules out a class of fix
before you spend effort on the next.

## 1. Measure instruction throughputs in isolation

Microbenchmark the suspect instructions with serial dependency chains
(8 independent chains/thread, feed output back to input so the loop can't be
hoisted; `%clock64` with `"memory"` clobber around the timed region — naive
volatile-asm loops get scheduled away). Template:
`agent_space/bench_cvt_throughput.cu`. This gave: ex2 = 32/clk/SM,
bf16-cvt = 62, e4m3-cvt = 32, e2m1-cvt = 57, mixed ex2+e4m3 = 54.6
(partial dual-issue).

## 2. Trace the real pipeline, coarsely

`FA4_PROFILE_PIPELINE=1 python3 flash_attn/cute/debug/trace_pipeline.py
--pv_mode {bf16,fp8,fp4}`. Use the default coarse mode for absolute numbers
(per-step period/busy, MMA wait-P). `FA4_PROFILE_DETAIL=1` gives the
per-phase breakdown but inflates spans ~15-30%: each `%clock` inline asm is
side-effecting and blocks ptxas from interleaving across the boundary.
**Never tune from detailed/instrumented runs alone.**

## 3. Verify scheduling hypotheses in SASS before reordering source

Don't software-pipeline by hand until you've checked the compiler hasn't
already done it: dump PTX (`CUTE_DSL_KEEP_PTX=1`, strip NULs with
`tr -d '\000'`), `ptxas -arch=sm_103a -O3`, `nvdisasm -c`, then compare
unit-run interleaving (e.g. MUFU↔F2FP transition counts) between variants.
On FA4 the whole softmax step is one fully-unrolled branch-free region
(`range_constexpr` = trace-time unrolled; its `unroll=` kwarg is silently
discarded), so ptxas already interleaves optimally — source reordering and
chunk-pipelining measured exactly 0.
Corollary: register-resident fragments REQUIRE the unrolled form — a
materialized `cutlass.range` loop makes the index dynamic and spills the
fragment to local memory (measured 8.4x slowdown).

## 4. If scheduling has no headroom: eliminate instructions algebraically

The wins come from algebra, not order. Pattern that paid off — **log-domain
refactoring**: exp2 is monotonic, so `max(exp2 s) = exp2(max s)`, and any
multiplicative post-exp scale folds into the exp2 argument as a subtract:
`exp2(s)/exp2(m) = exp2(s-m)`. For FA4 FP4 quant this deleted all per-group
divisions and the whole post-exp scaling pass (+15%). Look for: divisions
that could be exp2 of a negated log, multiply passes foldable into an
existing exp/log, reductions movable across a monotonic function to where
the data already is.

## 5. Tune the producer→consumer handoff split

Warp-specialized pipelines hand data over in chunks gated by two mbarriers
(first fraction at signal 1, rest at an embedded wait). The split fraction
is a tuning knob with hardware-generation-dependent optima: FP8 PV's 1/2 (a
B200 tuning) cost 24% on B300 vs 3/4. Safety contract: the producer-side
split fraction must be >= the consumer GEMM's `pre_mbar_tiles` fraction
(both derive from `mbar_p_split`); violating it reads unwritten TMEM (1/4
crashed). Sweep via `FA4_FP8_PV_P_SPLIT_NUM/DEN` (+ `FA4_FP4_PV_P_SPLIT_*`,
`FA4_FP4_PV_TMEM_STORE_REP` for chunk count).

## 6. Validation discipline (every change)

- **Bitwise harness with SAVED inputs** (`agent_space/check_pv_pipeline.py`):
  tensor creation (nvfp4 quantize amax) is NOT reproducible across processes
  — comparing fresh-tensor runs produces phantom diffs.
- Pure reorderings must be bitwise-identical; algebraic changes must show
  error vs an FP32 reference identical to baseline
  (`agent_space/check_fp4_log2.py`).
- **Flake-test ~20 runs** against the saved reference: the FP4 baseline had
  a pre-existing 1-in-20 nondeterminism — a single-run pass can lie, and a
  "failure" may predate your change (bisect with the env knobs).
- Env-knob variants need a fresh process per setting (constexprs read env at
  init; the in-memory compile cache won't recompile).
