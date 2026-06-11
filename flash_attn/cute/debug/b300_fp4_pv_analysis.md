# B300 Block-Scaled Attention Performance Analysis

**GPU**: NVIDIA GB300 SXM6 AC (SM 10.3, 2070 MHz max boost, 148 SMs, 1300W)
**Date**: 2026-06-08
**Branch**: fp4_B300
**cutlass-dsl**: 4.5.2
**Clock**: Unlocked (2070 MHz max boost, no frequency lock)

## Summary

B300 (SM 10.3) doubles MUFU.EX2 throughput (32 ops/clock/SM vs 16 on SM100).
This makes exp2 no longer the softmax co-bottleneck. The benefit depends on
the PV mode:

| PV Mode | B300 TF | vs BF16 ref | B200 TF | B200 vs BF16 ref |
|---------|---------|-------------|---------|------------------|
| BF16    | 2142    | 1.37x       | ~1887   | 1.22x            |
| FP8     | 1824    | 1.17x       | 2018    | 1.31x            |
| FP4     | 1340    | 0.86x       | ~1310   | ~0.85x           |
| BF16 ref| 1561    | 1.00x       | 1545    | 1.00x            |

Shape: b=1, s=4096, h=24, d=128 (BF16 ref uses non-FP4 QK).
Peak numbers (s=32768): BF16 PV 2142 TF, FP8 PV 1774 TF, FP4 PV 1340 TF.

**Key finding**: On B300, exp2 is no longer the bottleneck for any PV mode.
The new bottlenecks are:
- **BF16 PV**: MMA-bound → 1.37x (best result)
- **FP8 PV**: F2FP packing (32/clk, exactly half the BF16 cvt rate) → softmax-bound
- **FP4 PV**: P quantization instruction throughput → heavily softmax-bound

## Corrected Roofline Analysis

### FA4 Paper Table 1 (M=N=d=128, per tile, SM100 B200)

| Resource       | Cycles |
|----------------|--------|
| MMA (BF16 QK)  | 1024   |
| MMA (FP4 QK)   | 256    |
| SMEM load       | 768    |
| Exp (MUFU.EX2)  | 1024   |

The paper shows exp2 takes the **same** 1024 cycles as one BF16 MMA tile.
This means exp2 IS a co-bottleneck with MMA on B200, contradicting the
previous version of this analysis.

### B300 (SM103) Adjustments — Measured Instruction Throughput

Measured on this GB300 with `agent_space/bench_cvt_throughput.cu` (single SM,
8 independent serial chains per thread, 512 threads = saturated):

| Instruction                      | instr/clk/SM | Note                              |
|----------------------------------|--------------|-----------------------------------|
| `ex2.approx.ftz.f32` (MUFU)     | **32.0**     | 2x SM100's 16 — the B300 doubling |
| `cvt.rn.bf16x2.f32`             | **62**       | ~64 limit; trivial FP32 truncation|
| `cvt.rn.satfinite.e4m3x2.f32`   | **32.0**     | FP8: exactly half the BF16 rate   |
| `cvt.rn.satfinite.e2m1x2.f32`   | **57**       | FP4 cvt itself is fast            |
| mix 4x ex2 + 4x e4m3 cvt        | 54.6 combined| partial port overlap, not full 64 |

Key facts:
- MUFU.EX2 = 32/clk confirms the SM103 2x doubling (SM100: 16/clk).
- FP32→FP8 conversion runs at **exactly half** the FP32→BF16 rate. BF16 is a
  round+truncate of the top 16 bits of FP32 (full-width datapath); E4M3 needs
  exponent rebias, mantissa renormalization, saturation and NaN remapping —
  implemented at half rate.
- FP32→FP4 (E2M1) cvt is nearly as fast as BF16 — the FP4 PV cost is NOT the
  conversion, it's the surrounding group_max/scale/register traffic.
- MMA throughput: unchanged vs SM100 → same GEMM cycle counts.

### Pipeline Cycle Model (steady state, per iteration)

MMA warp processes `q_stage=2` tiles per KV block:
```
Per KV block: PV[0] + PV[1] + QK[0] + QK[1]
```

Each softmax WG handles one stage, ping-pong between WG0 (stage 0) and WG1 (stage 1):
```
Wait S → Load S → row_max → exp2 → row_sum → P pack/quant → Write P → Signal P_full
```

Softmax cycle counts below use the measured throughputs with a 2x contention
factor because in steady state the two softmax WGs overlap on the same SM and
share the vector pipes. Per softmax step each WG executes 16384 ex2 and 8192
packed-cvt thread-instructions (128×128 tile, 2 elements per cvt):

| Component                  | BF16 PV | FP8 PV  | FP4 PV  |
|----------------------------|---------|---------|---------|
| **MMA warp per KV block**    |         |         |         |
| QK GEMM (FP4, ×2 stages)  | 512     | 512     | 512     |
| PV GEMM (×2 stages)       | 2048    | 1024    | 512     |
| Total MMA per block        | 2560    | 1536    | 1024    |
| **Softmax WG per stage**   |         |         |         |
| TMEM load + row_max        | 150     | 150     | 150     |
| exp2 (16384 / (32/2))     | 1024    | 1024    | 1024    |
| row_sum                    | 100     | 100     | 100     |
| P pack/quant               | 256     | 512     | 1500    |
| TMEM store + signal        | 60      | 60      | 60      |
| Total softmax per stage    | 1590    | 1846    | 2834    |
| **Bottleneck**             | MMA     | softmax | softmax |

P pack: BF16 = 8192/(62/2) ≈ 256; FP8 = 8192/(32/2) = 512; FP4 quant is
dominated by group_max + scale + register shuffling (+914 PTX instructions),
not the E2M1 cvt itself (57/clk measured).

## Real In-Kernel Trace (measured, not modeled)

The kernel has flashinfer-style timestamp instrumentation
(`flash_attn/cute/profiler.py`), compiled in via `FA4_PROFILE_PIPELINE=1`.
One elected lane per warpgroup records `%clock` cycles at every pipeline
event into a gmem buffer. Run `flash_attn/cute/debug/trace_pipeline.py`
to capture and render a trace (3 rows: MMA warp, Softmax WG0, Softmax WG1).

Two granularities (see the instrumentation-artifact warning below):
- **Coarse (default)**: events only at wait/store boundaries that are already
  side-effect ordered — the softmax compute stream contains no timestamps, so
  ptxas keeps full scheduling freedom and the per-step numbers are close to
  the clean kernel (e.g. FP8: 96-iteration window 388K cycles coarse vs 472K
  with detailed events vs ~identical clean TFLOPS). One combined "softmax
  compute" span per step.
- **Detailed (`FA4_PROFILE_DETAIL=1`)**: per-phase load/exp/quant spans;
  phase boundaries inhibit compiler interleaving and inflate spans ~15%.

Average step costs, coarse mode, block 0, b=1 s=4096 h=24 d=128 (the trace
figures show this line under the title):

| Per step (cycles)            | BF16 PV | FP8 PV | FP4 PV |
|------------------------------|---------|--------|--------|
| softmax step period          | 4,855   | 4,001  | 5,389  |
| softmax step busy            | 2,352   | 2,288  | 3,469  |
| MMA QK+PV GEMM (one stage)  | 732     | 639    | 622    |
| MMA wait-P (per PV)          | 933     | 595    | 1,171  |

Measured means per event, block 0, b=1 s=4096 h=24 d=128 (96 softmax
iterations per WG, GB300 at 2070 MHz; cycles from %clock):

| Event (mean cycles)        | BF16 PV | FP8 PV  | FP4 PV  |
|----------------------------|---------|---------|---------|
| **MMA warp**                 |         |         |         |
| wait P (stall)             | 841 (33%)| 950 (38%)| 1306 (42%)|
| PV GEMM issue              | 560     | 405     | 549     |
| QK GEMM issue              | 262     | 267     | 246     |
| wait KV (TMA)              | 95      | 120     | 105     |
| **Softmax WG (per iter)**  |         |         |         |
| S load + row_max           | 697     | 821     | 666     |
| exp2 (+fused pack)         | 1581    | 1674    | 1579    |
| P quant / pack             | 75      | 325     | 1283    |
| P store + signal           | 380     | 230     | 177     |
| wait S                     | 581     | 759     | 594     |
| wait corr                  | 867     | 173     | 1013    |

Real-trace findings:
- Even BF16 PV is partially softmax-bound on GB300: the MMA warp spends 33%
  of its time waiting for P. The exp2 span (which includes the fused BF16
  pack) measures ~1580 cycles — well above the 512-cycle MUFU-only roofline,
  because both softmax WGs contend for the SM and the span includes the
  e2e bookkeeping around the MUFU burst.
- FP8 PV: the visible `quant` span only adds ~325 cycles because most of
  the F2FP work is interleaved into the exp2 span (`_apply_exp2_pack_fp8`),
  which grows by ~90 cycles; total extra ~340 cycles per iteration over
  BF16, pushing MMA wait-P to 38%.
- FP4 PV: P quantization measures ~1283 cycles per iteration on top of the
  same exp2 cost — the softmax iteration grows from ~2,700 to ~3,700 cycles
  while the PV GEMM gets cheaper, so the MMA warp stalls 42% of the time.
- GEMM *issue* spans are short (250-560 cycles) — tcgen05 MMAs execute
  asynchronously, so the MMA warp's real exposure is the waits, exactly what
  the trace shows.
- The PV issue sequence embeds a second wait: softmax stores P in two halves
  (`mbar_P_full_O_rescaled`, then `mbar_P_full_2`), and `gemm_ptx_partial`
  issues the first K-tiles, then `mbarrier.try_wait`s for the second half
  before issuing the rest. This wait is now MEASURED directly — the GEMM's
  inline PTX stores %clock right before and after the try_wait loop
  (`prof_ts_addrs` in `gemm_ptx_partial`/`gemm_ptx_partial_fp4`), and the
  trace draws it as a light-gray overlay at its actual position inside the
  PV bar. Measured steady state: wait-P2 mean is only 71-85 cycles across
  all three PV modes (the mbarrier round-trip when P's 2nd half is already
  signaled), so the ~420-cycle PV-span baseline is tcgen05 issue cost, and
  PV-span variance is dominated by issue backpressure, not the P2 wait.
  (An earlier run without this measurement showed a 1222-cycle PV tail that
  we wrongly attributed to the P2 wait — the measured data corrected this.)

The MMA WG must complete the full PV+QK cycle before the softmax results
from the same stage are needed again. With ping-pong, each softmax WG has
the entire MMA block cycle to complete its work. When softmax takes longer
than the MMA cycle, the MMA warp stalls waiting for P_full.

## PTX Instruction Analysis

Instruction counts from PTX (e2e=OFF, SM103 target, full kernel):

| Category              | BF16 PV | FP8 PV  | FP4 PV  | FP4 delta |
|-----------------------|---------|---------|---------|-----------|
| **Total instructions**| 5089    | 5089    | 6003    | +914 (18%)|
| ex2.approx (MUFU)    | 257     | 257     | 257     | 0         |
| fma.rn.f32x2          | 128     | 128     | 256     | +128      |
| mul.rn.f32x2          | 256     | 256     | 256     | 0         |
| add.rn.f32x2          | 127     | 127     | 127     | 0         |
| max.f32               | 132     | 132     | 308     | +176      |
| cvt (total)           | 266     | 266     | 276     | +10       |
| cvt.rn.bf16x2.f32     | 128/256 | 128     | 128     | 0         |
| cvt.e4m3x2 (F2FP)    | 0/128   | 128     | 8       | –          |
| cvt.e2m1x2            | 0       | 0       | 128     | +128      |
| selp                  | 160     | 160     | 157     | –3        |
| tcgen05 (MMA/TMEM)   | 107     | 107     | 119     | +12       |
| mbarrier              | 107     | 107     | 125     | +18       |
| mov (data movement)   | 2277    | 2277    | 2703    | +426      |
| INT ALU               | 1086    | 1086    | 1238    | +152      |

**FP8 PV vs BF16 PV**: Identical instruction count (5089). The only difference
is 128 `cvt.rn.bf16x2.f32` (BF16 P pack) replaced by 128 `cvt.rn.satfinite.e4m3x2.f32`
(FP8 P pack / F2FP). F2FP uses the MIO pipe and is slower on SM103.

**FP4 PV extra instructions** (+914, all in softmax WG):
- +128 fma.f32x2: scale computation in `_fused_group_max_scale_quant`
- +176 max.f32: group_max reduction (16-element groups × 8 groups)
- +128 cvt.e2m1x2: E2M1 packing
- +426 mov: register shuffling for group processing
- +18 mbarrier: extra sync for SFP SMEM copy

## Why FP8 PV is Slower than BF16 PV on B300

The instruction COUNT is identical — the PTX census shows the FP8 PV kernel
differs from BF16 PV only in 128 `cvt.rn.bf16x2.f32` replaced by 128
`cvt.rn.satfinite.e4m3x2.f32`. The difference is hardware THROUGHPUT:

- `cvt.rn.bf16x2.f32`: 62/clk/SM. BF16 is the top 16 bits of FP32, so the
  conversion is a round+truncate on the full-width FP32 datapath.
- `cvt.rn.satfinite.e4m3x2.f32`: exactly 32/clk/SM (half). E4M3 needs
  exponent rebias from 8-bit to 4-bit range, mantissa renormalization,
  saturation clamping, and NaN remapping — narrower dedicated hardware.

So P-pack costs 2x the cycles for FP8 (512 vs 256 per stage, contended).
On top of that, the FP8 PV GEMM is 2x faster than BF16 PV GEMM (1024 vs
2048 per block), so the MMA cycle shrinks from 2560 to 1536 while the
softmax stage grows from 1590 to 1846 — the kernel flips from MMA-bound
to softmax-bound, and the extra PV speed cannot be realized. e2e emulation
cannot help: the bottleneck is F2FP packing, not exp2.

On B200 the same FP8 path was relatively better (1.31x) because the BF16
baseline softmax was also exp2-bound (MUFU at 16/clk), hiding the F2FP cost.
B300 doubled MUFU (16→32/clk) but left F2FP at 32/clk, exposing it.

## Why FP4 PV is Slower than BF16 Reference

The softmax WG does significantly more work for FP4 PV:
1. `exp2()` — 512 cycles (same as BF16)
2. `update_row_sum()` — 100 cycles (same, but moved BEFORE quant for correctness)
3. **`compute_group_max()`** — per-group (16-element) max reduction → +176 max.f32
4. **`scale_groupwise()`** — per-element division by group_max → +128 fma.f32x2
5. **`_quant_fp4()` (E2M1 pack)** — 8 floats → 1 uint32 → +128 cvt.e2m1x2 + bit ops
6. **SF packing + R2S copy** — pack scale factors to UE4M3, copy to SMEM

Items 3–6 add ~1500 cycles to softmax, making it ~2834 cycles per stage vs
the 1024-cycle MMA block for FP4×FP4 PV GEMM. The MMA WG spends most of its
time stalled waiting for P. This explains the 0.86x performance. Note the
E2M1 cvt itself is fast (57/clk measured) — the cost is the group_max
reduction, per-group scaling, and the register traffic they generate.

## e2e Emulation on B300

| Mode       | e2e=ON  | e2e=OFF | Change | Reason                                |
|------------|---------|---------|--------|---------------------------------------|
| BF16 PV    | 1945 TF | 2142 TF | +10%   | e2e hurts: exp2 already fast (512 cy) |
| FP8 PV     | 1851 TF | 1824 TF | ~same  | F2FP is bottleneck, not exp2          |
| FP4 PV     | 1243 TF | 1340 TF | +8%    | e2e adds instructions to overloaded WG|

**Recommendation**: Disable e2e emulation on SM103 for all modes. Already
implemented in `_FP4_TUNING_CONFIG_SM103` with `enable_e2e: False`.

## Fixes Applied (fp4_B300 branch)

### 1. SM103-aware tuning config
`flash_fwd_sm100_fp4.py`: Added `_is_sm103()` detection and separate
`_FP4_TUNING_CONFIG_SM103` with `enable_e2e: False`. The config is selected
at import time based on `torch.cuda.get_device_capability()`.

### 2. `MmaF8F6F4Op` support (cutlass-dsl 4.5 rename)
`blackwell_helpers.py:_tcgen05_mma_kind` supports both `MmaFP8Op` (4.4)
and `MmaF8F6F4Op` (4.5+).

### 3. `max_offset = log2(6)` for FP4 PV
P output range shifted from [0, 1] to [0, 6] (E2M1 max) via `max_offset`.

### 4. Fused `_fused_group_max_scale_quant`
Processes each sf_vec_size group sequentially to reduce register pressure.

### 5. `nvidia-cutlass-dsl` 4.5 rounding mode compat
`utils.py:_RND_RN` auto-detects enum vs string API.

## Overlap Investigation: F2FP/quant vs Other Hardware Units (measured)

We tried to speed up FP8/FP4 PV by overlapping the bottleneck F2FP/quant
stream with instructions on other units (MUFU, TMEM stores, st.shared).
Result: **no speedup — the overlap already exists in the compiled SASS.**

What was tried (all bitwise-validated against baseline outputs):

1. **Knob sweep** (`FA4_FP8_PV_USE_FUSED_PACK`, `FA4_FORCE_E2E`, combinations):
   all variants land at 1909-1910 TF on (1, 32768, 24, 128). Source-order
   interleaving of exp2/F2FP has no effect.
2. **Chunk-pipelined FP8 path** (`FA4_FP8_PV_PACK_STORE_PIPELINE=1`,
   `_pack_fp8_store_pipelined`): software-pipelines exp2(chunk c) with
   F2FP-pack(c-1) and tcgen05.st(c-1) — three data-independent streams on
   MUFU / cvt / TMEM ports — and fires P_full right after the first
   `mbar_p_split` chunks. Measured: 1521/1638/1909 TF at s=4096/8192/32768,
   identical to baseline (1521/1638/1909).
3. **Chunk-pipelined FP4 path** (`FA4_FP4_PV_QUANT_STORE_PIPELINE=1`,
   `_fused_group_max_scale_quant_store_pipelined`): issues each P chunk's
   TMEM store as soon as its groups are quantized. Measured 1389 vs 1387
   TF baseline — parity.

Why: dumping SASS for baseline vs pipelined (ptxas -O3, sm_103a) shows
**ptxas already produces an equivalently interleaved schedule for the
baseline**. FP8: identical MUFU.EX2/F2FP run structure (83 transitions in
both). FP4: the per-group `MUFU.RCP → FMNMX×10 → F2FP×8` quant pattern is
already finely interleaved with the MUFU.EX2 stream (160 vs 168 unit-runs).
The whole softmax step is one fully-unrolled basic block, so ptxas freely
schedules across the source-level phases, and the hardware scoreboard
dual-issues across ports where possible (measured mixed ex2+F2FP throughput
54.6/clk vs 32 each in isolation — already reflected in kernel timing).

**Materialized-loop control (measured)**: to verify that the full unrolling
(not compiler magic) is what enables the overlap, we rebuilt the FP8
exp2+pack as a real `cutlass.range` IR loop over fragments
(`FA4_FP8_PV_RANGE_UNROLL`, `_exp2_pack_fp8_range`). With a live loop
(unroll=1 or 2), the dynamic fragment index makes the register-resident
S/P tensors unaddressable, so they spill to local memory (240 st.local +
208 ld.local in PTX vs 0 baseline): **227 TF, an 8.4x slowdown**. With
`unroll_full` the IR unroller restores constant indices and the result is
exactly baseline (1909 TF) — confirming `range_constexpr` ≡ fully-unrolled
`cutlass.range`, and that register residency requires the unrolled form.
(Also: the `unroll=` kwarg on `range_constexpr` is silently discarded by
the DSL preprocessor — it only means something on `cutlass.range`.)

**Instrumentation artifact warning**: with `FA4_PROFILE_PIPELINE=1`, the
chunk-pipelined FP8 variant looks ~14% faster per CTA than the instrumented
baseline (MMA wait-P 1055→713 cy). This is an artifact: the profiler's
side-effecting `%clock` inline asm between the exp/pack/store phases acts
as a scheduling barrier and prevents ptxas from interleaving the baseline.
The clean kernels are identical. Do not tune from instrumented runs alone.

The remaining FP8/FP4 PV gap is therefore raw issue-slot count on the
cvt/MUFU/ALU ports, not scheduling. Real improvement options:
- Fewer instructions: coarser SF groups (fewer group_max FMNMX), approximate
  group_max, packed-max if a 2-wide min/max op exists on SM103.
- Move quant work to the underutilized correction WG (requires a register →
  TMEM/SMEM round-trip of P — likely costs more than it saves).
- Hardware-assisted E2M1 packing (future arch).

Also observed while validating: the FP4 PV kernel has **pre-existing
run-to-run nondeterminism** — ~1 in 20 runs of the *unmodified baseline*
differs from its own reference output (diffs up to 3.8 in O on a couple of
adjacent seq rows). Bisects show it is unrelated to the new pipelined paths
(they reproduce it at the same rate). Likely a latent race in the FP4 PV
path; needs separate investigation.

## Other Notes

- **Register allocation**: Sweeping `num_regs_softmax` (168–224) showed no
  improvement for FP4 PV — the bottleneck is instruction throughput, not
  register pressure (register spills are 15K local ops vs 81M total instructions).

## Commands

```bash
# Check GPU clock and power
nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.max.sm,power.limit --format=csv

# Benchmark all PV modes
CUDA_VISIBLE_DEVICES=1 python3 -m flash_attn.cute.benchmarks.bench_fp4
CUDA_VISIBLE_DEVICES=1 python3 -m flash_attn.cute.benchmarks.bench_fp4 --pv_mode fp8
CUDA_VISIBLE_DEVICES=1 python3 -m flash_attn.cute.benchmarks.bench_fp4 --quant_v

# Generate PTX for instruction analysis
CUDA_VISIBLE_DEVICES=1 CUTE_DSL_KEEP_PTX=1 python3 agent_space/profile_stalls.py --pv_mode bf16

# Real in-kernel pipeline trace (instrumented kernel, %clock timestamps)
CUDA_VISIBLE_DEVICES=1 python3 flash_attn/cute/debug/trace_pipeline.py --pv_mode bf16
CUDA_VISIBLE_DEVICES=1 python3 flash_attn/cute/debug/trace_pipeline.py --pv_mode fp8
CUDA_VISIBLE_DEVICES=1 python3 flash_attn/cute/debug/trace_pipeline.py --pv_mode fp4

# Theoretical pipeline model (roofline-based, no kernel run needed)
python3 flash_attn/cute/debug/visualize_pipeline.py
```
