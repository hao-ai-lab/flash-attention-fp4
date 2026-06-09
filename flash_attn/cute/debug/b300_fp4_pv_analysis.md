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

MMA WG processes `q_stage=2` tiles per KV block:
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
| **MMA WG per KV block**    |         |         |         |
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

The MMA WG must complete the full PV+QK cycle before the softmax results
from the same stage are needed again. With ping-pong, each softmax WG has
the entire MMA block cycle to complete its work. When softmax takes longer
than the MMA cycle, the MMA WG stalls waiting for P_full.

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

## Potential Improvements

1. **FP8 PV**: Reduce F2FP latency by interleaving with exp2 (already attempted
   in `_apply_exp2_pack_fp8` — the fragment-level interleaving helps on B200 but
   not enough on B300 where the fundamental issue is F2FP MIO throughput).

2. **FP4 PV**: Reduce P quantization overhead. Options:
   - Hardware-assisted E2M1 packing (future GPU arch)
   - Coarser quantization groups (larger sf_vec_size → fewer group_max reductions)
   - Approximate group_max (skip reduction, use tile-level max)

3. **Register allocation**: Sweeping `num_regs_softmax` (168–224) showed no
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

# Visualize pipeline model
python3 flash_attn/cute/debug/visualize_pipeline.py
```
