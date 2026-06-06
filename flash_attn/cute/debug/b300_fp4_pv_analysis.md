# B300 Block-Scaled Attention Performance Analysis

**GPU**: NVIDIA B300 SXM6 AC (SM 10.3, 2032 MHz max boost)
**Date**: 2026-06-05
**Branch**: fp4
**cutlass-dsl**: 4.5.2

## Key Finding

B300's 2x MUFU.EX2 throughput benefits **all** attention paths equally (BF16, FP4+BF16,
FP4+FP8). The relative speedup of FP4 over BF16 is similar to B200, not better.
For FP4+FP4 PV, the bottleneck is P quantization (group_max + scale + E2M1 pack),
which is unrelated to exp2 throughput.

## B300 Results (TFLOPS, shape: b=1, s=32768, h=24, d=128)

| Mode | B300 TF | vs BF16 | B200 TF (README) | B200 vs BF16 |
|------|---------|---------|------------------|--------------|
| BF16 (ref) | 1562 | 1.00x | 1545 | 1.00x |
| NVFP4 QK + BF16 PV | 1770 | 1.13x | 1887 | 1.22x |
| NVFP4 QK + FP8 PV | 1785 | 1.14x | 2018 | 1.31x |
| NVFP4 QK + FP4 PV | 1371 | 0.88x | ~1310 | ~0.85x |

B200 vs B300 absolute TFLOPS differ by ~3-7% due to lower boost clock (2032 vs ~2100 MHz).
Relative speedups are similar except FP4+FP8 drops from 1.31x to 1.14x.

## Why FP4+FP8 Relative Speedup Drops on B300

On B200, MIO throttle for BF16 is higher (exp2 is the bottleneck for softmax WG).
FP4 QK halves the QK GEMM time, so the kernel becomes more softmax-bound.
The e2e emulation fills the gap by replacing some MUFU.EX2 with FP32 polynomial
approximation, overlapping softmax compute with MMA.

On B300 with 2x MUFU, the BF16 kernel's softmax is already fast (MIO throttle = 1.24%).
The BF16 baseline rises, reducing the relative advantage of FP4 QK.

NCU on B300 BF16 kernel: MIO throttle = 1.24% (not bottlenecked on exp2).

## Fixes Applied

### 1. `MmaF8F6F4Op` support (cutlass-dsl 4.5 rename)
`blackwell_helpers.py:_tcgen05_mma_kind` now supports both `MmaFP8Op` (4.4)
and `MmaF8F6F4Op` (4.5+). Without this, FP8 PV failed to compile on B300.

### 2. `max_offset = log2(6)` for FP4 PV
P output range shifted from [0, 1] to [0, 6] (E2M1 max) via `max_offset`
in `SoftmaxSm100.create`. Uses unified `max_offset` mechanism instead of
the duplicate `p_log2_offset`.

### 3. `FA4_FORCE_E2E` env var
The FP8 PV path previously hardcoded `force_e2e = True` in `__call__`,
ignoring the env var set in `__init__`. Now respects the env var for
testing pure hardware exp2 on B300.

### 4. Fused `_fused_group_max_scale_quant`
Processes each sf_vec_size group sequentially. Reduced register spills from
2.36M to 15K local ops. Same TFLOPS due to compute-bound bottleneck.

### 5. `nvidia-cutlass-dsl` 4.5 rounding mode compat
`utils.py:_RND_RN` auto-detects enum vs string API based on cutlass version.

## e2e Emulation Analysis on B300

| Mode | e2e=ON (B200 default) | e2e=OFF | Note |
|------|----------------------|---------|------|
| FP4+BF16 PV | 1945 TF | 1770 TF | e2e HELPS (overlaps FP32 ALU with MMA) |
| FP4+FP8 PV | 1851 TF | 1785 TF | e2e HELPS slightly |
| FP4+FP4 PV | 1243 TF | 1370 TF | e2e HURTS (adds instructions to softmax WG) |

e2e emulation helps BF16/FP8 PV because it replaces MUFU instructions with FP32 ALU
work, which overlaps better with the MMA warp group. For FP4 PV, the softmax WG is
already overloaded with P quantization — adding emulation makes it worse.

## NCU Profiling (b=1, s=4096, h=24, d=128, e2e=OFF)

| Metric | BF16 PV | FP4 PV |
|--------|---------|--------|
| SASS Instructions (M) | 81.8 | 98.1 |
| Local loads/stores | 0 / 0 | 15K / 6K |
| Wait stalls | 11.8% | 25.7% |
| MIO throttle | 0.93% | 0.01% |

FP4 PV adds 20% more instructions (P quantization) and the MMA WG waits
25.7% of the time (vs 11.8% for BF16 PV).

## Remaining FP4 PV Bottleneck

With e2e disabled, the softmax WG does:
1. `exp2()` — fast on B300, MIO throttle=0%
2. `update_row_sum()` — reduction
3. `compute_group_max()` — per-group (16-element) max reduction ← extra
4. `scale_groupwise()` — per-element division ← extra
5. `_quant_fp4()` — E2M1 packing ← extra
6. SF packing + SF→SMEM copy ← extra

Items 3-6 add ~20% more instructions. Since the MMA is already fast
(FP4 GEMMs), the softmax WG is the bottleneck.

## Commands

```bash
# Set clocks
nvidia-smi -pm 1; nvidia-smi -ac 3996,2032

# Test with e2e disabled
FA4_FORCE_E2E=0 python3 -m flash_attn.cute.benchmarks.bench_fp4
FA4_FORCE_E2E=0 python3 -m flash_attn.cute.benchmarks.bench_fp4 --pv_mode fp8
FA4_FORCE_E2E=0 python3 -m flash_attn.cute.benchmarks.bench_fp4 --quant_v
```
