# FP4 Kernel Investigation & Optimization

## Context
- Our FP4 block-scaled FA4 kernel peaks at ~1800 TFLOPS (1867 with quant removed)
- New FP8 FA4 PR ([Dao-AILab/flash-attention#2109](https://github.com/Dao-AILab/flash-attention/pull/2109)) claims ~1950 TFLOPS
- They use standard FP8 tcgen05.mma (not block-scaled), with per-head descaling folded into softmax
- Upstream FA4 main branch has 2-CTA support (`FA_DISABLE_2CTA=0` by default for hdim=128 non-causal)
- Our kernel uses 1-CTA (`tcgen05.mma.cta_group::1`)

## Tasks

- [x] **1. Investigate FP8 PR #2109 TFLOPS gap** (completed — root cause identified below)
  - Even with all quant instructions commented out (lines 3200-3212), our kernel only reaches 1867 TFLOPS vs their 1950
  - Key difference: they use 2-CTA gemm instructions; we use 1-CTA
  - Verify their numbers are real by running their code if possible
  - Identify all performance-relevant differences (2-CTA, tile sizes, pipeline depth, etc.)
  - **Commit**: git commit of flash-attention-fp4 repo used for benchmarks
  - **Command**: `CUTE_DSL_ENABLE_TVM_FFI=1 python -m flash_attn.cute.benchmarks.bench_fp4`

### Task 1 Findings

**PR #2109 does NOT introduce 2-CTA — the upstream FA4 main already has it.**
The PR is purely FP8 data type support. Key findings:
- FP8 uses standard `tcgen05.mma` with `kind::f8f6f4` (not block-scaled)
- Per-head descale factors (q_descale * k_descale) folded into softmax_scale
- V descale applied during output normalization
- `max_offset=8` trick to prevent FP8 underflow
- e2e softmax polynomial disabled for FP8 (hurts performance)

**Upstream FA4 2-CTA is enabled by default** for hdim∈{128,192}, non-causal, non-split-kv.
Controlled by `FA_DISABLE_2CTA` env var (`utils.py:60`).

#### Benchmark Results (B200 sm100, hdim=128, non-causal, TFLOPS)

All benchmarks on PR #2109 FP8 branch (`cc52c48f`), except "Our FP4" column which uses our fp4 branch (`1aa24eef`).

| batch | seqlen | BF16 2-CTA | BF16 1-CTA | FP8 2-CTA | FP8 1-CTA | Our FP4 1-CTA |
|-------|--------|-----------|-----------|----------|----------|---------------|
| 32    | 512    | 781.7     | 982.7     | 1102.3   | 1117.8   | 991.8         |
| 16    | 1024   | 1004.2    | 1223.4    | 1390.2   | 1425.6   | 1261.3        |
| 8     | 2048   | 1154.7    | 1375.0    | 1629.4   | 1639.4   | 1474.0        |
| 4     | 4096   | 1242.0    | 1457.4    | 1806.0   | 1801.9   | 1635.8        |
| 2     | 8192   | 1293.7    | 1509.7    | 1913.1   | 1897.7   | 1730.8        |
| 1     | 16384  | 1316.4    | 1557.8    | **1959.5** | 1939.9 | 1776.9        |
| 1     | 32768  | 1232.1    | 1503.6    | 1950.1   | **1967.7** | 1792.3      |
| 4     | 8192   | 1293.1    | 1522.4    | 1942.4   | 1931.5   | 1751.6        |

- **Commits**: PR#2109 `cc52c48f`, our fp4 `1aa24eef`
- **Command**: `CUTE_DSL_ENABLE_TVM_FFI=1 python /tmp/bench_fp8_branch.py [--fp8] [--disable-2cta]`

**Key observations:**
1. **BF16: 2-CTA is ~20% slower than 1-CTA** on B200 (1316 vs 1558 peak). Consistent across all shapes.
2. **FP8: 2-CTA ≈ 1-CTA** (~1960 vs ~1940-1968, within noise). 2-CTA edges out at medium seqlen, 1-CTA at very long seqlen.
3. **FP8 peak: ~1960 TFLOPS** — matches their claimed ~1950.
4. **Our FP4 1-CTA: 1792 TFLOPS peak** — 9% behind FP8's ~1960.
5. **Our FP4 vs their BF16 1-CTA**: Our FP4 is ~15% faster (1792 vs 1558).

**Critical question: why is our FP4 kernel (1792) slower than their FP8 (1960)?**
- Both use reduced-precision GEMM for QK (FP4 block-scaled vs FP8 standard)
- FP8 does NOT quantize P — it stays in higher precision for PV GEMM
- Our FP4 also does NOT quantize P currently (quant code commented out) — PV uses BF16 MMA
- So the only delta should be QK GEMM: FP4 block-scaled vs FP8 standard

#### Deep dive: FP4 vs FP8 GEMM performance gap

**Pipeline depth**: Both use kv_stage=4 for hdim=128. NOT the issue.
- Upstream FP8: Q=32KB (FP8) + O=64KB → KV/stage=32KB → (224-96)/32 = 4
- Our FP4: Q=16KB (FP4) + O=64KB + SF=6KB → KV/stage=32KB (V=BF16 dominates, K aliases V) → 4

**ROOT CAUSE: ~5-9% from outdated base kernel + 0.5-6% from FP4 block-scaled MMA overhead**

Normalized analysis (subtracting base kernel gap) across all shapes:

| shape | BaseGap | FP4/ourBF16 | FP8/upBF16 | NormFP4 | NormGap |
|---------|---------|------------|------------|---------|---------|
| 32x512  | 6.2%    | 1.07x      | 1.14x      | 1053    | **6.1%** |
| 16x1024 | 6.8%    | 1.10x      | 1.17x      | 1347    | **5.8%** |
| 8x2048  | 6.5%    | 1.14x      | 1.19x      | 1570    | **4.4%** |
| 4x4096  | 5.1%    | 1.18x      | 1.24x      | 1720    | **4.8%** |
| 2x8192  | 4.4%    | 1.20x      | 1.26x      | 1807    | **5.0%** |
| 1x16384 | 5.7%    | 1.21x      | 1.25x      | 1877    | **3.3%** |
| 1x32768 | 9.2%    | 1.30x      | 1.31x      | 1957    | **0.5%** |
| 4x8192  | 5.3%    | 1.21x      | 1.27x      | 1844    | **4.7%** |

- **BaseGap** = upstream BF16 / our BF16 (how much faster upstream's base kernel is)
- **NormFP4** = our FP4 × (upstreamBF16 / ourBF16) — what our FP4 would achieve on upstream's base
- **NormGap** = how much FP8 still beats normalized FP4 (actual instruction overhead)

Our BF16 re-run numbers: 925, 1146, 1291, 1386, 1446, 1474, 1377, 1446 TFLOPS.

**Pattern**: NormGap is **larger at small seqlen (6%) and shrinks at large seqlen (0.5%)**. This is expected: at small seqlen the kernel is more GEMM-bound (block-scaled MMA overhead matters), at large seqlen it's more memory/softmax-bound (GEMM difference masked).

The **base kernel gap (~6%)** comes from upstream optimizations. Confirmed by direct
benchmarking on 2026-05-08 (commit `cf3b58d8` vs upstream `09aa3222`):

| Shape | Upstream 1-CTA (TF) | Our BF16 (TF) | Gap |
|---|---|---|---|
| (2,8192,24,128) | 1502 | 1410 | −6.1% |
| (1,16384,24,128) | 1460 | 1381 | −5.4% |
| (1,32768,24,128) | 1403 | 1317 | −6.1% |

Key upstream optimizations responsible:
- `split_P_arrive` — softmax warps release P at 75%, MMA starts PV early
- Per-warp named barriers — finer-grained softmax↔correction sync
- `gemm_ptx_precomputed_varname` — pre-computed SMEM descriptors for QK gemm
- Pipeline abstractions — structured producer/consumer with phase tracking
- Register tuning table — per-config (2CTA/causal/hdim/arch) register allocation

**Note**: Piecemeal porting (register tuning + precomputed descriptors alone) showed
**no improvement** (1334 vs 1340 TF). The gains come from the full pipeline restructure
(split_P_arrive, named barriers, pipeline abstractions) working together.

**Rebase approach**: `fp4-rebase` branch (pushed to `org/fp4-rebase`, commit `ef84b115`)
is based on `public/main` (`09aa3222`) with FP4 functions appended to upstream helper files.

| Config | (1,32768,24,128) | vs old branch |
|---|---|---|
| Old branch BF16 | 1317 TF | baseline |
| Rebase BF16 | 1407 TF | **+6.8%** |
| Old branch NVFP4+BF16 | 1710 TF | — |
| Rebase NVFP4+BF16 | 1451 TF | **−15.2%** |

BF16 forward matches upstream (+6.8%). **FP4 kernel regressed 15%** — root cause
identified via NCU profiling (commit `ef84b115` vs `cf3b58d8`, 2026-05-09):

**NCU stall breakdown (NVFP4+BF16, 1,32768,24,128):**

| metric | old branch | rebase | delta |
|---|---|---|---|
| Duration | 11.93 ms | 14.75 ms | +23.6% |
| SM Active Cycles | 13.22M | 16.39M | +24.0% |
| Instructions/Scheduler | 5,197K | 5,185K | −0.2% |
| stall_long_sb | 262,140 | **399,098** | **+52%** |
| stall_mio | 15,423 | 3,288 | −79% |
| stall_wait | 112,548 | 112,809 | +0.2% |
| SASS instruction count | 3,440 | 3,416 | −0.7% |

**Root cause: ptxas scheduling artifact from upstream helpers.**
Same FP4 kernel code (`flash_fwd_sm100_fp4.py`) produces near-identical SASS
(same opcode counts, same MMA instructions, same barrier counts), but ptxas
schedules instructions differently due to the changed helper function code paths
(upstream `softmax.py`, `mask.py`, `blackwell_helpers.py`). This causes:
- Softmax warp spins 2.4× longer on the S-ready barrier (59K→142K stall_long_sb
  on the `SYNCS.PHASECHK + BRA` spin loop)
- Total stall samples +27% despite identical instruction count
- MIO stalls actually DECREASED (15K→3K) because the upstream softmax uses
  `cute.math.exp2(fastmath=True)` vs our `cute.arch.exp2()` — but the cycle
  savings from reduced MIO are overwhelmed by the long_sb increase

**Detailed NCU comparison:**
Both cubins have near-identical SASS (3440 vs 3416 instructions, same opcode counts:
257 MUFU.EX2, 256 F2FP.BF16.F32.PACK_AB, 24 STTM.x16, 8 LDTM.x32, 27 SYNCS.ARRIVE,
84 SYNCS.PHASECHK, 31 SYNCS.EXCH, 8 UTCOMMA.4X). MUFU burst patterns are also nearly
identical (max burst 32/32, avg 1.9/1.9, avg gap 1.7/1.7). The regression is entirely
from **cross-warp synchronization latency** — the softmax warp's PHASECHK+BRA spin loop
at the end of each iteration shows 142K vs 59K stall_long_sb samples (2.4× longer wait
for S from MMA warp).

**Root cause isolated (2026-05-09):** `cute.math.exp2(x, fastmath=True)` vs `cute.arch.exp2(x)`

Bisection results (swapping one file at a time back to old version):

| swapped file | NVFP4 TF | delta vs rebase (1451) |
|---|---|---|
| baseline rebase | 1450 | — |
| blackwell_helpers.py → old | 1451 | 0% |
| softmax.py: cute.math.exp2 → cute.arch.exp2 | **1716** | **+18.3%** |

The upstream softmax.py uses `cute.math.exp2(x, fastmath=True)` while our old code
uses `cute.arch.exp2(x)`. Both compile to the same MUFU.EX2 SASS instruction, but the
different MLIR IR path causes ptxas to generate different instruction scheduling, increasing
stall_long_sb by 52% in the FP4 kernel.

**Fix applied** (commit `9f9450ca`): replaced all `cute.math.exp2` with `cute.arch.exp2`
in the upstream softmax.py. Results:

| Config | Before fix | After fix |
|---|---|---|
| NVFP4+BF16 (1,32768,24,128) | 1451 TF | **1716 TF** |
| BF16 (1,32768,24,128) | 1427 TF | **1432 TF** |

Both FP4 and BF16 improved. The BF16 kernel is also slightly faster with `cute.arch.exp2`.

**Conclusion**: The raw 9-13% FP4-vs-FP8 gap decomposes into:
1. **~6% stale fork** — closed by rebasing onto upstream (`fp4-rebase` branch)
2. **~0.5-6% actual FP4 block-scaled overhead** — from SF TMA loads + block-scaled MMA instruction; larger at small seqlen (GEMM-bound), negligible at large seqlen (memory-bound)

---

- [x] **2. Debug: why does removing all quant code still give reasonable diffs?**
  - With quant code removed, FP4 vs BF16 ref: max_diff=0.1943, mean_diff=0.020350
  - P scale factors (SFP) should be polluted garbage — gemm_Pi would read FP4 from BF16-formatted data

### Task 2 Findings

**The output IS wrong — but bounded, not NaN/garbage.** Here's why:

**Code flow with P quant commented out (quant_pv=True):**
1. `tSrS_t2r` = P values loaded from TMEM (correct: exp2 of QK scores)
2. `apply_exp2_convert(tSrS_t2r)` — modifies S in-place, but does NOT write to `tSrP_r2t`
3. Lines 3200-3212 COMMENTED OUT — no `_quant_fp4`, no `scale_groupwise`, no SFP R2S copy
4. `tSrP_r2t_f32` (line 3185) = **uninitialized register fragment** — contains leftover register values
5. Line 3224-3225: copies `tSrP_r2t_f32` (garbage) to TMEM for PV MMA
6. sSFP (SMEM) = **uninitialized** — no R2S copy of scale factors was executed
7. Block-scaled PV MMA reads garbage P data + garbage SFP scale factors + valid FP4 V + valid SFV

**Why the output is "reasonable" (not NaN/random):**
- **FP4 E2M1 is bounded**: Only 16 possible values per nibble: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. Even random bit patterns produce finite values in [-6, 6].
- **E4M3 scale factors are bounded**: Range [~2e-10, 448]. Even garbage bytes produce finite scale values.
- **Product P_fp4 × SFP is bounded**: At most 6 × 448 = 2688. The PV gemm accumulates these into float32 accumulators — large but finite.
- **Softmax normalization renormalizes**: The correction loop divides O by row_sum. Even with garbage P weights, the output is a (wrong) weighted average of V values. Since V values are real data, the output stays in the same magnitude range.
- **max_diff=0.19 is actually large**: For outputs typically in [-0.1, 0.1], this is ~2x the output magnitude. The output is WRONG but BOUNDED.

**Verification**: The `force_fp4_impl bf16 test` (which runs quant_pv=False with BF16 P, same FP4 QK) shows exact element-wise matches — confirming the FP4 QK path is correct and the diffs come specifically from the garbage PV block-scaled MMA.

**Conclusion**: The "reasonable" diffs are an artifact of FP4 bounded arithmetic + softmax normalization. The output is numerically wrong but doesn't produce NaN/inf. This is expected behavior for garbage-in-bounded-out MMA pipelines.

- [x] **3. Integrate 2-CTA gemm instructions from FA4 main branch** (superseded — 2-CTA is slower on B200)
  - **UPDATE**: 2-CTA is slower on our B200 hardware (see Task 1 benchmarks above)
  - Main FA4 (flash_fwd_sm100.py) has full 2-CTA support on `public/main` (20+ commits ahead)
  - PR #2109 does NOT switch CTA groups — upstream already has it
  - Upstream `use_2cta_instrs` enabled for hdim∈{128,192}, non-causal, non-split-kv
  - Our flash_fwd_sm100_fp4.py has skeleton at line 885-886 but asserts False

### Task 3 Analysis

**2-CTA is SLOWER on our B200** (see Task 1 benchmarks). Upstream BF16 1-CTA (peak 1557) > 2-CTA (peak 1315). The FP8 PR reaches 1952 TFLOPS because FP8 has 2x MMA throughput which more than compensates for the 2-CTA overhead.

**Integration scope is massive** — 20+ components need changes:
- MMA tiler M doubles (128→256)
- Cluster shape (1,1)→(2,1), SMEM KV per stage halved (peer CTA has other half)
- TMA multicast to both CTAs, TMA copy bytes doubled
- Grid scheduling: cluster_idx vs block_idx, m_tile_idx recomputed
- mma_tile_coord_v = bidx % thr_id.shape for each CTA's row ownership
- TmemAllocator needs is_two_cta + tmem_dealloc_mbar_ptr in SharedStorage
- Cluster cooperative groups with doubled thread counts
- pipeline_init_arrive/wait with cluster shape
- gO partitioning: flat_divide by mma_tiler_pv[0]//cta_group_size
- get_tmem_load_op: use_2cta_instrs=True
- All gemm_ptx_* already wired for cta_group param (just needs value=2)
- Block-scaled SFB instruction shape: M halved per CTA (already coded at line 1172)

**Recommendation**: Skip 2-CTA integration. Focus on porting other upstream optimizations (pipeline improvements, CLC scheduler) which may explain the 1475→1557 TFLOPS gap between our BF16 and upstream 1-CTA BF16.

  - **Commit**: N/A (not implemented)
  - **Command**: N/A

- [x] **4. Remove duplicated softmax classes from FP4 kernel**
  - FP4 kernel (`flash_fwd_sm100_fp4.py`) had its own copy of `Softmax` (line 110) and `SoftmaxSm100` (line 254)
  - Main softmax lives in `softmax.py` with identical `Softmax` base class and `SoftmaxSm100`

### Task 4 Findings & Changes

**Removed 437 lines** from `flash_fwd_sm100_fp4.py`:
- Local `mul_packed_f32x2`, `add_packed_f32x2` helpers (used `calc_packed_f32x2_op` without explicit RN rounding)
- Local `fadd_reduce` function (identical logic to `utils.fadd_reduce`)
- Full `Softmax` base class copy (identical to `softmax.py`)
- Full `SoftmaxSm100` copy with FP4-specific extras

**Added to `softmax.py`**:
- `scale_groupwise()` method on `SoftmaxSm100` — normalizes P by per-group max before FP4 quantization (was in FP4 copy, currently commented out in usage but needed for future quant_pv work)

**Changed imports in `flash_fwd_sm100_fp4.py`**:
- `from flash_attn.cute.softmax import apply_score_mod_inner` → `from flash_attn.cute.softmax import SoftmaxSm100, apply_score_mod_inner`

**Analysis of removed methods**:
- `scale_groupwise`: Moved to softmax.py (commented out in usage at line 3202 but design-required)
- `update_row_sum_sage`: Already exists in softmax.py (slightly different signature — FP4 had extra `group_max_layout` param, but this method was never called)
- `apply_sage_sp1`: Dead code (referenced `self.sp1_scale` which isn't in the dataclass), not moved
- `fadd_reduce`: Identical to `utils.fadd_reduce`, used via `Softmax._compute_row_sum` which calls `utils.fadd_reduce` in the softmax.py version

**Rounding difference**: The local `mul_packed_f32x2`/`add_packed_f32x2` used default rounding (RZ) while `utils.*` versions use RN. This difference is negligible for softmax computation and both BF16 and FP4 paths produce identical numerical results.

**Test results** (no regressions):
- BF16 FA4: OK
- FP4 (quant_qk): max_diff=0.0295, mean_diff=0.001606 (identical to before)
- FP4 (quant_v): max_diff=0.0718, mean_diff=0.007285, 1869 TFLOPS (identical to before)
- **Command**: `CUTE_DSL_ENABLE_TVM_FFI=1 python -m flash_attn.cute.benchmarks.bench_fp4 --quant_v`

---

## AC-5: Group-128 V Scale Shootout

Two approaches for block-scaled V with per-128 group scale factors:

**(a) Block-scaled PV MMA** (pre-expand group-128 → group-32 SFV for hardware): Requires additional SF TMA loads, S2T copies, and block-scaled MMA infrastructure for PV gemm. Extra SMEM for SFV buffer.

**(b) Softmax-scale fusion** (plain FP8/BF16 V, fold V descale into softmax_scale): No extra infrastructure. V stays in standard dtype. Per-head V descale multiplied into softmax_scale during online softmax normalization.

### Results (bench_fp4.py, B200 GPU1)

| Approach | Mode | Peak TFLOPS | cos_sim | max_diff |
|----------|------|------------|---------|----------|
| **(b) Plain FP8 V** | NVFP4+FP8 | **1948** | **0.990** | 0.043 |
| **(b) Plain BF16 V** | NVFP4+BF16 | **1809** | **0.991** | 0.045 |
| (a) Block-scaled FP4 V | NVFP4+FP4 | 1261 | 0.790 | 0.112 |

**Winner: Approach (b) — softmax-scale fusion.** ~55% faster and much more precise (cos 0.99 vs 0.79) than block-scaled FP4 PV. Block-scaled PV MMA adds ~40% overhead from SF TMA loads and S2T copies while degrading precision from double FP4 quantization.

Note: NVFP4 cos 0.990-0.991 uses flashinfer adaptive per-block SF (amax/6). Previous table showed 0.976 due to uniform SF=1.0 (no dynamic range adaptation). TFLOPS measured on shared GPU (lower bound).

---

## Precision: All Mixed-Precision Modes vs BF16 Reference

Commit `e7d6527e` (fp4-rebase), cutlass-dsl 4.4.2, B200 sm_100a.
NVFP4 uses flashinfer `nvfp4_quantize` (adaptive per-block SF = amax/6 per 16-elem block).
MXFP8 uses torch-native FP8 + uniform E8M0 SF=1.0.
Command: `PYTHONPATH=$(pwd) python -m flash_attn.cute.benchmarks.bench_fp4 --qk_mode <mode> --pv_mode <pv>`

| Config (b,s,h,d) | NVFP4+BF16 cos | NVFP4+FP8 cos | MXFP8+FP8 cos |
|------------------|----------------|---------------|---------------|
| (1,256,16,128) | 0.9910 | 0.9904 | 0.9986 |
| (1,1024,16,128) | 0.9908 | 0.9901 | 0.9986 |
| (4,4096,16,128) | 0.9906 | 0.9899 | 0.9985 |
| (1,32768,16,128) | 0.9904 | 0.9897 | 0.9985 |
| (4,4096,32,128) | 0.9905 | 0.9898 | 0.9985 |
| (1,4096,12,128) | 0.9906 | 0.9899 | 0.9985 |
| (1,32768,12,128) | 0.9903 | 0.9896 | 0.9985 |
| (1,4096,24,128) | 0.9905 | 0.9898 | 0.9985 |
| (1,32768,24,128) | 0.9905 | 0.9899 | 0.9985 |
| (1,32768,24,64) | 0.9899 | 0.9892 | — |

NVFP4 cos ~0.990 flat across all shapes (adaptive per-block SF). Previous 0.976 was from
uniform SF=1.0 quantization (no dynamic range adaptation). MXFP8 cos ~0.9985 (uniform
E8M0 SF=1.0 → effectively plain FP8 GEMM; with adaptive SF expect ~0.999+).

TFLOPS (B200, shared GPU — lower bound due to thermal/contention):
| Mode | Peak TFLOPS | Shape |
|------|------------|-------|
| NVFP4+BF16 | 1809 | (1,32768,16,128) |
| NVFP4+FP8 | **1948** | (1,32768,24,128) |
| MXFP8+FP8 | 1881 | (1,32768,24,128) |
| BF16 ref | 1542 | (1,32768,16,128) |

- **Commit**: current fp4 branch, **GPU**: B200
- **Command**: `CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ENABLE_TVM_FFI=1 python -m flash_attn.cute.benchmarks.bench_fp4 --qk_mode {nvfp4,mxfp8} --pv_mode {bf16,fp8}`

---

## Full benchmark: fp4-rebase vs old branch (b,s,h=24,d=128, non-causal, do_bench, B200)

| mode | shape | old (TF) | rebase (TF) | delta |
|---|---|---|---|---|
| bf16 | (4,4096) | 1354 | 1444 | **+6.6%** |
| bf16 | (2,8192) | 1392 | 1474 | **+5.9%** |
| bf16 | (1,16384) | 1411 | 1497 | **+6.1%** |
| bf16 | (1,32768) | 1332 | 1559 | **+17.0%** |
| nvfp4_bf16 | (32,512) | 992 | 1261 | **+27.1%** |
| nvfp4_bf16 | (16,1024) | 1261 | 1516 | **+20.2%** |
| nvfp4_bf16 | (8,2048) | 1474 | 1654 | **+12.2%** |
| nvfp4_bf16 | (4,4096) | 1636 | 1735 | **+6.1%** |
| nvfp4_bf16 | (2,8192) | 1731 | 1779 | **+2.8%** |
| nvfp4_bf16 | (1,16384) | 1777 | 1806 | **+1.6%** |
| nvfp4_bf16 | (1,32768) | 1792 | 1918 | **+7.0%** |
| nvfp4_bf16 | (4,8192) | 1752 | 1880 | **+7.3%** |
| nvfp4_fp8 | (4,4096) | 1605 | 1823 | **+13.6%** |
| nvfp4_fp8 | (2,8192) | 1646 | 1871 | **+13.7%** |
| nvfp4_fp8 | (1,16384) | 1665 | 1897 | **+13.9%** |
| nvfp4_fp8 | (1,32768) | 1764 | 2016 | **+14.3%** |

**All shapes, all modes improved. Zero regressions.**

Fixes applied:
1. `cute.arch.exp2` instead of `cute.math.exp2` — fixed ptxas scheduling
2. Register tuning 216/48/24 → 192/80/48 — reduced correction warp spills
3. `_FP4_TUNING_CONFIG` with `enable_e2e=True, e2e_freq=16` for BF16 PV
4. `_FP4_FP8PV_TUNING_CONFIG` with `e2e_freq=9` for FP8 PV

---

## Final tuned results on fp4-rebase (2026-05-09)

Commit `a21acbe7`. Three fixes applied to fp4 kernel:
1. **`cute.arch.exp2`** instead of `cute.math.exp2` in upstream softmax — fixes ptxas scheduling
2. **Register tuning 216/48/24 → 192/80/48** — gives correction warps enough registers
3. **`_FP4_TUNING_CONFIG` table** with per-config e2e + register settings:
   - NVFP4+BF16: e2e_freq=16, start_frg=1, regs=192/80
   - NVFP4+FP8:  e2e_freq=9,  start_frg=0, regs=192/80

**Full results (bench_fp4.py, triton do_bench, B200):**

| shape | NVFP4+BF16 | NVFP4+FP8 | MXFP8+FP8 | BF16 ref |
|---|---|---|---|---|
| (1,256,16,128) | 34 | 39 | 40 | 35 |
| (1,1024,16,128) | 418 | 416 | 414 | 380 |
| (4,4096,16,128) | 1789 | 1875 | 1801 | 1479 |
| (1,32768,16,128) | 1920 | 2016 | 1942 | 1543 |
| (4,4096,32,128) | 1826 | 1920 | 1851 | 1471 |
| (1,4096,12,128) | 1081 | 1118 | 1070 | 940 |
| (1,32768,12,128) | 1820 | 1913 | 1846 | 1508 |
| (1,4096,24,128) | 1481 | 1548 | 1480 | 1274 |
| **(1,32768,24,128)** | **1887** | **2018** | **1948** | 1545 |
| (1,32768,24,64) | 919 | 986 | — | 949 |

h24 full shape sweep (NVFP4+BF16):

| shape | old (TF) | rebase (TF) | delta |
|---|---|---|---|
| (32,512,24,128) | 992 | 1261 | +27.1% |
| (16,1024,24,128) | 1261 | 1516 | +20.2% |
| (8,2048,24,128) | 1474 | 1654 | +12.2% |
| (4,4096,24,128) | 1636 | 1735 | +6.1% |
| (2,8192,24,128) | 1731 | 1779 | +2.8% |
| (1,16384,24,128) | 1777 | 1806 | +1.6% |
| (1,32768,24,128) | 1792 | 1918 | +7.0% |
| (4,8192,24,128) | 1752 | 1880 | +7.3% |

**All shapes improved. Zero regressions.** NVFP4+BF16 e2e=16 via `_FP4_TUNING_CONFIG`.
NVFP4+FP8 e2e=9 via `_FP4_FP8PV_TUNING_CONFIG`.

**Config-based e2e_freq sweep (editing _FP4_TUNING_CONFIG, all shapes, do_bench):**

| freq | NVFP4+BF16 avg (TF) | NVFP4+FP8 avg (TF) |
|---|---|---|
| off | 1388 | 1693 |
| 8 | 1247 | 1443 |
| **9** | 1265 | **1776** |
| 10 | 1210 | 1564 |
| 12 | 1615 | 1465 |
| **16** | **1683** | 1555 |

Current config (freq=16 BF16, freq=9 FP8) is optimal. Per-(nheads, seqlen_range)
tuning is not needed — same freq is best across all shapes within each mode.

**Note**: env var sweep (`FA4_FORCE_E2E=1`) produces different results than
config-based (`enable_e2e: True`) because they compile different kernels.
Always tune via config dict + bench_fp4.py.

**Command**: `CUDA_VISIBLE_DEVICES=N CUTE_DSL_ENABLE_TVM_FFI=1 PYTHONPATH=$(pwd) .venv/bin/python -m flash_attn.cute.benchmarks.bench_fp4 --qk_mode nvfp4 --pv_mode {bf16|fp8}`

### Remaining upstream features not ported

- **CLC scheduler**: disabled for dense non-causal (our main benchmark).
  Only relevant for causal + varlen. Would require 39+ code changes across
  shared storage, warp dispatch, and tile scheduling. Skipped.
- **Named barriers**: refactor of raw mbarrier offsets into NamedBarrier
  abstraction. Cosmetic improvement, no perf impact for 1-CTA. Skipped.
- **split_P_arrive**: already present as `mbar_p_split` / `pre_mbar_tiles`.

