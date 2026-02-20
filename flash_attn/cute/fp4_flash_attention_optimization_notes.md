# FP4 Flash Attention Optimization Debug Notes

## Overview

This document summarizes a debugging session investigating performance anomalies in a Blackwell FP4 quantized Flash Attention kernel. The kernel uses CUDA with CuTe DSL.

**Key Files:**
- `/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/flash_fwd_sm100_fp4.py` - Main kernel implementation
- SASS dumps: `mult.sass`, `no_mult.sass`, `no_mult_no_sync_warp.sass`

**Kernel Warp Architecture:**
```
Warps 0-7:   Softmax warps - read S from TMEM → compute softmax → _quant_fp4 → write P_fp4 to TMEM
Warps 8-11:  Correction warps - O rescaling when row_max changes
Warp 12:     MMA warp - reads P_fp4 from TMEM, V from SMEM → gemm_Pi
Warp 13:     Epilogue warp
Warp 14:     Load warp (TMA)
Warp 15:     Idle
```

**Profiling Command:**
```bash
CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_LINEINFO=1 CUTE_DSL_ENABLE_TVM_FFI=1 CUDA_VISIBLE_DEVICES=1 \
ncu --set full -f \
    --kernel-id ::regex:".*fp4FlashAttention.*":1 \
    -o <output>.ncu-rep \
    python benchmarks/bench_fp4.py --quant_v
```

---

## Level 1: Initial Symptom

**Observation:** In `update_row_sum_sage()` function, removing `mul_packed_f32x2` and using direct stores causes a **15% TFLOPs drop**, despite reducing instruction count.

**Code Paths:**
```python
# With mult (FAST):
acc_S_group_sum[i], acc_S_group_sum[i + 1] = mul_packed_f32x2(
    (self._compute_row_sum(...), self._compute_row_sum(...)),
    (acc_S_row_group_max_exp[i], acc_S_row_group_max_exp[i + 1]),
)

# Without mult (SLOW):
acc_S_group_sum[i] = self._compute_row_sum(...)
acc_S_group_sum[i + 1] = self._compute_row_sum(...)
cute.arch.sync_warp()  # Added to try to fix perf
```

**Initial Hypothesis:** Latency hiding - the multiply provides work while MUFU.EX2 (exp2) operations complete.

---

## Level 2: NCU Analysis - Misleading Indicators

**NCU Warp Stall Sampling showed:**
- With mult path: 100% Floating Point instructions
- Without mult path: 50% Uniform Datapath, 16.67% Load/Store, 16.67% Control

**Initial Interpretation:** Suspected different code generation, spilling, or control flow divergence.

**Key Insight:** NCU sample attribution can be misleading. The instruction mix percentages are for the *sampled hotspot region*, not the whole kernel.

**Verification:**
```bash
grep -c "FADD\|FMUL\|FFMA" mult.sass      # 553
grep -c "FADD\|FMUL\|FFMA" no_mult.sass   # 550

grep -c "BRA" mult.sass                    # 174
grep -c "BRA" no_mult.sass                 # 174

grep -c "SYNCS" mult.sass                  # 237
grep -c "SYNCS" no_mult.sass               # 237
```

**Conclusion:** Overall instruction counts are nearly identical. The difference is in **instruction scheduling/ordering**, not code generation.

---

## Level 2.5: Warp Stall Analysis

NCU provides warp stall sampling that shows *why* warps are waiting. Key metrics from the `smsp__pcsamp_warps_issue_stalled_*` family:

### Key Stall Reasons

| Stall Reason | Meaning | Typical Cause |
|-------------|---------|---------------|
| `stall_long_scoreboard` | Waiting for L2/DRAM/TMEM data | Memory latency not hidden |
| `stall_barrier` | Waiting at barrier (mbarrier, syncthreads) | Producer-consumer imbalance |
| `stall_math_pipe_throttle` | Math pipe full | Back-to-back math ops |
| `stall_mio_throttle` | Memory I/O throttled | Too many outstanding loads |
| `stall_short_scoreboard` | Waiting for L1/shared data | Short latency not hidden |
| `stall_wait` | Waiting for async operation | TMA/WGMMA completion |

### NCU Commands for Stall Analysis

```bash
# Get warp stall breakdown
ncu --import file.ncu-rep --page raw --csv 2>/dev/null | \
    grep -E "smsp__pcsamp_warps_issue_stalled" | head -20

# Key metrics to look for:
# smsp__pcsamp_warps_issue_stalled_long_scoreboard
# smsp__pcsamp_warps_issue_stalled_barrier
# smsp__pcsamp_warps_issue_stalled_wait
# smsp__pcsamp_warps_issue_stalled_math_pipe_throttle

# Get per-PC stall samples (source-correlated)
ncu --import file.ncu-rep --page source --csv > source_stalls.csv
```

### FP4 Flash Attention Stall Findings

**Quantization overhead (FP4 vs non-quant):**
| Metric | Non-Quant | FP4 Quant | Change |
|--------|-----------|-----------|--------|
| `stall_long_sb` (MMA warp) | 0.30% | 2.47% | **8x increase** |
| `stall_barrier` | baseline | elevated | MMA waiting for softmax+quant |

**Interpretation:** The MMA warp spends 8x more time waiting on long scoreboard when quantization is enabled. This indicates the quantization (`F2FP.SATFINITE.E2M1` instructions) is serialized in the critical path between softmax completion and MMA consumption.

**mul_packed_f32x2 removal effect:**
| Metric | With mult | Without mult | Interpretation |
|--------|-----------|--------------|----------------|
| `stall_long_scoreboard` | lower | **higher** | MUFU.EX2 latency exposed |
| Sampled instruction mix | 100% FP | 50% Uniform | Different hot region |

**Why `stall_long_scoreboard` increases without the multiply:**
1. `MUFU.EX2` (exp2) has ~20 cycle latency
2. With multiply: `FMUL2` → `FFMA2` chain provides work while `MUFU.EX2` completes
3. Without multiply: tight `FADD2` chain immediately depends on `MUFU.EX2` result
4. Result: warps stall waiting for exp2 to complete

### How to Use Stall Data

```python
def analyze_stalls(ncu_csv_path):
    """Extract and interpret warp stall metrics."""
    import pandas as pd
    
    df = pd.read_csv(ncu_csv_path)
    
    stall_cols = [c for c in df.columns if 'pcsamp_warps_issue_stalled' in c]
    
    # Key ratios to check
    long_sb = df['smsp__pcsamp_warps_issue_stalled_long_scoreboard'].iloc[0]
    barrier = df['smsp__pcsamp_warps_issue_stalled_barrier'].iloc[0]
    math_throttle = df['smsp__pcsamp_warps_issue_stalled_math_pipe_throttle'].iloc[0]
    
    print(f"Long scoreboard: {long_sb} (memory latency)")
    print(f"Barrier: {barrier} (sync overhead)")
    print(f"Math throttle: {math_throttle} (compute bound)")
    
    # Diagnosis
    if long_sb > 30:
        print("→ Memory latency dominant - need better prefetching or latency hiding")
    if barrier > 20:
        print("→ Barrier overhead high - check producer-consumer balance")
    if math_throttle > 40:
        print("→ Compute bound - good utilization but check ILP")
```

### Stall Sampling Caveats

1. **Sampling bias:** Stalls are sampled, not measured. High-frequency short stalls may be underrepresented.

2. **Attribution ambiguity:** A stall at instruction X may be caused by a dependency from instruction X-10. The stalled instruction isn't necessarily the problem.

3. **Warp-level aggregation:** Different warps may have different stall profiles. Per-warp breakdown requires `--set full` profiling.

4. **Instruction mix vs stalls:** The "instruction mix" in NCU warp state view shows what instruction *type* was sampled when stalled, not what caused the stall.

5. **`stall_long_sb` at mbarrier_wait:** You might expect `stall_barrier` at mbarrier instructions, but NCU often shows `stall_long_sb` instead. This is because:
   - `SYNCS.PHASECHK.TRYWAIT` (the mbarrier try_wait) polls barrier state in shared memory
   - Each poll iteration shows as `long_sb` (shared memory access latency)
   - The classification doesn't change the diagnosis - warps are waiting for producers to signal
   
   In FP4 FA, the 8x increase (0.30% → 2.47%) reflects MMA waiting longer for softmax+quant to produce P, regardless of the stall classification.

### Critical Path Analysis

The quantization adds to the critical path between softmax and MMA:

**Without quant:**
```
compute P_f32 → convert to fp16 → write TMEM → signal barrier
```

**With quant (slower):**
```
compute P_f32 → quant to P_fp4 → write TMEM → signal barrier
         ↑
    F2FP.SATFINITE.E2M1 instructions add latency here
```

The MMA warp's `mbarrier_wait` stalls longer because the quant serializes in the critical path.

---

## Level 3: Barrier Timing Hypothesis

**Observation:** Comparing SASS around the row_sum computation showed different barrier arrive positions:

**mult.sass:**
```
MUFU.EX2 R38, R2
MUFU.EX2 R39, R3
SYNCS.ARRIVE.TRANS64.ART0 [R2+UR5+0xb0], R138   ← EARLY barrier arrive
FENCE.VIEW.ASYNC.T
F2FP.SATFINITE.E2M1...  ← quant AFTER barrier
```

**no_mult.sass:**
```
MUFU.EX2 R64, R2
F2FP.SATFINITE.E2M1...  ← quant BEFORE barrier
... more F2FP ...
NOP × 4  ← sync_warp
FENCE.VIEW.ASYNC.T
SYNCS.ARRIVE.TRANS64.ART0...  ← LATE barrier arrive
```

**Hypothesis:** The mult version allows barrier arrive earlier, letting MMA warp wake up sooner.

**Tested:** Removed `sync_warp()` to see if barrier timing improves.

**Result:** `no_mult_no_sync_warp.sass` showed early barrier arrives (similar to mult), BUT performance was still worse.

**Additional finding:** Adding `sync_warp()` reduced the overhead from 15% to 9% - it helps by creating an explicit synchronization point, but doesn't fix the fundamental issue.

**Conclusion:** Barrier timing is NOT the root cause.

---

## Level 4: Root Cause - Instruction Reordering (Not Just FFMA2 vs FADD2)

**Key Discovery:** The `mul_packed_f32x2` changes the **entire instruction scheduling**, not just the final reduction.

### Visible Symptom: Final Reduction Difference

**mult.sass (0x7ca0-0x7d00):**
```asm
FADD R9, R6, R7
FMUL2.FTZ.RZ R2, R12.F32x2.HI_LO, R78.F32x2.HI_LO   ; mult by group_max_exp
FMUL2.FTZ.RZ R6, R4.F32x2.HI_LO, R72.F32x2.HI_LO   ; mult by group_max_exp
FFMA2.FTZ.RZ R4, R14, R76, R2   ; fused multiply-add
FFMA2.FTZ.RZ R2, R8, R30, R6    ; fused multiply-add
FADD2.FTZ.RZ R4, R4, R2
FADD R140, R4, R5
```

**no_mult_no_sync_warp.sass (0x7bb0-0x7c00):**
```asm
FADD R3, R8, R9
FADD2.FTZ.RZ R4, R14, R12
FADD2.FTZ.RZ R2, R6, R2
FADD2.FTZ.RZ R4, R4, R2
FADD R140, R4, R5
```

### Deeper Issue: Global Scheduling Differences

Diffing the SASS files reveals **pervasive differences** beyond the final reduction:

1. **Barrier placement differs:**
   - mult.sass: `SYNCS.ARRIVE` at 0x7290
   - no_mult.sass: `SYNCS.ARRIVE` at 0x7490

2. **MUFU.EX2 and F2FP interleaving:**
   - mult version interleaves more F2FP quant instructions between MUFU.EX2 pairs
   - Different latency hiding patterns

3. **Register assignments completely different:**
   - Same logical values use different physical registers (R80/R81 vs R62/R63)
   - Affects what operations can be reordered

4. **Reduction tree structure differs:**
   - Different FADD2 accumulation order
   - Different intermediate register usage

**Root Cause:**
The `mul_packed_f32x2` creates a different **dependency graph** that causes NVCC to schedule the entire function differently. The FFMA2 vs FADD2 difference is the most visible symptom, but the performance impact comes from the cumulative effect of different scheduling throughout the hot loop.

**Why FFMA2 helps:**
1. Hides latency - multiply and add in parallel pipeline stages
2. Better ILP - two independent FMUL2 issue, then two independent FFMA2 consume results
3. Same throughput as FADD2, so "extra work" is essentially free
4. **Creates scheduling constraints** that paradoxically lead to better overall code

---

## Level 5: Potential Solutions

### ~~Solution 1: Keep the Multiply~~ (DISPROVEN)
Multiplying by `Float32(1.0)` to preserve the `mul_packed_f32x2` dependency graph does NOT work — the compiler optimizes away the constant multiply, producing the same SASS and same performance as the no-mult path.

### Solution 1: Use Flat Reduction for Non-Quant Path
For non-quant path, use `update_row_sum()` instead of `update_row_sum_sage()` - it does a single flat reduction without the hierarchical structure that causes the scheduling issue.

### Solution 2: Explicit Scheduling Control
Use PTX inline assembly or compiler pragmas to force specific instruction ordering. This is fragile but gives direct control:

```python
# Example: force barrier arrive before certain operations
cute.arch.fence_view_async_tmem_store()
cute.arch.mbarrier_arrive(...)  # arrive early
# Then continue with remaining computation
```

### Solution 3: Investigate Compiler Flags
Try different `--ptxas-options` to see if scheduling behavior changes:
```bash
--ptxas-options=-v                    # verbose register info
--ptxas-options=-dlcm=ca              # cache all loads
--ptxas-options=-allow-expensive-optimizations=true
```

---

## Key Learnings

1. **"Less work" doesn't mean faster** - Removing operations can hurt performance if it removes latency-hiding opportunities or changes compiler scheduling decisions.

2. **Compiler scheduling is holistic** - A small code change (adding/removing a multiply) can cause NVCC to reschedule the *entire* function differently, affecting barrier placement, register allocation, and instruction interleaving.

3. **NCU sampling attribution is tricky** - Instruction mix percentages in sampled regions can be misleading. Always verify with full SASS analysis using `cuobjdump -sass` and diff comparisons.

4. **FFMA2 vs FADD2** - On Blackwell, fused multiply-add has same throughput as plain add, so the multiply is "free" and provides ILP benefits.

5. **Dependency chains matter** - A tight chain of FADD2 instructions serializes execution. Breaking it up with independent FMUL2/FFMA2 pairs improves ILP.

6. **sync_warp() can hurt** - It forces serialization and prevents compiler reordering optimizations, but removing it doesn't fix fundamental scheduling issues caused by different dependency graphs.

7. **Diff SASS files to find root cause** - When performance differs unexpectedly, diff the SASS output to see how instruction scheduling changed, not just what instructions are present.

8. **Warp stall metrics reveal latency issues** - `stall_long_scoreboard` increasing indicates exposed memory/MUFU latency. The multiply operation hides MUFU.EX2's ~20 cycle latency; without it, the tight FADD2 chain stalls waiting for exp2 results.

9. **Stall attribution is indirect** - A warp stalled at instruction X doesn't mean X is slow; it means X is waiting for a dependency. Trace back through the dependency chain to find the true cause.

---

## TODOs

1. **Fix no-mult path performance degradation:** The no-mult path (using plain stores instead of `mul_packed_f32x2`) is 15% slower due to compiler scheduling differences (see Level 4). Find a way to preserve the favorable scheduling without relying on the multiply. If fixed, we can compute group max on P after `exp(S)` and compute row sum, saving the cost of rescaling row sum for each group and subtracting S scale factors in `scale_subtract_rowmax`.

2. **Add reg→TMEM copy for P scale factor (`tSrPSF`):** After `_quant_fp4` computes `tSrPSF` (the per-block P scale factor in registers, line 3170), it is never copied to TMEM via `tcgen05.st`. See the `# TODO(wenxuan) tcgen05.st` on line 3171. The MMA warp reads P scale factors from TMEM (`tCtSFPs`) via `tiled_mma_pv.set(tcgen05.Field.SFA, ...)` (line 2534), so it currently uses stale/uninitialized values. Need to add the `tcgen05.st` and ensure precisions match.

3. **Optimize quantization critical path:** NCU showed MMA warp barrier wait increased 8x with quantization enabled. Quantization (`_quant_fp4` at line 3004) is serialized in the softmax path before signaling MMA warp. Try to speed up the quantization path.

---

## Useful Commands

```bash
# Dump SASS (actual cubin filename example)
cuobjdump -sass cutlass___call___flash_attncuteflash_fwd_sm100_fp4FlashAttentionForwardSm100_object_at__Tensorgmemodiv32i64div32i64i641_Tensorgmemodiv32i64div32i64i641_Tensorgmemodiv32i64div32i64i.sm_100a.cubin > no_mult_no_sync_warp.sass

# Check resource usage
cuobjdump --dump-resource-usage cutlass___call___flash_attncuteflash_fwd_sm100_fp4FlashAttentionForwardSm100_object_at__Tensorgmemodiv32i64div32i64i641_Tensorgmemodiv32i64div32i64i641_Tensorgmemodiv32i64div32i64i.sm_100a.cubin

# Example output:
# Resource usage:
#  Common:
#   GLOBAL:0
#  Function kernel_cutlass_kernel_flash_attn...:
#   REG:128 STACK:32 SHARED:1024 LOCAL:0 CONSTANT[2]:64 CONSTANT[0]:2128 TEXTURE:0 SURFACE:0 SAMPLER:0

# === WARP STALL ANALYSIS ===
# Export all metrics to CSV
ncu --import file.ncu-rep --page raw --csv > metrics.csv

# Extract warp stall breakdown
ncu --import file.ncu-rep --page raw --csv 2>/dev/null | \
    grep -E "smsp__pcsamp_warps_issue_stalled" 

# Key stall metrics to grep for:
grep "stall_long_scoreboard" metrics.csv      # Memory/TMEM latency
grep "stall_barrier" metrics.csv              # Barrier sync overhead
grep "stall_wait" metrics.csv                 # Async op (TMA/WGMMA) wait
grep "stall_math_pipe_throttle" metrics.csv   # Compute saturation

# Get source-correlated stalls (requires CUTE_DSL_LINEINFO=1)
ncu --import file.ncu-rep --page source --csv > source_stalls.csv

# === SASS ANALYSIS ===
# Compare specific regions between SASS files
sed -n '3730,3760p' mult.sass
sed -n '3730,3760p' no_mult.sass

# Diff specific regions to see scheduling differences
sed -n '3600,3800p' mult.sass > /tmp/mult_region.txt
sed -n '3600,3800p' no_mult.sass > /tmp/no_mult_region.txt
diff /tmp/mult_region.txt /tmp/no_mult_region.txt

# Find final reduction (look for FADD R140 which stores row_sum result)
grep -n "FADD R140" mult.sass no_mult.sass

# Look at N lines before the final reduction
sed -n '3950,4010p' mult.sass  # adjust line numbers based on grep output

# Count instruction types
grep -c "FADD\|FMUL\|FFMA" kernel.sass
grep -c "MUFU.EX2" kernel.sass
grep -c "F2FP.SATFINITE" kernel.sass

# Find specific patterns with context
grep -n -A5 -B5 "MUFU.EX2 R64" kernel.sass
grep -n -C20 "MUFU.EX2 R64" kernel.sass   # 20 lines context both sides
grep -C20 "USHF.L.U32 UR4, UR18, 0x1f, URZ" no_mult.sass  # Find specific instruction with context

# Search for instruction patterns across files
grep -n -C20 "SYNCS.ARRIVE.TRANS64.ART0 \[R2+UR5+0xb0\]" mult.sass
grep -n -C20 "F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C R10, R147, R146" no_mult.sass

# Find barrier arrivals to compare timing
grep -n "SYNCS.ARRIVE" mult.sass no_mult.sass
```

---

## SASS-to-Python Mapping

For SASS-to-Python line mapping, per-line register pressure analysis, diagnosing stack/local access, and NCU source-level profiling, use the [perf-optimizations skill](../../skills/perf-optimizations/SKILL.md#sass-to-python-line-mapping). The per-line analysis script is at [`sass_per_line_analysis.py`](../../skills/perf-optimizations/sass_per_line_analysis.py).

Apply these tools to the issues in this document:
- **Scheduling differences** (Level 4, main issue): use SASS-to-Python mapping to identify which Python lines the differently-scheduled SASS instructions originate from, then diff annotated SASS between two builds. The `mul_packed_f32x2` vs plain add changes the entire dependency graph and compiler scheduling.
- **Warp stall correlation** (Level 2.5): use `sass_per_line_analysis.py kernel.cubin --ncu-rep profile.ncu-rep --sort stalls` to see per-Python-line stall samples with dominant stall reason (e.g. `long_sb` for exposed MUFU.EX2 latency, `barrier` for producer-consumer imbalance, `wait` for TMA/WGMMA completion).
- **STACK/LOCAL access**: run `cuobjdump --dump-resource-usage kernel.cubin` to check if STACK or LOCAL is non-zero. If so, use `sass_per_line_analysis.py --sort local` to find which Python lines emit STL/LDL instructions. Check the Python source at those lines: non-static array indexing (e.g. `cutlass.range()` instead of `cutlass.range_constexpr()`) causes stack access; otherwise it's likely register spills to stack.
- **Quantization critical path** (Open Opportunity #1): use `sass_per_line_analysis.py --sort count` to see how many SASS instructions `_quant_fp4` and `softmax_step` emit, and `--sort max_gpr` to check their register pressure.

### Quick Landmark-Based Mapping

When debug info is unavailable, use structural pattern matching:

```bash
# Find quantization (F2FP.SATFINITE.E2M1 is FP4 quant)
grep -n "F2FP.SATFINITE.E2M1" kernel.sass | head -5

# Find MMA operations (WGMMA is Blackwell tensor core)
grep -n "WGMMA" kernel.sass | head -5

# Find exp2 operations (MUFU.EX2) - used in softmax, causes stall_long_sb
grep -n "MUFU.EX2" kernel.sass | head -10

# Find barrier arrives (marks phase boundaries)
grep -n "SYNCS.ARRIVE" kernel.sass
```

| SASS Landmark | Python Function | Stall Relevance |
|--------------|-----------------|-----------------|
| `MUFU.EX2` | softmax exp2 | ~20 cycle latency, causes `stall_long_scoreboard` |
| `F2FP.SATFINITE.E2M1` | `_quant_fp4()` | Serialized before MMA, causes `stall_barrier` |
| `SYNCS.ARRIVE` | barrier_arrive | Phase boundary, affects MMA warp wake-up |
| `WGMMA` | MMA operations | Consumer of P matrix, stalls on barrier |
| `FFMA2`/`FADD2` | reduction tree | FFMA2 hides MUFU latency, FADD2 exposes it |

---

## References

- Kernel file: `flash_fwd_sm100_fp4.py`
- Key functions:
  - `update_row_sum_sage()` - lines 326-348
  - `update_row_sum()` - lines 317-322  
  - `_compute_row_sum()` - lines 139-142
  - `fadd_reduce()` - lines 71-108
  - `_quant_fp4()` - lines 3004+
  - `softmax_step()` - around line 3155
