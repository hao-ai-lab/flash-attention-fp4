# SASS Analysis Learnings — FP4 Flash Attention on B200

## How to Generate Cubin from CuTe DSL Kernel

```bash
cd <working_dir>  # cubin is dumped HERE
CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_LINEINFO=1 CUTE_DSL_ENABLE_TVM_FFI=1 \
CUDA_VISIBLE_DEVICES=1 python benchmark.py
```

- `CUTE_DSL_KEEP_CUBIN=1` — saves the compiled cubin in the current working directory
- `CUTE_DSL_LINEINFO=1` — embeds DWARF `.debug_line` sections for SASS-to-Python mapping
- Cubin filename is long (includes full class path), use `ls *.cubin` to find it

## SASS Analysis Workflow

1. **Resource check**: `cuobjdump --dump-resource-usage kernel.cubin`
2. **Annotated SASS**: `nvdisasm -g -c kernel.cubin > annotated.sass`
3. **Per-line analysis**: `python sass_per_line_analysis.py kernel.cubin --sort count`
4. **Instruction counts**: `cuobjdump -sass kernel.cubin | grep -c "PATTERN"`
5. **With NCU stalls**: `python sass_per_line_analysis.py kernel.cubin --ncu-rep profile.ncu-rep --sort stalls`

**Important**: `nvdisasm -g` (line info) and `nvdisasm -lrm count` (register counts) cannot be combined — the tool runs both separately and joins by PC address.

## Key Blackwell (SM100) Instructions

| Instruction | Purpose | Notes |
|-------------|---------|-------|
| `UTCOMMA.4X` | tcgen05 MMA (Blackwell) | Not shown by `grep TCGEN05`; use `grep UTCOMMA` |
| `F2FP.SATFINITE.E2M1` | FP32→FP4 conversion | FP4 quantization; 128 per softmax step |
| `MUFU.EX2` | exp2 (special function) | ~20 cycle latency; needs latency hiding |
| `FMNMX3` | 3-input min/max | Blackwell-specific; reduces tree depth vs FMNMX |
| `FENCE.VIEW.ASYNC.T` | TMEM write fence | Ensures MMA warp sees written TMEM data |
| `SYNCS.ARRIVE.TRANS64.ART0` | Barrier arrive with data | Signals consumer warp |

## FP4 Critical Path Anatomy

```
softmax warps (0-7):
  MUFU.EX2 (exp2)         ~20 cycle latency
  → FMNMX3 tree           group max computation
  → F2FP.SATFINITE.E2M1   FP4 quantization (128 instructions!)
  → SYNCS.ARRIVE           barrier arrive (signals MMA)
  → FENCE.VIEW.ASYNC.T    TMEM fence

MMA warp (12):
  SYNCS.PHASECHK.TRYWAIT  barrier wait (shows as stall_long_sb)
  → UTCOMMA.4X             PV gemm (16 MMA ops)
```

The quant-to-barrier path is the bottleneck. The 128 F2FP instructions serialize here. Compiler partially hides exp2 latency by interleaving F2FP with MUFU.EX2.

## mul_packed_f32x2 Investigation — Red Herrings and Root Cause

### Initial observation (bad tvm-ffi binary)

Removing `mul_packed_f32x2` caused -15% TFLOPS. We investigated extensively:

| What we tried | Result | Why it failed |
|---------------|--------|---------------|
| Scheduling fence (inline asm) | No effect | Fences can reorder instructions but can't move barrier arrives earlier — that's a compiler heuristic decision |
| Scalar mul + fence | -1.5% vs packed | FMUL2 (packed) is more efficient than 2x FMUL; fence recovered scheduling but not the instruction efficiency |
| Detailed pre-barrier SASS diff | Explained symptoms, not cause | MULT had 168 insns pre-barrier vs NO-MULT 200 — but this was an effect, not the root cause |

### Lessons

1. **Exact binary matters, not just version string** — pip install from different mirrors can produce different codegen
2. **Scheduling fences can't move barriers** — they constrain instruction order, not barrier placement
3. **~30% of SASS instructions lose source line info** during MLIR→LLVM→PTX lowering (attributed to `@cute.jit` decorator line)

## Practical Gotchas

1. **nvdisasm vs cuobjdump**: `nvdisasm -g -c` for source line annotations, `cuobjdump -sass` for raw SASS. `-g` only works with nvdisasm.
2. **UTCOMMA naming**: Blackwell MMA = `UTCOMMA.4X` in SASS, not `TCGEN05.MMA` or `WGMMA`.
3. **Register pressure vs REG count**: `cuobjdump --dump-resource-usage` shows REG:128 (allocated), but live registers can reach 192 at hot spots.
4. **FP4 kernel bug**: `thr_tmem_store_psf.partition_S(tSrPSF)` fails on rank-1 tensors — use `make_fragment(partition_D(...).shape, ...)` instead.
5. **Cubin location**: Created in the **working directory**, not next to the .py source.

## P_full Barrier Interleave — Why mul_packed_f32x2 Enables Pipeline Overlap

### Three variants compared (good tvm-ffi, seqlen=16384)

| Variant | Placement | PSF multiply | TFLOPS |
|---------|-----------|-------------|--------|
| A (reference) | After barriers (line 3224) | Yes | 1755 |
| B | Before barriers (line 3190) | None | 1450 |
| C | Before barriers (line 3190) | Yes (`tSrPSF_f32`) | 1709 |

### Root cause: P_full arrive timing determines MMA pipeline overlap

The softmax warp signals two barriers to the MMA warp:
- `mbar_P_full_O_rescaled_offset` (`+0x80`) — first half of P is in TMEM, MMA can start PV GEMM
- `mbar_P_full_2_offset` (`+0x138`) — second half of P is ready

The MMA warp's pipeline is: wait P_full → PV GEMM first half → wait P_full_2 → PV GEMM second half.
If P_full arrives early, MMA overlaps first-half GEMM with softmax computing the second half.

### SASS instruction ordering (second iteration)

**Variant C (fast — P_full at 18 insns):**
```
      # Compiler distributes line 342 mul across lines 104-106 reduction tree:
      # (a+b+c+d)*e → a*e, c*e (FMUL2 seeds), b*e+a*e, d*e+c*e (FFMA2 fused), final add (FADD2)
      # Costs 5 insns instead of 4 (3×FADD2 + FMUL2) — likely MLIR/NVVM lowering artifact
7d40: FMUL2 R2, R12*R78        local_sum[0]*group_max_exp — seed, no prior sum to fuse
7d50: FMUL2 R6, R4*R72         local_sum[2]*group_max_exp — seed
7d60: FFMA2 R4, R14*R76 + R2   local_sum[1]*group_max_exp + R2 — line 104 add fused w/ line 342 mul
7d70: FFMA2 R2, R8*R30 + R6    local_sum[3]*group_max_exp + R6 — line 105 add fused w/ line 342 mul
7d80: FADD2 R4, R4 + R2        line 106: final reduction (both sides already multiplied)
7da0: FADD R140, R4 + R5       line 346: row_sum accumulator (output DEAD before arrive)
      ... FMNMX3 row-max → FSEL → FMUL R2 = -row_max*log2e ...
      ... R2 becomes bias for FFMA2s (line 377: S*log2e+bias) ...
      ... FMUL2 latency delays last 5 FFMA2 past arrive ...
8ef0: SYNCS.ARRIVE [+0x148]    sfqk_load_full arrive
      5x FFMA2 (line 377) + 5x MUFU.EX2 + 1x F2FP + STS
8fd0: SYNCS.ARRIVE [+0xb0]     sfqk_load_empty arrive
8fe0: FENCE.VIEW.ASYNC.T       flush first-half TMEM
9000: SYNCS.ARRIVE [+0x80]     *** P_full arrive (16 insns after sfqk_load_full) ***
      191 insns: 91 MUFU.EX2, 63 F2FP, 21 FADD2
9bd0: STTM.x16                 TMEM store (second half)
9c00: SYNCS.ARRIVE [+0x138]    *** P_full_2 arrive ***
```

**Variant B (slow — P_full at 87 insns):**
```
      (no FMUL2 — mul_packed_f32x2 skipped, no extra latency on bias)
      ... FMNMX3 row-max tree, then FFMA2 S*log2e+bias (line 377) ...
      ... all FFMA2 done by 8e30, before arrive ...
8e50: SYNCS.ARRIVE [+0x148]    sfqk_load_full arrive
      66x MUFU.EX2 + 5x F2FP + 7x FADD2 batched (all FFMA2 inputs ready):
8e60: MUFU.EX2 R104, R98       exp2 (independent — FFMA2 already completed)
8e80: MUFU.EX2 R96, R46        exp2 (independent)
8ed0: FADD2 R38, R106+R96      rowsum += exp2 output
      ... 40+ more MUFU.EX2 ...
9150: MUFU.EX2 R44, R66        exp2
9190: F2FP R19, R45, R44, RZ   quant — first F2FP, 82 insns after arrive
9350: STS [R186+0x400], R24    acc_scale → SMEM (line 3160)
9360: SYNCS.ARRIVE [+0xb0]     sfqk_load_empty arrive
9390: FENCE.VIEW.ASYNC.T       flush first-half TMEM
93a0: SYNCS.ARRIVE [+0x80]     *** P_full arrive (84 insns after sfqk_load_full) ***
      118 insns: 28 MUFU.EX2, 58 F2FP, 18 FADD2
9ae0: STTM.x16                 TMEM store (second half)
9b10: SYNCS.ARRIVE [+0x138]    *** P_full_2 arrive ***
```

### NCU stall data confirms the pipeline overlap

(Values are warp stall samples — scheduler cycles where warp was stalled at that PC.)

| Barrier wait | B samples | C samples | Meaning |
|-------------|----------|----------|---------|
| P_full (line 2530, `+0x80`) | **111** (long_sb:108) | **21** (long_sb:20) | MMA idles 5x longer in B |
| P_full_2 (line 2566, `+0x138`) | **37** (long_sb:37) | **156** (long_sb:156) | C trades this for early P_full |
| softmax_corr_empty (line 3218) | **1602** (long_sb:1568) | **346** (long_sb:327) | B's delayed barriers cascade |
| sfqk_load_full (line 3118) | 258 | 355 | Similar |
| sfqk_load iter0 (line 2454) | 37 | 23 | Similar |
| sfqk_load iterN (line 2619) | 0 | 0 | Never stalls |

**Instruction counts by region (sfqk_load_full → P_full → P_full_2):**

| Region | Variant C (multiply) | Variant B (no multiply) |
|--------|---------------------|------------------------|
| sfqk_load_full→P_full | **16 insns**: 5 MUFU, 1 F2FP, 5 FFMA2 | **84 insns**: 66 MUFU, 5 F2FP, 7 FADD2 |
| P_full→P_full_2 | **191 insns**: 91 MUFU, 63 F2FP, 21 FADD2 | **118 insns**: 28 MUFU, 58 F2FP, 18 FADD2 |
| Total | 207 insns | 202 insns |

### Why the compiler gets it wrong

The compiler optimizes the softmax warp **in isolation**. It sees MUFU.EX2 has ~20
cycle latency and greedily front-loads as many exp2/quant instructions as possible
before the FENCE/ARRIVE to hide that latency within the softmax warp. But it has
**no visibility into the MMA warp** waiting on the other side of `mbar_P_full_O_rescaled_offset`.
So "hiding latency" within the softmax warp creates a much larger pipeline bubble
on the MMA warp — 111 stall samples of MMA sitting idle in Variant B.

### Why mul_packed_f32x2 accidentally fixes it

Both variants compute `S * log2e + bias` via FFMA2 (line 377) → MUFU.EX2 (exp2)
→ F2FP (quant).

In B (no multiply), lines 104-106 (`_compute_row_sum` reduction) compile to 3×FADD2.
In C, `mul_packed_f32x2` (line 342) distributes the multiply across the reduction
tree, turning it into 2×FMUL2 + 2×FFMA2 + 1×FADD2 = 5 packed ops (see SASS above).
This heavier path has **no data dependency to the line-377 FFMA2 bias** — the chain
output (R140 = scaled group sum) is dead before the arrive, and the bias computation
(FSEL → FMUL → R2) is independent. But the 2 extra instructions delay the overall
schedule, pushing some line-377 FFMA2s past the arrive. This doesn't affect
correctness — TMEM stores and P_full arrives are done in chunks (first half /
second half), so FFMA2/MUFU.EX2 can continue between chunks.

- **B**: all FFMA2s (line 377) complete by 8e30, before 8e50 (sfqk_load_full arrive).
  All MUFU.EX2 inputs ready → compiler batches 66 MUFU.EX2 before barriers.
- **C**: last 5 FFMA2s (line 377) at 8f00-8f80 still issuing after 8ef0
  (sfqk_load_full arrive). MUFU.EX2 for those outputs can't issue yet → only
  5 MUFU.EX2 (whose inputs are from a prior phase) fit before barriers.

Not register pressure: GPR at P_full arrive: C=155, B=154 (both well below
169 max).

### Barrier offset mapping (for reference)

| Offset | Barrier | Producer | Consumer |
|--------|---------|----------|----------|
| `+0x80` | mbar_P_full_O_rescaled | softmax warp (line 3210) | MMA warp (line 2530) |
| `+0x88` | mbar_P_full_O_rescaled stage1 | softmax warp | MMA warp |
| `+0xb0` | sfqk_load_empty | softmax warp (line 3163) | load warp |
| `+0xc0` | softmax_corr_empty | MMA warp | softmax warp (line 3218) |
| `+0x138` | mbar_P_full_2 | softmax warp (line 3215) | MMA warp (line 2566) |
| `+0x148` | sfqk_load_full | load warp | softmax warp (line 3118) |

## scale_subtract_rowmax has the same MUFU front-load problem

After moving `update_row_sum_sage` to after barriers (line 3226, with `None`),
`scale_subtract_rowmax` (line 3167) causes the same symptom:
- With `tSrPSF_f32`: 1769 TFLOPS (fast)
- With `None`: 1383 TFLOPS (slow, -22%)

Same root cause — passing `tSrPSF_f32` adds per-group FMA ops (lines 367-377)
that interleave with MUFU.EX2 before sfqk_load_full arrive, preventing the
compiler from front-loading all MUFU.EX2 between sfqk→P_full.

| MUFU.EX2 placement | FAST (with PSF) | SLOW (no PSF) |
|---|---|---|
| Before sfqk_load_full | 35 | 17 |
| sfqk→P_full | **4** | **90** |
| P_full→P_full_2 | 90 | 22 |
| Total | 129 | 129 |

Cubins: `sass_analysis/fast_with_psf.cubin`, `sass_analysis/slow_no_psf.cubin`
Key PCs (second iteration):
- FAST: sfqk_load_full=8d60, P_full=8df0, P_full_2=9880
- SLOW: sfqk_load_full=8190, P_full=8800, P_full_2=8de0

## Resource Comparison: FP4 vs Reference

- FP4: REG:128, STACK:32, 3541 insns, 169 max live GPR
- Ref: REG:128, STACK:40, 3437 insns, 192 max live GPR
- FP4 has LESS stack spilling and lower peak register pressure despite more work
- This suggests the FP4 warp specialization keeps each warp's live register set smaller
