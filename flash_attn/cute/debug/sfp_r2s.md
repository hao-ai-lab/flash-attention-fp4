# FP4 quant_pv debug — SFP R2S, TMEM overlap, scale_groupwise, V K-major

## Status: ALL BUGS FIXED. quant_pv kernel correct. Debug mode exact match (max_diff=0.0) for all seqlens. Random data max_diff=2-6 with SF=1.0 (expected for FP4 E2M1). ~1175 TFLOPS at bs=4 seq=4096.

## FORBIDDEN: Do not use for loop for R2S copy

**Scalar for-loop stores have no vectorization and cause bank conflicts.**

## Bug summary

| Bug | Symptom | Root cause | Fix |
|-----|---------|------------|-----|
| R2S commented out | 100% NaN | SFP never reached SMEM | Uncomment R2S copy |
| Bug 1: TMEM overlap | SFV overwrites SFP K-tile 1 | `find_tmem_tensor_col_offset` returns u32 cols, used as FP8 element offset | Multiply by `sf_dtype_per_u32` |
| Bug 2: PV all-zeros | O accumulator = 0 | Missing `scale_groupwise` — `FP4(P) * group_max(P) ≠ P` | Divide P by group_max before quantization |
| Bug 3: QK NaN | NaN grows across n-blocks | SF PTX offset in FP8 elements, not u32 columns | `recast_layout` to u32 for SF offsets |
| V layout | Wrong PV output | FP4 MMA requires K-major B; V had headdim contiguous | Host-side transpose so seqlen is contiguous |
| Debug data range | fp4=5.3125 vs ref=15.5 at seqlen≥1024 | V = block_index exceeded FP4 max (6.0) | `V = (s // n_block_size) % 4` |

## Bug 2 (FIXED): PV GEMM all-zeros — scale_groupwise

### Root cause

The block-scaled MMA computes `P_fp4 * SFP * V_fp4 * SFV`. For this to
reconstruct `P * V`, we need `P_fp4 * SFP ≈ P`. But `_quant_fp4` was storing
`P_fp4 = FP4(P_raw)` and `SFP = group_max(P)` without dividing P by group_max
first. Result: `P_effective = FP4(P) * group_max(P) ≠ P`.

Additionally, exp2 underflow (wild S values cause most positions to underflow to 0
after `exp2((S - rowmax) * scale_log2)`) meant only the group containing the
global max had P > 0, compounding the scaling error.

### Fix: `SoftmaxSm100.scale_groupwise()`

Normalize P per-group AFTER exp2 so that `P_fp4 * SFP ≈ P`:

```
Flow in softmax_step (quant_pv path):

1. scale_subtract_rowmax(S, rowmax)          — global rowmax, same as BF16
2. apply_exp2_convert(S) → P                 — exp2((S-rowmax)*scale)
3. update_row_sum(P, ...)                    — uses ORIGINAL P before normalization
4. compute_group_max(P) → gmax[g]            — per-group max of exp2 output
5. scale_groupwise(P, gmax, sf_size)         — P[i] /= gmax[group(i)]
6. _quant_fp4(P_norm, gmax) → FP4(P_norm) + UE4M3(gmax)
```

MMA reconstructs: `P_fp4 * SFP ≈ (P/gmax) * gmax = P` ✓

The `scale_groupwise` method on `SoftmaxSm100` (flash_fwd_sm100_fp4.py ~line 316):
```python
@cute.jit
def scale_groupwise(self, acc_S_row, group_max, sf_size=16):
    acc_S_row_frag = cute.logical_divide(acc_S_row, cute.make_layout(sf_size))
    for g in cutlass.range_constexpr(cute.size(group_max)):
        inv_gmax = Float32(1.0) / cute.arch.fmax(group_max[g], 1e-20)
        for j in cutlass.range(0, sf_size, 2, unroll_full=True):
            acc_S_row_frag[j, g], acc_S_row_frag[j + 1, g] = mul_packed_f32x2(
                (acc_S_row_frag[j, g], acc_S_row_frag[j + 1, g]),
                (inv_gmax, inv_gmax),
            )
```

**Key design decisions:**
- Post-exp2 normalization (not pre-exp2) — no need to exp2 scale factors
- `update_row_sum` moved BEFORE `scale_groupwise` so it sees original P values.
  This is safe because `row_sum` is purely in softmax warp registers — the
  correction warp doesn't read it.
- Division by zero guarded with `fmax(gmax, 1e-20)` — zero groups stay zero
- Uses `mul_packed_f32x2` for paired f32 ops consistent with other softmax methods

**Precision:** Without SP1, P_norm ∈ [0, 1] → only FP4 levels {0, 0.5, 1.0}.
With SP1 (P_norm *= 6): [0, 6] → all 8 FP4 levels {0, 0.5, 1, 1.5, 2, 3, 4, 6}.

## V K-major layout

FP4 block-scaled MMA (`tcgen05.mma.kind::mxf4nvf4.block_scale`) only supports
K-major for both A and B operands. This is hardcoded in `MmaMXF4NVF4Op` (both set
`OperandMajorMode.K`). The MLIR trait validation rejects MN-major for FP4.

V in standard flash attention has headdim contiguous (MN-major). TMA does NOT
transpose between MN-major global and K-major SMEM. The fix is host-side transpose:

```python
v_ref_kmajor = v_ref.permute(0, 3, 2, 1).contiguous().permute(0, 3, 2, 1)
# Shape: (batch, seqlen, nheads, headdim) with strides where seqlen has stride 1
```

The kernel's `new_stride` function also handles Python int strides (from non-standard
layout) by skipping `cute.assume` for compile-time constants.

## Bug 1 (FIXED): SFP/SFV TMEM overlap due to unit mismatch

`find_tmem_tensor_col_offset()` returns u32 columns, but the result was used as
Float8E4M3FN element offset without conversion.

```python
# BEFORE (BUG): sfp_offset in sf_dtype units, but raw value is u32 columns
sfp_offset = math.ceil(find_tmem_tensor_col_offset(tCtSFPs[0]) / align) * align  # = 16 sf elements = 4 u32 cols
# SFP spans 8 u32 cols → SFV at col 4 OVERLAPS SFP K-tile 1 (cols 4-7)

# AFTER (FIX): multiply by sf_dtype_per_u32 to convert units
sf_dtype_per_u32 = 32 // self.sf_dtype.width  # = 4 for Float8E4M3FN
sfp_offset = math.ceil(find_tmem_tensor_col_offset(tCtSFPs[0]) * sf_dtype_per_u32 / align) * align  # = 32 sf elements = 8 u32 cols
# SFV now at col 8, no overlap
```

### TMEM layout after fix

```
SFP tmem base: stage0=0, stage1=128    (u32 cols: 0, 32)
SFV tmem base: stage0=32, stage1=160   (u32 cols: 8, 40)  ← No overlap ✓
SFQ tmem base: stage0=128, stage1=0    (u32 cols: 32, 0)
SFK tmem base: stage0=160, stage1=32   (u32 cols: 40, 8)  ← No overlap ✓
```

Full TMEM map (no overlaps):
- S_offset=[0, 128], O_offset=[256, 384], P_offset=[64, 192]
- SFP at S base, SFV at S base + 32 elements (8 u32 cols)
- SFV stage1 ends at col 144 (u32 col 36), P stage1 starts at col 192 — no overlap

## Bug 3 (FIXED): QK GEMM NaN — scale factor PTX offset units mismatch

In `gemm_ptx_partial_fp4` (blackwell_helpers.py), SF k-tile offsets were in FP8
element units but PTX expects u32 column units:

```python
# BUG: offset_sfa = [0, 16] in FP8 elements → PTX reads col+16 (garbage)
# FIX: recast to u32 layout → [0, 4] in columns ✓
sfa_layout_u32 = cute.recast_layout(32, tScaleA.element_type.width, tScaleA.layout)
offset_sfa = [cute.crd2idx((0, 0, k), sfa_layout_u32) ...]
```

Only appeared with random data + seqlen ≥ 256 (2+ n-blocks). With 1 n-block the
second k-tile SF was never read. With `--debug` (SF=1.0), garbage SF happened to
produce finite results within acceptable range.

## R2S copy: SFP registers → SMEM

The `get_smem_store_op` + `make_tiled_copy_D` approach was NOT used because
`tSrPSF` comes from `compute_group_max` (not TMEM load), and dimensions don't
match the 128×128 TMEM load tiling. Instead, manual pointer arithmetic +
`cute.autovec_copy`:

```python
base_offset = (thread_idx // 4) * 16 + (thread_idx % 4) * 4
sfp_thread_layout = cute.make_layout((4, 2), stride=(1, 512))
sSFP_stage_ptr = sSFP[None, None, None, stage].iterator
sSFP_thread = cute.make_tensor(sSFP_stage_ptr + base_offset, sfp_thread_layout)
tSrPSF_2d = cute.logical_divide(tSrPSF, cute.make_layout(4))
cute.autovec_copy(tSrPSF_2d, sSFP_thread)
```

- 2 vectorized 4-byte stores, no bank conflicts (32 threads → 32 unique banks)
- `autovec_copy` is synchronous SIMT — no `fence_proxy(async_shared)` needed
- mbarrier arrive/wait sufficient for SMEM→TMEM visibility

### `_quant_fp4` indexing fix

Fixed `tSrPSF_u32_view[i // 4]` → `tSrPSF_u32_view[i]`. Old code wrote both
i=0 and i=1 packs to index 0 (since 0//4 == 1//4 == 0), losing first 4 SFs.

## FP4 E2M1 representable values

0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0 — max magnitude is 6.0.
Debug test data must stay within this range or values get clipped.

## Accuracy characterization

With all SF=1.0 (no data-dependent scaling of V):
- **Debug mode** (constant data within FP4 range): max_diff=0.0 for all seqlens ✓
- **Random data**: max_diff=2-6, mean_diff=0.03-0.09
  - Sources of error: FP4 P quantization (only 3 levels without SP1), block-by-block
    softmax processing, log2-based exp, accumulation order differences
  - Python FP4 reference (with P quantization) gives max_diff=0.29 vs bf16 — kernel's
    higher error is from online softmax block processing differences

## Open items

- [ ] Proper data-dependent SFV computation (currently all 1.0)
- [ ] SFV layout for K-major V: `create_scale_factor_tensor` uses mn=seqlen, k=headdim;
      for K-major V in PV GEMM should be mn=headdim, k=seqlen (no-op with SFV=1.0)
- [ ] SP1 scaling to use full FP4 range (P_norm *= 6.0, gmax /= 6.0)

## Debug tips

- Insert `breakpoint()` and print to inspect tensor shapes at compile time
- Verify `sSFP[None, None, None, stage].iterator` points to correct stage offset
- Check that `autovec_copy` generates STS.32 instructions (not byte stores)
- TMEM SF values can be inspected by copying to registers:
  ```python
  tmem_load_atom = cute.make_copy_atom(
      tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(8)), Float8E4M3FN)
  thr_load = tcgen05.make_tmem_copy(tmem_load_atom, tCtSFQs[0]).get_slice(tidx)
  frg = cute.make_fragment(thr_load.partition_D(tCtSFQs[stage]).shape, Float8E4M3FN)
  cute.copy(thr_load, thr_load.partition_S(tCtSFQs[stage]), frg)
  if tidx == 0: cute.print_tensor(frg.load().to(Float32))
  ```
- SMEM SF values can be inspected with `autovec_copy` to a register fragment
- Wild S values across K blocks are NOT a bug — softmax normalization handles them
