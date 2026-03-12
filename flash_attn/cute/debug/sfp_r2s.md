# FP4 quant_v debug -- SFP/SFV TMEM overlap + all-zeros output

## Status: TMEM overlap fix applied. Output is now all-zeros (was 2.890625). PV GEMM O accumulator reads back as 0. Investigating.

## FORBIDDEN: Do not use for loop for R2S copy

**Scalar for-loop stores have no vectorization and cause bank conflicts.**

## Root cause of --quant_v 100% NaN: R2S copy was commented out

The R2S copy at `softmax_step()` line ~3182 was fully commented out:
```python
# if const_expr(sSFP is not None):
#     thread_idx = thr_tmem_load.thr_idx
#     base_offset = (thread_idx // 4) * 16 + (thread_idx % 4) * 4
#     sfp_thread_layout = cute.make_layout((4, 2), stride=(1, 512))
#     sSFP_stage_ptr = sSFP[None, None, None, stage].iterator
#     sSFP_thread = cute.make_tensor(sSFP_stage_ptr + base_offset, sfp_thread_layout)
#     tSrPSF_2d = cute.logical_divide(tSrPSF, cute.make_layout(4))
#     cute.autovec_copy(tSrPSF_2d, sSFP_thread)
```

This meant:
1. SFP scale factors were computed correctly in registers by the softmax warp
2. But never copied to shared memory (sSFP)
3. MMA warp's S2T copy (lines 2556-2562 and 2672-2678) read uninitialized SMEM
4. PV GEMM used garbage SFP scale factors -> NaN output

**Fix applied:** Uncommented the R2S copy block (lines 3182-3192).

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

Same bug existed for sfq_offset (QK path) but was MASKED because all scale factors = 1.0 in tests.

Reference: `dense_blockscaled_gemm_persistent.py` lines 1055-1084 does it correctly by
adding the offset to a Float32 pointer (same width as u32) then recasting to sf_dtype.

### Evidence for the overlap (from debug override experiments)

With SFP overridden to 0 in registers, SFV=1.0 in SMEM:
| P value | Stage 0 O_tmem | Stage 1 O_tmem | Expected (SFP=0) |
|---------|---------------|---------------|------------------|
| 0.0     | 0             | 0             | 0                |
| 1.0     | 370           | 128           | 0 (but SFV overwrites SFP K1) |
| 2.0     | 740           | 256           | 0                |

Stage 0's SFP K-tile 1 was overwritten by SFV (1.0) after S2T copy, causing nonzero output.
Stage 1 = expected/2 because only K-tile 1 contributed (K-tile 0 had SFP=0).

## Bug 2 (CURRENT): PV GEMM O accumulator is all zeros

After fixing the TMEM overlap, quant_v output changed from 2.890625 to **all zeros**.
Non-quant_v path still passes (output=2.0). The base FP4 NaN issue from previous
session no longer reproduces (environment was fixed).

### What we verified
- Quantization produces correct values:
  - exp2 ≈ 1.0 (correct for uniform Q=K=2.0 debug input)
  - group_max ≈ 1.0
  - P_u32 = 0x22222222 (all 1.0 in E2M1) ✓
  - SFP_u32 = 0x38383838 (all 1.0 in UE4M3) ✓
- row_sum ≈ 128 per work tile (correct, scale = 1/128) ✓
- correction_epilogue reads O_tmem = 0.0 for ALL elements
  → The PV GEMM itself produces zero accumulator

### What to investigate next
1. **P TMEM store**: Is the quantized P actually reaching the correct TMEM address?
   - P is stored at `tmem_p_offset = [64, 192]` (in qk_acc_dtype units)
   - TS GEMM `offset_a=[0, 8]` — does this match where P was stored?
   - The tOrPs pointer is computed with `qk_acc_dtype.width // v_dtype.width * tmem_p_offset[stage]`
   - Check: does the TS GEMM read from the same address that the softmax warp wrote to?
2. **SFP/SFV S2T copy**: Are scale factors reaching the correct TMEM columns?
   - With sfp_offset=32 (8 u32 cols), SFV is now at a different TMEM location
   - Does the TS GEMM descriptor know about the new SFV location?
3. **O accumulator address**: Does the TS GEMM write to the TMEM region that
   correction_epilogue reads from? (O at tmem_o_offset = [256, 384])
4. **GEMM execution**: Is `gemm_ptx_partial_fp4` TS path actually executing the MMA?
   - Could be a descriptor mismatch, wrong idesc, or stale tmem address

## What was implemented

### R2S copy approach: `cute.autovec_copy` with manual layout

The `get_smem_store_op` + `make_tiled_copy_D` approach from
`dense_blockscaled_gemm_persistent_amax.py` was NOT used because:
- `get_smem_store_op` requires a `tiled_tmem_load` to determine thread-value ownership
- `tSrPSF` (8 values per thread) comes from `compute_group_max`, not from TMEM load
- The TMEM load tiles 128 threads over 128x128 elements; SFP has only 128x8
- The dimensions don't match for `make_tiled_copy_D` / `retile`

Instead, we use manual pointer arithmetic + `cute.autovec_copy`:
- Each thread computes its base offset in sSFP using the BlockScaledBasicChunk strides
- A per-thread smem tensor view is created with layout `(4, 2)` stride `(1, 512)`
- `cute.autovec_copy` performs 2 vectorized 4-byte stores (no bank conflicts)

### Bank conflict analysis
- sSFP atom strides: `((16, 4), (0, 1))` for `((32,4), (16,4))`
- Thread t writes to base `(t//4)*16 + (t%4)*4`
- Within a warp (32 threads), each writes to a different 4-byte-aligned address
- No bank conflicts (32 threads -> 32 unique banks)

### Changes made

1. **R2S copy in `softmax_step`** (after `_quant_fp4`):
   ```python
   base_offset = (thread_idx // 4) * 16 + (thread_idx % 4) * 4
   sfp_thread_layout = cute.make_layout((4, 2), stride=(1, 512))
   sSFP_stage_ptr = sSFP[None, None, None, stage].iterator
   sSFP_thread = cute.make_tensor(sSFP_stage_ptr + base_offset, sfp_thread_layout)
   tSrPSF_2d = cute.logical_divide(tSrPSF, cute.make_layout(4))
   cute.autovec_copy(tSrPSF_2d, sSFP_thread)
   ```

2. **S2T copy in MMA warp** (after P_full wait, before PV GEMM):
   ```python
   tiled_copy_s2t_sfp_staged = [
       self.mainloop_s2t_copy_and_partition(sSFP, tCtSFPs[stage])
       for stage in range(self.q_stage)
   ]
   # Then in the PV GEMM loop:
   _, _, tCtSFP_compact_s2t = tiled_copy_s2t_sfp_staged[stage]
   tCsSFP_compact_s2t_cur = tCsSFP_compact_s2t[None, None, None, None, stage]
   cute.copy(tiled_copy_s2t_sfp, tCsSFP_compact_s2t_cur, tCtSFP_compact_s2t)
   ```

3. **Threading sSFP through call chain**:
   - Added `sSFP` param to `mma()`, `softmax_loop()`, `softmax_step()`
   - Passed from `forward()` through all calls

4. **SMEM fence NOT needed**: `autovec_copy` is a synchronous SIMT operation.
   No `fence_proxy(async_shared)` is needed after it. The mbarrier arrive/wait
   is sufficient to ensure visibility to the MMA warp's S2T copy.

### Bug fix: `_quant_fp4` indexing

Fixed `tSrPSF_u32_view[i // 4]` -> `tSrPSF_u32_view[i]`.
The old code wrote both i=0 and i=1 packs to index 0 (since 0//4 == 1//4 == 0),
effectively losing the first 4 scale factors.

## Layout analysis

- `sSFP` shape: `((((32, 4), 1), (16, 4)), 1, 2, 2)` =
  `(((Atom_Inst_M, Rest_M), (Atom_Inst_K, Rest_K)), MMA_M, MMA_K, STAGE)`
  - The `16` in K dimension is fake (stride 0)
- Per-stage unique elements: 128 x 4 x 2 = 1024 bytes = 128 threads x 8 values
- MMA_K tile cosize = 512 bytes, 2 tiles per stage
- Stage stride = 1024 bytes
- Address for (thread t, scale factor j):
  `(t//4)*16 + (t%4)*4 + (j%4) + (j//4)*512`

## TMEM overlap analysis (UPDATED after fix)

Scale factor TMEM addresses (in Float8E4M3FN element units, 4 per u32 col):

**Before fix** (sfp_offset=sfq_offset=16 sf elements = 4 u32 cols):
```
SFP tmem base: stage0=0, stage1=128    (u32 cols: 0, 32)
SFV tmem base: stage0=0+16, stage1=128+16  (u32 cols: 4, 36)  ← OVERLAPS SFP K-tile 1!
SFQ tmem base: stage0=128, stage1=0    (u32 cols: 32, 0)
SFK tmem base: stage0=128+16, stage1=0+16  (u32 cols: 36, 4)  ← OVERLAPS SFQ K-tile 1!
```

**After fix** (sfp_offset=sfq_offset=32 sf elements = 8 u32 cols):
```
SFP tmem base: stage0=0, stage1=128    (u32 cols: 0, 32)
SFV tmem base: stage0=0+32, stage1=128+32  (u32 cols: 8, 40)  ← No overlap ✓
SFQ tmem base: stage0=128, stage1=0    (u32 cols: 32, 0)
SFK tmem base: stage0=128+32, stage1=0+32  (u32 cols: 40, 8)  ← No overlap ✓
```

SFQ/SFK and SFP/SFV still share TMEM regions across stages (opposite stage mapping).
This is handled correctly because SFQ/SFK are reloaded before each QK GEMM.

## Debug tips

- Insert `breakpoint()` and print to inspect tensor shapes
- Verify `sSFP[None, None, None, stage].iterator` points to correct stage offset
- Check that `autovec_copy` generates STS.32 instructions (not byte stores)
- If switching to `get_smem_store_op` approach: need a tiled copy that matches
  SFP's 128-thread x 8-value partitioning (not the S matrix's 128x128 partitioning)
- To print scale factor values in tmem to see if they are correct, you can copy them to registers and print them like below (line 2488-2501 in flash_fwd_sm100_fp4.py):
```
                    # # make tmem to reg store atom for debugging
                    # if m_block == 0 and head_idx == 0 and batch_idx == 0 and split_idx == 0:
                    #     tidx = cute.arch.thread_idx()[0] % cute.arch.WARP_SIZE
                    #     tmem_load_atom = cute.make_copy_atom(
                    #         tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(8)),
                    #         Float8E4M3FN,
                    #     )
                        # thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tCtSFQs[0]).get_slice(tidx)
                        # tCtSFQs0_t2r = thr_tmem_load.partition_S(tCtSFQs[stage])
                        # tCrSFQs0_t2r_shape = thr_tmem_load.partition_D(tCtSFQs[stage]).shape
                        # tCrSFQs0_t2r = cute.make_fragment(tCrSFQs0_t2r_shape, Float8E4M3FN)
                        # cute.copy(thr_tmem_load, tCtSFQs0_t2r, tCrSFQs0_t2r)
                        # if tidx == 0:
                        #     cute.print_tensor(tCrSFQs0_t2r.load().to(Float32))
```
- You can print values in smem (e.g. sfP copied from reg to smem) (line 2396-2410 in flash_fwd_sm100_fp4.py) :
```
        # Copy sSFQ from smem to reg fragment for debugging
        # if const_expr(self.quant_qk) and sSFQ is not None:
        #     # Filter zeros to get compact layout and get stage 0
        #     sSFQ_compact = cute.filter_zeros(sSFQ[None, None, 0, 1])
        #     # sSFQ_compact = cute.filter_zeros(sSFK[None, None, 0, 1])
        #     sSFQ_slice = cute.logical_divide(sSFQ_compact, cute.make_layout(16))[None, 1]
        #     # Create register fragment with matching shape
        #     tSrSFQ = cute.make_fragment_like(sSFQ_slice, Float8E4M3FN)
        #     # Copy from smem to rmem using autovec_copy
        #     cute.autovec_copy(sSFQ_slice, tSrSFQ)
        #     tSrSFQ_f32 = cute.make_fragment_like(tSrSFQ, Float32)
        #     # Print to check for NaN
        #     if tidx == 0:
        #         tSrSFQ_f32.store(tSrSFQ.load().to(cute.Float32))
        #         cute.print_tensor(tSrSFQ_f32)
```

## Investigation timeline

### Session 1 (previous)
1. Identified R2S copy being commented out as a quant_v bug — uncommented it
2. Fixed `_quant_fp4` indexing: `tSrPSF_u32_view[i // 4]` → `tSrPSF_u32_view[i]`

### Session 2 (current)
8. Base FP4 NaN issue no longer reproduces (env fixed). Non-quant_v output=2.0 ✓
9. quant_v output was 2.890625 (wrong, expected 2.0) for s=128 debug case
10. Identified SFP/SFV TMEM overlap: `find_tmem_tensor_col_offset` returns u32 cols but
    was used as sf_dtype element offset → sfp_offset=16 sf elements (4 u32 cols) instead
    of 32 sf elements (8 u32 cols)
11. Confirmed with debug override experiments: set SFP=0 in registers, observed SFV
    overwriting SFP K-tile 1 data in TMEM (stage 0 O_tmem nonzero despite SFP=0)
12. Applied TMEM offset fix: `sfp_offset = ceil(col_offset * sf_dtype_per_u32 / align) * align`
    for both QK (sfq_offset) and PV (sfp_offset) paths
13. Removed all debug overrides and prints
14. **Result: output changed from 2.890625 to ALL ZEROS** — regression, not improvement
15. Verified quantization values are correct: P=0x22222222 (1.0 E2M1), SFP=0x38383838 (1.0 UE4M3) ✓
16. Verified row_sum ≈ 128 per work tile, scale ≈ 0.0078 — softmax warp is correct ✓
17. **Found: correction_epilogue reads O_tmem = 0.0** — PV GEMM produces zero accumulator
18. Active debug prints in code: QUANT printf (softmax_step), CORR printf (correction_loop),
    EPI printf (correction_epilogue)

### Next steps
- Verify P TMEM store address matches TS GEMM's A operand address
- Check if TS GEMM descriptor (idesc, offsets) is consistent with new TMEM layout
- Print P values in TMEM from MMA warp to confirm they were stored correctly
- Check O accumulator TMEM address vs where correction_epilogue reads
