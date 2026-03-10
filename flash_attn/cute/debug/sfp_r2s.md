# Copy tSrPSF (registers) to sSFP (SMEM) — IMPLEMENTED

## Status: Compiled successfully. Needs precision testing.

## FORBIDDEN: Do not use for loop for R2S copy

**Scalar for-loop stores have no vectorization and cause bank conflicts.**

## What was implemented

### R2S copy approach: `cute.autovec_copy` with manual layout

The `get_smem_store_op` + `make_tiled_copy_D` approach from
`dense_blockscaled_gemm_persistent_amax.py` was NOT used because:
- `get_smem_store_op` requires a `tiled_tmem_load` to determine thread-value ownership
- `tSrPSF` (8 values per thread) comes from `compute_group_max`, not from TMEM load
- The TMEM load tiles 128 threads over 128×128 elements; SFP has only 128×8
- The dimensions don't match for `make_tiled_copy_D` / `retile`

Instead, we use manual pointer arithmetic + `cute.autovec_copy`:
- Each thread computes its base offset in sSFP using the BlockScaledBasicChunk strides
- A per-thread smem tensor view is created with layout `(4, 2)` stride `(1, 512)`
- `cute.autovec_copy` performs 2 vectorized 4-byte stores (no bank conflicts)

### Bank conflict analysis
- sSFP atom strides: `((16, 4), (0, 1))` for `((32,4), (16,4))`
- Thread t writes to base `(t//4)*16 + (t%4)*4`
- Within a warp (32 threads), each writes to a different 4-byte-aligned address
- No bank conflicts (32 threads → 32 unique banks)

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

4. **No SMEM fence needed**: SIMT stores are synchronous — data is visible after barrier sync.

### Bug fix: `_quant_fp4` indexing

Fixed `tSrPSF_u32_view[i // 4]` → `tSrPSF_u32_view[i]`.
The old code wrote both i=0 and i=1 packs to index 0 (since 0//4 == 1//4 == 0),
effectively losing the first 4 scale factors.

## Layout analysis

- `sSFP` shape: `((((32, 4), 1), (16, 4)), 1, 2, 2)` =
  `(((Atom_Inst_M, Rest_M), (Atom_Inst_K, Rest_K)), MMA_M, MMA_K, STAGE)`
  - The `16` in K dimension is fake (stride 0)
- Per-stage unique elements: 128 × 4 × 2 = 1024 bytes = 128 threads × 8 values
- MMA_K tile cosize = 512 bytes, 2 tiles per stage
- Stage stride = 1024 bytes
- Address for (thread t, scale factor j):
  `(t//4)*16 + (t%4)*4 + (j%4) + (j//4)*512`

## Debug tips

- Insert `breakpoint()` and print to inspect tensor shapes
- Verify `sSFP[None, None, None, stage].iterator` points to correct stage offset
- Check that `autovec_copy` generates STS.32 instructions (not byte stores)
- If switching to `get_smem_store_op` approach: need a tiled copy that matches
  SFP's 128-thread × 8-value partitioning (not the S matrix's 128×128 partitioning)
