# Round 0 Summary (IN PROGRESS — not done yet)

## What Was Implemented

### P Dtype Fix (NaN Root Cause — FIXED)
The inline kernel's `tP_layout` and `tSrP_r2t` (P tensor for PV GEMM) used `self.q_dtype` (Float4E2M1FN) instead of `self.v_dtype` (BFloat16). P (softmax probabilities) is always BF16 for the block-scaled QK + BF16 PV path. The standalone FP4 kernel correctly uses `self.v_dtype`. This caused 100% NaN on every FP4 call.

**Fix**: Two conditional dtype selections:
1. `tP_layout`: use `self.v_dtype` when `block_scaled_qk`, else `self.q_dtype` (line 581)
2. `tSrP_r2t` recast: use `self.v_dtype` when `block_scaled_qk`, else `self.q_dtype` (line 2687)

**Result**: NaN eliminated. Output now finite with cos_sim=0.44 vs BF16 reference.

### S2T MLIR Legalization Fix
The `tcgen05.make_s2t_copy` produces a `tiled_copy` type that MLIR can't pass through `scf.for` loop iter_args in cutlass-dsl 4.4.2. Extracted `tiled_copy_s2t_sfq/sfk` outside the loop and passed them separately to `mma()`.

### S2T Pattern Restructure
Matched standalone kernel's exact pattern: single SRC (from stage 0, covers all stages via last dimension), per-stage DST from each `tCtSFQs[stage]`.

### SFK Load Guard
Added `if const_expr(self.block_scaled_qk)` guard around SFK TMA partition in load warp, preventing crash on None `tma_tensor_sfk` in BF16 path.

### Infrastructure
- Ported flash_attn_pr branch files to fp4 branch for integrated testing
- Moved benchmark files to `benchmarks/` directory
- Pinned cutlass-dsl to 4.4.2

## Files Changed

- `flash_attn/cute/flash_fwd_sm100.py` — P dtype fix, S2T MLIR fix, SFK guard, S2T restructure
- `flash_attn/__init__.py` — Remove C extension imports (allow bench to run)
- `flash_attn/cute/interface.py` — From flash_attn_pr (mSFQ/mSFK dispatch)
- `flash_attn/cute/blackwell_helpers.py` — From flash_attn_pr (block-scaled helpers)
- `flash_attn/cute/softmax.py` — From flash_attn_pr (scale_groupwise)
- `flash_attn/cute/fast_math.py`, `mma_sm100_desc.py`, `cute_dsl_utils.py` — From flash_attn_pr
- `flash_attn/cute/modified_utils/` — From flash_attn_pr (SF layout helpers)

## Validation

- BF16 path: works correctly (verified output shape + values)
- FP4 NVFP4+BF16 (with nvfp4_quantize): cos_sim=0.44, no NaN
  - Without S2T: cos_sim=0.61 (S2T makes it worse — writes wrong SF to TMEM)
  - Standalone kernel (same inputs, same cutlass-dsl): cos_sim=0.99
- Standalone FP4 kernel still works: cos=0.991
- Commands:
  ```
  CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ENABLE_TVM_FFI=1 PYTHONPATH=$(pwd) python -c "..."
  ```

## Remaining Items

**S2T cos issue (BLOCKING all tasks 4-14)**: The S2T copy writes incorrect scale factors to TMEM, degrading cos from 0.61 (no S2T, garbage SF) to 0.44 (with S2T). Despite matching the standalone kernel's `mainloop_s2t_copy_and_partition` method exactly (identical code), the inline kernel's S2T produces wrong partitions. Possible causes:
1. SF TMA loading (GMEM→SMEM) delivering wrong data before S2T
2. TMEM layout mismatch between `make_tmem_layout_sfa` output and what block-scaled gemm reads
3. Subtle JIT context difference affecting `partition_S`/`partition_D` inside the inline kernel

All plan tasks 4-14 are blocked on resolving this S2T issue.

## Commits

1. `79ccdccd` — P dtype fix + S2T MLIR fix + ported flash_attn_pr files
2. `bb910b47` — S2T restructure to match standalone pattern

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260522-p-dtype-inline
Notes: In the inline block-scaled kernel, P (softmax probabilities) tensor must use v_dtype (BF16), NOT q_dtype (FP4). The upstream kernel uses q_dtype for P because Q/K/V are all the same dtype. But when block-scaled QK uses FP4 Q/K with BF16 V, P must be BF16 (matching the PV GEMM's A operand dtype). Two locations: tP_layout (SMEM layout for P) and tSrP_r2t (register recast for TMEM store). Wrong dtype causes 100% NaN.
