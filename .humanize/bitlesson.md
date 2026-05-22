# BitLesson Knowledge Base

This file is project-specific. Keep entries precise and reusable for future rounds.

## Entry Template (Strict)

Use this exact field order for every entry:

```markdown
## Lesson: <unique-id>
Lesson ID: <BL-YYYYMMDD-short-name>
Scope: <component/subsystem/files>
Problem Description: <specific failure mode with trigger conditions>
Root Cause: <direct technical cause>
Solution: <exact fix that resolved the problem>
Constraints: <limits, assumptions, non-goals>
Validation Evidence: <tests/commands/logs/PR evidence>
Source Rounds: <round numbers where problem appeared and was solved>
```

## Entries

## Lesson: S2T MLIR Legalization — Compute SF Layouts Locally
Lesson ID: BL-20260517-s2t-mlir-local-layout
Scope: flash_fwd_sm100.py integration, CuTe DSL MLIR lowering, S2T copy
Problem Description: `make_s2t_copy(Cp4x32x128bOp, tCtSF_compact)` fails with "failed to legalize unresolved materialization" when SF SMEM layouts are passed as `@cute.kernel` parameters.
Root Cause: Passing `sSFQ_layout` as a kernel parameter loses MLIR type information needed by `make_tmem_layout_sfa`. The MLIR lowering can't reconstruct the layout type from a runtime parameter.
Solution: Compute `make_smem_layout_sfa/sfb` LOCALLY inside the kernel body using class attributes (self.sf_vec_size, self.mma_inst_tile_k, tiled_mma_qk, self.mma_tiler_qk). Don't pass SF layouts through the kernel parameter boundary.
Constraints: SF layout computation must be deterministic from const_expr class attributes.
Validation Evidence: Inline integration in flash-attention-pr branch compiles and runs all 4 modes (BF16 1540 TF, NVFP4+BF16 1894 TF, NVFP4+FP8 2031 TF, MXFP8+FP8 1902 TF). Commit 0bf487c.
Source Rounds: 0

## Lesson: make_ptr Required for Torch Block-Scaled Q/K (NOT cute tensors)
Lesson ID: BL-20260517-make-ptr-torch-blockscaled
Scope: interface.py dispatch, cute tensor creation
Problem Description: MXFP8 Q/K (torch.float8_e4m3fn) passed via `to_cute_tensor()` produces rank-0 tensors. Cute tensors from `cute_tensor_like` have byte-based strides that conflict with `make_ordered_layout`'s sub-byte element strides.
Root Cause: CuTe DSL's `from_dlpack` on fp8 tensors (viewed as uint8) does not preserve rank. For cute tensors, `cute_tensor_like` stores 1 FP4 per byte (byte-based strides), but `make_ordered_layout` assumes packed FP4 (2 per byte, element-based strides) — a 2x stride mismatch.
Solution: Use `make_ptr` + `q_ptr_shape` for **torch** int8/fp8 tensors only. For **cute** tensors from `cute_tensor_like`, pass empty `q_ptr_shape` and the tensor directly (preserving its byte-based layout). The kernel checks `len(q_ptr_shape) > 0` to choose rebuild vs pass-through.
Constraints: Applies to both NVFP4 (Float4E2M1FN) and MXFP8 (Float8E4M3FN) Q/K tensors.
Validation Evidence: All 3 modes produce correct output (cos >= 0.975) via bench, no NaN. Commit 6a7f95ad.
Source Rounds: 0

## Lesson: cute_tensor_like Sub-Byte Stride Mismatch
Lesson ID: BL-20260518-cute-tensor-like-stride
Scope: bench_fp4.py, cutlass_torch tensor creation, sub-byte types
Problem Description: bench_fp4.py produced NaN for 50% of heads when using `return_torch=True` with `make_ptr` path. Int8 backing tensor from `cute_tensor_like` has 1 FP4 per byte (stride=1 byte), but `make_ordered_layout(Float4E2M1FN)` computes stride=0.5 bytes.
Root Cause: `cute_tensor_like` with sub-byte types (FP4) creates int8 tensors at full element shape (headdim, not headdim/2). Each byte holds ONE FP4 value. `make_ordered_layout` assumes PACKED layout (2 FP4 per byte). Head stride off by 2x → reads wrong data → NaN.
Solution: Don't use `make_ptr` for cute tensors. Pass them directly to preserve byte-based strides. Use `return_torch=False` in bench. OR pack data 2-per-byte and halve last dim for `make_ptr` path.
Constraints: Only affects sub-byte element types (FP4). FP8 and wider types have 1:1 byte:element mapping.
Validation Evidence: Switching to return_torch=False eliminates NaN, all shapes produce cos >= 0.975.
Source Rounds: 0

## Lesson: P Dtype Must Be v_dtype for Block-Scaled Inline Integration
Lesson ID: BL-20260522-p-dtype-inline
Scope: flash_fwd_sm100.py inline integration, PV GEMM data types
Problem Description: Inline block-scaled kernel produces 100% NaN for all FP4 inputs. The NaN is NOT from S2T or SF loading — it persists even with S2T completely disabled.
Root Cause: `tP_layout` (P SMEM layout) and `tSrP_r2t` (P register recast for TMEM store) use `self.q_dtype` (Float4E2M1FN). In the standard BF16 kernel, q_dtype == v_dtype so this works. In block-scaled QK where Q is FP4 but V is BF16, P (softmax output) must be BF16. Using FP4 for P creates wrong SMEM layout and recast → all NaN from corrupted softmax→PV pipeline.
Solution: Use `self.v_dtype if const_expr(self.block_scaled_qk) else self.q_dtype` for both `tP_layout` and `tSrP_r2t` dtype.
Constraints: Only affects inline integration where q_dtype != v_dtype (block-scaled QK with BF16 PV). Standalone FP4 kernel already uses v_dtype correctly.
Validation Evidence: NaN eliminated, cos=0.44 (remaining cos issue is S2T partition, not P dtype).
Source Rounds: 0
