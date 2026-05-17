# Block-Scaled Mixed Precision Integration Into Upstream FA4 Kernel

## Original Idea

I have a fp4 version of flash attention 4 in examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/flash_fwd_sm100_fp4.py and a dev doc in examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/debug/fp4_kernel_vs_upstream_investigation.md. Now I want to integrate this into the upstream main flash_fwd_sm100.py in /sgl-workspace/flash-attention-pr with full mixed precision fp4/fp8 functionalities. The goal is to intergrate it into the flash-attention-pr with full functionality and tflops preseved as in the "**Full results (bench_fp4.py, triton do_bench, B200):**" table of the .md with zero regressions. The flash_fwd_sm100.py uses an optional block scale of group size 128 for pv fp8, so we need to support that and add a column "NVFP4+FP8(group 128)" with near the same tflops as without the block scale. Do your best to optimize it. when you are done, also log a full precision table for the columns. The integration should be as clean as performant as possible.

## Primary Direction: Inline Integration Architecture

### Rationale

Explores how to structurally merge the 3900-line FP4 kernel code into the existing 3100-line flash_fwd_sm100.py using conditional const_expr paths within the same class, extending patterns already present in the upstream skeleton.

### Approach Summary

Merge the FP4 kernel (flash_fwd_sm100_fp4.py) into upstream flash_fwd_sm100.py by filling in the existing block-scaled skeleton code (already present but dead) with the proven FP4 implementation:

1. **Unified Constructor with Block-Scaled Parameters**: Extend `FlashAttentionForwardSm100.__init__()` using `sf_vec_size`/`sf_dtype` parameters (already accepted at upstream lines 157-158). When None, use standard BF16/FP16 path; when set, enable block-scaled QK MMA.

2. **Conditional SMEM/Register Tuning**: Expand `_TUNING_CONFIG` dict with block-scaled keys `(is_causal, head_dim_padded, sf_vec_size)` mapping to register counts and exp2 frequency. Apply via `const_expr` dispatch at `__call__` time.

3. **const_expr Conditional Code Paths**: Use `if const_expr(self.block_scaled_qk)` to branch on MMA construction (`make_blockscaled_trivial_tiled_mma` vs `make_trivial_tiled_mma`), SMEM layout generation, SharedStorage variants, TMA load ops, and softmax methods.

4. **Shared Helper Functions**: Move quantization methods (`_quant_fp4`, `_pack_fp8`, `_apply_exp2_pack_fp8`) and block-scaled MMA logic (`gemm_blockscaled_generic`, `gemm_ptx_partial_fp4`) into the class, conditioned on `self.block_scaled_qk`.

5. **Single Export**: Replace two separate imports (`FlashAttentionForwardSm100` + `FlashAttentionForwardSm100FP4`) with one unified class. Update interface.py dispatch to pass `sf_vec_size`/`sf_dtype` instead of importing separate implementations.

Note: Integration is already in progress on the `flash_attn_pr` branch (commit 6230f99+). The SF TMA loading works (via `filter_zeros` fix), but S2T copy (SMEM->TMEM) has an MLIR legalization error that needs resolution. The BF16 path has verified zero regression.

### Objective Evidence

- `/sgl-workspace/flash-attention-pr/flash_attn/cute/flash_fwd_sm100.py` lines 157-158: `sf_vec_size` and `sf_dtype` constructor parameters already accepted
- `/sgl-workspace/flash-attention-pr/flash_attn/cute/flash_fwd_sm100.py` line 338: `self.block_scaled_qk = sf_vec_size is not None` flag already defined
- `/sgl-workspace/flash-attention-pr/flash_attn/cute/flash_fwd_sm100.py` lines 104-120: `_BLOCK_SCALED_TUNING_CONFIG` and `_BLOCK_SCALED_FP8PV_TUNING_CONFIG` dicts already present with validated entries
- `/sgl-workspace/flash-attention-pr/flash_attn/cute/flash_fwd_sm100.py` lines 537-546: `const_expr(self.block_scaled_qk)` branch for blockscaled `tiled_mma_qk` construction already present
- `/sgl-workspace/flash-attention-pr/flash_attn/cute/flash_fwd_sm100.py` lines 590-599: Block-scaled SMEM layout generation (`make_smem_layout_sfa/sfb`) already present
- `/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/flash_fwd_sm100_fp4.py` lines 73-93: Proven tuning configs (ex2_emu_freq=16 for NVFP4+BF16, freq=9 for NVFP4+FP8, freq=10 for MXFP8+FP8)
- `/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/softmax.py` line 385: `scale_groupwise()` method already present on `SoftmaxSm100`
- `/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/blackwell_helpers.py` lines 1145-1525: `gemm_blockscaled_generic`, `gemm_ptx_partial_fp4`, `packed_float_to_ue4m3` already ported to upstream

### Known Risks

- **Const-expr complexity**: Embedding 800+ lines of FP4-specific logic into conditional branches risks ptxas scheduler artifacts if branches diverge too much (as noted in debug doc: different code paths caused 52% stall_long_sb increase)
- **Register pressure**: FP4 and BF16 have different register tuning (FP4: 192/80 vs upstream BF16: 192/72 for causal 128). Combined kernel could exceed 512-reg budget if not carefully balanced
- **S2T Copy MLIR Legalization**: The `make_s2t_copy(Cp4x32x128bOp, tCtSF_compact)` currently fails with "failed to legalize unresolved materialization" — the TMEM layout from `make_tmem_layout_sfa/sfb` produces tiler_mn sizes incompatible with the S2T atom
- **2-CTA Incompatibility**: FP4 is 1-CTA only; upstream's 2-CTA support must not leak into block-scaled path
- **SF TMA Layout**: `sSFK` from `get_tensor(staged_layout)` has zero-stride modes requiring `filter_zeros` before `cpasync.tma_partition` (already fixed)

## Alternative Directions Considered

### Alt-1: SF Pipeline & TMA Strategy
- Gist: Focuses on how scale factor tensors (SFQ, SFK, SFV) flow through the kernel's TMA pipeline — from GMEM through SMEM to TMEM via S2T copies — fitting into the upstream's existing `pipeline_q`/`pipeline_kv` barrier infrastructure. Adds dedicated `mbar_sfqk_load_offset` and `mbar_sfpv_load_offset` barriers for SF synchronization, with SF TMA bytes riding on existing Q/KV barriers.
- Objective Evidence:
  - `flash_fwd_sm100_fp4.py` lines 990-1007: Dedicated mbarrier offset allocation for SF pipeline
  - `flash_fwd_sm100_fp4.py` lines 2251-2266: Consumer barrier wait pattern for S2T copy synchronization
  - `flash-attention-pr/flash_fwd_sm100.py` lines 1093-1130: Standard pipeline_q/pipeline_kv TmaUmma patterns
  - S2T copy cost is zero on critical path (executes in parallel with softmax output scaling)
- Why not primary: Higher technical risk due to the still-unresolved MLIR S2T legalization error; the integration architecture direction addresses the broader structural question first.

### Alt-2: Group-128 PV Block Scale Integration
- Gist: Extends the existing upstream `DescaleTensors.v_descale` mechanism to support optional group-level (group size 128) scale factors for V in the PV GEMM path. Leverages the existing `gemm_ptx_partial_fp4` pathway which already accepts `tScaleA`/`tScaleB` for block-scaled MMA, applying the same pattern to PV with `make_blockscaled_trivial_tiled_mma()` for tiled_mma_pv.
- Objective Evidence:
  - `flash_fwd_sm100_fp4.py` lines 2154-2162: `gemm_ptx_partial_fp4` already accepts tScaleA/tScaleB for PV
  - `flash_fwd_sm100_fp4.py` lines 1528-1559: SFV TMEM tensor construction via `make_tmem_layout_sfb()`
  - No CUTLASS precedent for sf_vec_size=128 (all examples use 16 or 32)
- Why not primary: Novel direction with no codebase precedent for group-128; depends on the core integration being complete first.

### Alt-3: Performance Tuning Methodology
- Gist: Ports verified tuning knobs (ex2_emu_freq, register budgets, kv_stage caps) into the upstream's `_TUNING_CONFIG` structure. Applies the critical `cute.arch.exp2` vs `cute.math.exp2` ptxas scheduling fix. Uses hierarchical config lookup with per-mode (NVFP4+BF16, NVFP4+FP8, MXFP8+FP8) entries keyed by `(is_causal, head_dim_padded, sf_vec_size)`.
- Objective Evidence:
  - Debug MD lines 336-394: Benchmark evidence showing 2018 TF peak with verified configs
  - Debug MD lines 159-182: ptxas scheduling fix documentation (`cute.arch.exp2` resolves 23.6% regression)
  - Upstream lines 104-120: Skeleton tuning dicts already present with correct key structure
  - FP4 kernel lines 73-93: Validated config values ready to port
- Why not primary: Tuning is applied after the structural integration is complete; it's a refinement step, not the foundational architecture.

### Alt-4: Interface & Dispatch Design
- Gist: Designs how mSFQ/mSFK/mSFV parameters flow from public `flash_attn_func()` through `FlashAttnFunc.forward`, `_flash_attn_fwd`, compile_key generation, cute tensor conversion, and kernel `__call__`. Handles FP4 dtype detection (`torch.float4_e2m1fn_x2`), sf_vec_size inference, and FP4→uint8 runtime conversion.
- Objective Evidence:
  - Upstream interface.py lines 250-258: `torch2cute_dtype_map` already registers float4_e2m1fn_x2
  - Upstream interface.py lines 704-710: _sf_dtype/_sf_vec_size inference logic exists
  - FP4 interface.py lines 598-640: Detailed SF dtype/vec_size inference for NVFP4(16) vs MXFP8(32)
  - Upstream interface.py lines 717-765: compile_key already includes block-scaled flags
- Why not primary: Mostly plumbing work that follows from the kernel architecture decisions; lower technical novelty.

### Alt-5: Precision Validation Framework
- Gist: Builds a modular precision measurement framework extending bench_fp4.py's existing max_diff/mean_diff structure to report cosine_similarity, max_diff, and mean_diff per mode (NVFP4+BF16, NVFP4+FP8, NVFP4+FP8(group128), MXFP8+FP8) in a markdown table. Uses flashinfer dequantization utilities for FP32 reference generation.
- Objective Evidence:
  - `bench_fp4.py` lines 642-670: Existing max_diff/mean_diff comparison structure
  - `debug/fp4_kernel_vs_upstream_investigation.md` lines 282-301: Multi-mode precision table precedent
  - flashinfer `bench_mxfp4_quantize_backend_comparison.py` lines 96-127: cosine_similarity validation pattern
  - flashinfer `fp4_quantization.py`: nvfp4_dequantize available for reference generation
- Why not primary: Measurement infrastructure, not core implementation; can be built independently after the kernel integration.

## Synthesis Notes

The primary direction (Inline Integration Architecture) is the foundation that all other directions build upon. The most impactful elements from the alternatives that should fold into the primary implementation are: (1) the `filter_zeros` fix for SF TMA partition from Alt-1 (already proven working), (2) the hierarchical tuning config structure from Alt-3 (validated configs ready to port), and (3) the interface dispatch plumbing from Alt-4 (partially complete in flash_attn_pr branch). The group-128 PV block scale (Alt-2) is the most novel feature requirement and should be tackled as a second phase after the core QK integration produces correct output. The precision framework (Alt-5) should be built last as the validation gate. A key insight from the existing progress: the `get_tensor` → `filter_zeros` → `cpasync.tma_partition` pattern resolves the SF SMEM/GMEM shape mismatch, and the S2T MLIR legalization error is the remaining critical blocker that likely requires matching the TMEM layout structure exactly as the standalone FP4 kernel creates it.
