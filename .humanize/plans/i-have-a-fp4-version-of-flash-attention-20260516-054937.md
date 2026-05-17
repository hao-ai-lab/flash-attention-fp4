# Block-Scaled Mixed Precision FP4/FP8 Integration Into Upstream FA4

## Goal Description

Integrate the proven block-scaled FP4/FP8 flash attention kernel (`flash_fwd_sm100_fp4.py`, ~3900 LOC) into the upstream `flash_fwd_sm100.py` (`/sgl-workspace/flash-attention-pr/`) as conditional `const_expr` code paths within the same `FlashAttentionForwardSm100` class. The integration must support four mixed-precision modes (NVFP4+BF16, NVFP4+FP8, MXFP8+FP8, NVFP4+FP8 with group-128 V scale) with zero BF16 regression and TFLOPS matching the standalone FP4 kernel (within 3% initially, optimized to match). A precision validation table (cosine_sim, max_diff, mean_diff) must be produced for all modes.

## Acceptance Criteria

- AC-1: BF16/FP16/FP8 forward paths have zero regression
  - Positive Tests:
    - Existing upstream test suite (`test_flash_attn.py`) passes unchanged
    - BF16 TFLOPS on (1,32768,24,128) within 1% of pre-integration baseline
    - Causal and non-causal BF16 produce identical output to pre-integration
  - Negative Tests:
    - Passing `mSFQ=None` must NOT activate block-scaled code paths
    - Block-scaled compile keys never collide with BF16/FP8 keys (compile both in same process without NaN)
- AC-2: NVFP4+BF16 produces correct output
  - Positive Tests:
    - cosine_sim >= 0.99 vs BF16 reference across all bench_fp4.py shapes
    - max_diff < 0.5 for random inputs with unit scale factors
    - Causal mask correctly applied (lower-triangle attention pattern)
  - Negative Tests:
    - Passing wrong SF layout (transposed) produces detectably wrong output
    - Missing mSFK when mSFQ is provided raises assertion error
  - AC-2.1: TFLOPS >= 1880 at (1,32768,24,128) non-causal
    - Positive: bench_fp4.py --qk_mode nvfp4 reports >= 1880 TF
    - Negative: Running with ex2_emu_freq=0 (suboptimal) gives lower TFLOPS
- AC-3: NVFP4+FP8 produces correct output with pure FP8 V (no V scale)
  - Positive Tests:
    - cosine_sim >= 0.99 vs BF16 reference
    - TFLOPS >= 1950 at (1,32768,24,128)
  - Negative Tests:
    - Passing BF16 V when FP8 PV mode is requested raises type error
    - Wrong sf_vec_size in compile key produces different kernel (cache miss, not collision)
- AC-4: MXFP8+FP8 produces correct output
  - Positive Tests:
    - cosine_sim >= 0.99 vs BF16 reference
    - TFLOPS >= 1890 at (1,32768,24,128)
    - sf_vec_size=32 correctly inferred from E8M0FNU scale dtype
  - Negative Tests:
    - MXFP8 with hdim=64 raises explicit "unsupported head_dim for MXFP8" error
    - MXFP8 with 2-CTA raises assertion
- AC-5: NVFP4+FP8(group-128 V scale) benchmark shootout completed
  - Positive Tests:
    - Both implementations built: (a) block-scaled PV MMA with pre-expanded group-32 SFV, (b) softmax_scale fusion (`softmax_scale_log2_eff = softmax_scale_log2 * qk_descale`)
    - Winner identified by TFLOPS comparison on bench_fp4.py shapes
    - Winner has cosine_sim >= 0.99 vs BF16 reference
    - Precision table includes both approaches
  - Negative Tests:
    - Pre-expansion produces valid per-32 scale layout (not garbage from wrong group boundary)
- AC-6: Full precision table logged
  - Positive Tests:
    - Table covers: NVFP4+BF16, NVFP4+FP8, MXFP8+FP8, NVFP4+FP8(group-128 winner)
    - Each row: shape, mode, cosine_sim, max_diff, mean_diff, TFLOPS
    - At least 8 shapes from bench_fp4.py included
    - Commit ID and command recorded per CLAUDE.md requirements
  - Negative Tests:
    - Table with NaN entries (indicating broken mode) fails review
- AC-7: Clean integration (no separate FP4 kernel file)
  - Positive Tests:
    - `flash_fwd_sm100_fp4.py` not imported by interface.py
    - All block-scaled logic gated by `const_expr(self.block_scaled_qk)`
    - Shared helpers in `blackwell_helpers.py` (not duplicated)
    - No disabled S2T paths, no debug skip flags, no env var overrides in final code
  - Negative Tests:
    - `grep "flash_fwd_sm100_fp4" interface.py` returns empty
    - No `os.getenv("FA4_...")` testing env vars remain in merged code
- AC-8: Compile key properly differentiates all modes
  - Positive Tests:
    - Compile NVFP4 then MXFP8 in same process: both produce correct output
    - Compile block-scaled then BF16: BF16 output unchanged
    - Key includes: QK ab_dtype, SF dtype, sf_vec_size, V dtype, mSFV presence, V group mode
  - Negative Tests:
    - Deliberately omitting sf_vec_size from key causes cache collision (wrong output)
- AC-9: Unsupported combinations fail explicitly
  - Positive Tests:
    - block-scaled + varlen raises NotImplementedError
    - block-scaled + paged_kv raises NotImplementedError
    - block-scaled + 2-CTA raises assertion
    - MXFP8 + hdim=64 raises ValueError
    - block-scaled + SplitKV raises NotImplementedError
  - Negative Tests:
    - Supported combinations (dense, causal, hdim=128, 1-CTA) do NOT raise

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The implementation includes all four mixed-precision modes (NVFP4+BF16, NVFP4+FP8, MXFP8+FP8, NVFP4+FP8 group-128) fully integrated into `flash_fwd_sm100.py` with const_expr gates, performance matching the standalone FP4 kernel within 1%, full precision table, explicit unsupported-mode guards, and benchmark-verified results across all bench_fp4.py shapes for both causal and non-causal.

### Lower Bound (Minimum Acceptable Scope)
The implementation includes NVFP4+BF16 and NVFP4+FP8 modes producing correct output (cosine_sim >= 0.99) with TFLOPS within 3% of standalone, inline in flash_fwd_sm100.py, with BF16 zero regression verified. MXFP8 and group-128 V scale may be partially complete with documented remaining work.

### Allowed Choices
- Can use: `const_expr` conditional paths, shared helpers in `blackwell_helpers.py`, `filter_zeros` for SF TMA partition, `blockscaled_utils` for layout generation, `cute.arch.exp2` for softmax (not `cute.math.exp2`)
- Cannot use: separate `flash_fwd_sm100_fp4.py` file as dispatch target, 2-CTA for block-scaled, env var debug overrides in production code, `cute.math.exp2(fastmath=True)` in softmax (causes ptxas scheduling regression)

## Feasibility Hints and Suggestions

### Conceptual Approach

**S2T MLIR Fix (Critical Path)**:
The S2T copy fails because `make_tmem_layout_sfa/sfb` produces a TMEM tensor whose tiler_mn is incompatible with `Cp4x32x128bOp`. The standalone FP4 kernel creates TMEM tensors the same way but succeeds. The fix likely involves:
1. Creating an isolated reproducer comparing TMEM tensor creation in both kernels
2. Checking if the TMEM base offset or pointer alignment differs
3. Verifying `cute.filter_zeros(tSF)` produces the same compact TMEM layout in both contexts

**Group-128 V Scale Shootout**:
- Approach A (block-scaled PV): Pre-expand group-128 scales to per-32 before kernel, load SFV via TMA, S2T to TMEM, use `make_blockscaled_trivial_tiled_mma` for PV
- Approach B (softmax_scale fusion): Compute `softmax_scale_log2_eff = softmax_scale_log2 * v_descale[head_idx]` — single multiply, no TMA/S2T overhead. Less precise (per-tensor vs per-group) but essentially free

### Relevant References
- `flash_attn/cute/flash_fwd_sm100_fp4.py` — proven FP4 implementation (donor code)
- `flash_attn/cute/debug/fp4_kernel_vs_upstream_investigation.md` — tuning results, ptxas fix documentation
- `/sgl-workspace/flash-attention-pr/flash_attn/cute/flash_fwd_sm100.py` — integration target (30+ commits of progress)
- `flash_attn/cute/blackwell_helpers.py` — `gemm_blockscaled_generic`, `gemm_ptx_partial_fp4`, `tcgen05_after_thread_sync`
- `flash_attn/cute/modified_utils/block_scaled_layout_test.py` — `make_smem_layout_sfa/sfb` (custom SF SMEM layouts)
- `flash_attn/cute/benchmarks/bench_fp4.py` — benchmark with do_bench, 10 shapes

## Dependencies and Sequence

### Milestones

1. **S2T Copy Fix**: Resolve MLIR legalization error in `make_s2t_copy`
   - Phase A: Create isolated reproducer comparing standalone FP4 vs PR TMEM creation
   - Phase B: Identify and fix the layout/pointer difference
   - Phase C: Verify S2T copy compiles and produces correct TMEM data

2. **NVFP4+BF16 End-to-End Correctness**: First mode producing correct attention output
   - Phase A: SF TMA loading (SFQ on Q barrier, SFK on KV barrier) — already working via filter_zeros
   - Phase B: S2T copies move SF from SMEM to TMEM (depends on Milestone 1)
   - Phase C: `gemm_blockscaled_generic` produces correct S matrix
   - Phase D: Softmax + BF16 PV GEMM produces correct output
   - Phase E: Precision validation (cosine_sim >= 0.99)

3. **NVFP4+FP8 PV Path**: Pure FP8 V without group scaling
   - Phase A: FP8 V TMA loading into separate sV buffer
   - Phase B: Standard PV MMA with FP8 V dtype
   - Phase C: Apply FP8 PV tuning (ex2_emu_freq=9, register budget)
   - Phase D: Performance verification (>= 1950 TF target)

4. **MXFP8+FP8 Path**: MXFP8 QK (sf_vec_size=32) with FP8 PV
   - Phase A: MXFP8 tuning config (ex2_emu_freq=10)
   - Phase B: Compile key differentiation from NVFP4
   - Phase C: Verify hdim=64 rejected, hdim=128 works

5. **Group-128 V Scale Shootout**: Implement and compare both approaches
   - Phase A: Block-scaled PV (pre-expand group-128 → per-32, SFV TMA+S2T, block-scaled PV MMA)
   - Phase B: Softmax_scale fusion (v_descale fused into scale_log2, single multiply)
   - Phase C: Benchmark both on all shapes, pick winner
   - Phase D: Document precision/perf tradeoffs

6. **Performance Tuning & Precision Table**: Match standalone FP4 kernel TFLOPS
   - Phase A: Port tuning configs (ex2_emu_freq, register budgets per mode)
   - Phase B: Benchmark all modes on all bench_fp4.py shapes
   - Phase C: Generate precision table (cosine_sim, max_diff, mean_diff)
   - Phase D: Optimize any modes with > 3% gap

Milestone 1 blocks Milestones 2-5. Milestones 3-4 depend on Milestone 2. Milestone 5 depends on Milestone 3. Milestone 6 depends on all prior milestones.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Create isolated S2T reproducer: compare TMEM layout creation in standalone FP4 vs PR kernel | AC-7 | analyze | - |
| task2 | Fix S2T copy MLIR legalization based on reproducer findings | AC-2 | coding | task1 |
| task3 | Verify SF TMA loading (SFQ on Q barrier, SFK on KV barrier) produces correct SMEM data | AC-2 | coding | - |
| task4 | End-to-end NVFP4+BF16 correctness: SF TMA + S2T + block-scaled QK gemm + softmax + BF16 PV | AC-2 | coding | task2, task3 |
| task5 | Add FP8 V path: separate sV buffer, FP8 PV MMA, FP8 PV tuning config | AC-3 | coding | task4 |
| task6 | Add MXFP8 QK path: sf_vec_size=32 config, compile key differentiation, hdim=64 guard | AC-4 | coding | task4 |
| task7 | Implement block-scaled PV with pre-expanded group-32 SFV (approach A) | AC-5 | coding | task5 |
| task8 | Implement softmax_scale fusion V descale (approach B) | AC-5 | coding | task5 |
| task9 | Benchmark shootout: compare approach A vs B on all shapes, pick winner | AC-5 | analyze | task7, task8 |
| task10 | Add unsupported-mode guards (varlen, paged KV, SplitKV, 2-CTA, MXFP8 hdim64) | AC-9 | coding | task4 |
| task11 | Verify compile key differentiates all modes (NVFP4/MXFP8/BF16 cache collision test) | AC-8 | analyze | task6 |
| task12 | Port tuning configs and optimize to match standalone FP4 TFLOPS | AC-2.1, AC-3, AC-4 | coding | task5, task6 |
| task13 | Generate full precision table (all modes, all shapes, cosine/max/mean) | AC-6 | coding | task9, task12 |
| task14 | BF16 regression test: verify existing tests pass, TFLOPS unchanged | AC-1 | analyze | task4 |
| task15 | Remove flash_fwd_sm100_fp4.py import from interface, clean dead code | AC-7 | coding | task13 |
| task16 | Final review: no env var overrides, no disabled paths, no debug flags | AC-7 | analyze | task15 |

## Claude-Codex Deliberation

### Agreements
- Dense-only v1 scope is correct (no varlen, paged KV, SplitKV, block sparsity for block-scaled)
- Separate K/V SMEM required when dtypes differ (FP4 K, BF16 V)
- SMEM budget must count K+V (not max) for block-scaled
- Compile key must include QK ab_dtype, SF dtype, sf_vec_size, V dtype, mSFV presence
- `filter_zeros` on SF SMEM before `cpasync.tma_partition` is the correct fix
- No disabled S2T paths or debug skip flags in final code
- `cute.arch.exp2` must be used (not `cute.math.exp2`) to avoid ptxas scheduling regression
- MXFP8 hdim=64 is explicitly unsupported

### Resolved Disagreements
- **TFLOPS ACs**: Claude proposed absolute numbers; Codex required benchmark env pinning. Resolution: within 3% initially (user decision), optimize to match. Benchmark env = B200, bench_fp4.py with do_bench.
- **Helper factoring**: Claude said "all inline"; Codex said "helpers in shared modules". Resolution: block-scaled MMA helpers already in `blackwell_helpers.py`; layout helpers in `modified_utils/`. Only const_expr gates are inline in the main kernel.
- **Group-128 V scale**: Claude proposed investigating; Codex raised sf_vec_size=128 infeasibility. Resolution: user confirmed pre-expand to group-32 (hardware limit). Implement both block-scaled PV and softmax_scale fusion, benchmark shootout.

### Convergence Status
- Final Status: `converged` (1 round, all REQUIRED_CHANGES accepted)

## Pending User Decisions

All decisions resolved during planning:
- DEC-1: Group-128 V scale → pre-expand to group-32, both approaches implemented + shootout. Decision Status: `RESOLVED`
- DEC-2: TFLOPS targets → within 3% initially, optimize to match standalone. Decision Status: `RESOLVED`
- DEC-3: MXFP8 hdim64 → explicitly unsupported. Decision Status: `RESOLVED`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead
- Block-scaled conditional paths use `if const_expr(self.block_scaled_qk):` — no runtime branching
- Tuning config keys follow pattern: `(is_causal, head_dim_padded, sf_vec_size)`
- SF tensor parameter names: `mSFQ`, `mSFK`, `mSFV` (matching existing codebase convention)

### Benchmark Environment
- GPU: NVIDIA B200
- Tool: `triton.testing.do_bench` via `bench_fp4.py`
- Shapes: 10 configurations from bench_fp4.py (b=1/4, s=256-32768, h=12/16/24/32, d=64/128)
- Warmup: 10 iterations, Rep: 25 iterations
- Clock: default (no frequency locking)

### Existing Integration Progress (flash_attn_pr branch)
- 30+ commits with structural integration
- BF16 path: verified zero regression
- SF TMA loading: GMEM→SMEM working (filter_zeros fix)
- S2T copy: SMEM→TMEM BLOCKED (MLIR legalization error — task1/task2)
- Interface dispatch: mSFQ/mSFK/mSFV params functional
- Block-scaled MMA atoms, TMEM SF tensor creation: working
- SharedStorage with separate sV, sSFQ, sSFK: working

---

--- Original Design Draft Start ---

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

- **S2T Copy MLIR Legalization**: The `make_s2t_copy(Cp4x32x128bOp, tCtSF_compact)` currently fails with "failed to legalize unresolved materialization" — the TMEM layout from `make_tmem_layout_sfa/sfb` produces tiler_mn sizes incompatible with the S2T atom
- **Register pressure**: FP4 and BF16 have different register tuning (FP4: 192/80 vs upstream BF16: 192/72 for causal 128)
- **2-CTA Incompatibility**: FP4 is 1-CTA only; upstream's 2-CTA support must not leak into block-scaled path
- **ptxas scheduling**: Different code paths can cause 52% stall_long_sb increase if not careful

--- Original Design Draft End ---
