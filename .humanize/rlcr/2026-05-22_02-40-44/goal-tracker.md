# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

Integrate the proven block-scaled FP4/FP8 flash attention kernel (`flash_fwd_sm100_fp4.py`, ~3900 LOC) into the upstream `flash_fwd_sm100.py` (`/sgl-workspace/flash-attention-pr/`) as conditional `const_expr` code paths within the same `FlashAttentionForwardSm100` class. The integration must support four mixed-precision modes (NVFP4+BF16, NVFP4+FP8, MXFP8+FP8, NVFP4+FP8 with group-128 V scale) with zero BF16 regression and TFLOPS matching the standalone FP4 kernel (within 3% initially, optimized to match). A precision validation table (cosine_sim, max_diff, mean_diff) must be produced for all modes.

## Acceptance Criteria

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->
<!-- Claude must extract or define these in Round 0 -->


- AC-1: BF16/FP16/FP8 forward paths have zero regression — baseline is the "Final tuned results on fp4-rebase" table in `fp4_kernel_vs_upstream_investigation.md` (commit `a21acbe7`)
  - Positive Tests:
    - Existing upstream test suite (`test_flash_attn.py`) passes unchanged
    - BF16 TFLOPS on (1,32768,24,128) within 1% of pre-integration baseline (1545 TF per investigation table)
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
    - Positive: bench_fp4.py --qk_mode nvfp4 reports >= 1880 TF (investigation table: 1887)
    - Negative: Running with ex2_emu_freq=0 (suboptimal) gives lower TFLOPS
- AC-3: NVFP4+FP8 produces correct output with pure FP8 V (no V scale)
  - Positive Tests:
    - cosine_sim >= 0.99 vs BF16 reference
    - TFLOPS >= 1950 at (1,32768,24,128) (investigation table: 2018)
  - Negative Tests:

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |

#### Active Tasks
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| task1: S2T reproducer (standalone vs inline) | AC-7 | completed | analyze | claude | Identified P dtype as NaN root cause |
| task2: Fix S2T MLIR + NaN | AC-2 | in_progress | coding | claude | MLIR fixed, NaN fixed (P dtype), cos=0.44 remaining |
| task3: Verify SF TMA loading | AC-2 | in_progress | coding | claude | S2T makes cos worse (0.44 vs 0.61 without S2T) |
| task4: NVFP4+BF16 end-to-end | AC-2 | blocked | coding | claude | Blocked on S2T cos issue |
| task5: FP8 V path | AC-3 | pending | coding | claude | Depends on task4 |
| task6: MXFP8 QK path | AC-4 | pending | coding | claude | Depends on task4 |
| task7: Verify softmax fusion (approach b) | AC-5 | pending | coding | claude | Depends on task5 |
| task8: Unsupported-mode guards | AC-9 | pending | coding | claude | |
| task9: Compile key verification | AC-8 | pending | analyze | codex | Depends on task6 |
| task10: Port tuning configs | AC-2.1,3,4 | pending | coding | claude | Depends on task5,6 |
| task11: Full precision table | AC-6 | pending | coding | claude | Depends on task7,10 |
| task12: BF16 regression test | AC-1 | pending | analyze | codex | Depends on task4 |
| task13: Remove fp4 import, clean code | AC-7 | pending | coding | claude | Depends on task11 |
| task14: Final review | AC-7 | pending | analyze | codex | Depends on task13 |

### Blocking Side Issues
<!-- Only issues that directly block current mainline progress belong here -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|

### Queued Side Issues
<!-- Non-blocking issues stay queued and must NOT replace the round objective -->
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|

