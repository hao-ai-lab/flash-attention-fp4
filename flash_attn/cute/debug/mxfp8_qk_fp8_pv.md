# MXFP8 QK + FP8 PV — status and plan

**Last merged:** 2026-04-15 (combines the checklist and the progress note)
**Last verified commit:** `7761bf12`

## Goal

Extend [`flash_fwd_sm100_fp4.py`](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/flash_fwd_sm100_fp4.py)
so the QK GEMM can run **block-scaled MXFP8** (Float8E4M3FN / Float8E5M2 + UE8M0
SF, `sf_vec_size=32`) while the PV GEMM runs **pure FP8** (no block-scale, no
P-quant in softmax). Verify end-to-end via
[`bench_fp4.py`](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/benchmarks/bench_fp4.py).

Motivation: the current `--quant_v` path puts P-quantization on the softmax
critical path. Moving to pure FP8 PV removes that, and MXFP8 QK should be
roughly precision-equivalent to NVFP4 QK while letting us drop block-scaled
MMA from both GEMMs.

Reference prior art:
- [`fp4_flash_attention_optimization_notes.md`](./fp4_flash_attention_optimization_notes.md)
- [`qkvp/QKVP_PRECISION_FIX.md`](./qkvp/QKVP_PRECISION_FIX.md)
- Upstream FA PR for pure FP8: https://github.com/Dao-AILab/flash-attention/pull/2109

## Exploration findings

- `interface.py` already accepts `Float8E4M3FN` / `Float8E5M2` A/B with
  `Float8E8M0FNU` SF and `sf_vec_size=32` for block-scaled paths.
- `make_blockscaled_trivial_tiled_mma` emits `MmaMXF8Op` for block-scaled FP8;
  `make_trivial_tiled_mma` emits `MmaFP8Op` for pure FP8.
- NVFP4-only assumptions that needed relaxing:
  - `__init__` asserted `sf_vec_size == 16` and `sf_dtype == Float8E4M3FN`
  - scale-factor SMEM storage hardcoded to `cute.Float8E4M3FN`
  - P-SF register path hardcoded to UE4M3 for block-scaled PV
- Inline PTX helpers in
  [`blackwell_helpers.py`](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/blackwell_helpers.py)
  hardcoded MMA `.kind::` for some paths (`.kind::f16` for BF16,
  `.kind::mxf4nvf4.block_scale.scale_vec::4X` for NVFP4).

## Implementation status

- [x] Relaxed `FlashAttentionForwardSm100.__init__` dtype assumptions.
  Block-scaled QK now supports:
  - NVFP4: `Float4E2M1FN + Float8E4M3FN + sf_vec_size=16`
  - MXFP8: `Float8E4M3FN` / `Float8E5M2` + `Float8E8M0FNU + sf_vec_size=32`
- [x] Replaced hardcoded FP8-E4M3 SMEM SF storage with `self.sf_dtype`.
- [x] Kept the existing on-the-fly P-quant path only for block-scaled PV.
- [x] Added a pure-FP8 PV path:
  - V is FP8
  - `mSFV is None`
  - softmax writes P directly in FP8 form for the PV GEMM
  - PV uses pure FP8 MMA (not block-scaled)
- [x] Generalized PTX-helper selection so the emitted MMA kind matches the op:
  `.kind::f16` for F16/BF16, pure FP8 kind for `MmaFP8Op`,
  `.kind::mxf8f6f4…` for block-scaled FP8, `.kind::mxf4nvf4…` for NVFP4.
- [x] Benchmark driver can exercise QK-only NVFP4 (baseline), MXFP8 QK + BF16 V,
  and MXFP8 QK + pure FP8 V.
- [x] Benchmark builds FP8 V tensors with the requested `--fp8_dtype` (was
  previously reusing the NVFP4 QK dtype by accident).
- [x] Softmax underflow handling for pure FP8 PV switched to the upstream-style
  `max_offset=8 / p_log2_offset=8` pattern (LSE subtracts the offset back out
  after `row_sum` accumulation).
- [x] P TMEM store shape for pure-FP8 PV: uses `St32x32bOp(Repetition(8))`
  (keying that off `quant_pv` was wrong for `NVFP4 QK + FP8 PV`; caused long-
  sequence sparse NaNs before the fix).
- [x] Explicit E4M3 packing for P: bypasses the generic `.to(Float8E4M3FN)`
  path and uses `packed_float_to_ue4m3(...)` directly from FP32. Saved PTX
  diff confirmed removal of `128x cvt.u32.u16` after the switch.
- [x] SFB helper tiled MMA uses `CtaGroup.ONE` (was `self.cta_group`).
- [x] MXFP8 QK SFQ/SFK TMEM spacing uses dense-style `16` u32-column separation
  (`0x10`) instead of the `4`-column split; that fixed one misalignment but
  not the current fault.
- [x] **Routed `gemm_ptx_partial_fp8` through the working generic block-scaled
  helper** — unblocks MXFP8 QK functionally. Both `MXFP8 QK + BF16 PV` and
  `MXFP8 QK + FP8 PV` now run without `cudaErrorMisalignedAddress`.

## Current blockers

### 1. Is the generic helper for MXFP8 QK actually optimal?

The dedicated `gemm_ptx_partial_fp8` helper faulted, so we fall back to the
generic block-scaled helper. This is functional but may be leaving perf on
the table. The dedicated helper should be revived once the real fault is
understood.

### 2. Pure-FP8 PV regression on long d=128 shapes

NVFP4 QK + FP8 PV beats NVFP4 QK baseline on `(1, 32768, 24, 64)` but not on
long-d shapes. Nsight confirms the FP8 path reduces DRAM and L2 traffic but
takes **more active cycles**, with higher `mio_throttle` and `wait` stalls —
the bottleneck shifted from memory to execution-side overhead (packing /
scheduling / pipeline balance), not raw BW.

## Current numbers (`bench_fp4.py`, repaired env)

Env to reproduce:
- `flash_attn/cute/.venv/bin/python`
- `CUTE_DSL_ARCH=sm_100a`, `CUTE_DSL_ENABLE_TVM_FFI=1`
- `PYTHONPATH=/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention:/sgl-workspace/flashinfer`

### NVFP4 QK-only baseline (reference)

| shape | ms | TFLOPS |
|---|---|---|
| `(1, 256, 16, 128)` | 0.014–0.015 | 37.1 |
| `(1, 1024, 16, 128)` | 0.023 | 376.8 |
| `(4, 4096, 16, 128)` | 0.330–0.336 | 1667.5 |
| `(4, 4096, 32, 128)` | 0.647–0.654 | 1699.0 |
| `(1, 4096, 12, 128)` | 0.102–0.105 | 1006.6 |
| `(1, 32768, 12, 128)` | 3.830–3.858 | 1722.6 |
| `(1, 4096, 24, 128)` | 0.148–0.153 | 1394.9 |
| `(1, 32768, 24, 128)` | 7.294–7.643 | 1808.9 |
| `(1, 32768, 24, 64)` | 7.138–7.176 | 924.2 |

### NVFP4 QK + pure FP8 PV (current best tuning: P-split=1/2, uncapped kv_stage)

| shape | ms | Δ vs baseline |
|---|---|---|
| `(1, 256, 16, 128)` | 0.013 | faster |
| `(1, 1024, 16, 128)` | 0.023 | tied |
| `(4, 4096, 16, 128)` | 0.343–0.344 | +0.008 ms |
| `(4, 4096, 32, 128)` | 0.673–0.676 | +0.022 ms |
| `(1, 4096, 12, 128)` | 0.106–0.107 | tied |
| `(1, 32768, 12, 128)` | 4.062–4.078 | +0.21 ms (worse) |
| `(1, 4096, 24, 128)` | 0.154–0.162 | ±0.01 ms |
| `(1, 32768, 24, 128)` | 7.736–7.807 | +0.1 ms |
| `(1, 32768, 24, 64)` | 6.945–7.641 | **−0.23 ms (faster)** |

Correctness against BF16 reference (max_diff) over the sweep:
- d=128 shapes: 0.05 – 0.51
- d=64 shapes: ~0.06

All configurations complete without NaNs. No cases exceed the benchmark
tolerance.

### MXFP8 QK — IS NOT actually unblocked (2026-04-15 late, post-bench)

The helper reroute stopped the `cudaErrorMisalignedAddress` crash but did
**not** restore correctness. End-to-end bench reveals two distinct bugs:

**A. Numerical: every MXFP8 QK config produces garbage output.**

Sample run on `MXFP8 QK + BF16 PV` (default `--qk_mode mxfp8`, `Float8E4M3FN`):

| shape | ms | max_diff vs BF16 |
|---|---|---|
| `(1, 256, 16, 128)`   | 0.015 | **6.9e37** |
| `(1, 1024, 16, 128)`  | 0.026 | **2.9e37** |
| `(4, 4096, 16, 128)`  | 0.350 | **NaN** |
| `(4, 4096, 32, 128)`  | 0.686 | **1.3e37** |
| `(1, 4096, 12, 128)`  | 0.111 | **NaN** |
| `(1, 32768, 12, 128)` | 4.016 | **NaN** |
| `(1, 4096, 24, 128)`  | 0.160 | **NaN** |
| `(1, 32768, 24, 128)` | 8.285 | **NaN** |

Same NaN/inf pattern with `--pv_mode fp8`. Latency numbers themselves are
plausible (1500–1700 TFLOPS, ~10–15% slower than NVFP4 baseline as expected
from MXFP8 having 2× wider operands), so the hot loop is running — it's
just computing the wrong thing.

**B. Compile-time: d=64 crashes during TMA SFQ atom creation.**

`(1, 32768, 24, 64)` with MXFP8 QK fails with:

```
loc("tma_atom_sfq, tma_tensor_sfq = cute.nvgpu.make_tiled_tma_atom_A("...
flash_fwd_sm100_fp4.py":759:39): error: expected top-level shape
equivalence between the SMEM layout and the CTA V-map, but got
'!cute.layout<"((((32,4),1),(32,1)),1,(2,4)):(((( 16,4),0),(0,0)),0,(0,1))">'
and
'!cute.layout<"(((32,4),32),1,2):(((1@0@0@0,1@1@0@0),1@0@0@1),0,1@1@0@1)">'
```

The two layouts both nominally describe the same total atom but disagree on
how the K-rest dim is grouped. The "32" in the actual layout is the K-rest
that's missing from the expected one — likely an `sf_vec_size=32` (MXFP8) vs
`sf_vec_size=16` (NVFP4) accounting bug in how the SFQ TMA SMEM layout is
built when `head_dim=64`.

Both bugs (numerical garbage on d=128, compile fault on d=64) point to the
same root cause: the SFQ/SFK plumbing was tuned for NVFP4's `sf_vec_size=16`
and doesn't correctly scale when `sf_vec_size=32`. Need to audit:
- `sfq_smem_size` / `sfk_smem_size` derivation
- `tile_atom_to_shape_SF` and `make_smem_layout_sfa/sfb` calls for MXFP8
- the dense_blockscaled_gemm_persistent_prefetch.py's MXFP8 path for the
  reference layout

## Nsight Compute findings

### `(4, 4096, 16, 128)` NVFP4 QK + BF16 vs + FP8 PV

- Tensor-pipe instructions stayed at `393,216` → PV MMA count already reduced.
- FP8 PV: `138.7M → 130.3M` total instructions after explicit E4M3 packing
  (saved non-tensor conversion/pack work).
- FP8 PV still launches at `128 regs/thread`, limited to 1 resident block by
  both register and SMEM budget.

### `(1, 32768, 12, 128)` BF16 vs FP8 PV

| metric | BF16 PV | FP8 PV |
|---|---|---|
| `gpu__time_duration.sum` | 6,337,600 | 6,645,504 |
| `dram__bytes.sum` | 235,973,376 | **175,292,416** |
| `lts__t_bytes.sum` | 14,943,254,528 | **8,322,595,552** |
| `smsp__cycles_active.sum` | 3,909,509,413 | 4,143,321,907 |
| `mio_throttle` | 0.33 | **0.88** |
| `wait` | 2.22 | **2.74** |

### `(1, 32768, 24, 64)` BF16 vs FP8 PV

| metric | BF16 PV | FP8 PV |
|---|---|---|
| `gpu__time_duration.sum` | 11,658,816 ns | 12,406,688 ns |
| `dram__bytes.sum` | 236,132,864 | 178,097,408 |
| `lts__t_bytes.sum` | 14,000,885,216 | 9,377,248,992 |
| `smsp__cycles_active.sum` | 7,676,213,467 | 8,191,504,535 |
| `shared_op_ld` bank conflicts | 1,004,597 | **35,413** |
| `shared_op_st` bank conflicts | 3,695,857 | **1,687,420** |
| SMEM / block | 217,088 | 222,208 |

Both datasets tell the same story: **memory pressure goes down, active cycles
go up.** The regression is execution-side, not bandwidth-side.

## Negative / reverted experiments

- **Single-phase pure-FP8 P handoff** (publish full P before first PV barrier):
  large regressions on every long-shape (`0.407 ms`, `0.801 ms`, `4.915 ms`,
  `9.351 ms`, `7.826 ms`). Reverted.
- **`TMEM_STORE_REP=16` for pure-FP8 P store**: `~0.451 ms` on the key shape
  vs `~0.344 ms` with `Repetition(8)`. Kept `Repetition(8)`.
- **E5M2 for pure-FP8 PV**: `~0.350 ms` vs E4M3 `~0.344 ms`. Kept E4M3.
- **`FA4_FP8_PV_KV_STAGE_CAP` cap on `(1, 32768, 24, 64)`**: uncapped 8.75 ms;
  cap=8 → 17.79 ms; cap=5 → 18.93 ms. Disproved the cap hypothesis.

## Best current tuning (pure FP8 PV)

- `kv_stage` uncapped by default.
- Release P to the PV consumer at `1/2` split (not `3/4`).
- `TMEM_STORE_REP = Repetition(8)`.
- E4M3 operand dtype.
- Explicit `packed_float_to_ue4m3(...)` pack, not generic `.to()`.
- Shape-dependent knobs:
  - `d=64` prefers lower `p_log2_offset`, smaller TMEM store rep, different P-split.
  - long `d=128` prefers uncapped KV stage + `p_log2_offset=8`.

## Verification matrix

| mode | functional | numerical (BF16 ref) | perf vs NVFP4 QK baseline |
|---|---|---|---|
| NVFP4 QK-only | ✓ | ✓ | reference |
| NVFP4 QK + pure FP8 PV | ✓ | ✓ (max_diff ≤ 0.51, no NaN) | mixed; wins on d=64, ties on small d=128, **regresses on long d=128** |
| MXFP8 QK + BF16 PV | ✓ (after helper reroute) | **pending** | **pending** |
| MXFP8 QK + FP8 PV | ✓ (after helper reroute) | **pending** | **pending** |

## Next steps (ordered)

1. **Re-run full bench sweep in all three new modes** with current code to
   establish numbers for MXFP8 QK + BF16/FP8 PV now that the crash is gone.
2. **Investigate FP8 PV `mio_throttle=0.88` / `wait=2.74`** on long `d=128`.
   Memory went down, cycles went up — so the extra work is in issue/scheduling,
   packing, or pipeline balance. Look at PTX/SASS for the long-d case the same
   way the Rep-32 vs Rep-64 `LDTM` investigation did.
3. **Revive the dedicated `gemm_ptx_partial_fp8` helper** (currently bypassed
   for the generic block-scaled helper). The dedicated one should be faster
   once the layout mismatch that made it fault is understood — compare FA
   SFQ/SFK TMA + TMEM plumbing against the working standalone dense
   `dense_blockscaled_gemm_persistent_prefetch.py --ab_dtype Float8E4M3FN
   --sf_dtype Float8E8M0FNU --sf_vec_size 32`.
4. **Verify PTX-helper / idesc agreement** on the new paths (QK block-scaled
   FP8, PV pure FP8) — the checklist marks this `~`.
5. **Recheck stride / descriptor assumptions** wherever byte-based spacing
   differs between BF16, NVFP4 and FP8. Several bugs so far have been from
   copy-pasted constants (SFP R2S base offset, TMEM u32-column spacing).

## Bottom line

- NVFP4 QK baseline: still the reference; peak 1805 TFLOPS on `(1, 32768, 24, 128)`.
- NVFP4 QK + pure FP8 PV: functional, numerically clean, but not yet a
  universal win — long `d=128` still regresses due to execution-side overhead.
- MXFP8 QK: **was** blocked on `cudaErrorMisalignedAddress`, now unblocked by
  routing through the generic block-scaled helper; numbers still need to be
  collected.
- The two real open questions are:
  - Is routing through the generic helper costing us perf on MXFP8 QK, and
    can the dedicated helper be revived?
  - Why does FP8 PV use more active cycles on long `d=128` despite lower
    memory traffic?

## 2026-04-15 update — MXFP8 NaN/inf still open

### Bugs diagnosed and fixed

1. `sf_layout_kwargs` only passed `mma_tile_inst_k` for NVFP4 (sf_vec_size 16).
   MXFP8 `d=64` was failing to compile with a TMA SFQ atom shape mismatch
   because the helper default (`4`) didn't match the required value (`2`).
   Fix: always thread the computed `mma_inst_tile_k` through regardless of
   `sf_vec_size`.

2. Compile-cache collision: `_flash_attn_fwd.compile_cache` keyed without
   QK ab_dtype / sf_dtype. An NVFP4 compile could land in the slot later
   looked up for MXFP8 (or vice-versa), returning a kernel whose PTX emitted
   `mxf4nvf4` when we needed `mxf8f6f4`. Fix: added `_key_qk_ab_dtype` and
   `_key_sf_dtype` to `compile_key`. Verified post-fix PTX now emits
   `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X` for
   MXFP8 and `...kind::mxf4nvf4.block_scale.scale_vec::4X` for NVFP4.

3. Inline-PTX helper `gemm_ptx_partial_fp4` computed per-K SF offset using
   `find_tmem_tensor_col_offset`, which returns the slice **cosize**, not
   the per-K offset. So `offset_sfa[k] = 0` for every `k`, meaning every
   block-scaled MMA inst read SFs from `k=0`. NVFP4 tolerated this (K=1 for
   our head-dims, or K=2 with scale_vec::4X which is FP-tolerant). MXFP8
   with K=4 scale_vec::1X amplifies the bug to 1e37 garbage.
   Fix attempt: refactored the helper to pass each K-slice's
   `tScaleA[None, None, k].iterator.toint()` as a separate `r` input so the
   PTX `[tmem_scale_a]` operand sees the right TMEM address per K-iter.
   NVFP4 max_diff stays `~0.03`, so the refactor preserved correctness, but
   **MXFP8 is still NaN/inf** — which means the root cause is NOT the
   per-K SF offset alone.

### Generic `cute.gemm` fallback also fails the same way

Added `sm100_utils.gemm_blockscaled_generic` that mirrors
`dense_blockscaled_gemm_persistent.py`: build `mma_atom = make_mma_atom(op)`,
for each k `mma_atom.set(Field.SFA/SFB, tScaleA[...,k].iterator)`, then
`cute.gemm(mma_atom, acc, tCrA[...,k], tCrB[...,k], acc)`. Wired through
`FA4_DEBUG_FORCE_GENERIC_MXFP8_QK=1`.

Result on `--qk_mode mxfp8 --debug` (Q=K=V=1.0, SF=1.0):
MXFP8 output is `1.3e36`, `2.6e36`, ..., NaN — **same as the inline-PTX path**.

This rules out the inline-PTX helper as the bug source. The block-scaled
MMA is reading SF bytes from uninitialized/garbage SMEM, not the `0x7F`
bytes we wrote. The bug is upstream — in SFQ/SFK smem layout, TMA partition,
or S2T copy for `sf_vec_size=32`. CUTLASS's
`make_smem_layout_sfa` uses the same `BlockScaledBasicChunk` atom for
`sf_vec_size in {16, 32}`; the modified copy in
`flash_attn/cute/modified_utils/block_scaled_layout_test.py` is byte-for-byte
identical to the stock helper. So the layout builder itself is not the
culprit.

### Hypotheses to chase next

- **TMA partition / box shape for MXFP8**: For MXFP8 `d=128`, the SFQ row
  has 4 SF bytes (vs 8 for NVFP4). If the TMA `cta_v_map` / multicast is
  sized for the NVFP4 row count, TMA will load the wrong byte positions
  into SMEM.
- **Effective swizzle differs**: `BlockScaledBasicChunk(32)` has the same
  atom shape as `(16)`, but the `tile_to_shape` over a tile with 4 instead
  of 8 SF bytes per row could produce a swizzle mismatch.
- **`mma_inst_tile_k=4` via helper default happens to match MXFP8 d=128**:
  flush this through one more time. `self.mma_inst_tile_k = 4` (from
  `128/(256/8) = 4`) matches the helper default, so the shape it produces
  may not matter. But `sfq_tmem_cols = (128 // 32) * 4 = 16` in Int32
  cols — need to verify this is correct for sf_vec_size=32.
- **S2T copy atom width**: the S2T copy that moves SFQ smem → SFQ tmem is
  sized for `sf_vec_size=16` (128 bits = 16 bytes = 8 SFs). With
  `sf_vec_size=32`, 16 bytes per row covers fewer M-rows; if the atom is
  unchanged, S2T could be writing wrong positions.

### State left in the tree

- `gemm_ptx_partial_fp4` non-TS path now passes per-K SF addresses as
  separate `r` inputs. NVFP4 still correct. MXFP8 still broken — but for a
  reason upstream of the helper.
- `gemm_blockscaled_generic` helper present; gated by
  `FA4_DEBUG_FORCE_GENERIC_MXFP8_QK=1`.
- `compile_key` includes `_key_qk_ab_dtype` and `_key_sf_dtype`.
- `sf_layout_kwargs` threads `mma_tile_inst_k` for both sf_vec sizes.

### Status: MXFP8 QK numerical correctness is **NOT** fixed.

Verification matrix update:

| mode | numerical |
|---|---|
| MXFP8 QK + BF16 PV | **NaN / 1e37 garbage** (both helper paths) |
| MXFP8 QK + FP8 PV | not tested (upstream QK broken) |

## 2026-04-15 stride audit (task #23)

Audited every `.width // 8`, `* width`, `<< 4`, hardcoded `* 16` /
`+ 512` style constant in the FA4 forward path. Three concrete bugs found
and fixed:

1. **`element_size=self.k_dtype.width // 8`** (`flash_fwd_sm100_fp4.py:929`)
   For FP4 (width=4), integer division rounds to 0; the LPT scheduler then
   computes `size_one_head=0` and divides by zero at MLIR trace time. Fixed
   by `max(width // 8, 1)` (re-applied from the `fp4` branch — the fix had
   been committed there but never landed on `mixed_precision`).

2. **`sf_dtype_per_u32` undefined** at `flash_fwd_sm100_fp4.py:1479`.
   This name was referenced inside the `quant_pv` block but never defined.
   The path didn't fail because `quant_pv` evaluates `--quant_v` (NVFP4 V
   only) and not the MXFP8-V path, but the symbol was a `NameError` ticking
   bomb. Defined locally as `sf_dtype_per_u32 = 32 // self.sf_dtype.width`
   (= 4 for both E4M3 and E8M0).

3. **SFP R2S hardcoded for `sf_vec_size=16`** at `flash_fwd_sm100_fp4.py:2932`
   The thread layout `(4, 2) stride=(1, 512)` and divisor `make_layout(4)`
   assume 8 SF bytes per row — only correct for NVFP4 (`128/16=8`). For MXFP8
   (`128/32=4`) we would over-write 4 extra bytes into the next atom or
   beyond the buffer. Generalized to derive `k_groups_per_row` from
   `mma_tiler_pv[2] // sf_vec_size`, and split into `(k_inner=min(k_groups,4),
   k_outer)`. NVFP4 path numerics unchanged (verified `--quant_v` max_diff
   identical to before).

The audit did NOT find a smoking-gun bug for the MXFP8 QK NaN/inf issue —
the relevant constants on the QK side are properly parameterized by
`sf_vec_size` already. The bug must be in either:
- the SFQ/SFK SMEM layout itself (`make_smem_layout_sfa/b` in
  `modified_utils/block_scaled_layout_test.py` — byte-for-byte identical
  to stock CUTLASS `blockscaled_utils.make_smem_layout_sfa`), or
- the SFQ/SFK TMA partition / cta_v_map (different SF-bytes-per-row count
  for MXFP8 may need a different TMA box shape), or
- the SFQ/SFK S2T copy atom which moves SMEM → TMEM.

## 2026-04-15 PTX-helper / idesc agreement audit (task #22)

For each path, compared the inline PTX `kind::*` qualifier against the
fields `mma_op_to_idesc` packs into the descriptor.

### MXFP8 QK (block-scaled, scale_vec::1X)
- PTX: `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`
- idesc captured at runtime: `0x08a00000`
  - bits 7-9 (a_format) = 0 = E4M3 ✓ matches `a_dtype=Float8E4M3FN`
  - bits 10-12 (b_format) = 0 = E4M3 ✓ matches `b_dtype=Float8E4M3FN`
  - bit 23 (scale_format) = 1 = UE8M0 ✓ matches `sf_dtype=Float8E8M0FNU`
  - bits 17-22 (n_dim) = 16 → N=128 ✓
  - bits 24-28 (m_dim) = 8 → M=128 ✓
  - bit 31 (k_size) = 0 → K32 dense ✓ matches scale_vec::1X K-per-inst
  - bits 29-30 (a_sf_id) and bits 4-5 (b_sf_id) = 0; documented as
    "set at runtime" — extracted from `tmem_scale_a/b` upper bits in the
    inline asm. **Agreement: confirmed.**

### NVFP4 QK (block-scaled, scale_vec::4X)
- PTX: `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X`
- idesc captured at runtime: `0x08201680`
  - bits 7-9 (a_format) = 5 = E2M1 ✓ matches `a_dtype=Float4E2M1FN`
  - bits 10-12 (b_format) = 1 = E5M2 ✗ — mismatch with `b_dtype=Float4E2M1FN`
    (should also be 5). Doesn't break NVFP4 because `kind::mxf4nvf4`
    implicitly fixes both operands to E2M1, so b_format is unused. Filed
    as cosmetic bug; should fix `make_instr_desc_block_scaled` to set
    b_format from `b_dtype` consistently.
  - bit 23 (scale_format) = 0 = UE4M3 ✓ matches `sf_dtype=Float8E4M3FN`
  - bits 17-22 (n_dim) = 16 → N=128 ✓
  - bits 24-28 (m_dim) = 8 → M=128 ✓
  - **Agreement: works in practice (kind takes precedence) but b_format
    encoding is sloppy.**

### Pure FP8 PV (no scale)
- PTX: `tcgen05.mma.cta_group::1.kind::f8f6f4`
- idesc: built via `make_instr_desc` (non-block-scaled path).
  - a_format / b_format / acc set from dtypes. No scale_format bit.
  - `--pv_mode fp8` runs end-to-end with max_diff `≤ 0.51` and no NaN
    across all benched shapes. **Agreement: confirmed by E2E correctness.**

### BF16 paths (fallback)
- Untouched by recent changes; kind = `tcgen05.mma.cta_group::1.kind::f16`.
  Matches `make_instr_desc(BFloat16, BFloat16, ...)` encoding. Confirmed
  by ongoing reference-attention parity in the FA4 baseline.

### Conclusion

PTX kind ↔ idesc agreement is **OK for all currently shipping paths**.
The NVFP4 b_format=1 quirk is harmless because `kind::mxf4nvf4` constrains
both operands to E2M1 by spec.

## 2026-04-15 FP8-PV active-cycle regression on long d=128 (task #20)

### Confirmed regression on current commit

Re-ran `bench_fp4 --pv_mode {bf16, fp8}` (commit `4631b50e`). Long d=128
shapes still favor BF16 PV; d=64 still favors FP8 PV:

| shape | BF16 PV | FP8 PV | Δ |
|---|---|---|---|
| `(1, 32768, 12, 128)` | 3.85 ms | 4.08 ms | **FP8 +6%** |
| `(1, 32768, 24, 128)` | 7.55 ms | 7.76 ms | **FP8 +3%** |
| `(1, 32768, 24, 64)`  | 7.17 ms | 6.94 ms | FP8 −3% |

### Structural diagnosis (without nsight on this build)

Looking at the `softmax_step` body (`flash_fwd_sm100_fp4.py:2940-2946`), the
pure-FP8 path runs **two** elementwise passes over `tSrS_t2r`:

1. `softmax.apply_exp2_convert(tSrS_t2r, ...)` — exp2(s - row_max) in
   place; result stays FP32.
2. `self._pack_fp8(tSrS_t2r, tSrP_r2t)` — FP32 → FP8 via 32 calls of
   `packed_float_to_ue4m3` (each = 2 `cvt.rn.satfinite.e4m3x2.f32`
   instructions), per softmax thread.

The BF16 path (`flash_fwd_sm100_fp4.py:2948-2954`) **fuses**: a single
`apply_exp2_convert(tSrS_t2r, tSrP_r2t, converted_scale=1.0, ...)`
writes BF16 directly into `tSrP_r2t`, no second pass.

With sf_size=16 and head_dim=128, the second-pass overhead is 32×2 = 64
extra `cvt` instructions per softmax thread per softmax_step iteration. On
short d (e.g. 64), the PV memory savings dominate; on long d=128, the pack
cost shows up as `mio_throttle 0.33→0.88` and `wait 2.22→2.74`
(unchanged from the previous nsight numbers in the table above — same
structural cause).

The bank-conflict numbers (`shared_op_st 1M→35K` for FP8) confirm that
the pure-FP8 staging itself isn't the bottleneck — it's the **issue-side
register / convert work** between exp2 and PV start.

### Proposed fix (not implemented this session)

Add a fused helper `apply_exp2_pack_fp8(tSrS_t2r, tSrP_r2t)` that walks
both tensors in lockstep:

```text
for k in 0..N_per_thread step 4:
    f0..f3 = exp2(tSrS_t2r[k:k+4] - row_max)  # already fused
    tSrP_r2t[k:k+4] = packed_float_to_ue4m3(f0, f1, f2, f3)
```

This collapses two passes into one, removes a register-resident FP32
intermediate, and matches the BF16 path's single-pass structure.

`SoftmaxSm100.apply_exp2_convert` already has a flag for in-place
conversion to dst dtype — extending it to take an explicit packer
(`fn(f0, f1, f2, f3) -> Int32`) is the smallest change. Tagging this for
follow-up; not landing this session because tuning the exp2/pack
interleaving for long d=128 needs nsight metrics that are unavailable on
this build host (CUPTI requires CUDA 13+ driver).

### Status: diagnosed, fix designed but not implemented.

## 2026-04-15 MXFP8 QK FIXED 🎯

### Root cause

`flash_fwd_sm100_fp4.py:1431` was dispatching SFQ's base TMEM offset
based on `sf_vec_size`:

```python
# BEFORE (broken for MXFP8):
sfq_base_offsets = self.tmem_o_offset if self.sf_vec_size == 32 else self.tmem_s_offset
sfq_stage_order  = (tuple(range(q_stage)) if sf_vec_size == 32
                    else tuple(q_stage - 1 - stage for stage in range(q_stage)))
```

MXFP8's branch put SFQ at `tmem_o_offset[stage]` (cols 256-511), which is
exactly where the **PV MMA writes the O accumulator**. The kernel
pipelines QK[k+1] against PV[k], so once PV[0] fires it corrupts the
SFQ[1] bytes sitting in cols 384+. QK[1] then reads garbage SF values,
which as UE8M0 exponents (0x00–0xFF = 2^−127 … 2^128) overflow to 1e37.

The branch was introduced "to avoid aliasing the S accumulator". That
reasoning was wrong — `make_tmem_layout_sfa` already encodes SFA in a
physically non-overlapping layout within the S range (via the sf_id bits
at address bits 29:30). NVFP4 uses `tmem_s_offset` and works fine.

### Fix

```python
# AFTER:
sfq_base_offsets = self.tmem_s_offset
sfq_stage_order  = tuple(q_stage - 1 - stage for stage in range(q_stage))
```

Both modes now share the NVFP4-proven layout.

### Results — `bench_fp4 --qk_mode mxfp8` on commit at HEAD

| shape | pre-fix max_diff | post-fix max_diff | TFLOPS | speedup vs BF16 ref |
|---|---|---|---|---|
| `(1, 256, 16, 128)`  | 6.9e37 | 0.043 | 30.4 | 0.87× (tiny shape, overhead-bound) |
| `(1, 1024, 16, 128)` | 2.9e37 | 0.025 | 167  | 0.50× (noisy, seqlen ~1K too small) |
| `(4, 4096, 16, 128)` | 9.9e36 | 0.016 | 1583 | 1.21× |
| `(4, 4096, 32, 128)` | NaN    | 0.020 | 1607 | 1.13× |
| `(1, 4096, 12, 128)` | NaN    | 0.012 | 948  | 1.07× |
| `(1, 32768, 12, 128)`| NaN    | 0.006 | 1638 | 1.19× |
| `(1, 4096, 24, 128)` | NaN    | 0.012 | 1307 | 1.08× |
| `(1, 32768, 24, 128)`| NaN    | 0.004 | 1651 | 1.22× |

### Verification

- NVFP4 max_diff unchanged (0.02–0.27, identical pre/post).
- `--quant_v` (FP4 PV) max_diff unchanged (0.08–1.18, identical pre/post).
- All three modes (NVFP4, MXFP8, FP4 V) produce valid output with no NaN.

Status: **#24 FIXED**. Unblocks #21 (dedicated FP8 helper perf A/B).

## 2026-04-15 MXFP8 inline-PTX vs generic cute.gemm A/B (task #21)

Now that MXFP8 numerics are fixed (#24), benched both helper paths:

| shape | inline-PTX | generic cute.gemm | winner |
|---|---|---|---|
| `(4, 4096, 16, 128)`   | 1583 | 1588 | generic +0.3% |
| `(4, 4096, 32, 128)`   | 1607 | 1611 | generic +0.3% |
| `(1, 4096, 12, 128)`   | 948  | 955  | generic +0.7% |
| `(1, 32768, 12, 128)`  | 1638 | 1650 | generic +0.7% |
| `(1, 4096, 24, 128)`   | 1307 | 1321 | generic +1.1% |
| `(1, 32768, 24, 128)`  | 1651 | 1726 | generic +4.5% |

Generic wins on every MXFP8 shape. Root cause: the inline-PTX helper's
per-K SF address plumbing (passing `tScaleA[..., k].iterator.toint()`
as separate `r` operands, and recomputing idesc with sf_id extraction
per K-iter) adds register pressure that the cleaner cute.gemm path
avoids. NVFP4 still wins on inline-PTX because scale_vec::4X packs 4
SFs per operand so the per-K advance is trivial.

### Default change

Routed MXFP8 (Float8E4M3FN / Float8E5M2 ab dtype) through
`sm100_utils.gemm_blockscaled_generic` by default; NVFP4 stays on
`gemm_ptx_partial_fp4`. Env `FA4_DEBUG_FORCE_GENERIC_MXFP8_QK=1` still
forces generic on NVFP4 too for A/B tests.

Status: **#21 resolved**. Dedicated helper offers no perf win on MXFP8;
keep generic.

## 2026-04-16 MXFP8+FP8 vs NVFP4 QK-only perf gap (task #25)

### Target
MXFP8 QK + FP8 PV should beat NVFP4 QK + BF16 PV baseline.

### 4-way comparison (bench_fp4 triton.do_bench rep=25)

| shape | NVFP4+BF16 | NVFP4+FP8 | MXFP8+BF16 | MXFP8+FP8 | MXFP8+FP8 Δ |
|---|---|---|---|---|---|
| (1,32768,12,128) | 1713 | 1620 | 1650 | 1580 | **-7.8%** |
| (1,32768,24,128) | 1783 | 1701 | 1670 | 1663 | **-6.7%** |
| (4,4096,16,128)  | 1646 | 1603 | 1588 | 1560 | **-5.2%** |
| (4,4096,32,128)  | 1683 | 1632 | 1607 | 1592 | **-5.4%** |

### Attribution

Decomposed for `(1, 32768, 24, 128)`:
- `NVFP4+BF16 → MXFP8+BF16` = **-113 TF (-6.3%)** ← QK path cost
- `MXFP8+BF16 → MXFP8+FP8` = **-7 TF (-0.4%)** ← PV path cost

For this shape, **90%+ of the gap is MXFP8 QK itself**, not FP8 PV.
For `(1, 32768, 12, 128)` the split is roughly 50/50.

### Root cause

MXFP8 uses `kind::mxf8f6f4.block_scale.scale_vec::1X` which has
**K=32 elements per MMA instruction**. NVFP4 uses
`kind::mxf4nvf4.block_scale.scale_vec::4X` with **K=64 per instruction**
(4 packed SFs per operand). For d=128:
- NVFP4: 2 MMA insts per (M=128, N=128, K=128) tile
- MXFP8: 4 MMA insts per tile

The extra MMA instructions have fixed scheduling / SF-fetch overhead
per inst. This shows as `smsp__cycles_active` going up even though
DRAM/L2 traffic goes down. No kv_stage tuning or softmax fusion closes
the gap because MMA throughput is the bottleneck.

### Attempts

1. **Fused `_apply_exp2_pack_fp8`** (commit pending): interleaves exp2
   with cvt.rn.satfinite.e4m3x2.f32 in the same 4-FP32 chunk, removing
   the explicit 2-pass. Benched at parity — the DSL IR optimizer already
   fuses at register-allocation level. Kept behind
   `FA4_FP8_PV_USE_FUSED_PACK=1` for future shapes where the 2-pass
   cost might matter.
2. **KV stage cap sweep** (`FA4_FP8_PV_KV_STAGE_CAP` in {0, 4, 6, 8}):
   no measurable difference; this path isn't kv_stage-bound on B200.

### Structural verdict

**MXFP8 QK + FP8 PV cannot beat NVFP4 QK + BF16 PV on this MMA shape
without one of:**
- Dropping block-scaling on QK (use `kind::f8f6f4` pure FP8, losing
  per-group SF precision), or
- A `scale_vec::2X` MMA kind that doubles K-per-inst for FP8 (would
  need PTX support and redesigned SF layout — currently `scale_vec::2X`
  is the sf_vec_size=16 MXFP8 variant, same 32-K-per-inst), or
- Layer-level optimization where MXFP8 is traded for better quant
  precision on downstream layers, not this kernel's TFLOPS.

### Current numerics (unchanged by this task)

- NVFP4 QK: max_diff 0.02–0.27
- MXFP8 QK + BF16 PV: max_diff 0.004–0.043
- MXFP8 QK + FP8 PV: max_diff 0.05–0.51
- FP4 PV (`--quant_v`): max_diff 0.08–1.18

## 2026-04-16 Full comparison: FP8 QK + BF16/FP8 PV apples-to-apples

Shape: `(1, 32768, 24, 128)`, non-causal, triton.do_bench rep=25 warmup=10.

| kernel | QK dtype | PV dtype | Duration | TFLOPs | IPC | SM Busy | cyc/inst | F2FP count / stalls |
|---|---|---|---|---|---|---|---|---|
| pr2109 | BF16 | BF16 | 15.67 ms | 1193 | 1.50 | 65.16% | 9.97 | 256 / 1564 |
| pr2109 | FP8  | FP8  | 10.85 ms | 1978 | 2.18 | 74.73% | 6.89 | 256 / 5851 |
| pr2109 | FP8  | BF16 |  7.77 ms | 1697 | 2.20 | 64.45% | 6.83 | 256 / pending |
| ours   | BF16 | BF16 |  9.26 ms | 1425 | —    | —      | —    | — |
| ours   | **FP8**  | **FP8**  | **11.99 ms** | **1794** | **1.69** | 82.31% | 8.89 | 256 / 6633 |
| ours   | FP8  | BF16 | pending  | —    | —    | —      | —    | — (mixed dtype requires kernel surgery) |
| ours   | NVFP4 | BF16 | 11.91 ms | 1767 | 1.57 | 82.97% | 9.53 | 256 / 3425 |
| ours   | NVFP4 | FP8  | 12.62 ms | 1705 | 1.46 | 78.14% | 10.26 | 256 / **14129** |
| ours   | MXFP8 | FP8  | 12.89 ms | 1667 | 1.44 | 76.57% | 10.41 | 256 / **14632** |

### Key findings

- **Pure FP8 QK + FP8 PV beats NVFP4+BF16** in our kernel (1794 vs 1767 TF). Block-scale
  wasn't essential; it was actually hurting the FP8 PV path.
- **F2FP stall attribution is clear**: pure FP8 paths show ~6K stalls; block-scaled FP8
  paths show ~14K stalls. The extra ~8K come from the SF plumbing, not from FP8 pack
  itself.
- Our FP8 kernel IPC jumps 1.44 → 1.69 when we drop block-scale.
- pr2109 still wins (10.85 vs 11.99 ms, +10%) on pure-FP8 apples-to-apples — their
  kernel has better FP8 scheduling (IPC 2.18 vs ours 1.69). Fixable kernel-level gap.
- **pr2109 FP8 QK + BF16 PV** (new row): 7.77 ms / 1697 TF at (1,32768,24,128).
  Sits between pr2109 FP8/FP8 (10.85 ms) and pr2109 BF16/BF16 (15.67 ms), showing
  the FP8 speedup in pr2109 is NOT exclusive to FP8 V — FP8 QK alone captures a
  large fraction of the improvement. IPC 2.20 matches their pure-FP8 path, so
  their QK MMA throughput isn't materially hurt by switching V back to BF16.
  Implementation: `hao-ai-lab/flash-attention-fp4` branch `pr2109-mixed-dtype`
  commit 65cbcc8a — key fixes: SharedStorage sizes sK for V's byte footprint,
  recast_ptr sets dtype=v_dtype, P operand dtype follows v_dtype, tmem store
  Repetition keyed on v_dtype.width.

### Ours: how to enable pure FP8 QK

`FA4_ALLOW_PURE_FP8_QK=1` — interface.py lets FP8 Q/K/V through to
`FlashAttentionForwardSm100` (the non-blockscaled path) when set. The kernel already
supports kind::f8f6f4 via `_mma_inst_kind`; only the assertion in interface.py
gated it off.
