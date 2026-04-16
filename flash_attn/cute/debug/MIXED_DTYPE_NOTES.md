# pr2109 Mixed FP8 QK + BF16 PV — Progress Notes

## Goal
Test FP8 QK (kind::f8f6f4) + BF16 PV (kind::f16) to isolate whether pr2109's
FP8 speedup at d=128 long seqlen comes from the FP8 QK path alone or also
requires FP8 V.

## Approach: K-aliases-V pattern (from flash_fwd_sm100_fp4.py)

Our FP4 kernel already handles mixed K/V byte sizes (FP4 K + BF16 V) without
allocating separate sV SMEM — it aliases the *smaller* tensor onto the
*larger*'s buffer with scaled stage stride. Applied the same pattern to pr2109:

```python
# SharedStorage: when k_dtype < v_dtype, sK is a placeholder and V owns
# the larger buffer. When k_dtype > v_dtype, they each get their own slot.
sK: MemRange[k_dtype, 1] if k_width < v_width else MemRange[k_dtype, cosize(sK_layout)]
sV: MemRange[v_dtype, cosize(sV_layout)] if k_dtype != v_dtype else MemRange[k_dtype, 1]

# Construction: stage stride of sK is scaled up by (v_width/k_width) so both
# point at matching bytes within the same physical stage slot.
if k_dtype == v_dtype:
    sK = storage.sK.get_tensor(sK_layout...)
    sV = cute.make_tensor(recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
elif k_dtype.width < v_dtype.width:
    sV = storage.sV.get_tensor(sV_layout...)
    stride_sK_aligned = sV_layout.outer.stride[-1] * (v_width / k_width)
    sK_outer_aligned = cute.make_layout(sK_layout.outer.shape,
                                         stride=(*sK_layout.outer.stride[:-1], stride_sK_aligned))
    sK = storage.sV.get_tensor(sK_outer_aligned, dtype=k_dtype)
else:
    sK = storage.sK.get_tensor(...); sV = storage.sV.get_tensor(...)
```

This matches `flash_fwd_sm100_fp4.py:1345-1361` exactly.

## Patches applied on this branch

1. `interface.py`
   - Relaxed `q.dtype == k.dtype == v.dtype` → `q.dtype == k.dtype` only.
   - Added `v_dtype_key` to `compile_key` to avoid cache collision.
   - `view(torch.uint8)` only applies to tensors whose actual dtype is FP8.
2. `flash_fwd_sm100.py`
   - Removed `q_dtype != v_dtype` TypeError.
   - `tP_layout`: uses `v_dtype` (was `q_dtype`).
   - `SharedStorage` + sK/sV construction: K-aliases-V pattern as above.
   - `kv_stage` sizing: `max(k_per_stage, v_per_stage)` when K-aliases-V
     (same as all-same-dtype).

## Current state

- **Compiles** with the K-aliases-V aliasing.
- **CUDA runtime error** `cudaErrorInvalidValue` at all tested shapes. The
  aliasing logic matches fp4 kernel's three-way pattern but some downstream
  bookkeeping still assumes `sV = recast(sK)` (pr2109's original code). Likely
  culprits to audit:
  - V TMA producer: builds `tVsV` from `sV`; should now be based on the new
    sV that owns the buffer.
  - V consumer partition (thr_mma_pv.partition_B(sV)): should be OK if sV
    layout is correct.
  - Producer pipeline barriers — may still assume sK and sV share base
    address for the `tma_copy_k` / `tma_copy_v` partition.
- Correctness **not verified**.

## Companion branches

- `hao-ai-lab/flash-attention-fp4` branch `pr2109-mixed-dtype`: this WIP.
- `hao-ai-lab/flash-attention-fp4` branch `mixed_precision`: our FA4 kernel
  with `FA4_ALLOW_PURE_FP8_QK=1` + full NCU table in
  `flash_attn/cute/debug/mxfp8_qk_fp8_pv.md`.

## Next steps

1. Add `print()` of `sK.iterator.toint()` and `sV.iterator.toint()` after
   construction to confirm K aliases V's base.
2. Audit TMA copy-fn setup for K: verify it uses the aligned sK_outer_aligned
   layout (not the original sK_layout).
3. Audit the pipeline barrier indexing for K vs V — they share stage slots
   but need consistent numbering.
