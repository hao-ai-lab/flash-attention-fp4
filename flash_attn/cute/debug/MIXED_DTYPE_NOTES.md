# pr2109 Mixed FP8 QK + BF16 PV — Blocker Notes

## Goal
Test FP8 QK (kind::f8f6f4) + BF16 PV (kind::f16) to isolate whether pr2109's
FP8 speedup at d=128 long seqlen comes from the FP8 QK path alone or also
requires FP8 V. This is an A/B against the all-FP8 baseline.

## Patches applied on this branch

1. `interface.py`: relaxed `q.dtype == k.dtype == v.dtype` assertion to
   `q.dtype == k.dtype` only.
2. `flash_fwd_sm100.py`:
   - Removed `q_dtype != v_dtype` TypeError.
   - Changed `tP_layout` to use `v_dtype` (was `q_dtype`) so the P operand
     format matches what the PV MMA's A-side expects (=v_dtype).

## Blocker

The SMEM tensor `sV` aliases `sK`'s storage via `cute.recast_ptr`:

```python
# flash_fwd_sm100.py ~line 1004
sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner),
                      sV_layout.outer)
```

This works when sizeof(K_dtype) == sizeof(V_dtype). For mixed FP8 K (1 byte)
+ BF16 V (2 bytes), the recast produces non-canonical UMMA_MN strides and
`make_smem_desc_base(sV, Major.MN)` raises:

```
ValueError: Not a canonical UMMA_MN Layout: Expected stride failure.
```

## Required change for native support

1. Allocate a **separate `sV` slot** in `SharedStorage` with
   `cute.struct.MemRange[v_dtype, cute.cosize(sV_layout)]` (at the cost
   of ~128 KB additional SMEM per stage for d=128 BF16 V, may reduce
   `kv_stage`).
2. Change `sV` construction to `storage.sV.get_tensor(...)` instead of
   aliasing `sK`.
3. Verify no downstream code assumes sK/sV share a base pointer (e.g.
   producer TMA partition or consumer pipeline barriers).

Estimate: 30-60 LOC change touching SharedStorage, kv_stage sizing, and
TMA dest tensor partitioning. Not landed in this branch.

## Related findings

See `flash_attn/cute/debug/mxfp8_qk_fp8_pv.md` on the companion
`mixed_precision` branch of `hao-ai-lab/flash-attention-fp4` for the
full NCU + SASS comparison (same (1, 32768, 24, 128) shape):

| kernel | QK | PV | TF | IPC | F2FP stalls |
|---|---|---|---|---|---|
| pr2109 | BF16 | BF16 | 1193 | 1.50 | 1564 |
| pr2109 | FP8  | FP8  | 1978 | 2.18 | 5851 |
| pr2109 | FP8  | BF16 | pending (this branch) | | |
| ours   | FP8  | FP8  | 1794 | 1.69 | 6633 |
| ours   | NVFP4| BF16 | 1767 | 1.57 | 3425 |
| ours   | MXFP8| FP8  | 1667 | 1.44 | 14632 |
