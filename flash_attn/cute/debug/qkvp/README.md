# QKVP correctness diagnostics

Tests collected during the 2026-04-08 investigation into the FP4 QKVP path
giving cos~0.5 vs BF16 reference.

**Quick state of the world** (commit 4afaace4):
- `bench_fp4 --quant_v` works (max_diff~1.14, mean correct, constant SF=1.0).
- The FP4 V `make_ptr` path used by FastVideo's `_nvfp4_quantize_v_for_fa4`
  is numerically broken because the permute trick produces a buffer where
  adjacent S values are in adjacent BYTES (not nibbles), violating FP4
  K-major MMA semantics.

## Tests

| File | What it does | Result |
|---|---|---|
| `test_real_mV.py` | Probes `tile_to_shape(BlockScaledBasicChunk(16).layout, mV_shape, (2,1,3,4))` with `mV_shape=(d,s,h,b)=(128,8192,16,1)`. | shape `(((32,4),1),((16,4),128),(1,16),(1,1))`, cosize 1MB. atom_M tiles d, atom_K tiles s. |
| `test_all_const.py` | Q=K=V=constant via FastVideo make_ptr path. | V=1.0 → out=1.5; V=2.0 → out=3.0; V=0.5 → out=0.42 with positional variance. WRONG. |
| `test_v_bench_path.py` | Q=K=V=constant via bench `cute_tensor_like` cute Tensor path. | V=1.0 → out=1.0 exactly. CORRECT. |
| `test_v_v2.py` | Random Q/K + V via FastVideo path. | cos~0.52, mean_diff~0.02 — wrong. |
| `test_kernel_correct_sf.py` | Bench V FP4 buffer + SF in kernel's tile_to_shape byte layout, write nvfp4 SF bytes directly. | Output uniform 0.5156 for V=1.0 (positional variance gone, but value half of expected ~1.03). The bench int8 V buffer is read as PACKED FP4 (2 per byte) by the kernel, so writing 1 FP4 per byte makes alternating positions read as 0. |

## Next steps

1. Determine empirically whether bench's V int8 buffer is read as packed (2
   FP4/byte) or unpacked (1 FP4/byte) by the cute kernel. Write distinct values
   in lower vs upper nibble of byte 0 and read back via the kernel.
2. If packed: pack our nvfp4 FP4 data 2 per byte at the right byte addresses
   in the bench-style int8 buffer.
3. Build SF tensor at the kernel's tile_to_shape byte addresses (not bench's
   `create_scale_factor_tensor` layout, which assumes M=seqlen, K=headdim — the
   opposite of what `tile_to_shape` produces for V).
