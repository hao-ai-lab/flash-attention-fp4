# Illegal Memory Access Bug: `gemm_ptx_partial_fp4` TS Path

## Summary

Changing from `nvidia-cutlass-dsl==4.3.5` to `4.4.0` caused `cudaErrorIllegalAddress` /
`cudaErrorIllegalInstruction` in the FP4 PV GEMM's TMEM-source (TS) inline PTX assembly path.
Both DSL versions use the same system ptxas (CUDA 12.9 / driver 570.158) and emit PTX 8.8,
but generate different PTX code (different IR lowering / instruction ordering), causing ptxas
to make different register allocation decisions.

## Root Cause

An **uninitialized PTX register** (`smem_desc_b_lo`) in the post-mbar loop of
`gemm_ptx_partial_fp4` (and `gemm_ptx_partial`), exposed by **different ptxas register
allocation** due to different PTX input from each DSL version.

### The bug in code (`blackwell_helpers.py`)

The TS path inline asm has this structure:

```
Initial MMA:  k=0, uses smem_desc_b = {smem_desc_b_lo_start, smem_desc_b_hi}
Pre-mbar loop:  k=1 to (k_tiles // 4 * 3 - 1), writes smem_desc_b_lo each iteration
mbar_wait
Post-mbar loop: k=(k_tiles // 4 * 3) to (k_tiles - 1), reads smem_desc_b_lo
```

The post-mbar loop used **incremental offsets**:
```ptx
add.u32 smem_desc_b_lo, smem_desc_b_lo, offset_b_diff[k-1];  // BUG: reads smem_desc_b_lo
```

When `k_tiles=2` (FP4 MMA), `k_tiles // 4 * 3 = 0`, so the pre-mbar loop is **empty**.
`smem_desc_b_lo` is never written before the post-mbar loop reads it → **undefined behavior**.

### Why it worked in DSL 4.3.5

ptxas **pre-computed** the descriptor offset before the initial MMA:

```sass
// 4.3.5 SASS:
/*2360*/  ULOP3.LUT UR6, UR8, 0x10000, URZ, 0xfc   // UR6 = smem_desc_start_b_lo (for k=0)
/*23c0*/  UIADD3 UR8, UPT, UPT, UR6, 0x2, URZ      // UR8 = UR6 + 0x2 (pre-computed k=1 desc)
/*23e0*/  UTCOMMA.4X ..., gdesc[UR6], ... UP0        // Initial MMA k=0
/*23f0*/  SYNCS.TRYWAIT ...                           // mbar_wait
          ...
/*2430*/  UMOV UR22, UR8                              // UR22 = pre-computed UR8
/*24e0*/  UTCOMMA.4X ..., gdesc[UR22], ... UPT        // Post-mbar MMA k=1, correct descriptor
```

ptxas mapped `smem_desc_b_lo` to UR8 which was **pre-computed as `UR6 + 0x2`** before the
mbar_wait. The register was effectively initialized by ptxas's scheduling.

### Why it broke in DSL 4.4.0

Different PTX input caused ptxas to make different register allocation decisions.
`smem_desc_b_lo` was never initialized in the PTX (the pre-mbar loop was empty), so ptxas
was free to assign any physical register to it. It chose UR4, which it also used for an
unrelated local memory load (`R2UR UR4, R4`). By the time the UIADD reads UR4 as
`smem_desc_b_lo`, it contains an unrelated value from local memory — not `smem_desc_b_lo_start`:

```sass
// 4.4.0 SASS:
/*2920*/  UTCOMMA.4X tmem[UR6], gdesc[UR10], ... UP1  // Initial MMA k=0, B desc in gdesc[UR10]
/*2930*/  SYNCS.TRYWAIT ...                             // mbar_wait
          ...
/*2af0*/  R2UR UR4, R4                                 // UR4 = unrelated local memory value
/*2b20*/  UIADD3 UR16, UPT, UPT, UR4, 0x2, URZ        // UR16 = UR4 + 0x2 (garbage + 0x2)
/*2b50*/  UTCOMMA.4X ..., gdesc[UR16], ... UPT          // Post-mbar MMA: corrupt descriptor → CRASH
```

UR4 never held `smem_desc_b_lo_start` — it was uninitialized from the start. ptxas simply
reused the same physical register for both the uninitialized `smem_desc_b_lo` and the
unrelated `R2UR` (Register-to-Uniform-Register, copies a per-thread register to a warp-uniform
register). The UIADD then computed `garbage + 0x2` instead of `smem_desc_b_lo_start + 0x2`,
producing a corrupt SMEM descriptor that caused an illegal memory access.

### Additional bug: k=0 duplication

When `k_tiles=2`, `k_tiles // 4 * 3 = 0`, so the post-mbar loop range is `range(0, 2)`.
Since k=0 was already handled by the initial MMA, k=0 executes **twice**, doubling its
contribution to the accumulator. This produces wrong results even after fixing the
uninitialized register.

The correct split for FP4 (4-bit elements) is `k_tiles // 2` (not `k_tiles // 4 * 3`),
defined in `flash_fwd_sm100_fp4.py` as `self.mbar_p_split`.

## Compile-time values

```
smem_desc_base_b_lo = 0x10000
smem_desc_b_hi      = 0x80004020
offset_b            = [0x0, 0x2]       (absolute offsets per k-tile)
offset_b_diff       = [0x2]            (incremental delta)
smem_desc_start_b_lo = 0x10000 | (smem_ptr >> 4)   (runtime)
```

## Fix

Two changes in `blackwell_helpers.py`:

1. **Use absolute offsets** instead of incremental in the post-mbar loop:
   ```python
   # OLD (buggy):
   f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
   # NEW (fixed):
   f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
   ```

2. **Pass `pre_mbar_tiles`** parameter using `self.mbar_p_split(k_tiles)` to avoid k=0
   duplication. The post-mbar loop starts at `max(1, pre_mbar_tiles)`:
   ```python
   # For FP4 (v_dtype.width < 8): mbar_p_split = k // 2  → k_tiles=2 → split=1
   # For F16 (v_dtype.width >= 8): mbar_p_split = k // 4 * 3 → k_tiles=8 → split=6
   ```

## Files

- `flash_attn/cute/blackwell_helpers.py`: `gemm_ptx_partial` (~line 363), `gemm_ptx_partial_fp4` (~line 573)
- `flash_attn/cute/flash_fwd_sm100_fp4.py`: `mbar_p_split` (line 1279), `gemm_Pi` creation (line 2321)
- Cubins: `/tmp/fp4_kernel_dsl435_quantv.cubin` (4.3.5), `/tmp/fp4_kernel_dsl440.cubin` (4.4.0)
- PTX: `/tmp/fp4_kernel_dsl435.ptx` (4.3.5), `/tmp/fp4_kernel_dsl440.ptx` (4.4.0)
