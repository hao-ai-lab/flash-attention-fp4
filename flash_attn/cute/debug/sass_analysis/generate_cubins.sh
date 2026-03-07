#!/bin/bash
# Generate cubins for Variant B (no multiply) and Variant C (with multiply)
# from the FP4 Flash Attention kernel.
#
# Variant B: update_row_sum_sage passes None for acc_S_row_group_max_exp (no FMUL2)
# Variant C: update_row_sum_sage passes tSrPSF_f32 (adds FMUL2 chain)
#
# The source file is modified in-place, so we use git stash to restore after each build.

set -e

REPO="/mlx_devbox/users/wenxuan.tan/playground/VibeCUDA/knowledge/repos/flash-attention-fp4"
SRC="$REPO/flash_attn/cute/flash_fwd_sm100_fp4.py"
OUTDIR="$(cd "$(dirname "$0")" && pwd)"
BENCHDIR="$REPO/flash_attn/cute/benchmarks"

echo "=== Generating Variant C cubin (with multiply — current code) ==="
cd "$OUTDIR"
CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_LINEINFO=1 CUTE_DSL_ENABLE_TVM_FFI=1 \
CUDA_VISIBLE_DEVICES=1 python "$BENCHDIR/bench_fp4.py" --quant_v 2>&1 | tail -5

# Find the cubin just created
CUBIN_C=$(ls -t "$OUTDIR"/*.cubin 2>/dev/null | head -1)
if [ -z "$CUBIN_C" ]; then
    echo "ERROR: No cubin generated for Variant C"
    exit 1
fi
mv "$CUBIN_C" "$OUTDIR/variantC_before_barriers_withmult.cubin"
echo "Variant C cubin: $OUTDIR/variantC_before_barriers_withmult.cubin"

echo ""
echo "=== Patching source for Variant B (no multiply) ==="
# Line 3190: change tSrPSF_f32 args to None
cd "$REPO"
git stash
sed -i 's/softmax.update_row_sum_sage(tSrS_t2r, tSrPSF_f32, tSrPSF_f32.layout, acc_scale, is_first)/softmax.update_row_sum_sage(tSrS_t2r, None, tSrPSF_f32.layout, acc_scale, is_first)/' "$SRC"
echo "Patched line 3190 — passing None for acc_S_row_group_max_exp"

echo ""
echo "=== Generating Variant B cubin (no multiply) ==="
cd "$OUTDIR"
CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_LINEINFO=1 CUTE_DSL_ENABLE_TVM_FFI=1 \
CUDA_VISIBLE_DEVICES=1 python "$BENCHDIR/bench_fp4.py" --quant_v 2>&1 | tail -5

CUBIN_B=$(ls -t "$OUTDIR"/*.cubin 2>/dev/null | head -1)
if [ -z "$CUBIN_B" ]; then
    echo "ERROR: No cubin generated for Variant B"
    cd "$REPO" && git checkout -- "$SRC"
    exit 1
fi
mv "$CUBIN_B" "$OUTDIR/variantB_before_barriers_nomult.cubin"
echo "Variant B cubin: $OUTDIR/variantB_before_barriers_nomult.cubin"

echo ""
echo "=== Restoring source ==="
cd "$REPO"
git checkout -- "$SRC"
echo "Source restored."

echo ""
echo "=== Done ==="
echo "Cubins in: $OUTDIR"
ls -lh "$OUTDIR"/*.cubin
