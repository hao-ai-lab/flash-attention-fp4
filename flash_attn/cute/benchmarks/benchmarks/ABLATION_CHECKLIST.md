# Quant-P ablation (bs=32, seq=1024, --quant_v)

Run from: `flash-attention/` (repo root)
```bash
CUTE_DSL_ENABLE_TVM_FFI=1 python flash_attn/cute/benchmarks/bench_fp4.py --quant_v
```

Rerun (all confirmed; ablation 1 is reproducible).

| # | What to change (flash_fwd_sm100_fp4.py) | Speedup vs bf16 |
|---|----------------------------------------|-----------------|
| baseline | (none) | **1.20x** |
| 1 | L2963: comment out `self._quant_fp4(...)` only (one line) | **1.24x** |
| 2 | L2955: comment out `softmax.update_row_sum_sage(...)` | **1.26x** |
| 3 | L2950–2954: comment out `softmax.apply_exp2_convert(tSrPSF_f32, ...)` block | **1.24x** |
| 4 | L2914–2917: no tSrPSF; quant_pv branch → else-branch P path only | **1.27x** |
| 5 | L2932: pass `None` instead of `tSrPSF_f32` to `scale_subtract_rowmax` | **1.20x** |

All five ablations run on B200; code reverted to baseline after each.
