# Flash Attention CUTE

Flash Attention CuTe-DSL implementation.

## NOTE
This branch is for debugging the performance of FP4.
See [debug notes](fp4_flash_attention_optimization_notes.md) for more details.

## Installation

### Editable Install

To install this package in editable mode for development:

```bash
pip install -e .
```

### Fixing Editable Install Import Issues

If you have a non-editable `flash-attn` package installed, Python may import from the installed package instead of your local editable installation. This happens because the editable finder is checked after the regular package in Python's import system.

**Solution:** Run the fix script once after installation:

```bash
python fix_editable_import.py
```

This script patches the editable finder to take precedence over the regular package. You only need to run it **once** after `pip install -e .` - the fix is permanent until you reinstall.

**Verify it's working:**

```bash
python -c "import flash_attn.cute.interface; print(flash_attn.cute.interface.__file__)"
```
This should show your local path (e.g., `/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/interface.py`), not the installed package path.

## Benchmarking FP4 attn

```bash
cd examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute
CUTE_DSL_ENABLE_TVM_FFI=1 python benchmarks/bench_fp4.py
```

## Debugging 
This uses cuda coredump and nvdisasm to locate the error ptx segment
```
./benchmarks/analyze_coredump.sh --run benchmarks/bench_fp4.py --quant_v --output my_analysis.txt
```

## Current pipeline graph
![pipeline graph](figures/pipeline.png)

## Changes to merge from FA4 main branch

### Commits after 43375aa (Nov 19, 2025)
Some are only partially merged (like q_stage=1)
- [ ] `052015a` - add fastdivmod for oob reads in mask_mods (#2020) - Nov 21, 2025
- [ ] `d063b33` - don't pass mask_fn to softmax_step generically (#2026) - Nov 22, 2025
- [ ] `92ca9da` - [Cute,Fwd] enable mask mod without blocksparsity (#2031) - Nov 25, 2025
- [ ] `672381f` - Bump pin (#2025) - Nov 25, 2025
- [ ] `fd8d5eb` - [Cute,Fwd] Extend score_mod to variable sequence length (#2043) - Dec 15, 2025
- [ ] `bba578d` - Fix IMA in fwd on m boundary (#2091) - Dec 20, 2025
- [ ] `58fe37f` - fix shuffle sync for pack gqa epilogue (#2097) - Dec 24, 2025
- [ ] `9b6dbac` - Add pack-gqa support for blcoksparse impl w/ braodcasted H dim (#2098) - Jan 4, 2026
- [ ] `f98d345` - [Cute,Fwd] improved block sparsity (#2100) - Jan 5, 2026
- [ ] `3c8ca4e` - [Cute,Fwd,Sm100] Support `q_stage=1` for inference (#1993) - Jan 8, 2026
- [ ] `68649fb` - [Cute][Flex]Add pack-gqa divmod (#2180) - Jan 15, 2026
- [ ] `fffabc3` - [Cute,Fwd,Sm100] distributed offset calculation for paged KV (#2104) - Jan 15, 2026