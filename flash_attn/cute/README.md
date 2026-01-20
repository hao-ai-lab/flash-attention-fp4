# Flash Attention CUTE

Flash Attention CuTe-DSL implementation.

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

**Note:** The `__init__.py` also includes code to move the editable finder to the front of `sys.meta_path` as a backup, but running the fix script is recommended for a permanent solution.

### Benchmarking FP4 attn

```bash
cd examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute
CUTE_DSL_ENABLE_TVM_FFI=1 python benchmarks/bench_fp4.py
```