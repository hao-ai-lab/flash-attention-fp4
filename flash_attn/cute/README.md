# FP4 Flash Attention 4 (FA4) on Blackwell

CuTe DSL implementation of FP4 block-scaled flash attention for NVIDIA Blackwell GPUs (sm100a/sm103a). Quantizes Q and K to FP4 E2M1 with per-block E4M3 scale factors, while V remains in BF16.

## Results

FP4 FA4 vs BF16 FA4 kernel speedup (CUPTI `bench_gpu_time`, vacant B200 GPU):

| Config | FP4 (ms) | FP4 TFLOPS | BF16 (ms) | BF16 TFLOPS | Speedup |
|--------|----------|------------|-----------|-------------|---------|
| b=1 s=256 h=16 d=128 | 0.014 | 37 | 0.015 | 35 | 1.07x |
| b=1 s=1024 h=16 d=128 | 0.024 | 365 | 0.026 | 336 | 1.09x |
| b=4 s=4096 h=16 d=128 | 0.336 | 1637 | 0.390 | 1409 | 1.16x |
| b=1 s=4096 h=12 d=128 | 0.104 | 987 | 0.118 | 871 | 1.13x |
| **b=1 s=32768 h=12 d=128** | **3.881** | **1700** | **4.834** | **1365** | **1.25x** |
| b=1 s=4096 h=24 d=128 | 0.152 | 1360 | 0.173 | 1194 | 1.14x |
| b=1 s=32768 h=24 d=128 | 7.578 | 1741 | 10.102 | 1306 | 1.33x |
| b=1 s=32768 h=24 d=64 | 7.186 | 918 | 7.276 | 907 | 1.01x |

Per-call precision: cosine similarity = 0.99, SNR = 7.25 (FP4 QK quantization vs BF16 reference).

## Installation

### Editable Install

```bash
pip install -e .
```

### Fixing Editable Install Import Issues

If you have a non-editable `flash-attn` package installed, Python may import from the installed package instead of your local editable installation.

**Solution:** Run the fix script once after installation:

```bash
python fix_editable_import.py
```

**Verify it's working:**

```bash
python -c "import flash_attn.cute.interface; print(flash_attn.cute.interface.__file__)"
```

## Benchmarking

```bash
cd examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute
CUTE_DSL_ENABLE_TVM_FFI=1 python benchmarks/bench_fp4.py          # QK quantized
CUTE_DSL_ENABLE_TVM_FFI=1 python benchmarks/bench_fp4.py --quant_v # QKV quantized
CUTE_DSL_ENABLE_TVM_FFI=1 python benchmarks/bench_fp4.py --debug   # correctness test
```

## Integration with FastVideo

See [debug/fastvideo_integrate.md](debug/fastvideo_integrate.md) for integration with [FastVideo](https://github.com/hao-ai-lab/FastVideo) video diffusion framework, including `nvfp4_quantize` scale factor layout conversion and end-to-end video generation results.

## Pipeline Graph
![pipeline graph](figures/pipeline.png)
