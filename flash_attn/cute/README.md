# FP4 Flash Attention 4 (FA4) on Blackwell

CuTe DSL implementation of FP4 block-scaled flash attention for NVIDIA Blackwell GPUs (sm100a/sm103a). Supports two modes:
- **QK quantization** (`--quant_qk`, default): Q and K quantized to NVFP4 E2M1 with per-block E4M3 scale factors, V remains BF16. **1.09–1.33x speedup** over BF16 FA4.
- **QKVP quantization** (`--quant_v`): additionally quantizes the softmax output P and V to NVFP4 with on-the-fly group-wise P quantization. Currently **slower than BF16** (0.84–0.95x) due to hardware limitations: the softmax exp (MUFU instruction) has the same throughput as on H100, but B200 MMA throughput doubles, making softmax warp the bottleneck (See [FA4 paper](https://arxiv.org/abs/2603.05451)). The added P quantization + scale factor R->SMEM->TMEM copy increases critical-path latency.
We speculate that on B300 and Rubin (w/ FP16 softmax) the QKVP quantization will be faster than BF16.

## Results — QK Quantization

FP4 FA4 vs BF16 FA4 kernel speedup (CUPTI `bench_gpu_time`, vacant B200 GPU):

| Config | FP4 (ms) | FP4 TFLOPS | BF16 (ms) | BF16 TFLOPS | Speedup |
|--------|----------|------------|-----------|-------------|---------|
| b=1 s=256 h=16 d=128 | 0.014 | 37 | 0.015 | 35 | 1.07x |
| b=1 s=1024 h=16 d=128 | 0.024 | 365 | 0.026 | 336 | 1.09x |
| b=4 s=4096 h=16 d=128 | 0.336 | 1637 | 0.390 | 1409 | 1.16x |
| b=1 s=4096 h=12 d=128 | 0.104 | 987 | 0.118 | 871 | 1.13x |
| **b=1 s=32768 h=12 d=128** ¹ | **3.881** | **1700** | **4.834** | **1365** | **1.25x** |
| b=1 s=4096 h=24 d=128 | 0.152 | 1360 | 0.173 | 1194 | 1.14x |
| b=1 s=32768 h=24 d=128 | 7.578 | **1741** | 10.102 | 1306 | **1.33x** |
| b=1 s=32768 h=24 d=64 | 7.186 | 918 | 7.276 | 907 | 1.01x |

¹ Matches [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) inference (480×832 video, 81 frames → latent seqlen 32760, nheads=12, headdim=128).

Per-call precision: cosine similarity = 0.99, SNR = 7.25 (FP4 QK quantization vs BF16 reference).

## Results — QKV Quantization (quant_v)

Additionally quantizes softmax output P and V to FP4. The PV GEMM uses block-scaled MMA with on-the-fly P quantization (`scale_groupwise`) and SFP R2S copy. **Currently slower than BF16** because the softmax warp is the pipeline bottleneck — P quantization adds to the critical path.

| Config | FP4 QKV (ms) | BF16 (ms) | Speedup |
|--------|-------------|-----------|---------|
| b=1 s=256 h=16 d=128 | 0.028 | 0.039 | 1.42x ² |
| b=1 s=1024 h=16 d=128 | 0.027 | 0.041 | 1.52x ² |
| b=4 s=4096 h=16 d=128 | 1.287 | 1.217 | 0.95x |
| b=1 s=4096 h=12 d=128 | 0.435 | 0.336 | 0.77x |
| **b=1 s=32768 h=12 d=128** | **13.693** | **12.775** | **0.93x** |
| b=1 s=4096 h=24 d=128 | 0.538 | 0.486 | 0.90x |
| b=1 s=32768 h=24 d=128 | 27.053 | 22.617 | 0.84x |

² Small shapes are faster due to reduced memory traffic, but the slowdown at large shapes reflects the softmax bottleneck.

Precision (debug mode, constant data within FP4 range): max_diff = 0.0 (exact match). Random data: max_diff = 0.9–2.9, mean_diff = 0.01–0.09 (expected for FP4 E2M1 with SF=1.0).

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

## Pipeline Graph
![pipeline graph](figures/pipeline.png)
