# FP4 Flash Attention 4 (FA4) on Blackwell

CuTe DSL implementation of FP4 block-scaled flash attention for NVIDIA Blackwell GPUs (sm100a/sm103a). Supports two modes:
- **QK quantization** (`--quant_qk`, default): Q and K quantized to NVFP4 E2M1 with per-block E4M3 scale factors, V remains BF16. **1.01–1.39x speedup** over BF16 FA4, peaking at **1801 TFLOPS**.
- **QKVP quantization** (`--quant_v`): additionally quantizes the softmax output P and V to NVFP4 with on-the-fly group-wise P quantization. Currently **slower than BF16** (0.84–0.95x) due to hardware limitations: the softmax exp (MUFU instruction) has the same throughput as on H100, but B200 MMA throughput doubles, making softmax warp the bottleneck (See [FA4 paper](https://arxiv.org/abs/2603.05451)). The added P quantization + scale factor R->SMEM->TMEM copy increases critical-path latency.
We speculate that on B300 and Rubin (w/ FP16 softmax) the QKVP quantization will be faster than BF16.

## Results — QK Quantization

FP4 FA4 vs BF16 FA4 kernel speedup (CUDA event timing, vacant B200 GPU):

| Config | FP4 (ms) | FP4 TFLOPS | BF16 (ms) | BF16 TFLOPS | Speedup |
|--------|----------|------------|-----------|-------------|---------|
| b=1 s=256 h=16 d=128 | 0.015 | 37 | 0.015 | 35 | 1.01x |
| b=1 s=1024 h=16 d=128 | 0.023 | 379 | 0.025 | 338 | 1.12x |
| b=4 s=4096 h=16 d=128 | 0.336 | 1637 | 0.389 | 1413 | 1.16x |
| b=4 s=8192 h=16 d=128 | 1.259 | **1747** | 1.511 | 1455 | 1.20x |
| b=2 s=16384 h=16 d=128 | 2.467 | **1782** | 3.003 | 1464 | 1.22x |
| **b=1 s=32768 h=16 d=128** | **4.884** | **1801** | **6.771** | **1299** | **1.39x** |
| b=4 s=4096 h=32 d=128 | 0.655 | 1678 | 0.775 | 1418 | 1.18x |
| b=4 s=8192 h=32 d=128 | 2.501 | **1759** | 3.027 | 1453 | 1.21x |
| b=1 s=4096 h=12 d=128 | 0.104 | 986 | 0.117 | 882 | 1.12x |
| **b=1 s=32768 h=12 d=128** ¹ | **3.856** | **1711** | **5.056** | **1305** | **1.31x** |
| b=1 s=4096 h=24 d=128 | 0.152 | 1352 | 0.172 | 1198 | 1.13x |
| b=1 s=32768 h=24 d=128 | 7.551 | **1747** | 10.061 | 1311 | **1.33x** |
| b=1 s=32768 h=24 d=64 | 7.170 | 920 | 7.284 | 906 | 1.02x |

¹ Matches [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) inference (480×832 video, 81 frames → latent seqlen 32760, nheads=12, headdim=128).

Per-call precision: cosine similarity = 0.99, SNR = 7.25 (FP4 QK quantization vs BF16 reference).

> Reproduce with `python -m flash_attn.cute.benchmarks.bench_fp4`. Uses raw `torch.cuda.Event` timing — CUPTI (`bench_gpu_time`) and CUDA graphs both cause B200 throttling and report ~5% lower TFLOPS.

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

## Pipeline Graph (scale factor TMEM overlap schedule)
![pipeline graph](figures/pipeline.png)

## Citation
If you find our FP4 kernel useful, please cite:
```
@misc{zhang2026attnqat4bitattentionquantizationaware,
      title={Attn-QAT: 4-Bit Attention With Quantization-Aware Training}, 
      author={Peiyuan Zhang and Matthew Noto and Wenxuan Tan and Chengquan Jiang and Will Lin and Wei Zhou and Hao Zhang},
      year={2026},
      eprint={2603.00040},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.00040}, 
}
```