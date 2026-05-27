# FP4 Flash Attention 4 (FA4) on Blackwell

CuTe DSL implementation of FP4 block-scaled flash attention for NVIDIA Blackwell GPUs (sm100a/sm103a). Supports two modes:
- **QK quantization** (`--quant_qk`, default): Q and K quantized to NVFP4 E2M1 or MXFP8 E4M3 with block scale factors. V can be BF16 or FP8. Peaks at **2018 TFLOPS** (NVFP4+FP8) and **1948 TFLOPS** (MXFP8+FP8).
- **QKVP quantization** (`--quant_v`): additionally quantizes the softmax output P and V to NVFP4 with on-the-fly group-wise P quantization. Currently **slower than BF16** (0.84–0.95x) due to hardware limitations: the softmax exp (MUFU instruction) has the same throughput as on H100, but B200 MMA throughput doubles, making softmax warp the bottleneck (See [FA4 paper](https://arxiv.org/abs/2603.05451)). The added P quantization + scale factor R->SMEM->TMEM copy increases critical-path latency.
We speculate that on B300 and Rubin (w/ FP16 softmax) the QKVP quantization will be faster than BF16.

## Results — QK Quantization

Block-scaled QK attention with BF16 or FP8 PV (triton `do_bench`, B200):

| Config | NVFP4+BF16 | NVFP4+FP8 | MXFP8+FP8 | BF16 ref |
|--------|-----------|----------|----------|---------|
| b=1 s=256 h=16 d=128 | 34 | 39 | **40** | 35 |
| b=1 s=1024 h=16 d=128 | **418** | 416 | 414 | 380 |
| b=4 s=4096 h=16 d=128 | 1789 | **1875** | 1801 | 1479 |
| b=1 s=32768 h=16 d=128 | 1920 | **2016** | 1942 | 1543 |
| b=4 s=4096 h=32 d=128 | 1826 | **1920** | 1851 | 1471 |
| b=1 s=4096 h=12 d=128 | 1081 | **1118** | 1070 | 940 |
| b=1 s=32768 h=12 d=128 ¹ | 1823 | **1913** | 1846 | 1508 |
| b=1 s=4096 h=24 d=128 | 1482 | **1548** | 1480 | 1274 |
| b=1 s=32768 h=24 d=128 | 1887 | **2018** | 1948 | 1545 |
| b=1 s=32768 h=24 d=64 | 919 | **986** | — | 949 |

All values in TFLOPS. Peak: **NVFP4+FP8 2018 TF**, **MXFP8+FP8 1948 TF**.

¹ Matches [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) inference (480×832 video, 81 frames → latent seqlen 32760, nheads=12, headdim=128).

Per-call precision: cosine similarity ≥ 0.99 (block-scaled QK vs BF16 reference).

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
CUTE_DSL_ENABLE_TVM_FFI=1 python benchmarks/bench_fp4.py --qk_mode nvfp4 --pv_mode bf16
CUTE_DSL_ENABLE_TVM_FFI=1 python benchmarks/bench_fp4.py --qk_mode nvfp4 --pv_mode fp8
CUTE_DSL_ENABLE_TVM_FFI=1 python benchmarks/bench_fp4.py --qk_mode mxfp8 --pv_mode fp8
```

## Precision (vs BF16 flash_attn_func reference)

Each cell: cos_sim / max_diff / mean_diff. NVFP4 uses flashinfer `nvfp4_quantize` (adaptive per-block SF). MXFP8 uses torch-native FP8 + uniform E8M0 SF. B200 sm_100a, cutlass-dsl 4.4.2.

| Config (b,s,h,d) | NVFP4+BF16 | NVFP4+FP8 | MXFP8+FP8 |
|---|---|---|---|
| (1,256,16,128) | 0.9910 / 0.1562 / 0.0106 | 0.9904 / 0.1475 / 0.0109 | 0.9986 / 0.0605 / 0.0042 |
| (1,1024,16,128) | 0.9908 / 0.1846 / 0.0055 | 0.9901 / 0.2119 / 0.0057 | 0.9986 / 0.0459 / 0.0022 |
| (4,4096,16,128) | 0.9906 / 0.0445 / 0.0028 | 0.9899 / 0.0432 / 0.0029 | 0.9985 / 0.0215 / 0.0011 |
| (1,32768,16,128) | 0.9904 / 0.0112 / 0.0010 | 0.9897 / 0.0122 / 0.0010 | 0.9985 / 0.0057 / 0.0004 |
| (4,4096,32,128) | 0.9905 / 0.0605 / 0.0028 | 0.9898 / 0.0713 / 0.0029 | 0.9985 / 0.0225 / 0.0011 |
| (1,4096,12,128) | 0.9906 / 0.0674 / 0.0028 | 0.9899 / 0.0771 / 0.0029 | 0.9985 / 0.0146 / 0.0011 |
| (1,32768,12,128) | 0.9903 / 0.0175 / 0.0010 | 0.9896 / 0.0194 / 0.0010 | 0.9985 / 0.0042 / 0.0004 |
| (1,4096,24,128) | 0.9905 / 0.0586 / 0.0028 | 0.9898 / 0.0645 / 0.0029 | 0.9985 / 0.0215 / 0.0011 |
| (1,32768,24,128) | 0.9905 / 0.0115 / 0.0010 | 0.9899 / 0.0142 / 0.0010 | 0.9985 / 0.0046 / 0.0004 |
| (1,32768,24,64) | 0.9899 / 0.0215 / 0.0010 | 0.9892 / 0.0223 / 0.0011 | — |

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