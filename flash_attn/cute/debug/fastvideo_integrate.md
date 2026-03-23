# FastVideo FP4 Flash Attention Integration

## Result
- **Attention kernel speedup** (FP4 FA4 vs BF16 FA4, vacant GPU, CUPTI bench_gpu_time):

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

- **Attention precision**: cos=0.99, SNR=7.25 per-call
- **Video quality**: Step 49 latent cos=0.96 vs BF16 after 50 denoising steps. Visually recognizable but accumulated FP4 error visible.
- **DIT profiling** (nsys, Wan2.1-T2V-1.3B, 480x832x81 video):
  - Attention (`flash_fwd_kernel`) = **70.1%** of total GPU time (1993ms / 2843ms)
  - Linear layers (nvjet GEMM) = 6.5%, elementwise = 14.5%, layernorm = 2.3%
  - Expected e2e speedup from 1.25x attention: 2843 / (1993/1.25 + 850) = **1.16x**
- **E2E timing** (both using FA4 backend, vacant GPUs):
  - Per-step: NVFP4=0.833s (1.20 it/s) vs BF16=0.877s (1.14 it/s) = **1.05x**
  - Total: NVFP4=51.9s vs BF16=51.2s (similar — FSDP overhead difference)
  - Gap vs expected 1.16x due to nvfp4_quantize + SF layout conversion overhead per call

## Setup
```bash
cd /sgl-workspace/FastVideo-Quantization
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
bash setup_fp4_fa4.sh   # clones fp4_quant branch, installs FA4 + cutlass-dsl + flashinfer
```
- Env vars: `FASTVIDEO_NVFP4_FA4=1`, `CUTE_DSL_ENABLE_TVM_FFI=1`

## Architecture

### Data flow
1. `FlashAttentionImpl._forward_nvfp4(q, k, v)` in `fastvideo/attention/backends/flash_attn.py`
2. `_nvfp4_quantize_for_fa4(tensor)` quantizes BF16→FP4 via flashinfer's `nvfp4_quantize`
3. FP4 data as `float4_e2m1fn_x2` torch tensors + SF in MMA layout passed to FA4 kernel
4. FA4 `interface.py` uses `make_ptr(Float4E2M1FN, data_ptr, gmem)` at compile and call time
5. FA4 kernel builds tensors from pointers via `make_ordered_layout` inside `@cute.jit`

### Key implementation details

**FP4 data**: `nvfp4_quantize` returns `(M, headdim/2)` uint8. Reshape to 4D, view as `float4_e2m1fn_x2`.

**SF layout conversion**: `nvfp4_quantize(layout_128x4)` outputs swizzled `[mTile, kTile, 32, 4, 4]` buffer. To convert to FA4's `(32, 4, rest_m, 4, rest_k, nheads, batch)`:
- Squash nheads into K dim: `t2d = (batch*seqlen, nheads*headdim)` so M tiles align with seqlen boundaries
- Reshape flat SF → `[batch, rest_m, nheads, rest_k, 32, 4, 4]`
- Permute to canonical `(batch, nheads, rest_m, rest_k, 32, 4, 4)` then to MMA layout
- stride[3] must equal 1 (matching `cvt_sf_MKL_to_M32x4xrm_K4xrk_L`)

**SF values**: `nvfp4_quantize` returns scale factors directly usable by FA4 MMA — no inversion needed despite the `inv_s` variable name.

**Pointer path**: FA4 kernel accepts `cute.Pointer` for Q/K with separate `q_ptr_shape`/`k_ptr_shape` tuples. `make_ordered_layout(shape, order=(3,2,1,0))` builds row-major tensor at compile time. Shapes are runtime (not Constexpr) to handle cross-attention where seqlen_q ≠ seqlen_k.

**Cross-attention**: Model has both self-attention (q_seq=k_seq=32760) and cross-attention (q_seq=32760, k_seq=512). Separate q/k shapes are essential — using the same shape for both caused the kernel to read out-of-bounds on K.

**Seqlen padding**: Non-multiple-of-128 seqlens are padded. V is also padded to match K. Output is trimmed back to original seqlen.

## Files modified

| File | Changes |
|------|---------|
| `fastvideo/attention/backends/flash_attn.py` | `_nvfp4_quantize_for_fa4()`, `FlashAttentionImpl._forward_nvfp4()`, env var detection |
| `flash_attn/cute/interface.py` | `make_ptr` compile/call path, `is_nvfp4_dtype` for e2m1, `head_dim*2` for float4_e2m1fn_x2, separate q/k shapes, `out_dtype=bf16` for FP4 |
| `flash_attn/cute/flash_fwd_sm100_fp4.py` | Accept pointers for mQ/mK, `make_ordered_layout` from q/k_ptr_shape, dtype via `value_type`/`element_type` |
| `examples/inference/fa4_fp4_inference.py` | `--nvfp4_fa4` flag, `use_fsdp_inference=False` for NVFP4 |
| `fastvideo/tests/ops/test_nvfp4_fa4.py` | Accuracy, cross-attention, speedup tests with CUDA events |

## Bugs found and fixed

1. **SF layout wrong**: Original reshape `(batch, seqlen, nheads, sf_k)` was wrong. `layout_128x4` is swizzled `[mTile, kTile, 32, 4, 4]`. Fixed by squashing nheads into K and decomposing correctly.
2. **Cross-attention crash**: Single `qk_shape` for both Q and K caused out-of-bounds read when seqlen_q ≠ seqlen_k. Fixed with separate `q_ptr_shape`/`k_ptr_shape`.
3. **`from_dlpack` on `float4_e2m1fn_x2`**: Gave wrong shape (D/2) and strides. Fixed by using `make_ptr` + `make_ordered_layout` inside kernel.
4. **Output dtype**: Was `q.dtype` (int8) instead of bf16 for FP4 path. Fixed.
5. **`is_nvfp4_dtype`**: Didn't detect `float4_e2m1fn_x2`. Added `e2m1` check.

## Open issues
- `use_fsdp_inference=True` incompatible with `make_ptr` path (FSDP shards invalidate pointers)
- NaN on 2nd video generation in same process (may be related to compile cache with different shapes)
- Step 3 in fastvideo_integrate TODO: zero-copy SF (currently allocates + copies for layout conversion)
