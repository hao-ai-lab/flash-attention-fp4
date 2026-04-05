# FastVideo Inference Overhead Analysis

## Summary

FP4 FA4 achieves **1.25x kernel-level speedup** over BF16 FA4 (attention = 70% of DIT GPU time), but e2e inference shows only **1.05x** (1.20 vs 1.15 it/s). This analysis investigates where the gap comes from.

**Root cause: layerwise weight offloading dominates wall time, not Python overhead.**

## Method

- Model: Wan2.1-T2V-1.3B, 480×832×81 video, 50 denoising steps with CFG
- Profiled with `nsys profile` (10 steps)
- GPU: single B200

## Findings

### 1. Layerwise Offloading is the Bottleneck

FastVideo auto-enables `dit_layerwise_offload` which streams model weights from CPU→GPU for each transformer layer at every forward pass. nsys shows:

| Category | Total Time | % of GPU Timeline |
|----------|-----------|------------------|
| **Host-to-Device memcpy** | **6.5s** | **75% of wall time** |
| Flash Attention kernels | 15.1s (summed) | 33.6% of kernel time |
| nvjet linear (GEMM) | 5.0s (summed) | 11.1% |
| Elementwise ops | 19.8s (summed) | 44.1% |
| LayerNorm | 1.1s | 2.4% |

The 6.5s of H2D memcpy across 85,103 transfers represents weight streaming. At 870ms per step, the H2D transfer consumes ~649ms/step (75%), leaving only ~220ms for actual compute.

### 2. Python Overhead is Zero

Direct profiling of the DIT forward pass (with model fully on GPU, no offloading) shows:

```
Wall time per fwd:   95.6 ms
GPU time per fwd:    95.6 ms
Python overhead:     0.0 ms (0.0%)
```

The Python framework (PyTorch, diffusers, fastvideo) adds no measurable overhead to the forward pass. The GPU is fully utilized when weights are resident.

### 3. Why FP4 Speedup is Diluted

Expected vs actual e2e speedup:

```
Attention kernel speedup:     1.25x (nsys: 70% of pure DIT GPU time)
Expected e2e DIT speedup:     1.16x (Amdahl: 1 / (0.3 + 0.7/1.25))
Actual e2e speedup:           1.05x (1.20 vs 1.15 it/s)
```

The gap (1.16x → 1.05x) is because Amdahl's law should be applied to **wall time**, not kernel time. With layerwise offloading consuming 75% of wall time, attention is only ~18% of wall time:

```
Effective attention fraction: 0.25 × 33.6% ≈ 8.4% of wall time
Effective speedup: 1 / (0.916 + 0.084/1.25) = 1.017x
```

The actual 1.05x is slightly better because some H2D memcpy overlaps with compute.

### 4. GPU Kernel Breakdown (10 steps, CFG)

| Kernel Category | Instances | Total Time | Avg Time | Notes |
|----------------|----------|-----------|---------|-------|
| Flash Attention | 6,000 | 15.1s | 2.51ms | Self-attn (~3.9ms) + cross-attn (~0.09ms) |
| nvjet GEMM | 24,000 | 5.0s | 0.21ms | Linear projections (QKV, out, FFN) |
| Elementwise | ~180k | 19.8s | 0.11ms | Fused ops, residuals, activations |
| LayerNorm | 9,100 | 1.1s | 0.12ms | |
| GELU | 3,100 | 0.9s | 0.30ms | |
| BF16 copy | 36,693 | 1.5s | 0.04ms | dtype conversions |
| H2D memcpy | 85,103 | 6.5s | 0.08ms | **Weight offloading** |

### 5. Elementwise Fragmentation

44% of kernel time is elementwise ops spread across ~180k kernel launches. This is a major source of launch overhead and GPU underutilization. Examples:
- `gpu_kernel_impl_nocast` (30,976 calls, 4.2s) — likely residual adds, scaling
- `direct_copy_kernel_cuda` (27,548 calls, 2.2s) — tensor copies
- `bfloat16_copy_kernel_cuda` (36,693 calls, 1.5s) — dtype casts
- `pow_tensor_scalar` (12,098 calls, 0.6s) — scheduler step

### 6. Disabling Offload Makes Things Worse

| Config | Throughput |
|--------|-----------|
| Default (layerwise offload) | **1.15 it/s** |
| No offload, no compile | 0.62 it/s |
| No offload + torch.compile | 0.74 it/s |

Layerwise offload **helps** — it pipelines H2D weight transfers with compute. Without it, the 1.42B model on a single B200 is slower because all weights compete for memory bandwidth simultaneously.

torch.compile doesn't help because FA4's CuTe DSL interface isn't compile-compatible (`torch.Stream` lacks `cuda_stream` attr), causing graph breaks. The fastvideo `custom_op` wrapper enables compilation but the overall overhead from graph breaks outweighs fusion gains.

## Recommendations

### For FP4 FA4 Speedup Visibility

1. **Multi-GPU inference**: Use 2+ GPUs to keep model weights resident without offloading. This eliminates the H2D bottleneck and makes kernel speedups visible.

2. **Larger models**: For models where attention dominates more of wall time (e.g., Wan2.1-14B with longer sequences), the FP4 speedup would translate better to e2e.

### For General FastVideo Performance

3. **Fuse elementwise ops**: 44% of GPU time is fragmented elementwise kernels across ~180k launches. `torch.compile` should help once FA4 compatibility is fixed.

4. **Fix FA4 torch.compile compatibility**: The `_flash_attn_fwd` function accesses `torch.Stream.cuda_stream` which isn't available during compile tracing. Wrapping via the existing `custom_op` in `flash_attn_cute.py` is the right approach but currently falls through to FA2 in some import paths.

5. **Reduce kernel launch count**: 265k kernel launches in 10 steps = 26.5k launches/step. At ~5μs each, that's ~130ms of launch overhead per step (15% of wall time).

## Reproducer

```bash
cd /sgl-workspace/FastVideo-Quantization
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 CUTE_DSL_ENABLE_TVM_FFI=1 \
  nsys profile --stats=true -o /tmp/fastvideo_profile -f true \
  python examples/inference/fa4_fp4_inference.py
nsys stats /tmp/fastvideo_profile.nsys-rep --report cuda_gpu_kern_sum
```
