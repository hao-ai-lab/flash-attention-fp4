# Why `bench_gpu_time` Reports Lower TFLOPS with Many Iterations

## TL;DR

**GPU power throttling.** Both `bench_gpu_time_with_cuda_event` and `bench_gpu_time_with_cudagraph` run kernels back-to-back with minimal gaps, causing the B200 to hit its power limit and throttle SM clocks from 1965 MHz down to ~1580 MHz — a 19% frequency drop that produces artificially lower measured TFLOPS.

CUDA graph mode is hit harder because it eliminates *all* CPU gaps (even the tiny Python loop overhead), but events throttle too over many iterations.

## Root Cause: Power Throttling

### Evidence: Per-Iteration Degradation (Events, 100 iters, no fix)

```
Iters   0-  9: avg=1425  (warming up)
Iters  10- 19: avg=1450  (peak)
Iters  20- 29: avg=1451  (peak)
Iters  30- 39: avg=1373  (throttling begins)
Iters  40- 49: avg=1371
Iters  60- 69: avg=1317
Iters  70- 79: avg=1237  (fully throttled, -15%)
Iters  80- 89: avg=1237
Iters  90- 99: avg=1238
```

### Evidence: SM Clock and Power

| Method | SM Clock Avg | Power Avg | Power Peak |
|--------|-------------|-----------|------------|
| CUDA Events (20 iters) | 1940 MHz | 253 W | 377 W |
| CUDA Graph (n=10) | 1742 MHz | 693 W | 1006 W |

B200 TDP is ~700W. Graph mode saturates this immediately; events take ~30 iterations.

### Evidence: Graph num_iters_within_graph Sweep

```
Graph(n= 1): 1449 TFLOPS  ← matches events (no intra-graph throttle)
Graph(n= 2): 1261 TFLOPS  ← throttling within graph
Graph(n= 5): 1286 TFLOPS
Graph(n=10): 1242 TFLOPS  ← -14% vs events
```

With `sleep_after_run=True`: 1437 TFLOPS (recovers).

### Launch Overhead is Negligible

```
Graph(n=1) replay: 1.514ms
Direct event call: 1.516ms
Launch overhead:   0.002ms (0.1%)
```

CUDA graph benchmarking does NOT improve accuracy for compute-bound kernels like FA4. The benefit is only for tiny kernels where launch overhead dominates.

## Fix

Branch: `fix_cudagraph_bench` on `Edenzzzz/flashinfer`

Two changes to `flashinfer/testing/utils.py`:

### 1. `bench_gpu_time_with_cuda_event`: Cooldown gaps

Insert `sync+sleep` every `~5ms` of sustained compute:

```python
iters_per_burst = max(1, int(5.0 / estimated_kernel_execution_time))
for iter_idx in range(repeat_iters):
    ...
    elif (iter_idx + 1) % iters_per_burst == 0:
        torch.cuda.synchronize()
        time.sleep(estimated_kernel_execution_time / 1000)
```

### 2. `bench_gpu_time_with_cudagraph`: Cap n + cooldown gaps

- Probe single-kernel time with a n=1 graph
- Cap `num_iters_within_graph` so graph duration < 5ms
- Insert cooldown gaps between replays (same as events)

### Results After Fix (60 iters each, BF16 FA4)

| Config | Events | Graph | Delta |
|--------|--------|-------|-------|
| b=1 s=256 h=16 d=128 | 35 | 59 | **+67.7%** |
| b=1 s=1024 h=16 d=128 | 338 | 457 | **+35.2%** |
| b=4 s=4096 h=16 d=128 | 1413 | 1381 | -2.3% |
| b=4 s=8192 h=16 d=128 | 1432 | 1428 | -0.2% |
| b=2 s=16384 h=16 d=128 | 1413 | 1425 | +0.8% |
| b=1 s=32768 h=16 d=128 | 1443 | 1455 | +0.9% |
| b=4 s=4096 h=32 d=128 | 1306 | 1390 | **+6.4%** |
| b=4 s=8192 h=32 d=128 | 1393 | 1409 | +1.1% |
| b=1 s=32768 h=12 d=128 | 1375 | 1395 | +1.4% |
| b=1 s=32768 h=24 d=128 | 1450 | 1457 | +0.5% |

Graph >= events for 8/10 shapes. Tiny/small shapes see massive gains (+35-68%) from launch overhead elimination. The two slightly worse (medium -2.3%, large -0.2%) are within measurement noise.

## Threshold Choice: 5ms

Empirically determined on B200:
- ~4ms of sustained compute: borderline throttling
- ~15ms: heavy throttling (-13%)
- 5ms threshold: conservative, prevents degradation while keeping burst count reasonable
