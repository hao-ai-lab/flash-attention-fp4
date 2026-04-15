# MXFP8 QK + FP8 PV support checklist

Date: 2026-04-14

Last updated: 2026-04-15

Goal: extend [`flash_fwd_sm100_fp4.py`](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/flash_fwd_sm100_fp4.py) so the QK GEMM can run block-scaled `mxfp8`, while the PV GEMM runs pure `fp8` instructions, then verify with [`bench_fp4.py`](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/benchmarks/bench_fp4.py).

Context from the current implementation:

- [x] Confirmed the current fast path is QK-only NVFP4, driven by [`bench_fp4.py`](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/benchmarks/bench_fp4.py) without `--quant_v`.
- [x] Confirmed the current `quant_v` path quantizes `P` inside softmax and uses block-scaled PV MMA, which is slower because softmax is on the critical path.
- [x] Confirmed existing notes already document the softmax bottleneck and the `quant_p` overhead:
  - [`fp4_flash_attention_optimization_notes.md`](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/debug/fp4_flash_attention_optimization_notes.md)
  - [`QKVP_PRECISION_FIX.md`](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/debug/qkvp/QKVP_PRECISION_FIX.md)

## Exploration findings

- [x] `interface.py` already accepts block-scaled FP8 combinations:
  - `ab_dtype in {Float8E4M3FN, Float8E5M2}`
  - `sf_dtype == Float8E8M0FNU`
  - `sf_vec_size == 32`
- [x] CUTLASS/CuTe helpers already expose the right MMA builders:
  - `make_blockscaled_trivial_tiled_mma(...)` uses `MmaMXF8Op` for block-scaled FP8
  - `make_trivial_tiled_mma(...)` uses `MmaFP8Op` for pure FP8
- [x] The FA kernel still has NVFP4-only assumptions that must be relaxed:
  - constructor asserts `sf_vec_size == 16` and `sf_dtype == Float8E4M3FN`
  - scale-factor shared-memory storage is hardcoded to `cute.Float8E4M3FN`
  - the P-scale-factor register path is hardcoded to UE4M3 packing for FP4 quantized PV
- [x] The inline PTX helper path in [`blackwell_helpers.py`](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/blackwell_helpers.py) still hardcodes MMA instruction kinds for some paths:
  - non-blockscaled helpers use `.kind::f16`
  - block-scaled FP4 helper uses `.kind::mxf4nvf4.block_scale.scale_vec::4X`
- [x] The referenced FlashAttention PR for pure FP8 support is the right direction for the PV GEMM instruction-kind update:
  - https://github.com/Dao-AILab/flash-attention/pull/2109

## Expected performance outcome

- [x] Working assumption recorded: changing QK from NVFP4 to MXFP8 should make little to no end-to-end difference because softmax remains the bottleneck in the QK-only path.
- [x] Working assumption recorded: changing PV to pure FP8 should help, because it removes the current `quant_p` / block-scaled PV overhead from the softmax critical path.

## Implementation checklist

- [x] Relax `FlashAttentionForwardSm100FP4.__init__` so block-scaled QK supports:
  - NVFP4: `Float4E2M1FN + Float8E4M3FN + sf_vec_size=16`
  - MXFP8: `Float8E4M3FN` or `Float8E5M2` + `Float8E8M0FNU + sf_vec_size=32`
- [x] Replace hardcoded FP8-E4M3 shared-memory scale-factor storage with `self.sf_dtype`.
- [x] Keep the existing on-the-fly `P` quantization path only for block-scaled PV.
- [x] Add a pure-FP8 PV path where:
  - `V` is FP8
  - `mSFV is None`
  - softmax writes `P` directly in FP8 form for the PV GEMM
  - PV uses pure FP8 MMA, not block-scaled PV MMA
- [x] Generalize inline PTX helper selection in `blackwell_helpers.py` so the emitted MMA kind matches the actual op:
  - `.kind::f16` for F16/BF16
  - pure FP8 kind for `MmaFP8Op`
  - `.kind::mxf8f6f4...` for block-scaled FP8
  - `.kind::mxf4nvf4...` for NVFP4
- [~] Verify the PTX helper path and `idesc` generation still agree for:
  - QK block-scaled FP8
  - PV pure FP8
- [x] Extend the benchmark driver so it can exercise:
  - current QK-only NVFP4 baseline
  - MXFP8 QK + BF16 V
  - MXFP8 QK + pure FP8 V
- [x] Keep the existing benchmark script as the source of truth for end-to-end verification instead of adding a separate benchmark.

## Current blocker

- [x] Confirmed the original NVFP4 QK-only benchmark still works after the refactor.
- [x] Confirmed standalone dense block-scaled GEMM works for MXFP8 on this machine:
  - `dense_blockscaled_gemm_persistent_prefetch.py --ab_dtype Float8E4M3FN --sf_dtype Float8E8M0FNU --sf_vec_size 32 ...`
  - result: `PASS`
- [x] Narrowed the remaining failure to the FA-specific MXFP8 QK path, not general CuTe/CUTLASS support.
- [x] Reproduced the failure for both:
  - MXFP8 QK + BF16 V
  - MXFP8 QK + FP8 V
- [x] Failure mode recorded:
  - kernel launch reaches compile/runtime setup, then fails inside `_flash_attn_fwd.compile_cache[compile_key](*call_args)`
  - CUDA error: `cudaErrorMisalignedAddress`
- [x] One FA mismatch vs dense GEMM was fixed:
  - SFB helper tiled MMA now uses `CtaGroup.ONE` instead of `self.cta_group`
  - this did **not** resolve the MXFP8 fault
- [x] A second FA mismatch vs dense GEMM was fixed:
  - MXFP8 QK SFQ/SFK TMEM spacing now uses the dense-style `16` u32-column separation (`0x10`) instead of the bad `4`-column split (`0x4`)
  - the FA kernel still fails with `cudaErrorMisalignedAddress`, so the remaining bug is elsewhere in the MXFP8 QK path
- [x] Pure-FP8 PV benchmark plumbing was fixed:
  - [`bench_fp4.py`](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/benchmarks/bench_fp4.py) now builds FP8 `V` tensors with the requested `--fp8_dtype` instead of accidentally reusing the NVFP4 QK dtype
  - NVFP4 QK + FP8 PV now runs end-to-end
- [x] Pure-FP8 PV softmax underflow handling was fixed:
  - switched from the ad hoc post-exp scaling hack to the upstream-style `max_offset=8` / `p_log2_offset=8` handling
  - LSE now subtracts the same offset back out after `row_sum` accumulation
- [x] Pure-FP8 PV `P` TMEM store layout bug was fixed:
  - the `tStP` store copy must use the FP8 repetition shape when `P` is consumed by pure FP8 PV MMA
  - the old code keyed that copy shape off `quant_pv`, which was wrong for `NVFP4 QK + FP8 PV`
  - after switching the pure-FP8 PV path to `St32x32bOp(Repetition(8))`, the long-sequence sparse `NaN` rows disappeared
- [x] Pure-FP8 PV scheduling sweep was run end-to-end with the benchmark script:
  - default no-cap `kv_stage` with the old `P` split at `3/4`
  - `kv_stage` capped at `4` with `P` split at `3/4`
  - no-cap `kv_stage` with earlier `P` split at `1/2`
  - `kv_stage=4` with earlier `P` split at `1/2`
- [x] Best current FP8-PV setting is recorded in code:
  - keep `kv_stage` uncapped by default for pure FP8 PV
  - release `P` to the PV consumer at `1/2` instead of `3/4`
- [x] Nsight Compute recheck was run on the key shape `(4, 4096, 16, 128)` comparing:
  - `NVFP4 QK + BF16 PV`
  - `NVFP4 QK + FP8 PV` with the best current `P` split (`1/2`)
- [x] Nsight finding recorded:
  - the FP8-PV kernel still launches at `128 regs/thread` and remains limited to `1` resident block by both register and shared-memory limits
  - FP8-PV reduces DRAM throughput but still has lower SM throughput than `NVFP4 QK + BF16 PV`, so the remaining regression is scheduler / overlap inefficiency inside the fused kernel rather than missing FP8 MMA instructions
- [x] PTX / instruction-mix root cause identified for pure FP8 PV:
  - PV MMA count was already reduced as expected, but the generic `Float32 -> Float8E4M3FN` `P` store path introduced extra scalar conversion/packing work
  - saved PTX diff showed:
    - BF16 PV: `256x cvt.rn.bf16x2.f32`
    - FP8 PV: `128x cvt.rn.satfinite.e4m3x2.f32` plus `128x cvt.u32.u16`
- [x] Pure FP8 PV `P` packing was optimized for `Float8E4M3FN`:
  - bypassed the generic `.to(Float8E4M3FN)` store path
  - pack `P` explicitly with `packed_float_to_ue4m3(...)` from FP32 registers before the TMEM store
  - correctness remains clean on the benchmark sweep
- [ ] Remaining suspected root cause:
  - FA-specific Q/K scale-factor TMA layout, TMEM layout, or descriptor plumbing for MXFP8
  - likely around `sfq/sfk` setup in [`flash_fwd_sm100_fp4.py`](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/flash_fwd_sm100_fp4.py)

## Verification checklist

- [x] Functional: benchmark runs successfully in current NVFP4 QK-only mode.
- [ ] Functional: benchmark runs successfully in MXFP8 QK + BF16 V mode.
- [ ] Functional: benchmark runs successfully in MXFP8 QK + FP8 V mode.
- [x] Functional: benchmark runs successfully in NVFP4 QK + pure FP8 V mode.
- [x] Correctness: current NVFP4 QK-only still matches the benchmark BF16 reference in the debug run.
- [~] Correctness: compare each new mixed-precision mode against BF16 reference output from the same benchmark.
  - NVFP4 QK + pure FP8 V is now clean across the current benchmark sweep
  - MXFP8 QK modes still pending because QK is still blocked on `cudaErrorMisalignedAddress`
- [ ] Performance: confirm MXFP8 QK-only is roughly neutral relative to current NVFP4 QK-only.
- [ ] Performance: confirm MXFP8 QK + FP8 V is faster than the current `--quant_v` path that quantizes `P`.
- [x] Performance: collect initial NVFP4 QK + pure FP8 V measurements from the same benchmark script.
- [x] Performance: record current baseline numbers from the benchmark run.

## Current numbers

- [x] NVFP4 QK-only baseline rechecked with [`bench_fp4.py`](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/benchmarks/bench_fp4.py)
  - `Batch=1, SeqLen=256, Nheads=16, Headdim=128`: `0.014 ms`, `37.1 TFLOPS`
  - `Batch=1, SeqLen=1024, Nheads=16, Headdim=128`: `0.023 ms`, `376.8 TFLOPS`
  - `Batch=4, SeqLen=4096, Nheads=16, Headdim=128`: `0.330 ms`, `1667.5 TFLOPS`
  - `Batch=4, SeqLen=4096, Nheads=32, Headdim=128`: `0.647 ms`, `1699.0 TFLOPS`
  - `Batch=1, SeqLen=4096, Nheads=12, Headdim=128`: `0.102 ms`, `1006.6 TFLOPS`
  - `Batch=1, SeqLen=32768, Nheads=12, Headdim=128`: `3.830 ms`, `1722.6 TFLOPS`
  - `Batch=1, SeqLen=4096, Nheads=24, Headdim=128`: `0.148 ms`, `1394.9 TFLOPS`
  - `Batch=1, SeqLen=32768, Nheads=24, Headdim=128`: `7.294 ms`, `1808.9 TFLOPS`
  - `Batch=1, SeqLen=32768, Nheads=24, Headdim=64`: `7.138 ms`, `924.2 TFLOPS`
- [x] MXFP8 QK-only attempt currently fails before a usable number is produced:
  - first repro shape was `Batch=1, SeqLen=256, Nheads=16, Headdim=128`
  - shared memory print before failure: `Total shared memory used: 201.00 KB`
  - runtime error: `cudaErrorMisalignedAddress`
- [x] MXFP8 QK + FP8 V attempt also currently fails before a usable number is produced:
  - first repro shape was `Batch=1, SeqLen=256, Nheads=16, Headdim=128`
  - shared memory print before failure: `Total shared memory used: 219.00 KB`
  - runtime error: `cudaErrorMisalignedAddress`
- [x] NVFP4 QK + FP8 V benchmark rechecked after the softmax-offset + FP8 `P` store fixes:
  - `Batch=1, SeqLen=256, Nheads=16, Headdim=128`: `0.015 ms`, `36.6 TFLOPS`, `1.08x` vs BF16 reference, `max_diff=0.5105`
  - `Batch=1, SeqLen=1024, Nheads=16, Headdim=128`: `0.024 ms`, `363.7 TFLOPS`, `1.06x`, `max_diff=0.2725`
  - `Batch=4, SeqLen=4096, Nheads=16, Headdim=128`: `0.357 ms`, `1542.0 TFLOPS`, `1.09x`, `max_diff=0.2381`
  - `Batch=4, SeqLen=4096, Nheads=32, Headdim=128`: `0.702 ms`, `1567.4 TFLOPS`, `1.10x`, `max_diff=0.2324`
  - `Batch=1, SeqLen=4096, Nheads=12, Headdim=128`: `0.111 ms`, `932.1 TFLOPS`, `1.06x`, `max_diff=0.1548`
  - `Batch=1, SeqLen=32768, Nheads=12, Headdim=128`: `4.180 ms`, `1578.3 TFLOPS`, `1.13x`, `max_diff=0.0782`
  - `Batch=1, SeqLen=4096, Nheads=24, Headdim=128`: `0.160 ms`, `1290.4 TFLOPS`, `1.08x`, `max_diff=0.1710`
  - `Batch=1, SeqLen=32768, Nheads=24, Headdim=128`: `7.949 ms`, `1659.8 TFLOPS`, `1.26x`, `max_diff=0.0532`
  - `Batch=1, SeqLen=32768, Nheads=24, Headdim=64`: `7.831 ms`, `842.4 TFLOPS`, `0.93x`, `max_diff=0.0620`
- [x] Correctness status for NVFP4 QK + FP8 V:
  - all current benchmark configurations complete without `NaN`s
  - all current benchmark configurations stay well within the benchmark tolerance against the BF16 reference
- [x] NVFP4 QK + FP8 V best-current tuning rechecked with `P` split at `1/2` and no `kv_stage` cap:
  - `Batch=1, SeqLen=256, Nheads=16, Headdim=128`: `0.013 ms`, `40.3 TFLOPS`
  - `Batch=1, SeqLen=1024, Nheads=16, Headdim=128`: `0.023 ms`, `376.5 TFLOPS`
  - `Batch=4, SeqLen=4096, Nheads=16, Headdim=128`: `0.343 ms`, `1604.8 TFLOPS`
  - `Batch=4, SeqLen=4096, Nheads=32, Headdim=128`: `0.673 ms`, `1634.3 TFLOPS`
  - `Batch=1, SeqLen=4096, Nheads=12, Headdim=128`: `0.107 ms`, `966.8 TFLOPS`
  - `Batch=1, SeqLen=32768, Nheads=12, Headdim=128`: `4.071 ms`, `1620.5 TFLOPS`
  - `Batch=1, SeqLen=4096, Nheads=24, Headdim=128`: `0.154 ms`, `1340.1 TFLOPS`
  - `Batch=1, SeqLen=32768, Nheads=24, Headdim=128`: `7.742 ms`, `1704.1 TFLOPS`
  - `Batch=1, SeqLen=32768, Nheads=24, Headdim=64`: `7.626 ms`, `865.0 TFLOPS`
- [x] Direct end-to-end comparison against the current NVFP4 QK-only baseline:
  - FP8 PV is now much closer after the explicit `E4M3` pack path, but it still does **not** beat `NVFP4 QK + BF16 PV` on the large benchmark shapes
  - representative deltas vs baseline:
    - `(4, 4096, 16, 128)`: `0.343 ms` vs `0.335 ms`
    - `(4, 4096, 32, 128)`: `0.673 ms` vs `0.655 ms`
    - `(1, 32768, 12, 128)`: `4.071 ms` vs `3.836 ms`
    - `(1, 32768, 24, 128)`: `7.742 ms` vs `7.664 ms`
- [x] Nsight follow-up after explicit `E4M3` packing:
  - `(4, 4096, 16, 128)` pure-FP8 PV dropped from about `138.7M` total instructions to about `130.3M`
  - tensor-pipe instructions stayed at `393,216`, confirming the improvement came from reducing non-tensor conversion/packing overhead

## Overnight task

- [ ] Overnight: continue from the current blocker by debugging the FA-specific MXFP8 QK scale-factor path:
  - compare `sfq/sfk` TMA layout and TMEM layout setup against the working dense block-scaled GEMM path
  - focus on `flash_fwd_sm100_fp4.py` sections around `sfq/sfk` TMA creation, `make_tmem_layout_sfa/sfb`, and `mainloop_s2t_copy_and_partition`
- [ ] Overnight: once the misaligned-address fault is fixed, rerun [`bench_fp4.py`](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/benchmarks/bench_fp4.py) in all three modes:
  - current NVFP4 QK-only baseline
  - MXFP8 QK-only
  - MXFP8 QK + pure FP8 PV
- [x] Overnight: append the measured latency / TFLOPS deltas and any correctness deltas to this file for the NVFP4 QK + pure FP8 PV path.

## 2026-04-15 update

- [x] Repaired the benchmark environment so measurements are reproducible again:
  - interpreter: [`flash_attn/cute/.venv/bin/python`](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/.venv/bin/python)
  - env: `CUTE_DSL_ARCH=sm_100a`, `CUTE_DSL_ENABLE_TVM_FFI=1`
  - `PYTHONPATH=/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention:/sgl-workspace/flashinfer`
  - added missing venv deps required by `bench_fp4.py`: `packaging`, `pynvml`, `requests`, `click`, `ninja`, `tabulate`, `tqdm`
- [x] Re-ran the current NVFP4 QK-only baseline in the repaired env:
  - `(1, 256, 16, 128)`: `0.015 ms`
  - `(1, 1024, 16, 128)`: `0.023 ms`
  - `(4, 4096, 16, 128)`: `0.336 ms`
  - `(4, 4096, 32, 128)`: `0.654 ms`
  - `(1, 4096, 12, 128)`: `0.105 ms`
  - `(1, 32768, 12, 128)`: `3.850 ms`
  - `(1, 4096, 24, 128)`: `0.153 ms`
  - `(1, 32768, 24, 128)`: `7.643 ms`
  - `(1, 32768, 24, 64)`: `7.175 ms`
- [x] Re-ran the current best NVFP4 QK + FP8 PV path in the repaired env:
  - `(1, 256, 16, 128)`: `0.013 ms`
  - `(1, 1024, 16, 128)`: `0.024 ms`
  - `(4, 4096, 16, 128)`: `0.344 ms`
  - `(4, 4096, 32, 128)`: `0.676 ms`
  - `(1, 4096, 12, 128)`: `0.107 ms`
  - `(1, 32768, 12, 128)`: `4.078 ms`
  - `(1, 4096, 24, 128)`: `0.155 ms`
  - `(1, 32768, 24, 128)`: `7.760 ms`
  - `(1, 32768, 24, 64)`: `7.641 ms`
- [x] Revalidated the current end-to-end gap versus NVFP4 QK-only:
  - `(4, 4096, 16, 128)`: `0.344 ms` vs `0.336 ms`
  - `(4, 4096, 32, 128)`: `0.676 ms` vs `0.654 ms`
  - `(1, 32768, 12, 128)`: `4.078 ms` vs `3.850 ms`
  - `(1, 32768, 24, 128)`: `7.760 ms` vs `7.643 ms`
  - `(1, 32768, 24, 64)`: `7.641 ms` vs `7.175 ms`
- [x] Tested and reverted a single-phase pure-FP8 `P` handoff experiment:
  - this published the full `P` tile before the first PV barrier
  - it regressed badly and is not kept
  - representative regressions:
    - `(4, 4096, 16, 128)`: `0.407 ms`
    - `(4, 4096, 32, 128)`: `0.801 ms`
    - `(1, 32768, 12, 128)`: `4.915 ms`
    - `(1, 32768, 24, 128)`: `9.351 ms`
    - `(1, 32768, 24, 64)`: `7.826 ms`
- [x] Rechecked pure-FP8 TMEM store repetition:
  - current `St32x32bOp(Repetition(8))` is still the correct setting
  - forcing `Repetition(16)` on the key shape made latency much worse: `~0.451 ms`
- [x] Rechecked FP8 operand dtype for pure-FP8 PV:
  - `E4M3` remains best on the key shape: `~0.3441 ms`
  - `E5M2` is slower: `~0.3502 ms`
- [~] Started wiring the dedicated `gemm_ptx_partial_fp8` helper into the MXFP8 QK path:
  - current status: helper dispatch no longer uses the original FP4-only route
  - remaining result: both `MXFP8 QK + BF16 PV` and `MXFP8 QK + FP8 PV` still fail with `cudaErrorMisalignedAddress`
  - this means the MXFP8 QK fault is not fixed yet and remains the current blocker

## 2026-04-15 late update

- [x] Isolated the MXFP8 QK crash more precisely:
  - skipping `SFQ` S2T copy still faulted
  - skipping `SFK` S2T copy still faulted
  - forcing MXFP8 QK back through the existing generic block-scaled helper ran successfully
  - conclusion: the fault was in the dedicated `gemm_ptx_partial_fp8` path, not in the SFQ/SFK copy staging
- [x] Unblocked MXFP8 QK functionally by routing `gemm_ptx_partial_fp8` through the working generic block-scaled helper path.
- [x] Verified in fresh processes that both of these now run without the previous misaligned-address failure:
  - `MXFP8 QK + BF16 PV`
  - `MXFP8 QK + FP8 PV`
- [x] Re-ran the full end-to-end comparison with the current code:
  - `NVFP4 QK` baseline
    - `(1, 256, 16, 128)`: `0.014 ms`
    - `(1, 1024, 16, 128)`: `0.023 ms`
    - `(4, 4096, 16, 128)`: `0.334 ms`
    - `(4, 4096, 32, 128)`: `0.653 ms`
    - `(1, 4096, 12, 128)`: `0.103 ms`
    - `(1, 32768, 12, 128)`: `3.858 ms`
    - `(1, 4096, 24, 128)`: `0.152 ms`
    - `(1, 32768, 24, 128)`: `9.528 ms`
    - `(1, 32768, 24, 64)`: `7.176 ms`
  - `NVFP4 QK + FP8 PV`
    - `(1, 256, 16, 128)`: `0.013 ms`
    - `(1, 1024, 16, 128)`: `0.023 ms`
    - `(4, 4096, 16, 128)`: `0.342 ms`
    - `(4, 4096, 32, 128)`: `0.673 ms`
    - `(1, 4096, 12, 128)`: `0.106 ms`
    - `(1, 32768, 12, 128)`: `4.071 ms`
    - `(1, 4096, 24, 128)`: `0.162 ms`
    - `(1, 32768, 24, 128)`: `7.807 ms`
    - `(1, 32768, 24, 64)`: `7.632 ms`
- [ ] `NVFP4 QK + FP8 PV` still does not beat `NVFP4 QK` on every benchmark shape.
- [x] Ran `ncu` on the still-regressing `(1, 32768, 24, 64)` case and recorded the main kernel counters:
  - BF16 PV:
    - `gpu__time_duration.sum`: `11,658,816 ns`
    - `dram__bytes.sum`: `236,132,864`
    - `lts__t_bytes.sum`: `14,000,885,216`
    - `smsp__cycles_active.sum`: `7,676,213,467`
    - `shared_op_ld` bank conflicts: `1,004,597`
    - `shared_op_st` bank conflicts: `3,695,857`
    - shared memory per block: `217,088`
  - FP8 PV:
    - `gpu__time_duration.sum`: `12,406,688 ns`
    - `dram__bytes.sum`: `178,097,408`
    - `lts__t_bytes.sum`: `9,377,248,992`
    - `smsp__cycles_active.sum`: `8,191,504,535`
    - `shared_op_ld` bank conflicts: `35,413`
    - `shared_op_st` bank conflicts: `1,687,420`
    - shared memory per block: `222,208`
- [x] Interpretation from `ncu` on `(1, 32768, 24, 64)`:
  - FP8 PV is already reducing DRAM traffic, L2 traffic, and shared bank conflicts
  - despite that, the kernel is slower because active cycles still increase
  - the remaining bottleneck is therefore not “memory got worse” in the simple sense
- [x] Rechecked the KV-stage-cap hypothesis on the same bad shape and disproved it:
  - uncapped FP8 PV: `~8.75 ms`
  - `FA4_FP8_PV_KV_STAGE_CAP=8`: `~17.79 ms`
  - `FA4_FP8_PV_KV_STAGE_CAP=5`: `~18.93 ms`
  - so capping KV stages is not a fix for the current regression
- [ ] Overnight next step:
  - identify what extra work in the pure-FP8 PV path is increasing active cycles despite lower memory traffic
  - continue retuning only after measuring that cost directly on the bad shapes
