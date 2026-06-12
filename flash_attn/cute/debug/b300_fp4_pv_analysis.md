# B300 Block-Scaled Attention Performance Analysis

**GPU**: NVIDIA GB300 SXM6 AC (SM 10.3, 2070 MHz max boost, 148 SMs, 1300W, unlocked clocks)
**Branch**: fp4_B300 · **cutlass-dsl**: 4.5.2 · **Updated**: 2026-06-12

## Summary

B300 (SM 10.3) doubles MUFU.EX2 throughput (32 ops/clk/SM vs 16 on SM100),
so exp2 is no longer the softmax co-bottleneck; the P conversion/quantization
stream is. Two optimizations shipped (both default-on, see below):

| PV mode (s=32768, h=24, d=128) | before | after | gain |
|--------------------------------|--------|-------|------|
| NVFP4 QK + BF16 PV             | 2069-2142 | same | — |
| NVFP4 QK + FP8 PV              | 1909   | **2306-2369** | +21-24% |
| NVFP4 QK + FP4 PV              | 1388   | **1595** | +15% |

FP8 PV now beats BF16 PV on B300 (it previously lost), restoring the
B200-style ordering. Full TFLOPS table at the end of this doc.

Per-mode bottlenecks (measured, see trace section):
- **BF16 PV**: MMA-bound (BF16 PV GEMM is the slow side).
- **FP8 PV**: softmax-bound via F2FP P-cast (exactly half the BF16 cvt rate);
  largely mitigated by the 3/4 P-handoff split.
- **FP4 PV**: softmax-bound via P quantization; mitigated by log-domain quant.

## Measured Instruction Throughput (GB300, SM103)

From `agent_space/bench_cvt_throughput.cu` (single SM, 8 independent serial
chains per thread, 512 threads = saturated):

| Instruction                      | instr/clk/SM | Note                              |
|----------------------------------|--------------|-----------------------------------|
| `ex2.approx.ftz.f32` (MUFU)     | **32.0**     | 2x SM100's 16 — the B300 doubling |
| `cvt.rn.bf16x2.f32`             | **62**       | ~64 limit; trivial FP32 truncation|
| `cvt.rn.satfinite.e4m3x2.f32`   | **32.0**     | FP8: exactly half the BF16 rate   |
| `cvt.rn.satfinite.e2m1x2.f32`   | **57**       | FP4 cvt itself is fast            |
| mix 4x ex2 + 4x e4m3 cvt        | 54.6 combined| partial port overlap, not full 64 |

Why FP8's P-cast costs 2x BF16's at identical instruction count: BF16 is a
round+truncate of the top FP32 bits (full-width datapath), while E4M3 needs
exponent rebias, mantissa renormalization, saturation and NaN remapping —
implemented at half rate. FP4's E2M1 cvt is nearly as fast as BF16; the FP4
quant cost is the surrounding group_max / scale / register traffic, not the
conversion. MMA throughput is unchanged vs SM100. For reference, FA4 paper
Table 1 (B200, M=N=d=128 per tile): BF16 MMA 1024 cy, FP4 MMA 256 cy,
SMEM 768 cy, Exp 1024 cy — i.e. exp2 WAS an MMA co-bottleneck on B200.

## Real In-Kernel Trace

Flashinfer-style timestamp instrumentation (`flash_attn/cute/profiler.py`),
compiled in via `FA4_PROFILE_PIPELINE=1`. One elected lane per warpgroup
records `%clock` cycles into a gmem buffer; `debug/trace_pipeline.py`
captures and renders a 3-row timeline (MMA warp, Softmax WG0, Softmax WG1)
with avg step costs under the title.

Two granularities:
- **Coarse (default)**: events only at wait/store boundaries that are
  already side-effect ordered — no timestamps inside the compute stream, so
  per-step numbers stay close to the clean kernel. One combined
  "softmax compute" span per step.
- **Detailed (`FA4_PROFILE_DETAIL=1`)**: per-phase load/exp/quant spans.
  **Each phase boundary's side-effecting `%clock` asm blocks ptxas from
  interleaving across it**, inflating spans 15-30% and shifting where waits
  land. Use for relative phase proportions only; never tune from detailed
  or instrumented runs alone — verify on clean kernels (and SASS).

Pre-optimization baseline characterization (coarse, block 0, b=1 s=4096
h=24 d=128, 96 softmax iterations/WG):

| Per step (cycles)            | BF16 PV | FP8 PV | FP4 PV |
|------------------------------|---------|--------|--------|
| softmax step period          | 4,855   | 4,001  | 5,389  |
| softmax step busy            | 2,352   | 2,288  | 3,469  |
| MMA QK+PV GEMM (one stage)  | 732     | 639    | 622    |
| MMA wait-P (per PV)          | 933     | 595    | 1,171  |

Key trace facts:
- GEMM *issue* spans are short (~250-650 cy) — tcgen05 MMAs execute
  asynchronously; the MMA warp's real exposure is its waits.
- The PV issue sequence embeds a second wait for P's 2nd half
  (`mbar_P_full_2` inside `gemm_ptx_partial*`), measured directly via
  `%clock` stores inside the GEMM's PTX (`prof_ts_addrs`) and drawn as an
  overlay in the PV bar. In steady state it is only 71-85 cy (mbarrier
  round-trip); PV-span variance is tcgen05 issue backpressure.
- FP4 PV (pre-log2-quant) spent ~1,250 cy/step on quantization on top of
  ~1,600 cy of exp — the MMA warp stalled 42% of its time waiting for P.

## PTX Instruction Census (baseline kernels, e2e=OFF, SM103)

| Category              | BF16 PV | FP8 PV  | FP4 PV (pre-log2) |
|-----------------------|---------|---------|---------|
| **Total instructions**| 5089    | 5089    | 6003    |
| ex2.approx (MUFU)    | 257     | 257     | 257     |
| cvt.rn.bf16x2.f32     | 128     | —       | —       |
| cvt.e4m3x2 (F2FP)    | —       | 128     | 8 (SF)  |
| cvt.e2m1x2            | —       | —       | 128     |
| max.f32               | 132     | 132     | 308     |
| fma/mul/add .f32x2    | 511     | 511     | 639     |
| mov (data movement)   | 2277    | 2277    | 2703    |

BF16 PV and FP8 PV differ ONLY in the 128 P-cast cvts — the perf gap is
pure hardware cvt throughput (table above). FP4 PV's +914 instructions are
the group quant: +176 max (group_max), +128 fma (scale; removed by
log-domain quant), +128 e2m1 cvt, +426 mov (group register traffic).

## Two Optimizations That Worked (June 11, default-on)

### 1. FP4 PV: log-domain group quantization (+15%)

`_fused_log2_group_quant` (`FA4_FP4_PV_LOG2_QUANT=1`, default). The baseline
quantized P as: exp2 all elements → per group: max → 1/x divide →
multiply-scale pass → E2M1 cvt. Using exp2 monotonicity
(`max(exp2 s) = exp2(max s)`), the group max moves to the PRE-exp scores and
the scale folds into the exp2 argument:

    m_g  = max(s_i)                      (same FMNMX count)
    P_i  = exp2(s_i - m_g + log2 6)      (subtract replaces the scale pass)
    SF_g = exp2(m_g - log2 6)            (1 extra ex2; == max(exp2 s)/6)

This deletes all 8 per-group `div.rn.f32` sequences (RCP + Newton FFMAs)
and the 64-FFMA post-exp scaling pass, and shortens the per-group dependency
chain from max→rcp→fma→cvt to max→sub→ex2. row_sum is rebuilt as
`sum_g SF_g * partial_g` (same value up to FP32 summation order). Measured:
1388 → 1595 TF at s=32768 (+15%), +8-9% at s=4096/8192, with error vs the
FP32 reference identical to baseline to all printed digits.

### 2. FP8 PV: P-handoff split 3/4 instead of 1/2 (+21-24%)

The first `mbar_P_full` signal releases the PV MMA after a fraction of P's
store chunks; the rest go through the embedded P_full_2 wait. The FP8
default (1/2, a B200 tuning) was far too early-release-biased for B300:
raising it to 3/4 (`FA4_FP8_PV_P_SPLIT_NUM/DEN`, now the default — the same
fraction BF16 PV always used) gives 1909 → 2366 TF at s=32768 and
1521 → 1812 at s=4096, bitwise-identical outputs. Safety contract: the
softmax-side split fraction must be >= the GEMM-side `pre_mbar_tiles`
fraction (both derive from `mbar_p_split`); 1/4 violates it for FP8's
K-tile count and crashes. FP4 is insensitive to the split (its PV GEMM is
too cheap for handoff latency to bind; tested 1/2, 3/4, 7/8 with store rep
8/4/2 via `FA4_FP4_PV_P_SPLIT_NUM/DEN`, `FA4_FP4_PV_TMEM_STORE_REP` —
all ~1595 TF).

## Overlap Investigation: Negative Results (measured)

Attempts to overlap the F2FP/quant stream with other hardware units
(MUFU, TMEM stores, st.shared) produced **no speedup — the overlap already
exists in the compiled SASS**:

1. Source-order knobs (`FA4_FP8_PV_USE_FUSED_PACK`, `FA4_FORCE_E2E`): all
   variants identical (1909-1910 TF).
2. Chunk-pipelined exp2/pack/TMEM-store paths
   (`FA4_FP8_PV_PACK_STORE_PIPELINE`, `FA4_FP4_PV_QUANT_STORE_PIPELINE`,
   kept env-gated off): bitwise-correct, exact parity at all shapes.
3. SASS comparison (ptxas -O3): baseline and pipelined compile to
   equivalently interleaved schedules (FP8: identical MUFU↔F2FP run
   structure; FP4: the per-group RCP→FMNMX→F2FP pattern already interleaves
   with the ex2 stream). The whole softmax step is one fully-unrolled
   branch-free region, so ptxas schedules across all source-level phases.
4. Materialized-loop control: rebuilding the FP8 exp+pack as a real
   `cutlass.range` loop (`FA4_FP8_PV_RANGE_UNROLL`) makes the fragment
   index dynamic → register-resident S/P spill to local memory
   (240 st.local/208 ld.local vs 0) → **227 TF, 8.4x slower**; with
   `unroll_full` the unroller restores constant indices and exactly matches
   baseline. Register residency REQUIRES the unrolled form; `range_constexpr`
   ≡ fully-unrolled `cutlass.range` (its `unroll=` kwarg is silently
   discarded by the DSL preprocessor).

Conclusion: the residual FP8/FP4 softmax cost is raw issue-slot count on
the cvt/MUFU/ALU ports, not scheduling — which is what pointed at
instruction elimination (log-domain quant) and handoff tuning (split)
instead of reordering.

## Other Findings

- **e2e exp2 emulation hurts on SM103** (hardware exp2 already fast):
  BF16 PV 1945→2142 TF with e2e off. Disabled via
  `_FP4_TUNING_CONFIG_SM103` (`enable_e2e: False`).
- **Register allocation is not the bottleneck**: sweeping
  `num_regs_softmax` 168-224 changed nothing for FP4 PV.
- **Pre-existing FP4 PV nondeterminism**: ~1 in 20 runs of the unmodified
  baseline differed from its own reference (diffs up to 3.8 on a couple of
  adjacent seq rows). Unrelated to the new code paths (bisected). With the
  log-domain quant default the repro went 0/20 — possibly timing-masked
  rather than fixed; keep watching.
- Compat fixes on this branch: `MmaF8F6F4Op` (cutlass-dsl 4.5 rename),
  `max_offset = log2(6)` for FP4 P range, rounding-mode API autodetect.

## Commands

```bash
# Benchmark all PV modes (NVFP4 QK)
CUDA_VISIBLE_DEVICES=1 python3 -m flash_attn.cute.benchmarks.bench_fp4              # BF16 PV
CUDA_VISIBLE_DEVICES=1 python3 -m flash_attn.cute.benchmarks.bench_fp4 --pv_mode fp8
CUDA_VISIBLE_DEVICES=1 python3 -m flash_attn.cute.benchmarks.bench_fp4 --quant_v    # FP4 PV

# Real in-kernel pipeline trace (coarse; add FA4_PROFILE_DETAIL=1 for phases)
CUDA_VISIBLE_DEVICES=1 python3 flash_attn/cute/debug/trace_pipeline.py --pv_mode {bf16,fp8,fp4}

# Instruction-throughput microbenchmark
nvcc -gencode arch=compute_103a,code=sm_103a -O3 -o bench_cvt agent_space/bench_cvt_throughput.cu

# Theoretical pipeline model (roofline-based, no GPU run needed)
python3 flash_attn/cute/debug/visualize_pipeline.py
```

## Results — PV Quantization (GB300)

NVFP4 QK attention with BF16, FP8 or NVFP4 PV (triton `do_bench`, GB300,
2026-06-12, log-domain FP4 quant + 3/4 FP8 P-split defaults):

| Config | NVFP4+BF16 | NVFP4+FP8 | NVFP4+FP4 | BF16 ref |
|--------|-----------|----------|----------|---------|
| b=1 s=256 h=16 d=128 | 13 | 14 | 10 | **16** |
| b=1 s=1024 h=16 d=128 | 205 | 230 | 168 | **279** |
| b=4 s=4096 h=16 d=128 | 2130 | **2215** | 1428 | 1514 |
| b=1 s=32768 h=16 d=128 | 2082 | **2182** | 1473 | 1651 |
| b=4 s=4096 h=32 d=128 | 1770 | **1934** | 1274 | 1530 |
| b=1 s=4096 h=12 d=128 | 1088 | **1152** | 768 | 984 |
| b=1 s=32768 h=12 d=128 ¹ | 2087 | **2271** | 1506 | 1624 |
| b=1 s=4096 h=24 d=128 | **1454** | 1356 | 1006 | 1303 |
| b=1 s=32768 h=24 d=128 | 2041 | **2306** | 1596 | 1586 |
| b=1 s=32768 h=24 d=64 | **1260** | 1114 | — | 1200 |

All values in TFLOPS. Peak: **NVFP4+FP8 2306 TF**, **NVFP4+BF16 2130 TF**,
**NVFP4+FP4 1596 TF**. **—** = unsupported (FP4 PV requires headdim ≥
sf_vec_size × 4 = 128 from the block-scaled MMA atom K constraint). Small
shapes (s ≤ 1024) are launch-latency dominated. The FP8 P-split gain grows
with seqlen; at s=4096 h=24 BF16 PV still edges FP8 PV.

¹ Matches [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) inference (480×832 video, 81 frames → latent seqlen 32760, nheads=12, headdim=128).
