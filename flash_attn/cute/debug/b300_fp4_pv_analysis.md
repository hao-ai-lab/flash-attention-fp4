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

### Pipeline Cycle Model (steady state, per iteration)

MMA warp processes `q_stage=2` tiles per KV block:
```
Per KV block: PV[0] + PV[1] + QK[0] + QK[1]
```

Each softmax WG handles one stage, ping-pong between WG0 (stage 0) and WG1 (stage 1):
```
Wait S → Load S → row_max → exp2 → row_sum → P pack/quant → Write P → Signal P_full
```

Softmax cycle counts below use the measured throughputs with a 2x contention
factor because in steady state the two softmax WGs overlap on the same SM and
share the vector pipes. Per softmax step each WG executes 16384 ex2 and 8192
packed-cvt thread-instructions (128×128 tile, 2 elements per cvt). This
models the PRE-optimization baseline (the log-domain quant removes part of
the FP4 P-quant row; the 3/4 split does not change per-step work):

| Component                  | BF16 PV | FP8 PV  | FP4 PV  |
|----------------------------|---------|---------|---------|
| **MMA warp per KV block**    |         |         |         |
| QK GEMM (FP4, ×2 stages)  | 512     | 512     | 512     |
| PV GEMM (×2 stages)       | 2048    | 1024    | 512     |
| Total MMA per block        | 2560    | 1536    | 1024    |
| **Softmax WG per stage**   |         |         |         |
| TMEM load + row_max        | 150     | 150     | 150     |
| exp2 (16384 / (32/2))     | 1024    | 1024    | 1024    |
| row_sum                    | 100     | 100     | 100     |
| P pack/quant               | 256     | 512     | 1500    |
| TMEM store + signal        | 60      | 60      | 60      |
| Total softmax per stage    | 1590    | 1846    | 2834    |
| **Bottleneck**             | MMA     | softmax | softmax |

P pack: BF16 = 8192/(62/2) ≈ 256; FP8 = 8192/(32/2) = 512; FP4 quant is
dominated by group_max + scale + register shuffling (+914 PTX instructions),
not the E2M1 cvt itself (57/clk measured).

The MMA warp must complete the full PV+QK cycle before the softmax results
from the same stage are needed again. With ping-pong, each softmax WG has
the entire MMA block cycle to complete its work. When softmax takes longer
than the MMA cycle, the MMA warp stalls waiting for P_full.

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

Measured means per event (detailed mode — relative proportions only, see
artifact warning above; pre-optimization baseline, block 0, b=1 s=4096
h=24 d=128, 96 softmax iterations per WG):

| Event (mean cycles)        | BF16 PV | FP8 PV  | FP4 PV  |
|----------------------------|---------|---------|---------|
| **MMA warp**                 |         |         |         |
| wait P (stall)             | 841 (33%)| 950 (38%)| 1306 (42%)|
| PV GEMM issue              | 560     | 405     | 549     |
| QK GEMM issue              | 262     | 267     | 246     |
| wait KV (TMA)              | 95      | 120     | 105     |
| **Softmax WG (per iter)**  |         |         |         |
| S load + row_max           | 697     | 821     | 666     |
| exp2 (+fused pack)         | 1581    | 1674    | 1579    |
| P quant / pack             | 75      | 325     | 1283    |
| P store + signal           | 380     | 230     | 177     |
| wait S                     | 581     | 759     | 594     |
| wait corr                  | 867     | 173     | 1013    |

(FP8's visible quant span is only ~325 cy because most of the F2FP work
interleaves into the exp2 span; FP4's ~1283-cy quant sat on top of the same
exp2 cost, which is what the log-domain quant attacked.)

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

## PTX Instruction Analysis

Static instruction counts from PTX (pre-optimization baseline kernels,
e2e=OFF, SM103 target, full kernel):

| Category              | BF16 PV | FP8 PV  | FP4 PV  | FP4 delta |
|-----------------------|---------|---------|---------|-----------|
| **Total instructions**| 5089    | 5089    | 6003    | +914 (18%)|
| ex2.approx (MUFU)    | 257     | 257     | 257     | 0         |
| fma.rn.f32x2          | 128     | 128     | 256     | +128      |
| mul.rn.f32x2          | 256     | 256     | 256     | 0         |
| add.rn.f32x2          | 127     | 127     | 127     | 0         |
| max.f32               | 132     | 132     | 308     | +176      |
| cvt (total)           | 266     | 266     | 276     | +10       |
| cvt.rn.bf16x2.f32     | 128     | 128     | 128     | 0         |
| cvt.e4m3x2 (F2FP)    | 0       | 128     | 8       | +8 (SF)   |
| cvt.e2m1x2            | 0       | 0       | 128     | +128      |
| selp                  | 160     | 160     | 157     | –3        |
| tcgen05 (MMA/TMEM)   | 107     | 107     | 119     | +12       |
| mbarrier              | 107     | 107     | 125     | +18       |
| mov (data movement)   | 2277    | 2277    | 2703    | +426      |
| INT ALU               | 1086    | 1086    | 1238    | +152      |

**FP8 PV vs BF16 PV**: Identical instruction count (5089). The only
difference is 128 `cvt.rn.bf16x2.f32` (BF16 P pack) replaced by 128
`cvt.rn.satfinite.e4m3x2.f32` (FP8 P pack / F2FP) — the perf gap is pure
hardware cvt throughput (table above).

**FP4 PV extra instructions** (+914, all in softmax WG, pre-log2-quant):
- +128 fma.f32x2: scale computation in `_fused_group_max_scale_quant`
  (removed by the log-domain quant)
- +176 max.f32: group_max reduction (16-element groups × 8 groups)
- +128 cvt.e2m1x2: E2M1 packing
- +426 mov: register shuffling for group processing
- +18 mbarrier: extra sync for SFP SMEM copy

## Why FP4 PV is Slower than BF16 Reference

The softmax WG does significantly more work for FP4 PV (baseline path):
1. `exp2()` — same MUFU cost as BF16
2. `update_row_sum()` — same (but moved BEFORE quant so it uses original
   P values)
3. **`compute_group_max()`** — per-group (16-element) max reduction → +176 max.f32
4. **`scale_groupwise()`** — per-element division by group_max → +128 fma.f32x2
   plus 8 `div.rn.f32` sequences per thread
5. **`_quant_fp4()` (E2M1 pack)** — 8 floats → 1 uint32 → +128 cvt.e2m1x2 + bit ops
6. **SF packing + R2S copy** — pack scale factors to UE4M3, copy to SMEM

Items 3–6 added ~1500 cycles per softmax stage vs the 1024-cycle MMA block
for FP4×FP4 PV GEMM — the MMA warp spent ~42% of its time stalled waiting
for P. Note the E2M1 cvt itself is fast (57/clk measured); the cost is the
group_max reduction, per-group scaling/divisions, and the register traffic
they generate. The log-domain quantization (below) removes items 4's
divisions and scaling pass entirely, taking FP4 PV from 1388 to 1595 TF;
items 3, 5, 6 remain the floor.

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

## Overlap Investigation: F2FP/quant vs Other Hardware Units (measured)

We tried to speed up FP8/FP4 PV by overlapping the bottleneck F2FP/quant
stream with instructions on other units (MUFU, TMEM stores, st.shared).
Result: **no speedup — the overlap already exists in the compiled SASS.**

What was tried (all bitwise-validated against baseline outputs):

1. **Knob sweep** (`FA4_FP8_PV_USE_FUSED_PACK`, `FA4_FORCE_E2E`, combinations):
   all variants land at 1909-1910 TF on (1, 32768, 24, 128). Source-order
   interleaving of exp2/F2FP has no effect.
2. **Chunk-pipelined FP8 path** (`FA4_FP8_PV_PACK_STORE_PIPELINE=1`,
   `_pack_fp8_store_pipelined`): software-pipelines exp2(chunk c) with
   F2FP-pack(c-1) and tcgen05.st(c-1) — three data-independent streams on
   MUFU / cvt / TMEM ports — and fires P_full right after the first
   `mbar_p_split` chunks. Measured: 1521/1638/1909 TF at s=4096/8192/32768,
   identical to baseline (1521/1638/1909).
3. **Chunk-pipelined FP4 path** (`FA4_FP4_PV_QUANT_STORE_PIPELINE=1`,
   `_fused_group_max_scale_quant_store_pipelined`): issues each P chunk's
   TMEM store as soon as its groups are quantized. Measured 1389 vs 1387
   TF baseline — parity.

Why: dumping SASS for baseline vs pipelined (ptxas -O3, sm_103a) shows
**ptxas already produces an equivalently interleaved schedule for the
baseline**. FP8: identical MUFU.EX2/F2FP run structure (83 transitions in
both). FP4: the per-group `MUFU.RCP → FMNMX×10 → F2FP×8` quant pattern is
already finely interleaved with the MUFU.EX2 stream (160 vs 168 unit-runs).
The whole softmax step is one fully-unrolled basic block, so ptxas freely
schedules across the source-level phases, and the hardware scoreboard
dual-issues across ports where possible (measured mixed ex2+F2FP throughput
54.6/clk vs 32 each in isolation — already reflected in kernel timing).

**Materialized-loop control (measured)**: to verify that the full unrolling
(not compiler magic) is what enables the overlap, we rebuilt the FP8
exp2+pack as a real `cutlass.range` IR loop over fragments
(`FA4_FP8_PV_RANGE_UNROLL`, `_exp2_pack_fp8_range`). With a live loop
(unroll=1 or 2), the dynamic fragment index makes the register-resident
S/P tensors unaddressable, so they spill to local memory (240 st.local +
208 ld.local in PTX vs 0 baseline): **227 TF, an 8.4x slowdown**. With
`unroll_full` the IR unroller restores constant indices and the result is
exactly baseline (1909 TF) — confirming `range_constexpr` ≡ fully-unrolled
`cutlass.range`, and that register residency requires the unrolled form.
(Also: the `unroll=` kwarg on `range_constexpr` is silently discarded by
the DSL preprocessor — it only means something on `cutlass.range`.)

**Instrumentation artifact warning**: with `FA4_PROFILE_PIPELINE=1`, the
chunk-pipelined FP8 variant looks ~14% faster per CTA than the instrumented
baseline (MMA wait-P 1055→713 cy). This is an artifact: the profiler's
side-effecting `%clock` inline asm between the exp/pack/store phases acts
as a scheduling barrier and prevents ptxas from interleaving the baseline.
The clean kernels are identical. Do not tune from instrumented runs alone.

The remaining FP8/FP4 PV gap is therefore raw issue-slot count on the
cvt/MUFU/ALU ports, not scheduling — which pointed at instruction
ELIMINATION (log-domain quant) and pipeline-handoff tuning (split) instead
of reordering. Both paid off (previous section).

## Other Findings

- **e2e exp2 emulation hurts on SM103** (hardware exp2 already fast at
  32/clk). Disabled by default via `_FP4_TUNING_CONFIG_SM103`
  (`enable_e2e: False`):

  | Mode    | e2e=ON  | e2e=OFF | Reason                                 |
  |---------|---------|---------|----------------------------------------|
  | BF16 PV | 1945 TF | 2142 TF | exp2 already fast; e2e adds FFMA work  |
  | FP8 PV  | 1851 TF | 1824 TF | F2FP is the bottleneck, not exp2       |
  | FP4 PV  | 1243 TF | 1340 TF | e2e adds instructions to overloaded WG |
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
