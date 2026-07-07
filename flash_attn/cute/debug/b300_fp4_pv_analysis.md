# B300 Block-Scaled Attention Performance Analysis

**GPU**: NVIDIA GB300 SXM6 AC (SM 10.3, 2070 MHz max boost, 148 SMs, 1300W, unlocked clocks)
**Branch**: fp4_B300 · **cutlass-dsl**: 4.5.2 · **Updated**: 2026-06-12

## Summary

B300 (SM 10.3) doubles MUFU.EX2 throughput (32 ops/clk/SM vs 16 on SM100),
so exp2 is no longer the softmax co-bottleneck; the P conversion/quantization
stream is. Optimizations shipped (all default-on, see below):
ld.red fused S-load+row-max, log-domain FP4 quant, the 3/4 FP8 P-split,
MXFP8 PV, and a block-scaled-PV SF-stepping correctness fix.

| PV mode (s=32768, h=24, d=128) | session start | after | gain | acc (mean_abs) |
|--------------------------------|--------|-------|------|------|
| NVFP4 QK + BF16 PV             | 2069-2142 | **2176-2246** | +4% | 0.0028 |
| NVFP4 QK + FP8 PV              | 1909   | **2549-2633** | +33-38% | 0.0040 |
| NVFP4 QK + NVFP4 PV           | 1388   | **1708-1726** | +23-24% | 0.0039 |
| NVFP4 QK + MXFP8 PV (new)      | —      | **1755** | — | **0.0029** |

FP8 PV now clearly beats BF16 PV on B300 (it previously lost), restoring
the B200-style ordering. MXFP8 PV trades FP8 PV's speed for near-BF16
accuracy at FP4-PV speed. The SF-stepping fix also improved FP4 PV
accuracy 0.0146 → 0.0039. Full TFLOPS table at the end of this doc.

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
- **Detailed (`FA4_PROFILE_DETAIL=1`)**: per-phase load/exp/quant spans
  (`figures/pipeline_trace_{mode}_pv_detailed.png`). For BF16/FP8 the exp2
  and the P cast are separable phases. For **NVFP4/MXFP8 the exp2 is fused
  into the per-group quant loop** (log-domain path: fma → exp2 → pack per
  group), so there is no separable exp2 phase — the figure shows one
  combined **exp2 + P quant** span (red), not an empty MUFU sliver next to
  a quant block. **Each phase boundary's side-effecting `%clock` asm blocks
  ptxas from interleaving across it**, inflating spans 15-30% and shifting
  where waits land. Use for relative phase proportions only; never tune
  from detailed or instrumented runs alone — verify on clean kernels (and
  SASS).

Metric definitions (`trace_pipeline.py:step_stats`): one softmax **step**
is one KV iteration = `wait S` → compute (S load + row_max + exp2 + P
quant/pack) → `P store + signal` → `wait corr`. **softmax step period**
is the full iteration wall-clock (median gap between consecutive `wait S`
starts, stalls included); **softmax step busy** is the time the softmax WG
is actually computing/storing, not blocked on an mbarrier.

Pre-optimization baseline characterization (coarse, block 0, b=1 s=4096
h=24 d=128, 96 softmax iterations/WG):

| Per step (cycles)            | BF16 PV | FP8 PV | FP4 PV |
|------------------------------|---------|--------|--------|
| softmax step period          | 4,855   | 4,001  | 5,389  |
| softmax step busy            | 2,352   | 2,288  | 3,469  |
| MMA QK+PV GEMM (one stage)  | 732     | 639    | 622    |
| MMA wait-P (per PV)          | 933     | 595    | 1,171  |

Post-optimization (2026-06-12 defaults: ld.red row-max, log-domain quant,
3/4 FP8 P-split, SF-stepping fix; coarse, same shape — the current
`figures/pipeline_trace_{bf16,fp8,fp4,mxfp8}_pv.png` figures):

| Per step (cycles)            | BF16 PV | FP8 PV | FP4 PV | MXFP8 PV |
|------------------------------|---------|--------|--------|----------|
| softmax step period          | 4,800   | 3,962  | 4,381  | 4,140    |
| softmax step busy            | 2,369   | 2,242  | 2,429  | 2,280    |
| MMA QK+PV GEMM (one stage)  | 707     | 632    | 618    | 615      |
| MMA wait-P (per PV)          | 973     | 619    | 633    | 625      |

FP4's softmax busy dropped 3,469 → 2,429 (log-domain quant + ld.red) and
its step period 5,389 → 4,381. MXFP8 PV's busy time sits just +38 cycles
over FP8's — in the *instrumented* kernel the extra group-bias FMA pass
largely hides under MUFU latency. The end-to-end TFLOPS gap to FP8 PV
(1148 vs 1555 at this shape) is much larger than the step-period delta
(4,140 vs 3,962), so most of it lives in what the coarse softmax span
doesn't isolate: the SF S2T staging + extra barriers on the softmax→MMA
handoff and the per-tile tails (see the PTX-count section; remember the
instrumentation caveat above — clean-kernel TFLOPS is the ground truth).

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

## Optimizations That Worked (June 11-12, default-on)

### 0. SM103 tcgen05.ld.red: fused S load + row max (+4-12%, all modes)

`tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max` (SM103-only) returns each
x32 TMEM load's 32 values plus their max in a 33rd register, computed in
the TMEM controller — row max becomes 3 `fmax` over tile maxes instead of
a ~127-op `fmax_reduce` tree (`tmem_ld_red_max` in blackwell_helpers.py,
ported from LopezCastroRoberto/flash-attention `perf/ld.red-upstream`;
`update_row_max_precomputed` in softmax.py; default on via
`FA4_LDRED_ROWMAX`). The hardware max is only used where masking is a
compile-time no-op (the unmasked main loop); masked iterations
(seqlen boundary, causal, mask_mod) and score_mod fall back to the
software reduce over post-mask values. Outputs are bitwise identical
(max is exact). Measured at s=32768 h=24 d=128:
BF16 PV 2127→2215 (+4%), FP8 PV 2342→2633 (+12%), FP4 PV 1595→1718 (+8%).

Bonus, pre-plumbed: for sf_vec_size=32 (MXFP8 P, group size 32) each x32
tile max IS the scale-factor group max — `_fused_log2_group_quant` already
accepts `hw_group_maxes` and converts the raw-S tile maxes with one FMA
per group (`m'_g = m_raw_g*scale_log2 + (max_offset - row_max*scale_log2)`),
eliminating the whole FMNMX group reduce. This is now active in MXFP8 PV
(below).

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

### 3. MXFP8 PV: NVFP4 QK + E4M3 P/V with E8M0 SFs per 32 (implemented)

`--pv_mode mxfp8`: P→E4M3 in the log domain with exponent-only (UE8M0)
group SFs over 32 columns, against a K-major MXFP8 V (flashinfer
`mxfp8_quantize`). Implementation: per-GEMM sf params
(`sf_dtype_pv`/`sf_vec_size_pv` split from the QK side across tiled MMAs,
smem/tmem SF layouts, and the quant helpers); E8M0 bias = `ceil` of the
log-domain bias (`cvt.rpi.f32.f32`) since E8M0 is exponent-only; SF bytes
packed via `cvt.rz.satfinite.ue8m0x2.f32`; P packed 4/u32 with
`packed_float_to_ue4m3`. sf_vec=32 matches the ld.red x32 tile width, so
the per-group maxes come free from the fused S-load row max (one FMA per
group, zero FMNMX group reduces).

**Accuracy** (vs FP32 sdpa, b=1 s=4096 h=24 d=128, mean_abs): MXFP8 PV
**0.0029** ≈ BF16 PV 0.0028 < FP4 PV 0.0039 ≈ FP8 PV 0.0040 — the best
of the quantized-PV modes, at FP4-PV-level speed (see table).

**Why MXFP8 PV (1755 TF) is ~30% slower than FP8 PV (2578 TF)** — PTX
instruction counts of the fwd kernel (per thread, whole 2-stage softmax
body; `CUTE_DSL_KEEP_PTX=1`, same NVFP4 QK in all modes):

| instruction | FP8 PV | MXFP8 PV | FP4 PV | what it is |
|---|---|---|---|---|
| `ex2.approx` | 257 | 265 | 273 | exp2 (+SF exp2 for quant modes) |
| `fma.rn.f32x2` | 128 | **256** | 256 | scale: row-wise only vs per-group bias |
| `max.f32` | 70 | **150** | 246 | sw group-max fallback for masked steps † |
| `cvt` P pack | 128 e4m3x2 | 128 e4m3x2 | 128 e2m1x2 | same pack count |
| `cvt.rpi.f32` | 0 | 8 | 0 | E8M0 ceil |
| `tcgen05.cp` (S2T) | 16 | **24** | 32 | +SFP/SFV smem→tmem copies |
| `st.shared.v4` | 32 | 32 | 32 | (FP8's are P chunks, quant modes' are SF) |
| PV MMA ×16 | `f8f6f4` | `mxf8f6f4 1X` | `mxf4nvf4 4X` | block-scaled needs SF tmem reads |

† unmasked main-loop steps use the free hw group maxes (ld.red); the
software FMNMX reduce remains compiled in for masked iterations.

The P-pack cvt count is identical to FP8 PV — the gap is (a) **+128
`fma.f32x2` on the FMA pipe**: the log-domain quant runs a *second*
packed-FMA pass over every element to subtract the per-group bias
(`_fused_log2_group_quant`), on top of the scale-subtract-rowmax pass both
modes share — FP8 PV's single row-wise scale folds entirely into that
first pass. (The two passes could in principle fuse into one FMA with a
per-group constant `c_g = -(row_max*scale_log2 + bias_g)`; that needs the
quant path to own the exp2 prep instead of receiving pre-subtracted
scores — a future lever worth ~128 f32x2 FMAs/step.) (b) the **SFP
R2S + S2T round-trip and its barriers** sitting on the softmax→MMA
critical path, and (c) the **handoff structure**: FP8 PV releases the PV
MMA after 3/4 of plain P chunks (chunk-pipelined pack/store), while the
block-scaled path must also stage SFs through smem→tmem before the MMA
can start — the FP8-style pack/store pipelining hasn't been ported to it
(`fp4_pv_quant_store_pipeline` is asserted off for MXFP8). Same reasons
FP4 PV sits at ~1726 (see "Why FP4 PV is Slower than BF16 Reference");
MXFP8 PV inherits the block-scaled-PV cost structure, paying for accuracy
with the SF machinery rather than with element width.

### Found along the way: block-scaled PV read K-tile 0's SFs for every K-tile

`gemm_ptx_partial_fp4`'s TMEM-A path computed per-K-tile scale-factor
offsets with `find_tmem_tensor_col_offset(tScaleA[None, None, k])`. The
catch: slicing `[None, None, k]` *fixes* the MMA_K coordinate and folds
`k * stride_k` into the tensor's **iterator base address**, dropping the
MMA_K mode from the resulting `.layout` entirely — so the slice layout is
**byte-identical for every k** (verified: `slice[:,:,0]` and `slice[:,:,1]`
print the same layout). `find_tmem_tensor_col_offset` reads **only**
`.layout` (`cosize(layout) & 0xFFFF`), never the iterator base, so it
returns the same column span for every k and `offset_sfa[k] = col(k) -
col(0) = 0` for all k. The real per-tile stride is there — the full
(unsliced) layout has `stride 16` on MMA_K, i.e. `crd2idx((0,0,k))` steps
+16 columns — but slicing had hidden it in the base pointer that the
helper ignores. Result: `offset_sfa/offset_sfb` were always all-zero and
every K-tile of the PV MMA read K-tile 0's SFs. NVFP4 PV
(vec16: SFs step whole tmem columns) silently lost half its V/P scale
factors — fixing it improved FP4 PV accuracy **0.0146 → 0.0039** mean_abs.
MXFP8 PV (vec32) was completely broken: its 4 per-tile SFs live in the 4
*bytes of one tmem word*, selected by the SF-ID bits of the instruction
descriptor (idesc 30:29 for SFA, 5:4 for SFB), not by the address. Fix:
derive per-tile element offsets from the layout via `crd2idx`, split into
tmem column (`elem // 4`) and static SF-ID (`elem % 4`), and emit a
per-tile idesc. Diagnosed by a uniform-P test being exactly 0 error while
varying-SF inputs failed; `FA4_DEBUG_SF_OFFSETS=1` now prints the
per-tile offsets/IDs at trace time. FP8 PV is bitwise-unaffected (it
folds scales into normalization, no block-scaled PV).

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

## Precision: descale / two-level quant (investigated, not adopted)

Motivating observation (README precision tables): **NVFP4+FP8 has lower
overall error** (higher cos, lower mean_diff) than NVFP4+NVFP4 — FP8's
wider mantissa wins on average — yet at some shapes its **max_diff is
higher** (e.g. (1,1024,16,128): 0.2119 vs 0.1504). We chased whether
adding scale factors to the FP8 PV/QK paths would close that max-diff gap.
Conclusion: not worth it. All measurements via torch emulation of the
exact quant before any kernel change.

**1. FP8 V descale (sm100.py per-head `v_descale`): implemented, no
precision benefit.** E4M3 is *floating point*, so a uniform per-head
descale is scale-invariant — relative quant error doesn't change. This was
**implemented in the kernel** (`v_descale` arg on the fp4 kernel /
interface / `flash_attn_func`; loaded per `[batch, kv_head]` in
`correction_loop` and folded into the output-norm `scale`, mirroring
sm100.py) and measured directly. NVFP4 QK + FP8 V, b1 s1024 h16 d128, vs
the BF16 reference:

| path | max | mean |
|---|---|---|
| A: direct cast V→fp8, no descale (current) | 0.5942 | 0.00737 |
| B: per-head amax-scaled V→fp8 + `v_descale` | 0.5922 | 0.00737 |
| C: scaled V, **descale omitted** (control) | 44.74 | 3.817 |

B ≡ A to 5 digits → no precision change, as scale-invariance predicts; C
blows up → confirms the descale path is actually exercised (not silently
ignored). Speed is unchanged: 2538 → 2562 TF at s=32768 h=24 d=128 (the
per-row global load + multiply in the correction warp is free). So
`v_descale` is a correct **dequant API for externally-quantized FP8 V**
(FA3 semantics), not a precision lever — V is not where the error lives.
(The full-kernel mean 0.0074 ≫ the isolated V-quant ~0.001 because QK+P
error dominates.)

**2. FP8 P error is underflow, but the kernel already mitigates it.** ~88%
of softmax probs sit below E4M3's subnormal floor (2⁻⁹) and would flush to
zero on a naive [0,1] cast (max 0.093 / mean 0.0137 isolated). But the
kernel scales P per *row* via `max_offset = log2(448)` (row max → 448,
full E4M3 range), which already recovers most of it: per-row 0.0133 /
0.00108. Going finer helps the tail — per-128 0.0062, per-32 0.0051,
per-16 0.0048 — but a per-key-group P scale **requires block-scaled MMA
along the contraction (keys) dim**, which is exactly NVFP4 PV (g16) and
MXFP8 PV (g32). Those already exist; and they need per-group SF computed
in the softmax WG (register pressure, the very thing the log-domain quant
fights). End-to-end the PV improvement (~3e-4 mean) is swamped by the
NVFP4 QK error (~2.9e-3 mean), so it doesn't move the table.

**3. Two-level QK quant (per-head E8M0 coarse + per-16 NVFP4 fine): also
dropped.** The level-1 per-head scale dequantizes as `qk_descale` folded
into `softmax_scale_log2` — and with an E8M0 (power-of-2) coarse scale the
dequant is a pure *add* in the log domain (no mantissa lost, no multiply).
The bench currently runs single-level (`nvfp4_quantize(t, one)`, global
scale = 1). But the per-16 E4M3 block SF *already adapts to local
magnitude*, so level-1 is redundant except when per-head magnitude spread
is extreme enough to push the per-block SF out of E4M3's range
(448 … ~2⁻⁹) — not the case for LayerNorm-scale Q/K. And quantization
needs a **second pass over Q/K** (a global per-head amax reduction before
the per-block pass), which is too costly for ~no gain on well-scaled data.
Cheap dequant, expensive quant.

Net: the existing per-row P scaling + per-16/32 block SF (in the NVFP4/
MXFP8 PV modes) already cover the realistic precision range; the extra
descale/two-level machinery only helps pathological per-head magnitude
spread, which the benchmark (and typical post-LayerNorm activations) don't
exhibit.

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
CUDA_VISIBLE_DEVICES=1 python3 -m flash_attn.cute.benchmarks.bench_fp4 --pv_mode mxfp8

# Real in-kernel pipeline trace (coarse; add FA4_PROFILE_DETAIL=1 for phases)
CUDA_VISIBLE_DEVICES=1 python3 flash_attn/cute/debug/trace_pipeline.py --pv_mode {bf16,fp8,fp4}

# Instruction-throughput microbenchmark
nvcc -gencode arch=compute_103a,code=sm_103a -O3 -o bench_cvt agent_space/bench_cvt_throughput.cu

# Theoretical pipeline model (roofline-based, no GPU run needed)
python3 flash_attn/cute/debug/visualize_pipeline.py
```

## Results — PV Quantization (GB300)

All block-scaled QK x PV combinations (triton `do_bench`, GB300, 2026-07,
re-measured with a per-shape cooldown ³ — includes the ld.red row-max (incl.
BF16 ref), log-domain FP4 quant, 3/4 FP8 P-split, MXFP8 PV, and the
block-scaled-PV SF-stepping fix):

| Config | NVFP4+BF16 | NVFP4+FP8 | NVFP4+NVFP4 | NVFP4+MXFP8 | MXFP8+BF16 | MXFP8+FP8 | BF16 ref ² |
|--------|----|----|----|----|----|----|----|
| b=1 s=256 h=16 d=128 | 13 | 11 | 12 | 12 | 11 | 12 | **17** |
| b=1 s=1024 h=16 d=128 | 232 | 203 | 200 | 209 | 221 | 242 | **289** |
| b=4 s=4096 h=16 d=128 | 2227 | **2502** | 1524 | 1612 | 1849 | 2291 | 1508 |
| b=1 s=32768 h=16 d=128 | 2337 | **2666** | 1720 | 1807 | 2072 | 2383 | 1585 |
| b=4 s=4096 h=32 d=128 | 2196 | **2540** | 1544 | 1638 | 1879 | 2290 | 1475 |
| b=1 s=4096 h=12 d=128 | 1458 | **1458** | 942 | 987 | 1227 | 1418 | 1017 |
| b=1 s=32768 h=12 d=128 ¹ | 2276 | **2582** | 1646 | 1726 | 2021 | 2350 | 1584 |
| b=1 s=4096 h=24 d=128 | 1949 | **2046** | 1291 | 1360 | 1611 | 1974 | 1322 |
| b=1 s=32768 h=24 d=128 | 2235 | **2677** | 1725 | 1809 | 1974 | 2362 | 1533 |
| b=1 s=32768 h=24 d=64 | 1209 | 1203 | — | — | — | — | **1221** |

All values in **TFLOPS**. Peak: **NVFP4+FP8 2677 TF**, **MXFP8+FP8 2383 TF**,
**NVFP4+BF16 2337 TF**, **MXFP8+BF16 2072 TF**, **NVFP4+MXFP8 1809 TF**,
**NVFP4+NVFP4 1725 TF**. **—** = unsupported (d=64 needs head_dim >=
sf_vec_size x 4: NVFP4 PV and MXFP8 require 128). MXFP8+x columns are MXFP8 QK GEMM (sf_vec 32,
E8M0) with BF16/plain-FP8 PV GEMM; NVFP4+MXFP8 is NVFP4 QK with the new MXFP8
PV (E4M3 P/V, E8M0 SFs per 32) — slowest-but-most-accurate of the
quantized-PV modes (mean_abs 0.0029 vs FP8 PV's 0.0040, FP4 PV's 0.0039).

¹ Matches [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) inference (480x832 video, 81 frames -> latent seqlen 32760, nheads=12, headdim=128).

² **BF16 ref** is the non-block-scaled SM100 reference kernel
(`flash_fwd_sm100.py`), and it too uses the SM103 ld.red fused S-load+row-max
(`FA4_LDRED_ROWMAX`, default-on) — every column in this table includes it.
The BF16-ref figures here are post-ld.red (~+1-4% over the pre-ld.red ref).

³ **Per-shape cooldown (now default-on in `bench_fp4`).** The original 2026-06
table ran every shape back-to-back with **no sleep**, so shapes late in the
config list ran on an already-hot GPU and were **thermally throttled** — a
measurement artifact, not slower kernels. It hit the short shapes hardest:
s=4096 h=24 NVFP4+FP8 read 1555 hot vs 2046 cool (~15-30% low); mid-size shapes
(h=12/16/32 s=4096, s=32768) were ~5-20% low too. Confirmed by construction —
running s=4096 h=24 right after 3× hot s=32768 reproduces ~1618 TF vs ~1949
cool (both at 2070 MHz idle; the throttle is a transient boost-state dip during
the short kernel's do_bench window). A brief settle before each shape fixes it;
the **measured minimum is ~0.5 s** (0 s → ~86% of cool, 0.5 s → 100%, ≥1 s
plateaus), so `bench_fp4` defaults to **0.8 s** — this whole table was
re-measured 2026-07 with it on.

The h=24 sequence-length sweep figure: `flash_attn/cute/figures/gb300_tflops_h24.png`.
See the h=24 sweep figure: `flash_attn/cute/figures/gb300_tflops_h24.png`.
