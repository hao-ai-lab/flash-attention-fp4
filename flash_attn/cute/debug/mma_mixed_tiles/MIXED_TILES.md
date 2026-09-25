# Mixed tcgen05.mma tile sizes on GB300: overlap, scheduling, and per-tile cost

Consolidated findings (2026-09-17 .. 09-21, GB300 / sm_103a, CUDA 13, cutlass-dsl 4.5.2)
on whether tensor-core MMAs of different tile shapes can overlap or be scheduled to
beat the sum of their costs — motivated by decode/prefill co-scheduling and by FA4's
mix of QK and PV GEMMs. Short answer: **no tensor-core overlap exists to exploit; the
levers are per-tile cost (`.ws` for small M, `cta_group::2`, N >= 128) and tile
grouping.**

All microbenchmarks are single-file CUDA/PTX programs in this directory:

| file | what it measures |
|---|---|
| `mma_overlap.cu`  | one issuing thread: large (M128 N256) vs small (N8..128) tiles, blocked / interleaved / same accumulator |
| `mma_overlap2.cu` | two issuing warps, one stream each, own commit + mbarrier |
| `mma_mixed.cu`    | cost ladder over M, N, kind; mixture ratios x schedules; background SIMT warps |
| `mma_mixed2.cu`   | hardware mechanisms: lane pairing, `.ws`, mixed kinds, mixed A source, 2 CTAs/SM |
| `mma_cg2.cu`      | `cta_group::2` (2-CTA) per-SM cost |

Build any of them with `nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o X X.cu`.

## Method

- One CTA (cluster of 2 for `mma_cg2.cu`), operands in shared memory in the
  SWIZZLE_NONE K-major canonical layout (8-row x 16 B core matrices, LBO 128 B,
  SBO 256 B); operand values are irrelevant for timing.
- Timing: `%clock64` from the first `tcgen05.mma` issue to completion of a
  `tcgen05.commit` observed through an mbarrier wait; 40-50 repetitions, first
  discarded. Sequences are compile-time unrolled with descriptors in registers, so
  issue overhead is a few instructions per MMA (an early version with runtime
  branches per instruction inflated every cost to ~155 cycles — keep sequences
  unrolled when reproducing).
- "Additive model" = `nL * cost(large) + nS * cost(small)` from the pure-stream runs.

## 1. What the ISA promises (and a wording trap)

PTX 9.7.16.6.2 / 9.7.18.6.2 ("Pipelined tcgen05 instructions"): asynchronous tcgen05
operations *may execute and complete out of issue order*; only listed pairs from the
same warp are ordered, e.g. `mma (A/metadata reads) -> mma (D writes)`,
`mma (A/metadata reads) -> cp (writes)`, `cp -> mma (A/metadata reads)`, and
`mma (D) -> mma (C/D)` only for the same accumulator address **and shape**.
`tcgen05.commit` tracks all prior async tcgen05 ops of the issuing thread.

The "N" in "(same N)" / "(same N and accumulator and shape)" is the `cta_group::N`
qualifier (1 or 2), **not** the tile N. It appears on pairs with no tile N at all
(`cp -> mma (same N)`), and "shape" is listed separately. Tile shape only enters
through the accumulator rules. The repo's local `debug/full-ptx-doc.md` is ISA 9.1,
which lists fewer pairs (no `mma -> cp`); newer revisions spell out operand-level
ordering.

So mixed-shape MMAs into different accumulators are exactly the case the ISA leaves
unordered — the hardware would be *allowed* to overlap them. Everything below
measures whether it does. None of the in-order behaviour observed is guaranteed.

## 2. Per-instruction cost ladder

32 back-to-back MMAs, same accumulator (`mma_mixed.cu ladder`), cycles per instruction:

| | N=8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| kind::f16 (K16), M=64  | 56 | 56 | 56 | 56 | 78 | 140 |
| kind::f16 (K16), M=128 | 56 | 56 | 56 | 63 | 80 | 141 |
| kind::f8f6f4 (K32), M=64 / 128 | 55 / 55 | 55 / 55 | 55 / 55 | 55 / 62 | 78 / 79 | 140 / 140 |

- ~56-cycle floor per instruction up to N~64, independent of M and K; above it
  ~0.55 cycles per N column. M128 N256 bf16 ~ 3900 MAC/clk: tensor-core bound.
- **M=64 costs exactly as much as M=128** (ISA layout table: non-.ws M=64 uses
  "4x1, 1/2 datapath utilized").
- `kind::f8f6f4` K=32 costs the same as `kind::f16` K=16: FP8's 2x comes purely
  from doing twice the K per instruction.
- The floor is mostly tensor-core side: with issue hidden behind another warp
  (section 3) a small N=32 tile still adds ~42 cycles.

## 3. Ordering, ratio, and issuer: no overlap

**One issuing thread** (`mma_overlap.cu`), 32 large (M128 N256) + 32 small:

| small tile | blocked / (A+B) | interleaved / blocked | small into same accumulator |
|---|---|---|---|
| N=8   | 0.94 | 1.01 | +-0.3% |
| N=32  | 0.94 | 1.02 | +-0.3% |
| N=64  | 0.97 | 0.97 | +-0.3% |
| N=128 | 0.96 | 1.00 | +-0.3% |

(the ~5% below A+B is one fewer commit/wait tail, not overlap).

**Two issuing warps** (`mma_overlap2.cu`, warp 0: 32 x N256, warp 1: 32 x small,
separate TMEM columns, own commit/mbarrier, wall = later completion):
concurrent = **0.93-0.96x of the sum**, far from max(L, S) (~0.73x of the sum).
Two warps both issuing 32 x N256: 0.98x of 2x alone.

**Mixture ratios** (`mma_mixed.cu mix`), 32 instructions, fraction f of small tiles,
three schedules vs the additive model:

small = M64 x N128 (decode-like), kind::f16

| f(small) | blocked | interleaved | two warps | model | blocked/model | inter/model | 2warp/model |
|---|---|---|---|---|---|---|---|
| 0.00 | 4477 | 4494 | 4477 | 4477 | 1.00 | 1.00 | 1.00 |
| 0.12 | 4277 | 4295 | 4261 | 4231 | 1.01 | 1.02 | 1.01 |
| 0.25 | 4121 | 4059 | 4007 | 3985 | 1.03 | 1.02 | 1.01 |
| 0.50 | 3646 | 3583 | 3506 | 3494 | 1.04 | 1.03 | 1.00 |
| 0.75 | 3157 | 3111 | 3070 | 3003 | 1.05 | 1.04 | 1.02 |
| 0.88 | 2935 | 2952 | 2805 | 2757 | 1.06 | 1.07 | 1.02 |
| 1.00 | 2512 | 2530 | 2512 | 2512 | 1.00 | 1.01 | 1.00 |

small = M128 x N32, kind::f16

| f(small) | blocked | interleaved | two warps | model | blocked/model | inter/model | 2warp/model |
|---|---|---|---|---|---|---|---|
| 0.12 | 4277 | 4184 | 4190 | 4143 | 1.03 | 1.01 | 1.01 |
| 0.25 | 3864 | 3864 | 3853 | 3810 | 1.01 | 1.01 | 1.01 |
| 0.50 | 3128 | 3266 | 3143 | 3143 | 1.00 | 1.04 | 1.00 |
| 0.75 | 2479 | 2582 | 2399 | 2476 | 1.00 | 1.04 | 0.97 |
| 0.88 | 2177 | 2220 | 2057 | 2142 | 1.02 | 1.04 | 0.96 |

kind::f8f6f4 with small = M64 x N128: same picture, 1.01-1.09x of the model.

Every schedule at every ratio lands within -4%..+9% of the additive model. Mid-ratio
mixing is slightly *worse* than additive (shape switches); a second issuing warp
recovers only the issue floor (<= 4%).

## 4. In a real kernel: BF16 FA4 with shrunk PV tiles

`FA4_DEBUG_MMA_N` experiment (debug knob in `gemm_ptx_partial`, not committed;
outputs wrong, timing only), GB300, b=1 s=32768 h=24 d=128, TFLOPS:

| variant | TFLOPS | vs baseline |
|---|---|---|
| baseline (PV N=128, QK N=128) | 1629 | — |
| PV N=64 on every K-tile | 1925 | +18% |
| PV N=64 on every other K-tile | 1799 | +10% |
| PV N=32 on every K-tile | 2102 | +29% |
| QK N=64 on every K-tile | 1605 | 0% |
| QK N=64 on every other K-tile | 1625 | 0% |

Additive again (half the tiles shrunk = half the gain), no mixing bonus; cost is not
proportional to N (matches the ladder). BF16 FA4 is bound by the PV GEMM's
tensor-core time; QK's is hidden under softmax.

## 5. Hardware mechanisms that could pair or pipeline tiles

`mma_mixed2.cu`, 32 instructions, N=256 K16 kind::f16 unless noted, cycles:

**a. Lane pairing of M=64 tiles** (non-.ws M=64 allows lane alignment 0 or 16):

| sequence | cycles |
|---|---|
| 32 x M128 (full datapath) | 4486 |
| 32 x M64, all at lane 0 | 4426 |
| 32 x M64, alternating lane 0 / lane 16 | 4480 |
| 16 at lane 0 then 16 at lane 16 | 4479 |
| 32 x M64, alternating column halves | 4449 |
| two warps, 16 each: lane 0 \| lane 16 | 4053 |
| two warps, 16 each: lane 0 \| lane 0 | 4053 |

No pairing; the two-warp 9% is issue hiding (identical with both at lane 0).

**b. Weight-stationary `tcgen05.mma.ws`**, cycles per instruction:

| .ws | N=64 | N=128 | N=256 |
|---|---|---|---|
| M=32 | 57.9 | 58.7 | **89.5** |
| M=64 | 57.4 | 64.0 | **92.7** |
| M=128 | 63.2 | 85.5 | 149.3 |

The one real per-tile lever for small M: `.ws` M=32/64 at N=256 is ~1.5x cheaper
than non-ws M=64/128 (90 vs 140). `.ws` M=128 is slightly slower than non-ws;
`.ws` M=32 requires lane alignment 0 (offsets 16/32/48 fault with
"misaligned address").

**c. `.ws` small tiles mixed with normal large tiles** (16 x M128N256 + 16 x ws
M32N256): blocked 3785 (1.03x additive), interleaved 4187 (**1.14x**) — switching
datapath organizations costs ~11%; keep same-mode tiles grouped.

**d. Mixed kinds** (16 x f16 M128N256 + 16 x f8f6f4 M128N64): blocked 1.02x,
interleaved 0.98x of additive.

**e. Mixed A source** (large A-from-smem + small A-from-TMEM, like QK vs PV in FA4):
A-from-TMEM is ~10% cheaper per small instruction (1833 vs 2037 for 32 x
M128N64); mixed 0.94x blocked / 1.00x interleaved of additive.

**f. Two co-resident CTAs per SM** (`__launch_bounds__(128,2)`, 256 TMEM columns
each, 152 SMs, 200 x 32 MMAs per CTA): 1 CTA/SM large 0.449 ms, small (N32)
0.164 ms; 2 CTAs/SM large+large 0.841 (1.87x), small+small 0.283 (1.73x),
large+small 0.842. Inconclusive — CTA-to-SM placement is not controllable, so wall
time is set by SMs that received two large CTAs; the per-SM TC queue is what the
two-warp test already exercises.

**g. Non-TC work alongside** (4 warps running MUFU.EX2 + FMA while the MMA stream
runs): MMA time changes < 0.5%. The tensor core is independent of the SIMT pipes;
the contended resource in practice is SM issue bandwidth, not the TC.

## 6. Why `.ws` helps, and `cta_group::2` as the alternative

The datapath is 4 quadrants x 32 lanes. Non-.ws maps a tile's M rows onto lanes:
M=128 fills all four quadrants, M=64 two. `.ws` reorganizes the mapping (M=32
"1x4", M=64 "2x3"): N is spread across quadrants, so all four stay busy at small M.
It bottoms out at ~90 cycles (not ~35) because the B operand — N x K = 256 x 16 bf16
= 8 KB per instruction, independent of M — streams from shared memory at ~128 B/clk
(~64 cycles). That is why `.ws` M=32 and M=64 cost the same and why `.ws` N <= 128
sits on the ~58-cycle floor. (`.collector::bN::{fill,use,lastuse}` exists to skip
the B re-read when B is reused; attention does not reuse B within a tile.)

`cta_group::2` gives small-M CTAs a full datapath another way: a 2-CTA pair issues
one M=128 (or 256) MMA, each SM contributes half the rows and holds N/2 accumulator
columns (Layout B "2x2" for M=128). Per-SM cost (`mma_cg2.cu`, leader clock):

| cta_group::2, kind::f16 K16 | N=64 | N=128 | N=256 |
|---|---|---|---|
| M=128 (each SM: 64 rows in, 128 x N/2 out) | 52 | 52 | **70** |
| M=256 (each SM: 128 rows in, 128 x N/2 out) | 52 | 70 | 134 |

M=128 N=256 via 2-CTA = 128 x 128 MAC-rows per SM per 70 cycles (234/cycle) vs
`.ws` M=64 N=256 at 177/cycle — 1.3x better, and each B (K/V) tile is loaded once
per pair.

## 7. Application: FA4 MLA (M=64 per CTA)

`flash_fwd_mla_sm100.py` already uses `cta_tile_m = 64` with `cta_group::2`
(`cluster_tile_m = 128`), i.e. the design in section 6. Its tiles: QK N = tile_n =
128 (52 cycles/instr), PV N = hdimv/2 = 256 (70 cycles/instr).

- Full cluster tile (heads x q_len >= 128 per KV head: 128-head models at q_len 1,
  or 64 heads with MTP q_len 2): `.ws` would be a 1.3x regression on the TC.
- Half-empty cluster tile (64 valid rows, e.g. 64 heads per GPU at q_len 1):
  per-SM valid throughput halves (117 row-cols/cycle) and a 1-CTA `.ws` M=64 design
  would be 1.5x better on the TC — but long-context decode is HBM-bound there:
  one 128-key K/V tile is 128 x 576 x 2 B = 147 KB (~2.7 us per SM at 8 TB/s over
  148 SMs) vs ~1.2 us of QK+PV MMA time, so the TC waste is hidden.
- The DSL exposes no `.ws` MmaOp (only MmaF16BF16Op / MmaF8F6F4Op / MmaMXF4NVF4Op
  / ...); using it would need inline-PTX MMAs plus Layout E/G-aware TMEM ld/st.

Verdict: not worth it for MLA.

Status: with cutlass-dsl 4.5.2 the MLA kernel does not currently run in this repo —
its own test/benchmark pass no CUDA stream (`.launch` asserts
`isinstance(arg, ir.Value)` on the `None`), and with an explicit stream it compiles
but faults on the device (illegal access) for every shape tried (dense and
top-k gather, 64/128 heads, q_len 1..4096). Pre-existing since the upstream rebase;
MLA numbers here are derived from the microbenchmarks, not the kernel.

## 8. Can `mma.sync` harvest the idle half?

Section 2 shows a non-ws `M=64` tile costs the same as `M=128`, so half the tensor
core looks idle; section 3 shows nothing on the `tcgen05` path can use it. The
remaining idea: issue warp-level `mma.sync` (the legacy HMMA path, register
operands) from warps that are already resident — e.g. softmax warps with spare
registers — when the incoming tile is small. That only pays if `mma.sync` reaches a
*different* datapath.

`mma_sync_overlap.cu`: warp 0 drives 32 x `tcgen05.mma`, warps 4..4+W-1 run
register-resident `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` chains with
NACC independent accumulators (no memory traffic in the loop). Timed alone and
together (started from the same `__syncthreads`). Independent paths would give
wall ~= max(alone); a shared tensor core gives wall ~= sum(alone).

| tcgen05 tile | tcgen05 alone | mma.sync alone (W=4, NACC=4) | both wall | vs max | vs sum |
|---|---|---|---|---|---|
| M128 N256 | 4250 cy (3948 MAC/cy) | 2306 cy (909 MAC/cy) | 6632 cy | 1.56x | **1.01x** |
| M64 N128 | 2186 cy (1919 MAC/cy) | 2306 cy (909 MAC/cy) | 4411 cy | 1.91x | **0.98x** |

wall/sum stays at 0.98-1.01x across every configuration tried (W = 1, 2, 4, 8;
NACC = 4, 8; both tcgen05 tiles): **`mma.sync` and `tcgen05.mma` fully serialize —
they are the same tensor core.** `mma.sync` alone is latency-bound, not
TC-saturated (its time is flat at 2306 cy while throughput scales 227 -> 455 -> 909
-> 1819 MAC/cy for W = 1..8), yet it never overlaps a single cycle of the
`tcgen05` stream.

Worse, `mma.sync` is a less efficient way to reach that tensor core. Best measured
peak (W=8, NACC=8): **1927 MAC/cy vs 3948 MAC/cy** for `tcgen05.mma` M128 N256 —
about half. So every MAC moved to `mma.sync` costs ~2x more tensor-core time than
issuing it as part of a `tcgen05` tile.

The M=64 case makes the verdict concrete: the "idle half" is not recoverable.
Adding the `mma.sync` stream to the M64 N128 tcgen05 stream yields an aggregate
1426 MAC/cy, *below* the 1919 MAC/cy of that tcgen05 stream on its own (0.74x) —
the extra work costs more TC time than it contributes.

Verdict: no. Spare registers in the softmax warps are not the scarce resource; the
tensor core is, and `mma.sync` contends for it at half efficiency (while also
consuming softmax-warp issue slots, which are the bound in the block-scaled FA4
kernels — see `b300_fp4_pv_analysis.md`). The fix for a small M stays the one in
sections 6-7: pack rows to M=128, use `cta_group::2` with 64 rows per CTA, or `.ws`.

## 9. Takeaways

1. The tensor core serializes MMAs of different shapes: by order, by ratio, by
   issuing warp, by kind, by A-source and by lane placement, total time is additive
   (best case a few % of issue hiding; worst case -11..-14% for interleaving tiles
   of different datapath modes).
2. Per-tile cost is what can be optimized:
   - never issue non-ws M=64 (same cost as M=128) — pack rows to M=128, use
     `cta_group::2` with 64 rows per CTA, or `.ws` for M <= 64 at large N;
   - keep N >= 128 to amortize the ~56-cycle floor (N <= 64 tiles are pure floor);
   - group tiles by datapath mode (ws vs non-ws);
   - prefer A from TMEM where the layout allows (~10% cheaper small tiles).
3. `mma.sync` shares the same tensor core (wall = 1.01x sum, never max) and reaches
   it at ~half the throughput (1927 vs 3948 MAC/cy), so it cannot harvest the idle
   half of an M=64 tile — it makes that case 0.74x worse.
4. For decode/prefill co-scheduling, TC time is simply the sum of the tiles' costs;
   co-scheduling can only help by filling otherwise idle SIMT/issue capacity (e.g.
   decode softmax under prefill MMAs), not by overlapping tensor-core work.
5. None of the in-order behaviour is guaranteed by the ISA; correctness must rest
   on the pipelined pairs and `tcgen05.commit`.
