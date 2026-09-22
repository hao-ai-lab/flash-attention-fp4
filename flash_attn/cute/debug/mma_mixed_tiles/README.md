# Mixed tcgen05.mma tile sizes: is there anything to exploit? (GB300, sm_103a)

Question: within one kernel (e.g. prefill + decode tiles co-scheduled), can
mixing large and small `tcgen05.mma` tiles be exploited through scheduling,
load balancing or overlap? Microbenchmark `mma_mixed.cu` (single CTA, one or
two issuing warps, `%clock64` around issue + `tcgen05.commit` + mbarrier wait,
compile-time unrolled sequences so issue overhead is a few instructions;
operands in smem, values irrelevant). Build/run:

```
nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o mma_mixed mma_mixed.cu
./mma_mixed ladder            # per-instruction cost vs M, N, kind
./mma_mixed mix 0 64 128      # f16, small tile = M64 x N128; also: mix 0 128 32, mix 1 64 128 (f8)
./mma_mixed bg                # MMA stream with 4 MUFU/FMA "softmax" warps running
```

## 1. Per-instruction cost ladder (cycles, 32 back-to-back, same accumulator)

| | N=8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| kind::f16 (K16), M=64  | 56 | 56 | 56 | 56 | 78 | 140 |
| kind::f16 (K16), M=128 | 56 | 56 | 56 | 63 | 80 | 141 |
| kind::f8f6f4 (K32), M=64 / M=128 | 55 / 55 | 55 / 55 | 55 / 55 | 55 / 62 | 78 / 79 | 140 / 140 |

- A ~56-cycle floor per instruction, independent of M, N (up to ~64) and K.
- **M=64 costs the same as M=128** at every N: the tensor core behaves as if
  it always processes 128 rows. A decode-like M=64 tile wastes half the TC.
- `kind::f8f6f4` (K=32) costs the same as `kind::f16` (K=16): FP8 is 2x only
  because K per instruction doubles; the instruction itself is not faster.
- Efficiency: N=256 is 2x the MACs of N=128 for 1.75x the time, and 8x the
  MACs of N=32 for 2.5x the time. Above the floor, cost ~ 0.55 cycles/N.

## 2. Mixture ratios (32 instructions: (1-f) x large M128xN256 + f x small)

Three schedules — blocked (large then small), interleaved (small spread
evenly), and two issuing warps (warp 0 all large, warp 1 all small, own
commit/mbarrier each, wall time = later completion) — vs the additive model
`nL*cost(L) + nS*cost(S)`:

small = M64 x N128 (decode-like: few query rows, many keys), kind::f16

| f(small) | blocked | interleaved | two warps | additive model | blocked/model | inter/model | 2warp/model |
|---|---|---|---|---|---|---|---|
| 0.00 | 4477 | 4494 | 4477 | 4477 | 1.00 | 1.00 | 1.00 |
| 0.12 | 4277 | 4295 | 4261 | 4231 | 1.01 | 1.02 | 1.01 |
| 0.25 | 4121 | 4059 | 4007 | 3985 | 1.03 | 1.02 | 1.01 |
| 0.50 | 3646 | 3583 | 3506 | 3494 | 1.04 | 1.03 | 1.00 |
| 0.75 | 3157 | 3111 | 3070 | 3003 | 1.05 | 1.04 | 1.02 |
| 0.88 | 2935 | 2952 | 2805 | 2757 | 1.06 | 1.07 | 1.02 |
| 1.00 | 2512 | 2530 | 2512 | 2512 | 1.00 | 1.01 | 1.00 |

small = M128 x N32, kind::f16

| f(small) | blocked | interleaved | two warps | additive model | blocked/model | inter/model | 2warp/model |
|---|---|---|---|---|---|---|---|
| 0.12 | 4277 | 4184 | 4190 | 4143 | 1.03 | 1.01 | 1.01 |
| 0.25 | 3864 | 3864 | 3853 | 3810 | 1.01 | 1.01 | 1.01 |
| 0.50 | 3128 | 3266 | 3143 | 3143 | 1.00 | 1.04 | 1.00 |
| 0.75 | 2479 | 2582 | 2399 | 2476 | 1.00 | 1.04 | 0.97 |
| 0.88 | 2177 | 2220 | 2057 | 2142 | 1.02 | 1.04 | 0.96 |

(kind::f8f6f4 with small = M64xN128: same picture, 1.01-1.09x of the model.)

Every schedule at every ratio lands within -4%..+9% of the additive model:
**the tensor core serializes mixed tiles; there is no overlap to exploit by
ordering, by splitting the streams across issuing warps, or by choosing the
ratio.** Mixing is if anything slightly worse than additive at mid ratios
(shape switches), and two warps recover only the issue floor (<= 4%).

## 3. Overlap with non-TC work

Four warps running a MUFU.EX2 + FMA loop (softmax-like) concurrently with
the MMA stream change its time by < 0.5% (32 large: 4495 -> 4482; 32 small:
2525 -> 2532; 16+16: 3592 -> 3584). The tensor core is independent of the
SIMT pipes, so a decode softmax can hide under prefill MMAs on the same SM
as far as the TC is concerned — the shared resource is the SM's issue
bandwidth, not the tensor core.

## What this means for decode/prefill co-scheduling

- No TC-level win from co-scheduling per se; total TC time is the sum of
  the tiles' costs. The lever is tile *shape*: never issue M=64 tiles
  (same cost as M=128 — pack two decode requests' query rows into one
  M=128 tile), and amortize the 56-cycle floor with N >= 128 (keys).
- Small-N (<= 64) tiles are pure floor: an N=16 "extra" MMA costs 40% of a
  full N=256 one. (This also rules out cheap narrow side-GEMMs such as a
  row-sum column.)
- Co-scheduling a decode CTA's softmax work with a prefill CTA's MMAs is
  free on the TC side; whether it pays depends on SM issue slots (the
  block-scaled FA4 kernels are already issue-bound in the softmax warps).
- Related in-kernel evidence (fp4 branch, `debug/b300_fp4_pv_analysis.md`):
  BF16 FA4 with the PV tile shrunk to N=64 on all vs every other K-tile
  gains +18% vs +10% — additive, no mixing bonus.
- Spec note: the "N" in the PTX tcgen05 pipelining rules is `cta_group::N`,
  not the tile N; ordering across different tile shapes is only promised
  for A/metadata-read -> D-write / cp-write pairs, so none of the above is
  guaranteed by the ISA — it is GB300 behaviour.

## 4. Hardware mechanisms that could pair/pipeline tiles (`mma_mixed2.cu`)

Follow-up: instead of ordering/ratio, test the mechanisms the ISA exposes
that could let the tensor core run two tiles at once. Build/run:
`nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o mma_mixed2 mma_mixed2.cu && ./mma_mixed2 all`
(32 instructions per sequence, N=256 K16 kind::f16 unless noted, cycles).

**a. Lane pairing of M=64 tiles.** The layout table says non-.ws M=64 uses
"1/2 datapath, lane alignment 0 or 16", so two M=64 MMAs at lane 0 and lane
16 could in principle occupy both halves.

| sequence | cycles |
|---|---|
| 32 x M128 (full datapath) | 4486 |
| 32 x M64, all at lane 0 | 4426 |
| 32 x M64, alternating lane 0 / lane 16 | 4480 |
| 16 at lane 0 then 16 at lane 16 | 4479 |
| 32 x M64 alternating column halves | 4449 |
| two warps, 16 each: lane 0 \| lane 16 | 4053 |
| two warps, 16 each: lane 0 \| lane 0 | 4053 |

No pairing: M=64 at complementary lanes costs exactly M=64 at the same
lanes (= M=128). The two-warp 9% is issue hiding (identical with both
warps at lane 0).

**b. Weight-stationary `tcgen05.mma.ws`** (M in {32,64,128}, N in
{64,128,256}; M=32 is "1x4", M=64 "2x3" datapath organization):

| ws, per instruction | N=64 | N=128 | N=256 |
|---|---|---|---|
| M=32 | 57.9 | 58.7 | **89.5** |
| M=64 | 57.4 | 64.0 | **92.7** |
| M=128 | 63.2 | 85.5 | 149.3 |

This is the one real lever for small-M (decode) tiles: `.ws` M=32/64 at
N=256 costs ~90 cycles vs 140 for the non-ws M=64/M=128 tile — 1.5x
cheaper (still not proportional to M: 4x fewer rows for 1.5x less time).
`.ws` M=128 is slightly slower than non-ws (149 vs 141), and `.ws` M=32 must
sit at lane alignment 0 (lane offsets 16/32/48 fault: "misaligned address").

**c. `.ws` small tiles mixed with normal large tiles:** 16 x M128N256
(non-ws) + 16 x ws M32N256: blocked 3785 (1.03x additive), interleaved
4187 (**1.14x**) — alternating datapath organizations costs ~11%; keep
same-mode tiles grouped.

**d. Mixed kinds** (16 x f16 M128N256 + 16 x f8f6f4 M128N64): blocked 1.02x,
interleaved 0.98x of additive — nothing.

**e. Mixed A source** (large A-from-smem + small A-from-tmem, as QK vs PV in
FA4): the A-from-tmem small tile is ~10% cheaper per instruction (1833 vs
2037 for 32 x M128N64); mixed blocked 0.94x / interleaved 1.00x of
additive — at most a few % from operand-path diversity.

**f. Two co-resident CTAs per SM** (one large-tile, one small-tile,
`__launch_bounds__(128,2)`, 256 TMEM cols each, 152 SMs, kernel wall time):
1 CTA/SM large 0.449 ms, small(N32) 0.164 ms; 2 CTA/SM large+large 0.841
(1.87x), small+small 0.283 (1.73x), large+small (by blockIdx parity) 0.842.
Inconclusive for TC concurrency: CTA-to-SM placement is not controllable, so
the wall time is the worst SM (two large CTAs). The per-SM TC queue is the
same one the two-warp tests exercise, which showed no concurrency.

### Bottom line for mixed tiles

Nothing pairs or pipelines on the tensor core: every schedule, issuer
split, kind mix, operand-path mix and lane placement lands on the additive
model (best case a few % of issue hiding, worst case -11..-14% for
interleaving unlike tiles). The exploitable knobs are per-tile cost, not
overlap: use `.ws` for M<=64 tiles at large N (1.5x cheaper), keep N >= 128
to amortize the ~56-cycle floor, group tiles of one datapath mode, and let
A come from TMEM where the layout allows.

## 5. Why `.ws` helps, and whether it applies to the MLA kernel (`mma_cg2.cu`)

The datapath is 4 quadrants x 32 lanes. Non-.ws maps a tile's M rows onto
lanes: M=128 fills all 4 quadrants, M=64 fills 2 ("1/2 datapath utilized"),
so M=64 costs the same as M=128. `.ws` reorganizes the mapping (M=32 "1x4",
M=64 "2x3"): N is spread across quadrants, so all four stay busy at small M.
Cost drops to ~90 cycles at N=256 but not to 35: the B operand (N x K =
256 x 16 bf16 = 8 KB per instruction, independent of M) streams from smem at
~128 B/clk (~64 cycles) — which is why ws M=32 and M=64 cost the same and
why ws N<=128 sits on the ~58-cycle floor. (The `.collector::bN` qualifiers
exist to skip that B re-read when B is reused; attention does not reuse B
within a tile.)

The MLA kernel (`flash_fwd_mla_sm100.py`) already solves small M another
way: `cta_tile_m = 64` with `cta_group::2` — the CTA pair issues one M=128
MMA (`cluster_tile_m`) where each SM contributes 64 Q rows and holds N/2
accumulator columns (Layout B, "2x2"), so every SM runs 128 rows x N/2 at
full datapath. Measured per-instruction cost (leader clock, per SM):

| cta_group::2, kind::f16 K16 | N=64 | N=128 | N=256 |
|---|---|---|---|
| M=128 (each SM: 128 rows x N/2) | 52 | 52 | **70** |
| M=256 (each SM: 128 rows x N/2) | 52 | 70 | 134 |

For the kernel's shapes (QK N=tile_n=128 -> 52 cycles; PV N=hdimv/2=256 ->
70 cycles) that is 128x128 MAC-rows per SM per 70 cycles = 234 row-cols/
cycle, vs 177 for `.ws` M=64 N=256 (64x256 / 92.7) — **1.3x better than .ws,
and each K/V tile is loaded once per pair instead of once per SM.** So `.ws`
would be a regression for MLA whenever the 128-row cluster tile is full
(heads x q_len >= 128 per KV head: 128-head models at q_len 1, or 64 heads
with MTP q_len 2).

The case where `.ws` could win is a half-empty cluster tile — 64 valid rows
total (e.g. 64 heads per GPU at q_len 1): per-SM valid throughput drops to
117 row-cols/cycle and a 1-CTA `.ws` M=64 kernel would be 1.5x better on
the tensor core (at 2x the K/V smem traffic per SM). That only matters if
such decode is TC-bound, which it is not at long context: per 128-key tile,
K/V bytes = 128 x 576 x 2 B = 147 KB -> ~2.7 us from HBM per SM at
8 TB/s / 148 SMs vs ~2.5k cycles (~1.2 us) of QK+PV MMA time at M=64 —
memory-bound by ~2x, so the TC waste is hidden. The DSL also exposes no
`.ws` MmaOp (only MmaF16BF16Op/F8F6F4/MXF4NVF4/...), so it would need inline
PTX for the MMAs and Layout E/G-aware TMEM ld/st — not worth it for MLA.

Status note: with cutlass-dsl 4.5.2 the MLA kernel currently does not run
in this repo: its benchmark/test passes no CUDA stream (`.launch` rejects
the None with `assert isinstance(arg, ir.Value)`), and with an explicit
stream it compiles but faults on the device (illegal access) for every
shape tried (dense and topk-gather). Pre-existing on the fp4 branch since
the upstream rebase; measured numbers above are from the microbenchmarks.
