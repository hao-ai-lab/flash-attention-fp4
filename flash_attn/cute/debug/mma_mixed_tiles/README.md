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
