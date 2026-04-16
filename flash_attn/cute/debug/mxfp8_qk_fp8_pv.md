# FA4 FP8/MXFP8/NVFP4 QK + FP8 PV — status

Tracks: block-scaled and pure-FP8 QK + PV paths in our kernel
(`flash_fwd_sm100_fp4.py` for block-scaled; `flash_fwd_sm100.py` for pure FP8
when `FA4_ALLOW_PURE_FP8_QK=1`). Benchmarks compare against upstream FA4 PR 2109
at `hao-ai-lab/flash-attention-fp4:pr2109-mixed-dtype` (local clone at
`/tmp/pr2109-mixed-dtype`).

## Current performance

Shape `(1, 32768, 24, 128)`, non-causal, `triton.do_bench rep=25 warmup=10`,
fresh compile (cache disabled). All numbers re-measured 2026-04-16 after
the mixed-dtype fix landed.

| kernel | QK | PV | ms | TFLOPs | ms vs BF16 |
|---|---|---|---|---|---|
| pr2109 | BF16  | BF16 | 11.23 | 1175 | baseline |
| pr2109 | FP8   | BF16 |  8.14 | 1620 | −27%  |
| pr2109 | FP8   | FP8  |  6.76 | 1951 | −40%  |
| ours   | BF16  | BF16 |  8.98 | 1470 | baseline |
| ours   | FP8   | FP8  |  7.35 | 1794 | −18%  |
| ours   | NVFP4 | BF16 |  7.36 | 1792 | −18%  |
| ours   | NVFP4 | FP8  |  7.74 | 1706 | −14% (↓ from NVFP4+BF16) |
| ours   | MXFP8 | BF16 |  8.29 | 1592 |  −8%  |
| ours   | MXFP8 | FP8  |  7.92 | 1666 | −12%  |

### Key observations

- **Our BF16 baseline (8.98 ms) is stronger than pr2109's (11.23 ms)** —
  our softmax / scheduling / pipelining is tuned; theirs is closer to the
  CUTLASS-default baseline.
- **Pure FP8 path:** both kernels speed up from BF16, pr2109 by more
  relative to their weaker baseline (−40% vs our −18%). In absolute TFLOPs
  pr2109 is ~9% ahead (1951 vs 1794).
- **Block-scaled + FP8 PV regresses our kernel** (NVFP4+BF16 at 7.36 ms
  vs NVFP4+FP8 at 7.74 ms — +5% slower with FP8 V). SF plumbing cost exceeds
  PV-MMA compute savings on long-d shapes.

## FP8 PV speedup: why pr2109 benefits more than ours

**Q:** pr2109 shows a large speedup from BF16 PV → FP8 PV. Our kernel's pure
FP8 path also speeds up, but our block-scaled QK + FP8 PV combo does not.
Why?

**A:** Two separate effects.

### 1. Relative vs absolute gains

pr2109's BF16 baseline is weaker (11.23 ms) so the absolute ms saved by
switching to FP8 MMA is larger. Ours is already running BF16 at 8.98 ms, so
the headroom left for FP8 is smaller. In absolute TFLOPs both kernels land
within 9% of each other on the pure-FP8 path (pr2109 1951 vs ours 1794),
meaning neither is leaving a massive factor on the table.

Derived:

| kernel | BF16 FLOPs/s  | FP8 FLOPs/s  | speedup |
|---|---|---|---|
| pr2109 | 1175 TF | 1951 TF | 1.66× |
| ours   | 1470 TF | 1794 TF | 1.22× |

The smaller relative gain in ours is **not** from ours having a worse
pure-FP8 path — it's from ours having a better BF16 starting point.

### 2. F2FP / SF stalls crush block-scaled + FP8 PV

On ours, going from NVFP4+BF16 → NVFP4+FP8 regresses 7.36 → 7.74 ms. NCU
F2FP stall counters from prior measurements:

| mode | F2FP stalls | cyc/inst | IPC |
|---|---|---|---|
| ours NVFP4 + BF16 |  3,425 |  9.53 | 1.57 |
| ours NVFP4 + FP8  | 14,129 | 10.26 | 1.46 |
| ours MXFP8 + FP8  | 14,632 | 10.41 | 1.44 |
| ours pure FP8/FP8 |  6,633 |  8.89 | 1.69 |
| pr2109 FP8/FP8    |  5,851 |  6.89 | 2.18 |

Switching PV from BF16 to FP8 **in the block-scaled path** adds ~10K F2FP
stalls (3,425 → 14,129) because P now goes through an f32→fp8 pack AND an
SFV lookup/broadcast for each PV tile. pr2109 doesn't have the SFV path at
all — its F2FP pack is standalone (~6K stalls). That's the direct cost that
wipes out the PV-MMA compute savings on long-d shapes.

Pure FP8/FP8 on ours (6,633 stalls) is close to pr2109's pack-only cost
(5,851) — the gap there is scheduling, not instruction mix.

### 3. Why pr2109 extracts more from FP8 on long seqlens

pr2109's FP8 QK already captures most of the speedup: BF16→FP8 QK alone
takes 11.23 → 8.14 ms (−27%). Adding FP8 V compounds to 6.76 ms (additional
−17%). IPC stays at 2.18 across both — the FP8 QK pipeline is the main
win; FP8 V is a second, smaller compounding factor from the MMA throughput
doubling on pure FP8 PV kind.

Our kernel can't test the equivalent mixed (FP8 QK + BF16 PV) config yet —
that requires the same K-aliases-V SMEM surgery we applied to pr2109
(pending on our kernel).

## How to run

### Our kernel pure FP8/FP8 (no block-scale)

```bash
FA4_ALLOW_PURE_FP8_QK=1 CUTE_DSL_ENABLE_TVM_FFI=1 python -c "
import torch
from flash_attn.cute.interface import flash_attn_func
b,s,h,d = 1,32768,24,128
q = torch.randn(b,s,h,d,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
k = q.clone(); v = q.clone()
out = flash_attn_func(q,k,v,causal=False)
print(out.shape, out.dtype)
"
```

`FA4_ALLOW_PURE_FP8_QK=1` lets `interface.py` dispatch pure FP8 Q/K/V
directly to `FlashAttentionForwardSm100` (non-block-scaled). Without it,
interface.py gates FP8 Q/K through the block-scaled MXFP8 path.

### Block-scaled NVFP4 / MXFP8

```bash
CUTE_DSL_ENABLE_TVM_FFI=1 python -m flash_attn.cute.benchmarks.bench_fp4 \
  --quant_v --qk_mode {nvfp4|mxfp8} --pv_mode {bf16|fp8}
```

### pr2109 FP8 QK + BF16 PV (mixed-dtype)

```bash
cd /tmp/pr2109-mixed-dtype
CUTE_DSL_ENABLE_TVM_FFI=1 python bench_mixed.py --shape 1 32768 24 128
```

Kernel is on `hao-ai-lab/flash-attention-fp4:pr2109-mixed-dtype @ 65cbcc8a`.

## Implementation notes (all merged)

- `FlashAttentionForwardSm100.__init__` accepts NVFP4 (FP4 + E4M3 + sfv=16)
  and MXFP8 (FP8 + E8M0 + sfv=32). SF SMEM uses `self.sf_dtype`.
- PTX helpers in `blackwell_helpers.py` pick MMA kind per op: `.kind::f16`,
  pure FP8, `.kind::mxf8f6f4…`, or `.kind::mxf4nvf4…`.
- Pure-FP8 PV: softmax writes FP8 P directly via `packed_float_to_ue4m3`
  (FP32 → E4M3 fused pack, avoids the 128× `cvt.u32.u16` detour).
- Underflow handling for pure FP8 PV uses the pr2109-style
  `max_offset=8 / p_log2_offset=8` pattern with LSE correction.
- P TMEM store uses `St32x32bOp(Repetition(8))` for FP8-P; Repetition(16)
  for BF16-P (keyed on v_dtype).
- MXFP8 SFQ TMEM slot moved off `tmem_o_offset` to `tmem_s_offset` to stop
  colliding with the O accumulator (fixed 2026-04-15, commit `9456a3de`).
- MXFP8 block-scaled QK defaults to the generic `cute.gemm` helper;
  `FA4_MXFP8_USE_INLINE_PTX=1` to toggle to the per-shape inline helper
  (faster on short-seqlen, slower on long-d; generic is safer default).

## Known open items

- **Our FP8 QK + BF16 PV mixed path** — requires K-aliases-V SMEM aliasing
  pattern in `flash_fwd_sm100.py` (same fix applied to pr2109). Not yet
  ported to ours.
- **Block-scaled + FP8 PV long-d regression** — structural (SFV plumbing
  adds ~10K F2FP stalls). Would need SFV fusion into the PV descriptor or
  a dedicated MXF8-PV atom to close.
- **Pure-FP8/FP8 ~9% gap to pr2109** — scheduling/IPC gap (1.69 vs 2.18).
  Their kernel issues more instructions per cycle; ours spends ~4.5 cyc/inst
  in L1TEX scoreboard stalls vs theirs fewer. Fixable but non-trivial.

## Repo branches

- `hao-ai-lab/flash-attention-fp4:mixed_precision` — ours with
  `FA4_ALLOW_PURE_FP8_QK` gate, MXFP8 fix, FP8 PV fused exp2+pack.
- `hao-ai-lab/flash-attention-fp4:pr2109-mixed-dtype` — upstream pr2109
  kernel with FP8 QK + BF16 PV mixed-dtype enabled.
