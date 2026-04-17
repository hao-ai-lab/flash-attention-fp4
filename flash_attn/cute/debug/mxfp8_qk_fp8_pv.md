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

### 2. NCU-confirmed root cause: MIO throttle from SFP + SFV plumbing

Per-warp avg stall contribution (`inst/warp`, single-shot NCU on
`(1, 32768, 24, 128)`). Summing across these columns ≈ cyc/inst.

| mode | long_sb | mio_throttle | math_pipe | wait | barrier | cyc/inst | ms | SM% |
|---|---|---|---|---|---|---|---|---|
| ours BF16 / BF16    | 8,508K |   264K |  47K | 2,878K |  0.1K | 10.64 | 13.54 | 78.0% |
| ours NVFP4 / BF16   | 6,860K |   394K |  82K | 2,883K |  0.2K |  9.53 | 11.91 | 81.5% |
| ours NVFP4 / FP8    | 6,290K | **1,129K** | 151K | 3,517K |  0.2K | 10.23 | 12.60 | 76.8% |
| ours MXFP8 / BF16   | 7,341K |   335K |  54K | 2,879K |  0.1K |  9.80 | — | — |
| ours MXFP8 / FP8    | 6,758K |   890K | 167K | 3,519K |  0.1K | 10.41 | — | — |
| ours pure FP8 / FP8 | 6,339K |   510K | 118K | 2,880K |  0.2K |  8.89 | 11.98 | 80.9% |
| pr2109 BF16 / BF16  | 5,765K |   168K | 166K | 2,997K | 4,495K |  9.99 | 15.87 | 65.2% |
| pr2109 FP8 / BF16   | 3,739K |   128K | 275K | 2,238K | 3,566K |  6.83 | 12.51 | 64.5% |
| pr2109 FP8 / FP8    | 2,771K |   267K | 253K | 2,239K | 3,025K |  6.90 | 10.83 | 73.3% |

#### Diffs for the switch-to-FP8-PV step

| transition | Δ long_sb | Δ mio_throttle | Δ wait | Δ cyc/inst |
|---|---|---|---|---|
| pr2109 FP8 BF16 → FP8 FP8          | **−968K** |  +139K |      0  | **+0.07** |
| ours NVFP4 BF16 → NVFP4 FP8        |  −570K   | **+735K** | +634K |  +0.70 |
| ours MXFP8 BF16 → MXFP8 FP8        |  −583K   |  +555K   | +640K |  +0.61 |
| ours BF16 BF16 → pure FP8 FP8      | **−2,169K** |  +246K |     0  | **−1.75** |

**Interpretation**:
- **pr2109's FP8 PV win is bandwidth-driven.** V's per-stage byte count halves
  (BF16 → FP8), which drops the long-scoreboard wait by ~1M cyc/inst (consumer
  warps stop stalling on V arrival). mio_throttle barely moves (+139K). Net
  cyc/inst is flat, but SM throughput jumps 64.5% → 73.3% because the unused
  cycles now get spent on productive instructions.
- **Ours pure FP8 / FP8 also wins** for the same bandwidth reason: long_sb
  drops ~2M cyc/inst, mio_throttle rises only modestly (+246K). Net cyc/inst
  falls 1.75 — FP8 PV is unambiguously good here.
- **Ours block-scaled + FP8 PV regresses** because mio_throttle spikes by
  **+735K (NVFP4)** or **+555K (MXFP8)** — 4–5× the rise pr2109 sees. That
  wipes out the bandwidth gain (−570K long_sb) and leaves cyc/inst +0.7
  worse. This is a direct symptom of SFP + SFV plumbing: softmax now has to
  compute and store SFP alongside FP8-packed P, and each PV MMA needs SFV
  loaded from smem → per-thread registers → broadcast among threads. Those
  operations run on the MIO pipe.

#### Why our pure-FP8/FP8 path DOES benefit

With `FA4_ALLOW_PURE_FP8_QK=1`, Q/K/V go through `flash_fwd_sm100.py`
(non-block-scaled) — no SFQ/SFK/SFP/SFV anywhere. That makes the FP8 PV
transition look exactly like pr2109's: a pure bandwidth/MMA-throughput win
with no MIO tax.

#### Why pr2109's MIO stays low

pr2109 has zero scale-factor plumbing. The FP8 PV step is just:
(a) halve V smem bytes → long_sb drops, (b) use `kind::f8f6f4` PV MMA
instead of `kind::f16` → 2× compute throughput. Nothing touches MIO.

#### Fixing the block-scaled + FP8 PV regression (speculative)

The +735K mio_throttle on ours is probably dominated by:
1. SFV load from smem to register per PV-MMA iteration (tcgen05 needs SFV in
   a specific register layout; `find_tmem_tensor_col_offset` returns cosize
   so we iterate addresses manually).
2. SFP compute + tmem-store in the softmax warp (fused exp2+pack already
   helps but the SF generation is additional).

Plausible mitigations:
- Keep SFV resident in registers across multiple PV iterations (amortize
  the load).
- Fuse the SFP+P pack into a single tmem-store (the `_apply_exp2_pack_fp8`
  helper does this for P; SF side is still separate).
- Skip block-scaling on PV entirely when V is FP8: use pure `MmaFP8Op` for
  PV even when QK is block-scaled. That removes SFV/SFP but needs
  descale handling for V in the correction step.

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
