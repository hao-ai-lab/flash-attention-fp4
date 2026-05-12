# FA4 block-scaled QK + FP8 PV — MIO stall analysis

## Performance table

Shape `(1, 32768, 24, 128)`, non-causal, `triton.do_bench rep=25 warmup=10`,
`CUDA_VISIBLE_DEVICES=2`. Measured 2026-04-21 on commit `51d3966c`.

| kernel | QK | PV | e2e | ms | TFLOPs | vs BF16 |
|---|---|---|---|---|---|---|
| pr2109 | BF16  | BF16 | on (freq=10) | 11.23 | 1175 | baseline |
| pr2109 | FP8   | BF16 | on (freq=10) |  8.14 | 1620 | −27% |
| pr2109 | FP8   | FP8  | on (freq=10) |  6.76 | 1951 | −40% |
| ours   | BF16  | BF16 | off |  9.44 | 1398 | baseline |
| ours   | FP8   | BF16 | off |  7.99 | 1651 | −15% |
| ours   | FP8   | FP8  | off |  7.35 | 1795 | −22% |
| ours   | NVFP4 | BF16 | off |  7.30 | 1808 | −23% |
| ours   | NVFP4 | FP8  | **auto (freq=8)** | **7.47** | **1766** | **−21%** |
| ours   | MXFP8 | BF16 | off |  7.61 | 1734 | −19% |
| ours   | MXFP8 | FP8  | **auto (freq=8)** | **7.77** | **1699** | −18% |
| ours   | MXFP8 | FP8  | manual freq=7 | **7.40** | **1784** | **−22%** |

e2e = exp2 emulation. Auto-enabled for FP8 PV at freq=8 (`51d3966c`).
Override: `FA4_E2E_FREQ=N` (frequency), `FA4_FORCE_E2E=1` (force on for BF16).

## Root cause: MMA warp's faster cycling saturates shared MIO pipe

On SM100, MUFU.EX2 (hardware exp2) dispatches through the **MIO pipe**,
which also handles shared memory loads, special math, barrier
exchanges (`SYNCS.EXCH`), and dynamic branches.

NCU PC sampling (`ncu --set full`, `--page source --csv`, `stall_mio`
column) confirms **99% of MIO stall samples land on MUFU.EX2** — but
MUFU is the *victim*, not the cause. The cause is cross-warp MIO
contention from the MMA warp.

### Why FP8 PV triggers MIO saturation but BF16 doesn't

Both cubins (BF16 PV and FP8 PV, without e2e) have **identical**
MIO-class instruction counts per iteration:

| static count | NVFP4+BF16 | NVFP4+FP8 |
|---|---|---|
| MUFU.EX2 | 259 | 259 |
| F2FP (all) | 256 | 256 |
| SYNCS.EXCH | 31 | 31 |
| SYNCS.ARRIVE | 34 | 34 |
| SYNCS.PHASECHK | 126 | 126 |
| UTCCP | 16 | 16 |
| MUFU avg gap | 4.0 inst | 4.0 inst |
| MUFU back-to-back pairs | 119 | 120 |

Same instructions, same scheduling. The difference is purely **runtime**:

**Open question**: the regression is specific to our fp4 kernel. Our
non-fp4 kernel (`flash_fwd_sm100.py`) gains cleanly from FP8 PV
(9.44→7.35 ms), as does pr2109 (8.14→6.76 ms). In a clean pipeline,
speeding up PV MMA while keeping softmax the same speed cannot slow
things down — at worst the bottleneck shifts to softmax and total stays
flat.

SASS analysis shows the BF16 and FP8 cubins have identical instruction
counts per MIO class (259 MUFU, 31 SYNCS.EXCH, 34 SYNCS.ARRIVE, same
MUFU spacing, same 128 REG, same STACK:32). The regression persists
regardless of P conversion method (explicit `packed_float_to_ue4m3` or
generic `.to()`). Yet NCU shows 3× more stall_mio samples for FP8 (44K
vs 15K), all on MUFU.EX2.

Per-instruction PC sampling reveals the mechanism — a **compiler
scheduling difference**, not a runtime contention effect:

BF16 MUFU stall profile (**bimodal**): a few MUFUs inside the burst of
32 have very high `stall_mio` (1517, 1451, 1125) and low `stall_wait`.
MUFUs outside the burst have low mio and moderate wait. Most MUFUs run
freely — total mio samples = 15K.

FP8 MUFU stall profile (**uniform**): every MUFU has `stall_mio`≈600
AND `stall_wait`≈900. No concentration. The compiler spread MUFUs more
evenly across the loop body, so every MUFU faces moderate MIO contention
instead of a few paying all of it. Total mio samples = 44K.

| metric | BF16 top MUFU | FP8 top MUFU |
|---|---|---|
| stall_mio | 1517 | 630 |
| stall_wait | 420 | 950 |
| stall_not_selected | 213 | 0 |
| total samples | 2256 | 1671 |

BF16's concentrated burst pays a high per-MUFU MIO cost but leaves most
MUFUs free (high `!sel` = they don't even compete for issue). FP8's
even spread means every MUFU faces contention — fewer per-MUFU stalls
but across ALL 259 MUFUs → higher total. This is a ptxas scheduling
artifact: different code paths (F2FP.BF16 vs F2FP.E4M3, STTM.x16 vs
STTM.x8) lead to different register allocation and instruction ordering
that happen to distribute MUFUs differently.

The e2e fix works by replacing ~40% of MUFUs with polynomial ALU,
reducing total MIO-class instructions enough that even the uniform
distribution doesn't saturate. The remaining 2.3% gap (7.47 vs 7.30 ms)
is the emulation's ALU overhead.

### NCU stall breakdown

| mode | long_sb | mio | wait | cyc/inst | ms |
|---|---|---|---|---|---|
| ours NVFP4 / BF16             | 6,860K |     394K | 2,883K |  9.53 | 11.91 |
| ours NVFP4 / FP8 (no e2e)    | 6,290K | **1,129K** | 3,517K | 10.23 | 12.60 |
| ours NVFP4 / FP8 **(e2e=8)** | 7,228K |    **40K** | 1,959K | **7.06** | **12.22** |
| ours MXFP8 / FP8 **(e2e=7)** | 6,820K |    **30K** | 1,991K | **6.89** | **12.08** |
| pr2109 FP8 / FP8             | 2,771K |     267K | 2,239K |  6.90 | 10.83 |

With e2e=7, our MXFP8+FP8 achieves `cyc/inst=6.89` — matching pr2109's 6.90.

### Why e2e fixes it

The exp2 emulation (`utils.ex2_emulation_2`) replaces some MUFU.EX2 with a
polynomial evaluation (FMAX + FADD + FMA×3). These ALU/FMA instructions do
NOT go through MIO, so they create natural breaks in the MUFU burst.

Without e2e, our softmax SASS has bursts of **17–21 consecutive MUFU.EX2**.
With e2e=8 the max burst drops to ~5, matching pr2109's pattern.

### Why e2e hurts BF16 PV

| NVFP4+BF16 | ms | total inst | long_sb | mio | cyc/inst |
|---|---|---|---|---|---|
| e2e=off | **7.30** | 3,077M | 6,858K | 395K | 9.53 |
| e2e=10 | 8.23 | **3,990M (+30%)** | **9,380K (+37%)** | 27K | 8.24 |
| e2e=16 | 9.79 | 3,707M (+20%) | **11,494K (+68%)** | 50K | 10.27 |

BF16 PV is **long_scoreboard-bound, not MIO-bound**. e2e adds ~30% more
instructions (polynomial ALU work) that increase long_sb (+37%) because the
emulated exp2 has higher latency than MUFU.EX2 (~8+ cycles polynomial chain
vs ~4 cycles MUFU). MIO drops to near-zero but that was never the
bottleneck — net effect is slower.

### e2e_freq sweep

| mode | off | freq=4 | freq=6 | freq=7 | freq=8 | freq=9 | freq=10 | freq=12 |
|---|---|---|---|---|---|---|---|---|
| NVFP4+BF16 | **7.30** | — | 13.24 | — | 13.33 | — | 8.23 | 9.38 |
| NVFP4+FP8 | 7.72 | 10.38 | 8.32 | 7.61 | **7.50** | 9.06 | 8.92 | 9.94 |
| MXFP8+BF16 | **7.61** | — | 13.35 | — | 13.54 | — | 9.00 | 9.65 |
| MXFP8+FP8 | 7.93 | — | 8.39 | **7.40** | 7.64 | 9.12 | 9.03 | 9.93 |

Sweet spot: freq=8 for NVFP4+FP8, freq=7 for MXFP8+FP8.
Auto-default is freq=8 for both (good enough; manual freq=7 gives extra
~3% for MXFP8).

## How to profile MIO stalls

```bash
bash flash_attn/cute/debug/mixed_prec_mio_stall_analysis/ncu_stall_mio.sh nvfp4_fp8
```

Or manually:

```bash
# 1. Capture full profile
ncu --profile-from-start off --target-processes all --launch-count 1 \
  --kernel-name regex:"flash_fwd" --set full -f -o /tmp/report \
  python <script.py>

# 2. Extract stall_mio by instruction
ncu -i /tmp/report.ncu-rep --page source --csv | python3 -c "
import csv, sys
from collections import defaultdict
lines = sys.stdin.readlines()
data = [l for l in lines if l.startswith('\"0x') or l.startswith('\"Address')]
reader = csv.DictReader(data)
by_op = defaultdict(int)
for r in reader:
    mio = int(r.get('stall_mio','0') or '0')
    if mio > 0:
        by_op[r.get('Source','').strip().split()[0]] += mio
for op, cnt in sorted(by_op.items(), key=lambda x: -x[1])[:10]:
    print(f'  {op:30s} {cnt:6d}')
"
```

## How to run

```bash
# Block-scaled NVFP4/MXFP8 + FP8 PV (auto e2e)
CUTE_DSL_ENABLE_TVM_FFI=1 python -m flash_attn.cute.benchmarks.bench_fp4 \
  --pv_mode fp8 --qk_mode {nvfp4|mxfp8}

# Override e2e freq
FA4_E2E_FREQ=7 CUTE_DSL_ENABLE_TVM_FFI=1 python -m flash_attn.cute.benchmarks.bench_fp4 \
  --pv_mode fp8 --qk_mode mxfp8

# Pure FP8/FP8 (no block-scale)
FA4_ALLOW_PURE_FP8_QK=1 CUTE_DSL_ENABLE_TVM_FFI=1 python <bench_script.py>
```

## Known open items

- **Pure-FP8/FP8 ~9% gap to pr2109** (1795 vs 1951 TF) — pr2109 uses
  2-CTA instructions (`UTCQMMA.2CTA` + `UTCBAR.2CTA.MULTICAST`) which our
  fp4 kernel doesn't support yet. Their 2-CTA path generates 2× more
  `SYNCS.EXCH.64` synchronization points that naturally break MUFU bursts,
  explaining their lower MIO baseline even without targeted e2e tuning.
- **MXFP8+FP8 optimal freq=7 vs auto freq=8** — 7.40 vs 7.77 ms. Could
  auto-detect based on `sf_vec_size` (32 for MXFP8 vs 16 for NVFP4).
