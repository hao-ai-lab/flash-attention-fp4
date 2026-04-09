# FP4 QKVP precision fix (2026-04-09)

## Symptom

`flash_fwd_sm100_fp4.py` with `quant_pv=True` (Q/K and V both NVFP4, P
quantized to FP4 inside the kernel via `scale_groupwise` + `_quant_fp4`) was
producing wildly wrong results for any non-uniform softmax:

- `bench_fp4 --quant_v` on (1, 8192, 16, 128) random Q/K/V:
  `max_diff ~ 1.1–3.0`, `mean_diff ~ 0.025–0.085` — looks like "FP4 noise"
  but is actually a real bug.
- Cosine similarity vs BF16 reference for the same shape: **0.52** (essentially
  uncorrelated).
- Diagnostic: V = 1.0 const + random Q/K should give uniform output ≈ 1.0,
  but the kernel produced range `[0.054, 23.87]`, mean 1.20 — wide variation
  even though the answer is mathematically constant.
- The issue reproduces equally in bench's `cute_tensor_like` V code path and
  the FastVideo `make_ptr` V code path, which rules out V layout / SF byte
  ordering on the V side and points squarely at the P quantization plumbing.

QK-only mode (`mSFV is None`) was unaffected (cos = 0.99) — the bug only
shows up when the kernel quantizes P internally for the PV MMA.

## Root causes

Three independent bugs in the on-the-fly P scale-factor (SFP) path. None
of them affect QK-only or BF16 because they only execute when `quant_pv` is
true.

### Bug 1 — `compute_group_max` returned `max` instead of `max/6`

```python
# Old
acc_S_row_group_max[i] = self._compute_row_max(acc_S_row_frag[None, i].load())
```

Downstream `scale_groupwise` divided P by this max, so the per-group P
landed in `[0, 1]`. The FP4 quantizer (`packed_float_to_e2m1`) then only
ever produced nibbles `{0, 0.5, 1.0}` — **3 of the 8 representable FP4
values**. For a peaked softmax this collapses 16-element groups to ~1
non-zero element, throwing away most of the per-element precision.

The PTX `tcgen05.mma...mxf4nvf4.block_scale.scale_vec::4X` instruction
treats the SF as a direct E4M3 multiplier; for the dequant chain
`P_q * SF_p ≈ P` to be tight at FP4 precision, P_q must use the full
`[0, 6]` range. That requires `P *= 6 / max` and `SF_p = max / 6`.

```python
# Fix
acc_S_row_group_max[i] = self._compute_row_max(acc_S_row_frag[None, i].load()) * (1.0 / 6.0)
```

### Bug 2 — SFP R2S byte placement was a linear `thread_idx << 2` mapping

```python
# Old
base_offset = thread_idx << 2
sfp_thread_layout = cute.make_layout((4, 2), stride=(1, 512))
sSFP_thread = cute.make_tensor(sSFP_stage_ptr + base_offset, sfp_thread_layout)
```

This says "thread N writes its 8 SF values to bytes `4N..4N+3` and
`4N+512..4N+515`" — i.e. consecutive threads occupy consecutive 4-byte
slots.

But `sSFP` is laid out by
`tile_to_shape(BlockScaledBasicChunk(16).layout, (M, K), (2, 1))`, which
encodes:

```
byte_offset(m, k_block) = (m % 32) * 16
                       + ((m // 32) % 4) * 4
                       + (k_block % 4) * 1
                       + (k_block // 4) * 512
```

Each softmax thread holds **one** M row of the t2r tile (lane within the
warp ↔ row in `[0, 32)`, warp within the softmax warpgroup ↔ row block in
`[0, 4)`), so the correct base offset is

```python
# Fix
lane_id = thread_idx % 32
warp_id = thread_idx // 32
base_offset = lane_id * 16 + (warp_id % 4) * 4
```

Verification: with both bugs in place, thread 0 happens to land at byte 0
either way, so V = constant + Q = K = constant gave the right answer.
For any other thread the SFP it wrote was being read by the MMA as the SF
for a different M row, and the dequantization was scrambled.

### Bug 3 (consumer-side) — `make_ptr` V code path expected the wrong byte packing

`flashinfer.nvfp4_quantize` packs 2 K-adjacent FP4 per byte. To get a true
K-major V (S-major in our notation, since K = seqlen for the PV MMA), feed
`v.permute(0, 2, 3, 1).reshape(b * h * d, s)` so each row's K = S — the
output is `(b*h*d, s/2)` int8 with adjacent S values in adjacent nibbles.

The previous interface decoded V as `(b, s, h, d/2)` (FastVideo's
`permute(0,3,2,1).contiguous().permute(0,3,2,1)` recipe), which packs 2
**D**-adjacent FP4 per byte — that's not K-major in the FP4-element sense
and the kernel reads V positions incorrectly. Updated `interface.py` and
`flash_fwd_sm100_fp4.py`'s `make_ordered_layout` to expect `(b, h, d, s/2)`
torch shape (S-major nibble packing), with element strides
`(h*d*s, 1, h*d, d)` — `order=(3, 0, 2, 1)`. FastVideo's
`_nvfp4_quantize_v_for_fa4` was updated in lockstep.

## Why these bugs were silent for so long

| Test | Constant SF? | Per-iteration drift? | Result |
|---|---|---|---|
| Q = K = V = const | yes | no | accidental pass (every position reads the same byte) |
| `bench --quant_v` w/ constant SF=1.0 | yes | yes | mean_diff is "small enough" to look like FP4 noise; max_diff is 1–3 |
| QK-only (no `compute_group_max`) | n/a | n/a | unchanged (the buggy path isn't called) |

Only V = const + **non-uniform** Q/K reveals the bug — and even then it
shows up as "wide output range" not as a NaN or a kernel hang, so a quick
benchmark check easily confirms the wrong behavior as "looks roughly
right".

## Empirical fix verification (1, 8192, 16, 128)

| Metric | Pre-fix | Post-fix |
|---|---|---|
| Random QKVP cos vs BF16 | 0.52 | **0.98** |
| `norm(out) / norm(ref)` | 1.85 | 1.006 |
| V = 1.0 + random Q/K out range | `[0.054, 23.87]`, mean 1.20 | `[1.00, 1.05]`, mean 1.02 |
| `bench_fp4 --quant_v` max_diff | 1.1 – 3.0 | **0.025 – 0.24** |
| `bench_fp4 --quant_v` mean_diff | 0.025 – 0.085 | 0.002 – 0.023 |
| `bench_fp4` (QK-only) max_diff | 0.02 – 0.27 | 0.02 – 0.27 (≤1e-3 perturbation, no consistent direction) |
| QK-only Test 3 cos | 0.9905 | 0.9905 (unchanged) |

## What about `rescale_threshold`?

The first commit in this fix series flipped `rescale_threshold` from `8.0`
to `0.0` and observed cos 0.52 → 0.58. After applying the real fixes
(compute_group_max + SFP R2S byte placement), reverting `rescale_threshold`
back to `8.0` does **not** change the QKVP accuracy — cos stays at 0.9817 ±
1e-4, max_diff and mean_diff stay within FP rounding noise. So the
rescale_threshold "fix" was a red herring; we kept it at 8.0 because it
matters for performance (skipping the rescale when `acc_scale_ ≥ -8` is the
common path for BF16-precision row_max tracking).

## Files touched

- `flash_attn/cute/softmax.py` — `compute_group_max` returns `max/6`
- `flash_attn/cute/flash_fwd_sm100_fp4.py` — SFP R2S byte placement, V
  `make_ordered_layout` order=(3, 0, 2, 1), `make_ptr` V shape decode
- `flash_attn/cute/interface.py` — `is_fp4_v` shape decode `(b, h, d, s/2)`
- `/sgl-workspace/FastVideo/fastvideo/attention/backends/flash_attn.py` —
  `_nvfp4_quantize_v_for_fa4` rewritten to feed nvfp4_quantize the
  `(b*h*d, s)` shape and emit the matching SF tensor in
  `(32, 4, rest_m, 4, rest_k, h, b)` form
