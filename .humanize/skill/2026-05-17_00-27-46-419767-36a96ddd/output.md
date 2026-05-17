AGREE:
- Dense-only v1 is a reasonable convergence boundary. Varlen, paged KV, SplitKV, block sparsity, and 2CTA should be explicit rejects for block-scaled modes.
- Keeping BF16/FP8 dense behavior under existing paths and gating block-scaled logic with compile-time constants is the right integration direction.
- Separate K and V SMEM handling is required once QK dtype and PV dtype diverge. The upstream kernel currently assumes `q_dtype == k_dtype == v_dtype`.
- SMEM budget must count K+V for block-scaled modes if K/V no longer share the same physical buffer.
- Compile-key differentiation is mandatory. It must include QK AB dtype, SF dtype, SF vector size, V dtype, whether PV is block-scaled, V SF mode/layout, and any force/debug flags that affect generated code.
- `filter_zeros` on SF TMA partitions is a credible required fix, based on the current FP4 path.
- “No disabled S2T paths in final code” is correct. A landed upstream path cannot depend on bypassed copies or commented alternatives.

DISAGREE:
- “hdim 64 and 128” is too broad for MXFP8. MXFP8 with `sf_vec=32` has a practical minimum K tile/head dimension of 128. State NVFP4 supports 64/128; MXFP8 supports 128 unless another layout is proven.
- The performance ACs are too absolute unless the benchmark environment is pinned. TFLOPS thresholds should name GPU, clocks, CUDA/CUTLASS DSL version, shape set, causal flag, warmup/iters, and whether pre/post quantization or V-scale expansion is included.
- “Group-128 V: pre-expand before kernel” may be reasonable functionally, but it conflicts with AC-5 unless benchmark accounting is explicit. If pre-expansion is outside timing, AC-5 is a kernel-only metric, not end-to-end.
- “All logic inline” is directionally right for upstream, but risky if interpreted as copying the whole `flash_fwd_sm100_fp4.py` into the main file without factoring helpers. Inline const_expr gates are fine; large reusable layout/helper routines should move to shared helper modules.

REQUIRED_CHANGES:
- Add explicit unsupported-mode guards before compile: block-scaled + varlen, paged KV, SplitKV, block sparsity, 2CTA, incompatible GQA/pack modes, unsupported head dims, unsupported V scale layouts.
- Define the public API surface for block-scaled tensors: expected Q/K/V dtypes, SF tensor shapes, SF dtypes, layout order, alignment, and output dtype.
- Add `sf_vec_size` and V scale mode to the compile key. The current plan mentions mode discriminators but must spell these out.
- Add a BF16 regression test matrix that proves the non-block-scaled compile path and cache keys are unchanged.
- Add cache-collision tests that compile NVFP4 and MXFP8 in both orders.
- Make MXFP8 hdim64 explicitly unsupported unless a separate validated implementation exists.
- Define precision reference precisely: BF16 reference should use the same mask/causal/scale semantics and compare only output regions that are valid.

OPTIONAL_IMPROVEMENTS:
- Land in two phases: first refactor helpers with no behavior change, then add block-scaled gates. This reduces upstream review risk.
- Keep a small standalone reproducer for the S2T/TMEM legalization issue in `agent_space/` or tests if it can run quickly.
- Add benchmark logging that prints kernel-only and end-to-end timings separately for quantization/scale expansion modes.
- Add SASS/PTX inspection checks for the intended block-scaled MMA instruction variants.

UNRESOLVED:
- Whether group-128 V scale expansion is acceptable as an API contract or only as a benchmark experiment.
- Whether the upstream maintainers will accept one large `flash_fwd_sm100.py` with many const_expr branches, or prefer shared helper extraction plus minimal branch points.
- Whether AC TFLOPS targets are hard gates or aspirational tuning targets pending fixed hardware/clock definitions.
