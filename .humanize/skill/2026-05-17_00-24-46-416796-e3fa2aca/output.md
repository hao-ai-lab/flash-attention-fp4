I inspected the upstream tree and the PR branch. Main conclusion: the S2T/TMEM scale-factor path is the gating risk; the rest is mostly scope control, API keying, and acceptance discipline.

**CORE_RISKS**

- The S2T legalization failure is a hard blocker, not an implementation detail. The standalone FP4 path depends on `mainloop_s2t_copy_and_partition` before QK/PV MMAs, while the PR integration currently disables it around [/sgl-workspace/flash-attention-pr/flash_attn/cute/flash_fwd_sm100.py:1435](/sgl-workspace/flash-attention-pr/flash_attn/cute/flash_fwd_sm100.py:1435).
- `const_expr` inline integration can still regress BF16 through shared register allocation, SMEM sizing, pipeline depth, TMA byte counts, and compile-key branching even if the BF16 arithmetic path is unchanged.
- JIT cache keying must distinguish QK operand dtype, SF dtype, V dtype, `mSFV` presence, and any group-size mode. The current workspace already notes the prior failure mode: NVFP4 and MXFP8 can silently reuse the wrong kernel if dtype metadata is omitted at [interface.py:598](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/interface.py:598).
- The 2018 TFLOPS target may be shape-specific. S2T copies, extra barriers, TMEM placement, and register pressure can move the bottleneck away from MMA issue rate.
- NVFP4+FP8 group-128 V scales are architecturally suspicious: existing block-scaled MMA paths appear built around `sf_vec_size` 16 or 32, not 128. Group-128 likely needs scale expansion, custom layout work, or a different algorithm.
- Torch dtype representation is fragile for FP4/SF tensors because several paths infer logical dtype from byte-backed tensors.

**MISSING_REQUIREMENTS**

- Explicit support matrix for: causal, local/windowed, varlen, paged KV, SplitKV, block sparsity, `pack_gqa`, `score_mod`, `mask_mod`, `learnable_sink`, SM100/SM103/SM110/SM120, `head_dim != head_dim_v`, and hdim 64/96/128/192/256.
- Public shape/layout contract for `mSFQ`, `mSFK`, `mSFV`: logical shape, physical byte layout, alignment, strides, varlen/page layout, and E8M0/E4M3 encoding.
- Definition of output dtype for each mixed mode and whether backward is explicitly unsupported.
- Required behavior for user-provided `out`/`lse`, non-contiguous inputs, fake tensor compile, disk cache, and env-driven debug flags.
- Precision reference definition: compare against BF16 attention, dequantized FP32 reference, or a quantization-aware reference.

**TECHNICAL_GAPS**

- `make_tmem_layout_sfa/sfb` layout compatibility with `tcgen05.Cp4x32x128bOp` needs an isolated reproducer before further integration.
- The PR branch integrates block-scaled QK, but the standalone FP4 file also has quantized PV machinery: SFP generation from softmax, SFP/SFV S2T, and block-scaled PV MMA around [flash_fwd_sm100_fp4.py:2182](/sgl-workspace/cutlass/examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/flash_fwd_sm100_fp4.py:2182).
- Group-128 V scaling does not map cleanly to the existing `sf_vec_size in {16, 32}` validation.
- SMEM sizing in the PR branch uses rough SF overhead; barrier transaction counts and staged layout cosizes need exact accounting.
- MXFP8 generic `cute.gemm` vs inline PTX is not settled; the helper comments already call out per-K SF addressing risk.

**ALTERNATIVE_DIRECTIONS**

- Keep `flash_fwd_sm100_fp4.py` separate and dispatch from `interface.py` first. Lowest BF16 regression risk, but preserves duplication.
- Land QK block-scaled modes first with BF16/regular FP8 PV; defer block-scaled PV and group-128 V. Faster path to correctness, misses peak target.
- Use generic block-scaled `cute.gemm` as the correctness baseline, then recover perf with inline PTX only where measured.
- Pre-expand group-128 V scales into hardware-supported per-16/per-32 scale layout before the kernel. More memory/preprocess cost, much less kernel risk.
- Produce a minimal CUTLASS/CuTe S2T reproducer and fix the layout helper upstream before integrating into FA4.

**QUESTIONS_FOR_USER**

- Is group-128 V scale required in the same fused kernel, or is preprocessing/scale expansion acceptable?
- Is a separate FP4 kernel file acceptable for v1, or is inline integration mandatory before merge?
- Which features are in scope for v1: dense only, or also varlen, paged KV, SplitKV, block sparsity, local masks, GQA packing?
- What exact benchmark shapes define “zero TFLOPS regression” and “peak 2018 TF”?
- What precision thresholds should gate each mode?
- Which architectures/toolkit versions are required: SM100 only, or SM103/SM110/SM120 too?

**CANDIDATE_CRITERIA**

- Existing BF16/FP16/FP8 SM100 tests pass with block-scaled code compiled out; perf delta within agreed noise, e.g. <=1%.
- Block-scaled compile keys include QK dtype, SF dtype, V dtype, `mSFQ/mSFK/mSFV` presence, and V group size.
- No disabled S2T path or debug skip flags in accepted mode.
- Precision table covers NVFP4+BF16, NVFP4+FP8, MXFP8+FP8, and NVFP4+FP8 group-128 across causal/non-causal and key hdim values.
- Benchmarks compare integrated kernel against standalone FP4 kernel and upstream BF16 baseline.
- Unsupported combinations fail early with clear API errors.
- At least one minimal S2T layout test protects the MLIR legalization fix.
