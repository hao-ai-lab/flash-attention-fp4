"""Standalone benchmark for FP4 Flash Attention.

This benchmark tests FP4 attention kernels and compares them against
the standard Python interface implementation.
"""

import math
import time
from typing import NamedTuple, Optional, Tuple

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from flash_attn.cute.interface import flash_attn_func as flash_attn_func_python
from flash_attn.cute.interface import flash_attn_varlen_func as flash_attn_varlen_func_python

from triton.testing import do_bench

Timing = NamedTuple('timing', [('mean', float)])


def flops(batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal=False, window_size=(None, None)):
    """Calculate FLOPS for attention computation."""
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        if window_size == (None, None):
            avg_seqlen = seqlen_k
        else:
            row_idx = torch.arange(seqlen_q, device='cuda')
            col_left = torch.maximum(row_idx + seqlen_k - seqlen_q - window_size[0], torch.tensor(0)) if window_size[0] is not None else torch.zeros_like(row_idx)
            col_right = torch.minimum(row_idx + seqlen_k - seqlen_q + window_size[1], torch.tensor(seqlen_k - 1)) if window_size[1] is not None else torch.full_like(row_idx, seqlen_k - 1)
            avg_seqlen = (col_right - col_left + 1).float().mean().item()
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


def create_scale_factor_tensor(batch, seqlen, nheads, headdim, sf_vec_size, sf_dtype, device='cuda'):
    """Create scale factor tensor for Q/K/V.
    
    Args:
        batch: Batch size
        seqlen: Sequence length
        nheads: Number of heads
        headdim: Head dimension
        sf_vec_size: Scale factor vector size (typically 16 for NVFP4)
        sf_dtype: Scale factor dtype (typically Float8E4M3FN for NVFP4)
        device: Device to create tensor on
    
    Returns:
        Tuple of (ref_tensor_cpu, cute_tensor, cute_torch_tensor)
    """
    def ceil_div(a, b):
        return (a + b - 1) // b
    
    # Scale factor shape: (batch, nheads, seqlen, ceil_div(headdim, sf_vec_size))
    # For attention, we need scale factors per head dimension
    l = batch * nheads
    mn = seqlen
    k = headdim
    sf_k = ceil_div(k, sf_vec_size)
    ref_shape = (l, mn, sf_k)
    
    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )
    
    ref_permute_order = (1, 2, 0)
    mma_permute_order = (3, 4, 1, 5, 2, 0)
    
    # Create f32 ref torch tensor (cpu)
    ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        ref_shape,
        torch.float32,
        permute_order=ref_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(
            min_val=1,
            max_val=3,
        ),
    )
    
    # Create f32 cute torch tensor (cpu)
    cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(
            min_val=0,
            max_val=1,
        ),
    )
    
    # convert ref f32 tensor to cute f32 tensor
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(ref_f32_torch_tensor_cpu),
        from_dlpack(cute_f32_torch_tensor_cpu),
    )
    cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()
    
    # reshape makes memory contiguous
    ref_f32_torch_tensor_cpu = (
        ref_f32_torch_tensor_cpu.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, mn, sf_k, sf_vec_size)
        .reshape(l, mn, sf_k * sf_vec_size)
        .permute(*ref_permute_order)
    )
    # prune to actual k dimension
    ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]
    
    # Create dtype cute torch tensor (cpu)
    cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
        cute_f32_torch_tensor_cpu,
        sf_dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    
    # Convert f32 cute tensor to dtype cute tensor
    cute_tensor = cutlass_torch.convert_cute_tensor(
        cute_f32_torch_tensor,
        cute_tensor,
        sf_dtype,
        is_dynamic_layout=True,
    )
    return ref_f32_torch_tensor_cpu, cute_tensor, cute_torch_tensor


def create_fp4_attention_tensors(batch, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, 
                                  device='cuda', dtype_gen=torch.bfloat16, quant_v=False, return_torch=True,
                                  ab_dtype=None, sf_dtype=None, sf_vec_size=None):
    """Create FP4 attention tensors (Q, K, V) with scale factors.
    
    Args:
        batch: Batch size
        seqlen_q: Query sequence length
        seqlen_k: Key sequence length
        nheads: Number of query heads
        nheads_kv: Number of key/value heads
        headdim: Head dimension for Q/K
        headdim_v: Head dimension for V
        device: Device to create tensors on
        dtype_gen: Dtype to generate random data in (before conversion to FP4)
        quant_v: Whether to quantize V to FP4 (default: False, only QK are quantized)
        return_torch: Whether to return torch tensors (default: True)
        ab_dtype: Data type for A/B matrices (default: Float4E2M1FN)
        sf_dtype: Scale factor dtype (default: Float8E4M3FN)
        sf_vec_size: Scale factor vector size (default: 16)
    Returns:
        Tuple of (q_fp4, k_fp4, v_tensor, q_sf, k_sf, v_sf, q_ref, k_ref, v_ref)
        where q_fp4, k_fp4 are FP4 tensors, v_tensor is FP4 if quant_v=True else regular dtype,
        q_sf, k_sf are scale factor tensors, v_sf is scale factor tensor if quant_v=True else None,
        and *_ref are reference FP32 tensors
    """
    # Default FP4 parameters
    if ab_dtype is None:
        ab_dtype = cutlass.Float4E2M1FN  # FP4 data type
    if sf_dtype is None:
        sf_dtype = cutlass.Float8E4M3FN  # Scale factor dtype
    if sf_vec_size is None:
        sf_vec_size = 16  # 1 scale factor per 16 elements
    
    # Create reference FP32 tensors
    q_ref = torch.randn(batch, seqlen_q, nheads, headdim, device=device, dtype=torch.float32)
    k_ref = torch.randn(batch, seqlen_k, nheads_kv, headdim, device=device, dtype=torch.float32)
    v_ref = torch.randn(batch, seqlen_k, nheads_kv, headdim_v, device=device, dtype=torch.float32)
    
    # Create FP4 tensors for Q and K (V quantization is optional)
    # First create CUTE tensors for Q and K
    q_tensor, q_torch_underlying = cutlass_torch.cute_tensor_like(
        q_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    k_tensor, k_torch_underlying = cutlass_torch.cute_tensor_like(
        k_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    # Get the correct stride_order from the reference tensors
    # stride_order should match the layout of the original tensor
    q_stride_order = tuple(q_ref.dim_order())
    k_stride_order = tuple(k_ref.dim_order())
    # Mark tensors to be byte aligned (FP4 needs divisibility of 2)
    # For flash attention, headdim is the last dimension (index 3), which should be mode 1
    # Mode 0 is for batch/seqlen/nheads, mode 1 is for headdim
    q_tensor.mark_compact_shape_dynamic(
        mode=1,  # headdim dimension needs divisibility for FP4
        stride_order=q_stride_order,
        divisibility=2 if ab_dtype == cutlass.Float4E2M1FN else 1,
    )
    k_tensor.mark_compact_shape_dynamic(
        mode=1,  # headdim dimension needs divisibility for FP4
        stride_order=k_stride_order,
        divisibility=2 if ab_dtype == cutlass.Float4E2M1FN else 1,
    )
    
    # Convert FP32 tensors to FP4 format for Q and K
    q_tensor = cutlass_torch.convert_cute_tensor(
        q_ref, q_tensor, ab_dtype, is_dynamic_layout=True
    )
    k_tensor = cutlass_torch.convert_cute_tensor(
        k_ref, k_tensor, ab_dtype, is_dynamic_layout=True
    )
    
    # Handle V: quantize to FP4 if quant_v=True, otherwise use regular dtype
    if quant_v:
        v_tensor, v_torch_underlying = cutlass_torch.cute_tensor_like(
            v_ref, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        # Get the correct stride_order from the reference tensor
        v_stride_order = tuple(v_ref.dim_order())
        v_tensor.mark_compact_shape_dynamic(
            mode=1,  # headdim_v dimension needs divisibility for FP4
            stride_order=v_stride_order,
            divisibility=2 if ab_dtype == cutlass.Float4E2M1FN else 1,
        )
        v_tensor = cutlass_torch.convert_cute_tensor(
            v_ref, v_tensor, ab_dtype, is_dynamic_layout=True
        )
    else:
        # V stays as regular dtype (not FP4 quantized) - create CUTE tensor
        # Convert torch dtype to CUTE dtype
        assert dtype_gen in [torch.bfloat16, torch.float16]
        if dtype_gen == torch.bfloat16:
            v_cute_dtype = cutlass.BFloat16
        elif dtype_gen == torch.float16:
            v_cute_dtype = cutlass.Float16

        v_tensor, v_torch_underlying = cutlass_torch.cute_tensor_like(
            v_ref, v_cute_dtype, is_dynamic_layout=True, assumed_align=16
        )
        # Get the correct stride_order from the reference tensor
        v_stride_order = tuple(v_ref.dim_order())
        v_tensor.mark_compact_shape_dynamic(
            mode=1,  # headdim_v dimension
            stride_order=v_stride_order,
            divisibility=1,  # Not FP4, so no divisibility requirement
        )
        v_tensor = cutlass_torch.convert_cute_tensor(
            v_ref, v_tensor, v_cute_dtype, is_dynamic_layout=True
        )
    
    # Create scale factor tensors for Q and K (V scale factors are optional)
    # For Q: (batch, nheads, seqlen_q, headdim) -> scale factors for headdim dimension
    # Scale factors are per (batch * nheads, seqlen_q, ceil_div(headdim, sf_vec_size))
    q_sf_ref, q_sf_tensor, q_sf_torch_underlying = create_scale_factor_tensor(
        batch, seqlen_q, nheads, headdim, sf_vec_size, sf_dtype, device
    )
    
    # For K: (batch, nheads_kv, seqlen_k, headdim) -> scale factors for headdim dimension
    k_sf_ref, k_sf_tensor, k_sf_torch_underlying = create_scale_factor_tensor(
        batch, seqlen_k, nheads_kv, headdim, sf_vec_size, sf_dtype, device
    )
    
    # Create V scale factors only if V is being quantized
    if quant_v:
        v_sf_ref, v_sf_tensor, v_sf_torch_underlying = create_scale_factor_tensor(
            batch, seqlen_k, nheads_kv, headdim_v, sf_vec_size, sf_dtype, device
        )
    else:
        v_sf_tensor = None
        v_sf_torch_underlying = None
    if return_torch:
        return (q_torch_underlying, k_torch_underlying, v_torch_underlying, q_sf_torch_underlying, k_sf_torch_underlying, v_sf_torch_underlying, 
                q_ref, k_ref, v_ref)
    else:
        return (q_tensor, k_tensor, v_tensor, q_sf_tensor, k_sf_tensor, v_sf_tensor, 
                q_ref, k_ref, v_ref)


def time_fwd(func, *args, repeats=30, verbose=True, desc="", **kwargs):
    """Time forward pass execution."""
    return Timing(do_bench(lambda: func(*args, **kwargs), warmup=5, rep=repeats) * 1e-3)


def main(ab_dtype, sf_dtype, sf_vec_size, quant_v=False):
    """Main benchmark function.
    
    Args:
        ab_dtype: Data type for A/B matrices
        sf_dtype: Scale factor dtype
        sf_vec_size: Scale factor vector size
        quant_v: Whether to quantize V to FP4 (default: False, only QK are quantized)
    """
    torch.manual_seed(0)
    repeats = 10
    device = 'cuda'
    verbose = True
    causal = False
    dtype_gen = torch.bfloat16
    
    # Benchmark configurations
    bs_seqlen_vals = [(32, 1024), (16, 2048), (8, 4096), (4, 8192), (2, 16384), (1, 32768)]
    headdim = 128
    nheads = 16
    nheads_kv = nheads
    headdim_v = headdim
    
    print("=" * 80)
    print("FP4 Flash Attention Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Headdim: {headdim}, Nheads: {nheads}, NheadsKV: {nheads_kv}, HeaddimV: {headdim_v}")
    print(f"Causal: {causal}")
    print(f"Quantize V: {quant_v}")
    print("=" * 80)
    
    for batch_size, seqlen in bs_seqlen_vals:
        seqlen_q = seqlen
        window_size = (None, None)
        
        print(f"\n### Batch={batch_size}, SeqLen={seqlen} ###")
        
        # Create FP4 tensors (V quantization is optional)
        try:
            (q_fp4, k_fp4, v_tensor, q_sf, k_sf, v_sf, 
             q_ref, k_ref, v_ref) = create_fp4_attention_tensors(
                batch_size, seqlen_q, seqlen, nheads, nheads_kv, 
                headdim, headdim_v, device, dtype_gen, quant_v=quant_v, return_torch=False,
                ab_dtype=ab_dtype, sf_dtype=sf_dtype, sf_vec_size=sf_vec_size
            )
        except Exception as e:
            print(f"Failed to create FP4 tensors: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Calculate FLOPS
        nFLOPS = flops(batch_size, nheads, seqlen_q, seqlen, headdim, headdim_v, 
                      causal=causal, window_size=window_size)
        
        # Benchmark FP4 attention
        # Pass CUTE tensors directly (like dense GEMM example)
        m_fp4 = None
        try:
            time.sleep(1)
            # The interface should detect nvfp4 dtype and dispatch to FP4 kernel
            # Pass scale factor tensors (V scale factors only if quant_v=True)
            desc_str = 'FP4 Attention (QKV quantized)' if quant_v else 'FP4 Attention (QK quantized)'
            m_fp4 = time_fwd(
                flash_attn_func_python,
                q_fp4, k_fp4, v_tensor,
                causal=causal,
                window_size=window_size,
                mSFQ=q_sf,
                mSFK=k_sf,
                mSFV=v_sf,  # None if quant_v=False, scale factor tensor if quant_v=True
                repeats=repeats,
                verbose=verbose,
                desc=desc_str
            )
            print(f'FP4 Attention fwd: {m_fp4.mean * 1e3:.3f}ms, {(nFLOPS / m_fp4.mean * 1e-12):.1f} TFLOPS')
        except Exception as e:
            print(f"FP4 attention failed: {e}")
            import traceback
            traceback.print_exc()
            # If FP4 fails, we might need to use a different approach
            # For now, skip FP4 benchmark if it fails
        
        # Benchmark reference (FP16/BF16) attention for comparison
        try:
            # Create reference tensors in standard dtype
            q_ref_std = q_ref.to(dtype_gen)
            k_ref_std = k_ref.to(dtype_gen)
            v_ref_std = v_ref.to(dtype_gen)
            
            time.sleep(1)
            m_ref = time_fwd(
                flash_attn_func_python,
                q_ref_std, k_ref_std, v_ref_std,
                causal=causal,
                window_size=window_size,
                repeats=repeats,
                verbose=verbose,
                desc='Reference (FP16/BF16) Attention'
            )
            print(f'Reference fwd: {m_ref.mean * 1e3:.3f}ms, {(nFLOPS / m_ref.mean * 1e-12):.1f} TFLOPS')
            
            if m_fp4 is not None:
                speedup = m_ref.mean / m_fp4.mean
                print(f'Speedup: {speedup:.2f}x')
        except Exception as e:
            print(f"Reference attention failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark FP4 Flash Attention")
    parser.add_argument(
        "--quant_v",
        action="store_true",
        help="Quantize V to FP4 (default: False, only QK are quantized)"
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    args = parser.parse_args()
    main(args.ab_dtype, args.sf_dtype, args.sf_vec_size, args.quant_v)

