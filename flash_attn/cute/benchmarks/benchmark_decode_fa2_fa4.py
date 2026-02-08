"""
Benchmark comparing FA2 (sm80) and FA4 (sm100) in decode mode (varlen, q_len=1 for each batch).

This benchmark tests decode mode performance with:
- 32 query heads and 8 key-value heads (pack_gqa=True)
- Variable length sequences with q_len=1 for each batch
- Causal masking enabled
"""

import time
import torch
import math
from einops import rearrange
from triton.testing import do_bench

from flash_attn.cute.benchmark import benchmark_forward

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_fa2
    FA2_AVAILABLE = True
except ImportError:
    flash_attn_varlen_func_fa2 = None
    FA2_AVAILABLE = False
    print("Warning: FA2 (flash_attn.flash_attn_interface) not available")

try:
    from flash_attn.cute.interface import flash_attn_varlen_func as flash_attn_varlen_func_fa4
    FA4_AVAILABLE = True
except ImportError:
    flash_attn_varlen_func_fa4 = None
    FA4_AVAILABLE = False
    print("Warning: FA4 (flash_attn.cute.interface) not available")

# Check device capability
try:
    device_capability = torch.cuda.get_device_capability()[0]
    print(f"Device compute capability: {device_capability}.x")
    print(f"FA2 (sm80) available: {device_capability >= 8}")
    print(f"FA4 (sm100) available: {device_capability >= 10}")
except RuntimeError as e:
    print(f"Warning: Could not detect CUDA device: {e}")
    print("This benchmark requires a CUDA-capable GPU.")
    print("Setting device_capability to 0 (will skip benchmarks)")
    device_capability = 0
print("=" * 80)

def time_fwd(func, *args, repeats=30, verbose=True, desc="", **kwargs):
    """Time forward pass using triton's do_bench."""
    return benchmark_forward(func, *args, **kwargs, repeats=repeats, verbose=verbose, desc=desc)[1]

def flops(batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal=False):
    """Calculate FLOPS for attention operation."""
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        avg_seqlen = seqlen_k
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)

def benchmark_decode_fa2_fa4():
    """Benchmark FA2 vs FA4 in decode mode."""
    # Check if at least one implementation is available
    if not FA2_AVAILABLE and not FA4_AVAILABLE:
        print("Error: Neither FA2 nor FA4 implementations are available!")
        print("Please ensure flash-attention is properly installed.")
        return
    
    torch.manual_seed(0)
    repeats = 30
    device = 'cuda'
    verbose = True
    causal = True  # Decode mode is always causal
    dtype = torch.bfloat16
    
    # Decode mode configuration: q_len=1 for each batch
    seqlen_q = 1  # Decode mode: single token per query
    nheads_q = 32
    nheads_kv = 8
    headdim = 128
    headdim_v = 128
    pack_gqa = True
    
    print("=" * 80)
    print("Decode Mode Benchmark: FA2 (sm80) vs FA4 (sm100)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Query heads: {nheads_q}")
    print(f"  KV heads: {nheads_kv}")
    print(f"  Head dim: {headdim}")
    print(f"  Head dim V: {headdim_v}")
    print(f"  Pack GQA: {pack_gqa}")
    print(f"  Query length: {seqlen_q} (decode mode)")
    print(f"  Causal: {causal}")
    print("=" * 80)
    
    # Test different batch sizes and KV sequence lengths
    # Typical decode scenarios: varying batch sizes and KV cache lengths
    batch_seqlen_k_vals = [
        (1, 1024),
        (4, 2048),
        (8, 4096),
        (16, 8192),
        (32, 16384),
        (64, 32768),
        (1, 65536),
    ]
    
    time_f_fa2 = {}
    time_f_fa4 = {}
    
    for batch_size, seqlen_k in batch_seqlen_k_vals:
        print(f"\n{'='*80}")
        print(f"Batch size: {batch_size}, KV sequence length: {seqlen_k}")
        print(f"{'='*80}")
        
        # Create varlen tensors for decode mode
        # Each batch has q_len=1, but KV sequences can vary
        # For simplicity, we'll use equal KV lengths per batch
        total_q = batch_size * seqlen_q  # batch_size * 1
        total_k = batch_size * seqlen_k
        
        # Create cumulative sequence lengths
        cu_seqlens_q = torch.arange(0, total_q + seqlen_q, step=seqlen_q, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.arange(0, total_k + seqlen_k, step=seqlen_k, dtype=torch.int32, device=device)
        
        # Create unflattened tensors: (total_q, nheads_q, headdim)
        q_unpad = torch.randn(total_q, nheads_q, headdim, device=device, dtype=dtype, requires_grad=False)
        k_unpad = torch.randn(total_k, nheads_kv, headdim, device=device, dtype=dtype, requires_grad=False)
        v_unpad = torch.randn(total_k, nheads_kv, headdim_v, device=device, dtype=dtype, requires_grad=False)
        
        # Calculate FLOPS
        nFLOPS = flops(batch_size, nheads_q, seqlen_q, seqlen_k, headdim, headdim_v, causal=causal)
        
        # Benchmark FA2 (sm80) - only if device supports it and FA2 is available
        m_fa2 = None
        if device_capability >= 8 and FA2_AVAILABLE and flash_attn_varlen_func_fa2 is not None:
            try:
                time.sleep(1)  # Sleep to avoid residual power throttling
                print(f"\nBenchmarking FA2 (sm80)...")
                m_fa2 = time_fwd(
                    flash_attn_varlen_func_fa2,
                    q_unpad, k_unpad, v_unpad,
                    cu_seqlens_q, cu_seqlens_k,
                    seqlen_q, seqlen_k,  # max_seqlen_q, max_seqlen_k for FA2
                    dropout_p=0.0,
                    causal=causal,
                    window_size=(-1, -1),  # FA2 uses (-1, -1) for infinite window
                    repeats=repeats,
                    verbose=verbose,
                    desc='FA2 (sm80)'
                )
                time_f_fa2[(batch_size, seqlen_k)] = m_fa2.mean
                print(f"FA2 (sm80) fwd: {m_fa2.mean * 1e3:.3f}ms, {(nFLOPS / m_fa2.mean * 1e-12):.1f} TFLOPS")
            except Exception as e:
                print(f"FA2 (sm80) benchmark failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("FA2 (sm80) not available on this device")
        
        # Benchmark FA4 (sm100) - only if device supports it and FA4 is available
        m_fa4 = None
        if device_capability >= 10 and FA4_AVAILABLE and flash_attn_varlen_func_fa4 is not None:
            try:
                time.sleep(1)  # Sleep to avoid residual power throttling
                print(f"\nBenchmarking FA4 (sm100)...")
                m_fa4 = time_fwd(
                    flash_attn_varlen_func_fa4,
                    q_unpad, k_unpad, v_unpad,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    page_table=None,
                    causal=causal,
                    window_size=(None, None),  # FA4 uses (None, None) for infinite window
                    pack_gqa=pack_gqa,
                    repeats=repeats,
                    verbose=verbose,
                    desc='FA4 (sm100)'
                )
                time_f_fa4[(batch_size, seqlen_k)] = m_fa4.mean
                print(f"FA4 (sm100) fwd: {m_fa4.mean * 1e3:.3f}ms, {(nFLOPS / m_fa4.mean * 1e-12):.1f} TFLOPS")
            except Exception as e:
                print(f"FA4 (sm100) benchmark failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("FA4 (sm100) not available on this device")
        
        # Compare results
        if m_fa2 is not None and m_fa4 is not None:
            speedup = m_fa2.mean / m_fa4.mean
            print(f"\n{'='*80}")
            print(f"Comparison:")
            print(f"  FA2 (sm80):  {m_fa2.mean * 1e3:.3f}ms, {(nFLOPS / m_fa2.mean * 1e-12):.1f} TFLOPS")
            print(f"  FA4 (sm100): {m_fa4.mean * 1e3:.3f}ms, {(nFLOPS / m_fa4.mean * 1e-12):.1f} TFLOPS")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"{'='*80}")
        elif m_fa2 is not None:
            print(f"\nOnly FA2 (sm80) available: {m_fa2.mean * 1e3:.3f}ms, {(nFLOPS / m_fa2.mean * 1e-12):.1f} TFLOPS")
        elif m_fa4 is not None:
            print(f"\nOnly FA4 (sm100) available: {m_fa4.mean * 1e3:.3f}ms, {(nFLOPS / m_fa4.mean * 1e-12):.1f} TFLOPS")
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    if time_f_fa2 and time_f_fa4:
        print("\nFA2 (sm80) results:")
        for (batch_size, seqlen_k), time_mean in time_f_fa2.items():
            nFLOPS = flops(batch_size, nheads_q, seqlen_q, seqlen_k, headdim, headdim_v, causal=causal)
            print(f"  Batch={batch_size:3d}, KV_len={seqlen_k:5d}: {time_mean*1e3:6.3f}ms, {(nFLOPS/time_mean*1e-12):6.1f} TFLOPS")
        
        print("\nFA4 (sm100) results:")
        for (batch_size, seqlen_k), time_mean in time_f_fa4.items():
            nFLOPS = flops(batch_size, nheads_q, seqlen_q, seqlen_k, headdim, headdim_v, causal=causal)
            print(f"  Batch={batch_size:3d}, KV_len={seqlen_k:5d}: {time_mean*1e3:6.3f}ms, {(nFLOPS/time_mean*1e-12):6.1f} TFLOPS")
        
        print("\nSpeedup (FA2/FA4):")
        for (batch_size, seqlen_k) in time_f_fa2.keys():
            if (batch_size, seqlen_k) in time_f_fa4:
                speedup = time_f_fa2[(batch_size, seqlen_k)] / time_f_fa4[(batch_size, seqlen_k)]
                print(f"  Batch={batch_size:3d}, KV_len={seqlen_k:5d}: {speedup:.2f}x")
    elif time_f_fa2:
        print("\nOnly FA2 (sm80) results available:")
        for (batch_size, seqlen_k), time_mean in time_f_fa2.items():
            nFLOPS = flops(batch_size, nheads_q, seqlen_q, seqlen_k, headdim, headdim_v, causal=causal)
            print(f"  Batch={batch_size:3d}, KV_len={seqlen_k:5d}: {time_mean*1e3:6.3f}ms, {(nFLOPS/time_mean*1e-12):6.1f} TFLOPS")
    elif time_f_fa4:
        print("\nOnly FA4 (sm100) results available:")
        for (batch_size, seqlen_k), time_mean in time_f_fa4.items():
            nFLOPS = flops(batch_size, nheads_q, seqlen_q, seqlen_k, headdim, headdim_v, causal=causal)
            print(f"  Batch={batch_size:3d}, KV_len={seqlen_k:5d}: {time_mean*1e3:6.3f}ms, {(nFLOPS/time_mean*1e-12):6.1f} TFLOPS")
    
    print(f"\n{'='*80}")
    print("Benchmark completed!")
    if time_f_fa2 and time_f_fa4:
        print(f"✓ Successfully compared FA2 (sm80) and FA4 (sm100) in decode mode")
        print(f"  - FA2 (sm80) results: {len(time_f_fa2)} configurations")
        print(f"  - FA4 (sm100) results: {len(time_f_fa4)} configurations")
    elif time_f_fa2:
        print(f"⚠ Only FA2 (sm80) results available ({len(time_f_fa2)} configurations)")
        print(f"  FA4 (sm100) was not available or failed")
    elif time_f_fa4:
        print(f"⚠ Only FA4 (sm100) results available ({len(time_f_fa4)} configurations)")
        print(f"  FA2 (sm80) was not available or failed")
    else:
        print(f"✗ No results available - both FA2 and FA4 failed or are unavailable")
    print(f"nheads_q: {nheads_q}, nheads_kv: {nheads_kv}, headdim: {headdim}, headdim_v: {headdim_v}")
    print(f"{'='*80}")

if __name__ == "__main__":
    benchmark_decode_fa2_fa4()
