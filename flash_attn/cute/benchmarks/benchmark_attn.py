from collections import namedtuple
from functools import partial
import math
import os
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

try:
    import cudnn
except ImportError:
    cudnn = None
# cudnn = None

Timing = NamedTuple('timing', [('mean', float)])


from einops import rearrange, repeat

# from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
from flash_attn.cute.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.interface import flash_attn_func as flash_attn_func_python
from flash_attn.cute.interface import flash_attn_varlen_func as flash_attn_varlen_func_python
try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    flash_attn_func_v3 = None
    flash_attn_varlen_func_v3 = None

if torch.cuda.get_device_capability()[0] != 9:
    flash_attn_func_v3 = None
# flash_attn_func_v3 = None

flash_attn_func = None

# Import FMHA kernel
import warnings
import sys
from pathlib import Path

# Add workspace root to path for importing fmha module
# File is at: examples/python/CuTeDSL/blackwell/flash-attention/flash_attn/cute/benchmarks/benchmark_attn.py
# Need to go up 9 levels to reach workspace root
_workspace_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cute.typing import Int32, Float32
    from examples.python.CuTeDSL.blackwell.fmha import BlackwellFusedMultiHeadAttentionForward, MaskType
    FMHA_AVAILABLE = True
except ImportError:
    warnings.warn("FMHA is not available")
    FMHA_AVAILABLE = False

from triton.testing import do_bench

def time_fwd(func, *args, repeats=30, verbose=True, desc="", **kwargs):
    # # Warmup
    # for _ in range(5):
    #     func(*args, **kwargs)
    # time.sleep(1)
    # return benchmark_forward(func, *args, **kwargs, repeats=repeats, verbose=verbose, desc=desc)[1]
    # s = torch.cuda.Stream()
    # s.wait_stream(torch.cuda.current_stream())
    # with torch.cuda.stream(s):
    #     for _ in range(2):
    #         out = func(*args, **kwargs)
    # torch.cuda.current_stream().wait_stream(s)
    # graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(graph):
    #     out = func(*args, **kwargs)
    # time_f = benchmark_forward(lambda: graph.replay(), repeats=repeats, verbose=verbose, desc=desc)
    # # return time_f[1].mean
    # return time_f[1]
    return Timing(do_bench(lambda: func(*args, **kwargs), warmup=5, rep=repeats) * 1e-3)


def flops(batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal=False, window_size=(None, None)):
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


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    else:
        raise ValueError("Unsupported tensor data type.")


def cudnn_spda_setup(q, k, v, causal=False, window_size_left=None):
    b, nheads, seqlen_q, headdim = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    headdim_v = v.shape[-1]
    assert v.shape == (b, nheads_k, seqlen_k, headdim_v)
    assert cudnn is not None, 'CUDNN is not available'
    q_gpu, k_gpu, v_gpu = q, k, v
    o_gpu = torch.empty((b, nheads, seqlen_q, headdim_v), dtype=q.dtype, device=q.device)
    stats_gpu = torch.empty(b, nheads, seqlen_q, 1, dtype=torch.float32, device=q.device)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu.detach())
    k = graph.tensor_like(k_gpu.detach())
    v = graph.tensor_like(v_gpu.detach())

    o, stats = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        is_inference=False,
        attn_scale=1.0 / math.sqrt(headdim),
        # use_causal_mask_bottom_right=causal or window_size_left is not None,
        use_causal_mask=causal or window_size_left is not None,
        sliding_window_length=window_size_left if window_size_left is not None and not causal else None,
    )

    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        stats: stats_gpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return o_gpu

    return run


def cudnn_spda_bwd_setup(q, k, v, o, g, lse, causal=False, window_size_left=None):
    b, nheads, seqlen_q, headdim = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    headdim_v = v.shape[-1]
    assert v.shape == (b, nheads_k, seqlen_k, headdim_v)
    assert g.shape == (b, nheads, seqlen_q, headdim_v)
    assert o.shape == (b, nheads, seqlen_q, headdim_v)
    assert lse.shape == (b, nheads, seqlen_q, 1)
    assert cudnn is not None, 'CUDNN is not available'
    q_gpu, k_gpu, v_gpu, o_gpu, g_gpu = q, k, v, o, g
    dq_gpu = torch.empty_like(q_gpu)
    dk_gpu = torch.empty_like(k_gpu)
    dv_gpu = torch.empty_like(v_gpu)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu.detach())
    k = graph.tensor_like(k_gpu.detach())
    v = graph.tensor_like(v_gpu.detach())
    o = graph.tensor_like(o_gpu.detach())
    g = graph.tensor_like(g_gpu.detach())
    stats = graph.tensor_like(lse.detach())

    dq, dk, dv = graph.sdpa_backward(
        name="sdpa_backward",
        q=q,
        k=k,
        v=v,
        o=o,
        dO=g,
        stats=stats,
        attn_scale=1.0 / math.sqrt(headdim),
        # use_causal_mask_bottom_right=causal or window_size_left is not None,
        use_causal_mask=causal or window_size_left is not None,
        sliding_window_length=window_size_left if window_size_left is not None and not causal else None,
        use_deterministic_algorithm=False,
    )

    dq.set_output(True).set_dim(dq_gpu.shape).set_stride(dq_gpu.stride())
    dk.set_output(True).set_dim(dk_gpu.shape).set_stride(dk_gpu.stride())
    dv.set_output(True).set_dim(dv_gpu.shape).set_stride(dv_gpu.stride())

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        g: g_gpu,
        stats: lse,
        dq: dq_gpu,
        dk: dk_gpu,
        dv: dv_gpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return dq_gpu, dk_gpu, dv_gpu

    return run


def fmha_setup(q, k, v, causal=False, is_persistent=True):
    """Setup FMHA kernel for benchmarking.
    
    Args:
        q: Query tensor of shape (batch, seqlen_q, nheads, headdim)
        k: Key tensor of shape (batch, seqlen_k, nheads_kv, headdim)
        v: Value tensor of shape (batch, seqlen_k, nheads_kv, headdim_v)
        causal: Whether to use causal masking
        is_persistent: Whether to use persistent kernel mode
    
    Returns:
        A callable function that executes the FMHA kernel
    """
    if not FMHA_AVAILABLE:
        return None
    
    # Check if we're on a compatible GPU (SM100/B100)
    if torch.cuda.get_device_capability()[0] < 10:
        return None
    
    b, s_q, h_q, d = q.shape
    _, s_k, h_k, d_k = k.shape
    _, _, _, d_v = v.shape
    
    # Check constraints
    if d not in {32, 64, 128}:
        return None
    if h_q % h_k != 0:
        return None
    
    # Convert PyTorch dtype to CUTE dtype
    if q.dtype == torch.float16:
        in_dtype = cutlass.Float16
        out_dtype = cutlass.Float16
    elif q.dtype == torch.bfloat16:
        in_dtype = cutlass.BFloat16
        out_dtype = cutlass.BFloat16
    else:
        return None
    
    qk_acc_dtype = Float32
    pv_acc_dtype = Float32
    
    # Default MMA tiler for SM100
    mma_tiler_mn = (128, 128)
    mma_tiler = (*mma_tiler_mn, d)
    
    # Determine mask type
    mask_type = MaskType.CAUSAL_MASK if causal else MaskType.NO_MASK
    if not causal and s_k % mma_tiler_mn[1] != 0:
        mask_type = MaskType.RESIDUAL_MASK
    
    # Create FMHA instance
    fmha = BlackwellFusedMultiHeadAttentionForward(
        qk_acc_dtype,
        pv_acc_dtype,
        mma_tiler,
        is_persistent,
        mask_type,
    )
    
    # Convert PyTorch tensors to CUTE tensors
    # FMHA expects layout with specific strides:
    # Q: (s_q, d, ((h_r, h_k), b)) with stride (d * h_r * h_k, 1, ((d, d * h_r), stride_b_qo))
    # K: (s_k, d, ((h_r, h_k), b)) with stride (d * h_k, 1, ((0, d), stride_b_kv))
    # V: (d_v, s_k, ((h_r, h_k), b)) with stride (1, d * h_k, ((0, d), stride_b_kv))
    # O: (s_q, d_v, ((h_r, h_k), b)) with stride (d * h_r * h_k, 1, ((d, d * h_r), stride_b_qo))
    h_r = h_q // h_k
    
    # For Q: need (b, s_q, h_r, h_k, d) in memory, then view as (s_q, d, h_r, h_k, b)
    # The stride for Q layout is: (d * h_r * h_k, 1, ((d, d * h_r), stride_b_qo))
    # where stride_b_qo = h_r * h_k * s_q * d
    # So memory layout should be: (b, s_q, h_r, h_k, d) -> stride (s_q * h_r * h_k * d, h_r * h_k * d, h_k * d, d, 1)
    # Then we view it as (s_q, d, h_r, h_k, b)
    # Detach to avoid gradient issues when converting to CUTE tensors
    q_reshaped = q.detach().view(b, s_q, h_r, h_k, d).permute(1, 4, 2, 3, 0).contiguous()
    q_tensor = from_dlpack(q_reshaped, assumed_align=16)
    q_tensor.element_type = in_dtype
    
    # For K: need to broadcast h_r, so we create (s_k, d, h_r, h_k, b) with h_r broadcasted (0-stride)
    # Create base tensor and expand (without contiguous to preserve broadcasting)
    k_base = k.detach().view(b, s_k, h_k, d).permute(1, 3, 2, 0).contiguous()  # (s_k, d, h_k, b)
    k_reshaped = k_base.unsqueeze(2).expand(s_k, d, h_r, h_k, b)  # Don't call contiguous() to preserve 0-stride
    k_tensor = from_dlpack(k_reshaped, assumed_align=16)
    k_tensor.element_type = in_dtype
    
    # For V: (d_v, s_k, h_r, h_k, b) with h_r broadcasted
    v_base = v.detach().view(b, s_k, h_k, d_v).permute(3, 1, 2, 0).contiguous()  # (d_v, s_k, h_k, b)
    v_reshaped = v_base.unsqueeze(2).expand(d_v, s_k, h_r, h_k, b)  # Don't call contiguous() to preserve 0-stride
    v_tensor = from_dlpack(v_reshaped, assumed_align=16)
    v_tensor.element_type = in_dtype
    
    # Create output tensor: (s_q, d_v, h_r, h_k, b)
    o_reshaped = torch.zeros(s_q, d_v, h_r, h_k, b, dtype=q.dtype, device=q.device)
    o_tensor = from_dlpack(o_reshaped, assumed_align=16)
    o_tensor.element_type = out_dtype
    
    # Setup problem size and scales
    problem_size = (b, s_q, s_k, h_q, h_k, d)
    scale_softmax = 1.0 / math.sqrt(d)
    log2_e = math.log2(math.exp(1.0))
    scale_softmax_log2 = scale_softmax * log2_e
    scale_output = 1.0
    
    # Get stream
    current_stream = cutlass_torch.default_stream()
    
    # Compile kernel
    try:
        compiled_fmha = cute.compile(
            fmha,
            q_tensor.iterator,
            k_tensor.iterator,
            v_tensor.iterator,
            o_tensor.iterator,
            problem_size,
            None,  # cum_seqlen_q
            None,  # cum_seqlen_k
            scale_softmax_log2,
            scale_output,
            current_stream,
        )
    except Exception as e:
        print(f"FMHA compilation failed: {e}")
        return None
    
    # Store references to keep tensors alive
    compiled_fmha._q_tensor = q_tensor
    compiled_fmha._k_tensor = k_tensor
    compiled_fmha._v_tensor = v_tensor
    compiled_fmha._o_tensor = o_tensor
    compiled_fmha._q_reshaped = q_reshaped
    compiled_fmha._k_reshaped = k_reshaped
    compiled_fmha._v_reshaped = v_reshaped
    compiled_fmha._o_reshaped = o_reshaped
    compiled_fmha._k_base = k_base
    compiled_fmha._v_base = v_base
    
    # Pre-compute and copy reshaped tensors once (input tensors don't change during benchmarking)
    # This avoids expensive reshape/copy operations on every iteration
    q_reshaped_input = q.detach().view(b, s_q, h_r, h_k, d).permute(1, 4, 2, 3, 0).contiguous()
    k_base_input = k.detach().view(b, s_k, h_k, d).permute(1, 3, 2, 0).contiguous()
    v_base_input = v.detach().view(b, s_k, h_k, d_v).permute(3, 1, 2, 0).contiguous()
    
    # Copy once to initialize - input tensors don't change during benchmarking, so no need to copy again
    q_reshaped.copy_(q_reshaped_input)
    compiled_fmha._k_base.copy_(k_base_input)
    compiled_fmha._v_base.copy_(v_base_input)
    
    def run(*args, **kwargs):
        # For benchmarking, input tensors are the same across iterations, so we don't need to
        # reshape or copy them again. The kernel will use the data that's already in q_reshaped,
        # k_base, and v_base. This eliminates expensive reshape/copy operations from the hot path.
        
        # Execute kernel
        compiled_fmha(
            q_tensor.iterator,
            k_tensor.iterator,
            v_tensor.iterator,
            o_tensor.iterator,
            problem_size,
            None,
            None,
            scale_softmax_log2,
            scale_output,
            current_stream,
        )
        
        return o_reshaped
    
    return run


torch.manual_seed(0)
repeats = 10
dropout_p = 0.0
causal = False
dtype = torch.bfloat16
# dtype = torch.float8_e4m3fn
dtype_gen = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
device = 'cuda'
verbose = True
varlen = False
has_backward = True
page_size = None
# page_size = 128
softcap = 0.0
V_colmajor = False
deterministic = False
batch_size = 2
# seqlen = 2048
seqlen = 8192
# seqlen = 4096
# seqlen = 2047
dim = 2048
# headdim = 128
# headdim = 64
headdim = 256
# for headdim in [64, 128, 256]:
# bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
bs_seqlen_vals = [(32, 1024), (16, 2048), (8, 4096), (4, 8192), (2, 16384), (1, 32768)]
# bs_seqlen_vals = [(32, 512), (16, 1024)]
# bs_seqlen_vals = [(2, 64 * 132)]
# bs_seqlen_vals = [(4, 8192)]
# bs_seqlen_vals = [(1, 16 * 1024)]
time_f = {}
time_b = {}

# for headdim in [64, 128, 256]:
# for headdim in [64, 96, 128, 192]:
# for headdim in [64, 96, 128, 192, 256]:
# for headdim in [64, 96, 128]:
# for headdim in [64, 128, 256]:
# for headdim in [64, 96, 128, 192, 256]:
for headdim in [128]:
    # nheads = dim // headdim
    nheads = 32 if headdim <= 64 else 16 if headdim <= 192 else 8
    # nheads = 128
    # headdim = 64
    # batch_size = 64
    # seqlen = 512
    # nheads = 8
    # headdim = 128
    nheads_kv = nheads
    # nheads_kv = nheads // 8
    # nheads_kv = 1
    # headdim_v = headdim
    headdim_v = 128 if headdim == 192 else headdim
    # headdim_v = 512
    has_qv = headdim == 64 and headdim_v == 512
    # has_qv = False
    # sinks = torch.randn(nheads, dtype=torch.bfloat16, device=device)
    sinks = None

    for batch_size, seqlen in bs_seqlen_vals:
        num_splits = 0
        # window_size = (-1, -1)
        window_size = (None, None)
        window_size_fa = (-1, -1)
        # window_size = (seqlen // 2 - 1, 0)
        pack_gqa = None
        # seqlen_q = 64
        seqlen_q = seqlen
        leftpad_k = None
        # leftpad_k = torch.full((batch_size,), 0, device=device, dtype=torch.int32)
        q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype_gen, requires_grad=has_backward)
        k = torch.randn(batch_size, seqlen, nheads_kv, headdim, device=device, dtype=dtype_gen, requires_grad=has_backward)
        v = torch.randn(batch_size, seqlen, nheads_kv, headdim_v, device=device, dtype=dtype_gen, requires_grad=has_backward)
        q, k, v = [x.detach().to(dtype).requires_grad_(has_backward) for x in [q, k, v]]
        v_colmajor = v.detach().transpose(-1, -3).contiguous().transpose(-1, -3).requires_grad_(has_backward)
        v_fa3 = v if not V_colmajor else v_colmajor
        qv = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen) if has_qv else None
        # q = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim), device=device, dtype=torch.int32).to(dtype)
        # k = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim), device=device, dtype=torch.int32).to(dtype)
        # v = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim_v), device=device, dtype=torch.int32).to(dtype)
        g = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen)
        o = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen)
        stats = torch.randn(batch_size, seqlen_q, nheads, 1, device=device, dtype=torch.float32)
        if varlen:
            q_unpad, k_unpad, v_unpad = [rearrange(x.detach(), "b s h d -> (b s) h d").requires_grad_(has_backward) for x in [q, k, v]]
            cu_seqlens_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen_q
            cu_seqlens_k = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen if page_size is None else None
            # cu_seqlens_q = torch.tensor([0, 248, 249, 250, 251, 252, 253, 254, 255, 256], device=device, dtype=torch.int32)
            # q_unpad = q_unpad[:256]
            # seqlen_q = 256
            # cu_seqlens_q = torch.tensor([0, 376, 377, 378, 379, 380, 381, 382, 383, 384], device=device, dtype=torch.int32)
            # q_unpad = q_unpad[:384]
            # seqlen_q = 384
        if page_size is not None:
            assert seqlen % page_size == 0
            k_paged, v_paged = [rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size) for x in [k, v]]
            page_table = rearrange(torch.arange(batch_size * seqlen // page_size, device=device, dtype=torch.int32),
                                   "(b s) -> b s", s=seqlen // page_size)
        else:
            page_table = None

        # for causal in [False, True]:
        for causal in [True]:
            print(f"\n### {headdim = }, {causal = }, {seqlen = } ###")
            nFLOPS = flops(batch_size, nheads, seqlen_q, seqlen, headdim if not has_qv else headdim + headdim_v, headdim_v, causal=causal, window_size=window_size)
            if cudnn is not None:
            # if False:
                if headdim <= 256 and dtype != torch.float8_e4m3fn:
                    cudnn_spda = cudnn_spda_setup(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal, window_size_left=window_size[0])
                    if has_backward and headdim == headdim_v:
                        cudnn_spda_bwd = cudnn_spda_bwd_setup(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), o.transpose(1, 2), g.transpose(1, 2), stats.transpose(1, 2), causal=causal, window_size_left=window_size[0])
            if dtype != torch.float8_e4m3fn and headdim == headdim_v and flash_attn_func is not None:
            # if False:
                if not varlen:
                    m0 = time_fwd(flash_attn_func, q, k, v, dropout_p, causal=causal, window_size=window_size, softcap=softcap, repeats=repeats, verbose=verbose, desc='Fav2')
                else:
                    m0 = time_fwd(flash_attn_varlen_func, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, dropout_p, causal=causal, window_size=window_size, softcap=softcap, repeats=repeats, verbose=verbose, desc='Fav2')
                time_f[(causal, headdim, batch_size, seqlen), "Flash2"] = m0.mean
                if has_backward:
                    time.sleep(1)
                    if not varlen:
                        _, m0b = benchmark_backward(flash_attn_func, q, k, v, dropout_p, causal=causal, window_size=window_size, softcap=softcap, deterministic=deterministic,
                                                    repeats=repeats, verbose=False, desc='Fav2')
                    else:
                        _, m0b = benchmark_backward(flash_attn_varlen_func, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, dropout_p, causal=causal, window_size=window_size, softcap=softcap, deterministic=deterministic,
                                                    repeats=repeats, verbose=False, desc='Fav2')
                    time_b[(causal, headdim, batch_size, seqlen), "Flash2"] = m0b.mean
            # pytorch_profiler(flash_attn_func, q, k, v, dropout_p, causal=causal, backward=True)

            if cudnn is not None:
            # if False:
                if headdim <= 256 and dtype != torch.float8_e4m3fn:
                    time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
                    m2 = time_fwd(cudnn_spda, repeats=repeats, verbose=verbose, desc='CuDNN')
                    time_f[(causal, headdim, batch_size, seqlen), "cuDNN"] = m2.mean
                    if has_backward:
                        time.sleep(1)
                        m2b = time_fwd(cudnn_spda_bwd, repeats=repeats, verbose=verbose, desc='CuDNN')
                        time_b[(causal, headdim, batch_size, seqlen), "cuDNN"] = m2b.mean
                # pytorch_profiler(cudnn_spda, backward=False)
                # pytorch_profiler(cudnn_spda_bwd, backward=False)
            time.sleep(1)
            if flash_attn_func_v3 is not None:
                if not varlen:
                    # m1 = time_fwd(flash_attn_func_v3, q, k if page_size is None else k_paged, v_fa3 if page_size is None else v_paged, cache_leftpad = leftpad_k, page_table=page_table, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, repeats=repeats, verbose=verbose, desc='Fav3')
                    m1 = time_fwd(flash_attn_func_v3, q, k if page_size is None else k_paged, v_fa3 if page_size is None else v_paged, causal=causal, window_size=window_size_fa, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, repeats=repeats, verbose=verbose, desc='Fav3')
                    # pytorch_profiler(flash_attn_func_v3, q, k if page_size is None else k_paged, v_fa3 if page_size is None else v_paged, page_table=page_table, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa)
                else:
                    m1 = time_fwd(flash_attn_varlen_func_v3, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, causal=causal, window_size=window_size_fa, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, repeats=repeats, verbose=verbose, desc='Fav3')
                    # pytorch_profiler(flash_attn_varlen_func_v3, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits)
                time_f[(causal, headdim, batch_size, seqlen), "Flash3"] = m1.mean
            if flash_attn_func_python is not None:
                if not varlen:
                    m1_py = time_fwd(flash_attn_func_python, q, k if page_size is None else k_paged, v_fa3 if page_size is None else v_paged, causal=causal, window_size=window_size, learnable_sink=sinks, softcap=softcap, pack_gqa=pack_gqa, repeats=repeats, verbose=verbose, desc='Fav3 python')
                else:
                    m1_py = time_fwd(flash_attn_varlen_func_python, q_unpad, k_unpad if page_size is None else k_paged, v_unpad if page_size is None else v_paged, cu_seqlens_q, cu_seqlens_k, page_table=page_table, causal=causal, window_size=window_size, softcap=softcap, pack_gqa=pack_gqa, repeats=repeats, verbose=verbose, desc='Fav3 python')
            
            # FMHA kernel benchmark
            m_fmha = None
            if FMHA_AVAILABLE and not varlen and headdim in {32, 64, 128} and dtype in {torch.float16, torch.bfloat16} and headdim == headdim_v:
                try:
                    time.sleep(1)
                    # Match the is_persistent logic from interface.py: persistent mode is disabled for causal
                    # Since we're already checking not varlen, cu_seqlens_q is None and local is False (window_size is None)
                    is_persistent = not causal
                    fmha_func = fmha_setup(q, k, v, causal=causal, is_persistent=is_persistent)
                    if fmha_func is not None:
                        m_fmha = time_fwd(fmha_func, repeats=repeats, verbose=verbose, desc='FMHA')
                        time_f[(causal, headdim, batch_size, seqlen), "FMHA"] = m_fmha.mean
                except Exception as e:
                    if verbose:
                        print(f"FMHA benchmark failed: {e}")
            
            if dtype != torch.float8_e4m3fn and headdim == headdim_v and flash_attn_func_v3 is not None and has_backward:
                time.sleep(1)
                if not varlen:
                    _, m1b = benchmark_backward(flash_attn_func_v3, q, k, v, causal=causal, softcap=softcap, repeats=repeats, verbose=False, desc='Fav3')
                else:
                    _, m1b = benchmark_backward(flash_attn_varlen_func_v3, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, causal=causal, window_size=window_size, softcap=softcap, deterministic=deterministic,
                                                repeats=repeats, verbose=False, desc='Fav3')
                time_b[(causal, headdim, batch_size, seqlen), "Flash3"] = m1b.mean
                time.sleep(1)
                # if not varlen:
                #     pytorch_profiler(flash_attn_func_v3, q, k, v, causal=causal, deterministic=deterministic, backward=True)
                # else:
                #     pytorch_profiler(flash_attn_varlen_func_v3, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, causal=causal, deterministic=deterministic, backward=True)
            # benchmark_forward(torch.clone, k, repeats=repeats, verbose=verbose, desc='Memcpy')
            if dtype != torch.float8_e4m3fn and headdim == headdim_v and flash_attn_func_python is not None and has_backward:
                _, m1b_py = benchmark_backward(flash_attn_func_python, q, k, v, causal=causal, softcap=softcap, repeats=repeats, verbose=False, desc='Fav2 python')

            if dtype != torch.float8_e4m3fn and headdim == headdim_v and flash_attn_func is not None:
            # if False:
                print(f'FAv2 fwd: {m0.mean * 1e3:.3f}ms, {(nFLOPS / m0.mean * 1e-12):.1f} TFLOPS')
                if has_backward:
                    print(f'FAv2 bwd: {m0b.mean * 1e3:.3f}ms, {(2.5 * nFLOPS / m0b.mean * 1e-12):.1f} TFLOPS')
            if cudnn is not None:
                print(f'CuDNN fwd: {m2.mean * 1e3:.3f}ms, {(nFLOPS / m2.mean * 1e-12):.1f} TFLOPS')
                if has_backward:
                    print(f'CuDNN bwd: {m2b.mean * 1e3:.3f}ms, {(2.5 * nFLOPS / m2b.mean * 1e-12):.1f} TFLOPS')
            if flash_attn_func_v3 is not None:
                print(f'FAv3 fwd: {m1.mean * 1e3:.3f}ms, {(nFLOPS / m1.mean * 1e-12):.1f} TFLOPS')
                if dtype != torch.float8_e4m3fn and headdim == headdim_v and has_backward:
                    print(f'FAv3 bwd: {m1b.mean * 1e3:.3f}ms, {(2.5 * nFLOPS / m1b.mean * 1e-12):.1f} TFLOPS')

            if flash_attn_func_python is not None:
                print(f'FA Python fwd: {m1_py.mean * 1e3:.3f}ms, {(nFLOPS / m1_py.mean * 1e-12):.1f} TFLOPS')
                # if dtype != torch.float8_e4m3fn and headdim == headdim_v and has_backward:
                #     print(f'FA Python bwd: {m1b_py.mean * 1e3:.3f}ms, {(2.5 * nFLOPS / m1b_py.mean * 1e-12):.1f} TFLOPS')
            
            if m_fmha is not None:
                print(f'FMHA fwd: {m_fmha.mean * 1e3:.3f}ms, {(nFLOPS / m_fmha.mean * 1e-12):.1f} TFLOPS')