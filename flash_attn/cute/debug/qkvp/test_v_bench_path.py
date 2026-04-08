"""Use bench's V wrapping approach with our nvfp4 data."""
import sys, importlib.util
sys.path.insert(0, '/sgl-workspace/FastVideo')
spec = importlib.util.spec_from_file_location(
    'fac', '/sgl-workspace/FastVideo/fastvideo/attention/utils/flash_attn_cute.py'
)
fac = importlib.util.module_from_spec(spec); sys.modules['fac'] = fac
spec.loader.exec_module(fac)

import torch
import cutlass
import cutlass.cute as cute
from cutlass.torch import cute_tensor_like
from flashinfer.quantization import nvfp4_quantize, SfLayout
from flash_attn.cute.benchmarks.bench_fp4 import create_scale_factor_tensor

b, s, h, d = 1, 8192, 16, 128

# Build a torch BF16 V with K-major storage like bench
v_ref_bf = torch.full((b, s, h, d), 1.0, device='cuda', dtype=torch.bfloat16)
v_kmajor_bf = v_ref_bf.permute(0, 3, 2, 1).contiguous().permute(0, 3, 2, 1)
print(f"v_kmajor_bf shape={v_kmajor_bf.shape} strides={v_kmajor_bf.stride()}")

# Create FP4 cute tensor with bench's helper
v_tensor, v_torch_underlying = cute_tensor_like(
    v_kmajor_bf, cutlass.Float4E2M1FN, is_dynamic_layout=True, assumed_align=16
)
v_stride_order = tuple(v_kmajor_bf.dim_order())
v_tensor.mark_compact_shape_dynamic(mode=1, stride_order=v_stride_order, divisibility=32)
print(f"v_tensor shape={v_tensor.shape} stride={v_tensor.stride}")
print(f"v_torch_underlying shape={v_torch_underlying.shape} strides={v_torch_underlying.stride()} dtype={v_torch_underlying.dtype}")

# Fill underlying buffer with FP4 bytes for V=1.0 (nibble 2 = 0x22)
v_torch_underlying.fill_(0x22)

# Q, K = RANDOM via bench's helper (so all three tensors are cute Tensors)
torch.manual_seed(0)
q_ref_bf = torch.randn(b, s, h, d, device='cuda', dtype=torch.bfloat16)
k_ref_bf = torch.randn(b, s, h, d, device='cuda', dtype=torch.bfloat16)
from cutlass.torch import convert_cute_tensor
q_tensor, q_torch_underlying = cute_tensor_like(q_ref_bf, cutlass.Float4E2M1FN, is_dynamic_layout=True, assumed_align=16)
k_tensor, k_torch_underlying = cute_tensor_like(k_ref_bf, cutlass.Float4E2M1FN, is_dynamic_layout=True, assumed_align=16)
q_tensor.mark_compact_shape_dynamic(mode=1, stride_order=tuple(q_ref_bf.dim_order()), divisibility=32)
k_tensor.mark_compact_shape_dynamic(mode=1, stride_order=tuple(k_ref_bf.dim_order()), divisibility=32)
q_tensor = convert_cute_tensor(q_ref_bf, q_tensor, cutlass.Float4E2M1FN, is_dynamic_layout=True)
k_tensor = convert_cute_tensor(k_ref_bf, k_tensor, cutlass.Float4E2M1FN, is_dynamic_layout=True)
q_fp4 = q_tensor
k_fp4 = k_tensor

# All SFs = 1.0 (E4M3 0x3C)
_, _, q_sf = create_scale_factor_tensor(b, s, h, d, 16, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN, sf_value=1.0)
_, _, k_sf = create_scale_factor_tensor(b, s, h, d, 16, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN, sf_value=1.0)
_, _, v_sf = create_scale_factor_tensor(b, s, h, d, 16, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN, sf_value=1.0)
torch.cuda.synchronize()

from flash_attn.cute.interface import _flash_attn_fwd
out, _ = _flash_attn_fwd(
    q_fp4, k_fp4, v_tensor,
    softmax_scale=None, causal=False,
    window_size_left=None, window_size_right=None,
    softcap=0.0, num_splits=1, pack_gqa=None,
    mSFQ=q_sf, mSFK=k_sf, mSFV=v_sf,
)
print(f"out[0,0,0,:8]={out[0,0,0,:8].tolist()}")
print(f"out range=[{out.min():.4f}, {out.max():.4f}], mean={out.mean():.4f}")
print(f"expected ~1.0 (V=1.0)")
