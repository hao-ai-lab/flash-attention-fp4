"""Everything constant test - QKVP."""
import sys, importlib.util
sys.path.insert(0, '/sgl-workspace/FastVideo')
spec = importlib.util.spec_from_file_location(
    'fac', '/sgl-workspace/FastVideo/fastvideo/attention/utils/flash_attn_cute.py'
)
fac = importlib.util.module_from_spec(spec); sys.modules['fac'] = fac
spec.loader.exec_module(fac)

import torch
import cutlass
from flash_attn.cute.benchmarks.bench_fp4 import create_scale_factor_tensor

b, s, h, d = 1, 8192, 16, 128

# All Q, K, V = constant 1.0 in BF16, then represented as FP4 + SF.
# Q = K = 1.0, V = 0.5
# QK^T elements = sum(Q*K) = d * 1 = 128. After scale = 128/sqrt(128) = sqrt(128) ≈ 11.3
# Softmax(uniform) gives 1/s. Output = sum(softmax * V) = V = 0.5.

# Build Q, K, V FP4 buffers as all constant FP4 1.0 = nibble 2 = 0x22 byte
# FP4 nibble values: 0=+0, 1=+0.5, 2=+1.0, 3=+1.5, 4=+2, 5=+3, 6=+4, 7=+6
# To get FP4 element 1.0, nibble = 2 → byte 0x22
q_fp4_buf = torch.full((b, s, h, d // 2), 0x22, device='cuda', dtype=torch.uint8)
k_fp4_buf = torch.full((b, s, h, d // 2), 0x22, device='cuda', dtype=torch.uint8)
# V = 0.5 → nibble 1 → 0x11
v_fp4_buf = torch.full((b, d, h, s // 2), 0x44, device='cuda', dtype=torch.uint8)  # V=2.0

q_fp4 = q_fp4_buf.view(torch.float4_e2m1fn_x2)
k_fp4 = k_fp4_buf.view(torch.float4_e2m1fn_x2)
v_fp4 = v_fp4_buf.view(torch.float4_e2m1fn_x2)

# All SFs = 1.0 (E4M3 0x3C)
_, _, q_sf = create_scale_factor_tensor(b, s, h, d, 16, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN, sf_value=1.0)
_, _, k_sf = create_scale_factor_tensor(b, s, h, d, 16, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN, sf_value=1.0)
_, _, v_sf = create_scale_factor_tensor(b, s, h, d, 16, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN, sf_value=1.0)
q_sf.fill_(0x3C); k_sf.fill_(0x3C); v_sf.fill_(0x3C)
torch.cuda.synchronize()

from flash_attn.cute.interface import _flash_attn_fwd
out, _ = _flash_attn_fwd(
    q_fp4, k_fp4, v_fp4,
    softmax_scale=None, causal=False,
    window_size_left=None, window_size_right=None,
    softcap=0.0, num_splits=1, pack_gqa=None,
    mSFQ=q_sf, mSFK=k_sf, mSFV=v_sf,
)
print(f"out[0,0,0,:8]={out[0,0,0,:8].tolist()}")
print(f"out range=[{out.min():.4f}, {out.max():.4f}], mean={out.mean():.4f}")
print(f"expected ~0.5")
