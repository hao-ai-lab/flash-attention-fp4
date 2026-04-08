"""V quantize: storage (b, d, h, s) row-major matches bench K-major V layout."""
import sys, importlib.util
sys.path.insert(0, '/sgl-workspace/FastVideo')
spec = importlib.util.spec_from_file_location(
    'fac', '/sgl-workspace/FastVideo/fastvideo/attention/utils/flash_attn_cute.py'
)
fac = importlib.util.module_from_spec(spec); sys.modules['fac'] = fac
spec.loader.exec_module(fac)

import torch
from flashinfer.quantization import nvfp4_quantize, SfLayout


def quantize_qk(t):
    b, s, h, d = t.shape
    one = torch.ones(1, device=t.device, dtype=torch.float32)
    fp4_data, sf_data = nvfp4_quantize(
        t.reshape(b * s, h * d), one,
        sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    fp4 = fp4_data.reshape(b, s, h, d // 2).view(torch.int8).view(torch.float4_e2m1fn_x2)
    rest_m = s // 128
    sf_kph = d // 16
    rest_k = sf_kph // 4
    total_m = b * rest_m
    total_k = (h * sf_kph) // 4
    sw = sf_data.reshape(total_m, total_k, 32, 4, 4)
    de = sw.reshape(b, rest_m, h, rest_k, 32, 4, 4)
    can = de.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
    return fp4, can.permute(4, 5, 2, 6, 3, 1, 0)


def quantize_v(v):
    """V K-major: SF needs `(b, h, d, s)` reshape so each 128-row m_block
    covers one (h) slice with d as the inner row index. The FP4 buffer is then
    reshaped + transposed into (b, d, h, s/2) physical storage to match the
    kernel's K-major V layout (order=(3,0,1,2)).
    """
    b, s, h, d = v.shape
    # (b, h, d, s) reshape so m_block_idx == h_idx
    v_perm = v.permute(0, 2, 3, 1).contiguous()  # (b, h, d, s)
    one = torch.ones(1, device=v.device, dtype=torch.float32)
    fp4_data, sf_data = nvfp4_quantize(
        v_perm.reshape(b * h * d, s), one,
        sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    # fp4_data is shape (b*h*d, s/2) in (b, h, d) row order. Reshape to
    # (b, h, d, s/2), then permute to (b, d, h, s/2) and force contiguous so
    # storage matches the kernel's V K-major layout (b,s,h,d) with strides
    # (s*h*d, 1, s, s*h).
    fp4_bhds = fp4_data.reshape(b, h, d, s // 2)
    fp4_buf = fp4_bhds.permute(0, 2, 1, 3).contiguous()  # (b, d, h, s/2)
    fp4_view = fp4_buf.view(torch.int8).view(torch.float4_e2m1fn_x2)

    # SF: build (32, 4, rest_m, 4, rest_k, h, b) view of the flat bytes.
    rest_m = max(d // 128, 1)
    rest_k = s // 64
    atom_bytes = 512
    m_block_stride = rest_k * atom_bytes
    h_stride = rest_m * m_block_stride
    b_stride = h * h_stride
    target_shape = (32, 4, rest_m, 4, rest_k, h, b)
    target_stride = (16, 4, m_block_stride, 1, atom_bytes, h_stride, b_stride)
    flat = sf_data.view(torch.uint8).contiguous().flatten()
    sf_tensor = flat.as_strided(target_shape, target_stride)
    return fp4_view, sf_tensor


torch.manual_seed(0)
b, s, h, d = 1, 8192, 16, 128
q_bf = torch.randn(b, s, h, d, device='cuda', dtype=torch.bfloat16)
k_bf = torch.randn(b, s, h, d, device='cuda', dtype=torch.bfloat16)
v_bf = torch.randn(b, s, h, d, device='cuda', dtype=torch.bfloat16)

q_fp4, q_sf = quantize_qk(q_bf)
k_fp4, k_sf = quantize_qk(k_bf)
v_fp4, v_sf = quantize_v(v_bf)
print(f"v_fp4 shape={v_fp4.shape} stride={v_fp4.stride()}")
print(f"v_sf shape={v_sf.shape} stride={v_sf.stride()}")

from flash_attn.cute.interface import _flash_attn_fwd, flash_attn_func
out, _ = _flash_attn_fwd(
    q_fp4, k_fp4, v_fp4,
    softmax_scale=None, causal=False,
    window_size_left=None, window_size_right=None,
    softcap=0.0, num_splits=1, pack_gqa=None,
    mSFQ=q_sf, mSFK=k_sf, mSFV=v_sf,
)
_r = flash_attn_func(q_bf, k_bf, v_bf)
out_bf16 = _r[0] if isinstance(_r, tuple) else _r


def metrics(a, bb):
    diff = (a - bb).abs().float()
    a_, b_ = a.float().flatten(), bb.float().flatten()
    cos = (a_ @ b_) / (a_.norm() * b_.norm() + 1e-9)
    return f"max={diff.max():.4f} mean={diff.mean():.6f} cos={cos:.4f} norm_a={a_.norm():.2f} norm_b={b_.norm():.2f}"


print(f"QKVP vs BF16: {metrics(out, out_bf16)}")

# Constant V sanity (V=0.5 → expected ~0.5156)
v_const = torch.full((b, s, h, d), 0.5, device='cuda', dtype=torch.bfloat16)
v_fp4c, v_sfc = quantize_v(v_const)
out_c, _ = _flash_attn_fwd(
    q_fp4, k_fp4, v_fp4c,
    softmax_scale=None, causal=False,
    window_size_left=None, window_size_right=None,
    softcap=0.0, num_splits=1, pack_gqa=None,
    mSFQ=q_sf, mSFK=k_sf, mSFV=v_sfc,
)
print(f"V=0.5 const: out[0,0,0,0]={out_c[0,0,0,0]:.4f} (expect ~0.516)")
print(f"V=0.5 const: out range=[{out_c.min():.4f}, {out_c.max():.4f}], mean={out_c.mean():.4f}")
