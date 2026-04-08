"""V quantize: bench's V FP4 buffer + SF in kernel's tile_to_shape byte layout."""
import sys, importlib.util
sys.path.insert(0, '/sgl-workspace/FastVideo')
spec = importlib.util.spec_from_file_location(
    'fac', '/sgl-workspace/FastVideo/fastvideo/attention/utils/flash_attn_cute.py'
)
fac = importlib.util.module_from_spec(spec); sys.modules['fac'] = fac
spec.loader.exec_module(fac)

import torch
import cutlass
from cutlass.torch import cute_tensor_like, convert_cute_tensor
from flashinfer.quantization import nvfp4_quantize, SfLayout
from flash_attn.cute.benchmarks.bench_fp4 import create_scale_factor_tensor


def quantize_qk_via_bench(t):
    """Use bench's convert_cute_tensor for Q/K. Returns (cute_tensor, sf_tensor)."""
    b, s, h, d = t.shape
    cute_t, _ = cute_tensor_like(t, cutlass.Float4E2M1FN, is_dynamic_layout=True, assumed_align=16)
    cute_t.mark_compact_shape_dynamic(mode=1, stride_order=tuple(t.dim_order()), divisibility=32)
    cute_t = convert_cute_tensor(t.float(), cute_t, cutlass.Float4E2M1FN, is_dynamic_layout=True)
    _, _, sf = create_scale_factor_tensor(b, s, h, d, 16, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN, sf_value=1.0)
    return cute_t, sf


def quantize_v_kernel_layout(v):
    """V quantize using bench cute Tensor + KERNEL-correct SF layout.

    1. Use bench's cute_tensor_like + convert_cute_tensor for V FP4 buffer.
    2. Quantize via nvfp4_quantize with (b*h*d, s) reshape for per-block SFs.
    3. Allocate SF as a flat 1MB buffer + view as (32, 4, rest_m, 4, rest_k, h, b)
       with KERNEL strides (16, 4, ., 1, 512, 65536, h*65536).
    4. Write SF bytes from nvfp4_quantize output (which is already in the correct
       byte layout because layout_128x4 with (b*h*d, s) reshape produces bytes that
       match the kernel's tile_to_shape byte order).
    """
    b, s, h, d = v.shape
    # Per-(b, h, d, s_block_of_16) max along s. Block of 16 in K dim.
    # Reshape v to (b, h, d, s/16, 16) and take abs().max() over last dim.
    v_perm = v.permute(0, 2, 3, 1).contiguous().float()  # (b, h, d, s)
    v_blocks = v_perm.reshape(b, h, d, s // 16, 16)
    block_max = v_blocks.abs().amax(dim=-1).clamp(min=1e-6)  # (b, h, d, s/16)
    sf_f32 = block_max / 6.0  # SF s.t. quant range is V/SF in [-6, 6]
    # Convert to E4M3 (Float8E4M3FN). Use round-to-nearest then bit-pack.
    sf_e4m3 = sf_f32.to(torch.float8_e4m3fn)  # torch supports this
    # Reconstruct dequantized SF for proper scaling
    sf_real = sf_e4m3.float()
    # Scale V by 1/sf_real along the matching block
    sf_real_full = sf_real.unsqueeze(-1).expand(b, h, d, s // 16, 16).reshape(b, h, d, s)
    v_scaled = v_perm / sf_real_full  # (b, h, d, s)
    v_scaled_kmajor = v_scaled.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1)  # back to (b, s, h, d) K-major
    # Wait, simpler: just scale the original v
    v_scaled_orig = (v.float() / sf_real.permute(0, 3, 1, 2).unsqueeze(-1).expand(-1, -1, -1, -1, 16).reshape(b, s // 16 * 16, h, d))
    # That's getting messy. Just use v_scaled (b, h, d, s) and permute back to (b, s, h, d).
    v_scaled_bshd = v_scaled.permute(0, 3, 1, 2).contiguous()  # (b, s, h, d)
    # K-major bench layout
    v_kmajor = v_scaled_bshd.to(torch.bfloat16).permute(0, 3, 2, 1).contiguous().permute(0, 3, 2, 1)
    v_t, v_und = cute_tensor_like(v_kmajor, cutlass.Float4E2M1FN, is_dynamic_layout=True, assumed_align=16)
    v_t.mark_compact_shape_dynamic(mode=1, stride_order=tuple(v_kmajor.dim_order()), divisibility=32)
    v_t = convert_cute_tensor(v_kmajor.float(), v_t, cutlass.Float4E2M1FN, is_dynamic_layout=True)

    # Now build SF tensor matching kernel's tile_to_shape byte layout.
    # Need byte at (d_inner=d%32, d_outer=(d//32)%4, rest_m_idx=d//128, sf_k_inner=(s//16)%4,
    # rest_k_idx=s//64, h, b) = sf_real[b, h, d, s//16] (in E4M3 byte form)
    sf_e4m3_bytes = sf_e4m3.view(torch.uint8)  # (b, h, d, s/16)
    sf_data = sf_e4m3_bytes

    # 3+4) Build SF tensor in kernel layout: shape (32, 4, rest_m=ceil(d,128), 4,
    # rest_k=ceil(s,64), h, b). Strides per the tile_to_shape probe:
    # ((16, 4), 0), ((0, 1), 512), (0, 65536), (0, h*65536)
    rest_m = max((d + 127) // 128, 1)
    rest_k = (s + 63) // 64
    atom_bytes = 512
    h_stride = rest_m * rest_k * atom_bytes  # rest_m=1 → 65536
    b_stride = h * h_stride
    target_shape = (32, 4, rest_m, 4, rest_k, h, b)
    target_stride = (16, 4, rest_k * atom_bytes, 1, atom_bytes, h_stride, b_stride)
    flat = sf_data.view(torch.uint8).contiguous().flatten()
    print(f"[v] sf flat size={flat.numel()} expected={b * b_stride}")
    sf_tensor = flat.as_strided(target_shape, target_stride)
    return v_t, sf_tensor


torch.manual_seed(0)
b, s, h, d = 1, 8192, 16, 128

# All-constant first
v_const = torch.full((b, s, h, d), 1.0, device='cuda', dtype=torch.bfloat16)
q_const = torch.full((b, s, h, d), 1.0, device='cuda', dtype=torch.bfloat16)
k_const = torch.full((b, s, h, d), 1.0, device='cuda', dtype=torch.bfloat16)

q_t, q_sf = quantize_qk_via_bench(q_const)
k_t, k_sf = quantize_qk_via_bench(k_const)
v_t, v_sf = quantize_v_kernel_layout(v_const)

from flash_attn.cute.interface import _flash_attn_fwd
out, _ = _flash_attn_fwd(
    q_t, k_t, v_t,
    softmax_scale=None, causal=False,
    window_size_left=None, window_size_right=None,
    softcap=0.0, num_splits=1, pack_gqa=None,
    mSFQ=q_sf, mSFK=k_sf, mSFV=v_sf,
)
print(f"[const] out range=[{out.min():.4f}, {out.max():.4f}], mean={out.mean():.4f} (expect ~1.0)")
