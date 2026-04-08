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
    # Allocate bench-style int8 V buffer (16 MB; cute interprets as packed FP4 in first 8 MB).
    v_kmajor_dummy = v.permute(0, 3, 2, 1).contiguous().permute(0, 3, 2, 1)
    v_t, v_und = cute_tensor_like(v_kmajor_dummy, cutlass.Float4E2M1FN, is_dynamic_layout=True, assumed_align=16)
    v_t.mark_compact_shape_dynamic(mode=1, stride_order=tuple(v_kmajor_dummy.dim_order()), divisibility=32)

    # nvfp4_quantize on (b*h*d, s) → packed FP4 (b*h*d, s/2) with adjacent s in adjacent
    # nibbles, AND SF in layout_128x4 byte order matching kernel's tile_to_shape result.
    v_perm = v.permute(0, 2, 3, 1).contiguous()  # (b, h, d, s)
    one = torch.ones(1, device=v.device, dtype=torch.float32)
    fp4_packed, sf_data = nvfp4_quantize(
        v_perm.reshape(b * h * d, s), one,
        sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    # fp4_packed shape (b*h*d, s/2). Reshape to (b, h, d, s/2), permute to (b, d, h, s/2)
    # to match bench's int8 V buffer's underlying physical layout.
    fp4_bhd = fp4_packed.view(torch.uint8).reshape(b, h, d, s // 2)
    fp4_bdh = fp4_bhd.permute(0, 2, 1, 3).contiguous()  # (b, d, h, s/2)
    # Write into the FIRST 8 MB of v_und (cute reads packed 2 FP4/byte from there).
    v_und_flat = v_und.view(torch.int8).flatten()
    v_und_flat.zero_()
    v_und_flat[:fp4_bdh.numel()].copy_(fp4_bdh.flatten().to(torch.int8))
    print(f"[v] wrote {fp4_bdh.numel()} bytes; first 16: {[hex(x & 0xFF) for x in v_und_flat[:16].cpu().tolist()]}")

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
