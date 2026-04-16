"""Minimal test for FP8 QK + BF16 PV in pr2109 kernel."""
import os
import sys
import traceback

sys.path.insert(0, "/tmp/pr2109-mixed-dtype")

import torch
import torch.nn.functional as F


def run(q_dtype, v_dtype, shape=(1, 4096, 24, 128), check=True):
    from flash_attn.cute.interface import flash_attn_func
    b, s, h, d = shape
    device = "cuda"
    torch.manual_seed(0)
    q_bf = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16) * 0.3
    k_bf = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16) * 0.3
    v_bf = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16) * 0.3
    q = q_bf.to(q_dtype)
    k = k_bf.to(q_dtype)
    v = v_bf.to(v_dtype)
    res = flash_attn_func(q, k, v, causal=False, softmax_scale=1.0 / (d ** 0.5))
    out = res[0] if isinstance(res, tuple) else res
    torch.cuda.synchronize()
    if check:
        # Reference: bf16 SDPA using the requantized tensors (so dtype-cast quant is in ref too).
        qr = q.to(torch.bfloat16)
        kr = k.to(torch.bfloat16)
        vr = v.to(torch.bfloat16)
        ref = F.scaled_dot_product_attention(
            qr.transpose(1, 2), kr.transpose(1, 2), vr.transpose(1, 2),
            is_causal=False, scale=1.0 / (d ** 0.5),
        ).transpose(1, 2).to(torch.bfloat16)
        cos = F.cosine_similarity(out.float().reshape(-1), ref.float().reshape(-1), dim=0).item()
        maxabs = (out.float() - ref.float()).abs().max().item()
        return out, cos, maxabs
    return out, None, None


def main():
    configs = [
        ("BF16", torch.bfloat16, torch.bfloat16),
        ("FP8/FP8", torch.float8_e4m3fn, torch.float8_e4m3fn),
        ("FP8/BF16", torch.float8_e4m3fn, torch.bfloat16),
    ]
    shapes = [(1, 1024, 4, 128), (1, 4096, 16, 128), (1, 32768, 24, 128)]
    for shape in shapes:
        print(f"\nshape={shape}")
        for label, qd, vd in configs:
            try:
                out, cos, maxabs = run(qd, vd, shape=shape)
                print(f"  {label:10s} cos={cos:.4f} max_abs={maxabs:.4f}")
            except Exception as e:
                print(f"  {label:10s} FAIL: {type(e).__name__}: {str(e).splitlines()[0]}")


if __name__ == "__main__":
    main()
