"""Profile script for NCU: NVFP4 QK + NVFP4 PV vs NVFP4 QK + BF16 PV.
Run with: ncu --set full -o profile_fp4pv python3 -m flash_attn.cute.benchmarks.profile_fp4_pv
"""
import torch
import os
import sys

# Disable compile cache to ensure fresh compilation
os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "0")

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.benchmarks.bench_fp4 import create_nvfp4_attention_tensors
from flashinfer.quantization import nvfp4_quantize, SfLayout
import torch.nn.functional as F

def q4(t):
    b, s, h, d = t.shape
    sp = (s + 127) // 128 * 128
    if sp != s:
        t = F.pad(t, (0, 0, 0, 0, 0, sp - s))
    t2 = t.reshape(b * sp, h * d)
    fp4, sf = nvfp4_quantize(t2, torch.ones(1, device=t.device, dtype=torch.float32),
                              sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    fp4 = fp4.reshape(b, sp, h, d // 2).view(torch.int8).view(torch.float4_e2m1fn_x2)
    rest_m = sp // 128; sf_k = d // 16; rest_k = sf_k // 4
    sf = sf.reshape(b * rest_m, (h * sf_k) // 4, 32, 4, 4)
    sf = sf.reshape(b, rest_m, h, rest_k, 32, 4, 4).permute(0, 2, 1, 3, 4, 5, 6).contiguous().permute(4, 5, 2, 6, 3, 1, 0)
    return fp4[:, :s], sf

mode = sys.argv[1] if len(sys.argv) > 1 else "bf16pv"
b, s, h, d = 1, 4096, 24, 128
torch.manual_seed(42)
q = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)

qf, qs = q4(q)
kf, ks = q4(k)

if mode == "fp4pv":
    qfp4, kfp4, vfp4, qsf, ksf, vsf, _, _, _ = create_nvfp4_attention_tensors(
        b, s, s, h, h, d, d, pv_mode="fp4")
    # Warmup
    _flash_attn_fwd(qfp4, kfp4, vfp4, softmax_scale=d**-0.5, causal=False,
                    mSFQ=qsf, mSFK=ksf, mSFV=vsf)
    torch.cuda.synchronize()
    # Profiled run
    torch.cuda.cudart().cudaProfilerStart()
    _flash_attn_fwd(qfp4, kfp4, vfp4, softmax_scale=d**-0.5, causal=False,
                    mSFQ=qsf, mSFK=ksf, mSFV=vsf)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"Profiled NVFP4+FP4PV: b={b} s={s} h={h} d={d}")
else:
    # Warmup
    _flash_attn_fwd(qf, kf, v, softmax_scale=d**-0.5, causal=False, mSFQ=qs, mSFK=ks)
    torch.cuda.synchronize()
    # Profiled run
    torch.cuda.cudart().cudaProfilerStart()
    _flash_attn_fwd(qf, kf, v, softmax_scale=d**-0.5, causal=False, mSFQ=qs, mSFK=ks)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"Profiled NVFP4+BF16PV: b={b} s={s} h={h} d={d}")
