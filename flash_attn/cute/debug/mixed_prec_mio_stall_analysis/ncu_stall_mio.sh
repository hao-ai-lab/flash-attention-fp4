#!/usr/bin/env bash
# Profile a flash attention kernel and extract top MIO-stall instructions.
#
# Usage:
#   bash ncu_stall_mio.sh <mode> [b s h d]
#
# Modes (our kernel):
#   bf16_bf16  fp8_fp8  nvfp4_bf16  nvfp4_fp8  mxfp8_bf16  mxfp8_fp8
#
# Examples:
#   bash ncu_stall_mio.sh nvfp4_fp8
#   bash ncu_stall_mio.sh nvfp4_fp8 1 32768 24 128
#   bash ncu_stall_mio.sh nvfp4_bf16

set -euo pipefail

MODE=${1:?Usage: $0 <mode> [b s h d]}
B=${2:-1}; S=${3:-32768}; H=${4:-24}; D=${5:-128}
REPORT="/tmp/ncu_stall_${MODE}.ncu-rep"
SCRIPT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"

export CUTE_DSL_ENABLE_TVM_FFI=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=0
export FA4_ALLOW_PURE_FP8_QK=1

# ---------- single-shot profiling script ----------
PROFILE_PY=$(mktemp /tmp/ncu_profile_XXXXX.py)
cat > "$PROFILE_PY" << 'PYEOF'
import sys, os, torch
sys.path.insert(0, os.environ["FA_ROOT"])
os.environ.setdefault("CUTE_DSL_ENABLE_TVM_FFI", "1")
os.environ.setdefault("FA4_ALLOW_PURE_FP8_QK", "1")

from flash_attn.cute.interface import flash_attn_func

mode = sys.argv[1]
b, s, h, d = (int(x) for x in sys.argv[2:6])
torch.manual_seed(0)

if mode in ("bf16_bf16", "fp8_fp8", "fp8_bf16"):
    q_bf = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16) * 0.3
    k_bf = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16) * 0.3
    v_bf = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16) * 0.3
    qk_str, pv_str = mode.split("_")
    qd = torch.float8_e4m3fn if qk_str == "fp8" else torch.bfloat16
    vd = torch.float8_e4m3fn if pv_str == "fp8" else torch.bfloat16
    q, k, v = q_bf.to(qd), k_bf.to(qd), v_bf.to(vd)
    def fn(): flash_attn_func(q, k, v, causal=False)
else:
    import cutlass
    from flash_attn.cute.benchmarks.bench_fp4 import create_blockscaled_attention_tensors
    qk_str, pv_str = mode.split("_")
    if qk_str == "nvfp4":
        ab, sf, sfv = cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16
    else:
        ab, sf, sfv = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32
    q, kk, v, qsf, ksf, vsf, *_ = create_blockscaled_attention_tensors(
        b, s, s, h, h, d, d, pv_mode=pv_str, return_torch=False,
        ab_dtype=ab, sf_dtype=sf, sf_vec_size=sfv,
        pv_fp8_dtype=cutlass.Float8E4M3FN)
    def fn(): flash_attn_func(q, kk, v, mSFQ=qsf, mSFK=ksf, mSFV=vsf, causal=False)

for _ in range(3):
    fn()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
fn()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
PYEOF

# ---------- run NCU ----------
echo ">>> Profiling ${MODE} (${B},${S},${H},${D}) → ${REPORT}"
FA_ROOT="$SCRIPT_DIR" ncu \
    --profile-from-start off \
    --target-processes all \
    --launch-count 1 \
    --kernel-name regex:"flash_fwd" \
    --set full \
    -f -o "${REPORT}" \
    python "$PROFILE_PY" "$MODE" "$B" "$S" "$H" "$D"

rm -f "$PROFILE_PY"

# ---------- extract stall_mio ----------
echo ""
echo ">>> Top stall_mio instructions:"
ncu -i "${REPORT}" --page source --csv 2>/dev/null | python3 -c "
import csv, sys
lines = sys.stdin.readlines()
data = [l for l in lines if l.startswith('\"0x') or l.startswith('\"Address')]
reader = csv.DictReader(data)
rows = []
for r in reader:
    mio = int(r.get('stall_mio', '0') or '0')
    if mio > 0:
        rows.append((mio, r.get('Source', '').strip(), r.get('Address', '')))
rows.sort(reverse=True)
total = sum(r[0] for r in rows)

# per-opcode summary
from collections import defaultdict
by_op = defaultdict(int)
for mio, src, _ in rows:
    op = src.split()[0] if src else '?'
    by_op[op] += mio

print(f'Total stall_mio samples: {total}')
print()
print('By instruction class:')
for op, cnt in sorted(by_op.items(), key=lambda x: -x[1]):
    print(f'  {op:40s} {cnt:6d} ({cnt*100//total:3d}%)')

print()
print('Top 30 individual instructions:')
for mio, src, addr in rows[:30]:
    print(f'  stall_mio={mio:5d}  {src}')
"

echo ""
echo ">>> Report saved to ${REPORT}"
echo "    Open in Nsight Compute GUI: ncu-ui ${REPORT}"
