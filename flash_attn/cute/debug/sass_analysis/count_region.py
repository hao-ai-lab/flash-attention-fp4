#!/usr/bin/env python3
"""Count instructions by type in a PC range of a cubin.

Usage:
    python count_region.py <cubin> <start_pc_hex> <end_pc_hex>
    python count_region.py variantC_before_barriers_withmult.cubin 8ef0 9c00

Counts MUFU.EX2, F2FP, FADD2, FFMA2, FMUL2, and total instructions between
start_pc (exclusive) and end_pc (exclusive).
"""
import subprocess
import sys
import re
from collections import Counter

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    cubin = sys.argv[1]
    start_pc = int(sys.argv[2], 16)
    end_pc = int(sys.argv[3], 16)

    # Optional: split at intermediate PCs
    splits = [int(x, 16) for x in sys.argv[4:]]
    boundaries = sorted([start_pc] + splits + [end_pc])

    result = subprocess.run(
        ['nvdisasm', '-g', '-sf', cubin],
        capture_output=True, text=True
    )

    # Parse instructions
    instructions = []
    for line in result.stdout.splitlines():
        m = re.match(r'\s+/\*([0-9a-f]+)\*/\s+(.+)', line)
        if not m:
            continue
        if '.byte' in line:
            continue
        pc = int(m.group(1), 16)
        inst = m.group(2).strip()
        instructions.append((pc, inst))

    TRACKED = ['MUFU.EX2', 'F2FP', 'FADD2', 'FFMA2', 'FMUL2', 'STTM', 'STS', 'SYNCS.ARRIVE', 'FENCE']

    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        region_insns = [(pc, inst) for pc, inst in instructions if lo < pc < hi]
        counts = Counter()
        for pc, inst in region_insns:
            for t in TRACKED:
                if t in inst:
                    counts[t] += 1
        total = len(region_insns)
        print(f"\n--- 0x{lo:x} → 0x{hi:x} ({total} insns) ---")
        for t in TRACKED:
            if counts[t] > 0:
                print(f"  {t:20s} {counts[t]}")

if __name__ == '__main__':
    main()
