#!/bin/bash
# SASS analysis helper for FP4 FA4 cubins.
# Usage: ./analyze_sass.sh [variantB|variantC|both]
#
# Generates annotated SASS dumps and instruction count summaries.

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

do_analysis() {
    local name="$1"
    local cubin="$2"

    if [ ! -f "$cubin" ]; then
        echo "ERROR: $cubin not found. Run generate_cubins.sh first."
        return 1
    fi

    echo "=== $name ==="

    # Resource usage
    echo "--- Resource usage ---"
    cuobjdump --dump-resource-usage "$cubin" 2>/dev/null | grep -E 'REG|STACK|insns'

    # Annotated SASS with source line info
    local sass_file="$DIR/${name}_annotated.sass"
    nvdisasm -g -sf "$cubin" > "$sass_file" 2>/dev/null
    echo "Annotated SASS: $sass_file ($(wc -l < "$sass_file") lines)"

    # Live register counts
    local lrm_file="$DIR/${name}_lrm.txt"
    nvdisasm -lrm count "$cubin" > "$lrm_file" 2>/dev/null
    echo "Register counts: $lrm_file"

    # Key instruction counts
    echo "--- Key instruction counts ---"
    for inst in "MUFU.EX2" "F2FP" "FADD2" "FFMA2" "FMUL2" "UTCOMMA" "SYNCS.ARRIVE" "FENCE.VIEW" "STTM"; do
        count=$(grep -c "$inst" "$sass_file" 2>/dev/null || echo 0)
        printf "  %-20s %d\n" "$inst" "$count"
    done

    # Find barrier arrive PCs
    echo "--- Barrier arrives ---"
    grep 'SYNCS.ARRIVE' "$sass_file" | grep -v '\.byte' | head -20

    echo ""
}

case "${1:-both}" in
    variantB|B|b)
        do_analysis "variantB" "$DIR/variantB_before_barriers_nomult.cubin"
        ;;
    variantC|C|c)
        do_analysis "variantC" "$DIR/variantC_before_barriers_withmult.cubin"
        ;;
    both|*)
        do_analysis "variantB" "$DIR/variantB_before_barriers_nomult.cubin"
        do_analysis "variantC" "$DIR/variantC_before_barriers_withmult.cubin"
        ;;
esac
