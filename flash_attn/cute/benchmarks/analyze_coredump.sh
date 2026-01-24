#!/bin/bash
# Script to run benchmark with CUDA core dump enabled and analyze the core dump
# Usage: 
#   ./analyze_coredump.sh --run <benchmark_script> [benchmark_args] [--output output_file]
#   ./analyze_coredump.sh <coredump_file> [output_file]

set -e

# Parse arguments
RUN_BENCHMARK=false
BENCHMARK_SCRIPT=""
BENCHMARK_ARGS=()
COREDUMP_FILE=""
OUTPUT_FILE="coredump_analysis.txt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run)
            RUN_BENCHMARK=true
            shift
            if [ $# -eq 0 ]; then
                echo "Error: --run requires a benchmark script"
                exit 1
            fi
            BENCHMARK_SCRIPT="$1"
            shift
            # Collect remaining arguments as benchmark args
            while [[ $# -gt 0 && "$1" != "--output" ]]; do
                BENCHMARK_ARGS+=("$1")
                shift
            done
            ;;
        --output)
            shift
            if [ $# -eq 0 ]; then
                echo "Error: --output requires a filename"
                exit 1
            fi
            OUTPUT_FILE="$1"
            shift
            ;;
        *)
            if [ -z "$COREDUMP_FILE" ]; then
                COREDUMP_FILE="$1"
            elif [ "$OUTPUT_FILE" = "coredump_analysis.txt" ]; then
                OUTPUT_FILE="$1"
            fi
            shift
            ;;
    esac
done

# If running benchmark, set up CUDA core dump and run
if [ "$RUN_BENCHMARK" = true ]; then
    if [ -z "$BENCHMARK_SCRIPT" ]; then
        echo "Error: Benchmark script not specified"
        echo "Usage: $0 --run <benchmark_script> [benchmark_args] [--output output_file]"
        exit 1
    fi
    
    if [ ! -f "$BENCHMARK_SCRIPT" ]; then
        echo "Error: Benchmark script not found: $BENCHMARK_SCRIPT"
        exit 1
    fi
    
    echo "=========================================="
    echo "Running benchmark with CUDA core dump enabled"
    echo "=========================================="
    echo "Benchmark: $BENCHMARK_SCRIPT"
    echo "Arguments: ${BENCHMARK_ARGS[@]}"
    echo ""
    
    # Set up CUDA core dump environment variables
    export NVCC_PREPEND_FLAGS='-lineinfo'
    export CUTE_DSL_ENABLE_TVM_FFI=1
    export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
    export CUDA_COREDUMP_PIPE="/tmp/cuda_coredump_pipe_%h.%p.%t"
    export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
    export CUDA_COREDUMP_SHOW_PROGRESS=1
    export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
    export CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h.%p.%t"
    
    # Get hostname and current time for pattern matching
    HOSTNAME=$(hostname)
    TIMESTAMP_BEFORE=$(date +%s)
    
    echo "CUDA core dump will be saved to: /tmp/cuda_coredump_${HOSTNAME}.*"
    echo ""
    
    # Run the benchmark
    echo "Starting benchmark..."
    set +e  # Temporarily disable exit on error to capture exit code
    python "$BENCHMARK_SCRIPT" "${BENCHMARK_ARGS[@]}"
    BENCHMARK_EXIT_CODE=$?
    set -e  # Re-enable exit on error
    
    # Wait a bit for core dump to be generated
    echo ""
    echo "Waiting for core dump to be generated..."
    sleep 3
    
    # Find the most recent core dump matching the pattern
    TIMESTAMP_AFTER=$(date +%s)
    COREDUMP_PATTERN="/tmp/cuda_coredump_${HOSTNAME}.*"
    
    # Try to find core dumps created after TIMESTAMP_BEFORE
    # First try using find with timestamp (if supported)
    COREDUMP_FILE=$(find /tmp -maxdepth 1 -name "cuda_coredump_${HOSTNAME}.*" -type f -newermt "@${TIMESTAMP_BEFORE}" 2>/dev/null | sort -t. -k3 -n | tail -n1)
    
    # Fallback: find most recent core dump by modification time
    if [ -z "$COREDUMP_FILE" ] || [ ! -f "$COREDUMP_FILE" ]; then
        COREDUMP_FILE=$(ls -t /tmp/cuda_coredump_${HOSTNAME}.* 2>/dev/null | head -n1)
    fi
    
    # Additional fallback: find any recent core dump (within last 60 seconds)
    if [ -z "$COREDUMP_FILE" ] || [ ! -f "$COREDUMP_FILE" ]; then
        COREDUMP_FILE=$(find /tmp -maxdepth 1 -name "cuda_coredump_*" -type f -mmin -1 2>/dev/null | sort -t. -k3 -n | tail -n1)
    fi
    
    if [ -z "$COREDUMP_FILE" ] || [ ! -f "$COREDUMP_FILE" ]; then
        echo "Warning: No core dump file found. The benchmark may have completed successfully or crashed before CUDA could generate a core dump."
        echo "Looking for core dumps in /tmp/cuda_coredump_*..."
        ls -lah /tmp/cuda_coredump_* 2>/dev/null || echo "No core dumps found."
        if [ -n "$BENCHMARK_EXIT_CODE" ] && [ "$BENCHMARK_EXIT_CODE" -ne 0 ]; then
            echo "Benchmark exited with code: $BENCHMARK_EXIT_CODE"
            echo "This might be a host-side crash (not GPU-side), which CUDA core dumps won't capture."
        fi
        exit 1
    fi
    
    echo "Found core dump: $COREDUMP_FILE"
    echo ""
fi

# If no core dump file specified and not running benchmark, show usage
if [ -z "$COREDUMP_FILE" ]; then
    echo "Usage:"
    echo "  $0 --run <benchmark_script> [benchmark_args] [--output output_file]"
    echo "  $0 <coredump_file> [output_file]"
    echo ""
    echo "Examples:"
    echo "  $0 --run benchmarks/bench_fp4.py"
    echo "  $0 /tmp/cuda_coredump_1d7c3631cfa8.1897845.1768858059"
    exit 1
fi

# Find the most recent core dump if pattern is used
if [[ "$COREDUMP_FILE" == *"*"* ]]; then
    COREDUMP_FILE=$(ls -t ${COREDUMP_FILE} 2>/dev/null | head -n1)
fi

if [ ! -f "$COREDUMP_FILE" ]; then
    echo "Error: Core dump file not found: $COREDUMP_FILE"
    exit 1
fi

echo "Analyzing core dump: $COREDUMP_FILE"
echo "Output will be written to: $OUTPUT_FILE"

# Create temporary files for cuda-gdb commands
GDB_SCRIPT=$(mktemp)
DISASM_OUTPUT=$(mktemp)

# Cleanup function
cleanup() {
    rm -f "$GDB_SCRIPT" "$DISASM_OUTPUT"
    # Only remove extracted binary if it's in /tmp (our temporary file)
    if [ -n "$EXTRACTED_BINARY" ] && [[ "$EXTRACTED_BINARY" == /tmp/cuda_extracted_binary_* ]]; then
        rm -f "$EXTRACTED_BINARY"
    fi
}
trap cleanup EXIT

# Create cuda-gdb script for info extraction and binary/disassembly extraction
# We need to do everything in one session before the binary is cleaned up
cat > "$GDB_SCRIPT" << EOF
set pagination off
set print elements 0
target cudacore $COREDUMP_FILE
info symbol \$errorpc
print/x \$errorpc
quit
EOF

# Run cuda-gdb and capture output
echo "Running cuda-gdb to extract error information..."
GDB_OUTPUT=$(cuda-gdb -batch -x "$GDB_SCRIPT" 2>&1)

# Extract binary file path from cuda-gdb output BEFORE it's cleaned up
BINARY_FILE_PATH=$(echo "$GDB_OUTPUT" | grep -oP '/tmp/cuda-dbg/[^ ]+\.o\.[A-Za-z0-9]+' | head -n1)

# The binary file only exists while cuda-gdb is running
# We need to extract it using cuobjdump or get disassembly directly from cuda-gdb
EXTRACTED_BINARY=""
if [ -n "$BINARY_FILE_PATH" ]; then
    echo "Found binary path in cuda-gdb output: $BINARY_FILE_PATH"
    echo "Note: This path only exists while cuda-gdb is running"
    echo "Attempting to extract binary from core dump using cuobjdump..."
    
    EXTRACTED_BINARY=$(mktemp /tmp/cuda_extracted_binary_XXXXXX.o)
    # Try to extract using cuobjdump
    cuobjdump --dump-elf "$COREDUMP_FILE" > "$EXTRACTED_BINARY" 2>/dev/null || true
    if [ -f "$EXTRACTED_BINARY" ] && [ -s "$EXTRACTED_BINARY" ]; then
        # Check if it's a valid ELF file
        if file "$EXTRACTED_BINARY" | grep -q "ELF\|CUDA"; then
            echo "Successfully extracted binary using cuobjdump to: $EXTRACTED_BINARY"
            BINARY_FILE="$EXTRACTED_BINARY"
            BINARY_EXISTS=true
        else
            echo "cuobjdump output doesn't appear to be a valid binary, will try disassembly directly"
            rm -f "$EXTRACTED_BINARY"
        fi
    else
        echo "cuobjdump extraction failed, will try disassembly directly from cuda-gdb"
        rm -f "$EXTRACTED_BINARY"
    fi
fi

# If we still don't have a binary, try cuobjdump as fallback
if [ -z "$BINARY_FILE" ] || [ "$BINARY_EXISTS" != true ]; then
    echo "Attempting to extract binary from core dump using cuobjdump..."
    if [ -z "$EXTRACTED_BINARY" ]; then
        EXTRACTED_BINARY=$(mktemp /tmp/cuda_extracted_binary_XXXXXX.o)
    fi
    cuobjdump --dump-elf "$COREDUMP_FILE" > "$EXTRACTED_BINARY" 2>/dev/null || true
    if [ -f "$EXTRACTED_BINARY" ] && [ -s "$EXTRACTED_BINARY" ]; then
        echo "Extracted binary using cuobjdump to: $EXTRACTED_BINARY"
        BINARY_FILE="$EXTRACTED_BINARY"
        BINARY_EXISTS=true
    else
        rm -f "$EXTRACTED_BINARY"
    fi
fi

# Use the extracted binary path or the one from cuda-gdb output
if [ -z "$BINARY_FILE" ]; then
    BINARY_FILE="$BINARY_FILE_PATH"
fi

# Extract error PC offset (the "+ offset" part after symbol name)
# Try multiple patterns to catch different formats
ERROR_OFFSET=$(echo "$GDB_OUTPUT" | grep -oP 'info symbol.*\+ \K[0-9]+' | head -n1)
if [ -z "$ERROR_OFFSET" ]; then
    # Try alternative format: "symbol_name + offset in section"
    ERROR_OFFSET=$(echo "$GDB_OUTPUT" | grep -oP '\+ \K[0-9]+(?= in section)' | head -n1)
fi

# Extract hex PC from print/x output
# Format: $1 = 0x71bfb799cb60
ERROR_PC_HEX=$(echo "$GDB_OUTPUT" | grep -oP '\$[0-9]+ = \K0x[0-9a-f]+' | head -n1)

# Also try to extract PC from exception message if available
# Format: "The exception was triggered at PC 0x7957539a1e70"
if [ -z "$ERROR_PC_HEX" ]; then
    ERROR_PC_HEX=$(echo "$GDB_OUTPUT" | grep -oP 'PC \K0x[0-9a-f]+' | head -n1)
fi

# Extract function name for finding the start address
FUNCTION_NAME=$(echo "$GDB_OUTPUT" | grep -oP '^[^+]+' | head -n1 | sed 's/[[:space:]]*$//')

# Check if we have a valid binary file
if [ -z "$BINARY_EXISTS" ]; then
    BINARY_EXISTS=false
fi

if [ "$BINARY_EXISTS" != true ]; then
    if [ -n "$BINARY_FILE" ] && [ -f "$BINARY_FILE" ]; then
        BINARY_EXISTS=true
        echo "Found binary file: $BINARY_FILE"
    else
        echo "Warning: Binary file not available"
        echo "The binary is embedded in the core dump and only accessible while cuda-gdb is running"
        echo "Will try to get disassembly directly from core dump using cuda-gdb"
        echo ""
    fi
fi

echo "Error offset: $ERROR_OFFSET"
echo "Error PC (hex): $ERROR_PC_HEX"
echo "Function name: $FUNCTION_NAME"
echo ""

# Get disassembly directly from cuda-gdb using the exact PC address
echo "Getting disassembly from cuda-gdb using exact PC address..."
if [ -n "$ERROR_PC_HEX" ]; then
    # Create a script to disassemble around the exact error PC
    cat > "$GDB_SCRIPT" << EOF
set pagination off
target cudacore $COREDUMP_FILE
disassemble \$errorpc
quit
EOF
    cuda-gdb -batch -x "$GDB_SCRIPT" > "$DISASM_OUTPUT" 2>&1 || true
    if [ -s "$DISASM_OUTPUT" ] && ! grep -q "No symbol" "$DISASM_OUTPUT"; then
        echo "Got disassembly from cuda-gdb using error PC"
        BINARY_EXISTS=true  # Mark as available since we have disassembly
    else
        echo "Could not get disassembly using error PC, trying function name..."
        if [ -n "$FUNCTION_NAME" ]; then
            cat > "$GDB_SCRIPT" << EOF
set pagination off
target cudacore $COREDUMP_FILE
disassemble $FUNCTION_NAME
quit
EOF
            cuda-gdb -batch -x "$GDB_SCRIPT" > "$DISASM_OUTPUT" 2>&1 || true
            if [ -s "$DISASM_OUTPUT" ]; then
                echo "Got disassembly from cuda-gdb using function name"
                BINARY_EXISTS=true
            fi
        fi
    fi
elif [ -n "$FUNCTION_NAME" ]; then
    echo "Trying to get disassembly using function name..."
    cat > "$GDB_SCRIPT" << EOF
set pagination off
target cudacore $COREDUMP_FILE
disassemble $FUNCTION_NAME
quit
EOF
    cuda-gdb -batch -x "$GDB_SCRIPT" > "$DISASM_OUTPUT" 2>&1 || true
    if [ -s "$DISASM_OUTPUT" ]; then
        echo "Got disassembly from cuda-gdb"
        BINARY_EXISTS=true
    fi
fi

# If we still have a binary file and nvdisasm works, try that too
if [ -n "$BINARY_FILE" ] && [ -f "$BINARY_FILE" ] && [ "$BINARY_EXISTS" != true ]; then
    echo "Trying nvdisasm on extracted binary..."
    nvdisasm -ndf -c -gi "$BINARY_FILE" > "$DISASM_OUTPUT" 2>&1 || true
    if [ -s "$DISASM_OUTPUT" ] && ! grep -q "fatal\|does not appear" "$DISASM_OUTPUT"; then
        echo "Got disassembly from nvdisasm"
        BINARY_EXISTS=true
    fi
fi

if [ "$BINARY_EXISTS" != true ]; then
    echo "Warning: Could not get disassembly"
    touch "$DISASM_OUTPUT"
fi
echo ""

# Extract the relevant region from disassembly
echo "Writing analysis to $OUTPUT_FILE..."

{
    echo "=== CUDA Core Dump Analysis ==="
    echo "Core dump file: $COREDUMP_FILE"
    echo "Binary file: $BINARY_FILE"
    echo "Error offset: $ERROR_OFFSET"
    echo "Error PC (hex): $ERROR_PC_HEX"
    echo ""
    echo "=== cuda-gdb output ==="
    echo "$GDB_OUTPUT"
    echo ""
    if [ "$BINARY_EXISTS" = true ]; then
        echo "=== Disassembly around error PC ==="
        echo ""
        # cuda-gdb's disassemble command already shows the exact PC location
        # Just display the disassembly - it should contain the error PC
        if [ -n "$ERROR_PC_HEX" ]; then
            echo "Error PC: $ERROR_PC_HEX"
            echo "Looking for this PC in disassembly..."
            echo ""
            # Search for the PC in the disassembly (cuda-gdb shows it as exact address)
            PC_SEARCH=$(echo "$ERROR_PC_HEX" | sed 's/0x//' | tr '[:upper:]' '[:lower:]')
            # Try to find the PC - cuda-gdb format is usually: 0x<address> <instruction>
            FOUND=$(grep -A 10 -B 10 "$PC_SEARCH" "$DISASM_OUTPUT" 2>/dev/null || \
                    grep -A 10 -B 10 "$ERROR_PC_HEX" "$DISASM_OUTPUT" 2>/dev/null || \
                    grep -A 10 -B 10 "$(echo $PC_SEARCH | tr '[:lower:]' '[:upper:]')" "$DISASM_OUTPUT" 2>/dev/null)
            
            if [ -n "$FOUND" ]; then
                echo "$FOUND"
            else
                echo "PC not found in disassembly, showing full disassembly below:"
                echo ""
            fi
        fi
        echo ""
    fi
    
    if [ "$BINARY_EXISTS" = true ]; then
        echo "=== Full disassembly ==="
        if [ -s "$DISASM_OUTPUT" ]; then
            cat "$DISASM_OUTPUT"
        else
            echo "(Disassembly not available)"
        fi
    else
        echo "=== Disassembly ==="
        echo "(Binary file not available - disassembly skipped)"
        echo ""
        echo "To get disassembly, you may need to:"
        echo "1. Check if the binary still exists at the path shown above"
        echo "2. Re-run the benchmark to generate a fresh core dump"
        echo "3. Use cuda-gdb directly: cuda-gdb -batch -ex 'target cudacore $COREDUMP_FILE' -ex 'info symbol \$errorpc'"
    fi
} > "$OUTPUT_FILE"

echo "Analysis complete! Results written to: $OUTPUT_FILE"
echo ""
if [ -n "$ERROR_PC_HEX" ]; then
    PC_SEARCH=$(echo "$ERROR_PC_HEX" | sed 's/0x//' | tr '[:upper:]' '[:lower:]')
    echo "To view the error region, run:"
    echo "  grep -C20 $PC_SEARCH $OUTPUT_FILE"
fi
