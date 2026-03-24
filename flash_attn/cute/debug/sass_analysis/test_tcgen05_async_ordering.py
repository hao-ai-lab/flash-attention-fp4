#!/usr/bin/env python3
"""Map PTX instructions to SASS by compiling a minimal standalone PTX kernel.

This is the only reliable method for PTX→SASS mapping. Line annotations in
real kernels are unreliable due to compiler reordering and elimination.

Usage:
    python ptx_to_sass.py                          # run default tcgen05 test
    python ptx_to_sass.py --ptx "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r_mbar];"
    python ptx_to_sass.py --ptx-file my_kernel.ptx # compile and disassemble a full PTX file

Examples:
    # What does tcgen05.fence::after_thread_sync compile to?
    python ptx_to_sass.py --ptx "tcgen05.fence::after_thread_sync;"

    # What about mbarrier.arrive?
    python ptx_to_sass.py --ptx "mbarrier.arrive.release.cta.shared::cta.b64 _, [%r_mbar], 1;"
"""

import argparse
import subprocess
import tempfile
import os
import re
import shutil

PTX_TEMPLATE = """\
.version 8.8
.target sm_100a
.address_size 64

.visible .entry test_mapping(
    .param .u64 smem_mbar_addr
)
{{
    .reg .u64 %rd1, %rd_desc;
    .reg .u32 %r_mbar, %r_tmem;
    .reg .pred %p0;

    // Setup: load mbarrier addr, zero out descriptors
    ld.param.u64 %rd1, [smem_mbar_addr];
    cvta.shared.u64 %rd1, %rd1;
    cvt.u32.u64 %r_mbar, %rd1;
    mov.u64 %rd_desc, 0;
    mov.u32 %r_tmem, 0;

    // Context: issue a real tcgen05 async op so fences/commits aren't optimized away
    tcgen05.cp.cta_group::1.32x128b.warpx4 [%r_tmem], %rd_desc;

    // === INSTRUCTION UNDER TEST ===
    // Marker: NOP before
    bar.sync 0;
{ptx_instructions}
    // Marker: NOP after
    bar.sync 1;

    ret;
}}
"""

DEFAULT_INSTRUCTIONS = [
    ("tcgen05.cp (async copy)", "tcgen05.cp.cta_group::1.32x128b.warpx4 [%r_tmem], %rd_desc;"),
    ("tcgen05.fence::after_thread_sync", "tcgen05.fence::after_thread_sync;"),
    ("tcgen05.commit", "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r_mbar];"),
    ("mbarrier.arrive.release", "mbarrier.arrive.release.cta.shared::cta.b64 _, [%r_mbar], 1;"),
    ("mbarrier.try_wait", "mbarrier.try_wait.parity.shared::cta.b64 %p0, [%r_mbar], 0;"),
]


def compile_and_disassemble(ptx_source, arch="sm_100a", verbose=False):
    """Compile PTX to cubin and disassemble to SASS."""
    ptxas = shutil.which("ptxas")
    nvdisasm = shutil.which("nvdisasm")
    if not ptxas or not nvdisasm:
        # Try CUDA default path
        cuda_bin = "/usr/local/cuda/bin"
        ptxas = ptxas or os.path.join(cuda_bin, "ptxas")
        nvdisasm = nvdisasm or os.path.join(cuda_bin, "nvdisasm")

    with tempfile.TemporaryDirectory() as tmpdir:
        ptx_path = os.path.join(tmpdir, "test.ptx")
        cubin_path = os.path.join(tmpdir, "test.cubin")

        with open(ptx_path, "w") as f:
            f.write(ptx_source)

        if verbose:
            print(f"PTX source:\n{ptx_source}\n")

        # Compile
        result = subprocess.run(
            [ptxas, f"-arch={arch}", "-o", cubin_path, ptx_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"ptxas FAILED:\n{result.stderr}")
            return None

        # Disassemble
        result = subprocess.run(
            [nvdisasm, "-c", cubin_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"nvdisasm FAILED:\n{result.stderr}")
            return None

        return result.stdout


def extract_instructions(sass_output):
    """Extract SASS instruction lines (/*addr*/ INSTR ;) from nvdisasm output."""
    pattern = re.compile(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;')
    instructions = []
    for line in sass_output.split('\n'):
        m = pattern.match(line)
        if m:
            addr = int(m.group(1), 16)
            instr = m.group(2).strip() + ' ;'
            instructions.append((addr, instr))
    return instructions


def find_between_markers(instructions, marker="BAR.SYNC"):
    """Find instructions between two BAR.SYNC markers."""
    found = []
    recording = False
    for addr, instr in instructions:
        if marker in instr.upper():
            if recording:
                return found  # second marker — done
            recording = True
            continue
        if recording:
            found.append((addr, instr))
    return found


def test_single_instruction(ptx_line, label=None, verbose=False):
    """Test what a single PTX instruction compiles to in SASS."""
    ptx_block = f"    {ptx_line}\n"
    ptx_source = PTX_TEMPLATE.format(ptx_instructions=ptx_block)

    sass = compile_and_disassemble(ptx_source, verbose=verbose)
    if sass is None:
        return

    all_instrs = extract_instructions(sass)
    between = find_between_markers(all_instrs)

    name = label or ptx_line.split()[0].rstrip(';')
    if not between:
        print(f"  {name:45s} -> (eliminated / no SASS between markers)")
    elif len(between) == 1 and 'NOP' in between[0][1]:
        print(f"  {name:45s} -> NOP (compiler directive, no hardware instruction)")
    else:
        sass_str = " | ".join(instr for _, instr in between)
        print(f"  {name:45s} -> {sass_str}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ptx", type=str, help="Single PTX instruction to test (e.g. 'tcgen05.fence::after_thread_sync;')")
    parser.add_argument("--ptx-file", type=str, help="Full PTX file to compile and disassemble")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print full PTX and SASS")
    args = parser.parse_args()

    if args.ptx_file:
        with open(args.ptx_file) as f:
            ptx_source = f.read()
        sass = compile_and_disassemble(ptx_source, verbose=args.verbose)
        if sass:
            print(sass)
        return

    if args.ptx:
        ptx_line = args.ptx.strip()
        if not ptx_line.endswith(';'):
            ptx_line += ';'
        print("PTX -> SASS mapping:")
        test_single_instruction(ptx_line, verbose=args.verbose)
        return

    # Default: test all tcgen05 instructions
    print("tcgen05 PTX -> SASS mapping (sm_100a):")
    print("=" * 80)
    for label, ptx_line in DEFAULT_INSTRUCTIONS:
        test_single_instruction(ptx_line, label=label, verbose=args.verbose)


if __name__ == "__main__":
    main()
