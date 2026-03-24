# tcgen05 PTX → SASS Instruction Mapping

Determined by compiling a minimal standalone PTX reproducer and reading the SASS output
directly. This is the only reliable method — line annotations in real kernels are unreliable
due to compiler reordering.

## Reproducer

```bash
cd debug/sass_analysis
# Automated (tests all tcgen05 instructions):
python test_tcgen05_async_ordering.py

# Manual:
ptxas -arch=sm_100a -o test_tcgen05_async_ordering.cubin test_tcgen05_async_ordering.ptx
nvdisasm -c test_tcgen05_async_ordering.cubin

# Test a single PTX instruction:
python test_tcgen05_async_ordering.py --ptx "tcgen05.fence::before_thread_sync;"
```

## Mapping

| PTX Instruction | SASS Instruction | Address | Notes |
|----------------|-----------------|---------|-------|
| `tcgen05.cp.cta_group::1.32x128b.warpx4` | `UTCCP.T.S.4x32dp128bit` | 0070, 00f0 | Async SMEM→TMEM copy |
| `tcgen05.fence::after_thread_sync` | **`NOP`** | 00b0 | Compiler scheduling directive only — no hardware instruction |
| `mbarrier.try_wait.parity` | `SYNCS.PHASECHK.TRANS64.TRYWAIT` | 00c0 | Cross-warp sync wait |
| `tcgen05.commit` | **`UTCBAR`** | 0150 | Binds pending async tcgen05 ops to mbarrier (deferred arrive) |
| `mbarrier.arrive.release.cta` | `SYNCS.ARRIVE.TRANS64.ART0` | 01a0 | Immediate mbarrier arrive |

## Key Findings

1. **`tcgen05.commit` ≠ `mbarrier.arrive`**: They are completely different SASS instructions.
   `UTCBAR` (Uniform TC Barrier) defers the arrive until pending async ops complete.
   `SYNCS.ARRIVE.TRANS64.ART0` arrives immediately.

2. **`tcgen05.fence::after_thread_sync` compiles to `NOP`**: It's a compiler directive that
   constrains instruction scheduling, not a real hardware instruction. Even with real tcgen05
   ops in the kernel, it produces no SASS instruction.

3. **`FENCE.VIEW.ASYNC.T`** in real kernels comes from `fence_view_async_tmem_load()` /
   `fence_view_async_tmem_store()` calls (which map to `tcgen05.wait::ld` / `tcgen05.wait::st`
   in PTX), NOT from `tcgen05.fence::after_thread_sync`.

4. **`.ART0` on `SYNCS.ARRIVE`** may be an nvdisasm annotation (inferred from mbarrier address
   being tcgen05-associated), not a hardware opcode modifier — the encoding bits for predicated
   ART0 and non-ART0 `SYNCS.ARRIVE` are identical in the opcode field.

## Verification in Real Kernel

In `fp4_rep8_annotated.sass`:
- `UTCBAR` count: **12** (matches 12 `tcgen05.commit` in PTX ✓)
- `UTCCP` is the tcgen05.cp instruction
- `SYNCS.ARRIVE.TRANS64.ART0` count: 28 (matches mbarrier.arrive calls)
- `SYNCS.ARRIVE.TRANS64` (no ART0) count: 6 (matches mbarrier.arrive.expect_tx calls)
