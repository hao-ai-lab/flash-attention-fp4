"""Lightweight in-kernel profiler for FA4 pipeline visualization.

Two modes:
1. **Model-based** (default): Uses roofline analysis and PTX instruction counts
   to estimate per-WG timing. No kernel modification needed.
   Run: `python3 flash_attn/cute/debug/visualize_pipeline.py`

2. **Trace-based** (opt-in): records %clock timestamps at pipeline events
   inside the kernel (modeled on flashinfer's profiler.cuh), written to a
   gmem buffer by one elected lane per warpgroup. Enabled at compile time
   via FA4_PROFILE_PIPELINE=1 — when off, the kernel has no trace code.
   Run: `FA4_PROFILE_PIPELINE=1 python3 flash_attn/cute/debug/trace_pipeline.py`

Buffer layout for trace mode (int32):
    [0] = num_blocks (CTAs), [1] = num_groups
    Event k of (block b, group g): tag at index 2 + 2*((b*G+g) + k*stride),
    timestamp at +1, where stride = num_blocks * num_groups.
    Tag encoding: sm_id[31:24] | block_id[23:12] | event_idx[11:2] | event_type[1:0]
"""

import os

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.arch import llvm


# --- Device-side PTX helpers ---

@dsl_user_op
def globaltimer_lo(*, loc=None, ip=None) -> Int32:
    """Read low 32 bits of the GPU global nanosecond timer."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %globaltimer_lo;",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def smid(*, loc=None, ip=None) -> Int32:
    """Read the SM ID of the current thread."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %smid;",
            "=r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def clock_lo(*, loc=None, ip=None) -> Int32:
    """Read the low 32 bits of the per-SM cycle counter (%clock).

    Cycle-accurate and coherent across warps of the same CTA (same SM),
    which is exactly what the FA4 pipeline trace needs — the MMA WG and
    both softmax WGs live in one CTA. Wraps every 2^32 cycles (~2s @ 2GHz).
    """
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %clock;",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def barrier_noop(*, loc=None, ip=None) -> None:
    """An empty side-effecting asm: a pure scheduling barrier (ptxas can't
    move instructions across it) with no clock read and no store. Used by
    FA4_PROF_EMPTY to isolate the cost of *preventing compiler reordering*
    from the clock-read and store costs."""
    llvm.inline_asm(
        None, [], "// prof barrier", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# ---------------------------------------------------------------------------
# In-kernel trace recording (flashinfer-style, see flashinfer/profiler.cuh)
#
# Buffer is int32. Layout:
#   [0] = num_blocks (CTAs), [1] = num_groups
#   entry k of (block b, group g): tag at 2 + 2*((b*num_groups+g) + k*stride),
#   timestamp at +1, where stride = num_blocks*num_groups.
# Tag: sm_id[31:24] | block_id[23:12] | event_idx[11:2] | event_type[1:0].
# One elected lane per warpgroup writes; the slot counter k is a loop-carried
# register in the calling warpgroup (no atomics, no gmem read-modify-write).
# ---------------------------------------------------------------------------

@cute.jit
def prof_init_meta(prof: cute.Tensor, num_blocks: Int32, num_groups: Int32):
    """Write buffer metadata. Call from one thread of block 0."""
    prof[0] = num_blocks
    prof[1] = num_groups


@cute.jit
def prof_make_tag_base(block_linear: Int32) -> Int32:
    """Tag base for the current SM/block (event bits filled per record)."""
    return (smid() << 24) | ((block_linear & 0xFFF) << 12)


@cute.jit
def prof_record(
    prof: cute.Tensor,
    bg_index: Int32,       # block_linear * num_groups + group
    stride: Int32,         # num_blocks * num_groups
    tag_base: Int32,
    k: Int32,              # per-(block, group) slot counter
    event_idx: cutlass.Constexpr[int],
    event_type: cutlass.Constexpr[int],
    pred,                  # only the elected lane writes
) -> Int32:
    """Record one event; returns the next slot counter (k + 1)."""
    if pred:
        # FA4_PROF_EMPTY: keep the per-call-site scheduling barrier but drop the
        # clock read and the store — isolates the compiler-reorder-prevention cost.
        if os.environ.get("FA4_PROF_EMPTY", "0") == "1":
            barrier_noop()
        else:
            slot32 = 2 + 2 * (bg_index + k * stride)
            if slot32 + 1 < cute.size(prof.shape):
                prof[slot32] = tag_base | Int32((event_idx << 2) | event_type)
                prof[slot32 + 1] = clock_lo()
    return k + 1


@cute.jit
def prof_flush(
    smem_buf: cute.Tensor,
    gmem_buf: cute.Tensor,
    block_linear: Int32,
    num_blocks: Int32,
    num_groups: Int32,
    group: Int32,
    pred,
):
    """Copy one group's smem-buffered events to their global gmem slots so
    decode_trace (which reads the gmem layout) works unchanged. Called once at
    each warp's exit. Relies on the smem ring being zero-initialised: a zero
    tag marks an unwritten slot and is skipped.

    smem local slot for (group, j):  2 + 2*(group + j*num_groups)
    gmem global slot for (block, group, j):
        2 + 2*(block*num_groups + group + j*num_blocks*num_groups)
    """
    if pred:
        for j in cutlass.range(PROF_SMEM_EVENTS_PER_GROUP):
            ls = 2 + 2 * (group + j * num_groups)
            tag = smem_buf[ls]
            if tag != Int32(0):
                gs = 2 + 2 * (block_linear * num_groups + group + j * num_blocks * num_groups)
                if gs + 1 < cute.size(gmem_buf.shape):
                    gmem_buf[gs] = tag
                    gmem_buf[gs + 1] = smem_buf[ls + 1]


@cute.jit
def prof_reserve_pair_ts(
    prof: cute.Tensor,
    bg_index: Int32,
    stride: Int32,
    tag_base: Int32,
    k: Int32,
    event_idx: cutlass.Constexpr[int],
    pred,
):
    """Reserve a BEGIN/END pair at slots k and k+1, writing only the tags.

    Returns (k + 2, ts_addr_begin, ts_addr_end): the global addresses of the
    two timestamp ints, to be filled by device code that knows the actual
    event boundaries (e.g. %clock stores inside a GEMM's inline PTX around
    its embedded mbarrier wait). Out-of-range slots are redirected to the
    last entry pair of the buffer as a sacrificial scratch slot.
    """
    slot_b = 2 + 2 * (bg_index + k * stride)
    slot_e = 2 + 2 * (bg_index + (k + 1) * stride)
    last_ok = cute.size(prof.shape) - 2
    if slot_e + 1 >= cute.size(prof.shape):
        slot_b = last_ok
        slot_e = last_ok
    if pred:
        prof[slot_b] = tag_base | Int32((event_idx << 2) | 0)  # EVENT_BEGIN
        prof[slot_e] = tag_base | Int32((event_idx << 2) | 1)  # EVENT_END
    base = Int64(prof.iterator.toint())
    ts_addr_b = base + Int64(slot_b + 1) * 4
    ts_addr_e = base + Int64(slot_e + 1) * 4
    return k + 2, ts_addr_b, ts_addr_e


# Event type constants
EVENT_BEGIN = 0
EVENT_END = 1
EVENT_INSTANT = 2

# Event index constants (10-bit, max 1023)
EVT_QK_GEMM = 0          # MMA WG: QK GEMM issue
EVT_PV_GEMM = 1          # MMA WG: PV GEMM issue
EVT_SOFTMAX_EXP = 2      # softmax WG: exp2 (+row_sum; +pack on fused paths)
EVT_SOFTMAX_QUANT = 3    # softmax WG: P quant / pack
EVT_SOFTMAX_ROWMAX = 4   # softmax WG: S t2r load + mask + row_max + subtract
EVT_SOFTMAX_ROWSUM = 5   # (unused; row_sum folded into EXP)
EVT_SOFTMAX_WAIT_S = 6   # softmax WG: mbarrier wait for S_full
EVT_SOFTMAX_STORE_P = 7  # softmax WG: P r2t store + P_full arrives
EVT_MMA_WAIT_P = 8       # MMA WG: mbarrier wait for P_full / O rescaled
EVT_MMA_SIGNAL_S = 9     # (unused)
EVT_SOFTMAX_WAIT_CORR = 10  # softmax WG: wait for correction empty
EVT_EPILOGUE = 11        # (unused)
EVT_MMA_WAIT_KV = 12     # MMA WG: pipeline_kv consumer wait (TMA load)
EVT_PV_WAIT_P2 = 13      # MMA WG: embedded wait for P 2nd half inside PV GEMM

EVENT_NAMES = [
    "QK GEMM", "PV GEMM", "exp2", "P quant", "load+row_max", "row_sum",
    "wait S", "store P", "wait P", "signal S", "wait corr", "epilogue",
    "wait KV", "wait P2",
]

# WG group indices
GRP_MMA = 0
GRP_SOFTMAX0 = 1
GRP_SOFTMAX1 = 2
GRP_CORRECTION = 3
NUM_GROUPS = 4

GROUP_NAMES = ["MMA warp", "Softmax WG0", "Softmax WG1", "Correction WG"]


def is_profiling_enabled():
    """Check if trace-based profiling is enabled."""
    return os.environ.get("FA4_PROFILE_PIPELINE", "0") == "1"


def is_smem_profiling():
    """Buffer trace events in shared memory and flush once at kernel exit,
    instead of one gmem store per event (which inflates runtime ~40-140%).
    Holds ~one stage's events; the rest of the kernel runs uninstrumented.
    Default ON — set FA4_PROF_SMEM=0 to compare against the gmem path."""
    return os.environ.get("FA4_PROF_SMEM", "1") == "1"


# Events buffered per (group) in the smem ring before recording stops. One
# stage (~25-32 K-steps x ~5 events) fits; sized to stay within bf16's ~8KB
# free smem: 2 + 2*NUM_GROUPS*EVENTS ints.
PROF_SMEM_EVENTS_PER_GROUP = 128


def prof_smem_capacity_ints(num_groups):
    return 2 + 2 * num_groups * PROF_SMEM_EVENTS_PER_GROUP


# Host-side handle to the buffer used by the most recent profiled kernel
# launch. Set by interface.py, read by the trace script after the call.
LAST_BUFFER = None

# Default buffer size: 8M int32 = 32 MB = up to ~9000 events per (block,
# group) at a 148-CTA persistent grid.
DEFAULT_BUFFER_INT32 = 8 * 1024 * 1024


def allocate_profiler_buffer(device="cuda", n_int32=DEFAULT_BUFFER_INT32):
    """Allocate (and zero) the int32 trace buffer."""
    import torch
    return torch.zeros(n_int32, dtype=torch.int32, device=device)


def decode_trace(profiler_buf):
    """Decode the int32-pair trace buffer into a list of event dicts.

    Returns (events, num_blocks, num_groups). Events carry timestamps in
    SM cycles (from %clock), coherent within one block.
    """
    import numpy as np

    buf = profiler_buf.cpu().numpy().view(np.uint32)
    num_blocks = int(buf[0])
    num_groups = int(buf[1])
    if num_blocks == 0 or num_groups == 0:
        return [], 0, 0

    stride = num_blocks * num_groups
    events = []
    data = buf[2:]
    n_pairs = len(data) // 2
    tags = data[0 : 2 * n_pairs : 2]
    times = data[1 : 2 * n_pairs : 2]
    nonzero = np.nonzero(tags)[0]
    for i in nonzero:
        tag = int(tags[i])
        bg = int(i % stride)
        events.append({
            "block_idx": bg // num_groups,
            "group_idx": bg % num_groups,
            "slot": int(i // stride),
            "event_idx": (tag >> 2) & 0x3FF,
            "event_type": tag & 0x3,
            "sm_id": (tag >> 24) & 0xFF,
            "timestamp": int(times[i]),
        })
    return events, num_blocks, num_groups


def pair_spans(events):
    """Pair sequential BEGIN/END events per (block, group) into spans.

    Events of one (block, group) are written by a single thread in slot
    order, so pairing is by slot order with a per-event-idx open stack.
    Returns dict (block, group) -> list of {event_idx, start, end} sorted
    by start, with timestamps rebased per block to that block's minimum.
    """
    from collections import defaultdict

    by_bg = defaultdict(list)
    for e in events:
        by_bg[(e["block_idx"], e["group_idx"])].append(e)

    spans = {}
    for bg, evs in by_bg.items():
        evs.sort(key=lambda e: e["slot"])
        open_ev = {}
        out = []
        for e in evs:
            if e["event_type"] == EVENT_BEGIN:
                open_ev[e["event_idx"]] = e
            elif e["event_type"] == EVENT_END and e["event_idx"] in open_ev:
                b = open_ev.pop(e["event_idx"])
                out.append({
                    "event_idx": e["event_idx"],
                    "start": b["timestamp"],
                    "end": e["timestamp"],
                })
        out.sort(key=lambda s: s["start"])
        spans[bg] = out
    return spans
