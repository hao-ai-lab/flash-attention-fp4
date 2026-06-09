"""Lightweight in-kernel profiler for FA4 pipeline visualization.

Two modes:
1. **Model-based** (default): Uses roofline analysis and PTX instruction counts
   to estimate per-WG timing. No kernel modification needed.
   Run: `python3 flash_attn/cute/debug/visualize_pipeline.py`

2. **Trace-based** (opt-in): Injects `cute.printf` calls at key pipeline points
   to capture actual `globaltimer_lo` timestamps. Requires kernel recompilation.
   Set `FA4_PROFILE_PIPELINE=1` before running the kernel.

Buffer layout for trace mode (uint64 per entry):
    Entry 0: metadata (num_blocks << 32 | num_groups)
    Entry 1+: strided by (block_idx * num_groups + group_idx)
    Each entry: (tag:u32, timestamp:u32) packed into u64
    Tag encoding: sm_id[31:24] | block_id[23:12] | event_idx[11:2] | event_type[1:0]
"""

import os

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
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


# Event type constants
EVENT_BEGIN = 0
EVENT_END = 1
EVENT_INSTANT = 2

# Event index constants (10-bit, max 1023)
EVT_QK_GEMM = 0
EVT_PV_GEMM = 1
EVT_SOFTMAX_EXP = 2
EVT_SOFTMAX_QUANT = 3
EVT_SOFTMAX_ROWMAX = 4
EVT_SOFTMAX_ROWSUM = 5
EVT_SOFTMAX_WAIT_S = 6
EVT_SOFTMAX_SIGNAL_P = 7
EVT_MMA_WAIT_P = 8
EVT_MMA_SIGNAL_S = 9
EVT_CORRECTION = 10
EVT_EPILOGUE = 11

EVENT_NAMES = [
    "QK GEMM", "PV GEMM", "exp2", "P quant", "row_max", "row_sum",
    "wait S", "signal P", "wait P", "signal S", "correction", "epilogue",
]

# WG group indices
GRP_MMA = 0
GRP_SOFTMAX0 = 1
GRP_SOFTMAX1 = 2
GRP_CORRECTION = 3
NUM_GROUPS = 4

GROUP_NAMES = ["MMA WG", "Softmax WG0", "Softmax WG1", "Correction WG"]


def is_profiling_enabled():
    """Check if trace-based profiling is enabled."""
    return os.environ.get("FA4_PROFILE_PIPELINE", "0") == "1"


def allocate_profiler_buffer(max_events_per_group=256, num_blocks=256):
    """Allocate a global memory buffer for profiling events."""
    import torch
    total_entries = 1 + max_events_per_group * num_blocks * NUM_GROUPS
    buf = torch.zeros(total_entries, dtype=torch.int64, device="cuda")
    return buf


def decode_events(profiler_buf):
    """Decode the profiler buffer into a list of event dicts."""
    import numpy as np

    buf = profiler_buf.cpu().numpy().view(np.uint64)
    if buf[0] == 0:
        return [], 0, 0

    raw = int(buf[0])
    num_blocks = raw & 0xFFFFFFFF
    num_groups = (raw >> 32) & 0xFFFFFFFF

    events = []
    for i in range(1, len(buf)):
        if buf[i] == 0:
            continue
        raw = int(buf[i])
        tag = raw & 0xFFFFFFFF
        timestamp = (raw >> 32) & 0xFFFFFFFF

        sm_id = (tag >> 24) & 0xFF
        block_group_idx = (tag >> 12) & 0xFFF
        event_idx = (tag >> 2) & 0x3FF
        event_type = tag & 0x3

        block_idx = block_group_idx // num_groups if num_groups > 0 else 0
        group_idx = block_group_idx % num_groups if num_groups > 0 else 0

        events.append({
            "block_idx": block_idx,
            "group_idx": group_idx,
            "event_idx": event_idx,
            "event_type": event_type,
            "sm_id": sm_id,
            "timestamp": timestamp,
        })

    return events, int(num_blocks), int(num_groups)


def parse_printf_trace(stdout_text):
    """Parse cute.printf trace output into events.

    Expected format per line:
        FA4_TRACE|<wg_id>|<event_idx>|<event_type>|<timestamp>|<sm_id>|<block_id>

    Returns list of event dicts compatible with visualize_pipeline().
    """
    import re
    events = []
    pattern = re.compile(
        r"FA4_TRACE\|(\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)"
    )
    for line in stdout_text.strip().split("\n"):
        m = pattern.search(line)
        if m:
            wg_id, event_idx, event_type, timestamp, sm_id, block_id = (
                int(x) for x in m.groups()
            )
            events.append({
                "block_idx": block_id,
                "group_idx": wg_id,
                "event_idx": event_idx,
                "event_type": event_type,
                "sm_id": sm_id,
                "timestamp": timestamp,
            })
    return events


def visualize_pipeline(events, block_idx=0, output_path="pipeline_trace.png",
                       max_iterations=None, title=None):
    """Create a 3-row pipeline timeline visualization.

    Rows: MMA WG, Softmax WG0, Softmax WG1
    MMA WG events are colored by which softmax WG's output they consume.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Filter to selected block
    block_events = [e for e in events if e["block_idx"] == block_idx]
    if not block_events:
        print(f"No events for block {block_idx}")
        return

    block_events.sort(key=lambda e: e["timestamp"])
    t0 = block_events[0]["timestamp"]

    # Pair begin/end events per group
    rows = {GRP_MMA: [], GRP_SOFTMAX0: [], GRP_SOFTMAX1: []}
    open_events = {}

    for e in block_events:
        grp = e["group_idx"]
        if grp not in rows:
            continue
        key = (grp, e["event_idx"])
        if e["event_type"] == EVENT_BEGIN:
            open_events[key] = e
        elif e["event_type"] == EVENT_END:
            if key in open_events:
                start = open_events.pop(key)
                rows[grp].append({
                    "event_idx": e["event_idx"],
                    "start": start["timestamp"] - t0,
                    "end": e["timestamp"] - t0,
                })

    # Color scheme
    stage0_color = "#4CAF50"   # green for stage 0
    stage1_color = "#FF9800"   # orange for stage 1
    qk_color = "#2196F3"       # blue for QK
    exp_color = "#9C27B0"      # purple for exp
    quant_color = "#E91E63"    # pink for quant
    rowop_color = "#00BCD4"    # cyan for row_max/row_sum
    wait_color = "#9E9E9E"     # gray for waits
    other_color = "#607D8B"    # blue-gray

    def get_color(grp, evt_idx):
        if grp == GRP_MMA:
            if evt_idx == EVT_QK_GEMM:
                return qk_color
            elif evt_idx == EVT_PV_GEMM:
                return stage0_color
            elif evt_idx == EVT_MMA_WAIT_P:
                return wait_color
        elif grp in (GRP_SOFTMAX0, GRP_SOFTMAX1):
            if evt_idx == EVT_SOFTMAX_EXP:
                return exp_color
            elif evt_idx == EVT_SOFTMAX_QUANT:
                return quant_color
            elif evt_idx in (EVT_SOFTMAX_ROWMAX, EVT_SOFTMAX_ROWSUM):
                return rowop_color
            elif evt_idx in (EVT_SOFTMAX_WAIT_S, EVT_SOFTMAX_SIGNAL_P):
                return wait_color
        return other_color

    fig, axes = plt.subplots(3, 1, figsize=(20, 4), sharex=True,
                             gridspec_kw={"hspace": 0.15})

    row_map = {GRP_MMA: 0, GRP_SOFTMAX0: 1, GRP_SOFTMAX1: 2}
    row_labels = ["MMA WG", "Softmax WG0", "Softmax WG1"]

    mma_counters = {"QK": 0, "PV": 0}
    pv_stage_toggle = 0

    for grp_idx, ax_idx in row_map.items():
        ax = axes[ax_idx]
        ax.set_ylabel(row_labels[ax_idx], fontsize=10, rotation=0,
                       ha="right", va="center")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        for span in rows.get(grp_idx, []):
            evt = span["event_idx"]
            start_ns = span["start"]
            duration = span["end"] - span["start"]
            color = get_color(grp_idx, evt)

            label = ""
            if grp_idx == GRP_MMA:
                if evt == EVT_QK_GEMM:
                    mma_counters["QK"] += 1
                    label = f"QK{mma_counters['QK']}"
                    color = qk_color
                elif evt == EVT_PV_GEMM:
                    mma_counters["PV"] += 1
                    label = f"PV{mma_counters['PV']}"
                    color = [stage0_color, stage1_color][pv_stage_toggle]
                    pv_stage_toggle = 1 - pv_stage_toggle
                elif evt == EVT_MMA_WAIT_P:
                    label = "wait"
            elif grp_idx in (GRP_SOFTMAX0, GRP_SOFTMAX1):
                if evt == EVT_SOFTMAX_EXP:
                    label = "exp"
                elif evt == EVT_SOFTMAX_QUANT:
                    label = "quant"
                elif evt == EVT_SOFTMAX_WAIT_S:
                    label = "wait"
                elif evt == EVT_SOFTMAX_ROWMAX:
                    label = "rmax"
                elif evt == EVT_SOFTMAX_ROWSUM:
                    label = "rsum"

            ax.barh(0.5, duration, left=start_ns, height=0.7,
                    color=color, edgecolor="white", linewidth=0.5)
            if label and duration > 20:
                ax.text(start_ns + duration / 2, 0.5, label,
                        ha="center", va="center", fontsize=7, color="white",
                        fontweight="bold")

    if max_iterations:
        all_ends = [s["end"] for grp in rows.values() for s in grp]
        if all_ends:
            axes[-1].set_xlim(0, sorted(all_ends)[
                min(len(all_ends) - 1, max_iterations * 10)
            ])

    axes[-1].set_xlabel("Time (cycles)")
    fig.suptitle(title or f"FA4 Pipeline Trace — Block {block_idx}",
                 fontsize=12, y=0.98)

    legend_patches = [
        mpatches.Patch(color=qk_color, label="QK GEMM"),
        mpatches.Patch(color=stage0_color, label="PV (stage 0)"),
        mpatches.Patch(color=stage1_color, label="PV (stage 1)"),
        mpatches.Patch(color=exp_color, label="exp2"),
        mpatches.Patch(color=quant_color, label="P quant/pack"),
        mpatches.Patch(color=rowop_color, label="row_max/row_sum"),
        mpatches.Patch(color=wait_color, label="barrier wait"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", ncol=4, fontsize=8,
               bbox_to_anchor=(0.98, 0.98))

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Pipeline visualization saved to {output_path}")
    return output_path
