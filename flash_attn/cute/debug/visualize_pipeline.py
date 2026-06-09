#!/usr/bin/env python3
"""
Pipeline visualization for FA4 FP4 kernel on B300 (SM103).

Generates a model-based timeline chart showing pipeline overlap between:
  - MMA WG (warp 12): QK GEMMs + PV GEMMs
  - Softmax WG0 (warps 0-3): even iterations (stage 0)
  - Softmax WG1 (warps 4-7): odd iterations (stage 1)

Cycle estimates are derived from PTX instruction counts and roofline analysis
(FA4 paper Table 1, B300 hardware specs). No kernel modification needed.

Usage:
    python3 flash_attn/cute/debug/visualize_pipeline.py
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Cycle model for M=N=d=128 tiles on B300 (SM103)
# ---------------------------------------------------------------------------

# MMA cycles
QK_GEMM_FP4 = 256      # FP4xFP4 QK GEMM
PV_GEMM_BF16 = 1024    # BF16 PV GEMM
PV_GEMM_FP8 = 512      # FP8 PV GEMM
PV_GEMM_FP4 = 256      # FP4 PV GEMM

# Softmax cycles on B300 (SM103, 2x MUFU throughput)
LOAD_S_TMEM = 50        # Load S from TMEM
ROW_MAX = 100           # row_max reduction
EXP2 = 512              # MUFU.EX2 (halved from B200's 1024)
UPDATE_ROW_SUM = 100    # update_row_sum
P_PACK_BF16 = 200       # cvt.bf16x2 instructions
P_PACK_FP8 = 1024       # F2FP (NOT 2x on B300)
P_QUANT_FP4 = 1500      # group_max + scale + E2M1 pack
WRITE_P_TMEM = 50       # Write P to TMEM
SIGNAL_OVERHEAD = 10    # mbarrier signal

# Small overhead for barrier waits when data IS ready
BARRIER_READY = 5

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COL_QK = "#4488CC"           # blue - QK GEMM
COL_PV_STAGE0 = "#44AA66"   # green - PV from Softmax WG0
COL_PV_STAGE1 = "#DD8844"   # orange - PV from Softmax WG1
COL_EXP2 = "#9966CC"        # purple - exp2
COL_QUANT_BF16 = "#DD5577"  # pink/red - P pack BF16
COL_QUANT_FP8 = "#CC3355"   # darker red - P pack FP8
COL_QUANT_FP4 = "#BB2244"   # deep red - P quant FP4
COL_ROWMAX = "#77BBDD"      # light blue - row_max/row_sum
COL_LOAD_S = "#AADDEE"      # very light blue - load S from TMEM
COL_WRITE_P = "#AADDEE"     # very light blue - write P to TMEM
COL_WAIT = "#CCCCCC"        # gray - mbarrier wait (stall)
COL_SIGNAL = "#DDDDDD"      # light gray - signal overhead
COL_BG = "#FAFAFA"          # background

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """A single timed event on the pipeline timeline."""
    name: str
    start: int
    duration: int
    color: str
    label: str = ""
    text_color: str = "white"
    is_wait: bool = False  # True for stall/wait bars (drawn thinner)


@dataclass
class PVMode:
    name: str
    pv_gemm_cycles: int
    p_quant_cycles: int
    p_quant_color: str
    p_quant_label_prefix: str


PV_MODES = [
    PVMode("BF16 PV", PV_GEMM_BF16, P_PACK_BF16, COL_QUANT_BF16, "pack"),
    PVMode("FP8 PV", PV_GEMM_FP8, P_PACK_FP8, COL_QUANT_FP8, "F2FP"),
    PVMode("FP4 PV", PV_GEMM_FP4, P_QUANT_FP4, COL_QUANT_FP4, "quant"),
]


def softmax_total(mode: PVMode) -> int:
    """Total softmax pipeline cycles for one stage."""
    return (
        LOAD_S_TMEM + ROW_MAX + EXP2 + UPDATE_ROW_SUM
        + mode.p_quant_cycles + WRITE_P_TMEM + SIGNAL_OVERHEAD
    )


# ---------------------------------------------------------------------------
# Pipeline simulation
# ---------------------------------------------------------------------------

def _run_softmax_iteration(
    sm_events: list[Event], t: int, stage: int, iter_label: int,
    s_full_ready: list[int], p_full_ready: list[int],
    quant_cycles: int, quant_color: str, quant_label_prefix: str,
) -> int:
    """Simulate one softmax WG iteration for a given stage. Returns updated time."""
    # Wait for S_full[stage]
    wait_until = s_full_ready[stage]
    if wait_until > t:
        wait_dur = wait_until - t
        sm_events.append(Event(
            f"wait_s", t, wait_dur, COL_WAIT,
            label="wait", text_color="#666666", is_wait=True
        ))
        t = wait_until

    # Load S from TMEM
    sm_events.append(Event(
        f"ldS{iter_label}", t, LOAD_S_TMEM, COL_LOAD_S,
        label="ldS", text_color="#333333"
    ))
    t += LOAD_S_TMEM

    # row_max
    sm_events.append(Event(
        f"rmax{iter_label}", t, ROW_MAX, COL_ROWMAX,
        label=f"max{iter_label}", text_color="#333333"
    ))
    t += ROW_MAX

    # exp2
    sm_events.append(Event(
        f"exp{iter_label}", t, EXP2, COL_EXP2,
        label=f"exp2.{iter_label}", text_color="white"
    ))
    t += EXP2

    # update_row_sum
    sm_events.append(Event(
        f"rsum{iter_label}", t, UPDATE_ROW_SUM, COL_ROWMAX,
        label=f"sum{iter_label}", text_color="#333333"
    ))
    t += UPDATE_ROW_SUM

    # P quantization / packing
    sm_events.append(Event(
        f"quant{iter_label}", t, quant_cycles, quant_color,
        label=f"{quant_label_prefix}{iter_label}", text_color="white"
    ))
    t += quant_cycles

    # Write P to TMEM
    sm_events.append(Event(
        f"wrP{iter_label}", t, WRITE_P_TMEM, COL_WRITE_P,
        label="wrP", text_color="#333333"
    ))
    t += WRITE_P_TMEM

    # Signal P_full[stage] -- P is now available for MMA to consume
    p_full_ready[stage] = t
    t += SIGNAL_OVERHEAD
    return t


def simulate_pipeline(mode: PVMode, n_iter: int = 6):
    """
    Simulate the interleaved pipeline for `n_iter` iterations.

    Returns (mma_events, sm0_events, sm1_events, total_cycles).

    Prologue (iteration 0):
        MMA: QK[0] -> signal S_full[0] -> QK[1] -> signal S_full[1]
        Softmax WG0/WG1: wait for S, then produce P

    Steady-state (iterations 1..n_iter-1):
        MMA: Wait P_full[0] -> PV[0] -> Wait P_full[1] -> PV[1] ->
             QK[0] -> signal S_full[0] -> QK[1] -> signal S_full[1]
        Softmax WG0/WG1: wait for S, then produce P (overlapped with MMA)
    """
    mma_events: list[Event] = []
    sm0_events: list[Event] = []
    sm1_events: list[Event] = []

    # Current time for each warp group
    mma_t = 0
    sm_t = [0, 0]  # [stage0_time, stage1_time]

    # Synchronization signals (cycle at which the signal becomes available)
    p_full_ready = [0, 0]  # softmax -> MMA: P is ready for PV
    s_full_ready = [0, 0]  # MMA -> softmax: S is ready for softmax

    pv_cycles = mode.pv_gemm_cycles
    quant_cycles = mode.p_quant_cycles

    sm_event_lists = [sm0_events, sm1_events]

    qk_idx = 0
    pv_idx = 0

    for it in range(n_iter):
        is_prologue = (it == 0)

        # ------------------------------------------------------------------
        # MMA WG
        # ------------------------------------------------------------------

        # PV GEMMs (skip in prologue -- no P available yet)
        if not is_prologue:
            for stage in [0, 1]:
                pv_idx += 1
                # Wait for P_full[stage]
                wait_until = p_full_ready[stage]
                if wait_until > mma_t:
                    wait_dur = wait_until - mma_t
                    mma_events.append(Event(
                        f"wait_p{stage}", mma_t, wait_dur, COL_WAIT,
                        label="wait", text_color="#666666", is_wait=True
                    ))
                    mma_t = wait_until
                # PV GEMM
                col = COL_PV_STAGE0 if stage == 0 else COL_PV_STAGE1
                mma_events.append(Event(
                    f"PV{pv_idx}", mma_t, pv_cycles, col,
                    label=f"PV{pv_idx}", text_color="white"
                ))
                mma_t += pv_cycles

        # QK GEMMs (always)
        for stage in [0, 1]:
            qk_idx += 1
            mma_events.append(Event(
                f"QK{qk_idx}", mma_t, QK_GEMM_FP4, COL_QK,
                label=f"QK{qk_idx}", text_color="white"
            ))
            qk_end = mma_t + QK_GEMM_FP4
            s_full_ready[stage] = qk_end
            mma_t = qk_end

        # ------------------------------------------------------------------
        # Softmax WG0 (stage 0) and WG1 (stage 1)
        # ------------------------------------------------------------------
        for stage in [0, 1]:
            sm_t[stage] = _run_softmax_iteration(
                sm_event_lists[stage], sm_t[stage], stage, it + 1,
                s_full_ready, p_full_ready,
                quant_cycles, mode.p_quant_color, mode.p_quant_label_prefix,
            )

    # Epilogue: final PV GEMMs consuming the last P produced
    for stage in [0, 1]:
        pv_idx += 1
        wait_until = p_full_ready[stage]
        if wait_until > mma_t:
            wait_dur = wait_until - mma_t
            mma_events.append(Event(
                f"wait_p{stage}", mma_t, wait_dur, COL_WAIT,
                label="wait", text_color="#666666", is_wait=True
            ))
            mma_t = wait_until
        col = COL_PV_STAGE0 if stage == 0 else COL_PV_STAGE1
        mma_events.append(Event(
            f"PV{pv_idx}", mma_t, pv_cycles, col,
            label=f"PV{pv_idx}", text_color="white"
        ))
        mma_t += pv_cycles

    total = max(mma_t, sm_t[0], sm_t[1])
    return mma_events, sm0_events, sm1_events, total


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def draw_timeline(ax, events: list[Event], y_center: float, bar_height: float):
    """Draw a row of events on the given axes."""
    for ev in events:
        if ev.duration <= 0:
            continue
        h = bar_height * (0.45 if ev.is_wait else 1.0)
        y = y_center - h / 2

        rect = FancyBboxPatch(
            (ev.start, y), ev.duration, h,
            boxstyle="round,pad=0,rounding_size=4",
            facecolor=ev.color, edgecolor="white", linewidth=0.8,
            zorder=3 if not ev.is_wait else 2,
        )
        ax.add_patch(rect)

        # Label (only if bar is wide enough)
        if ev.duration >= 80 and ev.label:
            fontsize = 7.5 if ev.duration >= 200 else 6
            ax.text(
                ev.start + ev.duration / 2, y_center,
                ev.label, ha="center", va="center",
                fontsize=fontsize, color=ev.text_color,
                fontweight="bold", zorder=4,
                clip_on=True,
            )


def render_mode(mode: PVMode, ax, n_iter: int = 6):
    """Render a single PV mode subplot."""
    mma_ev, sm0_ev, sm1_ev, total = simulate_pipeline(mode, n_iter)

    row_labels = ["MMA WG", "Softmax WG0", "Softmax WG1"]
    y_positions = [2, 1, 0]
    bar_height = 0.65

    for events, y in zip([mma_ev, sm0_ev, sm1_ev], y_positions):
        draw_timeline(ax, events, y, bar_height)

    # Axes configuration
    ax.set_xlim(-50, total + 50)
    ax.set_ylim(-0.6, 2.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_labels, fontsize=9, fontweight="bold")
    ax.set_xlabel("Cycles", fontsize=9)

    # Grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color="#999999")
    ax.yaxis.grid(False)

    # Title
    sm_total = softmax_total(mode)
    mma_cycle = mode.pv_gemm_cycles * 2 + QK_GEMM_FP4 * 2
    bottleneck = "MMA-bound" if mma_cycle >= sm_total else "Softmax-bound"
    ax.set_title(
        f"{mode.name}  |  MMA cycle = {mma_cycle}c, Softmax = {sm_total}c  "
        f"[{bottleneck}]",
        fontsize=10, fontweight="bold", pad=8,
    )

    # Tick formatting
    ax.tick_params(axis="x", labelsize=7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_facecolor(COL_BG)


def make_legend(fig):
    """Create a shared legend at the bottom."""
    legend_items = [
        mpatches.Patch(facecolor=COL_QK, edgecolor="white", label="QK GEMM (FP4)"),
        mpatches.Patch(facecolor=COL_PV_STAGE0, edgecolor="white", label="PV GEMM (stage 0)"),
        mpatches.Patch(facecolor=COL_PV_STAGE1, edgecolor="white", label="PV GEMM (stage 1)"),
        mpatches.Patch(facecolor=COL_EXP2, edgecolor="white", label="exp2 (MUFU)"),
        mpatches.Patch(facecolor=COL_QUANT_BF16, edgecolor="white", label="P pack (BF16)"),
        mpatches.Patch(facecolor=COL_QUANT_FP8, edgecolor="white", label="P pack (FP8 F2FP)"),
        mpatches.Patch(facecolor=COL_QUANT_FP4, edgecolor="white", label="P quant (FP4)"),
        mpatches.Patch(facecolor=COL_ROWMAX, edgecolor="white", label="row_max / row_sum"),
        mpatches.Patch(facecolor=COL_WAIT, edgecolor="white", label="mbarrier wait (stall)"),
    ]
    fig.legend(
        handles=legend_items, loc="lower center",
        ncol=5, fontsize=7.5, frameon=True,
        fancybox=True, shadow=False,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, -0.01),
    )


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Combined figure with all 3 PV modes ---
    fig, axes = plt.subplots(3, 1, figsize=(18, 9), dpi=150)
    fig.subplots_adjust(hspace=0.45, bottom=0.10, top=0.93, left=0.10, right=0.97)

    fig.suptitle(
        "FA4 Kernel Pipeline: MMA vs Softmax Overlap on B300 (SM103)\n"
        "M = N = d = 128  |  QK: FP4, MMA WG + 2 Softmax WGs (double-buffered)",
        fontsize=12, fontweight="bold", y=0.98,
    )

    for ax, mode in zip(axes, PV_MODES):
        render_mode(mode, ax, n_iter=6)

    make_legend(fig)

    combined_path = os.path.join(out_dir, "pipeline_model_b300.png")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {combined_path}")
    plt.close(fig)

    # --- Individual files per PV mode ---
    for mode in PV_MODES:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(18, 3.5), dpi=150)
        fig_single.subplots_adjust(bottom=0.22, top=0.82, left=0.10, right=0.97)
        render_mode(mode, ax_single, n_iter=6)
        make_legend(fig_single)

        tag = mode.name.lower().replace(" ", "_")
        path = os.path.join(out_dir, f"pipeline_model_b300_{tag}.png")
        fig_single.savefig(path, dpi=150, bbox_inches="tight",
                           facecolor="white", edgecolor="none")
        print(f"Saved: {path}")
        plt.close(fig_single)


if __name__ == "__main__":
    main()
