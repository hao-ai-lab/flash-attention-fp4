#!/usr/bin/env python3
"""
Pipeline visualization for FA4 FP4 kernel on B300 (SM103).

Generates a model-based timeline chart showing pipeline overlap between:
  - MMA warp (warp 12): QK GEMMs + PV GEMMs
  - Softmax WG0 (warps 0-3): even iterations (stage 0)
  - Softmax WG1 (warps 4-7): odd iterations (stage 1)

Cycle estimates are derived from PTX instruction counts and roofline analysis
(FA4 paper Table 1, B300 hardware specs). No kernel modification needed.

Usage:
    python3 flash_attn/cute/debug/visualize_pipeline.py
"""

import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Cycle model for M=N=d=128 tiles on B300 (SM103)
# ---------------------------------------------------------------------------

# MMA cycles
QK_GEMM_FP4 = 256      # FP4xFP4 QK GEMM
PV_GEMM_BF16 = 1024    # BF16 PV GEMM
PV_GEMM_FP8 = 512      # FP8 PV GEMM
PV_GEMM_FP4 = 256      # FP4 PV GEMM

# Softmax cycles on B300 (SM103). Throughputs measured on GB300 with
# agent_space/bench_cvt_throughput.cu (instr/clk/SM at saturation):
#   ex2.approx.ftz.f32           32.0  (2x SM100's 16 — the B300 MUFU doubling)
#   cvt.rn.bf16x2.f32            62    (~64 limit; trivial truncate of FP32)
#   cvt.rn.satfinite.e4m3x2.f32  32.0  (FP8 needs renorm/saturate — half rate)
#   cvt.rn.satfinite.e2m1x2.f32  57    (FP4 cvt itself is fast)
# Per softmax step each WG (128 threads) does 16384 ex2 and 8192 cvt
# thread-instructions (128x128 tile, x2 packed cvt). In steady state the two
# softmax WGs overlap on one SM, so each sees ~half throughput (factor 2).
LOAD_S_TMEM = 50        # Load S from TMEM
ROW_MAX = 100           # row_max reduction
EXP2 = 1024             # 16384 / (32/2) — MUFU, 2 WGs sharing (B200: 2048)
UPDATE_ROW_SUM = 100    # update_row_sum
P_PACK_BF16 = 256       # 8192 / (62/2) — fast truncating cvt
P_PACK_FP8 = 512        # 8192 / (32/2) — F2FP E4M3 at exactly half BF16 rate
P_QUANT_FP4 = 1500      # group_max + scale + E2M1 pack + reg shuffling
                        # (cvt is fast; the +914 extra PTX instrs dominate)
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
    PVMode("BF16 PV", PV_GEMM_BF16, P_PACK_BF16, COL_QUANT_BF16, "F2FP"),
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
        label=f"MUFU{iter_label}", text_color="white"
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

    # Write P to TMEM + P_full mbarrier signal (drawn as one bar so the
    # timeline has no undrawn time)
    sm_events.append(Event(
        f"wrP{iter_label}", t, WRITE_P_TMEM + SIGNAL_OVERHEAD, COL_WRITE_P,
        label="wrP", text_color="#333333"
    ))
    t += WRITE_P_TMEM
    # P is available for MMA to consume once the write completes
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
        # MMA warp
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
            # Global index matching the PV GEMM that will consume this P:
            # WG0 (stage 0) produces P for odd PVs (PV1, PV3, ...),
            # WG1 (stage 1) for even PVs (PV2, PV4, ...).
            sm_t[stage] = _run_softmax_iteration(
                sm_event_lists[stage], sm_t[stage], stage, 2 * it + 1 + stage,
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

def draw_timeline(ax, events: list[Event], y_center: float, bar_height: float,
                  x_scale: float = 1.0):
    """Draw a row of events on the given axes.

    x_scale: approximate pixels-per-cycle, used to decide label thresholds.
    """
    for ev in events:
        if ev.duration <= 0:
            continue
        h = bar_height * (0.45 if ev.is_wait else 1.0)
        y = y_center - h / 2

        # Flush rectangles (no rounding, no border) so adjacent events butt
        # together without white slivers that read as false pipeline gaps.
        rect = Rectangle(
            (ev.start, y), ev.duration, h,
            facecolor=ev.color, edgecolor="none",
            zorder=3 if not ev.is_wait else 2,
        )
        ax.add_patch(rect)

        # Label -- only bars wide enough to hold readable text; narrow bars
        # (ldS, max, sum, wrP at typical scales) stay unlabeled.
        bar_px = ev.duration * x_scale
        if ev.label and bar_px >= max(50, 11 * len(ev.label)):
            ax.text(
                ev.start + ev.duration / 2, y_center,
                ev.label, ha="center", va="center",
                fontsize=7.5 if bar_px >= 80 else 6.5,
                color=ev.text_color,
                fontweight="bold", zorder=4,
                clip_on=True,
            )


def _draw_dependency_arrows(ax, mma_ev, sm0_ev, sm1_ev, y_mma, y_sm0, y_sm1,
                            bar_height):
    """Draw arrows showing S_full (QK -> Softmax) and P_full (Softmax -> PV) sync."""
    arrow_kw_s = dict(
        arrowstyle="->,head_width=0.15,head_length=0.08",
        color="#2266AA", lw=0.8, alpha=0.5, zorder=1,
        connectionstyle="arc3,rad=0.0",
    )
    arrow_kw_p = dict(
        arrowstyle="->,head_width=0.15,head_length=0.08",
        color="#AA4422", lw=0.8, alpha=0.5, zorder=1,
        connectionstyle="arc3,rad=0.0",
    )

    # Build lookup: name -> event
    def lookup(events):
        d = {}
        for ev in events:
            d[ev.name] = ev
        return d

    mma_d = lookup(mma_ev)

    # S_full arrows: end of QK -> start of softmax (after wait)
    for name, ev in mma_d.items():
        if not name.startswith("QK"):
            continue
        qk_end_x = ev.start + ev.duration
        idx = name[2:]  # e.g. "1", "2", ...
        qk_num = int(idx)
        # QK with odd index -> stage 0 (Softmax WG0), even -> stage 1
        if qk_num % 2 == 1:
            target_events = sm0_ev
            y_target = y_sm0
        else:
            target_events = sm1_ev
            y_target = y_sm1
        # Find the softmax event that starts at or right after this QK ends
        for sev in target_events:
            if not sev.is_wait and sev.start >= qk_end_x - 5:
                ax.annotate("",
                    xy=(sev.start, y_target + bar_height * 0.35),
                    xytext=(qk_end_x, y_mma - bar_height * 0.35),
                    arrowprops=arrow_kw_s,
                )
                break

    # P_full arrows: end of softmax write_P -> start of PV
    for sev_list, y_src, stage in [
        (sm0_ev, y_sm0, 0), (sm1_ev, y_sm1, 1)
    ]:
        for sev in sev_list:
            if not sev.name.startswith("wrP"):
                continue
            wrp_end = sev.start + sev.duration
            # Find matching PV in MMA that starts near this time
            for mev in mma_ev:
                if not mev.name.startswith("PV"):
                    continue
                if mev.start >= wrp_end - 5 and mev.start <= wrp_end + 200:
                    # Only draw if PV color matches stage
                    expected_col = COL_PV_STAGE0 if stage == 0 else COL_PV_STAGE1
                    if mev.color == expected_col:
                        ax.annotate("",
                            xy=(mev.start, y_mma - bar_height * 0.35),
                            xytext=(wrp_end, y_src + bar_height * 0.35),
                            arrowprops=arrow_kw_p,
                        )
                        break


def render_mode(mode: PVMode, ax, n_iter: int = 6, fig_width_inches: float = 18.0):
    """Render a single PV mode subplot."""
    mma_ev, sm0_ev, sm1_ev, total = simulate_pipeline(mode, n_iter)

    row_labels = ["MMA warp", "Softmax WG0", "Softmax WG1"]
    y_positions = [2, 1, 0]
    bar_height = 0.65

    # Approximate pixels per cycle for label sizing (fig DPI * usable width / total cycles)
    usable_width = fig_width_inches * 0.87 * 150  # approximate
    x_scale = usable_width / max(total, 1)

    for events, y in zip([mma_ev, sm0_ev, sm1_ev], y_positions):
        draw_timeline(ax, events, y, bar_height, x_scale)

    # Draw dependency arrows (only for first few iterations to avoid clutter)
    _draw_dependency_arrows(ax, mma_ev, sm0_ev, sm1_ev,
                            y_positions[0], y_positions[1], y_positions[2],
                            bar_height)

    # Axes configuration (2% right margin so the last bars don't touch the frame)
    ax.set_xlim(-0.005 * total, 1.02 * total)
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


def make_legend(fig, modes=None):
    """Create a shared legend showing only colors that appear in the figure.

    If *modes* is None, all PV modes are assumed (combined figure).
    """
    if modes is None:
        modes = PV_MODES

    # Always-present entries
    legend_items = [
        mpatches.Patch(facecolor=COL_QK, edgecolor="white", label="QK GEMM (FP4)"),
        mpatches.Patch(facecolor=COL_PV_STAGE0, edgecolor="white", label="PV GEMM (stage 0)"),
        mpatches.Patch(facecolor=COL_PV_STAGE1, edgecolor="white", label="PV GEMM (stage 1)"),
        mpatches.Patch(facecolor=COL_EXP2, edgecolor="white", label="MUFU (exp2)"),
    ]

    # Per-mode quant/pack entries — only include if that mode is in the figure
    mode_colors = {m.p_quant_color for m in modes}
    if COL_QUANT_BF16 in mode_colors:
        legend_items.append(mpatches.Patch(facecolor=COL_QUANT_BF16, edgecolor="white", label="P cast (F2FP, BF16)"))
    if COL_QUANT_FP8 in mode_colors:
        legend_items.append(mpatches.Patch(facecolor=COL_QUANT_FP8, edgecolor="white", label="P cast (F2FP, FP8)"))
    if COL_QUANT_FP4 in mode_colors:
        legend_items.append(mpatches.Patch(facecolor=COL_QUANT_FP4, edgecolor="white", label="P quant (FP4)"))

    legend_items += [
        mpatches.Patch(facecolor=COL_ROWMAX, edgecolor="white", label="row_max / row_sum"),
        mpatches.Patch(facecolor=COL_WAIT, edgecolor="white", label="mbarrier wait (stall)"),
    ]

    fig.legend(
        handles=legend_items, loc="lower center",
        ncol=min(len(legend_items), 5), fontsize=7.5, frameon=True,
        fancybox=True, shadow=False,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, -0.01),
    )


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)

    # --- Combined figure with all 3 PV modes ---
    fig, axes = plt.subplots(3, 1, figsize=(18, 9.5), dpi=150)
    fig.subplots_adjust(hspace=0.50, bottom=0.10, top=0.88, left=0.10, right=0.97)

    fig.suptitle(
        "FA4 Kernel Pipeline: MMA vs Softmax Overlap on B300 (SM103)\n"
        "M = N = d = 128  |  QK: FP4, MMA warp + 2 Softmax WGs (double-buffered)",
        fontsize=12, fontweight="bold", y=0.97,
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
        make_legend(fig_single, modes=[mode])

        tag = mode.name.lower().replace(" ", "_")
        path = os.path.join(out_dir, f"pipeline_model_b300_{tag}.png")
        fig_single.savefig(path, dpi=150, bbox_inches="tight",
                           facecolor="white", edgecolor="none")
        print(f"Saved: {path}")
        plt.close(fig_single)


if __name__ == "__main__":
    main()
