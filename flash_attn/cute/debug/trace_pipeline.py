#!/usr/bin/env python3
"""Real in-kernel pipeline trace for the FA4 FP4 kernel.

Runs attention with FA4_PROFILE_PIPELINE=1 so the kernel is compiled with
%clock timestamp instrumentation (see flash_attn/cute/profiler.py, modeled
on flashinfer's profiler.cuh), then renders the recorded events as a 3-row
timeline: MMA WG, Softmax WG0, Softmax WG1.

Usage:
    python3 flash_attn/cute/debug/trace_pipeline.py --pv_mode bf16
    python3 flash_attn/cute/debug/trace_pipeline.py --pv_mode fp8
    python3 flash_attn/cute/debug/trace_pipeline.py --pv_mode fp4
"""

import argparse
import os
import sys

os.environ["FA4_PROFILE_PIPELINE"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

from flash_attn.cute import profiler as fa4_prof

# Colors per event index
COLORS = {
    fa4_prof.EVT_QK_GEMM: "#4488CC",           # blue
    fa4_prof.EVT_PV_GEMM: ("#44AA66", "#DD8844"),  # green / orange by stage
    fa4_prof.EVT_SOFTMAX_EXP: "#9966CC",       # purple
    fa4_prof.EVT_SOFTMAX_QUANT: "#CC3355",     # red
    fa4_prof.EVT_SOFTMAX_ROWMAX: "#77BBDD",    # cyan
    fa4_prof.EVT_SOFTMAX_WAIT_S: "#CCCCCC",    # gray
    fa4_prof.EVT_SOFTMAX_STORE_P: "#AADDEE",   # light blue
    fa4_prof.EVT_SOFTMAX_WAIT_CORR: "#B8A98F", # tan — distinct from KV wait
    fa4_prof.EVT_MMA_WAIT_P: "#CCCCCC",        # gray
    fa4_prof.EVT_MMA_WAIT_KV: "#555555",       # near-black, MMA row only
    fa4_prof.EVT_PV_WAIT_P2: "#E8E8E8",        # light gray overlay on PV
}

LABELS = {
    fa4_prof.EVT_QK_GEMM: "QK",
    fa4_prof.EVT_PV_GEMM: "PV",
    fa4_prof.EVT_SOFTMAX_EXP: "MUFU",
    fa4_prof.EVT_SOFTMAX_QUANT: "quant",  # overridden per PV mode in render()
    fa4_prof.EVT_SOFTMAX_ROWMAX: "ld+max",
    fa4_prof.EVT_SOFTMAX_WAIT_S: "wait S",
    fa4_prof.EVT_SOFTMAX_STORE_P: "stP",
    fa4_prof.EVT_SOFTMAX_WAIT_CORR: "wait corr",
    fa4_prof.EVT_MMA_WAIT_P: "wait P",
    fa4_prof.EVT_MMA_WAIT_KV: "wait KV",
    fa4_prof.EVT_PV_WAIT_P2: "wait P2",
}

WAIT_EVENTS = {
    fa4_prof.EVT_SOFTMAX_WAIT_S,
    fa4_prof.EVT_SOFTMAX_WAIT_CORR,
    fa4_prof.EVT_MMA_WAIT_P,
    fa4_prof.EVT_MMA_WAIT_KV,
}


def run_attention(pv_mode, batch, seqlen, nheads, headdim):
    import torch
    from flash_attn.cute.benchmarks.bench_fp4 import create_nvfp4_attention_tensors
    from flash_attn.cute.interface import flash_attn_func as flash_attn_func_python

    q, k, v, qsf, ksf, vsf, qr, kr, vr = create_nvfp4_attention_tensors(
        batch, seqlen, seqlen, nheads, nheads, headdim, headdim, pv_mode=pv_mode,
    )
    # First call compiles the instrumented kernel and runs once.
    flash_attn_func_python(q, k, v, mSFQ=qsf, mSFK=ksf, mSFV=vsf)
    torch.cuda.synchronize()
    # Zero the buffer, then do the profiled run.
    fa4_prof.LAST_BUFFER.zero_()
    flash_attn_func_python(q, k, v, mSFQ=qsf, mSFK=ksf, mSFV=vsf)
    torch.cuda.synchronize()
    return fa4_prof.LAST_BUFFER


def summarize(spans_by_bg, block):
    """Print per-event duration totals for the chosen block."""
    print(f"\n=== Per-event cycle totals (block {block}) ===")
    for grp, name in [(fa4_prof.GRP_MMA, "MMA warp"),
                      (fa4_prof.GRP_SOFTMAX0, "Softmax WG0"),
                      (fa4_prof.GRP_SOFTMAX1, "Softmax WG1")]:
        spans = spans_by_bg.get((block, grp), [])
        if not spans:
            continue
        total_window = max(s["end"] for s in spans) - min(s["start"] for s in spans)
        by_evt = {}
        for s in spans:
            d = s["end"] - s["start"]
            cnt, tot = by_evt.get(s["event_idx"], (0, 0))
            by_evt[s["event_idx"]] = (cnt + 1, tot + d)
        print(f"\n{name} (window {total_window} cycles):")
        for evt, (cnt, tot) in sorted(by_evt.items(), key=lambda kv: -kv[1][1]):
            nm = LABELS.get(evt, fa4_prof.EVENT_NAMES[evt] if evt < len(fa4_prof.EVENT_NAMES) else str(evt))
            print(f"  {nm:12s} n={cnt:5d} total={tot:9d} cy ({100.0*tot/total_window:5.1f}%) mean={tot/cnt:7.1f} cy")


def step_stats(spans_by_bg, block):
    """Average cycles of one softmax step and one MMA step (QK+PV) at block.

    Softmax step = median gap between consecutive wait-S starts (per WG,
    averaged over both WGs); its busy part = mean per-step non-wait time.
    MMA step (QK+PV) = mean QK span + mean PV span (one stage's GEMM pair;
    the PV span includes the embedded P2 wait). MMA wait-P = mean stall
    before each PV.
    """
    import statistics

    sm_periods, sm_busys = [], []
    for grp in (fa4_prof.GRP_SOFTMAX0, fa4_prof.GRP_SOFTMAX1):
        spans = spans_by_bg.get((block, grp), [])
        starts = [s["start"] for s in spans if s["event_idx"] == fa4_prof.EVT_SOFTMAX_WAIT_S]
        if len(starts) > 1:
            sm_periods.append(statistics.median(
                b - a for a, b in zip(starts, starts[1:])
            ))
        waits = (fa4_prof.EVT_SOFTMAX_WAIT_S, fa4_prof.EVT_SOFTMAX_WAIT_CORR)
        busy = sum(s["end"] - s["start"] for s in spans if s["event_idx"] not in waits)
        if starts:
            sm_busys.append(busy / len(starts))

    mma = spans_by_bg.get((block, fa4_prof.GRP_MMA), [])
    def _mean(evt):
        d = [s["end"] - s["start"] for s in mma if s["event_idx"] == evt]
        return sum(d) / len(d) if d else 0.0

    return {
        "softmax_period": sum(sm_periods) / len(sm_periods) if sm_periods else 0.0,
        "softmax_busy": sum(sm_busys) / len(sm_busys) if sm_busys else 0.0,
        "mma_qk_pv": _mean(fa4_prof.EVT_QK_GEMM) + _mean(fa4_prof.EVT_PV_GEMM),
        "mma_wait_p": _mean(fa4_prof.EVT_MMA_WAIT_P),
    }


def render(spans_by_bg, block, output_path, title, start_iter, num_iters,
           pv_mode="bf16"):
    rows = [
        (fa4_prof.GRP_MMA, "MMA warp"),
        (fa4_prof.GRP_SOFTMAX0, "Softmax WG0"),
        (fa4_prof.GRP_SOFTMAX1, "Softmax WG1"),
    ]
    # BF16/FP8 P conversion is a plain cast (F2FP in SASS); only FP4 does
    # real quantization (group_max + scale + E2M1 pack).
    quant_label = "quant" if pv_mode == "fp4" else "F2FP"
    quant_legend = "P quant (FP4)" if pv_mode == "fp4" else "P cast (F2FP)"
    # Coarse traces (default) have no per-phase events: EVT_SOFTMAX_EXP is one
    # combined compute span. Detect by the absence of ROWMAX spans.
    detail = any(
        s["event_idx"] == fa4_prof.EVT_SOFTMAX_ROWMAX
        for g in (fa4_prof.GRP_SOFTMAX0, fa4_prof.GRP_SOFTMAX1)
        for s in spans_by_bg.get((block, g), [])
    )

    # Pick a steady-state window: bounded by the start of softmax WG0's
    # (start_iter)-th and (start_iter+num_iters)-th wait-S span.
    sm0 = spans_by_bg.get((block, fa4_prof.GRP_SOFTMAX0), [])
    iter_starts = [s["start"] for s in sm0 if s["event_idx"] == fa4_prof.EVT_SOFTMAX_WAIT_S]
    if len(iter_starts) > start_iter + num_iters:
        w0 = iter_starts[start_iter]
        w1 = iter_starts[start_iter + num_iters]
    else:
        all_spans = [s for (b, g), sp in spans_by_bg.items() if b == block for s in sp]
        w0 = min(s["start"] for s in all_spans)
        w1 = max(s["end"] for s in all_spans)

    fig, axes = plt.subplots(3, 1, figsize=(20, 5), sharex=True,
                             gridspec_kw={"hspace": 0.25})

    for ax, (grp, name) in zip(axes, rows):
        ax.set_ylabel(name, fontsize=10, fontweight="bold", rotation=0,
                      ha="right", va="center")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.set_facecolor("#FAFAFA")
        ax.xaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        spans = spans_by_bg.get((block, grp), [])
        pv_idx = 0
        qk_idx = 0
        counters = {}
        for s in spans:
            evt = s["event_idx"]
            # Count every span (even outside the window) so indices stay global.
            counters[evt] = counters.get(evt, 0) + 1
            if grp == fa4_prof.GRP_MMA:
                if evt == fa4_prof.EVT_PV_GEMM:
                    pv_idx += 1
                elif evt == fa4_prof.EVT_QK_GEMM:
                    qk_idx += 1
            # Clip to the window; skip spans entirely outside it.
            vis_start = max(s["start"], w0)
            vis_end = min(s["end"], w1)
            if vis_end <= vis_start:
                continue

            color = COLORS.get(evt, "#607D8B")
            label = LABELS.get(evt, "")
            if grp == fa4_prof.GRP_MMA and evt == fa4_prof.EVT_PV_GEMM:
                # PV issue order alternates stage 0, stage 1
                color = color[(pv_idx - 1) % 2] if isinstance(color, tuple) else color
                label = f"PV{pv_idx}"
            elif grp == fa4_prof.GRP_MMA and evt == fa4_prof.EVT_QK_GEMM:
                label = f"QK{qk_idx}"
            elif evt in (fa4_prof.EVT_SOFTMAX_EXP, fa4_prof.EVT_SOFTMAX_QUANT):
                # Number softmax work by the PV that consumes it:
                # WG0 -> odd PVs, WG1 -> even PVs.
                if evt == fa4_prof.EVT_SOFTMAX_QUANT:
                    label = quant_label
                elif not detail:
                    # Coarse trace: one combined compute span per step.
                    label = "smax"
                n = counters[evt]
                glob = 2 * (n - 1) + (1 if grp == fa4_prof.GRP_SOFTMAX0 else 2)
                label = f"{label}{glob}"
            if isinstance(color, tuple):
                color = color[0]

            is_wait = evt in WAIT_EVENTS
            h = 0.35 if is_wait else 0.7
            zorder = 2 if is_wait else 3
            if evt == fa4_prof.EVT_PV_WAIT_P2:
                # Measured embedded wait (%clock stored inside the GEMM's
                # PTX around its mbar_P_full_2 try_wait) — drawn on top of
                # the enclosing PV bar at its actual position.
                h, zorder = 0.5, 4
            vis_dur = vis_end - vis_start
            ax.add_patch(Rectangle(
                (vis_start - w0, 0.5 - h / 2), max(vis_dur, 1), h,
                facecolor=color, edgecolor="none",
                zorder=zorder,
            ))
            # Label only when the bar is wide enough for the text
            # (~2550 usable px at figsize 20in x 150dpi).
            bar_px = vis_dur * 2550.0 / (w1 - w0)
            dark_text = is_wait or evt == fa4_prof.EVT_PV_WAIT_P2
            if label and bar_px >= 10 * len(label) + 8:
                ax.text(vis_start - w0 + vis_dur / 2, 0.5, label,
                        ha="center", va="center", fontsize=7,
                        color="#555555" if dark_text else "white",
                        fontweight="bold", zorder=6, clip_on=True)

        ax.set_xlim(0, w1 - w0)

    axes[-1].set_xlabel("Cycles (%clock, same SM)")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    # Average step costs measured over the whole trace (not just the window).
    st = step_stats(spans_by_bg, block)
    fig.text(
        0.5, 0.965,
        f"avg per step — softmax: {st['softmax_period']:,.0f} cy "
        f"(busy {st['softmax_busy']:,.0f} cy)  ·  "
        f"MMA QK+PV GEMM: {st['mma_qk_pv']:,.0f} cy  ·  "
        f"MMA wait-P: {st['mma_wait_p']:,.0f} cy",
        ha="center", fontsize=10, color="#333333",
    )

    if detail:
        sm_legend = [
            mpatches.Patch(color="#9966CC", label="MUFU (exp2)" if pv_mode == "fp4" else "MUFU (exp2, fused cast)"),
            mpatches.Patch(color="#CC3355", label=quant_legend),
            mpatches.Patch(color="#77BBDD", label="S load + row_max"),
        ]
    else:
        sm_legend = [
            mpatches.Patch(color="#9966CC",
                           label="softmax compute (S load + row_max + MUFU + "
                                 + ("quant" if pv_mode == "fp4" else "F2FP cast") + ")"),
        ]
    legend = [
        mpatches.Patch(color="#4488CC", label="QK GEMM"),
        mpatches.Patch(color="#44AA66", label="PV GEMM (stage 0)"),
        mpatches.Patch(color="#DD8844", label="PV GEMM (stage 1)"),
        mpatches.Patch(color="#E8E8E8", label="wait P 2nd half (measured, inside PV)"),
        *sm_legend,
        mpatches.Patch(color="#AADDEE", label="P store + signal"),
        mpatches.Patch(color="#CCCCCC", label="wait S / wait P (mbarrier)"),
        mpatches.Patch(color="#B8A98F", label="wait correction (softmax WG)"),
        mpatches.Patch(color="#555555", label="wait KV (TMA, MMA warp only)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.12))

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pv_mode", default="bf16", choices=["bf16", "fp8", "fp4"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--nheads", type=int, default=24)
    p.add_argument("--headdim", type=int, default=128)
    p.add_argument("--block", type=int, default=0, help="CTA to visualize")
    p.add_argument("--start-iter", type=int, default=0,
                   help="first softmax iteration of the window")
    p.add_argument("--num-iters", type=int, default=8,
                   help="number of softmax iterations to show")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    buf = run_attention(args.pv_mode, args.batch, args.seqlen, args.nheads, args.headdim)
    events, nblocks, ngroups = fa4_prof.decode_trace(buf)
    print(f"Decoded {len(events)} events from {nblocks} blocks x {ngroups} groups")
    if not events:
        print("No events recorded — was the kernel compiled with FA4_PROFILE_PIPELINE=1?")
        sys.exit(1)

    spans = fa4_prof.pair_spans(events)
    summarize(spans, args.block)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"pipeline_trace_{args.pv_mode}_pv.png",
    )
    render(
        spans, args.block, out,
        title=(
            f"FA4 real pipeline trace — {args.pv_mode.upper()} PV, "
            f"b={args.batch} s={args.seqlen} h={args.nheads} d={args.headdim}, "
            f"block {args.block} (GB300)"
        ),
        start_iter=args.start_iter, num_iters=args.num_iters,
        pv_mode=args.pv_mode,
    )


if __name__ == "__main__":
    main()
