#!/usr/bin/env python3
"""
MambaFusion Training Log Analyzer
Parses a BasicSR training log and generates a comprehensive diagnostic dashboard.
Usage: python analyze_mambafusion_log.py <path_to_log_file>
"""

import re
import sys
import os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import numpy as np

# ─────────────────────────── THEME ────────────────────────────────────────────
BG        = "#0d1117"
PANEL     = "#161b22"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
MUTED     = "#7d8590"
ACCENT_1  = "#58a6ff"   # pixel loss
ACCENT_2  = "#f78166"   # perceptual loss
ACCENT_3  = "#3fb950"   # sobel loss
ACCENT_4  = "#d2a8ff"   # total loss
ACCENT_5  = "#ffa657"   # PSNR
ACCENT_6  = "#ff7b72"   # best PSNR marker
ACCENT_LR = "#79c0ff"   # learning rate

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linewidth":    0.6,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  BORDER,
    "legend.labelcolor": TEXT,
    "font.family":       "monospace",
    "font.size":         9,
})


# ──────────────────────────── PARSER ──────────────────────────────────────────

def parse_log(path):
    """Return config dict, train_records list, val_records list."""

    config   = {}
    training = []   # {iter, epoch, lr, l_pix, l_percep, l_sobel, total, timestamp}
    val      = []   # {iter, psnr, best_psnr, best_iter}

    # Patterns
    p_train = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO: \[Mamba\.\.\]"
        r"\[epoch:\s*(\d+), iter:\s*([\d,]+),\s*lr:\(([\d.e+-]+),\)\]"
        r".*?l_pix:\s*([\d.e+-]+)"
        r".*?l_percep:\s*([\d.e+-]+)"
        r".*?l_sobel:\s*([\d.e+-]+)"
    )
    p_val   = re.compile(
        r"INFO: Validation.*\n\s*# psnr:\s*([\d.]+)\s*Best:\s*([\d.]+)\s*@\s*([\d,]+)\s*iter"
    )
    p_cfg_kv = re.compile(r"^\s{2}(\w+):\s+(.+)$")

    # Config block items we care about
    cfg_keys = {
        "name", "model_type", "scale", "num_gpu", "manual_seed",
        "total_iter",
    }

    in_config = False
    lines = open(path).read()

    # ── Parse training lines (line-by-line for speed) ──────────────────────────
    for line in lines.splitlines():
        m = p_train.search(line)
        if m:
            ts, epoch, it, lr, l_pix, l_percep, l_sobel = m.groups()
            it_val   = int(it.replace(",", ""))
            l_p      = float(l_pix)
            l_per    = float(l_percep)
            l_s      = float(l_sobel)
            # Weighted total matching the config weights: 1.0*pix + 0.05*percep + 0.1*sobel
            total    = l_p + 0.05 * l_per + 0.1 * l_s
            training.append({
                "iter":    it_val,
                "epoch":   int(epoch),
                "lr":      float(lr),
                "l_pix":   l_p,
                "l_percep":l_per,
                "l_sobel": l_s,
                "total":   total,
                "ts":      ts,
            })

    # ── Parse validation blocks ─────────────────────────────────────────────────
    # Need multi-line match; use regex over full text
    p_val2 = re.compile(
        r"iter:\s*([\d,]+),.*?\n.*?Saving models.*?\n.*?Validation.*?\n"
        r"\s+# psnr:\s*([\d.]+)\s+Best:\s*([\d.]+)\s*@\s*([\d,]+)\s*iter",
        re.MULTILINE
    )
    # Simpler: just find psnr lines paired with the preceding save iter
    save_iters = []
    psnr_lines = []
    for i, line in enumerate(lines.splitlines()):
        if "Saving models and training states" in line:
            # find the iter from the previous training line
            pass
        if "# psnr:" in line:
            m2 = re.search(r"psnr:\s*([\d.]+)\s+Best:\s*([\d.]+)\s*@\s*([\d,]+)\s*iter", line)
            if m2:
                psnr, best_psnr, best_iter = m2.groups()
                psnr_lines.append({
                    "psnr":      float(psnr),
                    "best_psnr": float(best_psnr),
                    "best_iter": int(best_iter.replace(",", "")),
                })

    # Assign iters to val records based on "Saving models" lines above each val
    all_lines = lines.splitlines()
    val_idx = 0
    for i, line in enumerate(all_lines):
        if "# psnr:" in line and val_idx < len(psnr_lines):
            # Look backward for the most recent save iter
            for j in range(i, max(i - 30, 0), -1):
                m3 = re.search(r"iter:\s*([\d,]+),", all_lines[j])
                if m3:
                    save_it = int(m3.group(1).replace(",", ""))
                    psnr_lines[val_idx]["iter"] = save_it
                    break
            else:
                psnr_lines[val_idx]["iter"] = 0
            val_idx += 1

    val = psnr_lines

    # ── Extract key config facts ────────────────────────────────────────────────
    config_patterns = {
        "name":          r"name:\s+(.+)",
        "model_type":    r"model_type:\s+(.+)",
        "scale":         r"^\s{2}scale:\s+(\d+)",
        "num_gpu":       r"num_gpu:\s+(\d+)",
        "manual_seed":   r"manual_seed:\s+(\d+)",
        "total_iter":    r"total_iter:\s+(\d+)",
        "batch_per_gpu": r"batch_size_per_gpu:\s+(\d+)",
        "num_frames":    r"num_frames:\s+(\d+)",
        "num_feat":      r"num_feat:\s+(\d+)",
        "depths":        r"depths:\s+(\[.+?\])",
        "num_heads":     r"num_heads:\s+(\[.+?\])",
        "mlp_ratio":     r"mlp_ratio:\s+(\d+)",
        "upsampler":     r"upsampler:\s+(.+)",
        "optimizer":     r"type:\s+(AdamW|Adam|SGD)",
        "lr":            r"lr:\s+([\d.e+-]+)",
        "weight_decay":  r"weight_decay:\s+([\d.e+-]+)",
        "scheduler":     r"type:\s+(CosineAnnealingRestartLR|StepLR|MultiStepLR)",
        "eta_min":       r"eta_min:\s+([\de.+-]+)",
        "train_images":  r"Number of train images:\s+([\d,]+)",
        "val_images":    r"Number of val images.*?:\s+([\d,]+)",
        "parameters":    r"with parameters:\s+([\d,]+)",
        "train_dataset": r"name:\s+(RealBSR_\w+)",
        "val_freq":      r"val_freq:\s+(\d+)",
        "print_freq":    r"print_freq:\s+(\d+)",
        "save_freq":     r"save_checkpoint_freq:\s+(\d+)",
        "time_consumed": r"Time consumed:\s+(.+)",
        "pytorch_ver":   r"PyTorch:\s+(.+)",
        "pixel_loss_w":  r"loss_weight:\s+([\d.]+)",
        "percep_weight": r"perceptual_weight:\s+([\d.]+)",
        "sobel_weight":  r"loss_weight:\s+([\d.]+)",
    }

    for key, pat in config_patterns.items():
        m = re.search(pat, lines, re.MULTILINE)
        if m:
            config[key] = m.group(1).strip()

    # Training stats
    config["start_iter"] = training[0]["iter"]  if training else 0
    config["end_iter"]   = training[-1]["iter"] if training else 0
    config["start_ts"]   = training[0]["ts"]    if training else "N/A"
    config["end_ts"]     = training[-1]["ts"]   if training else "N/A"

    return config, training, val


# ─────────────────────────── SMOOTHING ────────────────────────────────────────

def smooth(values, window=50):
    """Causal moving average."""
    if len(values) < window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


# ─────────────────────────── FORMATTING ───────────────────────────────────────

def fmt_k(x, _):
    return f"{int(x/1000)}k" if x >= 1000 else str(int(x))

def sci(x):
    return f"{x:.4e}"


# ─────────────────────────── PLOT ─────────────────────────────────────────────

def plot_dashboard(config, training, val, out_path):
    iters   = np.array([r["iter"]    for r in training])
    l_pix   = np.array([r["l_pix"]   for r in training])
    l_per   = np.array([r["l_percep"] for r in training])
    l_sob   = np.array([r["l_sobel"] for r in training])
    l_tot   = np.array([r["total"]   for r in training])
    lrs     = np.array([r["lr"]      for r in training])

    v_iters = np.array([r["iter"]    for r in val if "iter" in r])
    v_psnr  = np.array([r["psnr"]    for r in val if "iter" in r])
    best_psnr = max(r["best_psnr"] for r in val) if val else 0
    best_iter = next((r["best_iter"] for r in val if r["best_psnr"] == best_psnr), 0)

    W = 50  # smoothing window

    fig = plt.figure(figsize=(20, 26), facecolor=BG)
    fig.text(0.5, 0.985, "MambaFusion — Training Diagnostic Dashboard",
             ha="center", va="top", color=TEXT,
             fontsize=18, fontweight="bold", fontfamily="monospace")
    fig.text(0.5, 0.967, f"{config.get('name','N/A')}  ·  {config.get('start_ts','').split()[0]}  →  {config.get('end_ts','').split()[0]}",
             ha="center", va="top", color=MUTED, fontsize=10)

    gs = gridspec.GridSpec(5, 2, figure=fig,
                           left=0.07, right=0.97,
                           top=0.965, bottom=0.03,
                           hspace=0.55, wspace=0.35)

    # ── 0. Summary stats card (full row) ──────────────────────────────────────
    ax_info = fig.add_subplot(gs[0, :])
    ax_info.set_facecolor(PANEL)
    ax_info.axis("off")

    # Group into columns
    loss_weights = {
        "l_pix":   1.0,
        "l_percep": float(config.get("percep_weight", 0.05)),
        "l_sobel":  0.1,
    }

    col1 = [
        ("Model",         config.get("name",        "—")),
        ("Model Type",    config.get("model_type",  "—")),
        ("Scale Factor",  f"×{config.get('scale','—')}"),
        ("Parameters",    f"{config.get('parameters','—')}"),
        ("Num Frames",    config.get("num_frames",  "—")),
        ("Features",      config.get("num_feat",    "—")),
        ("Depths",        config.get("depths",      "—")),
    ]
    col2 = [
        ("Optimizer",     config.get("optimizer",   "—")),
        ("LR (init)",     config.get("lr",          "—")),
        ("Weight Decay",  config.get("weight_decay","—")),
        ("Scheduler",     config.get("scheduler",   "—")),
        ("LR η_min",      config.get("eta_min",     "—")),
        ("Batch/GPU",     config.get("batch_per_gpu","—")),
        ("GPUs",          config.get("num_gpu",     "—")),
    ]
    col3 = [
        ("Total Iters",   f"{int(config.get('total_iter',0)):,}"),
        ("Completed",     f"{config.get('end_iter',0):,}"),
        ("Val Freq",      config.get("val_freq",    "—")),
        ("Train Images",  config.get("train_images","—")),
        ("Val Images",    config.get("val_images",  "—")),
        ("Time Consumed", config.get("time_consumed","—")),
        ("PyTorch",       config.get("pytorch_ver", "—")),
    ]
    col4 = [
        ("Best PSNR",     f"▶  {best_psnr:.4f} dB"),
        ("Best @ Iter",   f"{best_iter:,}"),
        ("Final PSNR",    f"{v_psnr[-1]:.4f} dB" if len(v_psnr) else "—"),
        ("PSNR Drop",     f"{best_psnr - (v_psnr[-1] if len(v_psnr) else best_psnr):.4f} dB"),
        ("Loss: l_pix w", f"{loss_weights['l_pix']}"),
        ("Loss: percep w",f"{loss_weights['l_percep']}"),
        ("Loss: sobel w", f"{loss_weights['l_sobel']}"),
    ]

    cols = [col1, col2, col3, col4]
    x_positions = [0.0, 0.25, 0.50, 0.75]
    for ci, (col, xp) in enumerate(zip(cols, x_positions)):
        for ri, (label, val_str) in enumerate(col):
            y = 0.90 - ri * 0.135
            ax_info.text(xp,       y, label + ":", color=MUTED,  fontsize=8,  ha="left", va="top", transform=ax_info.transAxes)
            ax_info.text(xp+0.005, y - 0.055, val_str,   color=ACCENT_1 if ci == 3 and ri == 0 else TEXT,
                         fontsize=8.5, fontweight="bold" if ci == 3 and ri == 0 else "normal",
                         ha="left", va="top", transform=ax_info.transAxes)

    # Thin separator below
    line = plt.Line2D([0.01, 0.99], [-0.07, -0.07], color=BORDER, linewidth=0.8, transform=ax_info.transAxes, clip_on=False)
    ax_info.add_line(line)

    # ── 1. Individual losses ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    raw_alpha = 0.18
    ax1.plot(iters, l_pix,  color=ACCENT_1, alpha=raw_alpha, linewidth=0.5)
    ax1.plot(iters, l_per,  color=ACCENT_2, alpha=raw_alpha, linewidth=0.5)
    ax1.plot(iters, l_sob,  color=ACCENT_3, alpha=raw_alpha, linewidth=0.5)
    ax1.plot(iters, smooth(l_pix, W),  color=ACCENT_1, linewidth=1.4, label=f"l_pix (w=1.0)")
    ax1.plot(iters, smooth(l_per, W),  color=ACCENT_2, linewidth=1.4, label=f"l_percep (w=0.05)")
    ax1.plot(iters, smooth(l_sob, W),  color=ACCENT_3, linewidth=1.4, label=f"l_sobel (w=0.1)")
    ax1.set_title("Individual Losses (raw + smoothed)", color=TEXT, fontsize=10, pad=6)
    ax1.set_xlabel("Iteration", color=MUTED)
    ax1.set_ylabel("Loss Value", color=MUTED)
    ax1.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax1.legend(loc="upper right", fontsize=7.5)
    ax1.grid(True, axis="both", alpha=0.4)
    ax1.set_facecolor(PANEL)

    # ── 2. Total (weighted) loss ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(iters, l_tot,         color=ACCENT_4, alpha=0.18, linewidth=0.5)
    ax2.plot(iters, smooth(l_tot, W), color=ACCENT_4, linewidth=1.8, label="Total weighted loss")
    ax2.set_title("Total Weighted Loss", color=TEXT, fontsize=10, pad=6)
    ax2.set_xlabel("Iteration", color=MUTED)
    ax2.set_ylabel("Loss Value", color=MUTED)
    ax2.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, axis="both", alpha=0.4)
    ax2.set_facecolor(PANEL)

    # ── 3. PSNR over validation steps ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :])
    if len(v_iters):
        ax3.plot(v_iters, v_psnr, color=ACCENT_5, linewidth=1.8,
                 marker="o", markersize=4, markerfacecolor=ACCENT_5, label="PSNR (dB)")
        # Shade the gap between current and best
        ax3.axhline(best_psnr, color=ACCENT_6, linewidth=1.2, linestyle="--",
                    label=f"Best: {best_psnr:.4f} dB @ iter {best_iter:,}")
        ax3.axvline(best_iter, color=ACCENT_6, linewidth=0.8, linestyle=":", alpha=0.7)
        ax3.fill_between(v_iters, v_psnr, best_psnr,
                         where=(v_psnr < best_psnr),
                         alpha=0.12, color=ACCENT_6, label="Gap from best")
        # Mark best and final
        best_idx_v = np.argmax(v_psnr)
        ax3.scatter([v_iters[best_idx_v]], [v_psnr[best_idx_v]],
                    color=ACCENT_6, s=80, zorder=5, label=f"Peak @ {v_iters[best_idx_v]:,}")
        ax3.scatter([v_iters[-1]], [v_psnr[-1]],
                    color=ACCENT_5, s=80, marker="D", zorder=5,
                    label=f"Final: {v_psnr[-1]:.4f} dB")

    ax3.set_title("Validation PSNR at Each Checkpoint", color=TEXT, fontsize=10, pad=6)
    ax3.set_xlabel("Iteration", color=MUTED)
    ax3.set_ylabel("PSNR (dB)", color=MUTED)
    ax3.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax3.legend(loc="lower left", fontsize=8, ncol=3)
    ax3.grid(True, axis="both", alpha=0.4)
    ax3.set_facecolor(PANEL)

    # ── 4. Learning rate schedule ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(iters, lrs, color=ACCENT_LR, linewidth=1.2)
    ax4.set_title("Learning Rate Schedule", color=TEXT, fontsize=10, pad=6)
    ax4.set_xlabel("Iteration", color=MUTED)
    ax4.set_ylabel("LR", color=MUTED)
    ax4.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2e}"))
    ax4.grid(True, axis="both", alpha=0.4)
    ax4.set_facecolor(PANEL)

    # ── 5. Loss composition ratio ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 1])
    # Compute contribution of each UNWEIGHTED loss to total raw sum
    raw_total  = l_pix + l_per + l_sob
    r_pix_frac = l_pix / (raw_total + 1e-12)
    r_per_frac = l_per / (raw_total + 1e-12)
    r_sob_frac = l_sob / (raw_total + 1e-12)
    ax5.stackplot(iters,
                  smooth(r_pix_frac, W),
                  smooth(r_per_frac, W),
                  smooth(r_sob_frac, W),
                  colors=[ACCENT_1, ACCENT_2, ACCENT_3],
                  labels=["l_pix", "l_percep", "l_sobel"],
                  alpha=0.75)
    ax5.set_title("Loss Composition (fraction of raw total)", color=TEXT, fontsize=10, pad=6)
    ax5.set_xlabel("Iteration", color=MUTED)
    ax5.set_ylabel("Fraction", color=MUTED)
    ax5.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax5.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax5.set_ylim(0, 1)
    ax5.legend(loc="upper right", fontsize=8)
    ax5.grid(True, axis="y", alpha=0.4)
    ax5.set_facecolor(PANEL)

    # ── 6. Loss vs PSNR overlay (dual axis) ───────────────────────────────────
    ax6 = fig.add_subplot(gs[4, :])
    ax6b = ax6.twinx()
    ax6b.set_facecolor(PANEL)

    ax6.plot(iters, smooth(l_tot, W), color=ACCENT_4, linewidth=1.6, label="Total loss (smoothed)", zorder=3)
    ax6.set_xlabel("Iteration", color=MUTED)
    ax6.set_ylabel("Total Loss", color=ACCENT_4)
    ax6.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax6.tick_params(axis='y', colors=ACCENT_4)
    ax6.grid(True, axis="both", alpha=0.35)
    ax6.set_facecolor(PANEL)

    if len(v_iters):
        ax6b.plot(v_iters, v_psnr, color=ACCENT_5, linewidth=1.8,
                  marker="o", markersize=5, markerfacecolor=ACCENT_5,
                  label="PSNR", zorder=4)
        ax6b.axhline(best_psnr, color=ACCENT_6, linewidth=1.0,
                     linestyle="--", alpha=0.8, label=f"Best PSNR {best_psnr:.4f}")
    ax6b.set_ylabel("PSNR (dB)", color=ACCENT_5)
    ax6b.tick_params(axis='y', colors=ACCENT_5)

    # Combine legends
    lines_a, labels_a = ax6.get_legend_handles_labels()
    lines_b, labels_b = ax6b.get_legend_handles_labels()
    ax6.legend(lines_a + lines_b, labels_a + labels_b, loc="lower left",
               fontsize=8, ncol=3)
    ax6.set_title("Total Loss vs. PSNR — Joint View", color=TEXT, fontsize=10, pad=6)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Dashboard saved → {out_path}")


# ──────────────────────────── CONSOLE SUMMARY ─────────────────────────────────

def print_summary(config, training, val):
    sep = "─" * 62
    def row(k, v): print(f"  {k:<30} {v}")

    print(f"\n{'═'*62}")
    print(f"  MambaFusion Training Summary")
    print(f"{'═'*62}")

    print(f"\n  ── Model ─────────────────────────────────────────────")
    row("Name:",          config.get("name", "—"))
    row("Type:",          config.get("model_type", "—"))
    row("Scale:",         f"×{config.get('scale','—')}")
    row("Parameters:",    config.get("parameters","—"))
    row("Num frames:",    config.get("num_frames","—"))
    row("Num features:",  config.get("num_feat","—"))
    row("Depths:",        config.get("depths","—"))
    row("PyTorch:",       config.get("pytorch_ver","—"))

    print(f"\n  ── Training Setup ────────────────────────────────────")
    row("Optimizer:",     config.get("optimizer","—"))
    row("Initial LR:",    config.get("lr","—"))
    row("Weight decay:",  config.get("weight_decay","—"))
    row("Scheduler:",     config.get("scheduler","—"))
    row("η_min:",         config.get("eta_min","—"))
    row("GPUs:",          config.get("num_gpu","—"))
    row("Batch/GPU:",     config.get("batch_per_gpu","—"))
    row("Planned iters:", f"{int(config.get('total_iter',0)):,}")
    row("Completed iters:",f"{config.get('end_iter',0):,}")
    row("Training time:", config.get("time_consumed","—"))

    print(f"\n  ── Dataset ───────────────────────────────────────────")
    row("Train images:",  config.get("train_images","—"))
    row("Val images:",    config.get("val_images","—"))
    row("Val frequency:", config.get("val_freq","—"))

    print(f"\n  ── Loss Weights ──────────────────────────────────────")
    row("Pixel (Charbonnier):", "1.0")
    row("Perceptual (VGG19):", config.get("percep_weight","0.05"))
    row("Sobel:",              "0.1")

    if training:
        l_pix  = [r["l_pix"]   for r in training]
        l_per  = [r["l_percep"] for r in training]
        l_sob  = [r["l_sobel"] for r in training]
        l_tot  = [r["total"]   for r in training]
        # Compare early vs late
        n = max(len(training)//10, 1)
        early_tot = np.mean(l_tot[:n])
        late_tot  = np.mean(l_tot[-n:])

        print(f"\n  ── Loss Statistics ───────────────────────────────────")
        print(f"  {'Metric':<28} {'First 10%':>12} {'Last 10%':>12} {'Δ':>10}")
        print(f"  {sep}")
        for name_l, arr in [("l_pix", l_pix), ("l_percep", l_per), ("l_sobel", l_sob), ("total (weighted)", l_tot)]:
            e = np.mean(arr[:n])
            l = np.mean(arr[-n:])
            d = l - e
            print(f"  {name_l:<28} {e:>12.5f} {l:>12.5f} {d:>+10.5f}")

    if val:
        psnr_vals  = [r["psnr"] for r in val if "iter" in r]
        val_iters  = [r["iter"] for r in val if "iter" in r]
        best_psnr  = max(r["best_psnr"] for r in val)
        best_iter  = next(r["best_iter"] for r in val if r["best_psnr"] == best_psnr)

        print(f"\n  ── Validation PSNR ───────────────────────────────────")
        row("Validation steps:",  str(len(psnr_vals)))
        row("First PSNR:",        f"{psnr_vals[0]:.4f} dB  @ iter {val_iters[0]:,}" if psnr_vals else "—")
        row("Best PSNR:",         f"{best_psnr:.4f} dB  @ iter {best_iter:,}")
        row("Final PSNR:",        f"{psnr_vals[-1]:.4f} dB  @ iter {val_iters[-1]:,}" if psnr_vals else "—")
        drop = best_psnr - psnr_vals[-1] if psnr_vals else 0
        row("Drop (best→final):", f"{drop:.4f} dB")

        # Find where PSNR peaked
        if len(psnr_vals) > 1:
            peak_i = int(np.argmax(psnr_vals))
            pct = val_iters[peak_i] / int(config.get("total_iter", val_iters[-1])) * 100
            print(f"\n  ⚠  PSNR peaked at {pct:.1f}% of training ({val_iters[peak_i]:,} iters).")
            if drop > 0.1:
                print(f"  ⚠  Degradation of {drop:.4f} dB from peak — possible overfitting.")
            elif drop > 0:
                print(f"  ✓  Marginal degradation ({drop:.4f} dB) — training reasonably stable.")

    print(f"\n{'═'*62}\n")


# ──────────────────────────── MAIN ────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        log_path = "/groups/rls/blozanod/MambaFusion/experiments/MambaFusion_x4/MambaFusion_x4.log"
    else:
        log_path = sys.argv[1]

    if not os.path.exists(log_path):
        print(f"[ERROR] File not found: {log_path}")
        sys.exit(1)

    print(f"[·] Parsing log: {log_path}")
    config, training, val = parse_log(log_path)
    print(f"[·] Training records: {len(training):,}  |  Validation checkpoints: {len(val)}")

    print_summary(config, training, val)

    out_path = "/groups/rls/blozanod/MambaFusion/analysis/dashboards/train_MambaFusion_x4_run3_dashboard.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plot_dashboard(config, training, val, out_path)


if __name__ == "__main__":
    main()
