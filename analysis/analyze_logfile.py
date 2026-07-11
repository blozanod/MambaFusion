#!/usr/bin/env python3
"""
MambaFusion Training Log Analyzer

Parses a BasicSR-style training log and generates a diagnostic dashboard PNG
plus a markdown summary. Every loss term (any `l_<name>: <value>` token on a
training line) and every validation metric (any `# <name>: <value> Best: ...`
line in a Validation block) is discovered dynamically from the log itself, so
this works whether the run logs one loss/metric or a dozen — nothing about
the loss functions or metrics is hardcoded.

Usage:
    python analyze_logfile.py --log <path_to_log_file> [--output-dir <dir>] [--config <path_to_yaml>]

`--config` is optional; if given, per-loss weights are read from its `train`
section (any `<name>_opt.loss_weight`) and used to compute a weighted total
loss. Without it, all discovered losses are weighted equally (1.0) in the
"total" series.
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

try:
    import yaml
except ImportError:
    yaml = None

# ─────────────────────────── THEME ────────────────────────────────────────────
BG        = "#0d1117"
PANEL     = "#161b22"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
MUTED     = "#7d8590"
ACCENT_LR = "#79c0ff"
ACCENT_BEST = "#ff7b72"

# Color cycle for an arbitrary number of losses / metrics.
PALETTE = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#ffa657",
           "#ff7b72", "#79c0ff", "#e3b341", "#56d4dd", "#f778ba"]

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

# Matches the header of a training-progress line, e.g.:
#   2026-06-20 22:55:30,574 INFO: [MF_ST..][epoch:  0, iter:     100, lr:(1.000e-06,)] ...
RE_HEADER = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO: "
    r"\[[^\]]*\]\[epoch:\s*(?P<epoch>\d+),\s*iter:\s*(?P<iter>[\d,]+),"
    r"\s*lr:\((?P<lr>[\d.eE+-]+),?\)\]"
)
# Any loss token: `l_<name>: <value>`. Matches however many are present.
RE_LOSS = re.compile(r"\b(l_\w+):\s*([\d.eE+-]+)")
# Any bare "iter: N," occurrence — used to back-associate a validation block
# with the most recent training iteration above it.
RE_ITER_ANY = re.compile(r"iter:\s*([\d,]+),")
# Validation block header, e.g. "INFO: Validation RealBSR_val"
RE_VAL_HEADER = re.compile(r"INFO: Validation\s+(\S+)")
# One metric line inside a validation block, e.g.:
#   "\t # psnr_srgb: 21.9998\tBest: 21.9998 @ 5000 iter"
RE_VAL_METRIC = re.compile(r"#\s*(\w+):\s*([\d.]+)\s*Best:\s*([\d.]+)\s*@\s*([\d,]+)\s*iter")

# Config facts pulled from the log's own `dict2str(opt)` dump, for display only.
CONFIG_PATTERNS = {
    "name":          r"^\s*name:\s+(.+)$",
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
    "upsampler":     r"upsampler:\s+(.+)",
    "optimizer":     r"type:\s+(AdamW|Adam|SGD)",
    "lr":            r"^\s+lr:\s+([\d.eE+-]+)",
    "weight_decay":  r"weight_decay:\s+([\d.eE+-]+)",
    "scheduler":     r"type:\s+(CosineAnnealingRestartLR|StepLR|MultiStepLR)",
    "train_images":  r"Number of train images:\s+([\d,]+)",
    "val_images":    r"Number of val images.*?:\s+([\d,]+)",
    "parameters":    r"with parameters:\s+([\d,]+)",
    "val_freq":      r"val_freq:\s+(\d+)",
    "print_freq":    r"print_freq:\s+(\d+)",
    "save_freq":     r"save_checkpoint_freq:\s+(\d+)",
    "time_consumed": r"Time consumed:\s+(.+)",
    "pytorch_ver":   r"PyTorch:\s+(.+)",
}


def parse_log(path):
    """Parse a training log.

    Returns:
        config: dict of display facts (best-effort regex scrape of the log's
            own option dump)
        training: list of {ts, epoch, iter, lr, losses: {name: value}}
        val: list of {iter, metrics: {name: {value, best, best_iter}}}
        loss_keys: list of loss names in first-seen order
        metric_names: list of validation metric names in first-seen order
    """
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    lines = text.splitlines()

    training = []
    loss_keys = []
    for line in lines:
        m = RE_HEADER.search(line)
        if not m:
            continue
        it = int(m.group("iter").replace(",", ""))
        losses = {}
        for key, val_str in RE_LOSS.findall(line):
            try:
                losses[key] = float(val_str)
            except ValueError:
                continue
            if key not in loss_keys:
                loss_keys.append(key)
        training.append({
            "ts": m.group("ts"),
            "epoch": int(m.group("epoch")),
            "iter": it,
            "lr": float(m.group("lr")),
            "losses": losses,
        })

    val = []
    metric_names = []
    n = len(lines)
    i = 0
    while i < n:
        hm = RE_VAL_HEADER.search(lines[i])
        if not hm:
            i += 1
            continue
        block_metrics = {}
        j = i + 1
        while j < n:
            mm = RE_VAL_METRIC.search(lines[j])
            if not mm:
                break
            name, value, best, best_iter = mm.groups()
            block_metrics[name] = {
                "value": float(value),
                "best": float(best),
                "best_iter": int(best_iter.replace(",", "")),
            }
            if name not in metric_names:
                metric_names.append(name)
            j += 1
        if block_metrics:
            iter_val = None
            for k in range(i, max(i - 30, -1), -1):
                im = RE_ITER_ANY.search(lines[k])
                if im:
                    iter_val = int(im.group(1).replace(",", ""))
                    break
            val.append({"iter": iter_val if iter_val is not None else 0, "metrics": block_metrics})
        i = j if j > i else i + 1

    config = {}
    for key, pat in CONFIG_PATTERNS.items():
        m = re.search(pat, text, re.MULTILINE)
        if m:
            config[key] = m.group(1).strip()

    config["start_iter"] = training[0]["iter"] if training else 0
    config["end_iter"]   = training[-1]["iter"] if training else 0
    config["start_ts"]   = training[0]["ts"] if training else "N/A"
    config["end_ts"]     = training[-1]["ts"] if training else "N/A"

    return config, training, val, loss_keys, metric_names


# ─────────────────────────── LOSS WEIGHTS (from --config) ─────────────────────

def load_loss_weights(config_path, loss_keys):
    """Best-effort mapping of logged loss keys (e.g. 'l_pix') to a weight,
    read from a training YAML's `train.*_opt.loss_weight` entries.

    Matching is name-based: an option block named e.g. `pixel_opt` is matched
    against a logged key `l_pix` if either name is a prefix of the other
    (after stripping the `l_`/`_opt` decoration). Unmatched or unresolved
    loss keys default to weight 1.0.
    """
    weights = {k: 1.0 for k in loss_keys}
    resolved = {k: False for k in loss_keys}
    if not config_path or yaml is None:
        return weights, resolved

    try:
        with open(config_path, "r") as f:
            opt = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return weights, resolved

    train_opt = (opt or {}).get("train", {}) or {}
    opt_blocks = {
        key[:-4]: block for key, block in train_opt.items()
        if key.endswith("_opt") and isinstance(block, dict)
    }

    for key in loss_keys:
        short = key[2:] if key.startswith("l_") else key  # 'l_pix' -> 'pix'
        for opt_name, block in opt_blocks.items():
            if opt_name.startswith(short) or short.startswith(opt_name):
                weights[key] = float(block.get("loss_weight", 1.0))
                resolved[key] = True
                break

    return weights, resolved


# ─────────────────────────── PER-BLOCK STATS ──────────────────────────────────

def compute_blocks(training, val):
    """Partition training records into blocks bounded by validation
    iterations: (0, v1], (v1, v2], ... plus a trailing partial block after the
    last validation (if training continued past it, e.g. an in-progress run).
    """
    records = sorted(training, key=lambda r: r["iter"])
    boundaries = sorted({v["iter"] for v in val if v.get("iter")})

    blocks = []
    prev = 0
    for b in boundaries:
        block_recs = [r for r in records if prev < r["iter"] <= b]
        if block_recs:
            blocks.append((prev, b, block_recs))
        prev = b

    tail_recs = [r for r in records if r["iter"] > prev]
    if tail_recs:
        blocks.append((prev, tail_recs[-1]["iter"], tail_recs))

    if not blocks and records:
        blocks.append((0, records[-1]["iter"], records))

    return blocks


def block_loss_stats(block_recs, loss_keys, weights):
    """mean/std per loss key over one block, plus the weighted total."""
    stats = {}
    for key in loss_keys:
        vals = [r["losses"][key] for r in block_recs if key in r["losses"]]
        if vals:
            stats[key] = (float(np.mean(vals)), float(np.std(vals)))
        else:
            stats[key] = None

    totals = []
    for r in block_recs:
        if not r["losses"]:
            continue
        totals.append(sum(weights.get(k, 1.0) * v for k, v in r["losses"].items()))
    stats["__total__"] = (float(np.mean(totals)), float(np.std(totals))) if totals else None
    return stats


# ─────────────────────────── SMOOTHING / FORMATTING ───────────────────────────

def smooth(values, window=50):
    """Causal moving average."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def fmt_k(x, _):
    return f"{int(x/1000)}k" if x >= 1000 else str(int(x))


# ─────────────────────────── PLOT ─────────────────────────────────────────────

def plot_dashboard(config, training, val, loss_keys, metric_names, weights, out_path):
    iters = np.array([r["iter"] for r in training])
    lrs   = np.array([r["lr"] for r in training])
    W = 50  # smoothing window

    total = np.array([
        sum(weights.get(k, 1.0) * r["losses"].get(k, 0.0) for k in loss_keys)
        for r in training
    ]) if training else np.array([])

    per_metric_series = {}
    for name in metric_names:
        m_iters, m_vals, m_best, m_best_iter = [], [], [], []
        for v in val:
            if name in v["metrics"]:
                m_iters.append(v["iter"])
                m_vals.append(v["metrics"][name]["value"])
                m_best.append(v["metrics"][name]["best"])
                m_best_iter.append(v["metrics"][name]["best_iter"])
        per_metric_series[name] = (np.array(m_iters), np.array(m_vals), m_best, m_best_iter)

    # Build a dynamic list of (title, draw_fn) panels, packed 2-per-row below
    # a full-width summary card.
    panels = []

    def draw_losses(ax):
        for idx, key in enumerate(loss_keys):
            color = PALETTE[idx % len(PALETTE)]
            vals = np.array([r["losses"].get(key, np.nan) for r in training])
            ax.plot(iters, vals, color=color, alpha=0.15, linewidth=0.5)
            ax.plot(iters, smooth(vals, W), color=color, linewidth=1.4,
                    label=f"{key} (w={weights.get(key, 1.0):g})")
        ax.set_xlabel("Iteration", color=MUTED)
        ax.set_ylabel("Loss Value", color=MUTED)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_k))
        ax.legend(loc="upper right", fontsize=7.5)
        ax.grid(True, alpha=0.4)

    if loss_keys:
        panels.append(("Individual Losses (raw + smoothed)", draw_losses))

    def draw_total(ax):
        ax.plot(iters, total, color=PALETTE[3], alpha=0.15, linewidth=0.5)
        ax.plot(iters, smooth(total, W), color=PALETTE[3], linewidth=1.8, label="Total weighted loss")
        ax.set_xlabel("Iteration", color=MUTED)
        ax.set_ylabel("Loss Value", color=MUTED)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_k))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.4)

    if loss_keys:
        panels.append(("Total Weighted Loss", draw_total))

    def make_metric_panel(name):
        def draw(ax):
            m_iters, m_vals, m_best, m_best_iter = per_metric_series[name]
            if len(m_iters):
                best_val = m_best[-1]
                best_it = m_best_iter[-1]
                ax.plot(m_iters, m_vals, color=PALETTE[4], linewidth=1.8,
                        marker="o", markersize=4, markerfacecolor=PALETTE[4], label=name)
                ax.axhline(best_val, color=ACCENT_BEST, linewidth=1.2, linestyle="--",
                           label=f"Best: {best_val:.4f} @ {best_it:,}")
                ax.axvline(best_it, color=ACCENT_BEST, linewidth=0.8, linestyle=":", alpha=0.7)
                best_idx = int(np.argmax(m_vals))
                ax.scatter([m_iters[best_idx]], [m_vals[best_idx]], color=ACCENT_BEST, s=70, zorder=5)
                ax.scatter([m_iters[-1]], [m_vals[-1]], color=PALETTE[4], s=70, marker="D", zorder=5,
                           label=f"Final: {m_vals[-1]:.4f}")
            ax.set_title(name, color=TEXT, fontsize=10, pad=6)
            ax.set_xlabel("Iteration", color=MUTED)
            ax.set_ylabel(name, color=MUTED)
            ax.xaxis.set_major_formatter(FuncFormatter(fmt_k))
            ax.legend(loc="best", fontsize=7, ncol=2)
            ax.grid(True, alpha=0.4)
        return draw

    for name in metric_names:
        panels.append((f"Validation: {name}", make_metric_panel(name)))

    def draw_lr(ax):
        ax.plot(iters, lrs, color=ACCENT_LR, linewidth=1.2)
        ax.set_xlabel("Iteration", color=MUTED)
        ax.set_ylabel("LR", color=MUTED)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_k))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2e}"))
        ax.grid(True, alpha=0.4)

    panels.append(("Learning Rate Schedule", draw_lr))

    def draw_composition(ax):
        raw_total = np.sum([np.array([r["losses"].get(k, 0.0) for r in training]) for k in loss_keys], axis=0)
        fracs = []
        for key in loss_keys:
            vals = np.array([r["losses"].get(key, 0.0) for r in training])
            fracs.append(smooth(vals / (raw_total + 1e-12), W))
        ax.stackplot(iters, *fracs, colors=[PALETTE[i % len(PALETTE)] for i in range(len(loss_keys))],
                     labels=loss_keys, alpha=0.75)
        ax.set_xlabel("Iteration", color=MUTED)
        ax.set_ylabel("Fraction", color=MUTED)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_k))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.4)

    if len(loss_keys) > 1:
        panels.append(("Loss Composition (fraction of raw total)", draw_composition))

    n_panel_rows = (len(panels) + 1) // 2
    n_rows = 1 + n_panel_rows
    fig = plt.figure(figsize=(20, 6 + 4.2 * n_panel_rows), facecolor=BG)
    fig.text(0.5, 0.985 - 0.01 * (14 / n_rows), "MambaFusion — Training Diagnostic Dashboard",
              ha="center", va="top", color=TEXT, fontsize=18, fontweight="bold", fontfamily="monospace")
    fig.text(0.5, 0.965, f"{config.get('name', 'N/A')}  ·  {config.get('start_ts', '').split()[0] if config.get('start_ts') else ''}"
                          f"  →  {config.get('end_ts', '').split()[0] if config.get('end_ts') else ''}",
              ha="center", va="top", color=MUTED, fontsize=10)

    gs = gridspec.GridSpec(n_rows, 2, figure=fig, left=0.06, right=0.97, top=0.965, bottom=0.03,
                            hspace=0.55, wspace=0.35, height_ratios=[1.1] + [1.6] * n_panel_rows)

    # ── Summary card (full width, row 0) ────────────────────────────────────
    ax_info = fig.add_subplot(gs[0, :])
    ax_info.set_facecolor(PANEL)
    ax_info.axis("off")

    best_summary = []
    for name in metric_names:
        _, m_vals, m_best, m_best_iter = per_metric_series[name]
        if len(m_best):
            best_summary.append((name, m_best[-1], m_best_iter[-1], m_vals[-1] if len(m_vals) else None))

    col1 = [
        ("Model",       config.get("name", "—")),
        ("Model Type",  config.get("model_type", "—")),
        ("Scale",       f"×{config.get('scale', '—')}"),
        ("Parameters",  config.get("parameters", "—")),
        ("Num Frames",  config.get("num_frames", "—")),
        ("GPUs",        config.get("num_gpu", "—")),
    ]
    col2 = [
        ("Optimizer",   config.get("optimizer", "—")),
        ("LR (init)",   config.get("lr", "—")),
        ("Weight Decay", config.get("weight_decay", "—")),
        ("Scheduler",   config.get("scheduler", "—")),
        ("Batch/GPU",   config.get("batch_per_gpu", "—")),
        ("Val Freq",    config.get("val_freq", "—")),
    ]
    col3 = [
        ("Total Iters", f"{int(config.get('total_iter', 0)):,}" if config.get("total_iter") else "—"),
        ("Completed",   f"{config.get('end_iter', 0):,}"),
        ("Train Images", config.get("train_images", "—")),
        ("Val Images",  config.get("val_images", "—")),
        ("Time Consumed", config.get("time_consumed", "—")),
        ("PyTorch",     config.get("pytorch_ver", "—")),
    ]
    col4 = [("Losses tracked", ", ".join(loss_keys) if loss_keys else "—")]
    for name, best_v, best_it, final_v in best_summary:
        col4.append((f"Best {name}", f"{best_v:.4f} @ {best_it:,}"))

    cols = [col1, col2, col3, col4]
    x_positions = [0.0, 0.25, 0.50, 0.75]
    for ci, (col, xp) in enumerate(zip(cols, x_positions)):
        for ri, (label, val_str) in enumerate(col):
            y = 0.90 - ri * 0.135
            ax_info.text(xp, y, label + ":", color=MUTED, fontsize=8, ha="left", va="top",
                         transform=ax_info.transAxes)
            ax_info.text(xp + 0.005, y - 0.055, str(val_str), color=TEXT, fontsize=8.5,
                         ha="left", va="top", transform=ax_info.transAxes)

    line = plt.Line2D([0.01, 0.99], [-0.07, -0.07], color=BORDER, linewidth=0.8,
                       transform=ax_info.transAxes, clip_on=False)
    ax_info.add_line(line)

    # ── Remaining panels, 2 per row ──────────────────────────────────────────
    for idx, (title, draw_fn) in enumerate(panels):
        row = 1 + idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=10, pad=6)
        draw_fn(ax)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ─────────────────────────── MARKDOWN SUMMARY ─────────────────────────────────

def write_summary_md(config, training, val, loss_keys, metric_names, weights, weight_resolved, out_path):
    lines = []
    lines.append(f"# Training Summary — {config.get('name', 'N/A')}")
    lines.append("")
    lines.append(f"- Model type: `{config.get('model_type', '—')}`")
    lines.append(f"- Scale: ×{config.get('scale', '—')}, Parameters: {config.get('parameters', '—')}")
    lines.append(f"- GPUs: {config.get('num_gpu', '—')}, Batch/GPU: {config.get('batch_per_gpu', '—')}")
    lines.append(f"- Optimizer: {config.get('optimizer', '—')}, Initial LR: {config.get('lr', '—')}, "
                 f"Scheduler: {config.get('scheduler', '—')}")
    lines.append(f"- Planned iters: {config.get('total_iter', '—')}, Completed: {config.get('end_iter', 0):,}")
    lines.append(f"- Time consumed: {config.get('time_consumed', '—')}")
    lines.append(f"- Range: {config.get('start_ts', '—')} → {config.get('end_ts', '—')}")
    lines.append("")

    lines.append("## Loss Weights")
    lines.append("")
    lines.append("| Loss | Weight | Source |")
    lines.append("|---|---|---|")
    for key in loss_keys:
        source = "config" if weight_resolved.get(key) else "default (1.0)"
        lines.append(f"| `{key}` | {weights.get(key, 1.0):g} | {source} |")
    lines.append("")

    # ── Per-block loss stats (mean ± std), blocks bounded by val iters ──────
    blocks = compute_blocks(training, val)
    lines.append("## Loss Statistics per Validation Interval (mean ± std)")
    lines.append("")
    header = "| Block (iters] | N | " + " | ".join(loss_keys) + " | Total (weighted) |"
    sep = "|---|---|" + "---|" * len(loss_keys) + "---|"
    lines.append(header)
    lines.append(sep)
    for start, end, recs in blocks:
        stats = block_loss_stats(recs, loss_keys, weights)
        row = [f"({start:,}, {end:,}]", str(len(recs))]
        for key in loss_keys:
            s = stats.get(key)
            row.append(f"{s[0]:.5f} ± {s[1]:.5f}" if s else "—")
        tot = stats.get("__total__")
        row.append(f"{tot[0]:.5f} ± {tot[1]:.5f}" if tot else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Whole-run stats ──────────────────────────────────────────────────────
    if training:
        lines.append("## Loss Statistics — Whole Run (mean ± std)")
        lines.append("")
        whole_stats = block_loss_stats(training, loss_keys, weights)
        for key in loss_keys:
            s = whole_stats.get(key)
            if s:
                lines.append(f"- `{key}`: {s[0]:.5f} ± {s[1]:.5f}")
        tot = whole_stats.get("__total__")
        if tot:
            lines.append(f"- **Total (weighted)**: {tot[0]:.5f} ± {tot[1]:.5f}")
        lines.append("")

    # ── Validation metrics ───────────────────────────────────────────────────
    if val:
        lines.append("## Validation Metrics")
        lines.append("")
        header = "| Iter | " + " | ".join(metric_names) + " |"
        sep = "|---|" + "---|" * len(metric_names)
        lines.append(header)
        lines.append(sep)
        for v in val:
            row = [f"{v['iter']:,}"]
            for name in metric_names:
                m = v["metrics"].get(name)
                row.append(f"{m['value']:.4f}" if m else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        lines.append("### Best per Metric")
        lines.append("")
        lines.append("| Metric | Best | @ Iter | Final | Δ (final − best) |")
        lines.append("|---|---|---|---|---|")
        for name in metric_names:
            recs_with_metric = [v for v in val if name in v["metrics"]]
            if not recs_with_metric:
                continue
            last = recs_with_metric[-1]["metrics"][name]
            final_val = recs_with_metric[-1]["metrics"][name]["value"]
            best_val = last["best"]
            best_iter = last["best_iter"]
            delta = final_val - best_val
            lines.append(f"| {name} | {best_val:.4f} | {best_iter:,} | {final_val:.4f} | {delta:+.4f} |")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def print_console_summary(config, training, val, loss_keys, metric_names, weights):
    sep = "─" * 62
    print(f"\n{'═'*62}")
    print(f"  MambaFusion Training Summary — {config.get('name', 'N/A')}")
    print(f"{'═'*62}")
    print(f"  Records: {len(training):,} training lines, {len(val)} validation checkpoints")
    print(f"  Losses tracked: {', '.join(loss_keys) if loss_keys else '—'}")
    print(f"  Metrics tracked: {', '.join(metric_names) if metric_names else '—'}")

    blocks = compute_blocks(training, val)
    if blocks:
        print(f"\n  ── Loss stats per validation interval ({sep[:20]}")
        for start, end, recs in blocks:
            stats = block_loss_stats(recs, loss_keys, weights)
            parts = [f"{k}={stats[k][0]:.5f}±{stats[k][1]:.5f}" for k in loss_keys if stats.get(k)]
            tot = stats.get("__total__")
            tot_str = f" total={tot[0]:.5f}±{tot[1]:.5f}" if tot else ""
            print(f"  ({start:,}, {end:,}]  n={len(recs):<5} " + "  ".join(parts) + tot_str)

    if val:
        print(f"\n  ── Best per metric {sep[:20]}")
        for name in metric_names:
            recs_with_metric = [v for v in val if name in v["metrics"]]
            if not recs_with_metric:
                continue
            last = recs_with_metric[-1]["metrics"][name]
            print(f"  {name}: best={last['best']:.4f} @ iter {last['best_iter']:,}  "
                  f"final={recs_with_metric[-1]['metrics'][name]['value']:.4f}")
    print(f"\n{'═'*62}\n")


# ──────────────────────────── ENTRY POINT ─────────────────────────────────────

def run(log_path, output_dir, config_path=None):
    """Run the full analysis for one log file, writing outputs into output_dir.
    Callable directly (e.g. from run_analysis.py) as well as via the CLI."""
    os.makedirs(output_dir, exist_ok=True)

    config, training, val, loss_keys, metric_names = parse_log(log_path)
    if not training:
        print(f"[analyze_logfile][WARN] No training records parsed from {log_path}.")

    weights, resolved = load_loss_weights(config_path, loss_keys)

    print_console_summary(config, training, val, loss_keys, metric_names, weights)

    dashboard_path = os.path.join(output_dir, "logfile_dashboard.png")
    if training:
        plot_dashboard(config, training, val, loss_keys, metric_names, weights, dashboard_path)
        print(f"[analyze_logfile] Dashboard saved -> {dashboard_path}")

    summary_path = os.path.join(output_dir, "logfile_summary.md")
    write_summary_md(config, training, val, loss_keys, metric_names, weights, resolved, summary_path)
    print(f"[analyze_logfile] Summary saved -> {summary_path}")

    return {
        "config": config, "training": training, "val": val,
        "loss_keys": loss_keys, "metric_names": metric_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Parse a MambaFusion/BasicSR training log into a dashboard + summary.")
    parser.add_argument("--log", required=True, help="Path to the training .log file")
    parser.add_argument("--output-dir", default=None,
                         help="Directory to write logfile_dashboard.png / logfile_summary.md "
                              "(default: analysis/outputs/<run name>/)")
    parser.add_argument("--config", default=None,
                         help="Optional training YAML, used to resolve loss weights")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"[ERROR] File not found: {args.log}")
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        # Infer the run name straight from the log so this works standalone,
        # without requiring --config.
        with open(args.log, "r", errors="ignore") as f:
            head = f.read(4000)
        m = re.search(r"^\s*name:\s+(.+)$", head, re.MULTILINE)
        name = m.group(1).strip() if m else os.path.splitext(os.path.basename(args.log))[0]
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(repo_root, "analysis", "outputs", name)

    run(args.log, output_dir, config_path=args.config)


if __name__ == "__main__":
    main()
