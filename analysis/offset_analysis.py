#!/usr/bin/env python3
"""
DCN Offset Magnitude Analysis Across Training Checkpoints

For every net_g_*.pth checkpoint in a given experiment models/ directory,
this script runs inference on the full test set and measures the mean
absolute offset magnitude produced by each of the three DCN projection layers
in BurstAlign:

  offset_proj_lv2   — coarse pyramid level, resolution H/2 x W/2
  offset_proj_lv1   — fine   pyramid level, resolution H   x W
  casc_offset_proj  — cascading refinement,  resolution H   x W

Each projection layer outputs a tensor of shape [B, padded_C, H, W] where
the channels are interleaved as (Δx, Δy, mask) for each of the K=36 kernel
points.  Only the Δx and Δy channels (72 of the 112) are used to compute
mean |offset| in pixels at that level's spatial resolution.

Outputs:
  - Log file  : analysis/outputs/ablation_logs/offset_analysis_<timestamp>.log
  - Plot (PNG): analysis/outputs/ablation_logs/offset_analysis_<timestamp>.png

Run with:
    torchrun --nproc_per_node=4 analysis/offset_analysis.py \\
        --models_dir /path/to/experiments/STHAT_GW/models \\
        [--config      main/config_newarch.yml] \\
        [--data_root   /path/to/RealBSR_RAW_testpatch] \\
        [--seed        42] \\
        [--num_frames  5] \\
        [--log_dir     analysis/outputs/ablation_logs]
"""

import os
import sys
import glob
import pickle
import random
import logging
import argparse
import re
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.distributed as dist
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from burstISP.archs.mambafusion_arch import MambaFusionNet


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path, rank):
    logger = logging.getLogger('offset_analysis')
    logger.setLevel(logging.INFO)
    if rank == 0:
        fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Data loading (LQ only — GT not needed for offset measurement)
# ---------------------------------------------------------------------------

def load_lq(burst_dir, lq_indices):
    """Return stacked LQ frames as FloatTensor [N, 4, H, W] in [0, 1]."""
    pkl_file = glob.glob(os.path.join(burst_dir, '*.pkl'))[0]
    with open(pkl_file, 'rb') as f:
        meta = pickle.load(f)

    subtract_bl = not meta.get('black_level_subtracted', False)
    lq_paths    = sorted(glob.glob(os.path.join(burst_dir, '*_x1_*.png')))

    frames = []
    for idx in lq_indices:
        img   = cv2.imread(lq_paths[idx], cv2.IMREAD_UNCHANGED)
        frame = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        if subtract_bl:
            frame = frame - 512.0
        frame = frame / 16383.0
        frames.append(frame)

    return torch.stack(frames, dim=0)


def normal_indices(count=5, total_lq=14, seed=None):
    """Same frame-selection logic as BurstImageDataset._generate_lq_indices."""
    rng    = random.Random(seed)
    others = rng.sample(range(1, total_lq), count - 1)
    others.insert(count // 2, 0)
    return others


# ---------------------------------------------------------------------------
# Offset hooks
# ---------------------------------------------------------------------------

def _xy_mean_abs(offset_tensor, K):
    """Mean absolute displacement across all Δx/Δy channels, batch, and space.

    offset_tensor : [B, padded_C, H, W]
    K             : number of DCN kernel points (offset_groups × 9)

    Returns a Python float — mean |offset| in pixels at this level's resolution.
    """
    xy_idx = []
    for k in range(K):
        xy_idx.extend([k * 3, k * 3 + 1])          # Δx, Δy per point
    xy = offset_tensor[:, xy_idx, :, :]              # [B, 2K, H, W]
    return xy.abs().mean().item()


class OffsetAccumulator:
    """Collects per-call mean |offset| from the three projection layers."""

    def __init__(self, K):
        self.K = K
        self._data = {'lv2': [], 'lv1': [], 'casc': []}
        self._handles = []

    def register(self, model):
        align = model.alignment

        def make_hook(name):
            def hook(module, inp, output):
                self._data[name].append(_xy_mean_abs(output.detach(), self.K))
            return hook

        self._handles = [
            align.offset_proj_lv2.register_forward_hook(make_hook('lv2')),
            align.offset_proj_lv1.register_forward_hook(make_hook('lv1')),
            align.casc_offset_proj.register_forward_hook(make_hook('casc')),
        ]

    def clear(self):
        for v in self._data.values():
            v.clear()

    def burst_means(self):
        """Return (mean_lv2, mean_lv1, mean_casc) for the calls since last clear()."""
        return (
            float(np.mean(self._data['lv2'])),
            float(np.mean(self._data['lv1'])),
            float(np.mean(self._data['casc'])),
        )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Inference pass for one checkpoint
# ---------------------------------------------------------------------------

def run_checkpoint(model, all_burst_dirs, accumulator, args, device, rank, world_size):
    """Run inference on this rank's subset of bursts.

    Returns list of (name, mean_lv2, mean_lv1, mean_casc) — one entry per burst.
    """
    local_dirs = all_burst_dirs[rank::world_size]
    results    = []

    for i, burst_dir in enumerate(local_dirs):
        global_idx = rank + i * world_size
        indices    = normal_indices(count=args.num_frames, total_lq=14,
                                    seed=args.seed + global_idx)
        lq = load_lq(burst_dir, indices)

        accumulator.clear()

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _ = model(lq.unsqueeze(0).to(device))

        m_lv2, m_lv1, m_casc = accumulator.burst_means()
        results.append((os.path.basename(burst_dir), m_lv2, m_lv1, m_casc))

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def save_plot(iters, means, stds, output_path):
    """
    iters : list[int]
    means : list[(m_lv2, m_lv1, m_casc)]
    stds  : list[(s_lv2, s_lv1, s_casc)]
    """
    iters    = np.array(iters)
    lv2_m    = np.array([m[0] for m in means])
    lv1_m    = np.array([m[1] for m in means])
    casc_m   = np.array([m[2] for m in means])
    lv2_s    = np.array([s[0] for s in stds])
    lv1_s    = np.array([s[1] for s in stds])
    casc_s   = np.array([s[2] for s in stds])

    fig, ax  = plt.subplots(figsize=(11, 5))
    colours  = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for vals, errs, label, col in [
        (lv2_m,  lv2_s,  'offset_proj_lv2  (coarse, H/2)', colours[0]),
        (lv1_m,  lv1_s,  'offset_proj_lv1  (fine,   H)',   colours[1]),
        (casc_m, casc_s, 'casc_offset_proj (cascade, H)',  colours[2]),
    ]:
        ax.plot(iters, vals, marker='o', color=col, label=label)
        ax.fill_between(iters, vals - errs, vals + errs,
                        alpha=0.15, color=col)

    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Mean |Offset|  (pixels at level resolution)', fontsize=12)
    ax.set_title('BurstAlign DCN Offset Magnitude vs. Training Iteration', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', required=True,
                        help='Folder containing net_g_<iter>.pth checkpoints')
    parser.add_argument('--config', default='main/config.yml',
                        help='YAML config with network_g architecture params')
    parser.add_argument('--data_root',
                        default='/groups/rls/blozanod/MambaFusion/dataset/RealBSR_RAW_testpatch',
                        help='Root directory of test burst folders')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base seed for frame selection')
    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--log_dir', default='analysis/outputs/ablation_logs')
    args = parser.parse_args()

    # --- Distributed ---
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    device     = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_dir   = (args.log_dir if os.path.isabs(args.log_dir)
                 else os.path.join(repo_root, args.log_dir))
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = os.path.join(log_dir, f'offset_analysis_{timestamp}.log')
    logger    = setup_logger(log_path, rank)

    # --- Architecture config ---
    config_path = (args.config if os.path.isabs(args.config)
                   else os.path.join(repo_root, args.config))
    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)
    net_opt            = opt['network_g']
    net_opt['is_train'] = False

    K = net_opt['offset_groups'] * 9   # 4 × 9 = 36 kernel points

    # --- Checkpoints (sorted by iteration) ---
    ckpt_paths = sorted(
        glob.glob(os.path.join(args.models_dir, 'net_g_[0-9]*.pth')),
        key=lambda p: int(re.search(r'net_g_(\d+)\.pth', os.path.basename(p)).group(1))
    )
    if not ckpt_paths:
        if rank == 0:
            print(f'ERROR: No net_g_<iter>.pth files found in {args.models_dir}')
        dist.destroy_process_group()
        return

    # --- Dataset ---
    all_dirs = sorted(glob.glob(os.path.join(args.data_root, '*')))
    n_total  = len(all_dirs)

    if rank == 0:
        logger.info(f'Models dir    : {args.models_dir}')
        logger.info(f'Checkpoints   : {len(ckpt_paths)}')
        logger.info(f'Data root     : {args.data_root}')
        logger.info(f'Test bursts   : {n_total}  ({n_total // world_size}–{-(-n_total // world_size)} per GPU)')
        logger.info(f'K (kpt/group) : {K}  |  padded_C=112  |  xy_channels=72')
        logger.info(f'Seed          : {args.seed}\n')

    # Build model once; reload state dict per checkpoint
    model       = MambaFusionNet(**net_opt).to(device)
    model.eval()
    accumulator = OffsetAccumulator(K)
    accumulator.register(model)

    iters_list  = []
    means_list  = []
    stds_list   = []

    for ckpt_path in ckpt_paths:
        basename = os.path.basename(ckpt_path)
        iter_num = int(re.search(r'net_g_(\d+)\.pth', basename).group(1))

        ckpt  = torch.load(ckpt_path, map_location=device)
        state = ckpt.get('params_ema', ckpt.get('params', ckpt.get('state_dict', ckpt)))
        model.load_state_dict(state, strict=True)

        local_results = run_checkpoint(model, all_dirs, accumulator, args, device, rank, world_size)

        # Gather from all ranks
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_results)

        if rank == 0:
            flat = [r for rank_res in gathered for r in rank_res]
            n    = len(flat)

            arr_lv2  = np.array([r[1] for r in flat])
            arr_lv1  = np.array([r[2] for r in flat])
            arr_casc = np.array([r[3] for r in flat])

            m = (arr_lv2.mean(), arr_lv1.mean(), arr_casc.mean())
            s = (arr_lv2.std(),  arr_lv1.std(),  arr_casc.std())

            logger.info(
                f'iter {iter_num:>7d} | '
                f'lv2={m[0]:.5f}±{s[0]:.5f}  '
                f'lv1={m[1]:.5f}±{s[1]:.5f}  '
                f'casc={m[2]:.5f}±{s[2]:.5f}  '
                f'({n} bursts)'
            )

            iters_list.append(iter_num)
            means_list.append(m)
            stds_list.append(s)

        torch.cuda.empty_cache()

    if rank == 0:
        # Summary table
        SEP = '=' * 78
        logger.info('\n' + SEP)
        logger.info('  OFFSET MAGNITUDE SUMMARY  (mean ± std of mean|offset| in pixels)')
        logger.info('  Each value is averaged over all bursts × all frames × all spatial positions')
        logger.info(SEP)
        logger.info(f'  {"Iter":>8}  {"lv2 mean":>10}  {"lv2 std":>9}  '
                    f'{"lv1 mean":>10}  {"lv1 std":>9}  '
                    f'{"casc mean":>10}  {"casc std":>9}')
        logger.info('  ' + '-' * 72)
        for it, (m2, m1, mc), (s2, s1, sc) in zip(iters_list, means_list, stds_list):
            logger.info(f'  {it:>8d}  {m2:>10.5f}  {s2:>9.5f}  '
                        f'{m1:>10.5f}  {s1:>9.5f}  '
                        f'{mc:>10.5f}  {sc:>9.5f}')
        logger.info(SEP)

        # Plot
        plot_path = os.path.join(log_dir, f'offset_analysis_{timestamp}.png')
        save_plot(iters_list, means_list, stds_list, plot_path)
        logger.info(f'\nPlot → {plot_path}')
        logger.info(f'Log  → {log_path}')

    accumulator.remove()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
