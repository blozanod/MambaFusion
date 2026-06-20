#!/usr/bin/env python3
"""
Burst vs. Single-Image SR Ablation

Two-pass evaluation over the full test set:
  Pass 1 (Normal):  Reference frame at center, N-1 frames randomly sampled
                    (identical to validation during training)
  Pass 2 (All-Ref): All N input frames are the reference frame

If the metrics don't differ, the model is ignoring burst frames and
effectively doing single-image SR.

Run with:
    torchrun --nproc_per_node=4 analysis/burst_ablation.py \\
        --model_path /path/to/net_g_35000.pth \\
        [--config main/config_newarch.yml] \\
        [--data_root /path/to/RealBSR_RAW_testpatch] \\
        [--seed 42] \\
        [--crop_border 40] \\
        [--num_frames 5] \\
        [--log_dir analysis/ablation_logs]
"""

import os
import sys
import glob
import pickle
import random
import logging
import argparse
import copy
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from burstISP.archs.mambafusion_arch import MambaFusionNet
from burstISP.metrics.psnr_ssim import (
    calculate_psnr_srgb,
    calculate_psnr_linear,
    calculate_ssim_srgb,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logger(log_path, rank):
    logger = logging.getLogger('burst_ablation')
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


def load_burst(burst_dir, lq_indices):
    """Load GT image, metadata, and the requested LQ frames for one burst.

    Returns:
        lq  : FloatTensor [N, 4, H, W] in [0, 1]
        gt  : FloatTensor [3, H_gt, W_gt] in [0, 1]
        meta: dict loaded from .pkl
    """
    pkl_file = glob.glob(os.path.join(burst_dir, '*.pkl'))[0]
    with open(pkl_file, 'rb') as f:
        meta = pickle.load(f)

    subtract_bl = not meta.get('black_level_subtracted', False)

    # GT
    gt_file = glob.glob(os.path.join(burst_dir, '*_x4_rgb.png'))[0]
    gt_img = cv2.imread(gt_file, cv2.IMREAD_UNCHANGED)
    gt = torch.from_numpy(gt_img.astype(np.float32)).permute(2, 0, 1)
    if subtract_bl:
        gt = gt - 512.0
    gt = gt / 16383.0

    # LQ frames
    lq_paths = sorted(glob.glob(os.path.join(burst_dir, '*_x1_*.png')))
    frames = []
    for idx in lq_indices:
        img = cv2.imread(lq_paths[idx], cv2.IMREAD_UNCHANGED)
        frame = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        if subtract_bl:
            frame = frame - 512.0
        frame = frame / 16383.0
        frames.append(frame)

    lq = torch.stack(frames, dim=0)
    return lq, gt, meta


def normal_indices(count=5, total_lq=14, seed=None):
    """Same logic as BurstImageDataset._generate_lq_indices.

    Reference frame (index 0) is placed at the center position; the remaining
    count-1 frames are randomly drawn from [1, total_lq).
    """
    rng = random.Random(seed)
    ref_idx = 0
    others = rng.sample(range(1, total_lq), count - 1)
    others.insert(count // 2, ref_idx)
    return others


def run_inference(model, lq, device):
    """Forward pass. Returns output on CPU as FloatTensor [3, H_out, W_out]."""
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(lq.unsqueeze(0).to(device))
    return out.squeeze(0).float().cpu()


def compute_metrics(output, gt, meta, crop_border):
    """Compute PSNR-sRGB, PSNR-Linear, SSIM. All tensors CHW in [0, 1]."""
    p_srgb = calculate_psnr_srgb(output, gt, copy.deepcopy(meta), crop_border)
    p_lin  = calculate_psnr_linear(output, gt, crop_border)
    ssim   = calculate_ssim_srgb(output, gt, copy.deepcopy(meta), crop_border)
    return p_srgb, p_lin, ssim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True,
                        help='Path to checkpoint (.pth)')
    parser.add_argument('--config', default='main/config_newarch.yml',
                        help='YAML config for network architecture')
    parser.add_argument('--data_root',
                        default='/groups/rls/blozanod/MambaFusion/dataset/RealBSR_RAW_testpatch',
                        help='Root directory containing burst sub-folders')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base seed for Pass 1 frame selection (per-burst seeding)')
    parser.add_argument('--crop_border', type=int, default=40,
                        help='Border pixels excluded from metric computation')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of LQ frames per burst')
    parser.add_argument('--log_dir', default='analysis/ablation_logs',
                        help='Directory for log file output')
    args = parser.parse_args()

    # --- Distributed setup ---
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    device     = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # --- Logger ---
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_dir   = os.path.join(repo_root, args.log_dir) if not os.path.isabs(args.log_dir) else args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = os.path.join(log_dir, f'burst_ablation_{timestamp}.log')
    logger    = setup_logger(log_path, rank)

    if rank == 0:
        logger.info(f'Checkpoint : {args.model_path}')
        logger.info(f'Data root  : {args.data_root}')
        logger.info(f'Seed       : {args.seed}  |  Crop border: {args.crop_border}  |  Frames: {args.num_frames}')
        logger.info(f'World size : {world_size} GPU(s)')

    # --- Load architecture config ---
    config_path = args.config if os.path.isabs(args.config) else os.path.join(repo_root, args.config)
    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)
    net_opt = opt['network_g']
    net_opt['is_train'] = False

    # --- Build and load model ---
    model = MambaFusionNet(**net_opt).to(device)
    ckpt  = torch.load(args.model_path, map_location=device)
    state = ckpt.get('params_ema', ckpt.get('params', ckpt.get('state_dict', ckpt)))
    model.load_state_dict(state, strict=True)
    model.eval()

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f'Parameters : {n_params / 1e6:.3f}M')

    # --- Split dataset across GPUs ---
    all_dirs   = sorted(glob.glob(os.path.join(args.data_root, '*')))
    local_dirs = all_dirs[rank::world_size]
    n_total    = len(all_dirs)

    if rank == 0:
        logger.info(f'Test bursts : {n_total} total, ~{len(local_dirs)} per GPU')
        logger.info('Running two-pass inference...\n')

    # --- Two-pass inference ---
    local_results = []

    for i, burst_dir in enumerate(local_dirs):
        name = os.path.basename(burst_dir)
        # Deterministic per-burst seed: consistent regardless of which GPU handles it
        burst_seed = args.seed + (rank + i * world_size)

        # Pass 1 — Normal: ref at center, 4 random others
        idx_normal = normal_indices(count=args.num_frames, total_lq=14, seed=burst_seed)
        lq_n, gt, meta = load_burst(burst_dir, idx_normal)
        out_n = run_inference(model, lq_n, device)
        m_n   = compute_metrics(out_n, gt, meta, args.crop_border)

        # Pass 2 — All-Ref: every slot is the reference frame
        idx_all_ref = [0] * args.num_frames
        lq_a, _, _ = load_burst(burst_dir, idx_all_ref)
        out_a = run_inference(model, lq_a, device)
        m_a   = compute_metrics(out_a, gt, meta, args.crop_border)

        local_results.append({'name': name, 'normal': m_n, 'all_ref': m_a})

        if (i + 1) % 20 == 0:
            print(f'[Rank {rank}] {i + 1}/{len(local_dirs)} done', flush=True)

    # --- Gather all results on rank 0 ---
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_results)

    if rank == 0:
        all_results = [r for rank_res in gathered for r in rank_res]
        n = len(all_results)

        def col_mean(key, col):
            return float(np.mean([r[key][col] for r in all_results]))

        mn = (col_mean('normal', 0), col_mean('normal', 1), col_mean('normal', 2))
        ma = (col_mean('all_ref', 0), col_mean('all_ref', 1), col_mean('all_ref', 2))

        SEP = '=' * 72
        logger.info(SEP)
        logger.info(f'  BURST vs. SINGLE-IMAGE SR ABLATION  ({n} test bursts)')
        logger.info(SEP)
        logger.info(f'  {"Metric":<22} {"Pass 1  Normal":>16} {"Pass 2  All-Ref":>16} {"Delta (N-A)":>12}')
        logger.info('  ' + '-' * 68)

        rows = [
            ('PSNR-sRGB  (dB)', 0),
            ('PSNR-Linear (dB)', 1),
            ('SSIM', 2),
        ]
        for label, i_col in rows:
            d    = mn[i_col] - ma[i_col]
            sign = '+' if d >= 0 else ''
            logger.info(f'  {label:<22} {mn[i_col]:>16.4f} {ma[i_col]:>16.4f} {sign + f"{d:.4f}":>12}')

        logger.info(SEP)
        logger.info('  Delta > 0  →  burst helps (model uses extra frames)')
        logger.info('  Delta ≈ 0  →  model defaults to single-image SR')
        logger.info(SEP + '\n')

        # Per-sample table
        logger.info(f'  {"Burst Dir":<24} {"N:sRGB":>8} {"A:sRGB":>8} {"N:Lin":>8} {"A:Lin":>8} {"N:SSIM":>7} {"A:SSIM":>7}')
        logger.info('  ' + '-' * 72)
        for r in sorted(all_results, key=lambda x: x['name']):
            nm, am = r['normal'], r['all_ref']
            logger.info(
                f'  {r["name"]:<24}'
                f' {nm[0]:>8.4f} {am[0]:>8.4f}'
                f' {nm[1]:>8.4f} {am[1]:>8.4f}'
                f' {nm[2]:>7.4f} {am[2]:>7.4f}'
            )

        logger.info(f'\nFull log saved to: {log_path}')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
