#!/usr/bin/env python3
"""Pre-flight check for a training config: build the network, run one
forward/backward at the config's own batch size, and report peak GPU memory.

Catches shape errors and OOM in seconds instead of after a scheduler wait. The
per-arch `if __name__ == '__main__'` blocks cannot be used for this: the arch
package's __init__ auto-imports every *_arch.py, so executing one as a script
or with `python -m` registers its class twice and trips the registry assert.
This script imports the class instead.

Usage:
    python analysis/shape_check.py                      # default L5 Bayer config
    python analysis/shape_check.py main/configs/MF_STHAT_L5_PackedControl.yml
    python analysis/shape_check.py <config> --batch 4   # probe a larger batch
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from burstISP.utils.registry import ARCH_REGISTRY
import burstISP.archs  # noqa: F401  (populates ARCH_REGISTRY)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CFG = os.path.join(REPO_ROOT, 'main', 'configs', 'MF_STHAT_L5_BayerSpace.yml')


def human(n):
    return f'{n / 1e6:.3f}M'


def check(cfg_path, batch_override=None):
    with open(cfg_path) as f:
        opt = yaml.safe_load(f)

    net_opt = dict(opt['network_g'])
    net_type = net_opt.pop('type')
    ds = opt['datasets']['train']
    batch = batch_override if batch_override is not None else ds['batch_size_per_gpu']

    n_frames = net_opt['num_frames']
    img_size = net_opt['img_size']
    scale = net_opt['scale']

    print(f'\n=== {os.path.basename(cfg_path)} ===')
    print(f'  arch {net_type} | batch {batch} | {n_frames} frames | '
          f'{img_size}x{img_size} packed -> {img_size * scale}x{img_size * scale}')

    net = ARCH_REGISTRY.get(net_type)(**net_opt).cuda()

    total = sum(p.numel() for p in net.parameters())
    print('  parameters:')
    for name, mod in net.named_children():
        n = sum(p.numel() for p in mod.parameters())
        if n:
            print(f'    {name:<12} {human(n):>9}  {100 * n / total:5.1f}%')
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'    {"TOTAL":<12} {human(total):>9}   (trainable {human(trainable)})')

    lq = torch.randn(batch, n_frames, 4, img_size, img_size, device='cuda')
    gt = torch.randn(batch, 3, img_size * scale, img_size * scale, device='cuda')
    expected = (batch, 3, img_size * scale, img_size * scale)

    torch.cuda.reset_peak_memory_stats()
    net.train()

    # Mirrors MambaFusionModel.optimize_parameters: bf16 autocast, float() cast,
    # plain L1 on linear RGB.
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = net(lq)
    out = out.float()

    if tuple(out.shape) != expected:
        print(f'  FORWARD  FAIL  got {tuple(out.shape)}, expected {expected}')
        return False
    print(f'  forward  ok    {tuple(out.shape)}')

    loss = torch.nn.functional.l1_loss(out, gt)
    loss.backward()

    missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
    if missing:
        # find_unused_parameters is false in these configs, so DDP would error here.
        print(f'  BACKWARD FAIL  {len(missing)} parameter(s) got no gradient, first few:')
        for n in missing[:5]:
            print(f'      {n}')
        return False
    print(f'  backward ok    loss {loss.item():.4f}, all parameters received gradients')

    peak = torch.cuda.max_memory_allocated() / 2**30
    total_mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f'  peak memory    {peak:.2f} GiB of {total_mem:.1f} GiB '
          f'({100 * peak / total_mem:.0f}%) at batch {batch}')
    if peak / total_mem > 0.85:
        print('  NOTE: over 85% of the card. Training allocates more than this single '
              'step does (DDP gradient buckets, optimizer state). Consider '
              'batch_size_per_gpu 1 with accumulation_steps doubled to hold the '
              'effective batch.')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('configs', nargs='*', default=[DEFAULT_CFG],
                    help='config YAML path(s); defaults to the L5 Bayer config')
    ap.add_argument('--batch', type=int, default=None,
                    help="override batch_size_per_gpu (probe what fits)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print('ERROR: needs a GPU (DCNv4 is a CUDA extension). Run on a GPU node.')
        return 1

    ok = True
    for cfg in args.configs:
        try:
            ok &= check(cfg, args.batch)
        except torch.cuda.OutOfMemoryError:
            print(f'  OOM at batch {args.batch or "config default"} — lower '
                  f'batch_size_per_gpu and raise accumulation_steps to compensate.')
            ok = False
        torch.cuda.empty_cache()

    print('\n' + ('ALL CHECKS PASSED' if ok else 'CHECKS FAILED'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
