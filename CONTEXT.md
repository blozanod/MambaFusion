# MambaFusion — Repository Context

## Project Overview

MambaFusion is a **RAW burst super-resolution** model. Given a burst of N short-exposure, low-quality RAW frames, the model produces a single high-quality RGB image at 4× spatial upscale (8× in the packed RGGB coordinate system used throughout the code).

The project is a research prototype. Training runs on an HPC cluster (4× GPU). This local WSL2 repo is used for code development and inference/analysis only — the full dataset lives at `/groups/rls/blozanod/MambaFusion/dataset/` on the cluster.

---

## Dataset: RealBSR-RAW

- Paper: `papers/RealBSR-RAW.pdf`
- Each sample is a **burst folder** containing:
  - 14 LQ RAW frames: `*_x1_*.png` — stored as 4-channel packed RGGB, shape `[H/2, W/2, 4]`, 16-bit
  - 1 GT RGB image: `*_x4_rgb.png` — 48-bit RGB, at 4× the spatial resolution of the raw frames
  - 1 metadata file: `*.pkl` — holds camera info (`black_level_subtracted`, WB gains, CCM, etc.)
- **Black level**: 512 is subtracted if not already done per `meta_data['black_level_subtracted']`, then normalized to `[0, 1]` by dividing by 16383.
- **Scale convention**: LQ is packed RGGB at `[H, W, 4]` (so real spatial resolution is `[2H, 2W]` after unpacking). GT is at `[4×2H, 4×2W]` real pixels = `[8H, 8W]` in packed coordinates, hence `scale=8` in all configs.
- During training, 5 of the 14 frames are randomly sampled per iteration; the reference frame (index 0) is always placed at the center position `N//2`.
- The only data augmentation is a random RGGB-aware transpose (swaps G1↔G2 channels when transposing).
- Local preview dataset: `dataset/Inference_Set/` — 10 test-set bursts with GT.

---

## Architecture: `MambaFusionNet`

Defined in [burstISP/archs/mambafusion_arch.py](burstISP/archs/mambafusion_arch.py). Three sequential modules:

### 1. BurstAlign (`burstISP/archs/dcn_align_arch.py`)
- Pyramid (2-level) + Cascading + Deformable alignment via **DCNv4** (custom CUDA kernel, compiled at `burstISP/utils/DCNv4/`).
- Extracts features from each LQ frame, computes DCN offsets relative to the center/reference frame, and returns aligned feature maps `[B, N, C, H, W]` plus reference features `ref_feats`.
- Runs in **float32** (forced via `autocast(enabled=False)`) for numerical stability in offset computation.

### 2. ST_HAT Fusion (`burstISP/archs/st_hat_fusion_arch.py`)
- Input: aligned features `[B, N, C, H, W]`, output: single fused feature map `[B, C, H, W]`.
- **Stage 1** (depth_stage1 blocks, each with 3 sub-blocks):
  - `SpatialBlock`: window self-attention within each frame independently
  - `TemporalBlock`: per-pixel self-attention across the N frames (collapses burst dimension into batch dimension)
  - `SpatioTemporalBlock`: joint 3D window attention over (N × H × W) space
- **Stage 2** (dimension collapse):
  - `FusionBlock`: cross-attention where only the reference frame provides queries, all frames provide keys/values → collapses burst to single map
  - `SpatialBlock` for refinement
  - Residual from Stage 1 reference frame features (via 1×1 conv projection)
- **Stage 3** (depth_stage3 `RefinementBlock`s):
  - Each block: OCAB (overlapping cross-attention) → HAB (hybrid window + channel attention) → OCAB
  - Removes windowing artifacts and reweights features

### 3. MambaIRv2 Restoration (`burstISP/archs/mambairv2_arch.py`)
- Input: fused features `[B, C, H, W]` + `ref_feats` from BurstAlign (used as ControlNet-style injection when `global_skip=True`).
- Mamba-based state space model (SSM) backbone with window attention, produces upscaled output `[B, 3, 8H, 8W]`.
- Upsampler: `pixelshuffle` mode.

### Global Skip Connection (optional)
- Non-learnable Malvar-He-Cutler demosaicing (via kornia) + bicubic 4× upsampling of the center raw frame.
- Model learns residual on top of this baseline. Currently **disabled** (`global_skip: false`) in the active config.

---

## Code Structure

```
MambaFusion/
├── burstISP/              # Core library
│   ├── archs/             # Model architectures
│   │   ├── mambafusion_arch.py   ← Full model entry point
│   │   ├── st_hat_fusion_arch.py ← ST-HAT fusion module
│   │   ├── dcn_align_arch.py     ← BurstAlign with DCNv4
│   │   ├── mambairv2_arch.py     ← Restoration backbone
│   │   └── arch_util.py          ← Shared helpers (DCNv4Block, etc.)
│   ├── data/
│   │   └── burst_image_dataset.py ← BurstImageDataset
│   ├── models/
│   │   ├── mambafusion_model.py  ← Training/eval wrapper
│   │   └── sr_model.py           ← Base model class
│   ├── loss/losses.py            ← CharbonnierLoss, GWLoss, etc.
│   ├── metrics/psnr_ssim.py      ← calculate_psnr_srgb/linear, calculate_ssim_srgb
│   └── utils/
│       ├── img_util.py           ← ISP pipeline, image I/O
│       ├── options.py            ← YAML config parsing
│       └── DCNv4/                ← DCNv4 CUDA extension (must be compiled)
├── main/
│   ├── train.py                  ← Training entry point
│   ├── test.py                   ← Test/inference entry point
│   ├── config_newarch.yml        ← Current reference config
│   └── mamba_job.sh              ← HPC job submission script
├── analysis/                     ← Analysis and visualization scripts
│   ├── visualize_inference.py    ← Run model + ISP + save PNG
│   ├── visualize_progress.py     ← Training progress visualization
│   ├── visualize_dataset.py      ← Dataset inspection
│   └── analyze_logfile.py        ← Parse training logs
├── experiments/                  ← Saved runs (configs, checkpoints, logs)
│   └── STHAT_GW/                 ← Most recent completed run
├── dataset/
│   └── Inference_Set/            ← 10 local test bursts for inference
└── papers/                       ← Reference papers (RealBSR-RAW, MambaIR, HAT, etc.)
```

---

## Registry System

All models, datasets, archs, and losses are registered via decorators (e.g. `@ARCH_REGISTRY.register()`). They are selected in YAML configs by their `type` key and instantiated via `build_model()`, `build_dataset()`, etc.

---

## Training Details

- **Optimizer**: AdamW, lr=1e-4, betas=(0.9, 0.99)
- **Scheduler**: MultiStepLR (lr drops at milestones)
- **Loss**: Charbonnier (pixel) + optional GWLoss (gradient/edge, weight=0.25 in STHAT_GW)
- **Gradient accumulation**: `accumulation_steps=2` → effective batch size = 2 × batch_per_gpu × n_gpus
- **Mixed precision**: bfloat16 autocast in training; BurstAlign forced to float32
- **EMA**: supported via `ema_decay` config key
- **Logging**: file logger + optional TensorBoard; Weights & Biases supported

---

## Current Experiment: STHAT_GW (finished ~June 2026)

- 100k iterations, 4 GPUs, Charbonnier + GWLoss (0.25)
- **Best PSNR-sRGB**: ~24.09 dB @ ~35k iter; plateaued ~24.03 dB thereafter
- PSNR-Linear declined after ~10k iter (33.0 → 31.7), suggesting the sRGB ISP mapping is partially absorbing model error
- **Status**: Completed. Next step is empirical analysis to diagnose model failure modes before designing the next experiment.

---

## ISP Pipeline (Post-processing for Visualization)

The model outputs linear RAW-domain RGB. For visualization and sRGB PSNR:
1. Auto-exposure normalization (scale by `0.2 / mean`)
2. Clamp to `[1e-6, 1.0]`
3. Gamma correction (`^(1/2.2)`)
4. Smoothstep tone mapping (`3x² - 2x³`)

Implemented in `burstISP/utils/img_util.py: generate_processed_image_channel3`.
