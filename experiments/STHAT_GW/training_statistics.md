# MambaFusion (STHAT_GW) Summary

## 1. Architecture
* **Network Structure**: `MambaFusionNet` (Parameters: 5,021,120)
* **Alignment Module (`BurstAlign`)**:
    * Feature extractors composed of multi-level `Conv2d` layers.
    * Offset convolutions utilizing `DCNv4Block` elements (Deformable Convolutions).
    * Bilinear Upsampling mechanism.
* **Fusion Module (`ST_HAT`)**:
    * **Stage 1**: Features a `SpatialBlock` (Window Attention), a `TemporalBlock` (Multihead Attention), and a `SpatioTemporalBlock` (Window Attention 3D).
    * **Fusion Block**: Implements a Multi-head Softmax attention routing.
    * **Stage 3**: Uses `RefinementBlock`s consisting of `OCAB` and `HAB` modules (Channel and Window Attention mechanisms).
* **Restoration Module (`MambaIRv2`)**:
    * Configured with `ASSB` structural units housing basic blocks and an `AttentiveLayer`.
    * Integrates `WindowAttention` and `ASSM` computing loops featuring `Selective_Scan` parameterization for Mamba state space tracking.

## 2. Configuration Settings
* **Data Settings**:
    * `scale`: 8
    * `num_frames`: 5
    * `img_size`: 80
    * `num_feat`: 64
* **Network Hyperparameters**:
    * ST_HAT Details: `fusion_ws`: 8, `fusion_feat`: 96, `fusion_heads`: 4. Depths are set to 3 for Stage 1 and Stage 3.
    * MambaIR Details: `embed_dim`: 48, `d_state`: 8, array of depths: `[5, 5, 5, 5]`, array of attention heads: `[4, 4, 4, 4]`, `window_size`: 16.
* **Training & Optimization**:
    * `total_iter`: 100,000
    * `warmup_iter`: 10,000
    * Optimizer: `AdamW` (`lr`: 0.0001, `betas`: [0.9, 0.99])
    * LR Scheduler: `MultiStepLR` (`milestones`: [35000, 65000, 85000])
* **Loss Operations**:
    * `pixel_opt`: `CharbonnierLoss` (`loss_weight`: 1.0)
    * `edge_opt`: `GWLoss` (`loss_weight`: 0.25)
* **Validation**:
    * Metrics: `psnr_srgb`, `psnr_linear`, `ssim`
    * Trigger Frequency: Every 5,000 iterations.

## 3. Training Statistics Generation
### Training Validation & Loss Statistics

| Iteration Block | Phase | Avg `l_pix` ± Std Dev | Avg `l_edge` ± Std Dev | PSNR-SRGB | PSNR-Linear | SSIM |
|---|---|---|---|---|---|---|
| 1 - 5,000 | LR Warmup | 0.024987 ± 0.009927 | 0.008514 ± 0.003269 | **22.4424** | **32.2881** | **0.6577** |
| 5,001 - 10,000 | LR Warmup | 0.017915 ± 0.003162 | 0.006190 ± 0.001699 | **23.1392** | **33.0122** | **0.7050** |
| 10,001 - 15,000 | Training | 0.015941 ± 0.003319 | 0.005519 ± 0.001504 | **23.3366** | 32.4119 | **0.7151** |
| 15,001 - 20,000 | Training | 0.013641 ± 0.002166 | 0.004856 ± 0.001094 | **23.5324** | 32.2757 | **0.7187** |
| 20,001 - 25,000 | Training | 0.011614 ± 0.001717 | 0.004144 ± 0.000854 | **23.7281** | 32.1474 | **0.7190** |
| 25,001 - 30,000 | Training | 0.010401 ± 0.001414 | 0.003770 ± 0.000811 | **23.7320** | 32.0693 | **0.7240** |
| 30,001 - 35,000 | Training | 0.009568 ± 0.001579 | 0.003480 ± 0.000794 | **23.8365** | 31.5651 | 0.7190 |
| 35,001 - 40,000 | Training | 0.008159 ± 0.001418 | 0.003046 ± 0.000812 | **24.0728** | 31.9078 | 0.7236 |
| 40,001 - 45,000 | Training | 0.008109 ± 0.001246 | 0.003063 ± 0.000770 | 24.0352 | 31.8159 | 0.7224 |
| 45,001 - 50,000 | Training | 0.007706 ± 0.001323 | 0.002888 ± 0.000865 | **24.0856** | 31.8202 | 0.7232 |
| 50,001 - 55,000 | Training | 0.007847 ± 0.000990 | 0.002941 ± 0.000560 | 24.0751 | 31.8269 | 0.7227 |
| 55,001 - 60,000 | Training | 0.007849 ± 0.001161 | 0.002908 ± 0.000648 | 24.0384 | 31.6971 | 0.7223 |
| 60,001 - 65,000 | Training | 0.007282 ± 0.000989 | 0.002713 ± 0.000611 | 24.0801 | 31.7727 | 0.7236 |
| 65,001 - 70,000 | Training | 0.007638 ± 0.001142 | 0.002905 ± 0.000656 | 24.0412 | 31.8181 | 0.7220 |
| 70,001 - 75,000 | Training | 0.007348 ± 0.001263 | 0.002707 ± 0.000671 | 24.0396 | 31.8231 | 0.7221 |
| 75,001 - 80,000 | Training | 0.007389 ± 0.001241 | 0.002855 ± 0.000802 | 24.0124 | 31.8190 | 0.7212 |
| 80,001 - 85,000 | Training | 0.007465 ± 0.001118 | 0.002743 ± 0.000583 | 24.0105 | 31.8167 | 0.7214 |
| 85,001 - 90,000 | Training | 0.007495 ± 0.001183 | 0.002834 ± 0.000667 | 24.0254 | 31.7937 | 0.7218 |
| 90,001 - 95,000 | Training | 0.007604 ± 0.001014 | 0.002919 ± 0.000633 | 24.0244 | 31.8146 | 0.7218 |
| 95,001 - 100,000 | Training | 0.007234 ± 0.000997 | 0.002758 ± 0.000719 | 24.0268 | 31.8374 | 0.7219 |
