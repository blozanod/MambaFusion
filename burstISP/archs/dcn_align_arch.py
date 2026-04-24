import math
import torch
import torch.nn as nn
from burstISP.utils.registry import ARCH_REGISTRY
from burstISP.archs.arch_util import DCNv4Block


@ARCH_REGISTRY.register()
class BurstAlign(nn.Module):
    """ BurstAlign module for aligning burst frames with DCNv4.

    Alignment module using Pyramid, Cascading and Deformable convolution (PCD).
    Adapted to use DCNv4 and optimised for 2 pyramid levels.
    Adapted to apply upsampling AFTER DCN, not before. 

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
    num_feat (int): Channel number of middle features. Default: 64
    num_frames (int): Number of frames in the burst. Default: 5
    offset_groups (int): Number of groups for DCN offset prediction. Default: 4
    """

    def __init__(self, in_channels=4, num_feat=64, num_frames=5, offset_groups=4):
        super(BurstAlign, self).__init__()
        self.num_frames = num_frames
        self.center_frame_idx = num_frames // 2
        self.offset_groups = offset_groups
        self.K = offset_groups * (3 * 3)  # 36 kernel points
        self.padded_offset_channels = int(math.ceil((self.K * 3) / 8) * 8)  # 112

        # Pre-computed scale vector for offset mask tensor
        # the offsets are the only things that are scaled. If the mask was scaled, 
        # ie. by just multiplying by 2, the softmax temp in DCN would be ruined
        scale = torch.ones(1, self.padded_offset_channels, 1, 1)
        for k in range(self.K):
            base = k * 3
            scale[0, base,     0, 0] = 2.0  # delta x  – doubles with resolution
            scale[0, base + 1, 0, 0] = 2.0  # delta y  – doubles with resolution
            # scale[0, base + 2, 0, 0] remains 1.0  – mask, do not scale
        # Padding channels (indices K*3 .. padded_offset_channels-1) stay 1.0
        self.register_buffer('offset_scale', scale)

        # Feature Extraction
        self.feat_extractor_lv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_feat, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.feat_extractor_lv2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Level 2 Alignment
        self.offset_conv_lv2 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.offset_proj_lv2 = nn.Conv2d(num_feat, self.padded_offset_channels, kernel_size=3, padding=1)
        self.dcn_lv2 = DCNv4Block(channels=num_feat, kernel_size=3, pad=1, stride=1, groups=offset_groups)

        # Level 1 Alignment
        self.offset_conv_lv1_1 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.offset_conv_lv1_2 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.offset_proj_lv1 = nn.Conv2d(num_feat, self.padded_offset_channels, kernel_size=3, padding=1)
        self.dcn_lv1 = DCNv4Block(channels=num_feat, kernel_size=3, pad=1, stride=1, groups=offset_groups)

        # L1 Feature Fusion (aligned lv1 + upsampled coarse lv2)
        self.feat_fuse_lv1 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Cascading Refinement
        self.casc_offset_conv = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.casc_offset_proj = nn.Conv2d(num_feat, self.padded_offset_channels, kernel_size=3, padding=1)
        self.casc_dcn = DCNv4Block(channels=num_feat, kernel_size=3, pad=1, stride=1, groups=offset_groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self._init_offset_weights()

    def _init_offset_weights(self):
        """Zero-initialise all DCN projection layers.

        At t=0 all offsets and masks are zero DCNv4 starts as a regular
        convolution, giving stable early-training gradients.
        """
        for proj in [self.offset_proj_lv2, self.offset_proj_lv1, self.casc_offset_proj]:
            nn.init.constant_(proj.weight, 0)
            nn.init.constant_(proj.bias, 0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Burst of shape (B, N, C, H, W).

        Returns:
            Tensor: Aligned features of shape (B, N, num_feat, H, W).
        """
        B, N, C, H, W = x.size()

        # Feature Extraction (all frames as one batch)
        burst_reshaped = x.view(B * N, C, H, W)
        features_lv1_flat = self.feat_extractor_lv1(burst_reshaped)

        features_lv1 = features_lv1_flat.view(B, N, -1, H, W)
        features_lv2 = self.feat_extractor_lv2(features_lv1_flat).view(B, N, -1, H // 2, W // 2)

        ref_feat_lv1 = features_lv1[:, self.center_frame_idx] # (B, C, H, W)
        ref_feat_lv2 = features_lv2[:, self.center_frame_idx] # (B, C, H/2, W/2)

        aligned_feats = []

        for i in range(N):
            # The reference frame is intentionally NOT skipped:
            # Though in previous iterations it was, routing it through the
            # same pipeline as all others means it will look "identical" to them
            # thus it will be weighted as an equal with temporal fusion

            curr_feat_lv1 = features_lv1[:, i]   # (B, C, H, W)
            curr_feat_lv2 = features_lv2[:, i]   # (B, C, H/2, W/2)

            # Lv2 Alignment  (coarse alignment, H/2 x W/2 -> feats are compressed)
            concat_lv2 = torch.cat([curr_feat_lv2, ref_feat_lv2], dim=1)
            offset_feat_lv2 = self.offset_conv_lv2(concat_lv2) # (B, num_feat, H/2, W/2)
            offset_mask_lv2 = self.offset_proj_lv2(offset_feat_lv2) # (B, padded_C, H/2, W/2)

            aligned_lv2 = self.dcn_lv2(curr_feat_lv2, offset_mask_lv2) # (B, num_feat, H/2, W/2)

            # Coarse to Fine propagation
            # Usample the projected offset_mask tensor and apply offset_scaling to it
            # offset masks are geometric "pixel-displacements" and not the features
            # represented by offset_feat_lv. 
            # Thus it is multiplied by 2 for delta x/y chans and 1 for mask/padding chans
            up_offset_mask_lv2 = self.upsample(offset_mask_lv2) * self.offset_scale

            # Only upsample the features, thus no scaling is applied to these (feature-space)
            up_offset_feat_lv2 = self.upsample(offset_feat_lv2) # (B, num_feat, H, W), NO ×2

            # Upsample coarse aligned features (post DCN)
            up_aligned_lv2 = self.upsample(aligned_lv2) # (B, num_feat, H, W)

            # Lv1 Alignment  (fine alignment, H x W)
            concat_lv1 = torch.cat([curr_feat_lv1, ref_feat_lv1], dim=1)
            offset_feat_lv1_base = self.offset_conv_lv1_1(concat_lv1) # (B, num_feat, H, W)

            # Residual offset: condition on both the fine-level prior and the
            # upsampled coarse feature (which encodes where large-scale motion
            # was already estimated).
            offset_feat_lv1 = self.offset_conv_lv1_2(
                torch.cat([offset_feat_lv1_base, up_offset_feat_lv2], dim=1)
            )
            offset_mask_lv1 = self.offset_proj_lv1(offset_feat_lv1)      # (B, padded_C, H, W)

            # Now instead of adding the offset features, add the offset masks
            offset_mask_lv1 = offset_mask_lv1 + up_offset_mask_lv2

            aligned_lv1 = self.dcn_lv1(curr_feat_lv1, offset_mask_lv1) # (B, num_feat, H, W)

            # Fuse fine-aligned features with upsampled coarse aligned features
            aligned_lv1_fused = self.feat_fuse_lv1(
                torch.cat([aligned_lv1, up_aligned_lv2], dim=1) # (B, num_feat, H, W)
            )

            # Cascading Refinement
            concat_casc     = torch.cat([aligned_lv1_fused, ref_feat_lv1], dim=1)
            casc_offset_feat = self.casc_offset_conv(concat_casc)
            casc_offset_mask = self.casc_offset_proj(casc_offset_feat)

            final_aligned = self.casc_dcn(aligned_lv1_fused, casc_offset_mask)

            final_aligned = self.lrelu(final_aligned) # Mirror EVDR final activation

            aligned_feats.append(final_aligned)

        return torch.stack(aligned_feats, dim=1)   # (B, N, num_feat, H, W)