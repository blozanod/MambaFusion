import math
import torch
import torch.nn as nn
from burstISP.utils.registry import ARCH_REGISTRY
from burstISP.archs.arch_util import DCNv4Block

@ARCH_REGISTRY.register()
class BurstAlign(nn.Module):
    """ BurstAlign module for aligning burst frames DCNv4
    
    Using PCD align architecture from EDVR, but with Modulated Deform Conv, only 2 pyramid levels
    Will test effectiveness of 2 levels vs 3 levels to save on compute time
    """
    def __init__(self, num_feat=64, num_frames=5, offset_groups=4):
        super(BurstAlign, self).__init__()
        self.num_frames = num_frames
        self.center_frame_idx = num_frames // 2
        self.offset_groups = offset_groups
        self.K = offset_groups * (3*3) # 9 kernel points

        self.padded_offset_channels = int(math.ceil((self.K * 3) / 8) * 8) # DCNv4 expects channels to be multiple of 8

        # Shallow Feature Extraction
        self.feat_extractor_lv1 = nn.Sequential(
            nn.Conv2d(4, num_feat, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )

        # Shallow Feature Extraction for level 2 (downsampled by 2)
        self.feat_extractor_lv2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=2), # stride=2 cuts H and W in half
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )

        # Offset predictor for DCN
        self.offset_predictor_lv1 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1)
        )

        self.offset_predictor_lv2 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1)
        )

        # For feature maps in lv2 to cascade to lv1 without interpolation
        self.up_lv2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * 4, kernel_size=3, padding=1, stride=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Offset Projections (uncomment lv2 if performing DCNv4 on both levels)
        self.offset_proj_lv1 = nn.Conv2d(num_feat, self.padded_offset_channels, kernel_size=3, padding=1)
        # self.offset_proj_lv2 = nn.Conv2d(num_feat, self.padded_offset_channels, kernel_size=3, padding=1)

        # Modulated Deformable Convolution
        self.dcn_lv1 = DCNv4Block(channels=num_feat, kernel_size=3, pad=1, stride=1, groups=offset_groups)
        
        # Initialize final offset weights to 0
        self._init_offset_weights()
    
    # Initialize final conv layer to 0 for exploding gradients
    def _init_offset_weights(self):
        nn.init.constant_(self.offset_proj_lv1.weight, 0)
        nn.init.constant_(self.offset_proj_lv1.bias, 0)
    
    def forward(self, x):
        B, N, C, H, W = x.size()  # [B, N, C, H, W]

        # Feature Extraction
        burst_reshaped = x.reshape(B * N, C, H, W)  # [B*N, C, H, W]

        features_lv1 = self.feat_extractor_lv1(burst_reshaped)  # [B*N, num_feat, H, W]

        features_lv2 = self.feat_extractor_lv2(features_lv1)  # [B*N, num_feat, H/2, W/2]
        
        # Reshape back to [B, N, num_feat, H, W]
        features_lv1 = features_lv1.view(B, N, -1, H, W)  # [B, N, num_feat, H, W]
        features_lv2 = features_lv2.view(B, N, -1, H // 2, W // 2)  # [B, N, num_feat, H/2, W/2]

        # Select center reference features
        ref_feat_lv1 = features_lv1[:, self.center_frame_idx, :, :, :]
        ref_feat_lv2 = features_lv2[:, self.center_frame_idx, :, :, :]

        aligned_feats = []

        # Loop through each frame and align to the reference frame
        for i in range(N):
            if i == self.center_frame_idx:
                aligned_feats.append(ref_feat_lv1)
                continue

            # Select and concatenate current frame features
            curr_feat_lv1 = features_lv1[:, i, :, :, :]
            curr_feat_lv2 = features_lv2[:, i, :, :, :]

            # LV2 Coarse Alignment
            concat_lv2 = torch.cat([ref_feat_lv2, curr_feat_lv2], dim=1)  # [B, 2*num_feat, H/2, W/2]
            motion_feats_lv2 = self.offset_predictor_lv2(concat_lv2)

            # PixelShuffle Upsampling
            motion_lv2_upsampled = self.up_lv2(motion_feats_lv2)  # [B, num_feat, H, W]

            # LV1 Fine Alignment
            concat_lv1 = torch.cat([ref_feat_lv1, curr_feat_lv1], dim=1)
            motion_feats_lv1 = self.offset_predictor_lv1(concat_lv1)

            # Cascade
            cascaded_motions = motion_feats_lv1 + motion_lv2_upsampled

            # Projection to DCNv4 format
            offset_masks = self.offset_proj_lv1(cascaded_motions)

            # DCNv4 Conv
            aligned_feat = self.dcn_lv1(curr_feat_lv1, offset_masks)
            aligned_feats.append(aligned_feat)

        # Stack aligned features
        aligned_feats = torch.stack(aligned_feats, dim=1)  # [B, N, num_feat, H, W]
        return aligned_feats
