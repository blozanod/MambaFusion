import math
import torch
import torch.nn as nn
from burstISP.utils.registry import ARCH_REGISTRY
from burstISP.archs.arch_util import DCNv4Block

@ARCH_REGISTRY.register()
class BurstAlign(nn.Module):
    """ BurstAlign module for aligning burst frames DCNv4

    Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
    
    Using PCD align architecture from EDVR, but with Modulated Deform Conv, only 2 pyramid levels
    Will test effectiveness of 2 levels vs 3 levels to save on compute time

    Args:
    num_feat (int): Channel number of middle features. Default: 64
    num_frames (int): Number of frames in the burst. Default: 5
    offset_groups (int): Number of groups for DCN offset prediction. Default: 4
    """
    def __init__(self, num_feat=64, num_frames=5, offset_groups=4):
        super(BurstAlign, self).__init__()
        self.num_frames = num_frames
        self.center_frame_idx = num_frames // 2
        self.offset_groups = offset_groups
        self.K = offset_groups * (3*3) # 9 kernel points
        self.padded_offset_channels = int(math.ceil((self.K * 3) / 8) * 8) # DCNv4 expects channel to be multiple of 8

        # Feature Extraction
        self.feat_extractor_lv1 = nn.Sequential(
            nn.Conv2d(4, num_feat, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.feat_extractor_lv2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Level 2 Alignment (offset -> DCN)
        self.offset_conv_lv2 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.offset_proj_lv2 = nn.Conv2d(num_feat, self.padded_offset_channels, kernel_size=3, padding=1)
        self.dcn_lv2 = DCNv4Block(channels=num_feat, kernel_size=3, pad=1, stride=1, groups=offset_groups)

        # Level 1 Alignment (Offset lv1 -> Offset lv1 + lv2 -> DCN)
        self.offset_conv_lv1_1 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.offset_conv_lv1_2 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, padding=1), # Takes concat of L1 and upsampled L2 offsets
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.offset_proj_lv1 = nn.Conv2d(num_feat, self.padded_offset_channels, kernel_size=3, padding=1)
        self.dcn_lv1 = DCNv4Block(channels=num_feat, kernel_size=3, pad=1, stride=1, groups=offset_groups)
        
        # L1 Feature Fusion (fusing L1 warped features with upsampled L2 warped features)
        self.feat_fuse_lv1 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Cascading Refinement (Level 1)
        self.casc_offset_conv = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.casc_offset_proj = nn.Conv2d(num_feat, self.padded_offset_channels, kernel_size=3, padding=1)
        self.casc_dcn = DCNv4Block(channels=num_feat, kernel_size=3, pad=1, stride=1, groups=offset_groups)

        # Bilinear Upsampler
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self._init_offset_weights()
    
    def _init_offset_weights(self):
        # Initialize final projections to 0 for stability
        for proj in[self.offset_proj_lv2, self.offset_proj_lv1, self.casc_offset_proj]:
            nn.init.constant_(proj.weight, 0)
            nn.init.constant_(proj.bias, 0)
    
    def forward(self, x):
        B, N, C, H, W = x.size()

        burst_reshaped = x.view(B * N, C, H, W)
        features_lv1 = self.feat_extractor_lv1(burst_reshaped).view(B, N, -1, H, W)
        features_lv2 = self.feat_extractor_lv2(burst_reshaped.view(B * N, -1, H, W)).view(B, N, -1, H // 2, W // 2)

        ref_feat_lv1 = features_lv1[:, self.center_frame_idx, :, :, :]
        ref_feat_lv2 = features_lv2[:, self.center_frame_idx, :, :, :]

        aligned_feats =[]

        for i in range(N):
            if i == self.center_frame_idx:
                aligned_feats.append(ref_feat_lv1)
                continue

            curr_feat_lv1 = features_lv1[:, i, :, :, :]
            curr_feat_lv2 = features_lv2[:, i, :, :, :]

            # Lv2 Alignment
            concat_lv2 = torch.cat([curr_feat_lv2, ref_feat_lv2], dim=1)
            offset_feat_lv2 = self.offset_conv_lv2(concat_lv2)
            offset_mask_lv2 = self.offset_proj_lv2(offset_feat_lv2)
            
            # Lv2 DCN
            aligned_lv2 = self.dcn_lv2(curr_feat_lv2, offset_mask_lv2)

            # Lv1 Alignment
            # Upsample Lv2 offsets and aligned feats
            up_offset_feat_lv2 = self.upsample(offset_feat_lv2) 
            up_aligned_lv2 = self.upsample(aligned_lv2)

            # Lv1 Offsets (no Lv2)
            concat_lv1 = torch.cat([curr_feat_lv1, ref_feat_lv1], dim=1)
            offset_feat_lv1_base = self.offset_conv_lv1_1(concat_lv1)
            
            # Lv1 Offsets + Lv2 Offsets (concat)
            offset_feat_lv1 = self.offset_conv_lv1_2(torch.cat([offset_feat_lv1_base, up_offset_feat_lv2], dim=1))
            offset_mask_lv1 = self.offset_proj_lv1(offset_feat_lv1)
            
            # Lv1 DCN
            aligned_lv1 = self.dcn_lv1(curr_feat_lv1, offset_mask_lv1)
            
            # Lv1 Aligned Feats + Lv2 Aligned Feats (upsampled, concat)
            aligned_lv1_fused = self.feat_fuse_lv1(torch.cat([aligned_lv1, up_aligned_lv2], dim=1))

            # Cascading refinement
            # Compare the newly aligned feature to the reference again
            concat_casc = torch.cat([aligned_lv1_fused, ref_feat_lv1], dim=1)
            casc_offset_feat = self.casc_offset_conv(concat_casc)
            casc_offset_mask = self.casc_offset_proj(casc_offset_feat)
            
            # Final DCN (refinement)
            final_aligned = self.casc_dcn(aligned_lv1_fused, casc_offset_mask)
            
            aligned_feats.append(final_aligned)

        return torch.stack(aligned_feats, dim=1)