import torch
import torch.nn as nn
from burstISP.utils.registry import ARCH_REGISTRY

# from burstISP.dcn.deform_conv import ModulatedDeformConv
from torchvision.ops import DeformConv2d

import torch
import torch.nn as nn
from burstISP.archs.arch_util import ModulatedDeformableConv2d

@ARCH_REGISTRY.register()
class BurstAlign(nn.Module):
    """ BurstAlign module for aligning burst frames using Deformable Convolutional Networks (DCN).
    
    Using PCD align architecture from EDVR, but with Modulated Deform Conv, no pyramid levels
    Will test effectiveness of non-pyramid alignment vs pyramid alignment (2 or 3 levels) to save compute time
    """
    def __init__(self, num_feat=64, num_frames=5, offset_groups=8):
        super(BurstAlign, self).__init__()
        self.num_frames = num_frames
        self.center_frame_idx = num_frames // 2
        self.offset_groups = offset_groups

        # Shallow Feature Extraction
        self.feat_extractor = nn.Sequential(
            nn.Conv2d(4, num_feat, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )

        # Offset predictor for DCN
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, offset_groups * 27 , kernel_size=3, padding=1, stride=1)  # 3x3 kernel
        )

        # Modulated Deformable Convolution
        self.dcn = ModulatedDeformableConv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, stride=1, deformable_groups=offset_groups)

        # Initialize final offset weights to 0
        self._init_offset_weights()
    
    # Initialize final conv layer to 0 for exploding gradients
    def _init_offset_weights(self):
        final_conv = self.offset_predictor[-1]

        # Initialize the final convolution layer to predict zero offsets and masks
        nn.init.constant_(final_conv.weight, 0)
        nn.init.constant_(final_conv.bias, 0)
    
    def forward(self, x):
        B, N, C, H, W = x.size()  # [B, N, C, H, W]

        # Feature Extraction
        burst_reshaped = x.view(B * N, C, H, W)  # [B*N, C, H, W]
        features = self.feat_extractor(burst_reshaped)  # [B*N, num_feat, H, W]
        
        # Reshape back to [B, N, num_feat, H, W]
        features = features.view(B, N, -1, H, W)  # [B, N, num_feat, H, W]

        # Select center reference frames
        ref_frame = features[:, self.center_frame_idx, :, :, :]

        aligned_frames = []

        # Loop through each frame and align to the reference frame
        for i in range(N):
            if i == self.center_frame_idx:
                aligned_frames.append(ref_frame)
                continue

            # Select and concatenate current frame features
            curr_frame = features[:, i, :, :, :]
            concat_feat = torch.cat([ref_frame, curr_frame], dim=1)  # [B, 2*num_feat, H, W]

            # Predict offsets
            offsets = self.offset_predictor(concat_feat)  # [B, offset_groups*27, H, W]

            # Split offsets and masks
            offset_channels = self.offset_groups * 18

            offsets = offsets[:, :offset_channels, :, :]  # [B, offset_groups*18, H, W]
            masks = offsets[:, offset_channels:, :, :].sigmoid()  # [B, offset_groups*9, H, W]

            # Deformable Convolution
            aligned_feat = self.dcn(curr_frame, offsets, masks)  # [B, num_feat, H, W]
            aligned_frames.append(aligned_feat)

        # Stack aligned frames
        aligned_frames = torch.stack(aligned_frames, dim=1)  # [B, N, num_feat, H, W]
        return aligned_frames
