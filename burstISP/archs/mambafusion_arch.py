import torch
import torch.nn as nn
from burstISP.utils.registry import ARCH_REGISTRY
from burstISP.archs.mambairv2_arch import MambaIRv2
from burstISP.archs.dcn_align_arch import BurstAlign

@ARCH_REGISTRY.register()
class MambaFusionNet(nn.Module):
    """Full MambaFusion architecture for burst image restoration, including alignment, fusion, and restoration modules.

        Args:
            opt (dict): Config for the full MambaFusionNet. Expected keys:
                num_frames: Number of frames in the burst (e.g. 5)
                num_feat: Number of features for alignment (e.g. 64)
                out_chans: Number of output channels (e.g. 3, different from in chans due to burst)
                offset_groups: Number of groups for DCN offset prediction (e.g. 8)
                scale: upscaling factor (e.g. 8 -> 4 due to packed RGGB)
                depths: Model depth for Mamba block (e.g. [6, 6, 6, 6])
                num_heads: Number of stages per Mamba block (e.g. [4, 4, 4, 4])
                mlp_ratio: MLP ratio for Mamba block (e.g. 2)
                upsampler: upsampling method for Mamba block (e.g. 'pixelshuffledirect')
        """
    def __init__(self, **opt):
        super(MambaFusionNet, self).__init__()
        self.opt = opt
        self.num_frames = opt['num_frames']
        self.num_feat = opt['num_feat']
        self.offset_groups = opt['offset_groups']

        # Alignment module
        self.alignment = BurstAlign(num_feat=self.num_feat, num_frames=self.num_frames, offset_groups=self.offset_groups)

        # Restoration module
        self.restoration = MambaIRv2(
            in_chans= self.num_frames * self.num_feat,
            out_chans = 3, # For RGB image
            upscale=self.opt["scale"],
            depths=self.opt['depths'],
            num_heads=self.opt['num_heads'],
            mlp_ratio=self.opt['mlp_ratio'],
            upsampler=self.opt['upsampler'],
            use_checkpoint=False)

    def forward(self, x):
        # Align features from burst frames
        aligned_burst = self.alignment(x)  # Shape: [B, N, C, H, W]

        # Collapse burst dimension into batch dimension for restoration
        B, N, C, H, W = aligned_burst.shape
        fused_input = aligned_burst.view(B, N * C, H, W)  # Shape: [B, N * C, H, W]

        # Restore high-quality image from fused features
        output = self.restoration(fused_input)  # Shape: [B, C_out, H_out, W_out]

        return output