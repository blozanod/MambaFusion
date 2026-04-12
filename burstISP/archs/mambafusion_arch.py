import torch
import torch.nn as nn
from burstISP.utils.registry import ARCH_REGISTRY
from burstISP.archs.mambairv2_arch import MambaIRv2
from burstISP.archs.dcn_align_arch import BurstAlign
from burstISP.archs.temporal_fusion_arch import TemporalFusion

@ARCH_REGISTRY.register()
class MambaFusionNet(nn.Module):
    """Full MambaFusion architecture for burst image restoration, including alignment, fusion, and restoration modules.

        Args:
            opt (dict): Config for the full MambaFusionNet. Expected keys:
                num_frames: Number of frames in the burst (e.g. 5)
                num_feat: Number of features for alignment (e.g. 64)
                out_chans: Number of output channels (e.g. 3, different from in chans due to burst)
                offset_groups: Number of groups for DCN offset prediction (e.g. 8)
                embed_dim:
                d_state:
                scale: upscaling factor (e.g. 8 -> 4 due to packed RGGB)
                depths: Model depth for Mamba block (e.g. [6, 6, 6, 6])
                num_heads: Number of stages per Mamba block (e.g. [4, 4, 4, 4])
                window_size: Window size for both Mamba and Fusion blocks
                inner_rank:
                num_tokens:
                convffn_kernel_size:
                mlp_ratio: MLP ratio for Mamba block (e.g. 2)
                upsampler: upsampling method for Mamba block (e.g. 'pixelshuffledirect')
                resi_connection: residual connection for Mamba block (e.g. '1conv')
                is_train: Used for alignment loss
    """
    def __init__(self, **opt):
        super(MambaFusionNet, self).__init__()
        self.opt = opt
        self.num_frames = opt['num_frames']
        self.num_feat = opt['num_feat']
        self.offset_groups = opt['offset_groups']
        self.is_train = opt['is_train']
        self.fusion_heads = opt['fusion_heads']

        # Alignment module
        self.alignment = BurstAlign(num_feat=self.num_feat, num_frames=self.num_frames, offset_groups=self.offset_groups)

        # Fusion module & aux upsampler
        self.fusion = TemporalFusion(num_frames=self.num_frames, num_feat=self.num_feat, window_size=self.opt['window_size'], num_heads=self.fusion_heads)
        self.aux_upsampler = AuxUpsampler(in_channels=self.num_feat, scale_factor=self.opt['scale'], out_channels=3)

        # Restoration module
        self.restoration = MambaIRv2(
            img_size= self.opt['img_size'],
            in_chans= self.num_feat,
            out_chans = 3, # For RGB image
            embed_dim=self.opt['embed_dim'],
            d_state=self.opt['d_state'],
            upscale=self.opt["scale"],
            depths=self.opt['depths'],
            num_heads=self.opt['num_heads'],
            window_size=self.opt['window_size'],
            inner_rank=self.opt['inner_rank'],
            num_tokens=self.opt['num_tokens'],
            convffn_kernel_size=self.opt['convffn_kernel_size'],
            mlp_ratio=self.opt['mlp_ratio'],
            upsampler=self.opt['upsampler'],
            resi_connection=self.opt['resi_connection'],
            use_checkpoint=False)

    def forward(self, x):
        # Align features from burst frames
        with torch.amp.autocast("cuda", enabled=False):
            aligned_burst = self.alignment(x.float())  # Shape: [B, N, C, H, W]

        aligned_burst = aligned_burst.to(x.dtype)

        # Collapse burst dimension into batch dimension for restoration
        fused_input = self.fusion(aligned_burst)

        # Restore high-quality image from fused features
        output = self.restoration(fused_input)  # Shape: [B, C_out, H_out, W_out]

        if self.is_train:
            fusion_output = self.aux_upsampler(fused_input)
            return output, aligned_burst, fusion_output
        else:
            return output

@ARCH_REGISTRY.register()
class AuxUpsampler(nn.Module):
    """Auxiliary upsample uses PixelShuffle to upsample directly from fused features
    
    Used to calculate fusion loss to supervise fusion module
    """
    def __init__(self, in_channels, scale_factor=8, out_channels=3):
        super().__init__()
        mid_channels = out_channels * (scale_factor ** 2)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        return self.upsample(x)