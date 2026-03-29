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

        # Alignment module
        self.alignment = BurstAlign(num_feat=self.num_feat, num_frames=self.num_frames, offset_groups=self.offset_groups)

        self.fusion = TemporalFusion(num_frames=self.num_frames, num_feat=self.num_feat, window_size=self.opt['window_size'], num_heads=self.opt['num_heads'][0])

        # Restoration module
        self.restoration = MambaIRv2(
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
        aligned_burst = self.alignment(x)  # Shape: [B, N, C, H, W]

        """
        New Transformer-Based Cross-Frame Fusion Plan: Done!
        Center frame feature (frame 7): Query
        All frame features: key/values
        Use window paritioning for manageable compute (like in SwinIR)
        Output: Single fused feature map: [B, C, H, W]

        SR Head Modifications: Done!
        Drop shallow feature extraction (first conv), has already been done by other steps
        do this by changing kernel to 1x1

        New Alignment Loss:
        Calculate alignment loss by returning aligned_burst variable to training function
        if it is training. Then remove the center frame from the burst and use it as a reference instead.
        Then add the alignment loss to the total loss. This way alignment loss gradients reach alignment module.
        For this, it is wise to scale the loss over time (0.5x -> 0.10x -> 0.0x)

        New Alignment: Done!
        Swap DCNv2 for DCNv4, as well as interpolation for PixelShuffle for better upscaling in pyramid
        REMEMBER: REQUIRES COMPILING DCNV4 KERNELS AND CHANGING CONFIG.YML SO OFFSET GROUPS = 4, OFFSETS = 64

        Future Improvements:
        One whole joint architecture (not like 3-4 separate modules)
        Supervise training with PWC-Net for greater alignment accuracy
        """

        # Collapse burst dimension into batch dimension for restoration
        fused_input = self.fusion(aligned_burst)

        # Restore high-quality image from fused features
        output = self.restoration(fused_input)  # Shape: [B, C_out, H_out, W_out]

        if self.is_train:
            return output, aligned_burst
        else:
            return output