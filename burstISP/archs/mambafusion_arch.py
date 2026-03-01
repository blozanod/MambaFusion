import torch
import torch.nn as nn
from burstISP.utils.registry import ARCH_REGISTRY
from burstISP.archs.mambairv2_arch import MambaIRv2

@ARCH_REGISTRY.register()
class MambaFusionNet(nn.Module):
    def __init__(self, opt):
        """Full MambaFusion architecture for burst image restoration, including alignment, fusion, and restoration modules.

        Args:
            opt (dict): Config for the full MambaFusionNet. Expected keys:
                network_mamba: dict with keys for mambairv2 config:
                    scale: 8 (supports 2, 4, 8, for RealBSR, set to 8 -> 4x upscaling due to RGGB packing)
                    in_chans: TODO based on TSAFusion output channels
                    img_size: 64
                    img_range: 1.
                    embed_dim: 174
                    d_state: 16
                    depths: [6,6,6,6,6,6]
                    num_heads: [6,6,6,6,6,6]
                    window_size: 16
                    inner_rank: 64
                    num_tokens: 128
                    convffn_kernel_size: 5
                    mlp_ratio: 2.
                    upsampler: 'pixelshuffle'
        """
        super(MambaFusionNet, self).__init__()
        self.opt_m = opt
        self.scale = opt.get('scale', 1)
        self.num_frames = opt.get('num_frames', 5)
        self.center_frame_idx = self.num_frames // 2

        # Restoration module
        self.restoration = MambaIRv2(num_feat=64, num_blocks=20, scale=self.scale)

    def forward(self, x):
        b, n, c, h, w = x.size()  # [B, N, C, H, W]
        center_frame = x[:, self.center_frame_idx]  # [B, C, H, W]

        aligned_feats = []
        for i in range(self.num_frames):
            if i == self.center_frame_idx:
                aligned_feats.append(center_frame)
            else:
                aligned_feat = self.alignment(x[:, i], center_frame)
                aligned_feats.append(aligned_feat)

        aligned_feats = torch.stack(aligned_feats, dim=1)  # [B, N, C, H, W]
        fused_feat = self.fusion(aligned_feats)  # [B, C, H, W]
        output = self.restoration(fused_feat)  # [B, C*scale^2, H*scale, W*scale]

        return output