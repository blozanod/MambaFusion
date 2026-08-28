import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia
import cv2
from burstISP.utils.registry import ARCH_REGISTRY
from burstISP.archs.mambairv2_arch import MambaIRv2
from burstISP.archs.dcn_align_arch import BurstAlign, PreAlign
from burstISP.archs.st_hat_fusion_arch import ST_HAT

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
        self.is_train = opt['is_train']
        self.is_global_skip = opt['global_skip']

        # Pre Alignment: project packed RGGB into feature space and PixelShuffle
        # x2 so that alignment, fusion and restoration all run on the Bayer grid.
        # When disabled the network keeps the original packed-RGGB pipeline, so a
        # single config key switches between the two domains for ablation.
        self.use_pre_align = opt.get('pre_align', False)

        if self.use_pre_align:
            self.pre_align = PreAlign(in_channels=4, num_feat=self.num_feat)
            align_in_ch = self.num_feat        # PreAlign already projected to num_feat
            restoration_scale = self.opt['scale'] // 2   # PreAlign supplied the other x2
            restoration_img_size = self.opt['img_size'] * 2
        else:
            align_in_ch = 4                    # raw packed RGGB
            restoration_scale = self.opt['scale']
            restoration_img_size = self.opt['img_size']

        # Alignment module
        self.alignment = BurstAlign(in_channels=align_in_ch, num_feat=self.num_feat, num_frames=self.num_frames, offset_groups=self.opt['offset_groups'])

        # Fusion module
        self.fusion = ST_HAT(
            num_frames=self.num_frames,
            window_size=self.opt['fusion_ws'],
            in_feat=self.num_feat,
            num_feat=self.opt['fusion_feat'],
            mlp_ratio=self.opt['fusion_mlp_ratio'],
            num_heads=self.opt['fusion_heads'],
            overlap_ratio=self.opt['fusion_overlap'],
            depth_stage1=self.opt['fusion_depth_s1'],
            depth_stage3=self.opt['fusion_depth_s3'],
            out_feat=self.opt['embed_dim'],
            st_window_size=self.opt.get('fusion_st_ws', self.opt['fusion_ws'])
        )

        # Restoration module
        self.restoration = MambaIRv2(
            img_size=restoration_img_size,
            in_chans= self.opt['embed_dim'],
            out_chans = 3, # For RGB image
            embed_dim=self.opt['embed_dim'],
            d_state=self.opt['d_state'],
            upscale=restoration_scale,
            depths=self.opt['depths'],
            num_heads=self.opt['num_heads'],
            window_size=self.opt['window_size'],
            inner_rank=self.opt['inner_rank'],
            num_tokens=self.opt['num_tokens'],
            convffn_kernel_size=self.opt['convffn_kernel_size'],
            mlp_ratio=self.opt['mlp_ratio'],
            upsampler=self.opt['upsampler'],
            resi_connection=self.opt['resi_connection'],
            use_checkpoint=False,
            control_net=self.is_global_skip,
            upsample_feat=self.opt.get('upsample_feat', 64)
        )

    def forward(self, x):
        # Dynamic padding
        B, N, C_in, h_ori, w_ori = x.shape
        mod = 16 
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        x = torch.cat([x, torch.flip(x, [3])], 3)
        x = torch.cat([x, torch.flip(x, [4])], 4)[:, :, :, :h_ori + h_pad, :w_ori + w_pad]

        # Align features from burst frames
        x_align = self.pre_align(x) if self.use_pre_align else x
        with torch.amp.autocast("cuda", enabled=False):
            aligned_burst, ref_feats = self.alignment(x_align.float())  # Shape: [B, N, C, H, W]

        # ST-HAT Fusion
        fused_input = self.fusion(aligned_burst)

        # Refinement and Upsampling with MambaIRv2. The residual is the aligned
        # reference features, injected through a zero-init skip_proj (PLAN.md L1).
        output = self.restoration(fused_input, ref_feats)  # Shape: [B, C_out, H_out, W_out]

        output = output[..., :h_ori * self.opt['scale'], :w_ori * self.opt['scale']]
        return output

# NOTE: this module cannot be executed directly (`python -m burstISP.archs.
# mambafusion_arch` or `python burstISP/archs/mambafusion_arch.py`). The archs
# package __init__ auto-imports every *_arch.py, so the class is registered on
# import and a second execution as __main__ trips the ARCH_REGISTRY duplicate
# assert. For a build / forward / backward / memory check on a real config, use
# `python analysis/shape_check.py <config.yml>`, which imports instead.
