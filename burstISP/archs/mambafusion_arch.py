import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
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

        # Long skip connection
        self.global_skip = GlobalSkipConnection(scale=self.opt['scale'])
        self.alpha_residual = nn.Parameter(torch.tensor(0.1))

        # Alignment module
        self.alignment = BurstAlign(num_feat=self.num_feat, num_frames=self.num_frames, offset_groups=self.offset_groups)

        # Fusion module
        self.fusion = TemporalFusion(num_frames=self.num_frames, num_feat=self.num_feat, window_size=self.opt['fusion_window_size'], num_heads=self.fusion_heads)

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
        center_idx = self.num_frames // 2
        center_raw = x[:, center_idx, :, :, :].float()
        
        # Align features from burst frames
        with torch.amp.autocast("cuda", enabled=False):
            aligned_burst = self.alignment(x.float())  # Shape: [B, N, C, H, W]

        aligned_burst = aligned_burst.to(x.dtype)

        # Collapse burst dimension into batch dimension for restoration
        fused_input = self.fusion(aligned_burst)

        # Restore high-quality image from fused features
        deep_residual = self.restoration(fused_input)  # Shape: [B, C_out, H_out, W_out]

        # Add long skip connection
        output = base_img + (deep_residual*self.alpha_residual)

        return output
    
class GlobalSkipConnection(nn.Module):
    """
    Global skip connection baseline definition and application

    Performs non-learnable Malvar-He-Cutler demosaicing and bicubic upsampling to generate strong
    baseline. Model only has to learn residual from here.
    """
    def __init__(self, scale):
        super(GlobalSkipConnection, self).__init__()
        self.scale = scale
        self.rem_scale = self.scale // 2

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        # Detach to CPU 
        x_cpu = x.detach().float().cpu().numpy()
        
        # Scale to 16 bit
        x_uint16 = np.clip(x_cpu * 65535.0, 0, 65535).astype(np.uint16)
        
        out_batch =[]
        for i in range(B):
            # Unpack the 4-channel image into a 1-channel 2H x 2W Bayer array
            bayer = np.zeros((H * 2, W * 2), dtype=np.uint16)
            bayer[0::2, 0::2] = x_uint16[i, 0, :, :] # R
            bayer[0::2, 1::2] = x_uint16[i, 1, :, :] # G
            bayer[1::2, 0::2] = x_uint16[i, 2, :, :] # G
            bayer[1::2, 1::2] = x_uint16[i, 3, :, :] # B
            
            # Malvar-He-Cutler Demosaicing
            rgb_uint16 = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB_EA)
            out_batch.append(rgb_uint16)
            
        # Normalize to fp32
        out_np = np.stack(out_batch, axis=0) #[B, 2H, 2W, 3]
        out_np = out_np.astype(np.float32) / 65535.0
        
        # Cast back to tensor
        out_tensor = torch.from_numpy(out_np).permute(0, 3, 1, 2).to(device)
        
        # Upample with Bilinear Interpolation
        out = F.interpolate(out_tensor, scale_factor=self.rem_scale, mode='bicubic', align_corners=False)
        
        # Return bf16
        return out.to(x.dtype)
