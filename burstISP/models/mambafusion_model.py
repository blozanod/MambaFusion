import torch
from collections import OrderedDict
from torch.nn import functional as F

from burstISP.utils.registry import MODEL_REGISTRY
from burstISP.models.sr_model import SRModel
from burstISP.loss import build_loss


@MODEL_REGISTRY.register()
class MambaFusionModel(SRModel):
    """MambaFusion model for image restoration."""
    def __init__(self, opt):
        super(MambaFusionModel, self).__init__(opt)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device) # [B, N, C, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device) # [B, C, H, W]

    # Modified from sr_model.py to include bf16 and alignment loss
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        # Forward Pass
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            #self.output, aligned_burst, fusion_output = self.net_g(self.lq)
            self.output = self.net_g(self.lq)

        self.output = self.output.float()
        l_total = 0
        loss_dict = OrderedDict()

        # Applies part of ISP from visualize_results.py
        # Differentiates edges from noise
        epsilon = 1e-6
        out_float = self.output.float()
        gt_float = self.gt.float()
        FIXED_EXPOSURE = 4.0 

        out_scaled = torch.abs(out_float * FIXED_EXPOSURE) + epsilon
        gt_scaled = torch.abs(gt_float * FIXED_EXPOSURE) + epsilon

        # Gamma Compression 
        out_gamma = torch.clamp(out_scaled ** (1.0 / 2.2), 0.0, 1.0)
        gt_gamma = torch.clamp(gt_scaled ** (1.0 / 2.2), 0.0, 1.0)

        # Smoothstep
        out_final = 3 * out_gamma ** 2 - 2 * out_gamma ** 3
        gt_final = 3 * gt_gamma ** 2 - 2 * gt_gamma ** 3

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(out_final, gt_final)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # Edge Loss
        if self.cri_edge:
            l_edge = self.cri_edge(out_final, gt_final)
            l_total += l_edge
            loss_dict['l_edge'] = l_edge

        # Backpropagation
        l_total.backward()

        clip_norm = self.opt['datasets']['train'].get('grad_clip_norm',1.0)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), clip_norm)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            model = self.net_g_ema
        else:
            model = self.get_bare_model(self.net_g)

        model.eval()
        with torch.no_grad():
            self.output = model(self.lq)
        model.train()
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq[:, self.lq.shape[1]//2].detach().cpu()  # show center frame
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
