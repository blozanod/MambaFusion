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
        if self.is_train:
            train_opt = self.opt['train']
            
            if train_opt.get('alignment_opt'):
                self.cri_align = build_loss(train_opt['alignment_opt']).to(self.device)
            else:
                self.cri_align = None

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device) # [B, N, C, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device) # [B, C, H, W]

    # Modified from sr_model.py to include bf16 and alignment loss
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        # Forward Pass
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            self.output, aligned_burst = self.net_g(self.lq)

        ref_index = aligned_burst.shape[1] // 2
        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            l_rec = self.cri_pix(self.output, self.gt)
            l_total += l_rec
            loss_dict['l_rec'] = l_rec

        # alignment loss
        if hasattr(self, 'cri_align') and self.cri_align:
            # Extract and detach center frame
            ref_feat = aligned_burst[:, ref_index, :, :, :].detach()
            
            l_align = 0
            num_frames = aligned_burst.shape[1]
            
            # Compute Charbonnier between each aligned neighbor and the reference
            for i in range(num_frames):
                if i != 2:
                    l_align += self.cri_align(aligned_burst[:, i, :, :, :], ref_feat)
            
            # Average the loss across the 4 neighboring frames
            l_align = l_align / (num_frames - 1)

            # Anneal the Alignment Loss over time (0.5x -> 0.10x -> 0.0x)
            if current_iter < 50000:
                align_weight = 0.5
            elif current_iter < 100000:
                align_weight = 0.1
            else:
                align_weight = 0.0

            l_align = l_align * align_weight
            
            if align_weight > 0:
                l_total += l_align
                loss_dict['l_align'] = l_align

        # Backpropagation
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq[:, self.lq.shape[1]//2].detach().cpu()  # show center frame
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict