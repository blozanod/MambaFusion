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

            if train_opt.get('fusion_opt'):
                self.cri_fusion = build_loss(train_opt['fusion_opt']).to(self.device)
            else:
                self.cri_fusion = None
            
            if train_opt.get('cobi_opt'):
                self.cri_cobi = build_loss(train_opt['cobi_opt']).to(self.device)
            else:
                self.cri_cobi = None

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device) # [B, N, C, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device) # [B, C, H, W]

    # Modified from sr_model.py to include bf16 and alignment loss
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        # Forward Pass
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            self.output, aligned_burst, fusion_output = self.net_g(self.lq)

        self.output = self.output.float()
        aligned_burst = aligned_burst.float()
        fusion_output = fusion_output.float()

        ref_index = aligned_burst.shape[1] // 2
        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # CoBi loss
        if self.cri_cobi:
            l_cobi = self.cri_cobi(self.output, self.gt)
            l_total += l_cobi
            loss_dict['l_cobi'] = l_cobi

        # Sobel Loss
        if self.cri_sobel:
            l_sobel = self.cri_sobel(self.output.float(), self.gt.float())
            l_total += l_sobel
            loss_dict['l_sobel'] = l_sobel

        # alignment loss
        if self.cri_align:
            # Extract and detach center frame
            ref_feat = aligned_burst[:, ref_index, :, :, :].detach()
            
            l_align = 0
            num_frames = aligned_burst.shape[1]
            
            # Compute Charbonnier between each aligned neighbor and the reference
            for i in range(num_frames):
                if i != ref_index:
                    l_align += self.cri_align(aligned_burst[:, i, :, :, :], ref_feat)
            
            # Average the loss across the 4 neighboring frames
            l_align = l_align / (num_frames - 1)
            
            # Add alignment loss to total loss
            l_total += l_align
            loss_dict['l_align'] = l_align

        # fusion loss
        if self.cri_fusion:
            l_fusion = self.cri_fusion(fusion_output, self.gt)
            l_total += l_fusion
            loss_dict['l_fusion'] = l_fusion

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
