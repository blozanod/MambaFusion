import torch
from collections import OrderedDict
from torch.nn import functional as F

from burstISP.utils.registry import MODEL_REGISTRY
from burstISP.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class MambaFusionModel(SRModel):
    """MambaFusion model for image restoration."""

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device) # [B, N, C, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device) # [B, C, H, W]

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq[:, self.lq.shape[1]//2].detach().cpu()  # show center frame
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict