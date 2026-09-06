import math
import torch
from collections import OrderedDict
from torch.nn import functional as F
from contextlib import nullcontext

from burstISP.utils.registry import MODEL_REGISTRY
from burstISP.loss import build_loss
from burstISP.models.sr_model import SRModel
from burstISP.utils import get_root_logger
from burstISP.utils.img_util import differentiable_benchmark_isp


@MODEL_REGISTRY.register()
class MambaFusionModel(SRModel):
    """MambaFusion model for image restoration."""
    def __init__(self, opt):
        # Set before super().__init__, which may call init_training_settings.
        # A model built for inference never runs optimize_parameters, but this
        # keeps the attribute defined either way.
        self.cri_flow = None
        super(MambaFusionModel, self).__init__(opt)

    def init_training_settings(self):
        """SRModel's setup plus the optional alignment-flow criterion.

        The flow term supervises BurstAlign's per-level flow estimates against
        the generator's ground-truth `flow_vectors` (PLAN L6 step 8). It is a
        curriculum, not a constraint: lambda decays to zero so that late in
        training the network is free to pick a sampling pattern that beats
        perfect geometric registration.
        """
        super(MambaFusionModel, self).init_training_settings()
        train_opt = self.opt['train']

        if train_opt.get('flow_opt'):
            self.cri_flow = build_loss(train_opt['flow_opt']).to(self.device)
            # Piecewise-linear lambda(t). Duplicate a milestone to get a step.
            sched = train_opt.get('flow_lambda', {})
            self.flow_milestones = list(sched.get('milestones', [0]))
            self.flow_values = list(sched.get('values', [1.0]))
            if len(self.flow_milestones) != len(self.flow_values):
                raise ValueError('train.flow_lambda.milestones and .values must be the same length, '
                                 f'got {len(self.flow_milestones)} and {len(self.flow_values)}')
            # Equal weight per level; keys must exist in the arch's aux dict.
            self.flow_levels = list(train_opt.get('flow_levels', ['lv1', 'lv2', 'lv3']))
        else:
            self.cri_flow = None

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device) # [B, N, C, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device) # [B, C, H, W]
        # Ground-truth geometric flow, [B, N, 2, 2h, 2w] in LR-RGB pixel units.
        # Only SyntheticBurstDataset in train mode ships it; the official val
        # set does not, so this is None on every validation pass.
        self.flow_gt = data['flow_vectors'].to(self.device) if 'flow_vectors' in data else None

    def flow_lambda(self, current_iter):
        """lambda(t), linearly interpolated between the configured milestones
        and held flat outside them."""
        ms, vs = self.flow_milestones, self.flow_values
        if current_iter <= ms[0]:
            return float(vs[0])
        if current_iter >= ms[-1]:
            return float(vs[-1])
        for i in range(1, len(ms)):
            if current_iter <= ms[i]:
                span = max(ms[i] - ms[i - 1], 1)
                t = (current_iter - ms[i - 1]) / span
                return float(vs[i - 1] + t * (vs[i] - vs[i - 1]))
        return float(vs[-1])

    def flow_loss(self, flows):
        """Charbonnier between each level's predicted flow and the generator's
        flow, average-pooled to that level.

        The sign needs no negation here. `synthetic_burst_dataset.py:196` says
        the content the reference sees at p sits at `p - flow(p)` in frame i,
        and `BurstAlign.warp` samples at exactly `p - flow`, so the predicted
        flow and `flow_vectors` share a convention. The negation into DCN
        offset units lives in `BurstAlign.scatter_flow`, downstream of this.

        The reference frame is included on purpose: its ground-truth flow is
        identically zero (`flow_vectors = sample_pos_inv_all - ...[:1]`), so it
        contributes a free "do not move" constraint.

        The per-level conversion is derived from the tensors' own shapes, so it
        is correct in both the Bayer and the packed domain: pooling GT from
        96x96 to a level of side h scales magnitudes by h/96, which reproduces
        the x1 / x0.5 / x0.25 of `oracle_warp`'s pattern.
        """
        gt = self.flow_gt
        B, N = gt.shape[0], gt.shape[1]
        gt = gt.reshape(B * N, 2, gt.shape[-2], gt.shape[-1]).float()
        gt_h = gt.shape[-2]

        missing = [lv for lv in self.flow_levels if lv not in flows]
        if missing:
            raise KeyError(f'train.flow_levels names {missing}, which BurstAlign does not '
                           f'return; available levels are {sorted(flows)}')

        l_flow = 0
        for lvl in self.flow_levels:
            pred = flows[lvl].reshape(B * N, 2, *flows[lvl].shape[-2:]).float()
            h, w = pred.shape[-2:]
            # adaptive_avg_pool2d == avg_pool2d for the integer ratios here, and
            # degrades gracefully if a crop size is not an exact power-of-two
            # multiple of the level size.
            target = F.adaptive_avg_pool2d(gt, (h, w)) * (h / gt_h)
            l_flow = l_flow + self.cri_flow(pred, target)

        return l_flow / max(len(self.flow_levels), 1)

    @staticmethod
    def compand(x, mu=24.0):
        # Signed mu law
        return torch.sign(x) * torch.log1p(mu * x.abs()) / math.log1p(mu)

    # Modified from sr_model.py to include bf16 and alignment loss
    def optimize_parameters(self, current_iter):
        accumulation_steps = self.opt['train'].get('accumulation_steps', 1)

        if (current_iter - 1) % accumulation_steps == 0:
            self.optimizer_g.zero_grad()

        is_sync_step = (current_iter % accumulation_steps == 0)
        
        # Use DDP's no_sync() if we are accumulating, otherwise do nothing
        sync_context = self.net_g.no_sync if (not is_sync_step and self.opt['dist']) else nullcontext
        
        # Forward Pass
        with sync_context():
            # lambda is read up front: when the curriculum has decayed to zero
            # there is nothing to gain from carrying the aux flows. The flow
            # head stays on the gradient path either way, through the DCN
            # offsets, so find_unused_parameters can stay false.
            lam = self.flow_lambda(current_iter) if self.cri_flow else 0.0
            want_flow = self.cri_flow is not None and lam > 0 and self.flow_gt is not None

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if want_flow:
                    self.output, aux = self.net_g(self.lq, return_aux=True)
                else:
                    self.output, aux = self.net_g(self.lq), None

            self.output = self.output.float()
            l_total = 0
            loss_dict = OrderedDict()

            # Projection into SRGB via Signed Mu-Law. Config-gated: compand
            # defaults to true (RealBSR behavior); PLAN.md L3 sets it to false
            # to train on plain linear RGB.
            if self.opt['train'].get('compand', True):
                cp, cg = self.compand(self.output), self.compand(self.gt)
            else:
                cp, cg = self.output, self.gt

            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(cp, cg)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

            # Edge Loss
            if self.cri_edge:
                l_edge = self.cri_edge(cp, cg)
                l_total += l_edge
                loss_dict['l_edge'] = l_edge

            # Alignment flow loss (curriculum; lambda decays to zero)
            if want_flow:
                l_flow = lam * self.flow_loss(aux['flows'])
                l_total += l_flow
                loss_dict['l_flow'] = l_flow
                loss_dict['flow_lambda'] = torch.tensor(lam, device=self.device)

            # Backpropagation
            l_total = l_total / accumulation_steps
            l_total.backward()

            if is_sync_step:
                clip_norm = self.opt['datasets']['train'].get('grad_clip_norm',1.0)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), clip_norm)
                self.optimizer_g.step()

                if self.ema_decay > 0:
                    self.model_ema(decay=self.ema_decay)

                self.log_dict = self.reduce_loss_dict(loss_dict)
    
    def test(self):
        if hasattr(self, 'net_g_ema'):
            model = self.net_g_ema
        else:
            model = self.get_bare_model(self.net_g)

        model.eval()
        with torch.no_grad():
            self.output = model(self.lq)
        model.train()

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        # Log ReZero alpha
        bare_model = self.get_bare_model(self.net_g)
        if hasattr(bare_model, 'alpha_residual'):
            alpha_val = bare_model.alpha_residual.item()
            log_str += f'\t # rezero_alpha: {alpha_val:.10f}\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
            if hasattr(bare_model, 'alpha_residual'):
                tb_logger.add_scalar('train/rezero_alpha', alpha_val, current_iter)
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq[:, self.lq.shape[1]//2].detach().cpu()  # show center frame
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
