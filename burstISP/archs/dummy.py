import torch
import math
from burstISP.archs.arch_util import DCNv4Block

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(1, 3, 16, 16)
sample_x = 1.6
sample_y = 2.3
B,C,H,W = x.shape
out = torch.zeros_like(x)

# 1. Dumb Loop
for y in range(H):
    for xx in range(W):
        x_src = xx - sample_x
        y_src = y - sample_y

        # Interpolate
        x0 = math.floor(x_src)
        x1 = x0 + 1
        y0 = math.floor(y_src)
        y1 = y0 + 1
        wa = x_src - x0
        wb = y_src - y0

        w00 = (1-wa) * (1-wb)
        w01 = (1-wa) * wb
        w10 = wa * (1-wb)
        w11 = wa * wb

        v00 = (0 <= x0 < W) and (0 <= y0 < H)
        v01 = (0 <= x0 < W) and (0 <= y1 < H)
        v10 = (0 <= x1 < W) and (0 <= y0 < H)
        v11 = (0 <= x1 < W) and (0 <= y1 < H)

        x0_idx = max(0, min(W - 1, int(x0)))
        x1_idx = max(0, min(W - 1, int(x1)))
        y0_idx = max(0, min(H - 1, int(y0)))
        y1_idx = max(0, min(H - 1, int(y1)))

        p00 = x[:, :, y0_idx, x0_idx] if v00 else 0.0
        p01 = x[:, :, y1_idx, x0_idx] if v01 else 0.0
        p10 = x[:, :, y0_idx, x1_idx] if v10 else 0.0
        p11 = x[:, :, y1_idx, x1_idx] if v11 else 0.0

        out[:, :, y, xx] = p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11

import torch.nn.functional as F
import numpy as np

def warp_gs(x, sx, sy, align_corners, formula):
    B, C, H, W = x.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing='ij')
    sample_x = xs - sx
    sample_y = ys - sy

    if formula == 'A':
        gx = 2 * sample_x / (W - 1) - 1
        gy = 2 * sample_y / (H - 1) - 1
    else:
        gx = (2 * sample_x + 1) / W - 1
        gy = (2 * sample_y + 1) / H - 1

    grid = torch.stack((gx, gy), -1).unsqueeze(0).expand(B, -1, -1, -1)
    return F.grid_sample(x, grid, mode='bilinear',
                         padding_mode='zeros', align_corners=align_corners)

#for ac in (True, False):
#   for f in ('A', 'B'):
#        d = (warp_gs(x, 1.6, 2.3, ac, f) - out)[..., 3:-3, 3:-3].abs().max().item()
#        print(f'align_corners={str(ac):<5} formula={f}  ->  {d:.6f}')

def shift_fast(x, sy, sx, p=8):
    H, W = x.shape[-2:]
    xp = F.pad(x, (p, p, p, p))                        # L, R, T, B
    return xp[:, :, p-sy : p-sy+H, p-sx : p-sx+W]

def warp(x, flow, align_corners=False):
    """
    x: [B, C, H, W]
    flow: [B, 2, H, W] where flow[:,0] = x_shift, flow[:,1] = y_shift
    align_corners: set to False, determines formula used
    return: [B, C, H, W]
    """
    B, C, H, W = x.shape

    # base grid
    ys, xs = torch.meshgrid(torch.arange(H, device=x.device, dtype=torch.float32), 
                            torch.arange(W, device=x.device, dtype=torch.float32), indexing='ij')

    sample_x = xs - flow[:,0]
    sample_y = ys - flow[:,1]

    # selects correct formula based on align_corners
    if align_corners:
        gx = 2 * sample_x / (W - 1) - 1
        gy = 2 * sample_y / (H - 1) - 1
    else:
        gx = (2 * sample_x + 1) / W - 1
        gy = (2 * sample_y + 1) / H - 1

    # shift
    grid = torch.stack((gx, gy), -1)
    return F.grid_sample(x, grid, mode='bilinear',
                        padding_mode='zeros', align_corners=align_corners)

# Test warp function
flow = torch.zeros(1, 2, 16, 16)
flow[:, 0, :, :8] = 1.6        # left half only
flow[:, 1, :, :8] = 2.3

#w = warp(x, flow)
#print('left :', (w[..., 3:-3, 3:8] - out[..., 3:-3, 3:8]).abs().max().item())
#print('right:', (w[..., 3:-3, 8:]  - x  [..., 3:-3, 8:] ).abs().max().item())

def cost_vol(ref, cur, r):
    """
    ref: [B, C, H, W]
    cur: [B, C, H, W]
    r: radius (int)
    out: [B, (2r+1)^2, H, W]

    decoder: dx, dy
    """
    B, C, H, W = ref.shape    
    ref_n = F.normalize(ref, p=2, dim=1)
    cur_n = F.normalize(cur, p=2, dim=1)
    cur_pad = F.pad(cur_n, (r,r,r,r), mode='constant', value=0)

    cost_list = []
    for dx in range(-r, r+1):
        for dy in range(-r, r+1):
            cur_slice = cur_pad[:,:, r+dy:r+dy+H, r+dx:r+dx+W]
            cost_list.append((ref_n * cur_slice).sum(dim=1))

    return torch.stack(cost_list, dim=1)

ref = torch.randn(1, 8, 32, 32)
cur = shift_fast(ref, sy=2, sx=3)
r=4
corr = cost_vol(ref, cur, r)
peak = corr[0, :, 8:-8, 8:-8].mean((1, 2)).argmax().item()
dx, dy = divmod(peak, 2*r+1)
#print(dy - r, dx - r)

def up_flow(f):
    """f: [B, 2, h, w] in level-h pixel units -> [B, 2, 2h, 2w] in fine pixels"""
    return F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False) * 2.0

H = W = 12
ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                        torch.arange(W, dtype=torch.float32), indexing='ij')

fc = torch.stack([0.30*xs - 0.20*ys + 1.1,
                  0.15*xs + 0.40*ys - 0.6]).unsqueeze(0)      # [1,2,12,12]
ff = up_flow(fc)                                              # [1,2,24,24]

ys2, xs2 = torch.meshgrid(torch.arange(2*H, dtype=torch.float32),
                          torch.arange(2*W, dtype=torch.float32), indexing='ij')
ux, uy = (xs2 + 0.5)/2 - 0.5, (ys2 + 0.5)/2 - 0.5             # fine index -> coarse coord
gt = torch.stack([2*(0.30*ux - 0.20*uy + 1.1),
                  2*(0.15*ux + 0.40*uy - 0.6)]).unsqueeze(0)

#print((ff - gt)[..., 1:-1, 1:-1].abs().max().item())

def scatter_flow(self, om, flow):
    """
    om: [B, 112, H, w] offset proj
    flow: [B, 2, H, W] accumulated flow (sample = p - flow)
    out: [B, 112, H, W] ready for DCNv4 block
    """
    # houses dx dx for all respective channels
    delta = torch.zeros_like(om)
    delta[:, self.idx_dx] = -flow[:, 0:1]
    delta[:, self.idx_dy] = -flow[:, 1:]

    # apply flow to om
    out = om + delta
    return out

def test_scatter_flow_matches_warp():
    G, Kg = 4, 9
    grp = torch.arange(G) * (Kg * 3)
    pt = torch.arange(Kg) * 2
    idx_dx = (grp[:, None] + pt[None, :]).flatten()
    idx_dy = idx_dx + 1
    idx_m = (grp[:, None] + Kg * 2 + torch.arange(Kg)).flatten()

    class _Dummy:
        def __init__(self, idx_dx, idx_dy, idx_m):
            self.idx_dx, self.idx_dy, self.idx_m = idx_dx, idx_dy, idx_m
        scatter_flow = scatter_flow

    m = _Dummy(idx_dx, idx_dy, idx_m)

    B, C, H, W = 1, 64, 16, 16
    xt = torch.randn(B, C, H, W).cuda()
    a, b = 1.3, -0.7
    flow = torch.zeros(B, 2, H, W).cuda()
    flow[:, 0] = a
    flow[:, 1] = b

    om = torch.zeros(B, 112, H, W).cuda()
    centre = idx_m.view(G, Kg)[:, 4]   # [22, 49, 76, 103]
    om[:, centre] = 1.0

    dcn = DCNv4Block(channels=C, kernel_size=3, pad=1, stride=1,
                      groups=G, without_pointwise=True).cuda()

    out_dcn = dcn(xt, m.scatter_flow(om, flow))
    out_warp = warp(xt, flow)

    diff = (out_dcn - out_warp)[..., 3:-3, 3:-3].abs().max().item()
    print("max abs diff:", diff)   # want ~1e-3

test_scatter_flow_matches_warp()


"""def forward():
    B, N, C, H, W = x.size()

            # Feature Extraction (all frames as one batch)
            burst_reshaped = x.view(B * N, C, H, W)
            curr_feat_lv1 = self.feat_extractor_lv1(burst_reshaped)   # (B*N, num_feat, H, W)
            curr_feat_lv2 = self.feat_extractor_lv2(curr_feat_lv1)    # (B*N, num_feat, H/2, W/2)

            # Reference frame features
            ref_feat_lv1 = curr_feat_lv1.view(B, N, -1, H, W)[:, self.center_frame_idx]            # (B, C, H, W)
            ref_feat_lv2 = curr_feat_lv2.view(B, N, -1, H // 2, W // 2)[:, self.center_frame_idx]  # (B, C, H/2, W/2)
            ref_feat_lv1_bc = ref_feat_lv1.repeat_interleave(N, dim=0)  # (B*N, C, H, W)
            ref_feat_lv2_bc = ref_feat_lv2.repeat_interleave(N, dim=0)  # (B*N, C, H/2, W/2)

            # The reference frame is intentionally NOT skipped:
            # Though in previous iterations it was, routing it through the
            # same pipeline as all others means it will look "identical" to them
            # thus it will be weighted as an equal with temporal fusion

            # Lv2 Alignment  (coarse alignment, H/2 x W/2 -> feats are compressed)
            concat_lv2 = torch.cat([curr_feat_lv2, ref_feat_lv2_bc], dim=1)
            offset_feat_lv2 = self.offset_conv_lv2(concat_lv2) # (B, num_feat, H/2, W/2)
            offset_mask_lv2 = self.offset_proj_lv2(offset_feat_lv2) # (B, padded_C, H/2, W/2)

            aligned_lv2 = self.dcn_lv2(curr_feat_lv2, offset_mask_lv2) # (B, num_feat, H/2, W/2)

            # Coarse to Fine propagation
            # Usample the projected offset_mask tensor and apply offset_scaling to it
            # offset masks are geometric "pixel-displacements" and not the features
            # represented by offset_feat_lv. 
            # Thus it is multiplied by 2 for delta x/y chans and 1 for mask/padding chans
            up_offset_mask_lv2 = self.upsample(offset_mask_lv2) * self.offset_scale

            # Only upsample the features, thus no scaling is applied to these (feature-space)
            up_offset_feat_lv2 = self.upsample(offset_feat_lv2) # (B, num_feat, H, W), NO ×2

            # Upsample coarse aligned features (post DCN)
            up_aligned_lv2 = self.upsample(aligned_lv2) # (B, num_feat, H, W)

            # Lv1 Alignment  (fine alignment, H x W)
            concat_lv1 = torch.cat([curr_feat_lv1, ref_feat_lv1_bc], dim=1)
            offset_feat_lv1_base = self.offset_conv_lv1_1(concat_lv1) # (B, num_feat, H, W)

            # Residual offset: condition on both the fine-level prior and the
            # upsampled coarse feature (which encodes where large-scale motion
            # was already estimated).
            offset_feat_lv1 = self.offset_conv_lv1_2(
                torch.cat([offset_feat_lv1_base, up_offset_feat_lv2], dim=1)
            )
            offset_mask_lv1 = self.offset_proj_lv1(offset_feat_lv1)      # (B, padded_C, H, W)

            # Now instead of adding the offset features, add the offset masks
            offset_mask_lv1 = offset_mask_lv1 + up_offset_mask_lv2

            aligned_lv1 = self.dcn_lv1(curr_feat_lv1, offset_mask_lv1) # (B, num_feat, H, W)

            # Fuse fine-aligned features with upsampled coarse aligned features
            aligned_lv1_fused = self.feat_fuse_lv1(
                torch.cat([aligned_lv1, up_aligned_lv2], dim=1) # (B, num_feat, H, W)
            )

            # Cascading Refinement
            concat_casc     = torch.cat([aligned_lv1_fused, ref_feat_lv1_bc], dim=1)
            casc_offset_feat = self.casc_offset_conv(concat_casc)
            casc_offset_mask = self.casc_offset_proj(casc_offset_feat)

            final_aligned = self.casc_dcn(aligned_lv1_fused, casc_offset_mask)

            final_aligned = self.lrelu(final_aligned) # Mirror EVDR final activation

            return final_aligned.view(B, N, -1, H, W), ref_feat_lv1
"""