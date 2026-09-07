#!/usr/bin/env python3
"""Analytical FLOP / activation-memory model for ST-HAT's stage-1 blocks.

Companion to `param_budget.py`, which prices parameters. This prices the two
things parameters do not predict: multiply-accumulates per forward pass, and
bytes of activation held for the backward pass. Pure arithmetic, no torch, so
config variants and hypothetical blocks can be priced without a GPU.

Written to answer one question: what would replacing `SpatioTemporalBlock`'s
`WindowAttention3D` with a MambaIRv2-style selective scan actually buy?

Conventions
-----------
* "FLOPs" means multiply-accumulates, matching thop/fvcore and therefore the
  numbers BurstMamba (arXiv 2503.19634) and QMambaBSR (arXiv 2408.08665)
  report. Elementwise ops (norms, activations, residual adds) are excluded
  from the MAC count, as in those papers, but their *activations* are counted.
* Activation bytes = tensors autograd holds until backward. This, not the
  parameter count, is what sets `batch_size_per_gpu`.
* Byte widths follow `torch.autocast(bfloat16)` as actually used in
  `mambafusion_model.py:141`. Matmul/conv/linear inputs are held at 2 bytes.
  `softmax`, `log_softmax` and `layer_norm` are on autocast's **fp32 cast
  list**, so they run in fp32 and hold a 4-byte copy of what they saved --
  and where an fp32 result then feeds a matmul, autograd additionally holds
  the 2-byte cast copy. The attention matrix therefore costs 6 bytes per
  element, not 2. `Selective_Scan.forward_core` hardcodes `.float()` on its
  scan inputs (mambairv2_arch.py:388-393), so the scan is priced at 4 bytes.
"""
import math
from dataclasses import dataclass, field

BF16, FP32 = 2, 4


@dataclass
class Cost:
    """MACs and activation bytes, with a per-line breakdown for reporting."""
    macs: int = 0
    act: int = 0
    params: int = 0
    lines: list = field(default_factory=list)

    def add(self, name, macs=0, act=0, params=0):
        self.macs += macs
        self.act += act
        self.params += params
        if macs or act:
            self.lines.append((name, macs, act))
        return self

    def __iadd__(self, other):
        self.macs += other.macs
        self.act += other.act
        self.params += other.params
        self.lines += other.lines
        return self


# ---------------------------------------------------------------- primitives
# Each returns (macs, act_bytes). `act` is what the op saves for backward:
# a matmul/linear/conv saves its input; softmax saves its output; layer_norm
# and the elementwise ops save their input.

def linear(tok, cin, cout, w=BF16):
    return tok * cin * cout, tok * cin * w


def conv(px, cin, cout, k, w=BF16):
    return px * cin * cout * k * k, px * cin * w


def layernorm(tok, dim):
    # fp32 under autocast: holds an fp32 copy of the input on top of the bf16
    # tensor its producer already owns.
    return 0, tok * dim * FP32


def elemwise(tok, dim, n_saved=1, w=BF16):
    return 0, tok * dim * w * n_saved


def softmax_attn(elems):
    # fp32 saved output (softmax backward) + bf16 cast copy (the attn @ v matmul).
    return 0, elems * (FP32 + BF16)


def mlp(tok, dim, ratio):
    """LayerNorm -> Linear -> GELU -> Linear, the ST-HAT stage-1 FFN."""
    c = Cost()
    h = dim * ratio
    c.add('  norm', *layernorm(tok, dim))
    c.add('  fc1', *linear(tok, dim, h), params=dim * h + h)
    c.add('  gelu', *elemwise(tok, h))
    c.add('  fc2', *linear(tok, h, dim), params=h * dim + dim)
    return c


# ------------------------------------------------------------- ST-HAT blocks

def spatiotemporal_block(B, N, C, H, W, ws, heads, ratio):
    """Current block: 3D window attention over (N x ws x ws) tubes."""
    tok = B * N * H * W
    nwin = B * (H // ws) * (W // ws)
    n = N * ws * ws                       # tokens per window
    c = Cost()
    c.add('norm1', *layernorm(tok, C), params=2 * C)
    c.add('wqkv', *linear(tok, C, 3 * C), params=C * 3 * C + 3 * C)
    # q,k,v are slices of one contiguous permute; count the tensor once.
    c.add('qkv (saved by both matmuls)', 0, tok * 3 * C * BF16)
    c.add('q @ k^T', nwin * heads * n * n * (C // heads), 0)
    c.add('softmax(attn)', *softmax_attn(nwin * heads * n * n))
    c.add('attn @ v', nwin * heads * n * n * (C // heads), 0)
    c.add('proj', *linear(tok, C, C), params=C * C + C)
    c.add('residual', *elemwise(tok, C, 0))
    c += mlp(tok, C, ratio)
    c.params += 2 * C + (2 * ws - 1) ** 2 * (2 * N - 1) * heads
    return c


def spatial_block(B, N, C, H, W, ws, heads, ratio):
    tok = B * N * H * W
    nwin = B * N * (H // ws) * (W // ws)
    n = ws * ws
    c = Cost()
    c.add('norm1', *layernorm(tok, C), params=2 * C)
    c.add('wqkv', *linear(tok, C, 3 * C), params=C * 3 * C + 3 * C)
    c.add('qkv', 0, tok * 3 * C * BF16)
    c.add('q @ k^T', nwin * heads * n * n * (C // heads), 0)
    c.add('softmax', *softmax_attn(nwin * heads * n * n))
    c.add('attn @ v', nwin * heads * n * n * (C // heads), 0)
    c.add('proj', *linear(tok, C, C), params=C * C + C)
    c += mlp(tok, C, ratio)
    c.params += 2 * C + (2 * ws - 1) ** 2 * heads
    return c


def temporal_block(B, N, C, H, W, heads, ratio):
    """Per-pixel attention across frames: B*H*W sequences of length N."""
    tok = B * N * H * W
    nseq = B * H * W
    c = Cost()
    c.add('norm1', *layernorm(tok, C), params=2 * C)
    c.add('in_proj_qkv', *linear(tok, C, 3 * C), params=C * 3 * C + 3 * C)
    c.add('qkv', 0, tok * 3 * C * BF16)
    c.add('q @ k^T', nseq * heads * N * N * (C // heads), 0)
    c.add('softmax', *softmax_attn(nseq * heads * N * N))
    c.add('attn @ v', nseq * heads * N * N * (C // heads), 0)
    c.add('out_proj', *linear(tok, C, C), params=C * C + C)
    c += mlp(tok, C, ratio)
    c.params += 2 * C + N * C
    return c


# ------------------------------------------------------- SSM replacement(s)

def selective_scan(tok, d_inner, d_state, w=FP32):
    """mamba_ssm selective_scan_fn + the x_proj/dt_proj that feed it.

    The CUDA kernel recomputes states in backward, so it holds O(tok*d_inner),
    not O(tok*d_inner*d_state): u, delta and out at `w` bytes each, plus the
    per-token B and C. Scan MACs follow the official 9*L*D*N accounting.
    """
    c = Cost()
    dt_rank = math.ceil(d_inner / 16)
    c.add('  x_proj', tok * d_inner * (dt_rank + 2 * d_state), tok * d_inner * w,
          params=d_inner * (dt_rank + 2 * d_state))
    c.add('  dt_proj', tok * dt_rank * d_inner, tok * dt_rank * w,
          params=dt_rank * d_inner + d_inner)
    c.add('  scan (u, delta, out)', 9 * tok * d_inner * d_state, 3 * tok * d_inner * w)
    c.add('  scan (B, C)', 0, 2 * tok * d_state * w)
    c.params += d_inner * d_state + d_inner          # A_logs, Ds
    return c


def assm(tok, C, d_state, expand, num_tokens, inner_rank, scan_w=FP32):
    """MambaIRv2 ASSM: route -> gumbel -> SGN sort -> selective scan -> unsort."""
    c = Cost()
    h = int(C * expand)
    r = C // 3
    # --- SGN router (the 'learn the order' half of the proposal)
    c.add('route fc1', *linear(tok, C, r), params=C * r + r)
    c.add('route gelu', *elemwise(tok, r))
    c.add('route fc2', *linear(tok, r, num_tokens), params=r * num_tokens + num_tokens)
    c.add('log_softmax', 0, tok * num_tokens * FP32)
    c.add('gumbel_softmax(hard)', 0, tok * num_tokens * BF16)
    c.add('prompt = policy @ emb', tok * num_tokens * d_state, 0)
    c.params += num_tokens * inner_rank + inner_rank * d_state
    # --- body
    c.add('in_proj 1x1', *conv(tok, C, h, 1), params=C * h + h)
    c.add('CPE dwconv', *conv(tok, h, 1, 3), params=h * 9 + h)
    c.add('gate x * sigmoid(CPE)', 0, 2 * tok * h * BF16)
    c.add('SGN gather', 0, 0)                       # index only; tensor counted by scan
    c += selective_scan(tok, h, d_state, scan_w)
    c.add('out_norm', *layernorm(tok, h), params=2 * h)
    c.add('out_proj', *linear(tok, h, C), params=h * C + C)
    return c


def st_mamba_block(B, N, C, H, W, heads, ratio, d_state=16, expand=2,
                   num_tokens=64, inner_rank=32, scan_w=FP32):
    """Proposed swap: WindowAttention3D -> ASSM over the spatio-temporal volume.

    Same skeleton as SpatioTemporalBlock (norm -> mixer -> residual -> FFN),
    so the diff against the baseline isolates the mixer.
    """
    tok = B * N * H * W
    c = Cost()
    c.add('norm1', *layernorm(tok, C), params=2 * C)
    c += assm(tok, C, d_state, expand, num_tokens, inner_rank, scan_w)
    c.add('residual', *elemwise(tok, C, 0))
    c += mlp(tok, C, ratio)
    return c


# --------------------------------------------------------------- reporting

def gb(x):
    return x / 1024 ** 3


def report(name, c, per_sample_B=None):
    print(f"\n{name}")
    print(f"  {'':<34}{'GMACs':>10}{'act MB':>11}")
    for ln, m, a in c.lines:
        if m or a:
            print(f"  {ln:<34}{m/1e9:10.2f}{a/1024**2:11.1f}")
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<34}{c.macs/1e9:10.2f}{c.act/1024**2:11.1f}"
          f"   params {c.params/1e3:8.1f}k")
    if per_sample_B:
        print(f"  {'per sample':<34}{c.macs/1e9/per_sample_B:10.2f}"
              f"{c.act/1024**2/per_sample_B:11.1f}")


# ------------------------------------- rest of the model, for whole-net FLOPs
# Coarser than the stage-1 blocks above (MACs only, no activation breakdown):
# these exist to put the SpatioTemporalBlock's share in context and to compare
# the whole network against BurstMamba's published 63 + 1.3(L-1) GFLOPs.

def fusion_block_macs(B, N, C, H, W, ws, heads, ratio):
    tok = B * N * H * W
    nwin = B * (H // ws) * (W // ws)
    P, n = ws * ws, N * ws * ws
    m = tok * C * C                                   # wk over all frames
    m += (B * H * W) * C * C * 2                      # wq, proj on ref only
    m += tok * C * C                                  # wv on the signed diffs
    m += nwin * heads * P * n * (C // heads) * 2      # q@k^T and attn@v
    ref_tok = B * H * W
    m += ref_tok * (C * (C // 2) + (C // 2) * C + C * C)   # back-projection
    m += ref_tok * (C * ratio * C * 2)                # mlp on the collapsed map
    m += tok * (2 * C * C * 9) + ref_tok * C * C * 9  # gate_conv, diff_fuse
    return m


def ocab_macs(B, C, H, W, ws, heads, ratio, overlap):
    tok = B * H * W
    ows = int(ws * overlap) + ws
    nwin = B * (H // ws) * (W // ws)
    m = tok * C * 3 * C + tok * C * C
    m += nwin * heads * (ws * ws) * (ows * ows) * (C // heads) * 2
    m += tok * C * ratio * C * 2
    return m


def hab_macs(B, C, H, W, ws, heads, ratio):
    tok = B * H * W
    nwin = B * (H // ws) * (W // ws)
    n = ws * ws
    m = tok * C * 3 * C + tok * C * C
    m += nwin * heads * n * n * (C // heads) * 2
    m += tok * C * ratio * C * 2
    m += tok * (C * (C // 3) * 9 * 2 + C * (C // 30) + (C // 30) * C)   # CAB
    return m


def attentive_layer_macs(B, E, H, W, ws, heads, d_state, ratio, inner_rank,
                         num_tokens, kf):
    tok = B * H * W
    nwin = B * (H // ws) * (W // ws)
    n = ws * ws
    h = int(E * ratio)
    dt_rank = math.ceil(h / 16)
    m = tok * E * 3 * E + tok * E * E                        # wqkv, proj
    m += nwin * heads * n * n * (E // heads) * 2             # window attention
    m += tok * (E * (E // 3) + (E // 3) * num_tokens)        # route
    m += tok * num_tokens * d_state
    m += tok * (E * h + h * 9 + h * E)                       # in_proj, CPE, out_proj
    m += tok * (h * (dt_rank + 2 * d_state) + dt_rank * h)   # x_proj, dt_proj
    m += 9 * tok * h * d_state                               # scan
    m += 2 * tok * (E * h + h * kf * kf + h * E)             # two ConvFFNs
    return m


def whole_model_macs(cfg, B, st_block_macs):
    """Total MACs for one forward at the config's training resolution."""
    N, Hb, C = cfg['N'], cfg['H'], cfg['C']
    F, E, ws = cfg['num_feat'], cfg['embed_dim'], cfg['ws']
    heads, ratio = cfg['heads'], cfg['ratio']
    parts = {}
    # PreAlign: conv3x3(4->F) + conv3x3(F->4F) on the packed 48x48 grid, then x2.
    px_lr = B * N * (Hb // 2) * (Hb // 2)
    parts['PreAlign'] = px_lr * (4 * F * 9 + F * 4 * F * 9)
    # BurstAlign: 7 feature convs + flow heads + one DCN, mostly at 96x96.
    px = B * N * Hb * Hb
    cin = F if cfg['pre_align'] else 4
    px2, px3 = px // 4, px // 16                  # lv2 and lv3 are stride-2 each
    al = px * (cin * F * 9 + 2 * F * F * 9)       # feat_extractor_lv1 (3 convs)
    al += px2 * (2 * F * F * 9)                   # feat_extractor_lv2
    al += px3 * (2 * F * F * 9)                   # feat_extractor_lv3
    al += px3 * ((2 * cfg.get('offset_r', 2) + 1) ** 2 * 2 * 9)   # flow_head_lv3
    al += px2 * (2 * F * F * 9 + F * 2 * 9)       # offset_conv_lv2
    al += px * 2 * (2 * F * F * 9 + F * 2 * 9)    # offset_conv_lv1 + _casc
    al += px * (2 * F * F * 9)                    # offset_conv_dcn
    al += px * (F * 144 * 9)                      # offset_proj
    al += px * (F * F * 2 + F * 9)                # DCNv4 value/out proj + sampling
    parts['BurstAlign'] = al
    s1 = cfg['depth_s1']
    parts['ST-HAT s1 spatial'] = s1 * spatial_block(B, N, C, Hb, Hb, ws, heads, ratio).macs
    parts['ST-HAT s1 temporal'] = s1 * temporal_block(B, N, C, Hb, Hb, heads, ratio).macs
    parts['ST-HAT s1 spatiotemporal'] = s1 * st_block_macs
    parts['ST-HAT s2'] = (fusion_block_macs(B, N, C, Hb, Hb, ws, heads, ratio)
                          + spatial_block(B, 1, C, Hb, Hb, ws, heads, ratio).macs)
    parts['ST-HAT s3'] = cfg['depth_s3'] * (
        2 * ocab_macs(B, C, Hb, Hb, ws, heads, ratio, cfg['overlap'])
        + hab_macs(B, C, Hb, Hb, ws, heads, ratio))
    mb = sum(cfg['depths']) * attentive_layer_macs(
        B, E, Hb, Hb, cfg['mb_ws'], cfg['mb_heads'], cfg['d_state'],
        cfg['mb_ratio'], cfg['inner_rank'], cfg['num_tokens'], cfg['kf'])
    mb += len(cfg['depths']) * B * Hb * Hb * E * E * 9 * 2
    up = cfg['upsample_feat']
    mb += B * Hb * Hb * (E * up * 9 + 2 * up * 4 * up * 9) + B * (4 * Hb) ** 2 * up * 3 * 9
    parts['MambaIRv2'] = mb
    return parts


# ------------------------------------------------------------------- driver

L6 = dict(N=14, H=96, C=96, ws=8, st_ws=4, heads=4, ratio=4, overlap=0.25,
          depth_s1=3, depth_s3=1, num_feat=64, embed_dim=96, pre_align=True,
          depths=[2, 2, 2, 2], mb_ws=16, mb_heads=4, mb_ratio=2, d_state=16,
          inner_rank=32, num_tokens=64, kf=5, upsample_feat=64)


def main():
    cfg, B = L6, 2                      # batch_size_per_gpu from the L6 config
    N, H, C, heads, ratio = cfg['N'], cfg['H'], cfg['C'], cfg['heads'], cfg['ratio']
    print(f"MF_STHAT_L6  |  B={B}/gpu  N={N}  C={C}  {H}x{H} (Bayer)  "
          f"{B*N*H*H:,} tokens/fwd")
    print("FLOPs = multiply-accumulates (thop/fvcore convention). "
          "Activations = bytes held for backward under bf16 autocast.")

    base8 = spatiotemporal_block(B, N, C, H, H, 8, heads, ratio)
    base4 = spatiotemporal_block(B, N, C, H, H, cfg['st_ws'], heads, ratio)
    report("SpatioTemporalBlock, ws=8  (L3/L5 setting)", base8, B)
    report(f"SpatioTemporalBlock, ws={cfg['st_ws']}  (current L6 setting)", base4, B)

    variants = [
        ("ST-Mamba, expand=2, fp32 scan (repo's Selective_Scan as-is)",
         st_mamba_block(B, N, C, H, H, heads, ratio, expand=2, scan_w=FP32)),
        ("ST-Mamba, expand=2, bf16 scan (drop the .float() casts)",
         st_mamba_block(B, N, C, H, H, heads, ratio, expand=2, scan_w=BF16)),
        ("ST-Mamba, expand=1, bf16 scan (leanest defensible)",
         st_mamba_block(B, N, C, H, H, heads, ratio, expand=1, scan_w=BF16)),
    ]
    for name, c in variants:
        report(name, c, B)

    print("\n" + "=" * 72)
    print("STAGE-1 TOTALS (x3 blocks, the whole of stage 1's ST path)")
    print(f"  {'variant':<52}{'GMACs':>9}{'act GB':>9}")
    rows = [(f"attention ws=8", base8), (f"attention ws={cfg['st_ws']}", base4)] + variants
    for name, c in rows:
        print(f"  {name:<52}{3*c.macs/1e9:9.1f}{gb(3*c.act):9.2f}")

    print("\n  Reference points at the same B/N/resolution:")
    for name, c in [("SpatialBlock (x3)", spatial_block(B, N, C, H, H, cfg['ws'], heads, ratio)),
                    ("TemporalBlock (x3)", temporal_block(B, N, C, H, H, heads, ratio))]:
        print(f"  {name:<52}{3*c.macs/1e9:9.1f}{gb(3*c.act):9.2f}")

    print("\n" + "=" * 72)
    print("STAGE 1 AS A WHOLE (3 x [spatial + temporal + spatiotemporal])")
    sp = spatial_block(B, N, C, H, H, cfg['ws'], heads, ratio)
    tp = temporal_block(B, N, C, H, H, heads, ratio)
    s1_mac = 3 * (sp.macs + tp.macs + base4.macs)
    s1_act = 3 * (sp.act + tp.act + base4.act)
    # Everything after FusionBlock runs at N=1, so stage 1 holds essentially all
    # of the fusion module's activation memory.
    boundary = 3 * 3 * B * N * H * H * C * BF16      # one saved tensor per segment
    rows = [
        ("baseline (attention, st_ws=4)", s1_mac, s1_act),
        ("drop SpatioTemporalBlock entirely", 3 * (sp.macs + tp.macs),
         3 * (sp.act + tp.act)),
        ("checkpoint stage 1 (no arch change)", s1_mac * 4 // 3, boundary),
        ("ST-Mamba swap (expand=1, bf16 scan)",
         3 * (sp.macs + tp.macs + variants[2][1].macs),
         3 * (sp.act + tp.act + variants[2][1].act)),
        ("ST-Mamba swap + checkpoint stage 1",
         (3 * (sp.macs + tp.macs + variants[2][1].macs)) * 4 // 3, boundary),
    ]
    print(f"  {'intervention':<40}{'GMACs/smp':>11}{'act GB':>9}{'vs base':>9}")
    for nm, m, a in rows:
        print(f"  {nm:<40}{m/1e9/B:11.1f}{gb(a):9.2f}{gb(a)/gb(s1_act):8.0%}")

    print("\n" + "=" * 72)
    print("WHOLE NETWORK, one forward, 48x48 packed LR / 14 frames")
    for label, stc in [("attention ws=4 (current)", base4),
                       ("ST-Mamba expand=1 bf16", variants[2][1])]:
        parts = whole_model_macs(cfg, B, stc.macs)
        tot = sum(parts.values())
        print(f"\n  {label}")
        for k, v in parts.items():
            print(f"    {k:<30}{v/1e9/B:9.1f} GMACs/sample {100*v/tot:6.1f}%")
        print(f"    {'TOTAL':<30}{tot/1e9/B:9.1f} GMACs/sample")
        print(f"    {'BurstMamba (2503.19634), L=14':<30}{79.9:9.1f} GFLOPs/sample")

    # Sensitivity: if softmax turns out to run in bf16 rather than autocast's
    # fp32 (verify with torch.cuda.max_memory_allocated before trusting either),
    # the attention matrix costs 2 B/element instead of 6. That shrinks the
    # baseline and therefore shrinks what any mixer swap can recover -- the
    # ranking of the interventions below is unchanged.
    sm = next(a for nm, _, a in base4.lines if nm.startswith('softmax'))
    lean = s1_act - 3 * (sm * (FP32 + BF16 - BF16) // (FP32 + BF16))
    print(f"\n  Sensitivity: bf16 softmax -> stage 1 = {gb(lean):.2f} GB "
          f"(vs {gb(s1_act):.2f}); the swap recovers proportionally less.")


if __name__ == '__main__':
    main()
