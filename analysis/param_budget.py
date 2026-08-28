#!/usr/bin/env python3
"""Analytical parameter counter for MambaFusionNet.

Mirrors the module definitions arithmetically so config variants can be priced
without a GPU or a torch install. Validated against the reported 5.089M for
MF_STHAT_L3_SynBase (this script gives 5.088M).

Usage: python analysis/param_budget.py [config.yml ...]
"""
import sys, yaml
import math
def cv(i,o,k,bias=True): return i*o*k*k + (o if bias else 0)
def dw(c,k,bias=True):   return c*k*k + (c if bias else 0)
def ln(d):               return 2*d
def li(i,o,bias=True):   return i*o + (o if bias else 0)

def pre_align(in_ch,F):  return cv(in_ch,F,3) + cv(F,4*F,3)

def burst_align(Cin,F,G=4):
    K=G*9; pad=int(math.ceil(K*3/8)*8)
    t  = cv(Cin,F,3)+cv(F,F,3)+cv(F,F,3)          # feat_extractor_lv1
    t += cv(F,F,3)+cv(F,F,3)                      # feat_extractor_lv2
    for _ in range(3):                            # lv2 / lv1_2 / cascade offset convs
        t += cv(2*F,F,3)+cv(F,F,3)
    t += cv(2*F,F,3)                              # offset_conv_lv1_1
    t += 3*cv(F,pad,3)                            # 3 offset projections
    t += 3*(2*cv(F,F,1))                          # 3 DCNv4 value/output projs
    t += cv(2*F,F,3)                              # feat_fuse_lv1
    return t

def wattn(D,ws,H):       return li(D,D) + (2*ws-1)**2 * H
def spatial(D,ws,H,M):   return 2*ln(D) + li(D,3*D) + li(D,M*D)+li(M*D,D) + wattn(D,ws,H)
def st3d(D,ws,H,M,N):    return 2*ln(D) + li(D,3*D) + li(D,M*D)+li(M*D,D) + li(D,D) + (2*ws-1)**2*(2*N-1)*H
def temporal(D,H,M,N):   return N*D + 2*ln(D) + li(D,M*D)+li(M*D,D) + (3*D*D+3*D) + li(D,D)
def fusion_blk(D,ws,H,M,N): return 2*ln(D) + 4*li(D,D) + li(D,M*D)+li(M*D,D) + (2*ws-1)**2*(2*N-1)*H
def ocab(D,ws,OR,H,M):
    ows=int(ws*OR)+ws
    return 2*ln(D) + li(D,3*D) + (ws+ows-1)**2*H + li(D,D) + li(D,M*D)+li(M*D,D)
def hab(D,ws,H,M):
    cab = cv(D,D//3,3)+cv(D//3,D,3)+cv(D,D//30,1)+cv(D//30,D,1)
    return 2*ln(D) + li(D,3*D) + wattn(D,ws,H) + cab + li(D,M*D)+li(M*D,D)

def st_hat(N,ws,I,D,M,H,OR,s1,s3,O,st_ws=None):
    st_ws = ws if st_ws is None else st_ws
    t = cv(I,D,1)+cv(D,O,1)+cv(D,D,1)
    t += s1*(spatial(D,ws,H,M)+temporal(D,H,M,N)+st3d(D,st_ws,H,M,N))
    t += fusion_blk(D,ws,H,M,N) + spatial(D,ws,H,M)
    t += s3*(2*ocab(D,ws,OR,H,M)+hab(D,ws,H,M))
    return t

def attentive(E,S,W,H,R,T,KF,MR):
    h=int(E*MR); dr=math.ceil(h/16)
    ss = (dr+2*S)*h + h*dr + h + h*S + h          # x_proj, dt_projs w/b, A_logs, Ds
    assm = ss + ln(h) + li(h,E) + cv(E,h,1) + dw(h,3) + T*R + li(E,E//3) + li(E//3,T)
    ffn  = li(E,h) + dw(h,KF) + li(h,E)
    return 4*ln(E) + 2*E + li(E,3*E) + wattn(E,W,H) + assm + 2*ffn + R*S

def mambairv2(IC,E,depths,heads,W,R,T,KF,MR,S,upscale,UF,RC=None):
    RC = IC if RC is None else RC
    t = cv(IC,E,1) + cv(RC,E,1)                   # conv_first + skip_proj
    for d,H in zip(depths,heads):
        t += d*attentive(E,S,W,H,R,T,KF,MR) + cv(E,E,3)
    t += ln(E) + cv(E,E,3)                        # norm + conv_after_body
    t += cv(E,UF,3) + int(math.log2(upscale))*cv(UF,4*UF,3) + cv(UF,3,3)
    return t

def total(c, verbose=True):
    pa = pre_align(4,c['num_feat']) if c.get('pre_align') else 0
    al = burst_align(c['num_feat'] if c.get('pre_align') else 4, c['num_feat'], c['offset_groups'])
    fu = st_hat(c['num_frames'],c['fusion_ws'],c['num_feat'],c['fusion_feat'],c['fusion_mlp_ratio'],
                c['fusion_heads'],c['fusion_overlap'],c['fusion_depth_s1'],c['fusion_depth_s3'],c['out_feat'])
    up = c['scale']//2 if c.get('pre_align') else c['scale']
    mb = mambairv2(c['in_chans'],c['embed_dim'],c['depths'],c['num_heads'],c['window_size'],
                   c['inner_rank'],c['num_tokens'],c['convffn_kernel_size'],c['mlp_ratio'],
                   c['d_state'],up,c['upsample_feat'])
    tot = pa+al+fu+mb
    if verbose:
        for n,v in [('PreAlign',pa),('BurstAlign',al),('ST-HAT',fu),('MambaIRv2',mb)]:
            print(f"    {n:<12} {v/1e6:7.3f}M  {100*v/tot:5.1f}%")
        print(f"    {'TOTAL':<12} {tot/1e6:7.3f}M")
    return tot


def from_config(path):
    n = yaml.safe_load(open(path))['network_g']
    pa_on = n.get('pre_align', False)
    F, E = n['num_feat'], n['embed_dim']
    pa = pre_align(4, F) if pa_on else 0
    al = burst_align(F if pa_on else 4, F, n['offset_groups'])
    fu = st_hat(n['num_frames'], n['fusion_ws'], F, n['fusion_feat'], n['fusion_mlp_ratio'],
                n['fusion_heads'], n['fusion_overlap'], n['fusion_depth_s1'],
                n['fusion_depth_s3'], E, n.get('fusion_st_ws'))
    mb = mambairv2(E, E, n['depths'], n['num_heads'], n['window_size'], n['inner_rank'],
                   n['num_tokens'], n['convffn_kernel_size'], n['mlp_ratio'], n['d_state'],
                   n['scale'] // 2 if pa_on else n['scale'], n.get('upsample_feat', 64), F)
    tot = pa + al + fu + mb
    print(path)
    for name, v in [('PreAlign', pa), ('BurstAlign', al), ('ST-HAT', fu), ('MambaIRv2', mb)]:
        print(f"    {name:<12} {v/1e6:7.3f}M  {100*v/tot:5.1f}%")
    print(f"    {'TOTAL':<12} {tot/1e6:7.3f}M\n")
    return tot

if __name__ == '__main__':
    paths = sys.argv[1:] or ['main/configs/MF_STHAT_L3_SynBase.yml',
                             'main/configs/MF_STHAT_L5_PackedControl.yml',
                             'main/configs/MF_STHAT_L5_BayerSpace.yml']
    for p in paths:
        from_config(p)
