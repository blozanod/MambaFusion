import os
import sys

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from burstISP.archs.dcn_align_arch import BurstAlign


class _Dummy:
    """Minimal stand-in exposing only the buffers scatter_flow reads."""
    def __init__(self, idx_dx, idx_dy, idx_m):
        self.idx_dx, self.idx_dy, self.idx_m = idx_dx, idx_dy, idx_m

    scatter_flow = BurstAlign.scatter_flow


def test_scatter_flow_constant_flow():
    """Test A (structural): om = zeros, constant flow (a, b).
    Every idx_dx channel must read -a, every idx_dy channel -b,
    every idx_m channel must stay 0 (untouched).
    """
    G, Kg = 4, 9
    grp = torch.arange(G) * (Kg * 3)
    pt = torch.arange(Kg) * 2
    idx_dx = (grp[:, None] + pt[None, :]).flatten()
    idx_dy = idx_dx + 1
    idx_m = (grp[:, None] + Kg * 2 + torch.arange(Kg)).flatten()

    m = _Dummy(idx_dx, idx_dy, idx_m)

    B, H, W = 2, 4, 4
    om = torch.zeros(B, 112, H, W)
    a, b = 1.7, -2.3
    flow = torch.zeros(B, 2, H, W)
    flow[:, 0] = a
    flow[:, 1] = b

    out = m.scatter_flow(om, flow)

    assert torch.allclose(out[:, idx_dx], torch.full_like(out[:, idx_dx], -a))
    assert torch.allclose(out[:, idx_dy], torch.full_like(out[:, idx_dy], -b))
    assert torch.allclose(out[:, idx_m], torch.zeros_like(out[:, idx_m]))
    print("test_scatter_flow_constant_flow: PASS")


if __name__ == "__main__":
    test_scatter_flow_constant_flow()
