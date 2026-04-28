import torch
import torch.nn as nn
from einops import rearrange
from burstISP.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class TemporalFusion(nn.Module):
    def __init__(self, num_frames=5, num_feat=64, window_size=4, num_heads=4):
        super(TemporalFusion, self).__init__()
        self.num_frames = num_frames
        self.num_feat = num_feat
        self.window_size = window_size

        # Positional and Temporal Encodings
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, window_size * window_size, num_feat)) # [1, 1, pixels_per_window, C]
        self.temp_pos_embed = nn.Parameter(torch.zeros(1, num_frames, 1, num_feat)) # [1, num_frames, 1, C]

        # Encoding Initialization
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temp_pos_embed, std=0.02)

        # Normalization
        self.norm1 = nn.LayerNorm(num_feat)

        # Cross Attention (Internal Q, K, V projections are handled by PyTorch natively)
        # batch_first=True expects shape [Batch, Sequence, Feature]
        self.attn = nn.MultiheadAttention(embed_dim=num_feat, num_heads=num_heads, batch_first=True)

        # Feed Forward MLP
        self.norm2 = nn.LayerNorm(num_feat)
        self.mlp = nn.Sequential(
            nn.Linear(num_feat, 4 * num_feat),
            nn.GELU(),
            nn.Linear(4 * num_feat, num_feat)
        )

        # Final Fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        B, N, C, H, W = x.shape # [B, N=5, C=64, H=80, W=80]
        ws = self.window_size

        assert H % ws == 0 and W % ws == 0, f"H ({H}) and W ({W}) must be divisible by window size ({ws})"

        ref = N // 2

        # Partition image into non-overlapping windows
        x_win = rearrange(x, 'b n c (h ws1) (w ws2) -> (b h w) n (ws1 ws2) c', ws1=ws, ws2=ws) # [B * num_windows, N, pixels_per_window, C]

        # Add positional and temporal encodings to window
        x_pos = x_win + self.spatial_pos_embed + self.temp_pos_embed

        # Normalize
        x_pos_norm = self.norm1(x_pos)

        # Query, Key, and Value
        x_query = x_pos_norm[:, ref, :, :]
        x_k = rearrange(x_pos_norm, 'b n p c -> b (n p) c')

        x_win_norm = self.norm1(x_win)
        x_v = rearrange(x_win_norm, 'b n p c -> b (n p) c')

        # Multi-head Cross Attention
        attn_out, _ = self.attn(query=x_query, key=x_k, value=x_v) # [B*num_windows, 16, C]

        # Residual Connection (Add unnormalized reference frame)
        ref_unnorm = x_win[:, ref, :, :]
        out = attn_out + ref_unnorm

        # Feed Forward MLP with residual
        out = out + self.mlp(self.norm2(out))

        # Reverse Window Partitioning back to image shape
        out = rearrange(out, '(b h w) (ws1 ws2) c -> b c (h ws1) (w ws2)', 
                        b=B, h=H//ws, w=W//ws, ws1=ws, ws2=ws)
        
        out = self.fusion_conv(out)
        
        return out