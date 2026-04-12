import torch.nn as nn
from einops import rearrange
from burstISP.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class TemporalFusion(nn.Module):
    def __init__(self, num_frames=5, num_feat=64, window_size=16, num_heads=4):
        super(TemporalFusion, self).__init__()
        self.num_frames = num_frames
        self.num_feat = num_feat
        self.window_size = window_size

        # Normalization
        self.norm = nn.LayerNorm(num_feat)

        # Cross Attention
        self.q_proj = nn.Linear(num_feat, num_feat)
        self.k_proj = nn.Linear(num_feat, num_feat)
        self.v_proj = nn.Linear(num_feat, num_feat)
        self.attn = nn.MultiheadAttention(embed_dim=num_feat, num_heads=num_heads, batch_first=True)

        # Feed Forward MLP
        self.norm2 = nn.LayerNorm(num_feat)
        self.mlp = nn.Sequential(
            nn.Linear(num_feat, 4*num_feat),
            nn.GELU(),
            nn.Linear(4*num_feat, num_feat)
        )

        # Final Fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        )


    def forward(self, x):
        B, N, C, H, W = x.shape # [B, N = 5, C = 64, H = 80, W = 80]

        # Reference image (frame 7 in dataset, middle index)
        ref = N // 2

        # Partition into 20 windows size 16x16px (default)
        # Rearranges from B, N, C, H, W -> B*Windows, N, Window Tokens, C
        x_windows = rearrange(x, 'b n c (h w_h) (w w_w) -> (b h w) n (w_h w_w) c', w_h = self.window_size, w_w = self.window_size)
        
        # Norm
        x_norm = self.norm(x_windows)

        ref_feats = x_norm[:, ref:ref+1, :, :]

        # Calculate cross-attention for each window (query=frame 7 -> index 2, key=frame i)
        # do this by compairing the window of each key to the query
        # Flatten frames and spatial dimensions for K and V 
        # Q: [Batch*Windows, Tokens, C] 
        # K, V: [Batch*Windows, N*Tokens, C]
        x_query = self.q_proj(rearrange(ref_feats, 'bw 1 t c -> bw t c'))
        x_key = self.k_proj(rearrange(x_norm, 'bw n t c -> bw (n t) c'))
        x_value = self.v_proj(rearrange(x_norm, 'bw n t c -> bw (n t) c'))

        attn_out, _ = self.attn(query=x_query, key=x_key, value=x_value)

        # Concat
        out = attn_out + rearrange(ref_feats, 'bw 1 t c -> bw t c')

        # Norm + Feed fwd MLP
        out = out + self.mlp(self.norm2(out))

        # Reverse Partitioning
        out = rearrange(out, '(b h w) (w_h w_w) c -> b c (h w_h) (w w_w)', b=B, h=H//self.window_size, w=W//self.window_size, w_h=self.window_size, w_w=self.window_size)
        
        # Final Conv
        out = self.fusion_conv(out) # B C=num_feat H W
        
        return out
