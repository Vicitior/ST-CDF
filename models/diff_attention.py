import torch
import torch.nn as nn

try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False

class MultiHeadDifferentialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """
        x: (B, T, N, C) -> we flatten N and T or attend over T.
        Usually attention is over Time (T).
        Let's reshape to (B*N, T, C)
        """
        B, T, N, C = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, C) # (B*N, T, C)
        
        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)
        
        # Differential part: Add temporal difference to Queries and Keys
        q_diff = torch.cat([q[:, :1, :], q[:, 1:, :] - q[:, :-1, :]], dim=1)
        k_diff = torch.cat([k[:, :1, :], k[:, 1:, :] - k[:, :-1, :]], dim=1)
        
        q = q + 0.5 * q_diff
        k = k + 0.5 * k_diff
        
        if HAS_FLASH and q.dtype in [torch.float16, torch.bfloat16] and q.is_cuda:
            # FlashAttention requires (batch, seqlen, nheads, headdim)
            q_flash = q.view(B * N, T, self.num_heads, self.head_dim)
            k_flash = k.view(B * N, T, self.num_heads, self.head_dim)
            v_flash = v.view(B * N, T, self.num_heads, self.head_dim)
            
            out = flash_attn_func(q_flash, k_flash, v_flash, dropout_p=0.0)
            out = out.reshape(B * N, T, C)
        else:
            # Fallback to standard PyTorch attention
            q_std = q.view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)
            k_std = k.view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)
            v_std = v.view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)
            
            scores = torch.matmul(q_std, k_std.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_std).transpose(1, 2).reshape(B * N, T, C)
            
        out = self.out_proj(out)
        out = out.reshape(B, N, T, C).permute(0, 2, 1, 3) # Back to (B, T, N, C)
        return out
