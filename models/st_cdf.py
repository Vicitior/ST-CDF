import torch
import torch.nn as nn
from models.gat_layer import SpatialFeatureExtractor
from models.diff_attention import MultiHeadDifferentialAttention
from models.wavelet_transform import WaveletTransform, InverseWaveletTransform
from models.diffusion_process import DiffusionProcess

class ST_CDF(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim=128, diff_steps=50):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        # Spatial-Temporal features
        self.spatial_extractor = SpatialFeatureExtractor(num_nodes, hidden_dim, hidden_dim)
        self.diff_attention = MultiHeadDifferentialAttention(hidden_dim, num_heads=4)
        
        # Frequency decoupling
        self.dwt = WaveletTransform(wavelet='haar', level=1)
        self.idwt = InverseWaveletTransform(wavelet='haar')
        
        # Diffusion specific
        self.diffusion = DiffusionProcess(num_steps=diff_steps)
        self.time_embed_layer = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.output_proj = nn.Linear(hidden_dim, in_features)

    def forward(self, x, mask, t, edge_index):
        """
        x: (B, T, N, C) - corrupted data (or x_t during diffusion)
        mask: (B, T, N, C) - conditional mask
        t: (B,) - time steps
        """
        # Embed time step
        import math
        half_dim = 64
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        t_emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)
        t_emb = self.time_embed_layer(t_emb).unsqueeze(1).unsqueeze(2) # (B, 1, 1, hidden_dim)
        
        # Project input
        h = self.input_proj(x)
        
        # Add time embedding
        h = h + t_emb
        
        # DWT Decoupling (Optional application before attention)
        # To keep it differentiable in torch, we skip actual pywt in forward training loop 
        # unless custom autograd is defined. Usually DWT is used to process condition data.
        # Here we just use the spatial-temporal components directly as the core UNet-like structure.
        
        # Spatial Graph Attention
        h_spatial = self.spatial_extractor(h, edge_index)
        
        # Differential Attention
        h_attn = self.diff_attention(h_spatial)
        
        # Final projection to predict noise
        noise_pred = self.output_proj(h_attn)
        return noise_pred
