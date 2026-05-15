import torch
import torch.nn as nn

class LightweightStudentTransformer(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_layers=2, num_heads=4):
        super().__init__()
        # Aiming for ~3.8M parameters
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, in_features)

    def forward(self, x, mask=None):
        """
        x: (B, T, N, C)
        Student operates without heavy graph/diffusion parts, acting as a direct predictor.
        """
        B, T, N, C = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
        
        h = self.input_proj(x_flat)
        h = self.transformer(h)
        out = self.output_proj(h)
        
        out = out.reshape(B, N, T, C).permute(0, 2, 1, 3)
        return out

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
