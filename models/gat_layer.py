import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class SpatioTemporalGAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super(SpatioTemporalGAT, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=False)
        
    def forward(self, x, edge_index):
        """
        x: (Batch * Time * Nodes, Features) or (Nodes, Features) depending on reshaping.
        For batched spatio-temporal data, usually we reshape to process graphs per time step or 
        treat time as a feature. Here we assume x is reshaped to (Nodes, Features) per batch/time.
        """
        # In a real batched setup, you'd use torch_geometric.data.Batch
        # For simplicity in this module, we assume x is (N, C)
        return self.gat(x, edge_index)

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim):
        super().__init__()
        self.gat = SpatioTemporalGAT(in_features, hidden_dim, heads=4)
        
    def forward(self, x, edge_index):
        """
        x: (B, T, N, C)
        We process the spatial graph at each time step.
        """
        B, T, N, C = x.shape
        x_reshaped = x.reshape(B * T * N, C)
        
        # Adjust edge_index for batched graphs
        # This is a naive loop-based or block-diagonal approach. 
        # A more efficient way is to use PyG Batch.
        # Assuming edge_index is for a single graph (N nodes)
        
        # To avoid complex PyG batching overhead here, we use a loop for exactness,
        # but in production, `torch_geometric.data.Batch` is used.
        out = torch.zeros(B, T, N, self.gat.gat.out_channels, device=x.device)
        for b in range(B):
            for t in range(T):
                out[b, t] = self.gat(x[b, t], edge_index)
                
        return out
