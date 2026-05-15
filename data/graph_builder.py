import numpy as np
import torch
import logging

def build_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """
    coordinates: (N, 2) containing lat, lon for each sensor node
    Returns a distance matrix of shape (N, N)
    """
    N = coordinates.shape[0]
    dist_matrix = np.zeros((N, N), dtype=np.float32)
    # Simple Euclidean distance for physical location
    for i in range(N):
        for j in range(N):
            dist_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
    return dist_matrix

def build_adjacency_matrix(dist_matrix: np.ndarray, threshold: float = None) -> np.ndarray:
    """
    Convert distance matrix to adjacency matrix using a Gaussian kernel threshold.
    """
    logging.info("Building adjacency matrix for GAT.")
    N = dist_matrix.shape[0]
    if threshold is None:
        threshold = np.mean(dist_matrix) # heuristic default
    
    adj = np.zeros((N, N), dtype=np.float32)
    sigma = threshold / 2.0 if threshold > 0 else 1.0
    
    for i in range(N):
        for j in range(N):
            if dist_matrix[i, j] <= threshold:
                adj[i, j] = np.exp(-(dist_matrix[i, j]**2) / (2 * sigma**2))
    
    return adj

def get_edge_index_from_adj(adj: np.ndarray) -> torch.Tensor:
    """
    Convert dense adjacency matrix to edge_index format required by PyG (torch-geometric).
    """
    edge_indices = np.where(adj > 0)
    edge_index = torch.tensor(np.vstack((edge_indices[0], edge_indices[1])), dtype=torch.long)
    edge_weight = torch.tensor(adj[edge_indices], dtype=torch.float32)
    return edge_index, edge_weight
