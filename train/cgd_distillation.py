import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
import logging


class ClusterGuidedDistillationLoss(nn.Module):
    """
    Cluster-Guided Distillation (CGD) loss.
    Uses KMeans to find semantic anchors in teacher's hidden space,
    then aligns student's distribution to teacher's via KL divergence.
    """
    def __init__(self, num_clusters=10, temperature=2.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    @torch.no_grad()
    def get_cluster_centers(self, teacher_hidden: torch.Tensor):
        """
        teacher_hidden: (B, T, N, C) or (M, C)
        Use KMeans to find semantic anchors in the hidden space.
        Returns: (K, C) cluster centers tensor on same device.
        """
        if teacher_hidden.dim() == 4:
            B, T, N, C = teacher_hidden.shape
            hidden_flat = teacher_hidden.detach().cpu().numpy().reshape(-1, C)
        elif teacher_hidden.dim() == 2:
            hidden_flat = teacher_hidden.detach().cpu().numpy()
            C = hidden_flat.shape[1]
        else:
            raise ValueError(f"Expected 2D or 4D tensor, got {teacher_hidden.dim()}D")

        # Subsample if too large
        max_samples = 10000
        if hidden_flat.shape[0] > max_samples:
            indices = np.random.choice(hidden_flat.shape[0], max_samples, replace=False)
            hidden_flat = hidden_flat[indices]

        n_clusters = min(self.num_clusters, hidden_flat.shape[0])
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(hidden_flat)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=teacher_hidden.device)
        return centers

    def forward(self, student_hidden, teacher_hidden, cluster_centers):
        """
        student_hidden: (B, T, N, C) or (M, C)
        teacher_hidden: (B, T, N, C) or (M, C)
        cluster_centers: (K, C)
        """
        if student_hidden.dim() == 4:
            B, T, N, C = student_hidden.shape
            s_flat = student_hidden.reshape(-1, C)
            t_flat = teacher_hidden.reshape(-1, C)
        else:
            s_flat = student_hidden
            t_flat = teacher_hidden

        # Compute distances to cluster centers
        dist_s = torch.cdist(s_flat, cluster_centers)  # (M, K)
        dist_t = torch.cdist(t_flat, cluster_centers)  # (M, K)

        # Convert to probability distributions
        prob_s = nn.functional.log_softmax(-dist_s / self.temperature, dim=-1)
        prob_t = nn.functional.softmax(-dist_t / self.temperature, dim=-1)

        loss_cgd = self.kl_div(prob_s, prob_t)
        return loss_cgd
