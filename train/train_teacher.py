import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import numpy as np
from models.st_cdf import ST_CDF
from utils.physics_loss import PhysicsLossFAO56
from data.real_data_loader import load_real_datasets
from data.dataloader import create_dataloaders
from data.mask_generator import MaskGenerator
from data.preprocessing import DataPreprocessor
from data.graph_builder import build_distance_matrix, build_adjacency_matrix, get_edge_index_from_adj


def train_teacher_pipeline(args):
    """
    Teacher model training pipeline with physics-informed loss.
    L = L_recons + λ_phy * L_phy
    """
    logging.info("Initializing Teacher Training Pipeline...")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load data
    try:
        data_tensor, node_names, feature_names, timestamps = load_real_datasets()
        preprocessor = DataPreprocessor()
        data_tensor = preprocessor.fit_transform(data_tensor)
        num_nodes, total_time, in_features = data_tensor.shape
        logging.info(f"Loaded data: {num_nodes} nodes, {in_features} features, {total_time} timesteps.")
    except Exception as e:
        logging.warning(f"Could not load data: {e}. Using mock data.")
        num_nodes, in_features = 10, 5
        data_tensor = np.random.rand(num_nodes, 1000, in_features).astype(np.float32)

    # Generate missing mask
    mask_gen = MaskGenerator(missing_ratio=0.1, missing_type='random')
    mask_tensor = mask_gen.generate_mask(data_tensor)

    seq_len = 24
    train_loader, val_loader, test_loader = create_dataloaders(
        data_tensor, mask_tensor, seq_len=seq_len, batch_size=args.batch_size
    )

    # Fully connected graph
    edge_index = torch.cartesian_prod(
        torch.arange(num_nodes), torch.arange(num_nodes)
    ).t().contiguous().to(device)

    # Initialize model
    teacher = ST_CDF(num_nodes=num_nodes, in_features=in_features).to(device)
    optimizer = optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    physics_loss_fn = PhysicsLossFAO56().to(device)
    l2_loss_fn = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        teacher.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            x_0, mask = batch_data
            x_0 = x_0.to(device)
            mask = mask.to(device)
            if x_0.dim() == 3:
                x_0 = x_0.unsqueeze(0)
                mask = mask.unsqueeze(0)

            B = x_0.shape[0]
            optimizer.zero_grad()

            t = torch.randint(0, teacher.diffusion.num_steps, (B,), device=device).long()
            x_t, noise = teacher.diffusion.q_sample(x_0, t)
            noise_pred = teacher(x_t, mask, t, edge_index)

            loss_recons = l2_loss_fn(noise_pred, noise)

            sqrt_alpha_bar = teacher.diffusion.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar = teacher.diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            x_0_pred = (x_t - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar
            loss_phy = physics_loss_fn(x_0_pred, x_0)

            loss = loss_recons + args.lambda_phy * loss_phy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)

        # Validation
        teacher.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                x_0, mask = batch_data
                x_0 = x_0.to(device)
                mask = mask.to(device)
                if x_0.dim() == 3:
                    x_0 = x_0.unsqueeze(0)
                    mask = mask.unsqueeze(0)
                B = x_0.shape[0]
                t = torch.randint(0, teacher.diffusion.num_steps, (B,), device=device).long()
                x_t, noise = teacher.diffusion.q_sample(x_0, t)
                noise_pred = teacher(x_t, mask, t, edge_index)
                val_loss += l2_loss_fn(noise_pred, noise).item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{args.epochs} - Train: {avg_loss:.4f} - Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(root, 'teacher_model.pth')
            torch.save(teacher.state_dict(), save_path)

    logging.info(f"Teacher training completed. Best val loss: {best_val_loss:.4f}")
