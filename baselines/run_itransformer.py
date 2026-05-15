import torch
import torch.nn as nn
import numpy as np
import logging
import os
from utils.metrics import compute_metrics
from data.real_data_loader import load_real_datasets
from data.preprocessing import DataPreprocessor
from data.mask_generator import MaskGenerator


class iTransformerBaseline(nn.Module):
    """
    iTransformer: Inverted Transformer for time-series forecasting/imputation.
    Applies attention across variables (features) instead of time steps.
    Reference: https://github.com/thuml/iTransformer
    """
    def __init__(self, in_features, seq_len, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.in_features = in_features
        self.seq_len = seq_len

        self.input_proj = nn.Linear(seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, seq_len)

    def forward(self, x, mask=None):
        """
        x: (B, T, N, C) -> reshape to (B*N, C, T) -> attend across C
        """
        B, T, N, C = x.shape
        # Reshape: (B*N, C, T)
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, C, T)

        # If mask provided, zero out missing positions
        if mask is not None:
            m_flat = mask.permute(0, 2, 1, 3).reshape(B * N, C, T)
            x_flat = x_flat * m_flat

        h = self.input_proj(x_flat)  # (B*N, C, hidden)
        h = self.transformer(h)  # (B*N, C, hidden)
        out = self.output_proj(h)  # (B*N, C, T)

        # Reshape back to (B, T, N, C)
        out = out.reshape(B, N, C, T).permute(0, 3, 1, 2)
        return out


def run_baseline(args):
    logging.info("Running iTransformer baseline")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load data
    et0_path = os.path.join(root, 'data', 'raw', '蒸渗仪ET0计算-间隔30分钟-2025-5-28.xlsm')
    soil_path = os.path.join(root, 'data', 'raw', '副本阜康土壤传感器数据.xlsx')

    try:
        data_tensor, valid_nodes = load_real_datasets(et0_path, soil_path)
        preprocessor = DataPreprocessor()
        data_tensor = preprocessor.fit_transform(data_tensor)
        num_nodes, total_time, in_features = data_tensor.shape
    except Exception as e:
        logging.warning(f"Could not load real data: {e}. Using synthetic data.")
        num_nodes, total_time, in_features = 10, 500, 5
        data_tensor = np.random.rand(num_nodes, total_time, in_features).astype(np.float32)

    seq_len = 24
    missing_ratio = 0.3
    if hasattr(args, 'missing_ratio'):
        missing_ratio = args.missing_ratio

    # Create model
    model = iTransformerBaseline(in_features, seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Generate masks
    mask_gen = MaskGenerator(missing_ratio=missing_ratio, missing_type='random')

    # Simple training loop
    model.train()
    num_epochs = 50
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, total_time - seq_len, seq_len):
            chunk = data_tensor[:, start:start + seq_len, :]
            mask = mask_gen.generate_mask(chunk)
            data_missing = chunk * mask

            x = torch.tensor(data_missing, dtype=torch.float32).unsqueeze(0).to(device)
            m = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
            target = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)

            optimizer.zero_grad()
            pred = model(x, m)
            loss = loss_fn(pred * (1 - m), target * (1 - m))  # Only on missing
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0:
            logging.info(f"iTransformer Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss / max(num_batches, 1):.4f}")

    # Evaluation
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for start in range(0, total_time - seq_len, seq_len):
            chunk = data_tensor[:, start:start + seq_len, :]
            mask = mask_gen.generate_mask(chunk)
            data_missing = chunk * mask

            x = torch.tensor(data_missing, dtype=torch.float32).unsqueeze(0).to(device)
            m = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(x, m)
            all_true.append(chunk)
            all_pred.append(pred.squeeze(0).cpu().numpy())

    all_true = np.concatenate(all_true, axis=1)
    all_pred = np.concatenate(all_pred, axis=1)
    eval_mask = mask_gen.generate_mask(all_true)
    eval_mask = 1 - eval_mask

    metrics = compute_metrics(all_true, all_pred, mask=eval_mask)
    logging.info("iTransformer Baseline Results:")
    for k, v in metrics.items():
        if v is not None:
            logging.info(f"  {k}: {v:.4f}")

    return metrics
