import torch
import logging
import os
import numpy as np
from models.st_cdf import ST_CDF
from data.real_data_loader import load_real_datasets
from data.preprocessing import DataPreprocessor
from data.mask_generator import MaskGenerator
from data.dataloader import create_dataloaders
from utils.metrics import compute_metrics


class AblationSTCDF(torch.nn.Module):
    """ST-CDF with optional module removal for ablation study."""
    def __init__(self, num_nodes, in_features, remove_gat=False, remove_diffattn=False, remove_wavelet=False):
        super().__init__()
        self.base_model = ST_CDF(num_nodes, in_features)

        self.remove_gat = remove_gat
        self.remove_diffattn = remove_diffattn
        self.remove_wavelet = remove_wavelet

    def forward(self, x, mask, t, edge_index):
        """Forward with ablation flags."""
        import math
        # Time embedding
        half_dim = 64
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=x.device) * -emb)
        t_emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)
        t_emb = self.base_model.time_embed_layer(t_emb).unsqueeze(1).unsqueeze(2)

        h = self.base_model.input_proj(x)
        h = h + t_emb

        if not self.remove_gat:
            h = self.base_model.spatial_extractor(h, edge_index)

        if not self.remove_diffattn:
            h = self.base_model.diff_attention(h)

        noise_pred = self.base_model.output_proj(h)
        return noise_pred


def run_ablation(args):
    """Run ablation study: remove one module at a time and evaluate."""
    logging.info("Starting Ablation Study")

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

    # Define ablation configs
    ablation_configs = {
        "Full Model": {"remove_gat": False, "remove_diffattn": False, "remove_wavelet": False},
        "w/o GAT": {"remove_gat": True, "remove_diffattn": False, "remove_wavelet": False},
        "w/o DiffAttention": {"remove_gat": False, "remove_diffattn": True, "remove_wavelet": False},
        "w/o GAT & DiffAttn": {"remove_gat": True, "remove_diffattn": True, "remove_wavelet": False},
    }

    # If specific ablation requested
    if args.remove_gat:
        logging.info("Running ablation: w/o GAT (Spatial Feature Extractor disabled)")
        ablation_configs = {"w/o GAT": ablation_configs["w/o GAT"]}

    mask_gen = MaskGenerator(missing_ratio=0.3, missing_type='random')
    seq_len = 24
    edge_index = torch.cartesian_prod(torch.arange(num_nodes), torch.arange(num_nodes)).t().contiguous().to(device)

    results = {}

    for name, config in ablation_configs.items():
        logging.info(f"\n--- Ablation: {name} ---")
        model = AblationSTCDF(num_nodes, in_features, **config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        # Quick training
        model.train()
        mask_tensor = mask_gen.generate_mask(data_tensor)
        train_loader, _, _ = create_dataloaders(data_tensor, mask_tensor, seq_len=seq_len, batch_size=16)

        for epoch in range(20):
            for batch_data in train_loader:
                x_0, mask = batch_data
                x_0 = x_0.to(device)
                mask = mask.to(device)
                if x_0.dim() == 3:
                    x_0 = x_0.unsqueeze(0)
                    mask = mask.unsqueeze(0)
                B = x_0.shape[0]
                t = torch.randint(0, 50, (B,), device=device).long()
                x_t, noise = model.base_model.diffusion.q_sample(x_0, t)
                noise_pred = model(x_t, mask, t, edge_index)
                loss = loss_fn(noise_pred, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        eval_mask = mask_gen.generate_mask(data_tensor)
        data_missing = data_tensor * eval_mask
        with torch.no_grad():
            x = torch.tensor(data_missing, dtype=torch.float32).unsqueeze(0).to(device)
            m = torch.tensor(eval_mask, dtype=torch.float32).unsqueeze(0).to(device)
            t = torch.zeros(1, device=device).long()
            pred = model(x, m, t, edge_index).squeeze(0).cpu().numpy()

        eval_mask_inv = 1 - eval_mask
        metrics = compute_metrics(data_tensor, pred, mask=eval_mask_inv)
        results[name] = metrics
        logging.info(f"  {name}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")

    logging.info("\n=== Ablation Study Summary ===")
    for name, m in results.items():
        logging.info(f"  {name}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}")

    return results
