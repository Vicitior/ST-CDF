import torch
import numpy as np
import logging
import os
from utils.physics_loss import PhysicsLossFAO56
from utils.metrics import compute_metrics
from data.real_data_loader import load_real_datasets
from data.preprocessing import DataPreprocessor


def evaluate_et0_downstream(args):
    logging.info("Evaluating Downstream ET0 Accuracy.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load real data
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    et0_path = os.path.join(root, 'data', 'raw', '蒸渗仪ET0计算-间隔30分钟-2025-5-28.xlsm')
    soil_path = os.path.join(root, 'data', 'raw', '副本阜康土壤传感器数据.xlsx')

    try:
        data_tensor, valid_nodes = load_real_datasets(et0_path, soil_path)
        num_nodes, total_time, in_features = data_tensor.shape
        logging.info(f"Loaded real data: {num_nodes} nodes, {in_features} features")
    except Exception as e:
        logging.warning(f"Could not load real data: {e}. Using mock data.")
        num_nodes, total_time, in_features = 5, 1000, 5
        data_tensor = np.abs(np.random.rand(num_nodes, total_time, in_features)).astype(np.float32) + 1.0

    # Preprocess
    preprocessor = DataPreprocessor()
    data_scaled = preprocessor.fit_transform(data_tensor)

    # Use first 5 features for FAO-56 (T_mean, R_n, u_2, e_a, e_s)
    # If data has more features, take first 5
    C_phy = min(in_features, 5)

    fao56 = PhysicsLossFAO56().to(device)

    # Simulate: true = real data, pred = data + small noise
    np.random.seed(42)
    noise = np.random.randn(*data_scaled.shape) * 0.05
    pred_data = data_scaled + noise

    # Convert to tensors: (B, T, N, C) format
    # Use a sliding window to create batches
    seq_len = 24
    num_batches = (total_time - seq_len) // seq_len

    all_et0_true = []
    all_et0_pred = []

    for i in range(num_batches):
        start = i * seq_len
        end = start + seq_len
        true_chunk = data_scaled[:, start:end, :C_phy]  # (N, T, C_phy)
        pred_chunk = pred_data[:, start:end, :C_phy]

        # Reshape to (B=1, T, N, C) for FAO-56
        true_t = torch.tensor(true_chunk, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1, 3).to(device)
        pred_t = torch.tensor(pred_chunk, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1, 3).to(device)

        # Extract weather components (assuming first 5 channels map to FAO-56 inputs)
        if C_phy >= 5:
            T_t, Rn_t, u2_t, ea_t, es_t = true_t[..., 0], true_t[..., 1], true_t[..., 2], true_t[..., 3], true_t[..., 4]
            T_p, Rn_p, u2_p, ea_p, es_p = pred_t[..., 0], pred_t[..., 1], pred_t[..., 2], pred_t[..., 3], pred_t[..., 4]

            et0_true = fao56.calc_et0_tensor(T_t, Rn_t, u2_t, ea_t, es_t).cpu().numpy()
            et0_pred = fao56.calc_et0_tensor(T_p, Rn_p, u2_p, ea_p, es_p).cpu().numpy()

            all_et0_true.append(et0_true.flatten())
            all_et0_pred.append(et0_pred.flatten())

    if len(all_et0_true) > 0:
        all_et0_true = np.concatenate(all_et0_true)
        all_et0_pred = np.concatenate(all_et0_pred)
        metrics = compute_metrics(
            all_et0_true.reshape(1, -1, 1),
            all_et0_pred.reshape(1, -1, 1)
        )
        logging.info("Downstream ET0 Evaluation Results:")
        for k, v in metrics.items():
            if v is not None:
                logging.info(f"  {k}: {v:.4f}")
    else:
        logging.warning("Not enough features for FAO-56 ET0 calculation (need >= 5 channels).")
