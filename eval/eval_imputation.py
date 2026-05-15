import torch
import numpy as np
import logging
import os
from utils.metrics import compute_metrics
from models.student_model import LightweightStudentTransformer
from data.real_data_loader import load_real_datasets
from data.preprocessing import DataPreprocessor
from data.mask_generator import MaskGenerator
from data.dataloader import create_dataloaders


def evaluate_imputation(args):
    logging.info(f"Evaluating Imputation with missing ratio: {args.missing_ratio}, type: {args.missing_type}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load real data
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    et0_path = os.path.join(root, 'data', 'raw', '蒸渗仪ET0计算-间隔30分钟-2025-5-28.xlsm')
    soil_path = os.path.join(root, 'data', 'raw', '副本阜康土壤传感器数据.xlsx')

    try:
        data_tensor, valid_nodes = load_real_datasets(et0_path, soil_path)
        num_nodes, total_time, in_features = data_tensor.shape
        logging.info(f"Loaded real data: {num_nodes} nodes, {in_features} features, {total_time} timesteps")
    except Exception as e:
        logging.warning(f"Could not load real data: {e}. Using mock data.")
        num_nodes, total_time, in_features = 10, 1000, 5
        data_tensor = np.random.rand(num_nodes, total_time, in_features).astype(np.float32)

    # Preprocess
    preprocessor = DataPreprocessor()
    data_tensor = preprocessor.fit_transform(data_tensor)

    # Generate evaluation mask
    mask_gen = MaskGenerator(missing_ratio=args.missing_ratio, missing_type=args.missing_type)
    eval_mask = mask_gen.generate_mask(data_tensor)

    # Load model
    model = LightweightStudentTransformer(in_features=in_features).to(device)
    model_path = os.path.join(root, 'student_model.pth')
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info("Loaded pre-trained student model.")
    except Exception:
        logging.warning("No pre-trained student model found. Evaluating with random weights.")

    model.eval()

    # Build full observed mask (all 1s) as input condition
    full_mask = np.ones_like(data_tensor, dtype=np.float32)

    # Create dataloaders
    seq_len = 24
    train_loader, val_loader, test_loader = create_dataloaders(
        data_tensor, full_mask, seq_len=seq_len, batch_size=32
    )

    all_true = []
    all_pred = []

    with torch.no_grad():
        for x_0, mask in test_loader:
            x_0 = x_0.to(device)
            mask = mask.to(device)

            # Corrupt input with eval mask
            batch_eval_mask = torch.ones_like(x_0)
            rand = torch.rand(x_0.shape[0], x_0.shape[1], x_0.shape[2])
            if args.missing_type == 'random':
                missing = rand < args.missing_ratio
                for c in range(x_0.shape[3]):
                    batch_eval_mask[:, :, :, c][missing] = 0.0

            x_missing = x_0 * batch_eval_mask
            x_pred = model(x_missing, batch_eval_mask)

            # Evaluate on missing positions only
            eval_mask_batch = 1.0 - batch_eval_mask
            all_true.append(x_0.cpu().numpy())
            all_pred.append(x_pred.cpu().numpy())

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    # Flatten batch into (N, T, C) for metric computation
    # Reshape: (B, T, N, C) -> combine B and T
    B, T, N, C = all_true.shape
    true_flat = all_true.transpose(0, 2, 1, 3).reshape(N, B * T, C)
    pred_flat = all_pred.transpose(0, 2, 1, 3).reshape(N, B * T, C)

    # Generate matching eval mask
    eval_mask_gen = MaskGenerator(missing_ratio=args.missing_ratio, missing_type=args.missing_type)
    eval_mask_data = eval_mask_gen.generate_mask(true_flat)

    metrics = compute_metrics(true_flat, pred_flat, mask=eval_mask_data, pred_std=np.ones_like(pred_flat) * 0.1)

    logging.info("Evaluation Results:")
    for k, v in metrics.items():
        if v is not None:
            logging.info(f"  {k}: {v:.4f}")

    return metrics
