import torch
import numpy as np
import logging
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.student_model import LightweightStudentTransformer
from data.real_data_loader import load_real_datasets
from data.preprocessing import DataPreprocessor


def run_counterfactual_simulation(args):
    """
    Simulates counterfactual scenarios, e.g., "What if we irrigate Node A for N hours?"
    Modifies soil moisture for the target node and observes spatio-temporal propagation.
    """
    logging.info(f"Running Counterfactual Simulation on Node {args.target_node} for {args.simulate_duration} hours.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Try to load real data for realistic simulation
    et0_path = os.path.join(root, 'data', 'raw', '蒸渗仪ET0计算-间隔30分钟-2025-5-28.xlsm')
    soil_path = os.path.join(root, 'data', 'raw', '副本阜康土壤传感器数据.xlsx')

    try:
        data_tensor, valid_nodes = load_real_datasets(et0_path, soil_path)
        preprocessor = DataPreprocessor()
        data_tensor = preprocessor.fit_transform(data_tensor)
        num_nodes, total_time, in_features = data_tensor.shape
        logging.info(f"Using real data: {num_nodes} nodes, {in_features} features")
    except Exception as e:
        logging.warning(f"Could not load real data: {e}. Using synthetic data.")
        num_nodes, in_features = 10, 5
        data_tensor = np.random.rand(num_nodes, 48, in_features).astype(np.float32)

    # Load model
    model = LightweightStudentTransformer(in_features=in_features).to(device)
    model_path = os.path.join(root, 'student_model.pth')
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info("Loaded pre-trained student model.")
    except Exception:
        logging.warning("No pre-trained model found. Using random weights.")

    model.eval()

    # Take a window of data
    seq_len = min(24, data_tensor.shape[1])
    x_real = torch.tensor(data_tensor[:, :seq_len, :], dtype=torch.float32)
    # Reshape to (B=1, T, N, C)
    x = x_real.unsqueeze(0).to(device)  # (1, T, N, C)

    B, T, N, C = x.shape
    target = min(args.target_node, N - 1)
    sim_duration = min(args.simulate_duration, T - 1)
    sim_start = T // 3  # Start simulation 1/3 into the sequence
    sim_end = min(sim_start + sim_duration, T)

    # Create counterfactual: increase soil moisture (channel 0) at target node
    x_cf = x.clone()
    original_val = x_cf[0, sim_start, target, 0].item()
    x_cf[0, sim_start:sim_end, target, 0] = 0.9  # Simulate irrigation to high moisture

    # Mask other nodes' moisture AFTER the intervention to observe propagation
    mask = torch.ones_like(x_cf).to(device)
    for n in range(N):
        if n != target:
            mask[0, sim_start:, n, 0] = 0.0

    with torch.no_grad():
        x_pred = model(x_cf, mask)

    # Report results
    logging.info(f"Counterfactual: Irrigated Node {target} from t={sim_start} to t={sim_end}")
    logging.info(f"  Original moisture at Node {target}, t={sim_start}: {original_val:.4f}")
    logging.info(f"  Set moisture to: 0.9000")

    # Check propagation to neighboring nodes
    for n in range(min(N, 5)):
        if n != target:
            pred_val = x_pred[0, sim_end - 1, n, 0].item()
            logging.info(f"  Predicted moisture at Node {n}, t={sim_end - 1}: {pred_val:.4f}")

    # Plot results
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot target node moisture over time
        timesteps = list(range(T))
        orig_moisture = x[0, :, target, 0].cpu().numpy()
        cf_moisture = x_cf[0, :, target, 0].cpu().numpy()

        axes[0].plot(timesteps, orig_moisture, 'b-', label='Original', linewidth=2)
        axes[0].plot(timesteps, cf_moisture, 'r--', label='Counterfactual (Irrigated)', linewidth=2)
        axes[0].axvspan(sim_start, sim_end - 1, alpha=0.2, color='orange', label='Irrigation Period')
        axes[0].set_ylabel('Soil Moisture (scaled)')
        axes[0].set_title(f'Counterfactual Simulation: Node {target} Irrigation')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot propagation to other nodes
        for n in range(min(N, 4)):
            if n != target:
                pred_moisture = x_pred[0, :, n, 0].cpu().numpy()
                axes[1].plot(timesteps, pred_moisture, label=f'Node {n} (imputed)')

        axes[1].axvspan(sim_start, sim_end - 1, alpha=0.2, color='orange')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Soil Moisture (scaled)')
        axes[1].set_title('Propagation to Other Nodes')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(root, 'counterfactual_simulation.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Counterfactual simulation plot saved to {save_path}")
    except Exception as e:
        logging.warning(f"Could not save plot: {e}")
