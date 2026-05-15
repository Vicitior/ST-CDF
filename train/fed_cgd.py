import torch
import torch.nn as nn
import torch.optim as optim
import copy
import logging
import os
import numpy as np
from models.student_model import LightweightStudentTransformer
from models.st_cdf import ST_CDF
from train.cgd_distillation import ClusterGuidedDistillationLoss
from data.real_data_loader import load_real_datasets
from data.preprocessing import DataPreprocessor
from data.dataloader import create_dataloaders
from data.mask_generator import MaskGenerator


def fedavg_aggregate(global_model, client_models):
    """FedAvg: average client model weights."""
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack(
            [client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


def client_update(client_model, teacher_model, data_loader, cgd_loss_fn, device, epochs=1):
    """Local client training via distillation from frozen teacher."""
    client_model.train()
    teacher_model.eval()
    optimizer = optim.Adam(client_model.parameters(), lr=1e-3)
    l2_loss = nn.MSELoss()

    for epoch in range(epochs):
        for batch_data in data_loader:
            x_0, mask = batch_data
            x_0 = x_0.to(device)
            mask = mask.to(device)
            if x_0.dim() == 3:
                x_0 = x_0.unsqueeze(0)
                mask = mask.unsqueeze(0)

            B = x_0.shape[0]
            num_nodes = x_0.shape[2]
            optimizer.zero_grad()

            s_out = client_model(x_0, mask)

            with torch.no_grad():
                edge_index = torch.cartesian_prod(
                    torch.arange(num_nodes), torch.arange(num_nodes)
                ).t().contiguous().to(device)
                t = torch.zeros(B, dtype=torch.long, device=device)
                t_out = teacher_model(x_0, mask, t, edge_index)

            loss_recons = l2_loss(s_out, t_out)
            centers = cgd_loss_fn.get_cluster_centers(t_out)
            loss_cgd = cgd_loss_fn(s_out, t_out, centers)

            loss = loss_recons + 0.5 * loss_cgd
            loss.backward()
            optimizer.step()

    return client_model


def federated_cgd_pipeline(args):
    logging.info(f"Starting Federated CGD Pipeline with {args.clients} clients...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        data_tensor, node_names, feature_names, timestamps = load_real_datasets()
        preprocessor = DataPreprocessor()
        data_tensor = preprocessor.fit_transform(data_tensor)
        num_nodes, total_time, in_features = data_tensor.shape
    except Exception as e:
        logging.warning(f"Could not load data: {e}. Using mock data.")
        num_nodes, in_features = 10, 5
        data_tensor = np.random.rand(num_nodes, 500, in_features).astype(np.float32)

    mask_gen = MaskGenerator(missing_ratio=0.1, missing_type='random')
    mask_tensor = mask_gen.generate_mask(data_tensor)

    # Load pre-trained teacher
    teacher = ST_CDF(num_nodes=num_nodes, in_features=in_features).to(device)
    teacher_path = os.path.join(root, 'teacher_model.pth')
    try:
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        logging.info("Loaded pre-trained teacher model.")
    except Exception:
        logging.warning("No pre-trained teacher found. Using initialized weights.")
    teacher.eval()

    global_student = LightweightStudentTransformer(in_features=in_features).to(device)
    logging.info(f"Student Parameters: {global_student.get_num_params() / 1e6:.2f}M")

    cgd_loss_fn = ClusterGuidedDistillationLoss().to(device)

    # Split data across clients
    total_time = data_tensor.shape[1]
    client_chunk = total_time // args.clients

    num_rounds = 10
    for r in range(num_rounds):
        logging.info(f"--- Federated Round {r + 1}/{num_rounds} ---")
        client_models = []
        for c in range(args.clients):
            start_t = c * client_chunk
            end_t = min((c + 1) * client_chunk + 24, total_time)
            c_data = data_tensor[:, start_t:end_t, :]
            c_mask = mask_tensor[:, start_t:end_t, :]
            c_train, _, _ = create_dataloaders(c_data, c_mask, seq_len=24, batch_size=16)

            local_student = copy.deepcopy(global_student)
            local_student = client_update(local_student, teacher, c_train, cgd_loss_fn, device, epochs=2)
            client_models.append(local_student)

        global_student = fedavg_aggregate(global_student, client_models)

    save_path = os.path.join(root, 'student_model.pth')
    torch.save(global_student.state_dict(), save_path)
    logging.info(f"Federated CGD completed. Student saved to {save_path}")
