# ST-CDF: A Conditional Diffusion Framework for High-Fidelity Spatio-Temporal Sensor Data Imputation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4.0-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the **core model implementation** of **ST-CDF**, a Spatio-Temporal Conditional Diffusion Framework for high-fidelity sensor data imputation in agricultural applications.

> **Note**: This release includes the core model architecture and training pipelines. Data preprocessing utilities and baseline comparisons will be released upon paper acceptance.

## Key Features

1. **Physical Constraints ($\mathcal{L}_{phy}$)**: FAO-56 Penman-Monteith equation implemented purely in PyTorch tensors for gradient-aware physics-informed training.
2. **Cluster-Guided Distillation (CGD)** & **Federated Learning**: Distills a heavy Teacher model into a lightweight (3.8M) Student model for edge deployment via Fed-CGD.
3. **Multi-head Differential Attention & Wavelet Transform**: High-frequency decoupling with differentiable DWT/IDWT and spatio-temporal differential attention with FlashAttention support.
4. **Counterfactual Simulation**: Evaluate theoretical agricultural scenarios (e.g., virtual irrigation effects on neighboring sensors).

## Repository Structure

```
ST-CDF-Repo/
├── main.py                      # Entry point
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
│
├── data/
│   ├── dataloader.py            # PyTorch Dataset/DataLoader
│   ├── mask_generator.py        # Random & Block mask generation
│   ├── preprocessing.py         # Anomaly detection + normalization
│   └── graph_builder.py         # Graph construction for GAT
│
├── models/
│   ├── st_cdf.py                # Teacher: GAT + DiffAttention + Diffusion
│   ├── gat_layer.py             # Graph Attention Network (torch-geometric)
│   ├── diff_attention.py        # Multi-head Differential Attention
│   ├── wavelet_transform.py     # Differentiable DWT/IDWT
│   ├── diffusion_process.py     # DDPM forward/reverse process
│   └── student_model.py         # Lightweight Student Transformer (3.8M)
│
├── train/
│   ├── train_teacher.py         # Teacher training with L_phy
│   ├── cgd_distillation.py      # Cluster-Guided Distillation loss
│   └── fed_cgd.py               # Federated CGD pipeline (FedAvg)
│
├── eval/
│   ├── eval_imputation.py       # Imputation quality evaluation
│   ├── eval_et0_downstream.py   # Downstream ET0 accuracy
│   └── counterfactual_sim.py    # Counterfactual simulation
│
└── utils/
    ├── metrics.py               # MAE, RMSE, R², CRPS
    └── physics_loss.py          # FAO-56 Penman-Monteith (torch tensors)
```

## Environment Setup

```bash
conda create -n stcdf python=3.10 -y
conda activate stcdf
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Usage

```bash
# Teacher model training
python main.py --mode train_teacher --epochs 500 --lambda_phy 0.1

# Student distillation (Fed-CGD)
python main.py --mode train_student_fed --clients 5

# Evaluation
python main.py --mode eval --missing_ratio 0.5 --missing_type random

# Counterfactual simulation
python main.py --mode counterfactual --target_node 2 --simulate_duration 3
```

## Core Dependencies

| Category | Packages |
|----------|----------|
| Deep Learning | `torch 2.4.0`, `torch-geometric 2.6.0`, `flash-attn 2.6.3` |
| Data Processing | `numpy`, `pandas`, `scikit-learn`, `scipy`, `pywavelets` |
| Metrics | `properscoring` (CRPS) |
| Visualization | `matplotlib`, `seaborn`, `tensorboard` |

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{stcdf2025,
  title={ST-CDF: A Conditional Diffusion Framework for High-Fidelity Spatio-Temporal Sensor Data Imputation},
  journal={Applied Sciences},
  year={2025}
}
```
