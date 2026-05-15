import numpy as np
import logging


def load_real_datasets(et0_path: str = None, soil_path: str = None, use_daily_et0: bool = False):
    """
    Load and align ET0 weather data with soil sensor data.
    Returns (N, T, C) tensor where N=nodes, T=timesteps, C=features.

    Data format:
        - ET0: 30-min interval meteorological data (wind, temp, RH, radiation, rain)
        - Soil: 5 treatment nodes × 5 depths (10-16cm, 30-36cm, 50-56cm, 70-76cm, 90-96cm)
        - Combined: C = 5 soil depths + 5 weather features = 10 features
    """
    logging.info("Loading dataset...")

    # Try loading processed data first
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npz_path = os.path.join(root, 'data', 'processed', 'dataset.npz')

    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        data_tensor = data['data']
        import json
        meta_path = os.path.join(root, 'data', 'processed', 'metadata.json')
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        node_names = meta['node_names']
        feature_names = meta['feature_names']
        logging.info(f"Loaded processed dataset: {data_tensor.shape}")
        return data_tensor, node_names, feature_names, None

    # Fallback: generate synthetic demo data
    logging.warning("No processed dataset found. Using synthetic demo data.")
    logging.info("Run 'python data/build_dataset.py' to process real data.")

    np.random.seed(42)
    num_nodes, total_time, in_features = 5, 2000, 10
    data_tensor = np.random.rand(num_nodes, total_time, in_features).astype(np.float32) * 0.8 + 0.1

    node_names = ['稀植1', '稀植2传感器OK', '密植3玉黄金', '密植4苄氨乙烯利', '密植5胺鲜酯乙烯利']
    feature_names = ['soil_10-16cm', 'soil_30-36cm', 'soil_50-56cm', 'soil_70-76cm', 'soil_90-96cm',
                     'wind_speed', 'temperature', 'RH', 'solar_radiation', 'rainfall']

    return data_tensor, node_names, feature_names, None
