import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SensorDataset(Dataset):
    def __init__(self, data, mask, seq_len):
        """
        data: (N, T, C) - Nodes, Time, Features
        mask: (N, T, C) - 1 for observed, 0 for missing
        """
        self.data = data
        self.mask = mask
        self.seq_len = seq_len
        self.num_nodes, self.time_steps, self.num_features = data.shape
        self.num_samples = self.time_steps - seq_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[:, idx:idx+self.seq_len, :]
        m = self.mask[:, idx:idx+self.seq_len, :]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(m, dtype=torch.float32)

def create_dataloaders(data, mask, seq_len=24, batch_size=32):
    """
    Split data into Train 70%, Val 15%, Test 15% temporally.
    """
    total_time = data.shape[1]
    train_end = int(total_time * 0.7)
    val_end = int(total_time * 0.85)

    train_data = data[:, :train_end, :]
    train_mask = mask[:, :train_end, :]
    
    val_data = data[:, train_end:val_end, :]
    val_mask = mask[:, train_end:val_end, :]
    
    test_data = data[:, val_end:, :]
    test_mask = mask[:, val_end:, :]

    train_dataset = SensorDataset(train_data, train_mask, seq_len)
    val_dataset = SensorDataset(val_data, val_mask, seq_len)
    test_dataset = SensorDataset(test_data, test_mask, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
