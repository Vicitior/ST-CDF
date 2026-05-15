import numpy as np
import logging

class MaskGenerator:
    def __init__(self, missing_ratio: float, missing_type: str = 'random'):
        """
        missing_ratio: 0.0 to 1.0
        missing_type: 'random' or 'block'
        """
        self.missing_ratio = missing_ratio
        self.missing_type = missing_type

    def generate_mask(self, data: np.ndarray) -> np.ndarray:
        """
        data shape: (N, T, C)
        Returns: mask of same shape (1: observed, 0: missing)
        """
        logging.info(f"Generating {self.missing_type} mask with ratio {self.missing_ratio}")
        N, T, C = data.shape
        mask = np.ones((N, T, C), dtype=np.float32)

        if self.missing_ratio == 0:
            return mask

        if self.missing_type == 'random':
            # Pure random sampling over time and nodes
            rand_matrix = np.random.rand(N, T)
            missing_idx = rand_matrix < self.missing_ratio
            for c in range(C):
                mask[:, :, c][missing_idx] = 0.0

        elif self.missing_type == 'block':
            # Generate block missing scenarios (e.g., sensor failure for hours)
            # We randomly pick start times and mask consecutive steps
            block_length = int(T * 0.1) # Default 10% of time sequence as a block
            if block_length == 0: block_length = 1
            
            num_blocks = int((N * T * self.missing_ratio) / block_length)
            for _ in range(num_blocks):
                node = np.random.randint(0, N)
                start_t = np.random.randint(0, T - block_length)
                for c in range(C):
                    mask[node, start_t:start_t+block_length, c] = 0.0

        return mask
