import torch
import torch.nn as nn
import numpy as np

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


def _dwt_torch(x, wavelet='haar'):
    """Pure PyTorch DWT along the time dimension (axis=1).
    x: (B, T, N, C) -> returns low: (B, T//2, N, C), high: (B, T//2, N, C)
    Uses a simple Haar-like convolution to stay differentiable.
    """
    B, T, N, C = x.shape
    # Pad if odd
    if T % 2 != 0:
        x = torch.cat([x, x[:, -1:, :, :]], dim=1)
        T = x.shape[1]
    # Reshape to (B*N*C, 1, T)
    x_flat = x.permute(0, 2, 3, 1).reshape(-1, 1, T)
    # Haar low-pass and high-pass filters
    lo = torch.tensor([1.0, 1.0], device=x.device, dtype=x.dtype).view(1, 1, 2) / np.sqrt(2)
    hi = torch.tensor([1.0, -1.0], device=x.device, dtype=x.dtype).view(1, 1, 2) / np.sqrt(2)
    low = torch.nn.functional.conv1d(x_flat, lo, stride=2)
    high = torch.nn.functional.conv1d(x_flat, hi, stride=2)
    T2 = low.shape[2]
    low = low.reshape(B, N, C, T2).permute(0, 3, 1, 2)
    high = high.reshape(B, N, C, T2).permute(0, 3, 1, 2)
    return low, high


def _idwt_torch(low, high, target_len):
    """Pure PyTorch IDWT (Haar) to reconstruct signal.
    low, high: (B, T2, N, C) -> returns (B, target_len, N, C)
    """
    B, T2, N, C = low.shape
    lo = torch.tensor([1.0, 1.0], device=low.device, dtype=low.dtype).view(1, 1, 2) / np.sqrt(2)
    hi = torch.tensor([1.0, -1.0], device=low.device, dtype=low.dtype).view(1, 1, 2) / np.sqrt(2)
    low_flat = low.permute(0, 2, 3, 1).reshape(-1, 1, T2)
    high_flat = high.permute(0, 2, 3, 1).reshape(-1, 1, T2)
    # Transpose convolution to upsample
    low_up = torch.nn.functional.conv_transpose1d(low_flat, lo, stride=2)
    high_up = torch.nn.functional.conv_transpose1d(high_flat, hi, stride=2)
    rec = (low_up + high_up)[:, :, :target_len]
    rec = rec.reshape(B, N, C, target_len).permute(0, 3, 1, 2)
    return rec


class WaveletTransform(nn.Module):
    def __init__(self, wavelet='haar', level=1):
        super().__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        """
        x: (B, T, N, C)
        Applies 1D DWT along the time dimension (T).
        Uses pure PyTorch ops for gradient flow; falls back to pywt if needed.
        Returns: low_freq (B, T//2, N, C), high_freq (B, T//2, N, C)
        """
        if HAS_PYWT and not x.requires_grad:
            # Use pywt for non-differentiable inference path (exact)
            device = x.device
            x_np = x.detach().cpu().numpy()
            coeffs = pywt.wavedec(x_np, self.wavelet, level=self.level, axis=1)
            low_freq = torch.tensor(coeffs[0], device=device, dtype=x.dtype)
            high_freq = torch.tensor(coeffs[1], device=device, dtype=x.dtype)
            return low_freq, high_freq
        else:
            # Differentiable path using torch convolutions
            return _dwt_torch(x, self.wavelet)


class InverseWaveletTransform(nn.Module):
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet

    def forward(self, low_freq, high_freq, target_len=None):
        """
        Reconstruct the original signal from low and high frequencies.
        low_freq, high_freq: (B, T2, N, C)
        target_len: desired output time length (default: 2 * T2)
        """
        if target_len is None:
            target_len = low_freq.shape[1] * 2

        if HAS_PYWT and not low_freq.requires_grad:
            device = low_freq.device
            low_np = low_freq.detach().cpu().numpy()
            high_np = high_freq.detach().cpu().numpy()
            coeffs = [low_np, high_np]
            reconstructed_np = pywt.waverec(coeffs, self.wavelet, axis=1)
            return torch.tensor(reconstructed_np[:, :target_len], device=device, dtype=low_freq.dtype)
        else:
            return _idwt_torch(low_freq, high_freq, target_len)
