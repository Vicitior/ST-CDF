import numpy as np
import properscoring as ps
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(true, pred, mask=None, pred_std=None):
    """
    true: ground truth (N, T, C)
    pred: predictions (N, T, C)
    mask: boolean or 0/1 array where 1 means evaluate (e.g. missing data points)
    pred_std: standard deviation of predictions for CRPS
    """
    if mask is not None:
        mask_bool = mask == 1
        true = true[mask_bool]
        pred = pred[mask_bool]
        if pred_std is not None:
            pred_std = pred_std[mask_bool]

    true = true.flatten()
    pred = pred.flatten()

    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    
    crps = None
    if pred_std is not None:
        pred_std = pred_std.flatten()
        # compute CRPS for Gaussian distribution
        crps = np.mean(ps.crps_gaussian(true, mu=pred, sig=pred_std))
        
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CRPS": crps
    }
