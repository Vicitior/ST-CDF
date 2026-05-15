import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import logging

class DataPreprocessor:
    def __init__(self, contamination=0.01):
        self.scaler = MinMaxScaler()
        self.iso_forest = IsolationForest(contamination=contamination, random_state=42)
        
    def apply_3sigma(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply 3-sigma rule to remove extreme outliers.
        Replaces outliers with NaN.
        """
        logging.info("Applying 3-sigma rule for anomaly detection.")
        clean_df = df.copy()
        for col in clean_df.select_dtypes(include=[np.number]).columns:
            mean = clean_df[col].mean()
            std = clean_df[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            outlier_mask = (clean_df[col] < lower_bound) | (clean_df[col] > upper_bound)
            clean_df.loc[outlier_mask, col] = np.nan
        return clean_df
        
    def apply_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Isolation Forest for multivariate anomaly detection.
        """
        logging.info("Applying Isolation Forest.")
        # Drop rows with NaN for training IF (or handle appropriately)
        numeric_df = df.select_dtypes(include=[np.number])
        valid_mask = numeric_df.notna().all(axis=1)
        
        if valid_mask.sum() > 0:
            preds = self.iso_forest.fit_predict(numeric_df[valid_mask])
            # preds == -1 are anomalies
            anomaly_indices = numeric_df[valid_mask].index[preds == -1]
            df.loc[anomaly_indices, numeric_df.columns] = np.nan
        return df
        
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Scale data between 0 and 1. Flatten spatial-temporal before scaling.
        data shape: (N, T, C)
        """
        logging.info("Applying MinMaxScaler.")
        N, T, C = data.shape
        reshaped = data.reshape(-1, C)
        scaled = self.scaler.fit_transform(reshaped)
        return scaled.reshape(N, T, C)
        
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        N, T, C = scaled_data.shape
        reshaped = scaled_data.reshape(-1, C)
        inv = self.scaler.inverse_transform(reshaped)
        return inv.reshape(N, T, C)
