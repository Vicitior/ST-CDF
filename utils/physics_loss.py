import torch
import torch.nn as nn

class PhysicsLossFAO56(nn.Module):
    def __init__(self, elevation=500.0, latitude=35.0):
        super().__init__()
        self.elevation = elevation
        self.latitude = latitude
        
    def calc_et0_tensor(self, T_mean, R_n, u_2, e_a, e_s, G=0.0):
        """
        Calculate FAO-56 Penman-Monteith ET0 purely using torch operations.
        This ensures autograd can track gradients for L_phy.
        
        T_mean: Mean daily air temperature [°C]
        R_n: Net radiation at the crop surface [MJ m-2 day-1]
        u_2: Wind speed at 2 m height [m s-1]
        e_a: Actual vapour pressure [kPa]
        e_s: Saturation vapour pressure [kPa]
        G: Soil heat flux density [MJ m-2 day-1] (usually ~0 for daily)
        """
        # Psychrometric constant gamma [kPa °C-1]
        P = 101.3 * torch.pow((293.0 - 0.0065 * self.elevation) / 293.0, 5.26)
        gamma = 0.000665 * P
        
        # Slope of vapour pressure curve Delta [kPa °C-1]
        delta = (4098.0 * (0.6108 * torch.exp(17.27 * T_mean / (T_mean + 237.3)))) / torch.pow(T_mean + 237.3, 2)
        
        # FAO-56 Equation
        num1 = 0.408 * delta * (R_n - G)
        num2 = gamma * (900.0 / (T_mean + 273.0)) * u_2 * (e_s - e_a)
        den = delta + gamma * (1.0 + 0.34 * u_2)
        
        et0 = (num1 + num2) / den
        return et0

    def forward(self, pred_weather, true_weather):
        """
        pred_weather, true_weather: shape (B, T, N, C)
        Assume specific indices for weather variables.
        C_idx: 0: T_mean, 1: R_n, 2: u_2, 3: e_a, 4: e_s
        """
        # Extract components
        # In a real setup, mapping of features to indices is strictly managed.
        try:
            T_p, Rn_p, u2_p, ea_p, es_p = pred_weather[..., 0], pred_weather[..., 1], pred_weather[..., 2], pred_weather[..., 3], pred_weather[..., 4]
            T_t, Rn_t, u2_t, ea_t, es_t = true_weather[..., 0], true_weather[..., 1], true_weather[..., 2], true_weather[..., 3], true_weather[..., 4]
            
            et0_pred = self.calc_et0_tensor(T_p, Rn_p, u2_p, ea_p, es_p)
            et0_true = self.calc_et0_tensor(T_t, Rn_t, u2_t, ea_t, es_t)
            
            loss_phy = torch.mean(torch.abs(et0_pred - et0_true))
            return loss_phy
        except IndexError:
            # Fallback if features are not exactly 5 for demo purposes
            return torch.tensor(0.0, device=pred_weather.device, requires_grad=True)
