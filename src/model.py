import torch
import torch.nn as nn

__all__ = ["GatedPINN", "R_REF", "V_REF", "T_REF"]

# Physical constants for normalization (matching notebook v3.1.2)
MU_EARTH = 398600.4418  # km^3/s^2 (Gravitational parameter)
R_EARTH = 6378.137  # km (Earth equatorial radius)
J2 = 1.08263e-3  # J2 perturbation coefficient
OMEGA_E = 7.2921159e-5  # rad/s (Earth rotation rate)

# Normalization references (matching notebook v3.1.2)
R_REF = 6378.137  # km (Earth Radius)
V_REF = 7.905  # km/s (Orbital Velocity)
T_REF = 1440.0  # minutes (24h normalization)
ACC_REF = V_REF / (T_REF * 60.0)  # km/s^2


class GatedPINN(nn.Module):
    """
    ResidualPINN v3.1.2 — Physics-Informed Residual Correction for SGP4 with Space Weather.

    Architecture: 10 → 128 → 256 → 128 → 6 with tanh activations.
    Applies tanh(5t) gate so correction vanishes at t=0.

    Input features (10):
        [input_rx, input_ry, input_rz, input_vx, input_vy, input_vz,
         bstar, ndot, dt_minutes, f107]

    Output (6):
        [delta_rx, delta_ry, delta_rz, delta_vx, delta_vy, delta_vz]
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 6),
        )

    def forward(self, x, t=None):
        if t is None:
            t = x[:, -1:]
        out = self.net(x)
        gate = torch.tanh(5.0 * t)
        return out * gate
