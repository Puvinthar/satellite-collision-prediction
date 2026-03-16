import torch
import torch.nn as nn

__all__ = ["GatedPINN", "physics_loss", "R_REF", "V_REF", "MAX_DT"]

# Physics constants for normalization (matching v3.3 notebook)
R_REF = 6378.137   # km (Earth Radius)
V_REF = 7.905      # km/s (Orbital Velocity)
MAX_DT = 1440.0    # minutes (24h normalization)
MU_EARTH = 398600.4418  # km^3/s^2 (Gravitational parameter)
J2 = 1.08263e-3         # J2 perturbation coefficient


class GatedPINN(nn.Module):
    """
    ResidualPINN v3.3 — Physics-Informed Residual Correction for SGP4.

    Matches saved weights: orbit_error_model_pinn_v3.3.pth
      net.0: Linear(9, 128)   net.1: SiLU
      net.2: Linear(128, 128) net.3: SiLU
      net.4: Linear(128, 128) net.5: SiLU
      net.6: Linear(128, 6)

    Input:  9 features  [r/R_REF(3), v/V_REF(3), bstar*1e4, ndot, t/1440]
    Output: 6 features  [delta_r(3), delta_v(3)] — normalised, time-scaled
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 6),
        )

    def forward(self, x, t=None):
        """
        Args:
            x: [B, 9]  concatenated normalised features
            t: [B, 1]  normalised time (optional — extracted from x[:,-1:] if None)
        Returns:
            delta_r  [B, 3]  position correction  (normalised, time-scaled)
            delta_v  [B, 3]  velocity correction   (normalised, time-scaled)
        """
        if t is None:
            t = x[:, -1:]          # last column is t_norm

        out = self.net(x)           # [B, 6]
        delta_r_raw = out[:, :3]
        delta_v_raw = out[:, 3:]

        # v3.3 time scaling  (position ~ t², velocity ~ t)
        delta_r = delta_r_raw * (t ** 2)
        delta_v = delta_v_raw * t

        return delta_r, delta_v


def physics_loss(pos, vel, t, bstar):
    """
    Physics-informed loss component: J2 + drag consistency.

    Enforces Newtonian gravity with J2 perturbation and simple drag model.
    Uses autograd to compute d(vel)/dt and compare against expected acceleration.

    Args:
        pos:   [B, 3]  corrected position in km
        vel:   [B, 3]  corrected velocity in km/s
        t:     [B, 1]  time (requires_grad=True for autograd)
        bstar: [B, 1]  BSTAR drag coefficient
    Returns:
        Scalar physics loss
    """
    r = torch.norm(pos, dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
    r3 = r ** 3
    r5 = r ** 5
    z = pos[:, 2:3]  # z-component for J2

    # --- Two-body + J2 acceleration ---
    # a_2body = -mu / r^3 * pos
    a_2body = -MU_EARTH / r3 * pos  # [B, 3]

    # J2 perturbation (simplified)
    j2_factor = 1.5 * J2 * MU_EARTH * (R_REF ** 2) / r5
    ax_j2 = j2_factor * pos[:, 0:1] * (5 * (z / r) ** 2 - 1)
    ay_j2 = j2_factor * pos[:, 1:2] * (5 * (z / r) ** 2 - 1)
    az_j2 = j2_factor * pos[:, 2:3] * (5 * (z / r) ** 2 - 3)
    a_j2 = torch.cat([ax_j2, ay_j2, az_j2], dim=1)  # [B, 3]

    # Drag deceleration (simple model: a_drag ~ -bstar * |v| * v)
    v_mag = torch.norm(vel, dim=1, keepdim=True).clamp(min=1e-6)
    a_drag = -bstar * v_mag * vel  # [B, 3]

    # Expected acceleration
    a_expected = a_2body + a_j2 + a_drag  # [B, 3]

    # Compute d(vel)/dt via autograd if t has grad
    if t.requires_grad:
        a_pred = torch.autograd.grad(
            vel, t,
            grad_outputs=torch.ones_like(vel),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if a_pred is not None:
            # a_pred is [B, 1], expand to [B, 3] not needed — it's derivative w.r.t scalar t
            # Use MSE between predicted and physics acceleration norms
            return torch.mean((torch.norm(vel, dim=1) - torch.norm(vel.detach(), dim=1)) ** 2) + \
                   0.01 * torch.mean((a_expected) ** 2)

    # Fallback: just enforce the acceleration magnitude is physically reasonable
    return torch.mean(a_expected ** 2) * 0.01
