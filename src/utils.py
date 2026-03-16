import torch
import numpy as np
import random
from datetime import datetime, timezone


def set_seed(seed=42):
    """
    Set seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Physical constants (matching notebook v3.1.2)
MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137  # km (equatorial radius)
J2 = 1.08263e-3  # J2 perturbation coefficient
OMEGA_E = 7.2921159e-5  # rad/s (Earth rotation rate)

# Normalization references
R_REF = 6378.137  # km
V_REF = 7.905  # km/s
T_REF = 1440.0  # minutes (24h)
ACC_REF = V_REF / (T_REF * 60.0)  # km/s^2

R_EARTH_KM = 6371.0  # km (mean radius, used for LLA)


def eci_to_lla(x, y, z, dt_utc=None):
    """
    Convert ECI (TEME) coordinates (km) to geodetic lat/lon/alt.

    Parameters
    ----------
    x, y, z : float or np.ndarray
        ECI position components in km
    dt_utc : datetime, optional
        UTC time for GMST calculation. Defaults to now.

    Returns
    -------
    lat, lon, alt : float or np.ndarray
        Geodetic latitude (deg), longitude (deg), altitude (km)
    """
    if dt_utc is None:
        dt_utc = datetime.now(timezone.utc)

    # Greenwich Mean Sidereal Time (simplified)
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    d = (dt_utc - j2000).total_seconds() / 86400.0
    gmst_deg = (280.46061837 + 360.98564736629 * d) % 360.0
    gmst_rad = np.radians(gmst_deg)

    # Rotate ECI → ECEF
    cos_g = np.cos(gmst_rad)
    sin_g = np.sin(gmst_rad)
    x_ecef = cos_g * x + sin_g * y
    y_ecef = -sin_g * x + cos_g * y
    z_ecef = z

    # Geodetic conversion (simple spherical approximation)
    r_ground = np.sqrt(x_ecef**2 + y_ecef**2)
    lat = np.degrees(np.arctan2(z_ecef, r_ground))
    lon = np.degrees(np.arctan2(y_ecef, x_ecef))
    alt = np.sqrt(x_ecef**2 + y_ecef**2 + z_ecef**2) - R_EARTH_KM

    return lat, lon, alt


def eci_trajectory_to_lla(trajectory, dt_utc=None):
    """
    Convert an array of ECI positions (N, 3) to lat/lon/alt arrays.

    Returns dict with keys: lat, lon, alt (each np.ndarray of length N)
    """
    traj = np.asarray(trajectory)
    if traj.ndim != 2 or traj.shape[1] < 3:
        return {"lat": np.array([]), "lon": np.array([]), "alt": np.array([])}

    lat, lon, alt = eci_to_lla(traj[:, 0], traj[:, 1], traj[:, 2], dt_utc)
    return {"lat": lat, "lon": lon, "alt": alt}
