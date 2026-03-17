"""
OrbitDeployer — Deployment wrapper for GatedPINN v3.1.2.

Uses sklearn scalers, F10.7 solar flux from NOAA SWPC, and SGP4 propagation
to produce PINN-corrected orbital state vectors.
"""

import torch
import numpy as np
import pandas as pd
import os
import logging
import joblib
import requests
import time
from sgp4.api import Satrec, WGS72

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    from src.model import GatedPINN, T_REF
except ModuleNotFoundError:
    from model import GatedPINN, T_REF


# =========================================================================
# F10.7 Solar Flux Fetcher (NOAA SWPC Real-Time)
# =========================================================================


def get_latest_f107():
    """
    Fetch the most recent F10.7 solar flux observation from NOAA SWPC.

    Priority:
        1. Current observation from f107_cm_flux.json
        2. Today's forecast from 45-day-forecast.json
        3. Default fallback: 150.0 sfu
    """
    # Try current observation first
    try:
        url = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and len(data) > 0:
            # First entry is the most recent observation
            val = float(data[0].get("flux", 0))
            if val > 0:
                logger.info(
                    f"[F10.7] Current observation: {val} sfu (from {data[0].get('time_tag', 'unknown')})"
                )
                return val
    except Exception as e:
        logger.warning(f"[F10.7] Failed to fetch current observation: {e}")

    # Fallback to 45-day forecast
    try:
        url = "https://services.swpc.noaa.gov/json/45-day-forecast.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Find the latest f107 forecast entry
        for entry in data.get("data", []):
            if entry.get("metric") == "f107":
                val = float(entry.get("value", 0))
                if val > 0:
                    logger.info(
                        f"[F10.7] Forecast value: {val} sfu (from {entry.get('time', 'unknown')})"
                    )
                    return val
    except Exception as e:
        logger.warning(f"[F10.7] Failed to fetch forecast: {e}")

    # Final fallback
    logger.warning("[F10.7] Using default value: 150.0 sfu")
    return 150.0


_f107_cache = {
    "data": None,
    "fetched_at": 0
}
CACHE_TTL = 3600  # 1 hour

def get_f107_for_date(target_date_str):
    """
    Get F10.7 value for a specific target date from the 45-day forecast.
    Caches the forecast JSON for 1 hour to avoid redundant requests.
    """
    global _f107_cache
    
    target_date = target_date_str[:10]  # Extract YYYY-MM-DD
    
    # Check cache
    now = time.time()
    if _f107_cache["data"] and (now - _f107_cache["fetched_at"]) < CACHE_TTL:
        # Use cached data
        for entry in _f107_cache["data"].get("data", []):
            if entry.get("metric") == "f107":
                entry_date = entry.get("time", "")[:10]
                if entry_date == target_date:
                    val = float(entry.get("value", 0))
                    if val > 0:
                        logger.debug(f"[F10.7] Cache HIT for {target_date}: {val} sfu")
                        return val
        logger.debug(f"[F10.7] Cache HIT but no forecast for {target_date}")
    else:
        # Fetch fresh data
        try:
            logger.info("[F10.7] Cache MISS/STALE, fetching fresh 45-day forecast...")
            url = "https://services.swpc.noaa.gov/json/45-day-forecast.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Update cache
            _f107_cache["data"] = data
            _f107_cache["fetched_at"] = now
            
            for entry in data.get("data", []):
                if entry.get("metric") == "f107":
                    entry_date = entry.get("time", "")[:10]
                    if entry_date == target_date:
                        val = float(entry.get("value", 0))
                        if val > 0:
                            logger.info(f"[F10.7] Fetched and matched forecast for {target_date}: {val} sfu")
                            return val
        except Exception as e:
            logger.warning(f"[F10.7] Failed to get forecast for {target_date_str}: {e}")

    return get_latest_f107()


# =========================================================================
# OrbitDeployer
# =========================================================================


class OrbitDeployer:
    """
    Deployment wrapper for GatedPINN v3.1.2.
    Uses sklearn scalers and requires F10.7 solar flux.
    """

    MAX_ORBITAL_RADIUS = 50_000.0  # ~GEO is ~42,164 km
    PINN_MAX_T_NORM = 2.0

    def __init__(
        self,
        model_path: str = None,
        scaler_x_path: str = None,
        scaler_y_path: str = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GatedPINN().to(self.device)

        # Resolve paths
        self.model_path = self._resolve_path(
            model_path,
            [
                "models/pinn_model.pth",
                "./models/pinn_model.pth",
                "notebooks_temp/orbit_error_model_pinn_v3.1.2.pth",
            ],
        )
        self.scaler_x_path = self._resolve_path(
            scaler_x_path,
            [
                "models/scaler_X.pkl",
                "./models/scaler_X.pkl",
                "notebooks_temp/scaler_X.pkl",
            ],
        )
        self.scaler_y_path = self._resolve_path(
            scaler_y_path,
            [
                "models/scaler_Y.pkl",
                "./models/scaler_Y.pkl",
                "notebooks_temp/scaler_Y.pkl",
            ],
        )

        self.f107_val = get_latest_f107()

        self._load_model_and_scalers()

    def _resolve_path(self, user_path, defaults):
        if user_path:
            p = os.path.abspath(user_path)
            if os.path.exists(p):
                return p
            raise FileNotFoundError(f"File not found: {user_path}")

        for p in defaults:
            abs_p = os.path.abspath(p)
            if os.path.exists(abs_p):
                return abs_p
            # Also try parent dir
            abs_p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", p))
            if os.path.exists(abs_p):
                return abs_p

        logger.warning(f"Could not find preferred path for {defaults[0]}")
        return None

    def _load_model_and_scalers(self):
        # Model
        if not self.model_path:
            raise FileNotFoundError("Model file not found")
        logger.info(f"[MODEL] Loading PINN from {self.model_path}")
        state_dict = torch.load(
            self.model_path, map_location=self.device, weights_only=True
        )
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        # Scalers
        if self.scaler_x_path and os.path.exists(self.scaler_x_path):
            self.scaler_X = joblib.load(self.scaler_x_path)
        else:
            logger.error("scaler_X not found.")
            self.scaler_X = None

        if self.scaler_y_path and os.path.exists(self.scaler_y_path):
            self.scaler_y = joblib.load(self.scaler_y_path)
        else:
            logger.error("scaler_Y not found.")
            self.scaler_y = None

        logger.info("[MODEL] PINN v3.1.2 loaded with scalers.")

    @staticmethod
    def _sgp4_valid(r):
        return np.linalg.norm(r) < OrbitDeployer.MAX_ORBITAL_RADIUS

    def _pinn_correct(self, features, dt_minutes_array):
        if self.scaler_X is None or self.scaler_y is None:
            return np.zeros((features.shape[0], 3)), np.zeros((features.shape[0], 3))

        X_scaled = self.scaler_X.transform(features)
        X = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        # T_gate formulation from v3.1.2: t_norm = dt_minutes / T_REF
        t_norm_val = np.asarray(dt_minutes_array, dtype=np.float64) / T_REF
        t = torch.tensor(t_norm_val, dtype=torch.float32).to(self.device).reshape(-1, 1)

        with torch.no_grad():
            pred_scaled = self.model(X, t)

        pred = self.scaler_y.inverse_transform(pred_scaled.cpu().numpy())

        delta_r = pred[:, :3]
        delta_v = pred[:, 3:]

        # Simplistic fading for extrapolation safety
        abs_t = np.abs(t_norm_val).squeeze()
        max_t = float(np.max(abs_t)) if np.ndim(abs_t) > 0 else float(abs_t)
        if max_t > self.PINN_MAX_T_NORM:
            fade = np.clip(
                1.0 - (abs_t - self.PINN_MAX_T_NORM) / self.PINN_MAX_T_NORM, 0.0, 1.0
            )
            if delta_r.ndim == 2 and np.ndim(fade) == 1:
                fade = fade.reshape(-1, 1)
            delta_r = delta_r * fade if np.ndim(fade) > 0 else delta_r * float(fade)
            delta_v = delta_v * fade if np.ndim(fade) > 0 else delta_v * float(fade)

        return delta_r, delta_v

    def predict(self, line1, line2, target_epoch_str):
        try:
            sat = Satrec.twoline2rv(line1, line2, WGS72)
            jd_full = sat.jdsatepoch + sat.jdsatepochF
            ts_start = pd.Timestamp(jd_full - 2440587.5, unit="D")
            ts_target = pd.to_datetime(target_epoch_str)
            dt_minutes = (ts_target - ts_start).total_seconds() / 60.0

            e0, r0, v0 = sat.sgp4_tsince(0)
            e, r, v = sat.sgp4_tsince(dt_minutes)

            if e != 0 or e0 != 0:
                return (None, None), (None, None), (None, None)

            r_np, v_np = np.array(r), np.array(v)
            if not self._sgp4_valid(r_np):
                return (None, None), (None, None), (None, None)

            # Get F10.7 for target date if possible
            f107 = get_f107_for_date(target_epoch_str)

            # Build feature vector: 10 features
            # ['input_rx', 'input_ry', 'input_rz', 'input_vx', 'input_vy', 'input_vz',
            #  'bstar', 'ndot', 'dt_minutes', 'f107']
            features = np.array(
                [
                    r_np[0],
                    r_np[1],
                    r_np[2],
                    v_np[0],
                    v_np[1],
                    v_np[2],
                    sat.bstar,
                    sat.ndot,
                    dt_minutes,
                    f107,
                ]
            ).reshape(1, -1)

            correction_r, correction_v = self._pinn_correct(
                features, np.array([dt_minutes])
            )

            r_corr = r_np + correction_r[0]
            v_corr = v_np + correction_v[0]

            return (np.array(r0), np.array(v0)), (r_np, v_np), (r_corr, v_corr)

        except Exception as ex:
            logger.error(f"[PREDICT] Exception: {ex}")
            return (None, None), (None, None), (None, None)

    def get_trajectory(
        self, line1, line2, target_epoch_str, steps=600, window_minutes=None
    ):
        """
        Generate a trajectory for a satellite from NOW to the target epoch.

        Propagates from current UTC time forward to the target time.
        If that window is shorter than one orbital period, the window is
        extended to cover at least one full orbit from the start time.
        Also returns the TLE epoch datetime for operator display.

        Returns:
            (sgp4_positions, pinn_positions, tle_epoch_iso)
        """
        try:
            sat = Satrec.twoline2rv(line1, line2, WGS72)
            jd_full = sat.jdsatepoch + sat.jdsatepochF
            ts_tle_epoch = pd.Timestamp(jd_full - 2440587.5, unit="D", tz="UTC")
            tle_epoch_iso = ts_tle_epoch.strftime("%Y-%m-%d %H:%M:%S UTC")

            ts_target = pd.to_datetime(target_epoch_str).tz_localize("UTC")
            ts_now = pd.Timestamp.utcnow()

            # Minutes from TLE epoch to now and to target
            minutes_tle_to_now = (ts_now - ts_tle_epoch).total_seconds() / 60.0
            minutes_tle_to_target = (ts_target - ts_tle_epoch).total_seconds() / 60.0

            # Compute orbital period from mean motion (rad/min)
            if sat.no_kozai > 0:
                orbital_period = 2.0 * np.pi / sat.no_kozai  # minutes
            else:
                orbital_period = 90.0  # fallback for LEO

            # Propagation window: from NOW to TARGET
            # If now→target is < 1 orbital period, extend to cover at least 1 orbit
            window_from_now = minutes_tle_to_target - minutes_tle_to_now
            if window_from_now < orbital_period:
                prop_window = orbital_period
            else:
                prop_window = window_from_now

            # Apply user window override if provided and larger
            if window_minutes is not None:
                prop_window = max(prop_window, window_minutes)

            logger.debug(
                f"[TRAJ] TLE epoch: {tle_epoch_iso}, orbital period: {orbital_period:.1f} min, "
                f"window: {prop_window:.1f} min (now→target: {window_from_now:.1f} min)"
            )

            # Propagate: start at NOW, end at NOW + prop_window
            t_start = minutes_tle_to_now
            t_end = minutes_tle_to_now + prop_window
            times = np.linspace(t_start, t_end, steps)

            r_list, v_list, dt_list = [], [], []
            for t_min in times:
                e, r, v = sat.sgp4_tsince(t_min)
                if e == 0 and self._sgp4_valid(np.array(r)):
                    r_list.append(r)
                    v_list.append(v)
                    dt_list.append(t_min)

            if not r_list:
                return np.empty((0, 3)), np.empty((0, 3)), tle_epoch_iso

            r_arr = np.array(r_list)
            v_arr = np.array(v_list)
            dt_arr = np.array(dt_list)
            N = len(r_arr)

            # Get F10.7 for target date
            f107 = get_f107_for_date(target_epoch_str)

            bstar_arr = np.full((N, 1), sat.bstar)
            ndot_arr = np.full((N, 1), sat.ndot)
            dt_cols = dt_arr.reshape(-1, 1)
            f107_arr = np.full((N, 1), f107)

            features = np.hstack([r_arr, v_arr, bstar_arr, ndot_arr, dt_cols, f107_arr])

            corrections, _ = self._pinn_correct(features, dt_arr)
            r_corr = r_arr + corrections

            return r_arr, r_corr, tle_epoch_iso, prop_window

        except Exception as ex:
            logger.error(f"[TRAJ] Exception: {ex}")
            return np.empty((0, 3)), np.empty((0, 3)), "", 0.0
