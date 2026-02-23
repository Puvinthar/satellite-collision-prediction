import torch
import numpy as np
import pandas as pd
import os
from sgp4.api import Satrec, WGS72
try:
    from src.model import GatedPINN, R_REF, V_REF, MAX_DT
except ModuleNotFoundError:
    from model import GatedPINN, R_REF, V_REF, MAX_DT


class OrbitDeployer:
    """
    Deployment wrapper for GatedPINN v3.3.

    Uses **physical normalization** (R_REF, V_REF) — no sklearn scalers.
    """

    # Maximum reasonable orbital radius (GEO + margin) in km
    MAX_ORBITAL_RADIUS = 50_000.0   # ~GEO is ~42,164 km
    # PINN was trained with t_norm in [0, 1].  Beyond this, corrections are unreliable.
    PINN_MAX_T_NORM = 2.0

    def __init__(self, model_path='models/pinn_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GatedPINN().to(self.device)
        self.model_path = model_path
        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            state_dict = torch.load(self.model_path, map_location=self.device,
                                    weights_only=True)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict, strict=True)
            print("[+] PINN model loaded successfully (v3.3).")
        except Exception as e:
            print(f"[!] Critical Error loading model: {e}")
            raise
        self.model.eval()

    # ------------------------------------------------------------------
    @staticmethod
    def _sgp4_valid(r):
        """Return True if the SGP4 position is physically plausible."""
        return np.linalg.norm(r) < OrbitDeployer.MAX_ORBITAL_RADIUS

    # ------------------------------------------------------------------
    def _pinn_correct(self, features, t_norm_val):
        """
        Run PINN inference.  If t_norm is far outside training range,
        scale down the correction to avoid catastrophic extrapolation
        (the model's t² scaling blows up for large t).
        """
        X = torch.tensor(features, dtype=torch.float32).to(self.device)
        t = torch.tensor(t_norm_val, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            delta_r_nd, delta_v_nd = self.model(X, t)

        delta_r = delta_r_nd.cpu().numpy() * R_REF    # km
        delta_v = delta_v_nd.cpu().numpy() * V_REF     # km/s

        # Fade out corrections when extrapolating beyond training window
        abs_t = np.abs(t_norm_val if np.isscalar(t_norm_val) else t_norm_val.flatten())
        max_t = float(np.max(abs_t))
        if max_t > self.PINN_MAX_T_NORM:
            # Smooth fade: 1.0 at boundary → 0.0 well beyond
            fade = np.clip(1.0 - (abs_t - self.PINN_MAX_T_NORM) / self.PINN_MAX_T_NORM, 0.0, 1.0)
            if delta_r.ndim == 2:
                fade = fade.reshape(-1, 1)
            delta_r = delta_r * fade
            delta_v = delta_v * fade

        return delta_r, delta_v

    # ------------------------------------------------------------------
    def predict(self, line1, line2, target_epoch_str):
        """
        Returns
        -------
        initial_state : (r0, v0)  at TLE epoch
        sgp4_state    : (r, v)    at target time (baseline SGP4)
        pinn_state    : (r_corr, v_corr)  at target time (PINN-corrected)
        """
        # 1. SGP4 propagation
        sat = Satrec.twoline2rv(line1, line2, WGS72)
        jd_full = sat.jdsatepoch + sat.jdsatepochF
        ts_start = pd.Timestamp(jd_full - 2440587.5, unit='D')
        ts_target = pd.to_datetime(target_epoch_str)
        dt_minutes = (ts_target - ts_start).total_seconds() / 60.0

        e0, r0, v0 = sat.sgp4_tsince(0)
        e,  r,  v  = sat.sgp4_tsince(dt_minutes)

        if e != 0 or e0 != 0:
            return None, None, None          # SGP4 error

        r_np = np.array(r)
        v_np = np.array(v)

        # Sanity check: reject obviously non-physical SGP4 results
        if not self._sgp4_valid(r_np):
            return None, None, None

        # 2. Build normalised input (v3.3 scheme)
        r_norm = r_np / R_REF                         # [3]
        v_norm = v_np / V_REF                          # [3]
        drag   = np.array([sat.bstar * 1e4, sat.ndot]) # [2]
        t_norm = np.array([dt_minutes / MAX_DT])       # [1]

        features = np.concatenate([r_norm, v_norm, drag, t_norm]).reshape(1, -1)

        # 3. PINN inference (with extrapolation protection)
        correction_r, correction_v = self._pinn_correct(features, np.array([[t_norm[0]]]))
        correction_r = correction_r[0]
        correction_v = correction_v[0]

        # 4. Apply correction
        r_corr = r_np + correction_r
        v_corr = v_np + correction_v

        return (np.array(r0), np.array(v0)), (r_np, v_np), (r_corr, v_corr)

    # ------------------------------------------------------------------
    def get_trajectory(self, line1, line2, target_epoch_str, steps=300, window_minutes=50):
        """
        PINN-corrected trajectory for ±window_minutes around *target_epoch_str*.

        Matches the notebook's BatchConjunctionAssessor.get_trajectory():
        1.  SGP4 propagation at each timestep
        2.  Batch PINN inference for corrections along the full arc
        3.  Return corrected positions
        """
        sat = Satrec.twoline2rv(line1, line2, WGS72)
        jd_full = sat.jdsatepoch + sat.jdsatepochF
        ts_start = pd.Timestamp(jd_full - 2440587.5, unit='D')
        ts_target = pd.to_datetime(target_epoch_str)
        minutes_to_target = (ts_target - ts_start).total_seconds() / 60.0

        t_start = minutes_to_target - window_minutes
        t_end   = minutes_to_target + window_minutes
        times   = np.linspace(t_start, t_end, steps)

        # --- Step 1: SGP4 propagation at every time step ---
        r_list, v_list, dt_list = [], [], []
        for t_min in times:
            e, r, v = sat.sgp4_tsince(t_min)
            if e == 0 and self._sgp4_valid(np.array(r)):
                r_list.append(r)
                v_list.append(v)
                dt_list.append(t_min)

        if not r_list:
            return np.empty((0, 3))

        r_arr  = np.array(r_list)       # (N, 3)
        v_arr  = np.array(v_list)       # (N, 3)
        dt_arr = np.array(dt_list)      # (N,)
        N = len(r_arr)

        # --- Step 2: Batch PINN inference (with extrapolation protection) ---
        r_norm  = r_arr / R_REF                                     # (N, 3)
        v_norm  = v_arr / V_REF                                     # (N, 3)
        bstar   = np.full((N, 1), sat.bstar * 1e4)                  # (N, 1)
        ndot    = np.full((N, 1), sat.ndot)                          # (N, 1)
        t_norm  = (dt_arr / MAX_DT).reshape(-1, 1)                   # (N, 1)

        features = np.hstack([r_norm, v_norm, bstar, ndot, t_norm])  # (N, 9)

        corrections, _ = self._pinn_correct(features, t_norm)

        # --- Step 3: Corrected positions ---
        return r_arr + corrections

