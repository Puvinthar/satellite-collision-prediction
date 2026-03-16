import torch
import numpy as np
import pandas as pd
import os
import logging
from sgp4.api import Satrec, WGS72

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GatedPINN().to(self.device)
        
        # Resolve model path - support both absolute and relative paths
        if model_path is None:
            # Try multiple common locations
            possible_paths = [
                'models/pinn_model.pth',
                './models/pinn_model.pth',
                os.path.join(os.path.dirname(__file__), '..', 'models', 'pinn_model.pth'),
            ]
            self.model_path = None
            for p in possible_paths:
                abs_p = os.path.abspath(p)
                if os.path.exists(abs_p):
                    self.model_path = abs_p
                    break
            if not self.model_path:
                raise FileNotFoundError(f"PINN model not found in any of {possible_paths}")
        else:
            self.model_path = os.path.abspath(model_path)
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"PINN model not found at {self.model_path}")
        
        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            logger.info(f"[MODEL] Loading PINN from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device,
                                    weights_only=True)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict, strict=True)
            logger.info("[MODEL] PINN model loaded successfully (v3.3).")
            print("[+] PINN model loaded successfully (v3.3).")
        except FileNotFoundError as e:
            logger.critical(f"[MODEL] Model file not found: {e}")
            raise
        except Exception as e:
            logger.critical(f"[MODEL] Error loading model: {e}")
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
        Run PINN inference with proper batch handling.  If t_norm is far outside 
        training range, scale down the correction to avoid catastrophic extrapolation.
        
        Parameters
        ----------
        features : np.ndarray
            Shape (B, 9) - batch of normalized features
        t_norm_val : np.ndarray or float
            Normalized time value(s) - can be scalar or shape (B, 1) or (B,)
            
        Returns
        -------
        delta_r : np.ndarray
            Shape (B, 3) - position corrections in km
        delta_v : np.ndarray
            Shape (B, 3) - velocity corrections in km/s
        """
        X = torch.tensor(features, dtype=torch.float32).to(self.device)
        t = torch.tensor(t_norm_val, dtype=torch.float32).to(self.device)
        
        # Ensure t is at least 2D [B, 1]
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(0)  # scalar to [1, 1]
        elif t.dim() == 1:
            t = t.unsqueeze(-1)  # [B] to [B, 1]

        with torch.no_grad():
            delta_r_nd, delta_v_nd = self.model(X, t)

        delta_r = delta_r_nd.cpu().numpy() * R_REF    # km
        delta_v = delta_v_nd.cpu().numpy() * V_REF     # km/s

        # Fade out corrections when extrapolating beyond training window
        t_numpy = t.cpu().numpy()
        if t_numpy.ndim == 2:
            t_numpy = t_numpy.squeeze()
        abs_t = np.abs(t_numpy)
        max_t = float(np.max(abs_t))
        
        if max_t > self.PINN_MAX_T_NORM:
            logger.debug(f"[PINN] Extrapolation detected: t_norm={max_t:.3f} > {self.PINN_MAX_T_NORM}. Fading corrections.")
            # Smooth fade: 1.0 at boundary → 0.0 well beyond
            fade = np.clip(1.0 - (abs_t - self.PINN_MAX_T_NORM) / self.PINN_MAX_T_NORM, 0.0, 1.0)
            # Reshape fade to match delta_r shape for broadcasting
            if delta_r.ndim == 2 and fade.ndim == 1:
                fade = fade.reshape(-1, 1)  # (B,) to (B, 1)
            delta_r = delta_r * fade if fade.ndim > 0 else delta_r * fade.item()
            delta_v = delta_v * fade if fade.ndim > 0 else delta_v * fade.item()

        return delta_r, delta_v

    # ------------------------------------------------------------------
    def predict(self, line1, line2, target_epoch_str):
        """
        Predict satellite position at target epoch using SGP4 + PINN correction.
        
        Parameters
        ----------
        line1, line2 : str
            TLE lines
        target_epoch_str : str
            Target time as 'YYYY-MM-DD HH:MM:SS'
            
        Returns
        -------
        initial_state : (r0, v0)
            Position/velocity at TLE epoch, or (None, None) if SGP4 error
        sgp4_state : (r, v)
            Baseline SGP4 prediction at target time
        pinn_state : (r_corr, v_corr)
            PINN-corrected state at target time
        """
        try:
            # 1. SGP4 propagation
            sat = Satrec.twoline2rv(line1, line2, WGS72)
            jd_full = sat.jdsatepoch + sat.jdsatepochF
            ts_start = pd.Timestamp(jd_full - 2440587.5, unit='D')
            ts_target = pd.to_datetime(target_epoch_str)
            dt_minutes = (ts_target - ts_start).total_seconds() / 60.0
            
            logger.debug(f"[PREDICT] Target: {target_epoch_str}, dt_minutes: {dt_minutes:.1f}")

            e0, r0, v0 = sat.sgp4_tsince(0)
            e,  r,  v  = sat.sgp4_tsince(dt_minutes)

            if e != 0 or e0 != 0:
                logger.warning(f"[PREDICT] SGP4 error: e0={e0}, e={e}")
                return (None, None), (None, None), (None, None)

            r_np = np.array(r)
            v_np = np.array(v)

            # Sanity check: reject obviously non-physical SGP4 results
            if not self._sgp4_valid(r_np):
                logger.warning(f"[PREDICT] Invalid SGP4 result: r_norm={np.linalg.norm(r_np):.1f} km exceeds max {self.MAX_ORBITAL_RADIUS} km")
                return (None, None), (None, None), (None, None)

            # 2. Build normalised input (v3.3 scheme)
            r_norm = r_np / R_REF                                  # [3]
            v_norm = v_np / V_REF                                  # [3]
            drag   = np.array([sat.bstar * 1e4, sat.ndot])         # [2]
            t_norm = np.array([dt_minutes / MAX_DT])               # [1]

            features = np.concatenate([r_norm, v_norm, drag, t_norm]).reshape(1, -1)  # [1, 9]

            # 3. PINN inference (with extrapolation protection)
            correction_r, correction_v = self._pinn_correct(features, t_norm.reshape(1, 1))
            correction_r = correction_r[0]  # Extract first (only) batch element
            correction_v = correction_v[0]

            # 4. Apply correction
            r_corr = r_np + correction_r
            v_corr = v_np + correction_v
            
            logger.debug(f"[PREDICT] Successful: |correction_r|={np.linalg.norm(correction_r):.3f} km")

            return (np.array(r0), np.array(v0)), (r_np, v_np), (r_corr, v_corr)
            
        except Exception as ex:
            logger.error(f"[PREDICT] Exception: {str(ex)[:100]}")
            return (None, None), (None, None), (None, None)

    # ------------------------------------------------------------------
    def get_trajectory(self, line1, line2, target_epoch_str, steps=300, window_minutes=50):
        """
        PINN-corrected trajectory for ±window_minutes around *target_epoch_str*.

        Matches the notebook's BatchConjunctionAssessor.get_trajectory():
        1.  SGP4 propagation at each timestep
        2.  Batch PINN inference for corrections along the full arc
        3.  Return corrected positions
        
        Parameters
        ----------
        line1, line2 : str
            TLE lines
        target_epoch_str : str
            Target time as 'YYYY-MM-DD HH:MM:SS'
        steps : int
            Number of trajectory points
        window_minutes : int
            Total window (±minutes around target)
            
        Returns
        -------
        r_arr : np.ndarray
            Shape (N, 3) - SGP4 baseline positions in km
        r_corr : np.ndarray  
            Shape (N, 3) - PINN-corrected positions in km
        """
        try:
            sat = Satrec.twoline2rv(line1, line2, WGS72)
            jd_full = sat.jdsatepoch + sat.jdsatepochF
            ts_start = pd.Timestamp(jd_full - 2440587.5, unit='D')
            ts_target = pd.to_datetime(target_epoch_str)
            minutes_to_target = (ts_target - ts_start).total_seconds() / 60.0

            t_start = minutes_to_target - window_minutes / 2.0
            t_end   = minutes_to_target + window_minutes / 2.0
            times   = np.linspace(t_start, t_end, steps)
            
            logger.debug(f"[TRAJ] Computing {steps} points from t={t_start:.1f} to t={t_end:.1f} minutes")

            # --- Step 1: SGP4 propagation at every time step ---
            r_list, v_list, dt_list = [], [], []
            for t_min in times:
                e, r, v = sat.sgp4_tsince(t_min)
                if e == 0 and self._sgp4_valid(np.array(r)):
                    r_list.append(r)
                    v_list.append(v)
                    dt_list.append(t_min)

            if not r_list:
                logger.error(f"[TRAJ] No valid SGP4 points for {len(times)} attempts")
                return np.empty((0, 3)), np.empty((0, 3))
            
            logger.debug(f"[TRAJ] Got {len(r_list)}/{len(times)} valid SGP4 points")

            r_arr  = np.array(r_list)       # (N, 3)
            v_arr  = np.array(v_list)       # (N, 3)
            dt_arr = np.array(dt_list)      # (N,)
            N = len(r_arr)

            # --- Step 2: Batch PINN inference (with extrapolation protection) ---
            r_norm  = r_arr / R_REF                                     # (N, 3)
            v_norm  = v_arr / V_REF                                     # (N, 3)
            bstar   = np.full((N, 1), sat.bstar * 1e4)                  # (N, 1)
            ndot    = np.full((N, 1), sat.ndot)                         # (N, 1)
            t_norm  = (dt_arr / MAX_DT).reshape(-1, 1)                  # (N, 1)

            features = np.hstack([r_norm, v_norm, bstar, ndot, t_norm])  # (N, 9)
            
            logger.debug(f"[TRAJ] Running batch PINN inference on {N} points")
            corrections, _ = self._pinn_correct(features, t_norm)
            
            logger.debug(f"[TRAJ] Batch correction complete: avg correction |dr|={np.mean(np.linalg.norm(corrections, axis=1)):.3f} km")

            # --- Step 3: Return BOTH trajectories ---
            r_corr = r_arr + corrections
            return r_arr, r_corr
            
        except Exception as ex:
            logger.error(f"[TRAJ] Exception: {str(ex)[:100]}")
            return np.empty((0, 3)), np.empty((0, 3))

