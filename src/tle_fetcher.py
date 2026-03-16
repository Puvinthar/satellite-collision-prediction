"""
Real-time TLE fetcher using CelesTrak GP API (free, no auth).

Provides fresh TLEs for any NORAD catalog ID so that the PINN-corrected
SGP4 pipeline can predict *future* positions from *today's* data.
"""

import requests
import time
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

try:
    from spacetrack import SpaceTrackClient
except ImportError:
    SpaceTrackClient = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# CelesTrak General Perturbations (GP) API — free, public
CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"

# Cache: { norad_id: {"tle1":..., "tle2":..., "epoch":..., "fetched_at":...} }
_tle_cache: dict = {}
CACHE_TTL_SECONDS = 3600  # re-fetch if older than 1 hour


def fetch_tle(norad_id: str | int, timeout: float = 15.0, max_retries: int = 2) -> Optional[dict]:
    """
    Fetch the latest TLE for a single NORAD catalog ID from CelesTrak.

    Parameters
    ----------
    norad_id : str or int
        NORAD catalog ID
    timeout : float
        Request timeout in seconds (default: 15.0)
    max_retries : int
        Number of retry attempts on failure (default: 2)

    Returns
    -------
    dict
        {"tle1": str, "tle2": str, "name": str, "epoch_str": str}
        or None on failure.
    """
    norad_id = str(norad_id).strip()

    # Check cache
    cached = _tle_cache.get(norad_id)
    if cached and (time.time() - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        logger.debug(f"[TLE] Cache HIT for NORAD {norad_id} (age: {time.time() - cached['fetched_at']:.1f}s)")
        return cached
    
    if cached:
        logger.debug(f"[TLE] Cache STALE for NORAD {norad_id} (age: {time.time() - cached['fetched_at']:.1f}s, TTL: {CACHE_TTL_SECONDS}s)")

    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"[TLE] Fetching NORAD {norad_id} (attempt {attempt + 1}/{max_retries + 1})")
            resp = requests.get(
                CELESTRAK_GP_URL,
                params={"CATNR": norad_id, "FORMAT": "TLE"},
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.text.strip()

            if not text or "No GP data found" in text or "Invalid" in text:
                logger.warning(f"[TLE] No data for NORAD {norad_id} from CelesTrak")
                return None

            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if len(lines) < 2:
                logger.warning(f"[TLE] Incomplete response for NORAD {norad_id} (only {len(lines)} lines)")
                if attempt < max_retries:
                    time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                    continue
                break

            # CelesTrak returns: Name (line 0), TLE Line 1, TLE Line 2
            if len(lines) >= 3 and not lines[0].startswith("1 "):
                name = lines[0]
                tle1 = lines[1]
                tle2 = lines[2]
            else:
                name = f"NORAD-{norad_id}"
                tle1 = lines[0]
                tle2 = lines[1]

            # Validate basic TLE format
            if not tle1.startswith("1 ") or not tle2.startswith("2 "):
                logger.warning(f"[TLE] Invalid TLE format for NORAD {norad_id}")
                if attempt < max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                break

            # Extract epoch from TLE line 1  (cols 18-32: YYDDD.DDDDDDDD)
            epoch_str = _tle_epoch_to_utc(tle1)

            result = {
                "tle1": tle1,
                "tle2": tle2,
                "name": name,
                "epoch_str": epoch_str,
                "fetched_at": time.time(),
            }
            _tle_cache[norad_id] = result
            logger.info(f"[TLE] Successfully fetched NORAD {norad_id}: {name} (epoch: {epoch_str})")
            return result

        except requests.Timeout:
            logger.warning(f"[TLE] Timeout fetching NORAD {norad_id} (attempt {attempt + 1}/{max_retries + 1})")
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            break
        except requests.ConnectionError as e:
            logger.warning(f"[TLE] Connection error for NORAD {norad_id}: {e} (attempt {attempt + 1}/{max_retries + 1})")
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            break
        except requests.RequestException as e:
            logger.error(f"[TLE] Request error for NORAD {norad_id}: {e}")
            break

    # If we reached here, CelesTrak failed. Try fallback.
    logger.warning(f"[TLE] CelesTrak failed for NORAD {norad_id}. Attempting SpaceTrack fallback...")
    fallback_res = fetch_tle_spacetrack(norad_id)
    if fallback_res:
        _tle_cache[norad_id] = fallback_res
        return fallback_res
        
    return None


def fetch_batch(norad_ids: list[str | int], timeout: float = 15.0) -> dict:
    """
    Fetch TLEs for multiple NORAD IDs. Returns {norad_id: tle_dict or None}.
    """
    results = {}
    for nid in norad_ids:
        results[str(nid)] = fetch_tle(nid, timeout=timeout)
    return results


def _tle_epoch_to_utc(tle1: str) -> str:
    """Convert TLE epoch field (YYDDD.DDDDDDDD) to 'YYYY-MM-DD HH:MM:SS' string."""
    try:
        epoch_field = tle1[18:32].strip()
        yy = int(epoch_field[:2])
        year = 2000 + yy if yy < 57 else 1900 + yy
        day_frac = float(epoch_field[2:])
        dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_frac - 1)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown"


def fetch_tle_spacetrack(norad_id: str | int) -> Optional[dict]:
    """Fallback fetcher using SpaceTrack API."""
    norad_id = str(norad_id).strip()
    username = os.environ.get("SPACETRACK_USERNAME")
    password = os.environ.get("SPACETRACK_PASSWORD")
    
    if not username or not password or SpaceTrackClient is None:
        logger.warning(f"[TLE-Fallback] Missing credentials or spacetrack library for {norad_id}")
        return None
        
    try:
        logger.debug(f"[TLE-Fallback] Attempting SpaceTrack API for NORAD {norad_id}")
        st = SpaceTrackClient(identity=username, password=password)
        data = st.gp(
            norad_cat_id=norad_id,
            orderby='EPOCH desc',
            limit=1,
            format='json'
        )
        if not data or len(data) == 0:
            logger.warning(f"[TLE-Fallback] No data found for NORAD {norad_id}")
            return None
            
        sat_data = data[0]
        name = sat_data.get('OBJECT_NAME', f"NORAD-{norad_id}")
        tle1 = sat_data.get('TLE_LINE1')
        tle2 = sat_data.get('TLE_LINE2')
        
        if not tle1 or not tle2:
            return None
            
        epoch_str = _tle_epoch_to_utc(tle1)
        
        result = {
            "tle1": tle1,
            "tle2": tle2,
            "name": name,
            "epoch_str": epoch_str,
            "fetched_at": time.time()
        }
        logger.info(f"[TLE-Fallback] Successfully fetched NORAD {norad_id} from SpaceTrack: {name}")
        return result
    except Exception as e:
        logger.error(f"[TLE-Fallback] Error fetching NORAD {norad_id} from SpaceTrack: {e}")
        return None


def is_tle_fresh(tle_dict: dict, max_age_days: float = 7.0) -> bool:
    """Check if a TLE epoch is within max_age_days of now."""
    try:
        epoch = datetime.strptime(tle_dict["epoch_str"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - epoch).total_seconds() / 86400.0
        return age <= max_age_days
    except Exception:
        return False


def clear_cache():
    """Clear the TLE cache."""
    _tle_cache.clear()
