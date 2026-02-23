"""
Real-time TLE fetcher using CelesTrak GP API (free, no auth).

Provides fresh TLEs for any NORAD catalog ID so that the PINN-corrected
SGP4 pipeline can predict *future* positions from *today's* data.
"""

import requests
import time
from datetime import datetime, timezone
from typing import Optional

# CelesTrak General Perturbations (GP) API — free, public
CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"

# Cache: { norad_id: {"tle1":..., "tle2":..., "epoch":..., "fetched_at":...} }
_tle_cache: dict = {}
CACHE_TTL_SECONDS = 3600  # re-fetch if older than 1 hour


def fetch_tle(norad_id: str | int, timeout: float = 15.0) -> Optional[dict]:
    """
    Fetch the latest TLE for a single NORAD catalog ID from CelesTrak.

    Returns
    -------
    dict  {"tle1": str, "tle2": str, "name": str, "epoch_str": str}
    or None on failure.
    """
    norad_id = str(norad_id).strip()

    # Check cache
    cached = _tle_cache.get(norad_id)
    if cached and (time.time() - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached

    try:
        resp = requests.get(
            CELESTRAK_GP_URL,
            params={"CATNR": norad_id, "FORMAT": "TLE"},
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.text.strip()

        if not text or "No GP data found" in text or "Invalid" in text:
            print(f"[TLE] No data for NORAD {norad_id}")
            return None

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return None

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
            print(f"[TLE] Invalid format for NORAD {norad_id}")
            return None

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
        return result

    except requests.RequestException as e:
        print(f"[TLE] Fetch error for NORAD {norad_id}: {e}")
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
        dt = datetime(year, 1, 1, tzinfo=timezone.utc) + __import__('datetime').timedelta(days=day_frac - 1)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown"


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
