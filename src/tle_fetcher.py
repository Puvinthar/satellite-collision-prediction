"""
Real-time TLE fetcher using CelesTrak GP API (free, no auth).

Provides fresh TLEs for any NORAD catalog ID so that the PINN-corrected
SGP4 pipeline can predict *future* positions from *today's* data.
"""

import requests
import aiohttp
import asyncio
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


async def fetch_tle_async(norad_id: str | int, session: aiohttp.ClientSession, timeout: float = 15.0, max_retries: int = 2) -> Optional[dict]:
    """
    Asynchronously fetch the latest TLE for a single NORAD catalog ID from CelesTrak.
    """
    norad_id = str(norad_id).strip()

    # Check cache
    cached = _tle_cache.get(norad_id)
    if cached and (time.time() - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        logger.debug(f"[TLE-Async] Cache HIT for NORAD {norad_id}")
        return cached

    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"[TLE-Async] Fetching NORAD {norad_id} (attempt {attempt + 1})")
            async with session.get(
                CELESTRAK_GP_URL,
                params={"CATNR": norad_id, "FORMAT": "TLE"},
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                text = await resp.text()
                text = text.strip()

                if not text or "No GP data found" in text or "Invalid" in text:
                    logger.warning(f"[TLE-Async] No data for NORAD {norad_id}")
                    return None

                lines = [line.strip() for line in text.splitlines() if line.strip()]
                if len(lines) < 2:
                    if attempt < max_retries:
                        await asyncio.sleep(0.5 * (2 ** attempt))
                        continue
                    break

                if len(lines) >= 3 and not lines[0].startswith("1 "):
                    name, tle1, tle2 = lines[0], lines[1], lines[2]
                else:
                    name, tle1, tle2 = f"NORAD-{norad_id}", lines[0], lines[1]

                if not tle1.startswith("1 ") or not tle2.startswith("2 "):
                    if attempt < max_retries:
                        await asyncio.sleep(0.5 * (2 ** attempt))
                        continue
                    break

                epoch_str = _tle_epoch_to_utc(tle1)
                result = {
                    "tle1": tle1,
                    "tle2": tle2,
                    "name": name,
                    "epoch_str": epoch_str,
                    "fetched_at": time.time(),
                }
                _tle_cache[norad_id] = result
                logger.info(f"[TLE-Async] Successfully fetched NORAD {norad_id}: {name}")
                return result

        except Exception as e:
            logger.warning(f"[TLE-Async] Error fetching NORAD {norad_id} (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            break

    # Fallback to sync SpaceTrack if async CelesTrak fails
    return fetch_tle_spacetrack(norad_id)


async def fetch_batch_async(norad_ids: list[str | int], timeout: float = 15.0) -> dict:
    """
    Fetch TLEs for multiple NORAD IDs asynchronously in parallel.
    """
    results = {}
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_tle_async(nid, session, timeout=timeout) for nid in norad_ids]
        responses = await asyncio.gather(*tasks)
        for nid, resp in zip(norad_ids, responses):
            results[str(nid)] = resp
    return results


def fetch_tle(norad_id: str | int, timeout: float = 15.0, max_retries: int = 2) -> Optional[dict]:
    """
    Synchronous wrapper for fetch_tle_async (using a temporary event loop if needed, 
    but for now just keeping the original sync logic for compatibility).
    """
    norad_id = str(norad_id).strip()

    # Check cache
    cached = _tle_cache.get(norad_id)
    if cached and (time.time() - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(
                CELESTRAK_GP_URL,
                params={"CATNR": norad_id, "FORMAT": "TLE"},
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.text.strip()

            if not text or "No GP data found" in text or "Invalid" in text:
                return None

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) < 2:
                if attempt < max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                break

            if len(lines) >= 3 and not lines[0].startswith("1 "):
                name, tle1, tle2 = lines[0], lines[1], lines[2]
            else:
                name, tle1, tle2 = f"NORAD-{norad_id}", lines[0], lines[1]

            if not tle1.startswith("1 ") or not tle2.startswith("2 "):
                continue

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
        except Exception:
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            break

    return fetch_tle_spacetrack(norad_id)


def fetch_batch(norad_ids: list[str | int], timeout: float = 15.0) -> dict:
    """
    Synchronous batch fetcher (now uses asyncio.run for speed).
    """
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
            
        if loop and loop.is_running():
            # In a running loop (e.g. async flask view or test), we can't use asyncio.run
            # We must use a separate thread or just warn and fall back
            # For this simple app, we'll fall back to sync in a running loop or use a trick
            logger.debug("[TLE-Batch] Event loop already running. Falling back to sync loop.")
            results = {}
            for nid in norad_ids:
                results[str(nid)] = fetch_tle(nid, timeout=timeout)
            return results
        else:
            return asyncio.run(fetch_batch_async(norad_ids, timeout))
    except Exception as e:
        logger.error(f"[TLE-Batch] Async batch failed: {e}. Using sync loop.")
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
