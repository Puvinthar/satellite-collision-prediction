"""
EPOCH ZERO — Flask API Server
Serves the Three.js frontend and provides API endpoints for
batch conjunction scanning and TLE fetching.
"""

import sys
import os
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify, send_from_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure src is importable
try:
    if os.path.dirname(__file__) not in sys.path:
        sys.path.insert(0, os.path.dirname(__file__))
except NameError:
    pass

from src.utils import eci_to_lla, eci_trajectory_to_lla

# --- Backend imports ---
try:
    from src.deployer import OrbitDeployer

    deployer = OrbitDeployer()
    MODEL_LOADED = True
    print("[+] PINN Model loaded successfully (v3.1.2).")
except Exception as e:
    MODEL_LOADED = False
    print(f"[!] Warning: Model not loaded — {e}")

    deployer = None
    print(
        f"[!] CRITICAL: PINN Model not loaded — {e}. Running without prediction capabilities."
    )


try:
    from src.tle_fetcher import fetch_tle

    TLE_FETCHER_AVAILABLE = True
except Exception:
    TLE_FETCHER_AVAILABLE = False

# =========================================================================
# SATELLITE DATABASE & CACHE
# =========================================================================
TLE_CACHE = {}  # Volatile cache for live TLEs
SAT_DATABASE = {
    "25544": {
        "name": "ISS (ZARYA)",
        "short": "ISS",
        "norad": "25544",
        "type": "PAYLOAD",
        "country": "ISS",
        "tle1": None,
        "tle2": None,
    },
    "48274": {
        "name": "CSS (TIANHE)",
        "short": "CSS",
        "norad": "48274",
        "type": "PAYLOAD",
        "country": "PRC",
        "tle1": None,
        "tle2": None,
    },
    "46984": {
        "name": "SENTINEL-6A",
        "short": "SENT-6A",
        "norad": "46984",
        "type": "PAYLOAD",
        "country": "EU",
        "tle1": None,
        "tle2": None,
    },
    "20580": {
        "name": "HUBBLE SPACE TELESCOPE",
        "short": "HST",
        "norad": "20580",
        "type": "PAYLOAD",
        "country": "USA",
        "tle1": None,
        "tle2": None,
    },
    "25982": {
        "name": "SPOT-4",
        "short": "SPOT-4",
        "norad": "25982",
        "type": "PAYLOAD",
        "country": "FRA",
        "tle1": None,
        "tle2": None,
    },
    "39446": {
        "name": "JASON-2",
        "short": "JASON2",
        "norad": "39446",
        "type": "PAYLOAD",
        "country": "USA",
        "tle1": None,
        "tle2": None,
    },
    "41335": {
        "name": "CRYOSAT-2",
        "short": "CRYO-2",
        "norad": "41335",
        "type": "PAYLOAD",
        "country": "EU",
        "tle1": None,
        "tle2": None,
    },
    "39084": {
        "name": "ENVISAT",
        "short": "ENVI",
        "norad": "39084",
        "type": "PAYLOAD",
        "country": "EU",
        "tle1": None,
        "tle2": None,
    },
    "28654": {
        "name": "NOAA 18",
        "short": "NOAA18",
        "norad": "28654",
        "type": "PAYLOAD",
        "country": "USA",
        "tle1": None,
        "tle2": None,
    },
    "49271": {
        "name": "FREGAT DEB",
        "short": "FRG-DEB",
        "norad": "49271",
        "type": "DEBRIS",
        "country": "CIS",
        "tle1": None,
        "tle2": None,
    },
    "22285": {
        "name": "SL-16 R/B",
        "short": "SL16-RB",
        "norad": "22285",
        "type": "DEBRIS",
        "country": "CIS",
        "tle1": None,
        "tle2": None,
    },
    "54600": {
        "name": "CZ-6A DEB",
        "short": "CZ6-DEB",
        "norad": "54600",
        "type": "DEBRIS",
        "country": "PRC",
        "tle1": None,
        "tle2": None,
    },
}

SAT_PALETTE = ["#60a5fa", "#818cf8", "#34d399", "#a78bfa", "#93c5fd", "#6ee7b7"]
DEB_PALETTE = ["#fb923c", "#f472b6", "#f87171", "#fbbf24", "#fb7185", "#fde047"]
OBJECT_COLORS = {}
_si, _di = 0, 0
for _nid, _info in SAT_DATABASE.items():
    if _info["type"] == "PAYLOAD":
        OBJECT_COLORS[_nid] = SAT_PALETTE[_si % len(SAT_PALETTE)]
        _si += 1
    else:
        OBJECT_COLORS[_nid] = DEB_PALETTE[_di % len(DEB_PALETTE)]
        _di += 1

R_EARTH = 6371

# =========================================================================
# FLASK APP
# =========================================================================
app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/info", methods=["GET"])
def get_info():
    """Return catalog metadata and system status with Model Metadata for proof."""
    catalog = []
    for nid, info in SAT_DATABASE.items():
        catalog.append(
            {
                "id": nid,
                "name": info["name"],
                "short": info["short"],
                "type": info["type"],
                "country": info["country"],
                "color": OBJECT_COLORS.get(nid, "#ffffff"),
            }
        )
    return jsonify(
        {
            "catalog": [
                {
                    "id": k,
                    "name": v["name"],
                    "short": v["short"],
                    "type": v["type"],
                    "color": OBJECT_COLORS.get(k, "#ffffff"),
                }
                for k, v in SAT_DATABASE.items()
            ],
            "defaults": ["25544", "48274", "20580", "46984", "49271", "22285"],
            "model_loaded": MODEL_LOADED,
            "model_metadata": {
                "version": "GatedPINN v3.1.2",
                "weights": "models/pinn_model.pth",
                "R_REF": 6378.137,
                "V_REF": 7.905,
                "normalization": "StandardScaler (10 features, with Space Weather F10.7)",
            },
        }
    )


@app.route("/api/scan", methods=["POST"])
def api_scan():
    """Run batch conjunction scan with improved error handling."""
    data = request.json
    selected_ids = data.get("ids", [])
    target_str = data.get("target", "")
    prop_window = data.get("prop_window", 100)

    if not MODEL_LOADED:
        logger.error("PINN Model is offline")
        return jsonify(
            {"error": "PINN Model is offline. Server cannot run scans without PINN."}
        ), 500

    if len(selected_ids) < 2:
        logger.warning(f"Insufficient objects for scan: {len(selected_ids)}")
        return jsonify({"error": "Select at least 2 objects."}), 400

    # Define consistent start time for the scan
    start_dt = datetime.now(timezone.utc)

    # Parse target date with better error handling
    try:
        if target_str:
            try:
                target_dt = datetime.strptime(target_str, "%Y-%m-%d %H:%M:%S")
                target_dt = target_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                logger.warning(
                    f"Failed to parse target date '{target_str}', using current UTC"
                )
                target_dt = start_dt
        else:
            target_dt = start_dt
    except Exception as e:
        logger.error(f"Error parsing target date: {e}")
        target_dt = start_dt

    # Calculate required window to cover from NOW to TARGET
    delta_min = (target_dt - start_dt).total_seconds() / 60.0
    # Use max(user_prop_window, delta_min)
    effective_window = max(prop_window, delta_min)

    # Dynamic steps calculation (prevent trajectory threading)
    # Target ~1 point per degree of orbit (e.g., 360 points per 90-min orbit)
    # Min 600, Max 6000
    dynamic_steps = int(max(600, min(6000, (effective_window / 90.0) * 360)))

    logger.info(
        f"Scan started: {len(selected_ids)} objects, target={target_dt.strftime('%Y-%m-%d %H:%M:%S')}, window={effective_window:.1f}m (now->target: {delta_min:.1f}m)"
    )

    objects = {}
    errors = []
    for obj_id in selected_ids:
        try:
            sat = SAT_DATABASE.get(obj_id)
            if not sat:
                logger.warning(f"Satellite {obj_id} not in database")
                continue

            # Prioritize TLEs from volatile cache, then from SAT_DATABASE
            tle_data = TLE_CACHE.get(obj_id)
            if not tle_data:
                tle_data = sat

            tle1, tle2 = tle_data.get("tle1"), tle_data.get("tle2")
            if not tle1 or not tle2:
                # Try to fetch live TLE if missing
                if TLE_FETCHER_AVAILABLE:
                    try:
                        from src.tle_fetcher import fetch_tle

                        result = fetch_tle(obj_id)
                        if result and result.get("tle1") and result.get("tle2"):
                            tle1, tle2 = result["tle1"], result["tle2"]
                            (
                                SAT_DATABASE[obj_id]["tle1"],
                                SAT_DATABASE[obj_id]["tle2"],
                            ) = tle1, tle2
                            logger.info(
                                f"Auto-fetched TLE for {sat['name']} [NORAD {obj_id}]"
                            )
                        else:
                            logger.warning(f"Skip {sat['name']}: No TLE available")
                            continue
                    except Exception as e:
                        logger.warning(
                            f"Skip {sat['name']}: TLE fetch failed ({str(e)})"
                        )
                        continue
                else:
                    logger.warning(f"Skip {sat['name']}: No TLE data")
                    continue

            try:
                # 2. Get trajectories (Dual: SGP4 vs PINN)
                traj_sgp4, traj_pinn, tle_epoch_str, actual_prop_window = (
                    deployer.get_trajectory(
                        tle1,
                        tle2,
                        target_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        steps=dynamic_steps,
                        window_minutes=effective_window,
                    )
                )

                # Update global prop_window to the maximum used by any object
                prop_window = max(prop_window, actual_prop_window)

                if traj_pinn.size == 0:
                    msg = f"{sat['name']}: PINN trajectory empty"
                    logger.warning(msg)
                    errors.append(msg)
                    continue

                # Convert ECI trajectories to LLA (still in km for ECI)
                lla_pinn = eci_trajectory_to_lla(
                    traj_pinn, target_dt
                )  # Trajectories in km
                lla_sgp4 = eci_trajectory_to_lla(traj_sgp4, target_dt)

                logger.debug(
                    f"  {sat['name']}: trajectory computed ({len(traj_pinn)} points)"
                )

                # Single point prediction for table/markers
                res = deployer.predict(
                    tle1, tle2, target_dt.strftime("%Y-%m-%d %H:%M:%S")
                )
                if res[0][0] is None:
                    msg = f"{sat['name']}: Prediction failed (SGP4/PINN error)"
                    logger.warning(msg)
                    errors.append(msg)
                    continue

                (r0, v0), (r_sgp4, v_sgp4), (r_pinn, v_pinn) = res
                pos_lla = eci_to_lla(r_pinn[0], r_pinn[1], r_pinn[2], target_dt)

                # Compute prediction confidence based on correction magnitude
                dr = float(np.linalg.norm(r_pinn - r_sgp4))
                orbital_radius = float(np.linalg.norm(r_pinn))
                correction_ratio = dr / orbital_radius if orbital_radius > 0 else 1.0

                # Confidence: high when corrections are small relative to orbit
                # ratio < 0.001 → HIGH, < 0.01 → MEDIUM, else LOW
                if correction_ratio < 0.001:
                    confidence = min(1.0, 1.0 - correction_ratio * 500)
                    confidence_label = "HIGH"
                elif correction_ratio < 0.01:
                    confidence = max(0.3, 0.8 - correction_ratio * 50)
                    confidence_label = "MEDIUM"
                else:
                    confidence = max(0.1, 0.3 - correction_ratio * 5)
                    confidence_label = "LOW"

                objects[obj_id] = {
                    "name": sat["name"],
                    "short": sat["short"],
                    "type": sat["type"],
                    "color": OBJECT_COLORS.get(obj_id, "#ffffff"),
                    "trajectory_pinn_eci": (
                        traj_pinn / R_EARTH
                    ).tolist(),  # Normalized ECI for Three.js
                    "trajectory_sgp4_eci": (
                        traj_sgp4 / R_EARTH
                    ).tolist(),  # Normalized ECI for Three.js
                    "trajectory_pinn_lla": {k: v.tolist() for k, v in lla_pinn.items()},
                    "trajectory_sgp4_lla": {k: v.tolist() for k, v in lla_sgp4.items()},
                    "pos_pinn": r_pinn.tolist(),
                    "pos_lla": {
                        "lat": float(pos_lla[0]),
                        "lon": float(pos_lla[1]),
                        "alt": float(pos_lla[2]),
                    },
                    "dr": dr,
                    "altitude": float(np.linalg.norm(r_pinn) - R_EARTH),
                    "speed": float(np.linalg.norm(v_pinn)),
                    "confidence": round(confidence, 3),
                    "confidence_label": confidence_label,
                    "tle_epoch": tle_epoch_str,
                }
                logger.info(
                    f"  {sat['name']}: OK (alt={objects[obj_id]['altitude']:.1f}km, dr={dr:.3f}km, confidence={confidence_label})"
                )

            except Exception as ex:
                msg = f"{sat['name']}: {str(ex)[:80]}"
                logger.error(msg)
                errors.append(msg)

        except Exception as ex:
            logger.error(f"Error processing {obj_id}: {str(ex)[:80]}")
            errors.append(f"{obj_id}: {str(ex)[:60]}")

    if len(objects) < 2:
        logger.warning(
            f"Insufficient objects propagated: {len(objects)} < 2, errors: {errors}"
        )
        return jsonify(
            {"error": "Less than 2 objects propagated.", "details": errors}
        ), 400

    # Build collision pairs
    ids = list(objects.keys())
    n = len(ids)
    pairs = []

    # Calculate step size in minutes (estimated from prop_window and trajectory points)
    # get_trajectory auto-computes from orbital period; estimate step from prop_window as fallback
    first_traj = next(iter(objects.values()), {}).get("trajectory_pinn_eci", [])
    n_traj_points = len(first_traj) if first_traj else 600
    step_minutes = prop_window / n_traj_points
    logger.info(f"Computing collision pairs for {n} objects ({n * (n - 1) // 2} pairs)")

    for i in range(n):
        for j in range(i + 1, n):
            try:
                # Use the ECI PINN trajectories for distance calculation (in km!)
                # Denormalize from display coordinates back to km
                traj_a_eci = (
                    np.array(objects[ids[i]]["trajectory_pinn_eci"]) * R_EARTH
                )  # back to km
                traj_b_eci = (
                    np.array(objects[ids[j]]["trajectory_pinn_eci"]) * R_EARTH
                )  # back to km

                # Calculate distance across all 200 frames
                dists = np.linalg.norm(traj_a_eci - traj_b_eci, axis=1)
                min_idx = np.argmin(dists)
                dist = float(dists[min_idx])

                # Predict TCA Time
                tca_offset = min_idx * step_minutes
                tca_dt = target_dt + timedelta(minutes=tca_offset)
                tca_str = (
                    tca_dt.strftime("%Y-%m-%d %H:%M:%S") + f" (+{int(tca_offset):d}m)"
                )

                threat = "LOW"
                if dist < 100:
                    threat = "CRITICAL"
                elif dist < 500:
                    threat = "HIGH"
                elif dist < 2000:
                    threat = "WARNING"

                pairs.append(
                    {
                        "a": ids[i],
                        "b": ids[j],
                        "name_a": objects[ids[i]]["short"],
                        "name_b": objects[ids[j]]["short"],
                        "miss_dist": round(dist, 1),
                        "threat": threat,
                        "tca": tca_str,
                        "min_idx": int(min_idx),
                    }
                )
                logger.debug(
                    f"  {objects[ids[i]]['name']} <-> {objects[ids[j]]['name']}: {dist:.1f} km ({threat})"
                )

            except Exception as ex:
                logger.error(
                    f"Error computing collision pair {ids[i]}-{ids[j]}: {str(ex)[:80]}"
                )

    pairs.sort(key=lambda p: p["miss_dist"])

    n_critical = sum(1 for p in pairs if p["threat"] in ("CRITICAL", "HIGH"))
    min_miss = pairs[0]["miss_dist"] if pairs else 9999

    logger.info(
        f"Scan complete: {n} objects, {len(pairs)} pairs, {n_critical} critical, closest={min_miss:.1f}km"
    )

    # Determine actual number of trajectory frames from first object
    first_obj_val = next(iter(objects.values()), {})
    actual_frames = (
        len(first_obj_val.get("trajectory_pinn_eci", [])) if first_obj_val else 600
    )

    # Start epoch = start_dt (propagation starts from consistent now)
    return jsonify(
        {
            "objects": objects,
            "pairs": pairs[:20],
            "n_objects": n,
            "n_pairs": len(pairs),
            "n_critical": n_critical,
            "min_miss": min_miss,
            "closest_pair": f"{pairs[0]['name_a']} ↔ {pairs[0]['name_b']}"
            if pairs
            else "---",
            "errors": errors,
            "total_frames": actual_frames,
            "start_epoch": start_dt.isoformat(),
            "target_epoch": target_dt.isoformat(),
            "prop_window": prop_window,
        }
    )


@app.route("/api/fetch-tles", methods=["POST"])
def api_fetch_tles():
    """Fetch live TLEs from CelesTrak with detailed logging."""
    if not TLE_FETCHER_AVAILABLE:
        logger.error("TLE Fetcher not available")
        return jsonify({"error": "TLE Fetcher not available."}), 500

    data = request.json
    ids_to_fetch = data.get("ids", list(SAT_DATABASE.keys()))
    logger.info(f"Fetching TLEs for {len(ids_to_fetch)} satellites")

    n_ok, n_fail = 0, 0
    for nid in ids_to_fetch:
        try:
            result = fetch_tle(nid)
            if result:
                # Update SAT_DATABASE for existing ones
                if str(nid) in SAT_DATABASE:
                    SAT_DATABASE[str(nid)]["tle1"] = result["tle1"]
                    SAT_DATABASE[str(nid)]["tle2"] = result["tle2"]
                    logger.info(
                        f"✓ Updated {SAT_DATABASE[str(nid)]['name']} from CelesTrak"
                    )
                # Always store in cache for immediate use
                TLE_CACHE[str(nid)] = result
                n_ok += 1
            else:
                logger.warning(f"✗ Failed to fetch NORAD {nid}")
                n_fail += 1
        except Exception as ex:
            logger.error(f"Exception fetching NORAD {nid}: {str(ex)}")
            n_fail += 1

    logger.info(f"TLE fetch complete: {n_ok} OK, {n_fail} failed")
    return jsonify(
        {"ok": n_ok, "fail": n_fail, "source": f"LIVE ({n_ok}/{n_ok + n_fail})"}
    )


@app.route("/api/add-sat", methods=["POST"])
def add_sat():
    """Live proof: Fetch any NORAD ID from CelesTrak and add it to tracking pool."""
    norad_id = request.json.get("id")
    if not norad_id:
        logger.warning("add_sat called with no ID")
        return jsonify({"error": "No ID provided"}), 400

    if not TLE_FETCHER_AVAILABLE:
        logger.error("TLE Fetcher not available for add_sat")
        return jsonify({"error": "TLE Fetcher not available."}), 500

    logger.info(f"Adding satellite NORAD {norad_id} to tracking")
    result = fetch_tle(norad_id)
    if not result:
        logger.warning(f"Satellite NORAD {norad_id} not found on CelesTrak")
        return jsonify({"error": f"Satellite {norad_id} not found on CelesTrak"}), 404

    # Add to local volatile database
    SAT_DATABASE[str(norad_id)] = {
        "name": result["name"],
        "short": result["name"][:10],
        "norad": str(norad_id),
        "type": "PAYLOAD",  # Default to payload for custom track
        "country": "UNK",
        "tle1": result["tle1"],
        "tle2": result["tle2"],
    }

    # Store the TLE for immediate use in scan
    TLE_CACHE[str(norad_id)] = result

    logger.info(f"✓ Added satellite: {result['name']}")
    return jsonify(
        {
            "success": True,
            "name": result["name"],
            "id": norad_id,
            "msg": f"Target {result['name']} locked and loaded for active tracking.",
        }
    )


if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print("  EPOCH ZERO — Fleet Surveillance System (Three.js UI)")
    print("  Open http://localhost:3000")
    print(f"{'=' * 60}\n")
    app.run(debug=True, port=5000, host="0.0.0.0", use_reloader=True)
