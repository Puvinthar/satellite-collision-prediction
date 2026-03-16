from datetime import datetime, timezone
import numpy as np
import time

from src.deployer import OrbitDeployer
from server import SAT_DATABASE, TLE_CACHE

print("Loading deployer...")
deployer = OrbitDeployer(model_path="models/pinn_model.pth")
tle1 = SAT_DATABASE["25544"]["tle1"]
tle2 = SAT_DATABASE["25544"]["tle2"]
start_dt = datetime.now(timezone.utc)
target_dt = start_dt

print("Computing trajectory...")
start_time = time.time()
traj_sgp4, traj_pinn, tle_epoch_str, act_win = deployer.get_trajectory(
    tle1, tle2, target_dt.strftime("%Y-%m-%d %H:%M:%S"), steps=3600, window_minutes=900
)
end_time = time.time()

print(f"Prop window: {act_win}")
print(f"Points generated: {len(traj_pinn)}")
print(f"Time taken: {end_time - start_time:.2f} seconds")
