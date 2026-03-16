"""
EPOCH ZERO — Hybrid PINN-SGP4 Satellite Collision Prediction System.

Modules:
    model      – GatedPINN v3.3 neural network architecture
    deployer   – OrbitDeployer inference wrapper (SGP4 + PINN correction)
    tle_fetcher – Real-time TLE fetching from CelesTrak GP API
    train      – Training loop for PINN model
    utils      – Physical constants and helper functions
    app        – Dash web application (3D mission control UI)
"""

__version__ = "4.0.0"
__all__ = ["model", "deployer", "tle_fetcher", "train", "utils", "app"]
