"""
EPOCH ZERO — Hybrid PINN-SGP4 Satellite Collision Prediction System.

Modules:
    model      – GatedPINN v3.1.2 neural network architecture
    deployer   – OrbitDeployer inference wrapper (SGP4 + PINN correction)
    tle_fetcher – Real-time TLE fetching from CelesTrak GP API
    train      – Training loop for PINN model
    utils      – Physical constants and helper functions
"""

__version__ = "3.1.2"
__all__ = ["model", "deployer", "tle_fetcher", "train", "utils"]
