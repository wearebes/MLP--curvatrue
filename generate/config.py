from pathlib import Path


# config.py — Global hyperparameters for the level-set curvature dataset pipeline.

# Target grid resolutions (rho)
TARGET_RHOS = [256, 266, 276]

# RNG seed for CircleGeometryGenerator
GEOMETRY_SEED = 42

# Number of center-position variations per radius
VARIATIONS = 5

# Reinitialization pseudo-time CFL coefficient
# Δτ = CFL * h  (0.1 = conservative; original paper used 0.5)
CFL = 0.2

# WENO5 smoothness-indicator epsilon
EPS_WENO = 1e-6

# Smoothed sign function: eps = EPS_SIGN_FACTOR * h
EPS_SIGN_FACTOR = 1.0

# Reinit step counts for non-SDF blueprints
REINIT_STEPS = [5, 10, 15, 20]

# Project root and output directory for generated HDF5 datasets
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data"
