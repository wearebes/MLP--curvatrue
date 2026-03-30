#!/usr/bin/env python3
"""Quick smoke test for the refactored generate/ module."""
import sys
sys.path.insert(0, '/mnt/e/Research/PDE/code1')

print("Testing generate.pde import...")
from generate.pde import LevelSetReinitializer
print("  LevelSetReinitializer imported OK")

print("Testing generate.numerics import...")
from generate.numerics import (
    build_grid, compute_hkappa, hkappa_analytic,
    build_flower_phi0, vectorized_exact_sdf, find_projection_theta,
    ReinitQualityEvaluator,
)
print("  All numerics imported OK")

print("Testing generate.train_data import...")
from generate.train_data import LevelSetFieldBuilder, ReinitFieldPackBuilder
print("  generate.train_data imports OK")

print("Testing generate.test_data import...")
from generate.test_data import generate_test_datasets
print("  generate.test_data imports OK")

print("Testing test.evaluator import...")
from test.evaluator import evaluate_test_data, evaluate_train_data
print("  test.evaluator imports OK")

print("\n=== LevelSetReinitializer ij-mode sanity check ===")
import numpy as np
N = 33
reinforcer_ij = LevelSetReinitializer(indexing="ij", cfl=0.5, time_order=3, space_order=5)
X, Y, h = build_grid(1.0, N, indexing="ij")
phi0 = np.sqrt(X**2 + Y**2) - 0.5
phi_re = reinforcer_ij.reinitialize(phi0, h, 5)
gx = (phi_re[2:, 1:-1] - phi_re[:-2, 1:-1]) / (2*h)
gy = (phi_re[1:-1, 2:] - phi_re[1:-1, :-2]) / (2*h)
gn = np.sqrt(gx**2 + gy**2)
print(f"  grad_norm near center: mean={gn.mean():.4f} std={gn.std():.4f} (expect ~1.0)")

print("\n=== LevelSetReinitializer xy-mode sanity check ===")
reinforcer_xy = LevelSetReinitializer(indexing="xy", cfl=0.5, time_order=3, space_order=5)
X2, Y2, h2 = build_grid(1.0, N, indexing="xy")
phi0_xy = np.sqrt(X2**2 + Y2**2) - 0.5
phi_re_xy = reinforcer_xy.reinitialize(phi0_xy, h2, 5)
gx2 = (phi_re_xy[1:-1, 2:] - phi_re_xy[1:-1, :-2]) / (2*h2)
gy2 = (phi_re_xy[2:, 1:-1] - phi_re_xy[:-2, 1:-1]) / (2*h2)
gn2 = np.sqrt(gx2**2 + gy2**2)
print(f"  grad_norm near center: mean={gn2.mean():.4f} std={gn2.std():.4f} (expect ~1.0)")

print("\n=== ALL SMOKE TESTS PASSED ===")
