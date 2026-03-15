import numpy as np
from typing import List, Dict

class CircleGeometryGenerator:
    def __init__(self, resolution_rho: int, seed: int = 42, variations: int = 5):
        self.rho = int(resolution_rho)
        self.global_seed = int(seed)
        self.variations = int(variations)

        if self.rho not in (256, 266, 276):
            print(f"[Warning] rho={self.rho} is not one of the paper's listed training resolutions (256, 266, 276).")
        if self.variations < 1:
            raise ValueError("variations must be >= 1")
        if self.variations > 5:
            print(f"[Warning] variations={self.variations} exceeds the paper's 'up to 5' setting.")

        self.h = 1.0 / (self.rho - 1)
        self.r_min = 1.6 * self.h
        self.r_max = 0.5 - 2.0 * self.h
        self.num_radii = int(np.floor((self.rho - 8.2) / 2.0)) + 1

        if self.num_radii < 1 or self.r_min >= self.r_max:
            raise ValueError(f"[Error] Resolution rho={self.rho} is too small to generate valid circular interfaces. Required: num_radii >= 1 and r_min < r_max.")

        self.radii_set = np.linspace(self.r_min, self.r_max, self.num_radii, dtype=float)

        if not np.isclose(self.radii_set[0], self.r_min, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"First radius mismatch: {self.radii_set[0]} vs r_min={self.r_min}")
        if not np.isclose(self.radii_set[-1], self.r_max, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"Last radius mismatch: {self.radii_set[-1]} vs r_max={self.r_max}")

        self.center_min = 0.5 - self.h / 2.0
        self.center_max = 0.5 + self.h / 2.0

    def _subseed(self, r_idx: int, v_idx: int) -> int:
        # Use explicit 64-bit wrapping in Python ints so the intended hash-style
        # overflow does not emit NumPy RuntimeWarning during long evaluations.
        mask64 = (1 << 64) - 1
        x = int(self.global_seed) & mask64
        x ^= 1469598103934665603
        x &= mask64
        x ^= ((int(r_idx) + 1) * 1099511628211) & mask64
        x &= mask64
        x ^= ((int(v_idx) + 1) * 14029467366897019727) & mask64
        x &= mask64
        return int(x % (1 << 32))

    def generate_blueprints(self) -> List[Dict]:
        blueprints = []
        for r_idx, r in enumerate(self.radii_set):
            analytic_kappa = 1.0 / r
            analytic_h_kappa = self.h / r

            for v_idx in range(self.variations):
                sub_seed = self._subseed(r_idx, v_idx)
                local_rng = np.random.default_rng(sub_seed)
                cx = local_rng.uniform(self.center_min, self.center_max)
                cy = local_rng.uniform(self.center_min, self.center_max)

                blueprint_id_str = f"rho{self.rho}_r{r_idx:03d}_v{v_idx:02d}_s{sub_seed}"
                blueprint_idx_int = int(self.rho * 100000 + r_idx * self.variations + v_idx)

                blueprints.append({
                    "meta": {
                        "blueprint_id": blueprint_id_str,
                        "blueprint_idx": blueprint_idx_int,
                        "geometry_type": "circle",
                        "resolution": self.rho,
                        "radius_idx": int(r_idx),
                        "variation_idx": int(v_idx),
                        "global_seed": self.global_seed,
                        "sub_seed": sub_seed,
                    },
                    "params": {
                        "h": float(self.h),
                        "radius": float(r),
                        "center": (float(cx), float(cy))
                    },
                    "label": {
                        "source": "analytic_circle",
                        "kappa": float(analytic_kappa),
                        "h_kappa": float(analytic_h_kappa)
                    }
                })
        return blueprints


if __name__ == "__main__":
    rho_test = 256

    gen1 = CircleGeometryGenerator(rho_test, seed=42, variations=5)
    bps1 = gen1.generate_blueprints()

    gen2 = CircleGeometryGenerator(rho_test, seed=42, variations=5)
    bps2 = gen2.generate_blueprints()

    print(f"Resolution: {rho_test}")
    print(f"h = {gen1.h:.12f}")
    print(f"Radius range = [{gen1.r_min:.12f}, {gen1.r_max:.12f}]")
    print(f"Num Radii: {gen1.num_radii}")
    print(f"Total Blueprints: {len(bps1)}")

    assert len(bps1) == gen1.num_radii * gen1.variations
    assert bps1[0]["params"] == bps2[0]["params"], "Reproducibility failed on first blueprint."
    assert bps1[-1]["params"] == bps2[-1]["params"], "Reproducibility failed on last blueprint."

    arr1 = np.array(
        [[bp["params"]["radius"], *bp["params"]["center"]] for bp in bps1],
        dtype=float
    )
    arr2 = np.array(
        [[bp["params"]["radius"], *bp["params"]["center"]] for bp in bps2],
        dtype=float
    )
    assert np.allclose(arr1, arr2), "Full blueprint array reproducibility failed."

    assert np.isclose(gen1.radii_set[0], gen1.r_min, atol=1e-12, rtol=0.0)
    assert np.isclose(gen1.radii_set[-1], gen1.r_max, atol=1e-12, rtol=0.0)
    dr = np.diff(gen1.radii_set)
    assert np.all(dr > 0), "Radii must be strictly increasing."

    for bp in bps1[:10]:
        cx, cy = bp["params"]["center"]
        assert gen1.center_min <= cx <= gen1.center_max
        assert gen1.center_min <= cy <= gen1.center_max

    labels = np.array([bps1[i * gen1.variations]["label"]["h_kappa"] for i in range(gen1.num_radii)])
    assert np.all(np.diff(labels) < 0), "h/r labels should decrease as radius increases."

    for r_idx in range(gen1.num_radii):
        vals = [bps1[r_idx * gen1.variations + v]["label"]["h_kappa"] for v in range(gen1.variations)]
        assert np.allclose(vals, vals[0]), f"Label mismatch at radius_idx={r_idx}"

    print(f"Sample first blueprint = {bps1[0]}")
    print("All checks passed.")
