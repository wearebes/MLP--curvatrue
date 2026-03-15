#!/usr/bin/env python3
"""
generate_data.py — End-to-end train-data generation pipeline.
"""
import argparse
import os
import sys
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

if __package__ in (None, ""):
    # Support running from inside generate/ via: python -m generate_data
    # by injecting the project root and importing through the package name.
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_HERE)
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    from generate.config import TARGET_RHOS, OUTPUT_DIR, PROJECT_ROOT, CFL, REINIT_STEPS, GEOMETRY_SEED, VARIATIONS
    from generate.geometry import CircleGeometryGenerator
    from generate.field_builder import LevelSetFieldBuilder
    from generate.reinitializer import ReinitFieldPackBuilder
    from generate.dataset_compiler import HDF5DatasetCompiler
    from generate.validation import validate_curvature_dataset
else:
    from .config import TARGET_RHOS, OUTPUT_DIR, PROJECT_ROOT, CFL, REINIT_STEPS, GEOMETRY_SEED, VARIATIONS
    from .geometry import CircleGeometryGenerator
    from .field_builder import LevelSetFieldBuilder
    from .reinitializer import ReinitFieldPackBuilder
    from .dataset_compiler import HDF5DatasetCompiler
    from .validation import validate_curvature_dataset


def generate_full_dataset_for_resolution(rho: int, h5_filename: str) -> None:
    """
    End-to-end generation of complete multi-million sample dataset for a given resolution.
    Includes safe file handling and rigorous error catching.
    """
    if os.path.exists(h5_filename):
        os.remove(h5_filename)

    generator     = CircleGeometryGenerator(resolution_rho=rho, seed=GEOMETRY_SEED, variations=VARIATIONS)
    field_builder = LevelSetFieldBuilder(dtype=np.float64)
    reinit_builder = ReinitFieldPackBuilder(cfl=CFL)

    print(f"\n>>> Starting Pipeline for Resolution: {rho}")
    dataset_compiler = HDF5DatasetCompiler(h5_filename, mode='w')

    try:
        blueprints = generator.generate_blueprints()
        print(f"[*] Successfully generated {len(blueprints)} circle blueprints.")

        for bp in tqdm(blueprints, desc=f"Processing rho={rho}"):
            pack_sdf    = field_builder.build_circle_sdf(bp, return_grid=False)
            pack_nonsdf = field_builder.build_circle_nonsdf(bp, return_grid=False)

            out_sdf    = reinit_builder.build(pack_sdf)
            out_nonsdf = reinit_builder.build(pack_nonsdf, steps_list=REINIT_STEPS)

            batch_packs = [out_sdf["0"]]
            for step in [str(s) for s in REINIT_STEPS]:
                batch_packs.append(out_nonsdf[step])

            dataset_compiler.append_data(batch_packs)

        print(f"[*] Dataset generation complete! Saved to {h5_filename}")
        dataset_compiler.verify_final()

    except Exception as e:
        print(f"\n[!] Pipeline interrupted for rho={rho} due to error: {e}")
        raise
    finally:
        dataset_compiler.close()


def print_train_summary(target_rhos, output_dir) -> None:
    print("\nFinal Verification Results (Train data)")
    for rho in target_rhos:
        db_name = os.path.join(output_dir, f"train_rho{rho}.h5")
        if os.path.exists(db_name):
            with h5py.File(db_name, 'r') as f:
                total_samples = f['X'].shape[0]
                steps = np.unique(np.array(f['reinit_steps'][:, 0], dtype=np.int32)).tolist()
                blueprint_idx = np.array(f['blueprint_idx'][:, 0], dtype=np.int32)
                radius_idx    = np.array(f['radius_idx'][:, 0],    dtype=np.int32)
                print(
                    f"Resolution {rho}: {total_samples:>9,} samples | steps={steps} | "
                    f"blueprint_idx=[{int(blueprint_idx.min())}, {int(blueprint_idx.max())}] "
                    f"(unique={np.unique(blueprint_idx).size}) | "
                    f"radius_idx=[{int(radius_idx.min())}, {int(radius_idx.max())}] -> {db_name}"
                )
        else:
            print(f"Resolution {rho}: Failed to locate {db_name}")


def generate_train_datasets(target_rhos=None, output_dir: str | os.PathLike[str] = OUTPUT_DIR) -> None:
    target_rhos = list(target_rhos or TARGET_RHOS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Launching Train Data Generation Pipeline")

    for rho in target_rhos:
        db_name = output_dir / f"train_rho{rho}.h5"
        generate_full_dataset_for_resolution(rho, str(db_name))
        validate_curvature_dataset(str(db_name), rho=rho)

    print_train_summary(target_rhos, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate level-set curvature datasets.")
    parser.add_argument(
        "--rho", type=int, nargs="+", default=TARGET_RHOS,
        help=f"Resolution(s) to generate (default: {TARGET_RHOS})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Output directory for HDF5 files (default: {OUTPUT_DIR})"
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    generate_train_datasets(target_rhos=args.rho, output_dir=output_dir)
