import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple
import copy
from .reinitializer import ReinitQualityEvaluator

class HDF5DatasetCompiler:
    """
    Stage: Extracts 3x3 stencils from field_packs and writes them into an HDF5 database.
    """
    def __init__(self, h5_filepath: str, mode: str = "w"):
        self.filepath = h5_filepath
        self.file = h5py.File(self.filepath, mode)
        if mode == "w":
            self.X_dset = self.file.create_dataset("X", shape=(0, 9), maxshape=(None, 9), dtype=np.float32, compression="gzip")
            self.Y_dset = self.file.create_dataset("Y", shape=(0, 1), maxshape=(None, 1), dtype=np.float32, compression="gzip")
            self.Steps_dset = self.file.create_dataset("reinit_steps", shape=(0, 1), maxshape=(None, 1), dtype=np.int32, compression="gzip")
            # blueprint_idx/radius_idx are stored as int32 to prevent float32
            # precision loss for IDs > 2^24 (e.g. rho*100000 + r_idx*variations + v_idx).
            self.Blueprint_dset = self.file.create_dataset("blueprint_idx", shape=(0, 1), maxshape=(None, 1), dtype=np.int32, compression="gzip")
            self.Radius_dset = self.file.create_dataset("radius_idx", shape=(0, 1), maxshape=(None, 1), dtype=np.int32, compression="gzip")

    @staticmethod
    def extract_stencils(field_pack: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        phi = field_pack["field"]["phi"]
        h = float(field_pack["params"]["h"])
        h_kappa = float(field_pack["label"]["h_kappa"])

        I, J = ReinitQualityEvaluator.get_sampling_coordinates(phi)
        if len(I) == 0:
            return np.zeros((0, 9), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
        patches = []

        for i, j in zip(I, J):
            patch_2d = phi[i-1:i+2, j-1:j+2]
            patch_correct = patch_2d[:, ::-1].T
            patches.append(patch_correct.flatten())

        patches = np.stack(patches, axis=0)

        X_b = patches.astype(np.float32)
        Y_b = np.full((X_b.shape[0], 1), h_kappa, dtype=np.float32)

        X_batch_aug = np.vstack([X_b, -X_b])
        Y_batch_aug = np.vstack([Y_b, -Y_b])

        return X_batch_aug, Y_batch_aug

    @staticmethod
    def _extract_required_meta_int(meta: Dict[str, Any], key: str, phi_type: str) -> int:
        if key in meta:
            return int(meta[key])
        nested_meta = meta.get("meta")
        if isinstance(nested_meta, dict) and key in nested_meta:
            return int(nested_meta[key])
        raise KeyError(f"[Error] Field pack (phi_type: {phi_type}) missing required meta['{key}'].")

    @staticmethod
    def _validate_nonnegative_meta_value(value: int, key: str, phi_type: str) -> int:
        value = int(value)
        if value < 0:
            raise ValueError(
                f"[Error] Field pack (phi_type: {phi_type}) has invalid meta['{key}']={value}. "
                f"This usually means metadata was dropped upstream."
            )
        return value

    def append_data(self, field_packs: List[Dict[str, Any]]):
        """Appends batches and writes to HDF5. (Augmentation is already handled in extract_stencils)"""
        X_list, Y_list = [], []
        steps_list, bp_list, rad_list = [], [], []

        for pack in field_packs:
            X_b, Y_b = self.extract_stencils(pack)
            if X_b.shape[0] == 0:
                continue
            meta = pack["meta"]
            phi_type = pack["field"].get("phi_type", "")
            is_sdf = meta.get("is_sdf")
            if is_sdf is None:
                is_sdf = "sdf" in phi_type and "nonsdf" not in phi_type
            else:
                is_sdf = bool(is_sdf)

            if is_sdf:
                steps = 0
            else:
                reinfo = meta.get("reinit", None)
                if reinfo is None or "steps" not in reinfo:
                    raise KeyError(f"[Error] Non-SDF pack (phi_type: {phi_type}) missing meta['reinit']['steps'].")
                steps = int(reinfo["steps"])

            blueprint_idx = self._extract_required_meta_int(meta, "blueprint_idx", phi_type)
            radius_idx = self._extract_required_meta_int(meta, "radius_idx", phi_type)
            # Validate non-negative before writing — catches any upstream metadata loss
            blueprint_idx = self._validate_nonnegative_meta_value(blueprint_idx, "blueprint_idx", phi_type)
            radius_idx = self._validate_nonnegative_meta_value(radius_idx, "radius_idx", phi_type)

            n_b = X_b.shape[0]

            X_list.append(X_b)
            Y_list.append(Y_b)
            steps_list.append(np.full((n_b, 1), steps, dtype=np.int32))
            bp_list.append(np.full((n_b, 1), blueprint_idx, dtype=np.int32))
            rad_list.append(np.full((n_b, 1), radius_idx, dtype=np.int32))

        if not X_list:
            return

        X_all = np.vstack(X_list)
        Y_all = np.vstack(Y_list)
        Steps_all = np.vstack(steps_list)
        BP_all = np.vstack(bp_list)
        Rad_all = np.vstack(rad_list)

        curr_len = self.X_dset.shape[0]
        new_len = curr_len + X_all.shape[0]

        self.X_dset.resize((new_len, 9))
        self.Y_dset.resize((new_len, 1))
        self.Steps_dset.resize((new_len, 1))
        self.Blueprint_dset.resize((new_len, 1))
        self.Radius_dset.resize((new_len, 1))

        self.X_dset[curr_len:new_len] = X_all
        self.Y_dset[curr_len:new_len] = Y_all
        self.Steps_dset[curr_len:new_len] = Steps_all
        self.Blueprint_dset[curr_len:new_len] = BP_all
        self.Radius_dset[curr_len:new_len] = Rad_all

    def verify_final(self) -> None:
        """
        Samples up to 10 000 metadata rows from the completed file and raises
        RuntimeError if any blueprint_idx / radius_idx are negative or any
        reinit_steps value is outside the expected set {0, 5, 10, 15, 20}.
        Call this BEFORE close() so a corrupt file never silently enters training.
        """
        self.file.flush()
        n = self.X_dset.shape[0]
        if n == 0:
            print("[verify_final] WARNING: file is empty (0 samples).")
            return

        check_n = min(10_000, n)
        rng = np.random.default_rng(seed=0)
        idx = np.sort(rng.choice(n, check_n, replace=False))

        bp  = np.array(self.Blueprint_dset[idx, 0], dtype=np.int32)
        rad = np.array(self.Radius_dset[idx, 0],    dtype=np.int32)
        stp = np.array(self.Steps_dset[idx, 0],     dtype=np.int32)

        errors = []
        if np.any(bp < 0):
            n_bad = int((bp < 0).sum())
            errors.append(
                f"blueprint_idx: {n_bad}/{check_n} sampled rows are negative "
                f"(likely -1; metadata was dropped upstream in CircleGeometryGenerator)"
            )
        if np.any(rad < 0):
            n_bad = int((rad < 0).sum())
            errors.append(f"radius_idx: {n_bad}/{check_n} sampled rows are negative")
        unknown_steps = set(np.unique(stp).tolist()) - {0, 5, 10, 15, 20}
        if unknown_steps:
            errors.append(f"reinit_steps: unexpected values {unknown_steps}")

        if errors:
            msg = "\n  ".join(errors)
            raise RuntimeError(
                f"[verify_final] \u274c FAILED ({n:,} samples):\n  {msg}"
            )

        print(
            f"[verify_final] \u2713 OK \u2014 {n:,} total samples, "
            f"{check_n} sampled rows all valid "
            f"(blueprint_idx\u22650, radius_idx\u22650, reinit_steps\u2208{{0,5,10,15,20}})."
        )

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


class LevelSetCurvatureDataset(Dataset):
    """
    HDF5 file,Includes multiprocessing safety protocols for DataLoader compatibility.
    """
    def __init__(self, h5_filepath: str):
        super().__init__()
        self.filepath = h5_filepath
        self.file = None

        # Open briefly just to get the total length, then safely close
        with h5py.File(self.filepath, "r") as f:
            self.total_samples = f["X"].shape[0]

    def __len__(self) -> int:
        return self.total_samples

    def __getstate__(self):
        """
        Prevents DataLoader from crashing when num_workers > 0.
        Drops the un-picklable HDF5 file handle before distributing to workers.
        """
        state = self.__dict__.copy()
        state["file"] = None
        return state

    def _ensure_open(self):
        """Lazy loading: every worker process opens its own safe file handle."""
        if self.file is None:
            self.file = h5py.File(self.filepath, "r")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:

        self._ensure_open()
        x = np.array(self.file["X"][idx])
        y = np.array(self.file["Y"][idx])

        step    = int(np.array(self.file["reinit_steps"][idx])[0])
        bp_idx  = int(np.array(self.file["blueprint_idx"][idx])[0])
        rad_idx = int(np.array(self.file["radius_idx"][idx])[0])

        return torch.from_numpy(x), torch.from_numpy(y), step, bp_idx, rad_idx

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
