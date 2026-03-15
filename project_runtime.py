from __future__ import annotations

import shutil
import sys
from pathlib import Path


def disable_bytecode_cache() -> None:
    sys.dont_write_bytecode = True


def cleanup_bytecode_caches(project_root: str | Path) -> None:
    root = Path(project_root)
    code_dirs = [
        root,
        root / "generate",
        root / "train",
        root / "test",
    ]

    for code_dir in code_dirs:
        if not code_dir.exists():
            continue

        direct_pycache = code_dir / "__pycache__"
        if direct_pycache.exists():
            shutil.rmtree(direct_pycache, ignore_errors=True)

        for pyc_file in code_dir.glob("*.pyc"):
            pyc_file.unlink(missing_ok=True)
