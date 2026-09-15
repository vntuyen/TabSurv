#!/usr/bin/env python3
"""Fail-fast checks for the TabSurv project on Gadi before long experiments."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from experiment_config import SCENARIOS, REC_SCENARIOS, check_ckpt, tabpfn_model_source

PROJECT_DIR = Path(__file__).resolve().parent


def main() -> int:
    print("=== TabSurv project preflight ===")
    print("project:", PROJECT_DIR)
    print("python :", sys.executable)
    print("version:", sys.version.replace("\n", " "))
    print("cwd    :", Path.cwd())

    # Ensure Python resolves the local shared modules, not stale copies elsewhere.
    for name in ["datasets", "models", "utils", "experiment_config", "tabsurv_REC", "baselines_REC"]:
        mod = importlib.import_module(name)
        path = Path(mod.__file__).resolve()
        print(f"{name:18s}: {path}")
        if PROJECT_DIR not in path.parents and path != PROJECT_DIR / f"{name}.py":
            raise RuntimeError(f"{name} resolved outside project directory: {path}")

    # Import runtime dependencies now so missing/incompatible installs fail early.
    import numpy as np
    import pandas as pd
    import torch
    import sklearn
    import sksurv
    import pycox
    import torchtuples
    import lifelines
    import tabpfn
    import geomloss

    print("numpy :", np.__version__)
    print("pandas:", pd.__version__)
    print("torch :", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device   :", torch.cuda.get_device_name(0))
    print("sklearn:", sklearn.__version__)
    print("sksurv :", getattr(sksurv, "__version__", "unknown"))
    print("pycox  :", getattr(pycox, "__version__", "unknown"))
    print("tabpfn :", getattr(tabpfn, "__version__", "unknown"))
    print("geomloss:", getattr(geomloss, "__version__", "installed"))

    ckpt = check_ckpt(required=None)
    print("checkpoint:", ckpt if ckpt is not None else "not present locally")
    print("TabPFN source:", tabpfn_model_source())

    input_dir = PROJECT_DIR / "input"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    expected = set()
    for cfg in SCENARIOS.values():
        expected.add(f"{cfg['training_dataset']}.csv")
        expected.update(f"{d}.csv" for d in cfg["testing_datasets"])
    for rec_cfg in REC_SCENARIOS.values():
        expected.update(f"{d}_treatment.csv" for d in rec_cfg["datasets"])

    missing = sorted(name for name in expected if not (input_dir / name).is_file())
    if missing:
        raise FileNotFoundError(
            "Missing required input files:\n  " + "\n  ".join(missing)
        )
    print(f"input files: OK ({len(expected)} required files found)")

    # Environment should be offline on Gadi.
    print("HF_HUB_OFFLINE:", os.environ.get("HF_HUB_OFFLINE", "<unset>"))
    print("TABPFN_DEVICE :", os.environ.get("TABPFN_DEVICE", "<unset>"))
    print("=== preflight OK ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
