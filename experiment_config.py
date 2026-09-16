"""Shared configuration for TabSurv OOD and treatment-recommendation experiments."""

from pathlib import Path
from functools import lru_cache
import os

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Flexible TabPFN model-source configuration
# -----------------------------------------------------------------------------
# Local/Gadi: if TABPFN_CKPT_PATH exists, use it explicitly.
# Networked machines (e.g. local Mac): if the checkpoint is absent, allow
# TabPFN to resolve/download the SAME v2.5 regressor checkpoint into its cache.
# Gadi PBS sets TABPFN_REQUIRE_LOCAL=1 and HF_HUB_OFFLINE=1, so a missing local
# checkpoint still fails immediately there instead of attempting the network.
TABPFN_CKPT_PATH = Path(os.environ.get(
    "TABPFN_CKPT_PATH",
    "/scratch/sq95/tv9849/tabsurv/tabpfn/tabpfn-v2.5-regressor-v2.5_default.ckpt",
)).expanduser()
TABPFN_DEVICE = os.environ.get("TABPFN_DEVICE", "auto")
TABPFN_NETWORK_MODEL = os.environ.get(
    "TABPFN_NETWORK_MODEL",
    "tabpfn-v2.5-regressor-v2.5_default.ckpt",
)


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def tabpfn_local_required() -> bool:
    """Whether this process must use a local checkpoint and may not download."""
    return (
        _env_truthy("TABPFN_REQUIRE_LOCAL")
        or _env_truthy("HF_HUB_OFFLINE")
        or _env_truthy("TRANSFORMERS_OFFLINE")
    )


@lru_cache(maxsize=2)
def check_ckpt(required: bool | None = None) -> Path | None:
    """Return a readable local checkpoint, or ``None`` when network fallback is allowed.

    Parameters
    ----------
    required:
        ``True`` forces a local checkpoint. ``False`` allows fallback. ``None``
        (default) derives the policy from TABPFN_REQUIRE_LOCAL / offline flags.
    """
    if required is None:
        required = tabpfn_local_required()

    path = TABPFN_CKPT_PATH
    if path.is_file():
        if not os.access(path, os.R_OK):
            raise PermissionError(f"TabPFN checkpoint is not readable: '{path}'")
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  [INFO] TabPFN checkpoint OK: '{path}' ({size_mb:.0f} MB)")
        return path

    if required:
        raise FileNotFoundError(
            f"TabPFN checkpoint not found: '{path}'\n"
            "A local checkpoint is required in this environment. Set "
            "TABPFN_CKPT_PATH to a readable .ckpt file, or unset "
            "TABPFN_REQUIRE_LOCAL/HF_HUB_OFFLINE on a machine with internet access."
        )

    print(
        f"  [INFO] Local TabPFN checkpoint not found at '{path}'. "
        f"Falling back to TabPFN network/cache resolution for '{TABPFN_NETWORK_MODEL}'."
    )
    return None


def tabpfn_model_source() -> str:
    """Human-readable active TabPFN model source for logging/preflight."""
    ckpt = check_ckpt(required=None)
    return str(ckpt) if ckpt is not None else f"network/cache:{TABPFN_NETWORK_MODEL}"


def tabpfn_regressor_kwargs(device: str | None = None) -> dict:
    """Common kwargs for TabPFNRegressor with local-first/network-fallback logic.

    When the configured checkpoint exists it is used exactly. Otherwise, on an
    online machine, a bare v2.5 checkpoint filename is supplied so TabPFN uses
    its cache/download mechanism while preserving the intended v2.5 model
    rather than silently switching to the package's newest default version.
    """
    ckpt = check_ckpt(required=None)
    model_path = str(ckpt) if ckpt is not None else TABPFN_NETWORK_MODEL
    return {
        "model_path": model_path,
        "device": device if device is not None else TABPFN_DEVICE,
    }

# -----------------------------------------------------------------------------
# Global reproducibility settings
# -----------------------------------------------------------------------------
SEEDS = [40, 41, 42, 43, 44]
OUTPUT_ROOT = Path("./output")

# -----------------------------------------------------------------------------
# OOD survival experiments (RFS / DMFS)
# -----------------------------------------------------------------------------
TEST_SIZE = 0.5

# Training feature matrices are not needed by evaluations and can be very large.
# Leave False for normal experiments. Set True only for debugging/auditing.
SAVE_TRAINING_ARTIFACTS = False

SCENARIOS = {
    "RFS": {
        "endpoint": "RFS",
        "training_dataset": "METABRIC",
        "testing_datasets": ["TCGA500", "GEO", "GSE6532", "GSE19783", "UK", "UPP"],
        "prediction_dir": OUTPUT_ROOT / "prediction_RFS",
        "results_dir": OUTPUT_ROOT / "results_RFS",
    },
    "DMFS": {
        "endpoint": "DMFS",
        "training_dataset": "nki",
        "testing_datasets": ["hel", "unt", "transbig", "mainz"],
        "prediction_dir": OUTPUT_ROOT / "prediction_DMFS",
        "results_dir": OUTPUT_ROOT / "results_DMFS",
    },
}

BASELINE_MODELS = ["DeepHS", "DeepSurv", "ENCox", "LH", "MTLR", "PCHazard", "PMF", "RSF"]

# TabSurv_M remains the proposed method for paired statistical comparisons.
# TabSurv_P and TabSurv_A now also produce InD predictions (50/50 source-cohort
# split) in addition to their OOD predictions. They are descriptive variants /
# ablations and are not added to the proposed-vs-baseline Wilcoxon family.
PROPOSED_METHOD = "TabSurv_M"
TAB_SURV_VARIANTS = [PROPOSED_METHOD, "TabSurv_P", "TabSurv_A"]
IND_EVALUATION_MODELS = [*TAB_SURV_VARIANTS, *BASELINE_MODELS]
ALL_EVALUATION_MODELS = [*TAB_SURV_VARIANTS, *BASELINE_MODELS]
FINAL_REPORT_MODELS = [*TAB_SURV_VARIANTS, *BASELINE_MODELS]
# Both InD and OOD descriptive report tables now include M/P/A + 8 baselines.
IND_REPORT_MODELS = FINAL_REPORT_MODELS.copy()
OOD_REPORT_MODELS = FINAL_REPORT_MODELS.copy()

# -----------------------------------------------------------------------------
# Treatment-recommendation experiment
# -----------------------------------------------------------------------------
# Two REC gene-set scenarios. Both TabSurv_REC and baselines_REC can run either
# scenario (or both) using the same split seeds and test fraction.
REC_72GENES_DATASET = ["METABRIC_72genes"]
REC_35genes_DATASET = ["METABRIC_35genes"]
REC_30GENES_DATASET = ["METABRIC_30genes"]
REC_ALLGENES_DATASET = ["METABRIC_allgenes"]
REC_TEST_SIZE = 0.3
REC_MODEL_RANDOM_STATE = 42

# User-specified output roots.
REC_PREDICTION_DIR = OUTPUT_ROOT / "prediction_REC"
REC_RESULTS_DIR = OUTPUT_ROOT / "results_REC"
REC_PLOTS_DIR = REC_RESULTS_DIR / "plots"

BASELINES_REC_PREDICTION_DIR = REC_PREDICTION_DIR / "prediction_BASELINES_REC"
BASELINES_REC_RESULTS_DIR = OUTPUT_ROOT / "results_REC"
BASELINES_REC_PLOTS_DIR = BASELINES_REC_RESULTS_DIR / "plots"

# Keep the conventional REC baseline definition exactly aligned with the
# survival baselines, then add the two causal-survival baselines separately.
REC_BASELINE_MODELS = BASELINE_MODELS.copy()
REC_CAUSAL_BASELINE_MODELS = ["SurvITE", "BITES"]
REC_ALL_BASELINE_MODELS = [*REC_BASELINE_MODELS, *REC_CAUSAL_BASELINE_MODELS]
REC_PREDICTIVE_BASELINE_MODELS = REC_BASELINE_MODELS.copy()

# Gene-set scenario registry.  The scenario is the dataset representation,
# not the method: each scenario contains method-specific output paths so both
# TabSurv_REC and baselines_REC can run on BOTH 72genes and allgenes without
# overwriting one another.
REC_SCENARIOS = {
    "72genes": {
        "datasets": REC_72GENES_DATASET,
        "TabSurv_REC": {
            "prediction_dir": REC_PREDICTION_DIR / "72genes" / "TabSurv_REC",
            "results_dir": REC_RESULTS_DIR / "72genes",
            "plots_dir": REC_PLOTS_DIR / "72genes" / "TabSurv_REC",
        },
        "baselines_REC": {
            "prediction_dir": BASELINES_REC_PREDICTION_DIR / "72genes",
            "results_dir": BASELINES_REC_RESULTS_DIR / "72genes",
            "plots_dir": BASELINES_REC_PLOTS_DIR / "72genes" / "baselines_REC",
        },
    },
    "35genes": {
        "datasets": REC_35genes_DATASET,
        "TabSurv_REC": {
            "prediction_dir": REC_PREDICTION_DIR / "35genes" / "TabSurv_REC",
            "results_dir": REC_RESULTS_DIR / "35genes",
            "plots_dir": REC_PLOTS_DIR / "35genes" / "TabSurv_REC",
        },
    },
    "30genes": {
        "datasets": REC_30GENES_DATASET,
        "TabSurv_REC": {
            "prediction_dir": REC_PREDICTION_DIR / "30genes" / "TabSurv_REC",
            "results_dir": REC_RESULTS_DIR / "30genes",
            "plots_dir": REC_PLOTS_DIR / "30genes" / "TabSurv_REC",
        },
    },
    "allgenes": {
        "datasets": REC_ALLGENES_DATASET,
        "TabSurv_REC": {
            "prediction_dir": REC_PREDICTION_DIR / "allgenes" / "TabSurv_REC",
            "results_dir": REC_RESULTS_DIR / "allgenes",
            "plots_dir": REC_PLOTS_DIR / "allgenes" / "TabSurv_REC",
        },
        "baselines_REC": {
            "prediction_dir": BASELINES_REC_PREDICTION_DIR / "allgenes",
            "results_dir": BASELINES_REC_RESULTS_DIR / "allgenes",
            "plots_dir": BASELINES_REC_PLOTS_DIR / "allgenes" / "baselines_REC",
        },
    },
}

# Causal survival baseline settings.
# Shared RMST horizon for SurvITE/BITES treatment-effect summaries.
REC_CAUSAL_HORIZON = 20.0

# SurvITE settings.
REC_SURVITE_Z_DIM = 100
REC_SURVITE_REP_HIDDEN = 100
REC_SURVITE_HEAD_HIDDEN = 100
REC_SURVITE_REP_LAYERS = 3
REC_SURVITE_HEAD_LAYERS = 2
REC_SURVITE_ACTIVATION = "elu"
REC_SURVITE_DROPOUT = 0.3
REC_SURVITE_BETA = 1e-3
REC_SURVITE_GAMMA = 0.0
REC_SURVITE_LR = 1e-3
REC_SURVITE_WEIGHT_DECAY = 0.0
REC_SURVITE_BATCH_SIZE = 512
REC_SURVITE_EPOCHS = int(os.environ.get("REC_SURVITE_EPOCHS", "3000"))
REC_SURVITE_PATIENCE = int(os.environ.get("REC_SURVITE_PATIENCE", "20"))
REC_SURVITE_CHECK_EVERY = int(os.environ.get("REC_SURVITE_CHECK_EVERY", "25"))
REC_SURVITE_DEVICE = os.environ.get("REC_SURVITE_DEVICE", "auto")

# BITES settings following Schrod et al. (Bioinformatics, 2022).
REC_BITES_SHARED_LAYERS = [15, 10]
REC_BITES_INDIVIDUAL_LAYERS = [10, 5]
REC_BITES_ALPHA = 0.1
REC_BITES_BLUR = 0.05
REC_BITES_LR = 1e-3
REC_BITES_WEIGHT_DECAY = 0.2
REC_BITES_DROPOUT = 0.1
REC_BITES_EPOCHS = 1000
REC_BITES_PATIENCE = 50
REC_BITES_DEVICE = os.environ.get("REC_BITES_DEVICE", "auto")


def selected_rec_scenarios(name: str):
    """Return REC gene-set scenario names in deterministic order."""
    key = str(name).lower()
    if key == "both":
        return ["72genes", "allgenes"]
    if key not in REC_SCENARIOS:
        raise ValueError(
            f"Unknown REC scenario '{name}'. Choose 72genes, allgenes, or both."
        )
    return [key]


def ensure_rec_output_dirs(scenario_name: str, method_name: str):
    """Create and return method-specific paths for one REC gene-set scenario."""
    if scenario_name not in REC_SCENARIOS:
        raise KeyError(f"Unknown REC scenario: {scenario_name}")
    if method_name not in {"TabSurv_REC", "baselines_REC"}:
        raise KeyError(f"Unknown REC method: {method_name}")

    scenario_cfg = REC_SCENARIOS[scenario_name]
    method_cfg = scenario_cfg[method_name]
    method_cfg["prediction_dir"].mkdir(parents=True, exist_ok=True)
    method_cfg["results_dir"].mkdir(parents=True, exist_ok=True)
    method_cfg["plots_dir"].mkdir(parents=True, exist_ok=True)
    return {
        "scenario": scenario_name,
        "datasets": list(scenario_cfg["datasets"]),
        "prediction_dir": method_cfg["prediction_dir"],
        "results_dir": method_cfg["results_dir"],
        "plots_dir": method_cfg["plots_dir"],
    }

def selected_scenarios(name: str):
    """Return OOD scenario names in deterministic order."""
    key = name.upper()
    if key == "BOTH":
        return ["RFS", "DMFS"]
    if key not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Choose RFS, DMFS, or both.")
    return [key]


def selected_settings(name: str):
    """Return evaluation settings in deterministic order."""
    key = name.upper()
    if key == "BOTH":
        return ["InD", "OOD"]
    if key == "IND":
        return ["InD"]
    if key == "OOD":
        return ["OOD"]
    raise ValueError(f"Unknown setting '{name}'. Choose InD, OOD, or both.")


def ensure_output_dirs(scenario_name: str):
    """Create and return the configured directories for one OOD scenario."""
    cfg = SCENARIOS[scenario_name]
    cfg["prediction_dir"].mkdir(parents=True, exist_ok=True)
    cfg["results_dir"].mkdir(parents=True, exist_ok=True)
    return cfg



def format_mean_std(mean, std, ndigits=4):
    """Format a numeric mean and standard deviation consistently."""
    if mean is None or not np.isfinite(mean):
        return "NaN"
    if std is None or not np.isfinite(std):
        std = 0.0
    return f"{mean:.{ndigits}f} ± {std:.{ndigits}f}"


def compact_prediction_frame(
    times,
    events,
    predictions,
    prediction_col: str,
    patient_ids=None,
) -> pd.DataFrame:
    """Create a minimal prediction table used by evaluation scripts.

    Parameters
    ----------
    times, events, predictions
        One-dimensional arrays of equal length.
    prediction_col
        Usually ``predicted`` (expected survival time) or ``risk_score``.
    patient_ids
        Optional stable patient identifiers. If omitted, deterministic 0-based
        row IDs are used. REC scripts pass the original dataframe indices so a
        test patient's ID remains meaningful across different random splits.
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=int).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)

    n = len(times)
    if len(events) != n or len(predictions) != n:
        raise ValueError(
            f"Prediction length mismatch: time={n}, event={len(events)}, "
            f"{prediction_col}={len(predictions)}"
        )

    if patient_ids is None:
        patient_ids = np.arange(n, dtype=np.int64)
    else:
        patient_ids = np.asarray(patient_ids).reshape(-1)
        if len(patient_ids) != n:
            raise ValueError(
                f"patient_id length mismatch: patient_id={len(patient_ids)}, expected={n}"
            )

    return pd.DataFrame({
        "patient_id": patient_ids,
        "time": times,
        "event": events,
        prediction_col: predictions,
    })
