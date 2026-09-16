#!/usr/bin/env python3
"""TabSurv_P: pseudo-time variant for RFS/DMFS InD + OOD evaluation.

Experimental design
-------------------
For every seed and endpoint scenario two independent fits are performed:

InD
    * stratified 50/50 split of the source training cohort (TEST_SIZE=0.5)
    * fit Stage 1/Stage 2 only on the 50% training partition
    * evaluate on the held-out 50% partition

OOD
    * independently refit Stage 1/Stage 2 on 100% of the source training cohort
    * evaluate on the configured external OOD cohorts

Thus adding InD reporting does NOT reduce the OOD training cohort. The OOD
algorithm remains the full-source TabSurv_P method.

Memory safety
-------------
The memory-safe behaviour is retained:
  * --device auto uses CPU on macOS, CUDA elsewhere when available
  * TabPFN low-memory mode is used
  * prediction is batched with automatic OOM backoff
  * caches/models are released between fits, datasets and seeds

Prediction CSVs remain compact: patient_id, time, event, predicted.
"""

import argparse
import gc
import os
import platform
import random
from typing import Optional

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor

from datasets import (
    load_datafile_gene,
    load_full_dataset_censoring,
    load_tab_survival_dataset_censoring,
    load_tab_survival_dataset_test,
)
from experiment_config import (
    SAVE_TRAINING_ARTIFACTS,
    SEEDS,
    TEST_SIZE,
    compact_prediction_frame,
    ensure_output_dirs,
    selected_scenarios,
    selected_settings,
    tabpfn_regressor_kwargs,
)
from utils import manual_c_index_expected_time

MODEL_NAME = "TabSurv_P"
DEFAULT_PREDICT_BATCH_SIZE = 32


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_numpy_float32(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy(dtype=np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def _fmt(mean, std):
    if mean is None or not np.isfinite(mean):
        return "NaN"
    if std is None or not np.isfinite(std):
        std = 0.0
    return f"{mean:.4f} ± {std:.4f}"


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in (
        "out of memory",
        "mps backend out of memory",
        "cuda out of memory",
        "cannot allocate memory",
    ))


def _clear_memory(device: Optional[str] = None):
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _resolve_device(requested: str) -> str:
    requested = requested.lower()
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        if requested == "mps":
            mps_ok = bool(
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
            if not mps_ok:
                raise RuntimeError("--device mps requested but MPS is not available.")
        return requested

    # Deliberately avoid MPS by default for high-dimensional TabPFN because
    # the Apple unified-memory allocator can OOM during large OOD inference.
    if platform.system() == "Darwin":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _new_tabpfn(seed: int, device: str) -> TabPFNRegressor:
    return TabPFNRegressor(
        ignore_pretraining_limits=True,
        random_state=seed,
        fit_mode="low_memory",
        memory_saving_mode=True,
        **tabpfn_regressor_kwargs(device=device),
    )


def _predict_batched(model, X, batch_size: int, device: str, label: str):
    X_np = _to_numpy_float32(X)
    n = len(X_np)
    if n == 0:
        return np.empty(0, dtype=float)

    batch_size = min(max(1, int(batch_size)), n)
    predictions = []
    start = 0

    while start < n:
        current_bs = min(batch_size, n - start)
        while True:
            stop = start + current_bs
            try:
                pred = model.predict(X_np[start:stop])
                predictions.append(np.asarray(pred, dtype=float).reshape(-1))
                start = stop
                _clear_memory(device)
                break
            except RuntimeError as exc:
                if not _is_oom_error(exc) or current_bs <= 1:
                    raise
                new_bs = max(1, current_bs // 2)
                print(
                    f"[MEMORY] {label}: OOM for rows {start}:{stop} "
                    f"with batch_size={current_bs}; retrying with batch_size={new_bs}."
                )
                _clear_memory(device)
                current_bs = new_bs
                batch_size = min(batch_size, new_bs)

    out = np.concatenate(predictions)
    if len(out) != n:
        raise RuntimeError(f"Prediction length mismatch for {label}: expected {n}, got {len(out)}")
    return out


def _fit_pseudo_time_model(
    X_dead,
    y_dead,
    X_alive,
    y_alive_observed,
    seed,
    device,
    predict_batch_size,
    label,
):
    """Fit the two-stage TabSurv_P model on the supplied training partition."""
    if len(X_dead) < 2:
        raise ValueError(f"{label}: only {len(X_dead)} uncensored samples are available.")

    print(f"[{label}] Stage 1: n_uncensored={len(X_dead)}, n_censored={len(X_alive)}")
    model1 = _new_tabpfn(seed, device)
    model1.fit(
        _to_numpy_float32(X_dead),
        np.asarray(y_dead, dtype=np.float32).reshape(-1),
    )

    if len(X_alive):
        y_pred_alive = _predict_batched(
            model1,
            X_alive,
            predict_batch_size,
            device,
            label=f"{label} censored pseudo-time prediction",
        )
        y_alive = np.maximum(
            np.asarray(y_pred_alive, dtype=float).ravel(),
            np.asarray(y_alive_observed, dtype=float).ravel(),
        )
    else:
        y_alive = np.empty(0, dtype=float)

    del model1
    _clear_memory(device)

    X_train_full = pd.concat([X_dead, X_alive], axis=0).reset_index(drop=True)
    y_train_full = np.concatenate([
        np.asarray(y_dead, dtype=float).ravel(),
        y_alive,
    ]).astype(np.float32)

    print(f"[{label}] Stage 2: n_train={len(X_train_full)}")
    model2 = _new_tabpfn(seed, device)
    model2.fit(_to_numpy_float32(X_train_full), y_train_full)

    return model2, X_train_full, y_train_full, y_alive


def _evaluate_expected_time(model, X_test, y_time, y_event, batch_size, device, label, patient_ids=None):
    y_pred = _predict_batched(model, X_test, batch_size, device, label=label)
    df_pred = compact_prediction_frame(
        y_time,
        y_event,
        y_pred,
        "predicted",
        patient_ids=patient_ids,
    )
    c_index = manual_c_index_expected_time(
        df_pred,
        time_col="time",
        event_col="event",
        prediction_col="predicted",
    )
    return float(c_index), df_pred


def _save_summary(rows, results_dir):
    results_df = pd.DataFrame(rows)
    raw_path = results_dir / f"{MODEL_NAME}_metrics_all_seeds.csv"
    results_df.to_csv(raw_path, index=False)

    if results_df.empty:
        summary_df = pd.DataFrame(columns=[
            "setting", "dataset", "model", "n_seeds",
            "mean_C-index", "std_C-index", "C-index (mean ± std)",
        ])
    else:
        summary_rows = []
        for (setting, dataset), group in results_df.groupby(["setting", "dataset"], sort=False):
            vals = pd.to_numeric(group["C-index"], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            mean_c = float(np.mean(vals)) if vals.size else np.nan
            std_c = float(np.std(vals, ddof=1)) if vals.size > 1 else (0.0 if vals.size == 1 else np.nan)
            summary_rows.append({
                "setting": setting,
                "dataset": dataset,
                "model": MODEL_NAME,
                "n_seeds": int(vals.size),
                "mean_C-index": mean_c,
                "std_C-index": std_c,
                "C-index (mean ± std)": _fmt(mean_c, std_c),
            })
        summary_df = pd.DataFrame(summary_rows)

    summary_path = results_dir / f"{MODEL_NAME}_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_df, raw_path, summary_path


def run_scenario(scenario_name, setting="both", device_request="auto", predict_batch_size=DEFAULT_PREDICT_BATCH_SIZE):
    selected = selected_settings(setting)
    run_ind = "InD" in selected
    run_ood = "OOD" in selected
    cfg = ensure_output_dirs(scenario_name)
    training_dataset = cfg["training_dataset"]
    testing_datasets = list(cfg["testing_datasets"])
    prediction_dir = cfg["prediction_dir"]
    results_dir = cfg["results_dir"]
    device = _resolve_device(device_request)

    print("\n" + "=" * 88)
    print(f"{MODEL_NAME} | scenario={scenario_name}")
    print(f"settings={selected}")
    if run_ind:
        print(f"InD : {training_dataset}, held-out test_size={TEST_SIZE}")
    if run_ood:
        print(f"OOD : train on 100% {training_dataset} -> {testing_datasets}")
    print(f"Seeds={SEEDS}")
    print(f"TabPFN device={device} (requested={device_request})")
    print(f"fit_mode=low_memory | memory_saving_mode=True | predict_batch_size={predict_batch_size}")
    if platform.system() == "Darwin" and device_request == "auto":
        print("[INFO] macOS detected: CPU selected instead of MPS for memory safety.")
    print("=" * 88)

    all_results = []

    for seed in SEEDS:
        _seed_everything(seed)
        _clear_memory(device)
        print(f"\n--- {MODEL_NAME} {scenario_name}, seed={seed} ---")

        df_full, _, _, time_col, event_col, _feature_names = load_datafile_gene(training_dataset)

        if run_ind:
            # ==================================================================
            # InD branch: train ONLY on the 50% source training partition.
            # ==================================================================
            model_in = X_train_in_full = y_train_in_full = y_alive_in = None
            X_dead_in = y_dead_in = X_alive_in = y_alive_obs_in = None
            X_test_in = y_test_time_in = y_test_event_in = df_pred_in = None
            try:
                (
                    X_dead_in,
                    y_dead_in,
                    X_alive_in,
                    y_alive_obs_in,
                    X_test_in,
                    y_test_time_in,
                    y_test_event_in,
                ) = load_tab_survival_dataset_censoring(
                    df_full,
                    time_col,
                    event_col,
                    TEST_SIZE,
                    seed,
                )

                model_in, X_train_in_full, y_train_in_full, y_alive_in = _fit_pseudo_time_model(
                    X_dead_in,
                    y_dead_in,
                    X_alive_in,
                    y_alive_obs_in,
                    seed,
                    device,
                    predict_batch_size,
                    label=f"{training_dataset} InD seed={seed}",
                )

                patient_ids_in = np.asarray([
                    f"{training_dataset}_seed{seed}_row{int(idx)}"
                    for idx in X_test_in.index
                ], dtype=object)

                cidx_in, df_pred_in = _evaluate_expected_time(
                    model_in,
                    X_test_in,
                    y_test_time_in,
                    y_test_event_in,
                    predict_batch_size,
                    device,
                    label=f"{training_dataset} InD seed={seed}",
                    patient_ids=patient_ids_in,
                )
                pred_path_in = prediction_dir / f"{training_dataset}_{MODEL_NAME}_seed{seed}_predict.csv"
                df_pred_in.to_csv(pred_path_in, index=False)
                print(f"{training_dataset} [InD]: C-index={cidx_in:.4f} -> {pred_path_in}")

                all_results.append({
                    "scenario": scenario_name,
                    "training_dataset": training_dataset,
                    "dataset": training_dataset,
                    "setting": "InD",
                    "model": MODEL_NAME,
                    "seed": seed,
                    "C-index": cidx_in,
                    "n_train": int(len(X_train_in_full)),
                    "training_fraction": 1.0 - TEST_SIZE,
                    "device": device,
                    "predict_batch_size": predict_batch_size,
                })

                if SAVE_TRAINING_ARTIFACTS:
                    pd.DataFrame({
                        "patient_id": np.arange(len(X_train_in_full), dtype=np.int64),
                        "time": np.asarray(y_train_in_full, dtype=float),
                        "event": np.concatenate([
                            np.ones(len(y_dead_in), dtype=int),
                            np.zeros(len(y_alive_obs_in), dtype=int),
                        ]),
                        "is_predicted": np.concatenate([
                            np.zeros(len(y_dead_in), dtype=int),
                            np.ones(len(y_alive_obs_in), dtype=int),
                        ]),
                    }).to_csv(
                        prediction_dir / f"{training_dataset}_{MODEL_NAME}_seed{seed}_InD_train_full.csv",
                        index=False,
                    )

            except Exception as exc:
                print(f"[ERROR] InD {training_dataset}, seed={seed}: {type(exc).__name__}: {exc}")
                all_results.append({
                    "scenario": scenario_name,
                    "training_dataset": training_dataset,
                    "dataset": training_dataset,
                    "setting": "InD",
                    "model": MODEL_NAME,
                    "seed": seed,
                    "C-index": np.nan,
                    "device": device,
                    "predict_batch_size": predict_batch_size,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            finally:
                # Drop the complete InD branch before constructing the independent
                # full-source OOD fit. This is important on high-dimensional cohorts.
                model_in = None
                X_train_in_full = y_train_in_full = y_alive_in = None
                X_dead_in = y_dead_in = X_alive_in = y_alive_obs_in = None
                X_test_in = y_test_time_in = y_test_event_in = df_pred_in = None
                _clear_memory(device)

        if run_ood:
            # ==================================================================
            # OOD branch: independent refit on 100% of the source cohort.
            # ==================================================================
            model_ood = X_train_ood_full = y_train_ood_full = y_alive_ood = None
            try:
                X_dead_full, y_dead_full, X_alive_full, y_alive_obs_full = load_full_dataset_censoring(
                    df_full,
                    time_col,
                    event_col,
                    seed,
                )

                model_ood, X_train_ood_full, y_train_ood_full, y_alive_ood = _fit_pseudo_time_model(
                    X_dead_full,
                    y_dead_full,
                    X_alive_full,
                    y_alive_obs_full,
                    seed,
                    device,
                    predict_batch_size,
                    label=f"{training_dataset} OOD-full seed={seed}",
                )

                if SAVE_TRAINING_ARTIFACTS:
                    pd.DataFrame({
                        "patient_id": np.arange(len(X_train_ood_full), dtype=np.int64),
                        "time": np.asarray(y_train_ood_full, dtype=float),
                        "event": np.concatenate([
                            np.ones(len(y_dead_full), dtype=int),
                            np.zeros(len(y_alive_obs_full), dtype=int),
                        ]),
                        "is_predicted": np.concatenate([
                            np.zeros(len(y_dead_full), dtype=int),
                            np.ones(len(y_alive_obs_full), dtype=int),
                        ]),
                    }).to_csv(
                        prediction_dir / f"{training_dataset}_{MODEL_NAME}_seed{seed}_OOD_train_full.csv",
                        index=False,
                    )

                # Remove duplicate training matrices before external inference.
                del X_train_ood_full, y_train_ood_full, y_alive_ood
                X_train_ood_full = y_train_ood_full = y_alive_ood = None
                _clear_memory(device)

                for dataset_name in testing_datasets:
                    X_test_out = y_time_out = y_event_out = df_pred_out = None
                    try:
                        X_test_out, y_time_out, y_event_out = load_tab_survival_dataset_test(dataset_name)
                        cidx_out, df_pred_out = _evaluate_expected_time(
                            model_ood,
                            X_test_out,
                            y_time_out,
                            y_event_out,
                            predict_batch_size,
                            device,
                            label=f"{dataset_name} OOD seed={seed}",
                        )
                        pred_path_out = prediction_dir / f"{dataset_name}_{MODEL_NAME}_seed{seed}_predict.csv"
                        df_pred_out.to_csv(pred_path_out, index=False)
                        print(f"{dataset_name} [OOD]: C-index={cidx_out:.4f} -> {pred_path_out}")

                        all_results.append({
                            "scenario": scenario_name,
                            "training_dataset": training_dataset,
                            "dataset": dataset_name,
                            "setting": "OOD",
                            "model": MODEL_NAME,
                            "seed": seed,
                            "C-index": cidx_out,
                            "n_train": int(len(df_full)),
                            "training_fraction": 1.0,
                            "device": device,
                            "predict_batch_size": predict_batch_size,
                        })
                    except Exception as exc:
                        print(f"[ERROR] OOD {dataset_name}, seed={seed}: {type(exc).__name__}: {exc}")
                        all_results.append({
                            "scenario": scenario_name,
                            "training_dataset": training_dataset,
                            "dataset": dataset_name,
                            "setting": "OOD",
                            "model": MODEL_NAME,
                            "seed": seed,
                            "C-index": np.nan,
                            "device": device,
                            "predict_batch_size": predict_batch_size,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                    finally:
                        _clear_memory(device)

            except Exception as exc:
                print(f"[ERROR] OOD full-source fit {training_dataset}, seed={seed}: {type(exc).__name__}: {exc}")
                for dataset_name in testing_datasets:
                    all_results.append({
                        "scenario": scenario_name,
                        "training_dataset": training_dataset,
                        "dataset": dataset_name,
                        "setting": "OOD",
                        "model": MODEL_NAME,
                        "seed": seed,
                        "C-index": np.nan,
                        "device": device,
                        "predict_batch_size": predict_batch_size,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            finally:
                if model_ood is not None:
                    del model_ood
                _clear_memory(device)

        del df_full
        _clear_memory(device)

    summary_df, raw_path, summary_path = _save_summary(all_results, results_dir)
    print(f"\nSaved raw metrics: {raw_path}")
    print(f"Saved summary:     {summary_path}")
    if not summary_df.empty:
        print(summary_df[["setting", "dataset", "model", "n_seeds", "C-index (mean ± std)"]].to_string(index=False))
    return summary_df


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["RFS", "DMFS", "both"], default="both")
    p.add_argument("--setting", choices=["InD", "OOD", "both"], default="both",
                   help="Evaluation setting to run. InD uses the 50/50 split; OOD independently trains on 100% of the source cohort.")
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help=(
            "TabPFN device. 'auto' uses CPU on macOS, CUDA when available "
            "elsewhere, otherwise CPU."
        ),
    )
    p.add_argument(
        "--predict-batch-size",
        type=int,
        default=DEFAULT_PREDICT_BATCH_SIZE,
        help="Rows per TabPFN predict() call; automatically reduced on OOM.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.predict_batch_size < 1:
        raise SystemExit("--predict-batch-size must be >= 1")
    for scenario_name in selected_scenarios(args.scenario):
        run_scenario(
            scenario_name,
            setting=args.setting,
            device_request=args.device,
            predict_batch_size=args.predict_batch_size,
        )


if __name__ == "__main__":
    main()
