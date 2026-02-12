#!/usr/bin/env python3
"""
OOD pipeline (baselines only; improved error reporting + robust y unpacking)
"""

import os
import time
import traceback
import numpy as np
import pandas as pd
import torch
from typing import Any, Tuple

# project-local dataset & metric utilities
from datasets import (
    preprocess_dataset,
    load_datafile_gene,
    get_target,
    preprocess_dataset_test,
)

from utils import manual_c_index_expected_time
from utils import get_labtrans, evaluate_model_sksurv

from models import get_model, model_dict

# models
import torchtuples as tt
from pycox.models import LogisticHazard, PMF, DeepHitSingle, PCHazard, MTLR, CoxPH
from sksurv.ensemble import RandomSurvivalForest

# ---------------- Config ----------------
TRAINING_DATASET = "METABRIC"
TESTING_DATASETS = [
   "TCGA500", "GEO", "GSE6532", "GSE19783", "HEL",
   "unt", "nki", "transbig", "UK", "mainz", "upp"
]



TIMESTRING = time.strftime("%Y%m%d%H%M")
RESULTS_DIR = f"./output/results_{TIMESTRING}"
os.makedirs(RESULTS_DIR, exist_ok=True)
#PREDICTION_DIR = f"./output/prediction_OOD_{TIMESTRING}"
PREDICTION_DIR = f"./output/prediction_OOD"
os.makedirs(PREDICTION_DIR, exist_ok=True)

TEST_SIZE = 0.5
RANDOM_STATE = 42
SEED = 42

np.random.seed(SEED)
_ = torch.manual_seed(SEED)

# ---------------- Helpers ----------------
def _to_numpy_float32(X):
    if isinstance(X, pd.DataFrame):
        return X.values.astype(np.float32, copy=False)
    if isinstance(X, np.ndarray):
        return X.astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)

def _to_frame(X, ref_cols=None):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    X = np.asarray(X)
    if ref_cols is not None and len(ref_cols) == X.shape[1]:
        return pd.DataFrame(X, columns=list(ref_cols))
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

def save_csv(df, folder: str, filename: str) -> str:
    path = os.path.join(folder, filename)
    _ensure_parent_dir(path)
    df.to_csv(path, index=False)
    return path

MODELS = [ "DeepHitSingle", "DeepSurv", "LogisticHazard", "MTLR", "PCHazard", "PMF",     "RSF"]


# ---------------- Main pipeline ----------------
def main():
    all_results = []


    for model_name in MODELS:
        print(f"  Training model: {model_name}")

        ci_list = []

        # ----- Load & split training dataset (METABRIC) -----

        df, cols_standardize, cols_leave, time_col, event_col, feature_names = load_datafile_gene(TRAINING_DATASET)

        df_train, df_val, df_test, x_train, x_val, x_test, x_mapper = preprocess_dataset(
            df, cols_standardize, cols_leave, time_col, event_col, TEST_SIZE, RANDOM_STATE)


        # ----- Train TabSurv (TabPFNRegressor) -----
        print(f"\n=== Training model: {model_name} on {TRAINING_DATASET} ===")



        print(f"  Training model: {model_name}")
        model_class = model_dict[model_name]
        labtrans = get_labtrans(model_class, 10) if model_name in ["LogisticHazard", "PMF", "DeepHitSingle", "PCHazard",
                                                                   "MTLR"] else None

        if labtrans is not None:
            y_train = labtrans.fit_transform(*get_target(df_train, time_col, event_col))
            y_val = labtrans.transform(*get_target(df_val, time_col, event_col))
            times_test, events_test = get_target(df_test, time_col, event_col)
            model = get_model(model_name, x_train.shape[1], labtrans.out_features, labtrans)
        else:
            times_train, events_train = get_target(df_train, time_col, event_col)
            times_val, events_val = get_target(df_val, time_col, event_col)
            times_test, events_test = get_target(df_test, time_col, event_col)
            y_train = (times_train, events_train)
            y_val = (times_val, events_val)
            model = get_model(model_name, x_train.shape[1])

        # Train model
        if model_name == "RSF":
            model = RandomSurvivalForest(
                n_estimators=200,
                min_samples_split=10,
                min_samples_leaf=15,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            )
            y_train_rsf = np.array(
                [(bool(e), t) for e, t in zip(df_train[event_col], df_train[time_col])],
                dtype=[(event_col, 'bool'), (time_col, 'float')]
            )
            model.fit(x_train, y_train_rsf)
        else:
            callbacks = [tt.cb.EarlyStopping()]
            model.fit(x_train, y_train, batch_size=256, epochs=100,
                      callbacks=callbacks, val_data=(x_val, y_val))


        # ----- OOD evaluation -----
        for dataset_name in TESTING_DATASETS:
            print(f"Evaluating {model_name} on {dataset_name}")
            try:

                df, cols_standardize, cols_leave, time_col, event_col, feature_names = load_datafile_gene(dataset_name)

                df_test, X_test_out, x_mapper = preprocess_dataset_test(df, cols_standardize, cols_leave, time_col, event_col)
                times_test, events_test = get_target(df_test, time_col, event_col)

                # Evaluate model
                metrics = evaluate_model_sksurv(model, X_test_out, times_test, events_test, model_name, feature_names)
                df_results = metrics["df_results"]

                cidx_out = metrics["c_index"][0]
                print(f"C-index_{dataset_name}_{model_name}: {cidx_out}")

                df_results_rename = df_results.rename(columns={
                    "real_survival": "time",
                    "event": "event",
                    "predicted_risk_score": "predicted"
                })
                df_results.to_csv(os.path.join(f"{PREDICTION_DIR}/{dataset_name}_{model_name}_predict.csv"), index=False)

                all_results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "C-index": float(cidx_out),
                })
                ci_list.append(float(cidx_out))

            except Exception as e:
                print(f"[ERROR] Skipping {dataset_name}: {e}")
                continue


        # stability summary
        ci_arr = np.asarray(ci_list, dtype=float)
        ci_clean = ci_arr[~np.isnan(ci_arr)]
        stability_ci = float(np.mean(ci_clean) - np.std(ci_clean)) if ci_clean.size else np.nan
        stability_row = {
            "model": model_name,
            "mean_C-index": float(np.mean(ci_clean)) if ci_clean.size else np.nan,
            "std_C-index": float(np.std(ci_clean)) if ci_clean.size else np.nan,
            "Stability_CI": stability_ci,
        }
        save_csv(pd.DataFrame([stability_row]), RESULTS_DIR, f"{model_name}_stability.csv")
        print(f"Saved stability for {model_name}")

    # Save aggregated results
    results_df = pd.DataFrame(all_results)
    save_csv(results_df, RESULTS_DIR, "all_baseline_models_metrics_debug.csv")
    print("Saved overall results:", os.path.join(RESULTS_DIR, "all_baseline_models_metrics_debug.csv"))


if __name__ == "__main__":
    main()
