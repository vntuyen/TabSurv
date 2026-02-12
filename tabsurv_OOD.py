import os
import time
import numpy as np
import pandas as pd
import torch

from datasets import (
    load_datafile_gene,  # (df, cols_standardize, cols_leave, time_col, event_col)
    load_tab_survival_dataset_censoring,
    load_tab_survival_dataset_test,  # (X_test, y_test_time, y_test_event)
)
from utils import manual_c_index_expected_time
from tabpfn import TabPFNRegressor

# --------------- Config ---------------
TRAINING_DATASET = "METABRIC"
TESTING_DATASETS = [
   "TCGA500", "GEO", "GSE6532", "GSE19783", "HEL",
   "unt", "nki", "transbig", "UK", "mainz", "upp"]


TIMESTRING = time.strftime("%Y%m%d%H%M")
RESULTS_DIR = f"./output/results_{TIMESTRING}"
os.makedirs(RESULTS_DIR, exist_ok=True)

PREDICTION_DIR = f"./output/prediction_OOD_{TIMESTRING}"
os.makedirs(PREDICTION_DIR, exist_ok=True)

MODEL_NAME = "TabSurv"
TEST_SIZE = 0.5
RANDOM_STATE = 42

np.random.seed(42)
_ = torch.manual_seed(42)


# --------------- Helpers ---------------
def _to_numpy_float32(X):
    if isinstance(X, pd.DataFrame):
        return X.values.astype(np.float32, copy=False)
    if isinstance(X, np.ndarray):
        return X.astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def _to_frame(X, ref_cols=None):
    """Return a DataFrame for export, preserving ref_cols if provided."""
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


# --------------- Main ---------------
def main():
    all_results = []
    ci_list = []

    # ----- Load & split training dataset (METABRIC) -----
    df_full, cols_standardize, cols_leave, time_col, event_col, feature_names = load_datafile_gene(TRAINING_DATASET)

    X_train_dead, y_train_dead, X_train_alive, y_train_alive, X_test_in, y_test_time_in, y_test_event_in = load_tab_survival_dataset_censoring(
        df_full, time_col, event_col, TEST_SIZE, RANDOM_STATE
    )
    feature_names = list(X_train_dead.columns) if isinstance(X_train_dead, pd.DataFrame) else None

    # ----- Train TabSurv (TabPFNRegressor) -----
    print(f"\n=== Training model: {MODEL_NAME} on {TRAINING_DATASET} ===")
    model1 = TabPFNRegressor(ignore_pretraining_limits=True)
    model1.fit(_to_numpy_float32(X_train_dead), _to_numpy_float32(y_train_dead))

    # ----- Predict survival time for censored/alive patients -----
    y_pred_alive = model1.predict(_to_numpy_float32(X_train_alive))

    # ======================================================
    # NEW: Construct train_full dataset (dead + alive)
    # ======================================================

    # Combine feature matrices
    X_train_full = pd.concat([X_train_dead, X_train_alive], axis=0).reset_index(drop=True)

    # Combine times:
    #    - dead → true observed survival time
    #    - alive → pseudo survival time predicted by model1
    y_train_full = pd.concat([
        pd.Series(y_train_dead).reset_index(drop=True),
        pd.Series(y_pred_alive).reset_index(drop=True)
    ], axis=0)

    print("Shapes:")
    print("X_train_dead :", X_train_dead.shape)
    print("X_train_alive:", X_train_alive.shape)
    print("X_train_full :", X_train_full.shape)
    print("y_train_full :", y_train_full.shape)

    # Ensure y_pred_alive is a 1-D numpy array
    y_pred_alive = np.asarray(y_pred_alive).ravel()

    # Build the time column used for training: dead -> true, alive -> predicted
    time_col_full = pd.concat([
        pd.Series(y_train_dead).reset_index(drop=True),
        pd.Series(y_pred_alive).reset_index(drop=True)
    ], axis=0).reset_index(drop=True).rename(time_col)

    # Real observed time for everyone: dead -> true, alive -> observed (censored) time
    time_real_full = pd.concat([
        pd.Series(y_train_dead).reset_index(drop=True),
        pd.Series(y_train_alive).reset_index(drop=True)
    ], axis=0).reset_index(drop=True).rename(time_col + "_real")

    # Event indicator: dead=1, alive=0
    event_full = pd.concat([
        pd.Series(np.ones(len(y_train_dead))),
        pd.Series(np.zeros(len(y_train_alive)))
    ], axis=0).reset_index(drop=True).rename(event_col)

    # Is the target a prediction? dead -> 0 (true), alive -> 1 (predicted)
    is_predicted = pd.concat([
        pd.Series(np.zeros(len(y_train_dead))),
        pd.Series(np.ones(len(y_train_alive)))
    ], axis=0).reset_index(drop=True).rename("is_predicted")

    # Sanity checks: all lengths must match X_train_full
    n = len(X_train_full)
    checks = {
        "X_train_full": n,
        "time_col_full": len(time_col_full),
        "time_real_full": len(time_real_full),
        "event_full": len(event_full),
        "is_predicted": len(is_predicted),
    }
    mismatches = {k: v for k, v in checks.items() if v != n}
    if mismatches:
        raise ValueError(
            "Length mismatch between X_train_full and one or more constructed columns:\n"
            + "\n".join([f"{k}: {v} != {n}" for k, v in mismatches.items()])
        )

    # Convert series to single-column DataFrames (avoids index/column alignment issues)
    df_time = time_col_full.to_frame(name=time_col)
    df_time_real = time_real_full.to_frame(name=time_col + "_real")
    df_event = event_full.to_frame(name=event_col)
    df_is_pred = is_predicted.to_frame(name="is_predicted")

    # Final concat along columns (axis=1) — all indices are RangeIndex and match
    train_full_df = pd.concat(
        [X_train_full.reset_index(drop=True),
         df_time,
         df_time_real,
         df_event,
         df_is_pred],
        axis=1
    )

    # Save
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    save_train_full = os.path.join(PREDICTION_DIR, f"{TRAINING_DATASET}_{MODEL_NAME}_train_full.csv")
    train_full_df.to_csv(save_train_full, index=False)

    print(f"\nSaved combined training dataset to: {save_train_full}")
    print("train_full_df shape:", train_full_df.shape)



    print(f"\n=== Training model: {MODEL_NAME} on {TRAINING_DATASET} full dataset with censoring ===")
    model = TabPFNRegressor(ignore_pretraining_limits=True)
    model.fit(_to_numpy_float32(X_train_full), _to_numpy_float32(y_train_full))


    # ----- In-distribution evaluation (METABRIC hold-out) -----
    print(f"Evaluating in-distribution on {TRAINING_DATASET} (held-out test)")
    y_pred_in = model.predict(_to_numpy_float32(X_test_in))

    df_pred_in = _to_frame(X_test_in, ref_cols=feature_names)
    df_pred_in["time"] = np.asarray(y_test_time_in, dtype=float)
    df_pred_in["event"] = np.asarray(y_test_event_in, dtype=int)
    df_pred_in["predicted"] = np.asarray(y_pred_in, dtype=float)

    cidx_in = manual_c_index_expected_time(
        df_pred_in, time_col="time", event_col="event", prediction_col="predicted"
    )
    print(f"C-index {TRAINING_DATASET}: {cidx_in:.4f}")
    pred_path = save_csv(df_pred_in, PREDICTION_DIR, f"{TRAINING_DATASET}_{MODEL_NAME}_predict.csv")
    print(f"Saved: {pred_path}")

    all_results.append({
        "dataset": TRAINING_DATASET,
        "model": MODEL_NAME,
        "C-index": float(cidx_in),
    })
    ci_list.append(float(cidx_in))

    # ----- OOD evaluation across external datasets -----
    for dataset_name in TESTING_DATASETS:
        print(f"Evaluating OOD on {dataset_name}")
        try:
            # If your loader supports feature alignment kwargs, add them here.
            X_test_out, y_test_time_out, y_test_event_out = load_tab_survival_dataset_test(dataset_name)

            y_pred_out = model.predict(_to_numpy_float32(X_test_out))

            df_pred_out = _to_frame(X_test_out, ref_cols=feature_names)
            df_pred_out["time"] = np.asarray(y_test_time_out, dtype=float)
            df_pred_out["event"] = np.asarray(y_test_event_out, dtype=int)
            df_pred_out["predicted"] = np.asarray(y_pred_out, dtype=float)

            cidx_out = manual_c_index_expected_time(
                df_pred_out, time_col="time", event_col="event", prediction_col="predicted"
            )
            print(f"C-index {dataset_name}: {cidx_out:.4f}")

            pred_path = save_csv(df_pred_out, PREDICTION_DIR, f"{dataset_name}_{MODEL_NAME}_predict.csv")
            print(f"Saved: {pred_path}")

            all_results.append({
                "dataset": dataset_name,
                "model": MODEL_NAME,
                "C-index": float(cidx_out),
            })
            ci_list.append(float(cidx_out))

        except Exception as e:
            print(f"[ERROR] Skipping {dataset_name}: {e}")
            continue

    # ----- Save per-dataset metrics -----
    results_df = pd.DataFrame(all_results)
    metrics_path = save_csv(results_df, RESULTS_DIR, "TabSurv_metrics.csv")
    print(f"Saved: {metrics_path}")

    # ----- Stability across datasets (C-index only) -----
    ci_arr = np.asarray(ci_list, dtype=float)
    ci_clean = ci_arr[~np.isnan(ci_arr)]
    stability_ci = float(np.mean(ci_clean) - np.std(ci_clean)) if ci_clean.size else np.nan

    stability_df = pd.DataFrame([{
        "model": MODEL_NAME,
        "mean_C-index": float(np.mean(ci_clean)) if ci_clean.size else np.nan,
        "std_C-index": float(np.std(ci_clean)) if ci_clean.size else np.nan,
        "Stability_CI": stability_ci,
    }])
    stab_path = save_csv(stability_df, RESULTS_DIR, "TabSurv_stability.csv")
    print(f"Saved: {stab_path}")

    print("\n=== Final Stability Scores ===")
    print(stability_df)


if __name__ == "__main__":
    main()
