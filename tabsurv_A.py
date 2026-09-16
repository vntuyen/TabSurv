#!/usr/bin/env python3
"""TabSurv_A: uncensored-only ablation for RFS/DMFS InD + OOD evaluation.

Experimental design
-------------------
For every seed and endpoint scenario two independent fits are performed:

InD
    * stratified 50/50 split of the source training cohort (TEST_SIZE=0.5)
    * fit TabSurv_A on uncensored patients from the 50% training partition only
    * evaluate on the held-out 50% partition

OOD
    * use 100% of the configured source training cohort
    * fit TabSurv_A on all uncensored patients in that full source cohort
    * evaluate on the configured external OOD cohorts

The OOD algorithm is therefore not weakened by the addition of InD reporting.
Prediction CSVs remain compact: patient_id, time, event, predicted.
"""

import argparse
import random

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

MODEL_NAME = "TabSurv_A"


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


def _fit_uncensored_only(X_uncensored, y_uncensored, seed, label):
    y = np.asarray(y_uncensored, dtype=np.float32).reshape(-1)
    if len(X_uncensored) < 2:
        raise ValueError(
            f"{label}: only {len(X_uncensored)} uncensored training samples are available."
        )
    model = TabPFNRegressor(
        ignore_pretraining_limits=True,
        random_state=seed,
        **tabpfn_regressor_kwargs(),
    )
    model.fit(_to_numpy_float32(X_uncensored), y)
    return model


def _evaluate_expected_time(model, X_test, y_time, y_event, patient_ids=None):
    y_pred = np.asarray(model.predict(_to_numpy_float32(X_test)), dtype=float).reshape(-1)
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
    return results_df, summary_df, raw_path, summary_path


def run_scenario(scenario_name, setting="both"):
    selected = selected_settings(setting)
    run_ind = "InD" in selected
    run_ood = "OOD" in selected
    cfg = ensure_output_dirs(scenario_name)
    training_dataset = cfg["training_dataset"]
    testing_datasets = list(cfg["testing_datasets"])
    prediction_dir = cfg["prediction_dir"]
    results_dir = cfg["results_dir"]

    print("\n" + "=" * 80)
    print(f"{MODEL_NAME} | scenario={scenario_name}")
    print(f"settings={selected}")
    if run_ind:
        print(f"InD : {training_dataset}, held-out test_size={TEST_SIZE}")
    if run_ood:
        print(f"OOD : train on 100% {training_dataset} -> {testing_datasets}")
    print(f"Seeds: {SEEDS}")
    print("=" * 80)

    all_results = []

    for seed in SEEDS:
        _seed_everything(seed)
        print(f"\n--- {MODEL_NAME} {scenario_name}, seed={seed} ---")

        df_full, _, _, time_col, event_col, _feature_names = load_datafile_gene(training_dataset)

        if run_ind:
            # ==================================================================
            # InD branch: 50% train / 50% held-out test.
            # Only uncensored patients from the TRAINING HALF are used to fit A.
            # ==================================================================
            X_train_unc_in = y_train_unc_in = X_train_cens_in = None
            X_test_in = y_test_time_in = y_test_event_in = df_pred_in = None
            model_in = None
            try:
                (
                    X_train_unc_in,
                    y_train_unc_in,
                    X_train_cens_in,
                    _y_train_cens_in,
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

                model_in = _fit_uncensored_only(
                    X_train_unc_in,
                    y_train_unc_in,
                    seed,
                    label=f"{training_dataset} InD seed={seed}",
                )

                # X_test_in retains the source dataframe row index after the same
                # stratified outer split used by TabSurv_M and matched baselines.
                patient_ids_in = np.asarray([
                    f"{training_dataset}_seed{seed}_row{int(idx)}"
                    for idx in X_test_in.index
                ], dtype=object)

                cidx_in, df_pred_in = _evaluate_expected_time(
                    model_in,
                    X_test_in,
                    y_test_time_in,
                    y_test_event_in,
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
                    "n_train_uncensored": int(len(X_train_unc_in)),
                    "n_train_censored_ignored": int(len(X_train_cens_in)),
                    "training_fraction": 1.0 - TEST_SIZE,
                })

                if SAVE_TRAINING_ARTIFACTS:
                    pd.DataFrame({
                        "patient_id": np.arange(len(y_train_unc_in), dtype=np.int64),
                        "time": np.asarray(y_train_unc_in, dtype=float).reshape(-1),
                        "event": 1,
                        "is_predicted": 0,
                    }).to_csv(
                        prediction_dir / f"{training_dataset}_{MODEL_NAME}_seed{seed}_InD_train_uncensored.csv",
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
                    "error": f"{type(exc).__name__}: {exc}",
                })
            finally:
                model_in = None
                X_train_unc_in = y_train_unc_in = X_train_cens_in = None
                X_test_in = y_test_time_in = y_test_event_in = df_pred_in = None

        if run_ood:
            # ==================================================================
            # OOD branch: independent refit on 100% of the source cohort.
            # TabSurv_A still ignores censored patients by design, but the source
            # cohort itself is NOT split/held out for OOD training.
            # ==================================================================
            try:
                (
                    X_unc_full,
                    y_unc_full,
                    X_cens_full,
                    _y_cens_full,
                ) = load_full_dataset_censoring(
                    df_full,
                    time_col,
                    event_col,
                    seed,
                )

                model_ood = _fit_uncensored_only(
                    X_unc_full,
                    y_unc_full,
                    seed,
                    label=f"{training_dataset} OOD-full seed={seed}",
                )

                if SAVE_TRAINING_ARTIFACTS:
                    pd.DataFrame({
                        "patient_id": np.arange(len(y_unc_full), dtype=np.int64),
                        "time": np.asarray(y_unc_full, dtype=float).reshape(-1),
                        "event": 1,
                        "is_predicted": 0,
                    }).to_csv(
                        prediction_dir / f"{training_dataset}_{MODEL_NAME}_seed{seed}_OOD_train_uncensored.csv",
                        index=False,
                    )

                for dataset_name in testing_datasets:
                    try:
                        X_test_out, y_time_out, y_event_out = load_tab_survival_dataset_test(dataset_name)
                        cidx_out, df_pred_out = _evaluate_expected_time(
                            model_ood,
                            X_test_out,
                            y_time_out,
                            y_event_out,
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
                            "n_train_uncensored": int(len(X_unc_full)),
                            "n_train_censored_ignored": int(len(X_cens_full)),
                            "training_fraction": 1.0,
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
                            "error": f"{type(exc).__name__}: {exc}",
                        })

                del model_ood

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
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    _, summary_df, raw_path, summary_path = _save_summary(all_results, results_dir)
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
    return p.parse_args()


def main():
    args = parse_args()
    for scenario_name in selected_scenarios(args.scenario):
        run_scenario(scenario_name, setting=args.setting)


if __name__ == "__main__":
    main()
