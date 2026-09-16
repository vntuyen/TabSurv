#!/usr/bin/env python3
"""TabSurv_M: multiple-imputation TabSurv for RFS/DMFS InD + OOD evaluation.

InD and OOD deliberately share the same 50% source-training split and fitted
multiple-imputation ensemble for each seed. ``--setting`` controls evaluation
only; OOD never triggers a 100%-source retraining for TabSurv_M.
"""

import argparse
import random

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor

from datasets import (
    load_datafile_gene,
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

MODEL_NAME = "TabSurv_M"
N_IMPUTATIONS = 5
QUANTILE_LEVELS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]


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
    if not np.isfinite(mean):
        return "NaN"
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:.4f} ± {std:.4f}"


def sample_truncated_from_quantiles(quantile_preds, quantile_levels, lower_bounds, n_draws, rng):
    """Sample imputed times from predictive quantiles conditional on T >= censoring time."""
    quantile_preds = np.asarray(quantile_preds, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float).ravel()
    qs = np.asarray(quantile_levels, dtype=float)
    draws = np.empty((n_draws, quantile_preds.shape[0]), dtype=float)

    for i in range(quantile_preds.shape[0]):
        vals = np.maximum.accumulate(quantile_preds[i])
        c = lower_bounds[i]
        p_below_c = np.interp(c, vals, qs, left=0.0, right=1.0)
        u = rng.uniform(low=min(p_below_c, 0.999), high=1.0, size=n_draws)
        sampled = np.interp(u, qs, vals)
        draws[:, i] = np.maximum(sampled, c)
    return draws


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
            std_c = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
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


def run_scenario(scenario_name, setting="both"):
    selected = selected_settings(setting)
    run_ind = "InD" in selected
    run_ood = "OOD" in selected
    cfg = ensure_output_dirs(scenario_name)
    training_dataset = cfg["training_dataset"]
    testing_datasets = cfg["testing_datasets"]
    prediction_dir = cfg["prediction_dir"]
    results_dir = cfg["results_dir"]

    print("\n" + "=" * 72)
    print(f"{MODEL_NAME} | scenario={scenario_name} | train={training_dataset}")
    print(f"settings={selected}")
    if run_ind:
        print(f"InD={training_dataset} held-out split (test_size={TEST_SIZE})")
    if run_ood:
        print(f"OOD={testing_datasets} using the SAME 50% source-training split/model as InD")
    print(f"seeds={SEEDS}")
    print("=" * 72)

    all_results = []

    for seed in SEEDS:
        _seed_everything(seed)
        rng = np.random.default_rng(seed)
        print(f"\n--- {MODEL_NAME} {scenario_name}, seed={seed} ---")

        df_full, _, _, time_col, event_col, _feature_names = load_datafile_gene(training_dataset)
        (
            X_train_dead, y_train_dead, X_train_alive, y_train_alive,
            X_test_in, y_test_time_in, y_test_event_in,
        ) = load_tab_survival_dataset_censoring(
            df_full, time_col, event_col, TEST_SIZE, seed
        )

        model1 = TabPFNRegressor(
            ignore_pretraining_limits=True,
            random_state=seed,
            **tabpfn_regressor_kwargs(),
        )
        model1.fit(_to_numpy_float32(X_train_dead), _to_numpy_float32(y_train_dead).reshape(-1))

        quantile_preds = model1.predict(
            _to_numpy_float32(X_train_alive),
            output_type="quantiles",
            quantiles=QUANTILE_LEVELS,
        )
        quantile_preds = np.asarray(quantile_preds, dtype=float)
        if quantile_preds.shape[0] == len(QUANTILE_LEVELS) and quantile_preds.shape[1] == len(X_train_alive):
            quantile_preds = quantile_preds.T
        if quantile_preds.shape != (len(X_train_alive), len(QUANTILE_LEVELS)):
            raise ValueError(
                f"Unexpected quantile prediction shape {quantile_preds.shape}; "
                f"expected ({len(X_train_alive)}, {len(QUANTILE_LEVELS)})."
            )

        imputed_draws = sample_truncated_from_quantiles(
            quantile_preds,
            QUANTILE_LEVELS,
            np.asarray(y_train_alive, dtype=float).ravel(),
            N_IMPUTATIONS,
            rng,
        )

        X_train_full = pd.concat([X_train_dead, X_train_alive], axis=0).reset_index(drop=True)

        if SAVE_TRAINING_ARTIFACTS:
            time_real_full = np.concatenate([
                np.asarray(y_train_dead, dtype=float).ravel(),
                np.asarray(y_train_alive, dtype=float).ravel(),
            ])
            event_full = np.concatenate([
                np.ones(len(y_train_dead), dtype=int),
                np.zeros(len(y_train_alive), dtype=int),
            ])
            train_diag = pd.DataFrame({
                "patient_id": np.arange(len(X_train_full), dtype=np.int64),
                "time_real": time_real_full,
                "event": event_full,
                "is_predicted": np.concatenate([
                    np.zeros(len(y_train_dead), dtype=int),
                    np.ones(len(y_train_alive), dtype=int),
                ]),
                "imputation_mean": np.concatenate([
                    np.full(len(y_train_dead), np.nan),
                    imputed_draws.mean(axis=0),
                ]),
                "imputation_std": np.concatenate([
                    np.full(len(y_train_dead), np.nan),
                    imputed_draws.std(axis=0),
                ]),
            })
            train_diag.to_csv(
                prediction_dir / f"{training_dataset}_{MODEL_NAME}_seed{seed}_train_full.csv",
                index=False,
            )

        stage2_models = []
        for m in range(N_IMPUTATIONS):
            y_train_full_m = np.concatenate([
                np.asarray(y_train_dead, dtype=float).ravel(),
                imputed_draws[m],
            ]).astype(np.float32)
            model_m = TabPFNRegressor(
                ignore_pretraining_limits=True,
                random_state=seed + m,
                **tabpfn_regressor_kwargs(),
            )
            model_m.fit(_to_numpy_float32(X_train_full), y_train_full_m)
            stage2_models.append(model_m)

        if run_ind:
            # ------------------------------------------------------------------
            # In-distribution evaluation on the held-out split of the training
            # cohort. The same split seed is used by the baseline script.
            # ------------------------------------------------------------------
            try:
                X_test_in_np = _to_numpy_float32(X_test_in)
                preds_in_by_model = np.stack(
                    [mdl.predict(X_test_in_np) for mdl in stage2_models], axis=0
                )
                y_pred_in = preds_in_by_model.mean(axis=0)

                # Preserve the original dataframe row index in the seed-specific
                # patient ID. This allows a downstream check that TabSurv_M and
                # baselines used the same InD held-out patients for a given seed.
                patient_ids_in = np.asarray([
                    f"{training_dataset}_seed{seed}_row{idx}" for idx in X_test_in.index
                ], dtype=object)
                df_pred_in = compact_prediction_frame(
                    y_test_time_in,
                    y_test_event_in,
                    y_pred_in,
                    "predicted",
                    patient_ids=patient_ids_in,
                )
                cidx_in = manual_c_index_expected_time(
                    df_pred_in,
                    time_col="time",
                    event_col="event",
                    prediction_col="predicted",
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
                    "C-index": float(cidx_in),
                })
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

        if run_ood:
            # ------------------------------------------------------------------
            # OOD evaluation on external cohorts.
            # ------------------------------------------------------------------
            for dataset_name in testing_datasets:
                try:
                    X_test, y_time, y_event = load_tab_survival_dataset_test(dataset_name)
                    X_np = _to_numpy_float32(X_test)
                    preds = np.stack([mdl.predict(X_np) for mdl in stage2_models], axis=0)
                    y_pred = preds.mean(axis=0)

                    df_pred = compact_prediction_frame(y_time, y_event, y_pred, "predicted")
                    c_index = manual_c_index_expected_time(
                        df_pred, time_col="time", event_col="event", prediction_col="predicted"
                    )
                    pred_path = prediction_dir / f"{dataset_name}_{MODEL_NAME}_seed{seed}_predict.csv"
                    df_pred.to_csv(pred_path, index=False)
                    print(f"{dataset_name} [OOD]: C-index={c_index:.4f} -> {pred_path}")

                    all_results.append({
                        "scenario": scenario_name,
                        "training_dataset": training_dataset,
                        "dataset": dataset_name,
                        "setting": "OOD",
                        "model": MODEL_NAME,
                        "seed": seed,
                        "C-index": float(c_index),
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
                   help="Evaluation setting to run. TabSurv_M always trains on the same 50% source-training split per seed; OOD reuses that fitted ensemble.")
    return p.parse_args()


def main():
    args = parse_args()
    for scenario_name in selected_scenarios(args.scenario):
        run_scenario(scenario_name, setting=args.setting)


if __name__ == "__main__":
    main()
