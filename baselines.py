#!/usr/bin/env python3
"""OOD baseline survival experiments for the common RFS and DMFS scenarios."""

import argparse
import random

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

from datasets import load_datafile_gene, get_target, preprocess_dataset_test
from experiment_config import (
    BASELINE_MODELS, SEEDS, TEST_SIZE, compact_prediction_frame,
    ensure_output_dirs, selected_scenarios, selected_settings,
)
from models import get_model, model_dict, fit_coxnet_survival
from utils import get_labtrans, evaluate_model_sksurv

L1_RATIOS = (0.5,)
CV_FOLDS = 3


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def _preprocess_ind_matched(df, cols_standardize, cols_leave, time_col, event_col, test_size, seed):
    """Create the same stratified outer InD split used by TabSurv_M.

    The outer held-out test patients are generated exactly like
    load_tab_survival_dataset_censoring(): reset row IDs, stratify on event
    when possible, and split with the same seed/test_size. The remaining
    training portion is then split into train/validation for baseline fitting,
    preserving the baseline models' existing train/validation workflow.
    Preprocessing is fitted on df_train only.
    """
    df = df.dropna(subset=[time_col, event_col]).reset_index(drop=True).copy()
    df["__patient_row_id"] = np.arange(len(df), dtype=int)

    stratify_col = df[event_col] if df[event_col].nunique() > 1 else None
    train_ids, test_ids = train_test_split(
        df["__patient_row_id"],
        test_size=test_size,
        random_state=seed,
        stratify=stratify_col,
    )
    train_full = df[df["__patient_row_id"].isin(train_ids)].copy()
    df_test = df[df["__patient_row_id"].isin(test_ids)].copy()

    # Preserve the original baseline inner train/validation split behaviour.
    df_train, df_val = train_test_split(
        train_full, test_size=test_size, random_state=seed
    )

    cols_standardize = [
        c for c in cols_standardize
        if c not in {time_col, event_col, "__patient_row_id"}
    ]
    cols_leave = [
        c for c in cols_leave
        if c not in {time_col, event_col, "__patient_row_id"}
    ]
    standardize = [([c], StandardScaler()) for c in cols_standardize]
    leave = [(c, None) for c in cols_leave]
    x_mapper = DataFrameMapper(standardize + leave)

    drop_cols = [time_col, event_col, "__patient_row_id"]
    X_train = df_train.drop(columns=drop_cols, errors="ignore")
    X_val = df_val.drop(columns=drop_cols, errors="ignore")
    X_test = df_test.drop(columns=drop_cols, errors="ignore")

    x_train = x_mapper.fit_transform(X_train).astype("float32")
    x_val = x_mapper.transform(X_val).astype("float32")
    x_test = x_mapper.transform(X_test).astype("float32")
    return df_train, df_val, df_test, x_train, x_val, x_test, x_mapper

def _fmt(mean, std):
    if not np.isfinite(mean):
        return "NaN"
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:.4f} ± {std:.4f}"


def run_scenario(scenario_name, models_to_run=None, setting="both"):
    selected = selected_settings(setting)
    run_ind = "InD" in selected
    run_ood = "OOD" in selected
    cfg = ensure_output_dirs(scenario_name)
    training_dataset = cfg["training_dataset"]
    testing_datasets = cfg["testing_datasets"]
    prediction_dir = cfg["prediction_dir"]
    results_dir = cfg["results_dir"]
    models_to_run = list(models_to_run or BASELINE_MODELS)

    unknown = [m for m in models_to_run if m not in BASELINE_MODELS]
    if unknown:
        raise ValueError(f"Unknown baseline model(s): {unknown}. Valid: {BASELINE_MODELS}")

    print("\n" + "=" * 72)
    print(f"Baselines | scenario={scenario_name} | train={training_dataset}")
    print(f"settings={selected}")
    if run_ind:
        print(f"InD={training_dataset} held-out test_size={TEST_SIZE}")
    if run_ood:
        print(f"OOD={testing_datasets}")
    print(f"seeds={SEEDS}")
    print(f"models={models_to_run}")
    print("=" * 72)

    all_results = []
    encox_rows = []

    for model_name in models_to_run:
        for seed in SEEDS:
            _seed_everything(seed)
            print(f"\n--- {model_name} {scenario_name}, seed={seed} ---")

            df, cols_standardize, cols_leave, time_col, event_col, feature_names = load_datafile_gene(training_dataset)
            df_train, df_val, df_test, x_train, x_val, x_test, _x_mapper = _preprocess_ind_matched(
                df, cols_standardize, cols_leave, time_col, event_col, TEST_SIZE, seed
            )

            # The outer held-out InD split is now identical to TabSurv_M's
            # stratified split for the same seed.
            times_test_in, events_test_in = get_target(df_test, time_col, event_col)

            if model_name in ("RSF", "ENCox"):
                times_train, events_train = get_target(df_train, time_col, event_col)
                model = None
            else:
                model_class = model_dict[model_name]
                labtrans = get_labtrans(model_class, 10) if model_name in [
                    "LH", "PMF", "DeepHS", "PCHazard", "MTLR"
                ] else None

                if labtrans is not None:
                    y_train = labtrans.fit_transform(*get_target(df_train, time_col, event_col))
                    y_val = labtrans.transform(*get_target(df_val, time_col, event_col))
                    model = get_model(model_name, x_train.shape[1], labtrans.out_features, labtrans)
                else:
                    times_train, events_train = get_target(df_train, time_col, event_col)
                    times_val, events_val = get_target(df_val, time_col, event_col)
                    y_train = (times_train, events_train)
                    y_val = (times_val, events_val)
                    model = get_model(model_name, x_train.shape[1])

            if model_name == "RSF":
                model = RandomSurvivalForest(
                    n_estimators=200,
                    min_samples_split=10,
                    min_samples_leaf=15,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=seed,
                )
                y_train_rsf = np.array(
                    [(bool(e), t) for e, t in zip(df_train[event_col], df_train[time_col])],
                    dtype=[(event_col, "bool"), (time_col, "float")],
                )
                model.fit(x_train, y_train_rsf)
            elif model_name == "ENCox":
                model, selection = fit_coxnet_survival(
                    x_train,
                    times_train,
                    events_train,
                    l1_ratios=L1_RATIOS,
                    cv_folds=CV_FOLDS,
                    random_state=seed,
                )
                encox_rows.append({
                    "scenario": scenario_name,
                    "training_dataset": training_dataset,
                    "seed": seed,
                    **selection,
                })
                print(
                    f"Selected ENCox: l1_ratio={selection['l1_ratio']}, "
                    f"alpha={selection['alpha']:.6g}, CV C-index={selection['cv_score']:.4f}"
                )
            else:
                callbacks = [tt.cb.EarlyStopping()]
                model.fit(
                    x_train,
                    y_train,
                    batch_size=256,
                    epochs=100,
                    callbacks=callbacks,
                    val_data=(x_val, y_val),
                )

            if run_ind:
                # -------------------------------------------------------------
                # In-distribution evaluation: held-out split of the training
                # cohort. This uses x_test returned by preprocess_dataset(), so
                # preprocessing parameters are fitted on the training split only.
                # -------------------------------------------------------------
                try:
                    metrics_in = evaluate_model_sksurv(
                        model, x_test, times_test_in, events_test_in, model_name, feature_names
                    )
                    c_index_in = float(metrics_in["c_index"][0])
                    df_results_in = metrics_in["df_results"]

                    # Include the split seed in the identifier because the held-
                    # out patient subset changes from seed to seed. The original
                    # dataframe index is retained inside the ID so methods run on
                    # the same seed can still be matched patient-by-patient.
                    patient_ids_in = np.asarray([
                        f"{training_dataset}_seed{seed}_row{int(idx)}"
                        for idx in df_test["__patient_row_id"].to_numpy()
                    ], dtype=object)

                    df_compact_in = compact_prediction_frame(
                        df_results_in["time"],
                        df_results_in["event"],
                        df_results_in["risk_score"],
                        "risk_score",
                        patient_ids=patient_ids_in,
                    )
                    pred_path_in = prediction_dir / (
                        f"{training_dataset}_{model_name}_seed{seed}_predict.csv"
                    )
                    df_compact_in.to_csv(pred_path_in, index=False)
                    print(
                        f"{training_dataset} [InD]: C-index={c_index_in:.4f} "
                        f"-> {pred_path_in}"
                    )

                    all_results.append({
                        "scenario": scenario_name,
                        "training_dataset": training_dataset,
                        "dataset": training_dataset,
                        "setting": "InD",
                        "model": model_name,
                        "seed": seed,
                        "C-index": c_index_in,
                    })
                except Exception as exc:
                    print(
                        f"[ERROR] InD {training_dataset}, {model_name}, seed={seed}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    all_results.append({
                        "scenario": scenario_name,
                        "training_dataset": training_dataset,
                        "dataset": training_dataset,
                        "setting": "InD",
                        "model": model_name,
                        "seed": seed,
                        "C-index": np.nan,
                        "error": f"{type(exc).__name__}: {exc}",
                    })

            if run_ood:
                # -------------------------------------------------------------
                # Out-of-distribution evaluation on the external cohorts.
                # -------------------------------------------------------------
                for dataset_name in testing_datasets:
                    try:
                        df_out, out_standardize, out_leave, out_time_col, out_event_col, out_feature_names = load_datafile_gene(dataset_name)

                        # Preserve the baseline script's existing OOD preprocessing behaviour.
                        # The scenario refactor changes orchestration/output only, not the baseline method.
                        df_test_out, X_test_out, _ = preprocess_dataset_test(
                            df_out, out_standardize, out_leave, out_time_col, out_event_col
                        )
                        times_test, events_test = get_target(df_test_out, out_time_col, out_event_col)

                        metrics = evaluate_model_sksurv(
                            model, X_test_out, times_test, events_test, model_name, out_feature_names
                        )
                        c_index = float(metrics["c_index"][0])
                        df_results = metrics["df_results"]

                        # Save only the columns required for later evaluation.
                        # The full feature matrix remains in memory but is not written to disk.
                        df_compact = compact_prediction_frame(
                            df_results["time"],
                            df_results["event"],
                            df_results["risk_score"],
                            "risk_score",
                        )
                        pred_path = prediction_dir / f"{dataset_name}_{model_name}_seed{seed}_predict.csv"
                        df_compact.to_csv(pred_path, index=False)
                        print(f"{dataset_name}: C-index={c_index:.4f} -> {pred_path}")

                        all_results.append({
                            "scenario": scenario_name,
                            "training_dataset": training_dataset,
                            "dataset": dataset_name,
                            "setting": "OOD",
                            "model": model_name,
                            "seed": seed,
                            "C-index": c_index,
                        })
                    except Exception as exc:
                        print(f"[ERROR] {dataset_name}, {model_name}, seed={seed}: {type(exc).__name__}: {exc}")
                        all_results.append({
                            "scenario": scenario_name,
                            "training_dataset": training_dataset,
                            "dataset": dataset_name,
                            "setting": "OOD",
                            "model": model_name,
                            "seed": seed,
                            "C-index": np.nan,
                            "error": f"{type(exc).__name__}: {exc}",
                        })

    results_df = pd.DataFrame(all_results)
    raw_path = results_dir / "baselines_metrics_all_seeds.csv"
    results_df.to_csv(raw_path, index=False)

    if not results_df.empty:
        summary_df = (
            results_df.groupby(["setting", "model", "dataset"], sort=False)["C-index"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_C-index", "std": "std_C-index", "count": "n_seeds"})
        )
        summary_df["std_C-index"] = summary_df["std_C-index"].fillna(0.0)
        summary_df["C-index (mean ± std)"] = [
            _fmt(m, s) for m, s in zip(summary_df["mean_C-index"], summary_df["std_C-index"])
        ]
    else:
        summary_df = pd.DataFrame(columns=[
            "setting", "model", "dataset", "mean_C-index", "std_C-index", "n_seeds", "C-index (mean ± std)"
        ])

    summary_path = results_dir / "baselines_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # if encox_rows:
    #     pd.DataFrame(encox_rows).to_csv(results_dir / "ENCox_hyperparameters_all_seeds.csv", index=False)

    print(f"\nSaved raw metrics: {raw_path}")
    print(f"Saved summary:     {summary_path}")
    if not summary_df.empty:
        print(summary_df[["setting", "dataset", "model", "n_seeds", "C-index (mean ± std)"]].to_string(index=False))
    return summary_df


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["RFS", "DMFS", "both"], default="both")
    p.add_argument("--setting", choices=["InD", "OOD", "both"], default="both",
                   help="Evaluation setting to run. Baselines fit the same 50% source-training split per seed and can report InD, OOD, or both.")
    p.add_argument(
        "--models",
        nargs="+",
        choices=BASELINE_MODELS,
        default=BASELINE_MODELS,
        help="Baseline models to run. Default: all baseline models.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    for scenario_name in selected_scenarios(args.scenario):
        run_scenario(scenario_name, args.models, setting=args.setting)


if __name__ == "__main__":
    main()
