#!/usr/bin/env python3
"""TabSurv treatment-recommendation experiment using the shared master config.

This is the master-runner-compatible form of the existing TabSurv REC
sensitivity experiment. The treatment-recommendation algorithm is unchanged:
RSF provides a continuous risk-score target, TabPFN/TabSurv learns that target,
and counterfactual treatment plans are evaluated on the held-out test set.

By default the experiment uses the shared five seeds from experiment_config.py:
    [40, 41, 42, 43, 44]

Examples
--------
    python tabsurv_REC.py
    python tabsurv_REC.py --seeds 42 43 44 45 46
    python tabsurv_REC.py --scenario 72genes
    python tabsurv_REC.py --test-size 0.3
    python tabsurv_REC.py --full-reseed
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.ensemble import RandomSurvivalForest
from tabpfn import TabPFNRegressor

from datasets import load_datafile_treatment, preprocess_dataset_rec, get_target
from experiment_config import (
    SEEDS,
    REC_TEST_SIZE,
    REC_MODEL_RANDOM_STATE,
    REC_RESULTS_DIR,
    selected_rec_scenarios,
    compact_prediction_frame,
    ensure_rec_output_dirs,
    format_mean_std,
    tabpfn_regressor_kwargs,
)
from utils import (
    recommend_treatment,
    compare_recommendations,
    manual_c_index_risk_score,
    survival_curves,
    mean_survival_time_km,
)

METHOD_NAME = "TabSurv"
RISK_BASE_MODEL = "RSF"


def _as_float(value):
    return float(value) if value is not None and np.isfinite(value) else np.nan


def run_one_seed(
    dataset_name: str,
    split_seed: int,
    prediction_dir: Path,
    plots_dir: Path,
    test_size: float,
    full_reseed: bool = False,
) -> dict:
    """Run one treatment-recommendation split for TabSurv."""
    model_seed = split_seed if full_reseed else REC_MODEL_RANDOM_STATE

    np.random.seed(split_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    (
        df,
        cols_standardize,
        cols_leave,
        duration_col,
        event_col,
        feature_names,
        treatments,
    ) = load_datafile_treatment(dataset_name)

    df_train, df_val, df_test, x_train, x_val, x_test = preprocess_dataset_rec(
        df,
        cols_standardize,
        cols_leave,
        duration_col,
        event_col,
        test_size,
        split_seed,
    )

    durations_train, events_train = get_target(df_train, duration_col, event_col)
    durations_test, events_test = get_target(df_test, duration_col, event_col)

    # ------------------------------------------------------------------
    # Stage 1: RSF risk target for TabSurv.
    # ------------------------------------------------------------------
    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        random_state=model_seed,
    )
    y_train_rsf = np.array(
        [(bool(e), t) for e, t in zip(df_train[event_col], df_train[duration_col])],
        dtype=[(event_col, "bool"), (duration_col, "float")],
    )
    rsf.fit(x_train, y_train_rsf)
    risk_scores_train = rsf.predict(x_train)

    x_train_df = (
        x_train.copy()
        if isinstance(x_train, pd.DataFrame)
        else pd.DataFrame(x_train, columns=feature_names)
    )
    x_test_df = (
        x_test.copy()
        if isinstance(x_test, pd.DataFrame)
        else pd.DataFrame(x_test, columns=feature_names)
    )

    # ------------------------------------------------------------------
    # Stage 2: TabSurv learns the continuous RSF risk score.
    # ------------------------------------------------------------------
    model_name = METHOD_NAME
    model_rec = TabPFNRegressor(
        ignore_pretraining_limits=True,
        random_state=model_seed,
        **tabpfn_regressor_kwargs(),
    )
    model_rec.fit(x_train_df, pd.Series(risk_scores_train, index=x_train_df.index))

    # ------------------------------------------------------------------
    # Counterfactual recommendation and test-set evaluation.
    # ------------------------------------------------------------------
    rec_df = recommend_treatment(model_rec, x_test_df, treatments)
    combined_df, _, _ = compare_recommendations(rec_df, x_test_df, treatments)
    combined_df["time"] = np.asarray(durations_test, dtype=float)
    combined_df["event"] = np.asarray(events_test, dtype=int)
    combined_df["risk_score"] = np.asarray(model_rec.predict(x_test_df), dtype=float)

    c_index_rec = manual_c_index_risk_score(
        combined_df,
        time_col="time",
        event_col="event",
        prediction_col="risk_score",
    )

    rec_tp_counts = combined_df["REC_TP"].value_counts().to_dict()
    current_tp_counts = combined_df["CURRENT_TP"].value_counts().to_dict()

    plot_path = plots_dir / f"{dataset_name}_{METHOD_NAME}_seed{split_seed}_risk_KM_plot.png"
    p_value = survival_curves(
        combined_df,
        "time",
        "event",
        f"{METHOD_NAME} (seed={split_seed})",
        str(plot_path),
    )

    follow_df = combined_df[combined_df["FOLLOW_REC"] == True]
    not_follow_df = combined_df[combined_df["FOLLOW_REC"] == False]

    mean_surv_follow = (
        mean_survival_time_km(follow_df["time"], follow_df["event"])
        if len(follow_df) > 0
        else np.nan
    )
    mean_surv_not_follow = (
        mean_survival_time_km(not_follow_df["time"], not_follow_df["event"])
        if len(not_follow_df) > 0
        else np.nan
    )

    # Compact test prediction/recommendation output. Use original dataframe
    # indices as patient_id so IDs remain stable when the split seed changes.
    pred_df = compact_prediction_frame(
        durations_test,
        events_test,
        combined_df["risk_score"].to_numpy(),
        "risk_score",
        patient_ids=df_test.index.to_numpy(),
    )
    pred_df["REC_TP"] = combined_df["REC_TP"].to_numpy()
    pred_df["CURRENT_TP"] = combined_df["CURRENT_TP"].to_numpy()
    pred_df["FOLLOW_REC"] = combined_df["FOLLOW_REC"].astype(int).to_numpy()

    pred_path = prediction_dir / f"{dataset_name}_{METHOD_NAME}_seed{split_seed}_predict.csv"
    pred_df.to_csv(pred_path, index=False)

    return {
        "seed": split_seed,
        "dataset": dataset_name,
        "model": METHOD_NAME,
        "model_seed": model_seed,
        "test_size": test_size,
        "c_index_rec": _as_float(c_index_rec),
        "p_value": _as_float(p_value),
        "mean_survival_followed": _as_float(mean_surv_follow),
        "mean_survival_not_followed": _as_float(mean_surv_not_follow),
        "delta_mean_survival": _as_float(mean_surv_follow - mean_surv_not_follow),
        "n_followed": int(len(follow_df)),
        "n_not_followed": int(len(not_follow_df)),
        "rec_tp_counts": json.dumps(rec_tp_counts, sort_keys=True),
        "current_tp_counts": json.dumps(current_tp_counts, sort_keys=True),
        "prediction_file": str(pred_path),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenario",
        choices=["72genes", "35genes","30genes", "allgenes", "both"],
        default="both",
        help="REC gene-set scenario. Default: both.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=f"Split seeds. Default: shared SEEDS={SEEDS}.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Optional custom dataset override for a single REC scenario. Normally omit this "
            "and use --scenario so 72genes/allgenes come from experiment_config.py."
        ),
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=REC_TEST_SIZE,
        help=f"Held-out test fraction. Default from experiment_config: {REC_TEST_SIZE}.",
    )
    p.add_argument(
        "--full-reseed",
        action="store_true",
        help=(
            "Also use each split seed as the RSF/TabPFN model seed. By default, "
            f"model random_state stays fixed at {REC_MODEL_RANDOM_STATE}, preserving "
            "the original split-sensitivity design."
        ),
    )
    return p.parse_args()


def _run_rec_scenario(rec_scenario, seeds, test_size, full_reseed, datasets_override=None):
    dirs = ensure_rec_output_dirs(rec_scenario, "TabSurv_REC")
    prediction_dir = dirs["prediction_dir"]
    results_dir = dirs["results_dir"]
    plots_dir = dirs["plots_dir"]
    datasets = list(datasets_override) if datasets_override is not None else list(dirs["datasets"])

    metadata = {
        "method": METHOD_NAME,
        "rec_scenario": rec_scenario,
        "datasets": datasets,
        "seeds": seeds,
        "test_size": test_size,
        "model_random_state": REC_MODEL_RANDOM_STATE,
        "full_reseed": full_reseed,
        "risk_base_model": RISK_BASE_MODEL,
    }
    (results_dir / "tabsurv_REC_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print(f"TabSurv treatment recommendation | scenario={rec_scenario}")
    print(f"datasets  : {datasets}")
    print(f"seeds     : {seeds}")
    print(f"test_size : {test_size}")
    print(f"full_reseed: {full_reseed}")
    print("=" * 78)

    rows = []
    for dataset_name in datasets:
        for seed in seeds:
            print(f"\n=== TabSurv_REC | scenario={rec_scenario} | dataset={dataset_name} | seed={seed} ===")
            try:
                row = run_one_seed(
                    dataset_name,
                    seed,
                    prediction_dir,
                    plots_dir,
                    test_size=test_size,
                    full_reseed=full_reseed,
                )
                row["rec_scenario"] = rec_scenario
            except Exception as exc:
                print(f"[ERROR] {dataset_name}, seed={seed}: {type(exc).__name__}: {exc}")
                rows.append({
                    "rec_scenario": rec_scenario,
                    "seed": seed,
                    "dataset": dataset_name,
                    "model": METHOD_NAME,
                    "model_seed": seed if full_reseed else REC_MODEL_RANDOM_STATE,
                    "test_size": test_size,
                    "c_index_rec": np.nan,
                    "p_value": np.nan,
                    "mean_survival_followed": np.nan,
                    "mean_survival_not_followed": np.nan,
                    "delta_mean_survival": np.nan,
                    "n_followed": np.nan,
                    "n_not_followed": np.nan,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue

            rows.append(row)
            print(
                f"C-index={row['c_index_rec']:.4f}, p={row['p_value']:.4g}, "
                f"Delta mean survival={row['delta_mean_survival']:.3f} years"
            )

    raw_df = pd.DataFrame(rows)
    raw_path = results_dir / "tabsurv_REC_all_seeds.csv"
    raw_df.to_csv(raw_path, index=False)

    valid_df = raw_df[np.isfinite(pd.to_numeric(raw_df["delta_mean_survival"], errors="coerce"))].copy()
    summary_rows = []
    for dataset_name, group in valid_df.groupby("dataset", sort=False):
        delta = group["delta_mean_survival"].astype(float)
        cidx = group["c_index_rec"].astype(float)
        delta_mean = float(delta.mean())
        delta_std = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
        cidx_mean = float(cidx.mean())
        cidx_std = float(cidx.std(ddof=1)) if len(cidx) > 1 else 0.0
        summary_rows.append({
            "rec_scenario": rec_scenario,
            "dataset": dataset_name,
            "model": METHOD_NAME,
            "n_seeds": int(len(group)),
            "mean_C-index": cidx_mean,
            "std_C-index": cidx_std,
            "C-index (mean ± std)": format_mean_std(cidx_mean, cidx_std),
            "mean_delta_mean_survival": delta_mean,
            "std_delta_mean_survival": delta_std,
            "Delta mean survival (mean ± std)": format_mean_std(delta_mean, delta_std, ndigits=3),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / "tabsurv_REC_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved raw REC results: {raw_path}")
    print(f"Saved REC summary:     {summary_path}")
    if not summary_df.empty:
        print("\nTabSurv REC summary:")
        print(summary_df.to_string(index=False))
    return raw_df, summary_df


def main():
    args = parse_args()
    seeds = list(SEEDS if args.seeds is None else args.seeds)
    scenarios = selected_rec_scenarios(args.scenario)

    if not 0.0 < args.test_size < 1.0:
        raise SystemExit("--test-size must be between 0 and 1.")
    if not seeds:
        raise SystemExit("At least one seed is required.")
    if args.datasets is not None and len(scenarios) != 1:
        raise SystemExit("--datasets can only be used with a single --scenario (72genes or allgenes).")

    all_raw = []
    all_summary = []
    for rec_scenario in scenarios:
        raw, summary = _run_rec_scenario(
            rec_scenario,
            seeds,
            args.test_size,
            args.full_reseed,
            datasets_override=args.datasets,
        )
        all_raw.append(raw)
        all_summary.append(summary)

    if len(scenarios) > 1:
        REC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        combined_raw = pd.concat(all_raw, ignore_index=True, sort=False)
        combined_summary = pd.concat(all_summary, ignore_index=True, sort=False)
        combined_raw_path = REC_RESULTS_DIR / "tabsurv_REC_all_scenarios_all_seeds.csv"
        combined_summary_path = REC_RESULTS_DIR / "tabsurv_REC_all_scenarios_summary.csv"
        combined_raw.to_csv(combined_raw_path, index=False)
        combined_summary.to_csv(combined_summary_path, index=False)
        print(f"\nSaved combined TabSurv REC results: {combined_raw_path}")
        print(f"Saved combined TabSurv REC summary: {combined_summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
