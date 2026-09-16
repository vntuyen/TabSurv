#!/usr/bin/env python3
"""Evaluate RFS/DMFS predictions for InD and OOD final reporting.

Reporting rules
---------------
InD:
    - held-out training cohort only (METABRIC for RFS, NKI for DMFS)
    - TabSurv_M + TabSurv_P + TabSurv_A + eight baselines
    - all TabSurv variants use test_size=0.5 for the InD branch
    - report C-index mean +/- SD across the five matched seeds/splits
    - NO Stability_CI is reported, because stability is defined across cohorts

OOD:
    - configured external cohorts
    - TabSurv_M + TabSurv_P + TabSurv_A + eight baselines
    - per-cohort C-index is first averaged across seeds
    - Stability_CI is then computed ACROSS COHORTS:
          Stability_CI = mean(cohort mean C-index) - SD(cohort mean C-index)

The paired statistical comparison remains TabSurv_M versus the eight baselines;
TabSurv_P and TabSurv_A are descriptive variants/ablations in both InD and OOD
reporting tables and are not added to the Wilcoxon comparison family.
"""

import argparse

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

from experiment_config import (
    ALL_EVALUATION_MODELS,
    BASELINE_MODELS,
    IND_EVALUATION_MODELS,
    IND_REPORT_MODELS,
    OOD_REPORT_MODELS,
    PROPOSED_METHOD,
    OUTPUT_ROOT,
    SEEDS,
    ensure_output_dirs,
    selected_scenarios,
    selected_settings,
)

EXPECTED_TIME_MODELS = {"TabSurv_M", "TabSurv_P", "TabSurv_A"}

EVALUATION_COLUMNS = {
    "patient_id", "time", "real_survival", "event", "event_observed",
    "predicted", "predicted_survival", "risk_score", "predicted_risk_score",
}


def compute_c_index(times, events, risk_scores):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int).astype(bool)
    risk_scores = np.asarray(risk_scores, dtype=float)
    finite = np.isfinite(times) & np.isfinite(risk_scores)
    if finite.sum() < 2:
        return np.nan
    return float(concordance_index_censored(events[finite], times[finite], risk_scores[finite])[0])


def _read_prediction_file(file_path):
    """Read only evaluation columns, including from older feature-heavy CSVs."""
    columns = pd.read_csv(file_path, nrows=0).columns.tolist()
    usecols = [c for c in columns if c in EVALUATION_COLUMNS]
    if not usecols:
        raise KeyError(f"{file_path} contains none of the recognised evaluation columns.")
    return pd.read_csv(file_path, usecols=usecols)


def _harmonise_prediction_file(df, model_name, file_path):
    df = df.copy()
    if "time" not in df.columns and "real_survival" in df.columns:
        df.rename(columns={"real_survival": "time"}, inplace=True)
    if "event" not in df.columns and "event_observed" in df.columns:
        df.rename(columns={"event_observed": "event"}, inplace=True)

    missing_targets = [c for c in ("time", "event") if c not in df.columns]
    if missing_targets:
        raise KeyError(f"{file_path} missing required column(s): {missing_targets}")

    if model_name in EXPECTED_TIME_MODELS:
        if "predicted" in df.columns:
            df["risk_score_eval"] = -pd.to_numeric(df["predicted"], errors="coerce")
        elif "predicted_survival" in df.columns:
            df["risk_score_eval"] = -pd.to_numeric(df["predicted_survival"], errors="coerce")
        elif "risk_score" in df.columns:
            df["risk_score_eval"] = pd.to_numeric(df["risk_score"], errors="coerce")
        else:
            raise KeyError(
                f"{file_path} must contain 'predicted', 'predicted_survival', "
                f"or 'risk_score' for {model_name}."
            )
    else:
        if "risk_score" in df.columns:
            df["risk_score_eval"] = pd.to_numeric(df["risk_score"], errors="coerce")
        elif "predicted_risk_score" in df.columns:
            df["risk_score_eval"] = pd.to_numeric(df["predicted_risk_score"], errors="coerce")
        elif "predicted" in df.columns:
            df["risk_score_eval"] = pd.to_numeric(df["predicted"], errors="coerce")
        else:
            raise KeyError(
                f"{file_path} must contain 'risk_score', 'predicted_risk_score', or 'predicted'."
            )
    return df


def _fmt(mean, std):
    if mean is None or not np.isfinite(mean):
        return "NaN"
    if std is None or not np.isfinite(std):
        std = 0.0
    return f"{mean:.4f} ± {std:.4f}"


def _check_ind_patient_pairing(prediction_dir, training_dataset, models):
    """Check that all InD methods use the same held-out patients as TabSurv_M."""
    rows = []
    if PROPOSED_METHOD not in models:
        return pd.DataFrame(rows)

    for seed in SEEDS:
        prop_path = prediction_dir / f"{training_dataset}_{PROPOSED_METHOD}_seed{seed}_predict.csv"
        if not prop_path.exists():
            continue
        prop = _read_prediction_file(prop_path)
        if "patient_id" not in prop.columns:
            continue
        prop_ids = set(prop["patient_id"].astype(str))

        for model in models:
            if model == PROPOSED_METHOD or model not in IND_EVALUATION_MODELS:
                continue
            comp_path = prediction_dir / f"{training_dataset}_{model}_seed{seed}_predict.csv"
            if not comp_path.exists():
                continue
            comp = _read_prediction_file(comp_path)
            if "patient_id" not in comp.columns:
                rows.append({
                    "dataset": training_dataset, "seed": seed,
                    "proposed_method": PROPOSED_METHOD, "comparator": model,
                    "n_proposed_patients": len(prop_ids), "n_comparator_patients": np.nan,
                    "n_common_patients": np.nan, "same_patient_set": False,
                    "note": "comparator file has no patient_id",
                })
                continue
            comp_ids = set(comp["patient_id"].astype(str))
            rows.append({
                "dataset": training_dataset, "seed": seed,
                "proposed_method": PROPOSED_METHOD, "comparator": model,
                "n_proposed_patients": len(prop_ids), "n_comparator_patients": len(comp_ids),
                "n_common_patients": len(prop_ids & comp_ids),
                "same_patient_set": prop_ids == comp_ids,
                "note": "" if prop_ids == comp_ids else "rerun with matched InD split before final reporting",
            })
    return pd.DataFrame(rows)


def _build_setting_report(raw_df, summary_df, scenario_name, training_dataset, testing_datasets, models, settings):
    """Build final setting-level report with no InD stability and cohort-based OOD stability."""
    rows = []
    settings = set(settings)

    # ---------------- InD: one held-out cohort, variability only across seeds ----------------
    if "InD" in settings:
        ind_models = [m for m in IND_REPORT_MODELS if m in models]
        ind_raw = raw_df[(raw_df["setting"] == "InD") & (raw_df["model"].isin(ind_models))]
        for model in ind_models:
            vals = ind_raw.loc[ind_raw["model"] == model, "C-index"].astype(float).to_numpy()
            vals = vals[np.isfinite(vals)]
            mean_c = float(np.mean(vals)) if vals.size else np.nan
            std_c = float(np.std(vals, ddof=1)) if vals.size > 1 else (0.0 if vals.size == 1 else np.nan)
            rows.append({
                "scenario": scenario_name,
                "setting": "InD",
                "model": model,
                "n_datasets": 1 if vals.size else 0,
                "n_obs": int(vals.size),
                "expected_n_obs": len(SEEDS),
                "mean_C-index": mean_c,
                "std_C-index": std_c,
                "C-index (mean ± std)": _fmt(mean_c, std_c),
                "variability_basis": "seeds",
                "Stability_CI": np.nan,
                "Stability_note": "N/A for InD: stability is defined across cohorts",
                "complete": int(vals.size) == len(SEEDS),
            })

    # ---------------- OOD: first average seeds per cohort, then assess across cohorts ---------
    if "OOD" in settings:
        ood_models = [m for m in OOD_REPORT_MODELS if m in models]
        ood_summary = summary_df[
            (summary_df["setting"] == "OOD") & (summary_df["model"].isin(ood_models))
        ].copy()

        for model in ood_models:
            model_ds = ood_summary[ood_summary["model"] == model].copy()
            cohort_means = pd.to_numeric(model_ds["mean_C-index"], errors="coerce").to_numpy(dtype=float)
            cohort_means = cohort_means[np.isfinite(cohort_means)]
            mean_c = float(np.mean(cohort_means)) if cohort_means.size else np.nan
            std_c = (
                float(np.std(cohort_means, ddof=1)) if cohort_means.size > 1
                else (0.0 if cohort_means.size == 1 else np.nan)
            )
            n_obs = int(raw_df[(raw_df["setting"] == "OOD") & (raw_df["model"] == model)]["C-index"].notna().sum())
            expected_n_obs = len(testing_datasets) * len(SEEDS)
            rows.append({
                "scenario": scenario_name,
                "setting": "OOD",
                "model": model,
                "n_datasets": int(cohort_means.size),
                "n_obs": n_obs,
                "expected_n_obs": expected_n_obs,
                "mean_C-index": mean_c,
                "std_C-index": std_c,
                "C-index (mean ± std)": _fmt(mean_c, std_c),
                "variability_basis": "cohorts (after averaging seeds within cohort)",
                "Stability_CI": mean_c - std_c if np.isfinite(mean_c) and np.isfinite(std_c) else np.nan,
                "Stability_note": "mean cohort C-index - SD across cohort mean C-indices",
                "complete": int(cohort_means.size) == len(testing_datasets) and n_obs == expected_n_obs,
            })

    report = pd.DataFrame(rows)
    if report.empty:
        return report

    # Ranking is setting-specific. Both settings now include TabSurv_M/P/A + baselines.
    report["rank_by_mean_C_index"] = (
        report.groupby("setting")["mean_C-index"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    model_order = list(dict.fromkeys([*OOD_REPORT_MODELS]))
    report["setting"] = pd.Categorical(report["setting"], ["InD", "OOD"], ordered=True)
    report["model"] = pd.Categorical(report["model"], model_order, ordered=True)
    report = report.sort_values(["setting", "model"]).reset_index(drop=True)
    report[["setting", "model"]] = report[["setting", "model"]].astype(str)
    return report


def evaluate_scenario(scenario_name, models=None, setting="both"):
    settings = selected_settings(setting)
    cfg = ensure_output_dirs(scenario_name)
    training_dataset = cfg["training_dataset"]
    testing_datasets = list(cfg["testing_datasets"])
    prediction_dir = cfg["prediction_dir"]
    results_dir = cfg["results_dir"]
    models = list(models or ALL_EVALUATION_MODELS)

    print("\n" + "=" * 88)
    print(f"Evaluation | scenario={scenario_name}")
    print(f"settings={settings}")
    if "InD" in settings:
        print(f"InD dataset={training_dataset} | Stability_CI: NOT APPLICABLE")
    if "OOD" in settings:
        print(f"OOD datasets={testing_datasets} | Stability_CI: across cohort mean C-indices")
    print(f"models={models}")
    print(f"seeds={SEEDS}")
    print("=" * 88)

    rows, missing, errors = [], [], []

    # InD and OOD both evaluate all requested configured models. TabSurv_A/P
    # now save seed-specific held-out source-cohort predictions as well.
    tasks = []
    for model_name in models:
        if "InD" in settings and model_name in IND_EVALUATION_MODELS:
            tasks.append(("InD", training_dataset, model_name))
        if "OOD" in settings:
            for dataset in testing_datasets:
                tasks.append(("OOD", dataset, model_name))

    for setting, dataset, model_name in tasks:
        for seed in SEEDS:
            file_path = prediction_dir / f"{dataset}_{model_name}_seed{seed}_predict.csv"
            if not file_path.exists():
                missing.append({
                    "scenario": scenario_name, "setting": setting, "dataset": dataset,
                    "model": model_name, "seed": seed, "prediction_file": str(file_path),
                })
                continue
            try:
                df = _read_prediction_file(file_path)
                df = _harmonise_prediction_file(df, model_name, file_path)
                c_index = compute_c_index(
                    pd.to_numeric(df["time"], errors="coerce").to_numpy(),
                    pd.to_numeric(df["event"], errors="coerce").fillna(0).to_numpy(dtype=int),
                    df["risk_score_eval"].to_numpy(dtype=float),
                )
                rows.append({
                    "scenario": scenario_name, "setting": setting,
                    "training_dataset": training_dataset, "dataset": dataset,
                    "model": model_name, "seed": seed, "C-index": c_index,
                    "prediction_file": str(file_path),
                })
                print(f"{setting:3s} {dataset:10s} {model_name:10s} seed={seed}: {c_index:.4f}")
            except Exception as exc:
                errors.append({
                    "scenario": scenario_name, "setting": setting, "dataset": dataset,
                    "model": model_name, "seed": seed, "prediction_file": str(file_path),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"[ERROR] {setting} {file_path}: {type(exc).__name__}: {exc}")

    raw_df = pd.DataFrame(rows)
    raw_path = results_dir / "evaluation_all_methods_all_seeds.csv"
    raw_df.to_csv(raw_path, index=False)

    if raw_df.empty:
        summary_df = pd.DataFrame()
        pivot_df = pd.DataFrame()
        setting_report_df = pd.DataFrame()
        stability_df = pd.DataFrame()
    else:
        summary_df = (
            raw_df.groupby(["setting", "dataset", "model"], sort=False)["C-index"]
            .agg(mean_C_index="mean", std_C_index="std", n_seeds="count")
            .reset_index()
            .rename(columns={"mean_C_index": "mean_C-index", "std_C_index": "std_C-index"})
        )
        summary_df["std_C-index"] = summary_df["std_C-index"].fillna(0.0)
        summary_df["C-index (mean ± std)"] = [
            _fmt(m, s) for m, s in zip(summary_df["mean_C-index"], summary_df["std_C-index"])
        ]
        dataset_order = [training_dataset, *testing_datasets]
        summary_df["setting"] = pd.Categorical(summary_df["setting"], ["InD", "OOD"], ordered=True)
        summary_df["dataset"] = pd.Categorical(summary_df["dataset"], dataset_order, ordered=True)
        summary_df["model"] = pd.Categorical(summary_df["model"], models, ordered=True)
        summary_df = summary_df.sort_values(["setting", "dataset", "model"]).reset_index(drop=True)
        summary_df[["setting", "dataset", "model"]] = summary_df[["setting", "dataset", "model"]].astype(str)

        pivot_df = summary_df.pivot_table(
            index=["setting", "dataset"], columns="model",
            values="C-index (mean ± std)", aggfunc="first", observed=False,
        ).reindex(columns=models).reset_index()

        setting_report_df = _build_setting_report(
            raw_df, summary_df, scenario_name, training_dataset, testing_datasets, models, settings
        )
        stability_df = setting_report_df[setting_report_df["setting"] == "OOD"].copy()

    # Verify matched InD patient sets only when InD is part of this evaluation.
    pairing_path = results_dir / "evaluation_ind_patient_pairing_check.csv"
    if "InD" in settings:
        pairing_df = _check_ind_patient_pairing(prediction_dir, training_dataset, models)
        pairing_df.to_csv(pairing_path, index=False)
        if not pairing_df.empty and (~pairing_df["same_patient_set"].fillna(False)).any():
            bad = pairing_df[~pairing_df["same_patient_set"].fillna(False)]
            print("\n[WARNING] Some InD proposed-vs-baseline files do not contain the same patient set:")
            print(bad[["seed", "comparator", "n_proposed_patients", "n_comparator_patients", "n_common_patients", "note"]].to_string(index=False))
    else:
        pairing_df = pd.DataFrame()

    summary_path = results_dir / "evaluation_all_methods_summary_mean_std.csv"
    pivot_path = results_dir / "evaluation_all_methods_pivot_mean_std.csv"
    report_path = results_dir / f"{scenario_name}_final_report_InD_OOD.csv"
    report_pivot_path = results_dir / f"{scenario_name}_final_report_InD_OOD_pivot.csv"
    stability_path = results_dir / f"{scenario_name}_OOD_stability_score.csv"
    legacy_stability_path = results_dir / f"{scenario_name}_stability_score_by_setting.csv"

    summary_df.to_csv(summary_path, index=False)
    pivot_df.to_csv(pivot_path, index=False)
    setting_report_df.to_csv(report_path, index=False)
    stability_df.to_csv(stability_path, index=False)
    # Backward-compatible filename, but now intentionally OOD-only.
    stability_df.to_csv(legacy_stability_path, index=False)

    if setting_report_df.empty:
        report_pivot_df = pd.DataFrame()
    else:
        report_pivot_df = setting_report_df.pivot(
            index="model", columns="setting", values="C-index (mean ± std)"
        ).reindex(OOD_REPORT_MODELS).reset_index().rename(columns={
            "InD": "InD C-index (mean ± std across seeds)",
            "OOD": "OOD C-index (mean ± std across cohorts)",
        })
        # Add OOD stability as a separate column. InD intentionally has no
        # stability metric because stability is defined across cohorts.
        stab_map = stability_df.set_index("model")["Stability_CI"] if not stability_df.empty else pd.Series(dtype=float)
        report_pivot_df["OOD Stability_CI"] = report_pivot_df["model"].map(stab_map)
    report_pivot_df.to_csv(report_pivot_path, index=False)

    if missing:
        missing_path = results_dir / "evaluation_missing_prediction_files.csv"
        pd.DataFrame(missing).to_csv(missing_path, index=False)
        print(f"\nMissing prediction files: {len(missing)} (listed in {missing_path})")
    if errors:
        errors_path = results_dir / "evaluation_errors.csv"
        pd.DataFrame(errors).to_csv(errors_path, index=False)
        print(f"Evaluation errors: {len(errors)} (listed in {errors_path})")

    print(f"\nSaved raw evaluation:          {raw_path}")
    print(f"Saved per-dataset summary:      {summary_path}")
    print(f"Saved per-dataset pivot:        {pivot_path}")
    print(f"Saved final InD/OOD report:     {report_path}")
    print(f"Saved final report pivot:       {report_pivot_path}")
    print(f"Saved OOD-only stability table: {stability_path}")
    if "InD" in settings:
        print(f"Saved InD pairing check:        {pairing_path}")

    if not pivot_df.empty:
        print("\nPer-dataset C-index mean ± std across seeds:")
        print(pivot_df.to_string(index=False))
    if not setting_report_df.empty:
        print("\nFinal report summary")
        print("  InD: C-index mean ± SD across seeds; Stability_CI = N/A")
        print("  OOD: C-index mean ± SD across cohort means; Stability_CI = mean - SD across cohorts")
        cols = [
            "setting", "model", "n_datasets", "n_obs", "C-index (mean ± std)",
            "variability_basis", "Stability_CI", "rank_by_mean_C_index", "complete",
        ]
        print(setting_report_df[cols].to_string(index=False))

    return raw_df, summary_df, pivot_df, setting_report_df, stability_df


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["RFS", "DMFS", "both"], default="both")
    p.add_argument("--setting", choices=["InD", "OOD", "both"], default="both",
                   help="Evaluate only InD, only OOD, or both. Existing files from unselected settings are ignored.")
    p.add_argument(
        "--models", nargs="+", choices=ALL_EVALUATION_MODELS,
        default=ALL_EVALUATION_MODELS,
        help=(
            "Models to evaluate. Both InD and OOD can include TabSurv_M, TabSurv_P, "
            "TabSurv_A and all eight baselines. Stability_CI remains OOD-only."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    reports = []
    per_dataset = []

    scenarios = selected_scenarios(args.scenario)
    for scenario_name in scenarios:
        raw_df, summary_df, pivot_df, report_df, stability_df = evaluate_scenario(
            scenario_name, args.models, setting=args.setting
        )
        if not report_df.empty:
            reports.append(report_df.copy())
        if not summary_df.empty:
            # Both settings: M/P/A + baselines; Stability_CI remains OOD-only.
            keep_ind = summary_df[(summary_df["setting"] == "InD") & summary_df["model"].isin(IND_REPORT_MODELS)]
            keep_ood = summary_df[(summary_df["setting"] == "OOD") & summary_df["model"].isin(OOD_REPORT_MODELS)]
            keep = pd.concat([keep_ind, keep_ood], ignore_index=True)
            keep.insert(0, "scenario", scenario_name)
            per_dataset.append(keep)

    if len(scenarios) > 1 and reports:
        out_dir = OUTPUT_ROOT / "results_final_report"
        out_dir.mkdir(parents=True, exist_ok=True)

        combined = pd.concat(reports, ignore_index=True, sort=False)
        combined_path = out_dir / "TabSurv_final_InD_OOD_report.csv"
        combined.to_csv(combined_path, index=False)

        if per_dataset:
            combined_dataset = pd.concat(per_dataset, ignore_index=True, sort=False)
            combined_dataset.to_csv(
                out_dir / "TabSurv_all_dataset_results_InD_OOD.csv", index=False
            )

        display = combined[[
            "scenario", "setting", "model", "n_datasets", "n_obs",
            "C-index (mean ± std)", "Stability_CI", "rank_by_mean_C_index", "complete",
        ]]
        print("\n" + "=" * 108)
        print("Combined final report")
        print("Both settings include TabSurv_M/P/A + 8 baselines; InD has no Stability_CI.")
        print("=" * 108)
        print(display.to_string(index=False))
        print(f"Saved combined final report: {combined_path}")


if __name__ == "__main__":
    main()
