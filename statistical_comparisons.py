#!/usr/bin/env python3
"""Paired Wilcoxon signed-rank comparisons: TabSurv_M vs eight baselines.

Primary analyses
----------------
RFS-InD
    One training cohort (METABRIC), five matched held-out split seeds. The
    paired statistical units are seeds/splits.
RFS-OOD
    Six external cohorts. For each cohort, both methods are first averaged
    over the same common seeds; the paired statistical units are datasets.
DMFS-InD
    One training cohort (NKI), five matched held-out split seeds. The paired
    statistical units are seeds/splits.
DMFS-OOD
    Four external cohorts, paired at dataset level after common-seed averaging.

Secondary analysis
------------------
OVERALL_10_OOD pools the six RFS and four DMFS external cohorts. It is kept
separate because RFS and DMFS are different endpoints/training scenarios.

Holm correction is applied separately within each analysis family across the
pre-specified proposed-vs-baseline comparisons.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiment_config import (
    BASELINE_MODELS,
    PROPOSED_METHOD,
    SCENARIOS,
    SEEDS,
    ensure_output_dirs,
    selected_settings,
)

ALPHA = 0.05
EXPECTED_SEED_SET = set(SEEDS)


def _holm_adjust(p_values, alpha=ALPHA):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for i, p in enumerate(sorted_p):
        candidate = (m - i) * p
        running_max = max(running_max, candidate)
        adjusted_sorted[i] = min(1.0, running_max)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted <= alpha, adjusted


def _safe_wilcoxon(differences):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return np.nan, np.nan, "not_run"
    if np.allclose(differences, 0.0, rtol=0.0, atol=1e-15):
        return 0.0, 1.0, "all_differences_zero"
    try:
        res = wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            correction=False,
            method="auto",
        )
        return float(res.statistic), float(res.pvalue), "auto"
    except TypeError:
        res = wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            correction=False,
        )
        return float(res.statistic), float(res.pvalue), "scipy_default"


def _fmt(x, digits=4):
    if x is None or not np.isfinite(x):
        return "NaN"
    return f"{x:.{digits}f}"


def _load_seed_level_results(scenario_name, baselines, strict_seeds=False, settings=("InD", "OOD")):
    settings = set(settings)
    cfg = ensure_output_dirs(scenario_name)
    raw_path = cfg["results_dir"] / "evaluation_all_methods_all_seeds.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}. Run evaluations.py --scenario {scenario_name} first."
        )

    raw = pd.read_csv(raw_path)
    required = {"setting", "dataset", "model", "seed", "C-index"}
    missing = required.difference(raw.columns)
    if missing:
        raise KeyError(
            f"{raw_path} is missing {sorted(missing)}. Re-run the updated evaluations.py "
            "so InD/OOD settings are recorded explicitly."
        )

    wanted = [PROPOSED_METHOD, *baselines]
    raw = raw[raw["model"].isin(wanted)].copy()
    raw["C-index"] = pd.to_numeric(raw["C-index"], errors="coerce")
    raw["seed"] = pd.to_numeric(raw["seed"], errors="coerce")
    raw = raw[np.isfinite(raw["C-index"]) & raw["seed"].notna()].copy()
    raw["seed"] = raw["seed"].astype(int)
    raw = (
        raw.groupby(["setting", "dataset", "model", "seed"], as_index=False)["C-index"]
        .mean()
    )
    raw["scenario"] = scenario_name

    if strict_seeds:
        expected_rows = []
        training_dataset = SCENARIOS[scenario_name]["training_dataset"]
        for model in wanted:
            if "InD" in settings:
                expected_rows.append(("InD", training_dataset, model))
            if "OOD" in settings:
                for dataset in SCENARIOS[scenario_name]["testing_datasets"]:
                    expected_rows.append(("OOD", dataset, model))
        coverage = (
            raw.groupby(["setting", "dataset", "model"])["seed"]
            .agg(lambda s: set(int(x) for x in s))
            .to_dict()
        )
        bad = []
        for key in expected_rows:
            seeds = coverage.get(key, set())
            if seeds != EXPECTED_SEED_SET:
                bad.append((*key, sorted(seeds)))
        if bad:
            details = "\n".join(
                f"  {setting}/{dataset}/{model}: seeds={seeds}" for setting, dataset, model, seeds in bad
            )
            raise RuntimeError(
                f"{scenario_name}: incomplete seed coverage. Expected {SEEDS} for every final-report pair:\n{details}"
            )

    return raw


def _check_ind_pairing_file(scenario_name, baselines, strict=False):
    cfg = ensure_output_dirs(scenario_name)
    path = cfg["results_dir"] / "evaluation_ind_patient_pairing_check.csv"
    if not path.exists():
        message = f"{scenario_name}: missing InD pairing check {path}. Run evaluations.py first."
        if strict:
            raise RuntimeError(message)
        print("[WARNING] " + message)
        return
    df = pd.read_csv(path)
    if df.empty:
        message = f"{scenario_name}: InD pairing check is empty."
        if strict:
            raise RuntimeError(message)
        print("[WARNING] " + message)
        return
    subset = df[df["comparator"].isin(baselines)].copy()
    if "same_patient_set" in subset.columns:
        same = subset["same_patient_set"].astype(str).str.lower().isin(["true", "1"])
        bad = subset[~same]
    else:
        bad = subset
    if not bad.empty:
        message = (
            f"{scenario_name}: {len(bad)} InD proposed-vs-baseline seed pairs do not "
            "use identical patient sets. Rerun TabSurv_M and baselines with the matched split scripts."
        )
        if strict:
            raise RuntimeError(message)
        print("[WARNING] " + message)


def _apply_holm(results):
    results = results.copy()
    results["holm_adjusted_p"] = np.nan
    results["significant_raw_0.05"] = False
    results["significant_holm_0.05"] = False
    valid = results["p_value"].notna()
    if valid.any():
        reject, p_adj = _holm_adjust(results.loc[valid, "p_value"].to_numpy(dtype=float))
        results.loc[valid, "holm_adjusted_p"] = p_adj
        results.loc[valid, "significant_raw_0.05"] = (
            results.loc[valid, "p_value"].to_numpy(dtype=float) < ALPHA
        )
        results.loc[valid, "significant_holm_0.05"] = reject
    results["effect_direction"] = np.where(
        results["mean_delta_C_index"] > 0,
        f"{PROPOSED_METHOD} higher",
        np.where(results["mean_delta_C_index"] < 0, "Baseline higher", "Tie"),
    )
    return results


def compare_ind(seed_level, scenario_name, baselines):
    """InD: one paired C-index difference per matched split seed."""
    dataset = SCENARIOS[scenario_name]["training_dataset"]
    ind = seed_level[(seed_level["setting"] == "InD") & (seed_level["dataset"] == dataset)]
    rows, pair_rows = [], []

    for baseline in baselines:
        prop = ind[ind["model"] == PROPOSED_METHOD][["seed", "C-index"]].rename(
            columns={"C-index": "proposed_C_index"}
        )
        base = ind[ind["model"] == baseline][["seed", "C-index"]].rename(
            columns={"C-index": "baseline_C_index"}
        )
        paired = prop.merge(base, on="seed", how="inner")
        paired = paired[paired["seed"].isin(SEEDS)].sort_values("seed").copy()
        paired["delta_C_index"] = paired["proposed_C_index"] - paired["baseline_C_index"]
        diffs = paired["delta_C_index"].to_numpy(dtype=float)
        stat, p_value, calc = _safe_wilcoxon(diffs)
        tol = 1e-12

        rows.append({
            "analysis": f"{scenario_name}_InD",
            "scenario": scenario_name,
            "setting": "InD",
            "pairing_unit": "seed/split",
            "dataset": dataset,
            "proposed_method": PROPOSED_METHOD,
            "baseline": baseline,
            "n_pairs": int(len(paired)),
            "pair_ids": ",".join(str(int(x)) for x in paired["seed"]),
            "all_five_seeds": set(paired["seed"].astype(int)) == EXPECTED_SEED_SET,
            "mean_proposed_C_index": float(paired["proposed_C_index"].mean()) if len(paired) else np.nan,
            "mean_baseline_C_index": float(paired["baseline_C_index"].mean()) if len(paired) else np.nan,
            "mean_delta_C_index": float(diffs.mean()) if diffs.size else np.nan,
            "median_delta_C_index": float(np.median(diffs)) if diffs.size else np.nan,
            "wins": int(np.sum(diffs > tol)),
            "ties": int(np.sum(np.abs(diffs) <= tol)),
            "losses": int(np.sum(diffs < -tol)),
            "wilcoxon_statistic": stat,
            "p_value": p_value,
            "wilcoxon_calculation": calc,
        })

        for r in paired.itertuples(index=False):
            pair_rows.append({
                "analysis": f"{scenario_name}_InD",
                "scenario": scenario_name,
                "setting": "InD",
                "pairing_unit": "seed/split",
                "pair_id": int(r.seed),
                "dataset": dataset,
                "seed": int(r.seed),
                "proposed_method": PROPOSED_METHOD,
                "baseline": baseline,
                "proposed_C_index": float(r.proposed_C_index),
                "baseline_C_index": float(r.baseline_C_index),
                "delta_C_index": float(r.delta_C_index),
            })

    return _apply_holm(pd.DataFrame(rows)), pd.DataFrame(pair_rows)


def _common_seed_dataset_pair(seed_level, dataset, baseline):
    prop = seed_level[
        (seed_level["dataset"] == dataset) & (seed_level["model"] == PROPOSED_METHOD)
    ][["seed", "C-index"]].rename(columns={"C-index": "proposed_C_index_seed"})
    base = seed_level[
        (seed_level["dataset"] == dataset) & (seed_level["model"] == baseline)
    ][["seed", "C-index"]].rename(columns={"C-index": "baseline_C_index_seed"})
    common = prop.merge(base, on="seed", how="inner")
    common = common[common["seed"].isin(SEEDS)].sort_values("seed")
    if common.empty:
        return None, common
    seeds = common["seed"].astype(int).tolist()
    record = {
        "dataset": dataset,
        "proposed_C_index": float(common["proposed_C_index_seed"].mean()),
        "baseline_C_index": float(common["baseline_C_index_seed"].mean()),
        "n_common_seeds": len(seeds),
        "common_seed_ids": ",".join(str(x) for x in seeds),
        "all_five_common": set(seeds) == EXPECTED_SEED_SET,
    }
    record["delta_C_index"] = record["proposed_C_index"] - record["baseline_C_index"]
    return record, common


def compare_ood(seed_level, analysis_name, baselines, dataset_order):
    """OOD: one paired observation per dataset after common-seed averaging."""
    ood = seed_level[seed_level["setting"] == "OOD"].copy()
    rows, pair_rows, seed_rows = [], [], []

    for baseline in baselines:
        records = []
        for dataset in dataset_order:
            record, common = _common_seed_dataset_pair(ood, dataset, baseline)
            if record is None:
                print(f"[WARNING] {analysis_name}: no common seeds for {dataset}, {baseline}")
                continue
            records.append(record)
            for r in common.itertuples(index=False):
                seed_rows.append({
                    "analysis": analysis_name,
                    "dataset": dataset,
                    "proposed_method": PROPOSED_METHOD,
                    "baseline": baseline,
                    "seed": int(r.seed),
                    "proposed_C_index": float(r.proposed_C_index_seed),
                    "baseline_C_index": float(r.baseline_C_index_seed),
                    "delta_C_index": float(r.proposed_C_index_seed - r.baseline_C_index_seed),
                })

        paired = pd.DataFrame(records)
        if not paired.empty:
            order = {d: i for i, d in enumerate(dataset_order)}
            paired = paired.sort_values(by="dataset", key=lambda x: x.map(order)).reset_index(drop=True)
            diffs = paired["delta_C_index"].to_numpy(dtype=float)
        else:
            diffs = np.asarray([], dtype=float)
        stat, p_value, calc = _safe_wilcoxon(diffs)
        tol = 1e-12

        rows.append({
            "analysis": analysis_name,
            "scenario": analysis_name.split("_")[0] if analysis_name.startswith(("RFS_", "DMFS_")) else "RFS+DMFS",
            "setting": "OOD",
            "pairing_unit": "dataset",
            "dataset": ";".join(paired["dataset"].astype(str)) if len(paired) else "",
            "proposed_method": PROPOSED_METHOD,
            "baseline": baseline,
            "n_pairs": int(len(paired)),
            "pair_ids": ";".join(paired["dataset"].astype(str)) if len(paired) else "",
            "all_five_seeds": bool(paired["all_five_common"].all()) if len(paired) else False,
            "min_common_seeds_per_dataset": int(paired["n_common_seeds"].min()) if len(paired) else 0,
            "mean_proposed_C_index": float(paired["proposed_C_index"].mean()) if len(paired) else np.nan,
            "mean_baseline_C_index": float(paired["baseline_C_index"].mean()) if len(paired) else np.nan,
            "mean_delta_C_index": float(diffs.mean()) if diffs.size else np.nan,
            "median_delta_C_index": float(np.median(diffs)) if diffs.size else np.nan,
            "wins": int(np.sum(diffs > tol)),
            "ties": int(np.sum(np.abs(diffs) <= tol)),
            "losses": int(np.sum(diffs < -tol)),
            "wilcoxon_statistic": stat,
            "p_value": p_value,
            "wilcoxon_calculation": calc,
        })

        for r in paired.itertuples(index=False):
            pair_rows.append({
                "analysis": analysis_name,
                "scenario": analysis_name.split("_")[0] if analysis_name.startswith(("RFS_", "DMFS_")) else "RFS+DMFS",
                "setting": "OOD",
                "pairing_unit": "dataset",
                "pair_id": r.dataset,
                "dataset": r.dataset,
                "seed": np.nan,
                "proposed_method": PROPOSED_METHOD,
                "baseline": baseline,
                "proposed_C_index": r.proposed_C_index,
                "baseline_C_index": r.baseline_C_index,
                "delta_C_index": r.delta_C_index,
                "n_common_seeds": int(r.n_common_seeds),
                "common_seed_ids": r.common_seed_ids,
                "all_five_common": bool(r.all_five_common),
            })

    return _apply_holm(pd.DataFrame(rows)), pd.DataFrame(pair_rows), pd.DataFrame(seed_rows)


def _save_family(results, pairs, seed_pairs, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"wilcoxon_{PROPOSED_METHOD}_vs_baselines_{stem}.csv"
    pair_path = output_dir / f"wilcoxon_pairs_{PROPOSED_METHOD}_vs_baselines_{stem}.csv"
    seed_path = output_dir / f"wilcoxon_common_seed_values_{stem}.csv"
    results.to_csv(result_path, index=False)
    pairs.to_csv(pair_path, index=False)
    seed_pairs.to_csv(seed_path, index=False)
    return result_path, pair_path, seed_path



def _save_final_overall_ood_table(results, output_dir):
    """
    Save the compact manuscript-facing Overall OOD statistical table.

    Each row compares TabSurv_M with one baseline across the 10 external
    OOD cohorts (6 RFS + 4 DMFS). C-index values entering the Wilcoxon test
    are cohort-level means over the common seeds shared by the two methods.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "baseline", "mean_proposed_C_index", "mean_baseline_C_index",
        "mean_delta_C_index", "wins", "ties", "losses",
        "p_value", "holm_adjusted_p", "significant_holm_0.05",
        "n_pairs", "all_five_seeds",
    }
    missing = required.difference(results.columns)
    if missing:
        raise KeyError(
            "Cannot build Final Overall OOD table; missing columns: "
            + ", ".join(sorted(missing))
        )

    # Preserve the configured baseline order rather than sorting alphabetically.
    order = {name: i for i, name in enumerate(BASELINE_MODELS)}
    final = results.copy()
    final = final.sort_values(
        "baseline", key=lambda s: s.map(lambda x: order.get(x, len(order)))
    ).reset_index(drop=True)

    final["W/T/L"] = (
        final["wins"].astype(int).astype(str) + "/"
        + final["ties"].astype(int).astype(str) + "/"
        + final["losses"].astype(int).astype(str)
    )

    manuscript = pd.DataFrame({
        "Baseline": final["baseline"],
        f"{PROPOSED_METHOD} C-index": final["mean_proposed_C_index"],
        "Baseline C-index": final["mean_baseline_C_index"],
        "Delta C-index": final["mean_delta_C_index"],
        "W/T/L": final["W/T/L"],
        "Raw p": final["p_value"],
        "Holm-adjusted p": final["holm_adjusted_p"],
        "Significant after Holm (0.05)": final["significant_holm_0.05"].astype(bool),
        "n OOD cohorts": final["n_pairs"].astype(int),
        "all five common seeds": final["all_five_seeds"].astype(bool),
    })

    csv_path = output_dir / "Final_Overall_OOD_statistical_comparison.csv"
    manuscript.to_csv(csv_path, index=False, float_format="%.6f")

    # Also write a publication-ready LaTeX table containing the compact columns
    # used in the manuscript. Raw p-values are retained alongside Holm-adjusted
    # values for transparency.
    all_sig = bool(final["significant_holm_0.05"].astype(bool).all())
    sig_sentence = (
        "All comparisons remain significant after correction."
        if all_sig
        else f"{int(final['significant_holm_0.05'].astype(bool).sum())} of {len(final)} "
             "comparisons remain significant after correction."
    )
    caption = (
        r"Paired statistical comparison of TabSurv\_M against eight baselines "
        r"over the 10 OOD cohorts (six RFS and four DMFS). C-index is first "
        r"averaged over the five common seeds within each cohort. $\Delta$ "
        r"C-index denotes TabSurv\_M minus the baseline. W/T/L denotes the "
        r"number of cohorts in which TabSurv\_M wins, ties, or loses. "
        r"Two-sided Wilcoxon signed-rank tests are used with Holm correction "
        r"over the eight comparisons. " + sig_sentence
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\label{tab:statistical_comparison}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Baseline & TabSurv\_M & Baseline & $\Delta$ C-index & W/T/L & Raw $p$ & Holm $p$ \\",
        r"\midrule",
    ]

    for _, r in final.iterrows():
        sig = bool(r["significant_holm_0.05"])
        tab = f"{r['mean_proposed_C_index']:.4f}"
        base = f"{r['mean_baseline_C_index']:.4f}"
        delta = f"{r['mean_delta_C_index']:+.4f}"
        rawp = f"{r['p_value']:.5f}"
        holmp = f"{r['holm_adjusted_p']:.5f}"
        if sig:
            tab = rf"\textbf{{{tab}}}"
            delta = rf"\textbf{{{delta}}}"
            holmp = rf"\textbf{{{holmp}}}"
        wtl = f"{int(r['wins'])}/{int(r['ties'])}/{int(r['losses'])}"
        lines.append(
            f"{r['baseline']} & {tab} & {base} & {delta} & {wtl} & {rawp} & {holmp} " + "\\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    tex_path = output_dir / "Final_Overall_OOD_statistical_comparison.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    # A short machine-readable summary is useful for checks in run_all/PBS logs.
    summary_path = output_dir / "Final_Overall_OOD_statistical_comparison_summary.txt"
    summary_path.write_text(
        f"Proposed method: {PROPOSED_METHOD}\n"
        f"OOD cohorts: {int(final['n_pairs'].min()) if len(final) else 0}\n"
        f"Baseline comparisons: {len(final)}\n"
        f"All Holm-adjusted p < 0.05: {all_sig}\n"
        f"Holm-adjusted p range: "
        f"{final['holm_adjusted_p'].min():.6f} to {final['holm_adjusted_p'].max():.6f}\n",
        encoding="utf-8",
    )

    return csv_path, tex_path, summary_path

def _print_results(results, name):
    print("\n" + "=" * 120)
    print(f"Paired Wilcoxon | {name} | proposed={PROPOSED_METHOD}")
    print("=" * 120)
    display = results.copy()
    for c in ["mean_delta_C_index", "wilcoxon_statistic", "p_value", "holm_adjusted_p"]:
        if c in display.columns:
            display[c] = display[c].map(_fmt)
    cols = [
        "baseline", "pairing_unit", "n_pairs", "mean_delta_C_index",
        "wins", "ties", "losses", "wilcoxon_statistic", "p_value",
        "holm_adjusted_p", "significant_holm_0.05",
    ]
    print(display[cols].to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--analysis",
        choices=["RFS", "DMFS", "overall-ood", "all"],
        default="all",
        help="RFS/DMFS run both InD and OOD families; overall-ood is secondary.",
    )
    p.add_argument("--setting", choices=["InD", "OOD", "both"], default="both",
                   help="Run paired comparisons for InD, OOD, or both. overall-ood requires OOD.")
    p.add_argument(
        "--baselines",
        nargs="+",
        choices=BASELINE_MODELS,
        default=BASELINE_MODELS,
        help="Baseline subset. Default: all eight baselines.",
    )
    p.add_argument("--strict-seeds", action="store_true", help="Require all configured seeds everywhere.")
    p.add_argument(
        "--strict-ind-pairing",
        action="store_true",
        help="Require identical InD patient sets for proposed/baseline within each seed.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    baselines = list(args.baselines)
    requested = args.analysis.lower()
    settings = selected_settings(args.setting)
    run_ind = "InD" in settings
    run_ood = "OOD" in settings
    if requested == "overall-ood" and not run_ood:
        raise SystemExit("--analysis overall-ood requires --setting OOD or both")

    need_rfs = requested in {"rfs", "overall-ood", "all"}
    need_dmfs = requested in {"dmfs", "overall-ood", "all"}
    rfs = _load_seed_level_results("RFS", baselines, strict_seeds=args.strict_seeds, settings=settings) if need_rfs else None
    dmfs = _load_seed_level_results("DMFS", baselines, strict_seeds=args.strict_seeds, settings=settings) if need_dmfs else None

    combined_results, combined_pairs, combined_seed_rows = [], [], []

    for scenario_name, data in [("RFS", rfs), ("DMFS", dmfs)]:
        if data is None or requested not in {scenario_name.lower(), "all"}:
            continue
        if run_ind:
            _check_ind_pairing_file(scenario_name, baselines, strict=args.strict_ind_pairing)
            ind_results, ind_pairs = compare_ind(data, scenario_name, baselines)
            empty_seed_rows = pd.DataFrame(columns=[
                "analysis", "dataset", "proposed_method", "baseline", "seed",
                "proposed_C_index", "baseline_C_index", "delta_C_index",
            ])
            paths = _save_family(
                ind_results, ind_pairs, empty_seed_rows,
                SCENARIOS[scenario_name]["results_dir"], f"{scenario_name}_InD",
            )
            _print_results(ind_results, f"{scenario_name}-InD")
            print("Saved:", paths[0])
            combined_results.append(ind_results)
            combined_pairs.append(ind_pairs)

        if run_ood:
            ood_results, ood_pairs, seed_rows = compare_ood(
                data,
                f"{scenario_name}_OOD",
                baselines,
                SCENARIOS[scenario_name]["testing_datasets"],
            )
            paths = _save_family(
                ood_results, ood_pairs, seed_rows,
                SCENARIOS[scenario_name]["results_dir"], f"{scenario_name}_OOD",
            )
            _print_results(ood_results, f"{scenario_name}-OOD")
            print("Saved:", paths[0])
            combined_results.append(ood_results)
            combined_pairs.append(ood_pairs)
            combined_seed_rows.append(seed_rows)

    if run_ood and requested in {"overall-ood", "all"}:
        overall = pd.concat([rfs, dmfs], ignore_index=True)
        dataset_order = [*SCENARIOS["RFS"]["testing_datasets"], *SCENARIOS["DMFS"]["testing_datasets"]]
        results, pairs, seed_rows = compare_ood(overall, "OVERALL_10_OOD", baselines, dataset_order)
        out_dir = Path("./output/results_overall")
        paths = _save_family(results, pairs, seed_rows, out_dir, "OVERALL_10_OOD")
        final_paths = _save_final_overall_ood_table(results, out_dir)
        _print_results(results, "OVERALL-10-OOD")
        print("Saved:", paths[0])
        print("Final Overall OOD table:", final_paths[0])
        print("Final Overall OOD LaTeX:", final_paths[1])
        print("Final Overall OOD summary:", final_paths[2])
        combined_results.append(results)
        combined_pairs.append(pairs)
        combined_seed_rows.append(seed_rows)

    if combined_results:
        out_dir = Path("./output/results_statistical_comparisons")
        out_dir.mkdir(parents=True, exist_ok=True)
        all_results = pd.concat(combined_results, ignore_index=True, sort=False)
        all_pairs = pd.concat(combined_pairs, ignore_index=True, sort=False)
        all_seed_rows = (
            pd.concat(combined_seed_rows, ignore_index=True, sort=False)
            if combined_seed_rows else pd.DataFrame()
        )
        all_results.to_csv(
            out_dir / f"wilcoxon_{PROPOSED_METHOD}_vs_baselines_all_settings_scenarios.csv",
            index=False,
        )
        all_pairs.to_csv(
            out_dir / f"wilcoxon_pairs_{PROPOSED_METHOD}_vs_baselines_all_settings_scenarios.csv",
            index=False,
        )
        all_seed_rows.to_csv(
            out_dir / "wilcoxon_common_seed_values_all_OOD_analyses.csv",
            index=False,
        )
        print(f"\nCombined final statistical table: {out_dir}")

    if not args.strict_seeds or (run_ind and not args.strict_ind_pairing):
        strict_hint = "--strict-seeds" + (" --strict-ind-pairing" if run_ind else "")
        print(f"\nFor the final manuscript, use {strict_hint} after all requested runs are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
