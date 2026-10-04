#!/usr/bin/env python3
"""Accuracy and calibration of predicted survival functions.

Reads the S(t|x) files written by tabsurv_M.py and baselines.py
(``<prediction_dir>/survival_functions/<dataset>_<model>_seed<seed>_surv.npz``)
and reports, for every endpoint, setting, cohort, model and seed:

    IBS (with a Kaplan-Meier reference), prediction-error curves,
    time-dependent AUC, calibration curves, calibration slope and O/E ratio
    at the prespecified horizons in experiment_config.CALIBRATION_HORIZONS.

Outputs (per scenario, in results_dir/calibration/):
    calibration_all_seeds.csv          one row per setting/cohort/model/seed
    calibration_by_cohort.csv          mean ± SD over seeds
    Table_calibration_<scenario>.csv   Supplementary table (OOD: mean ± SD across
    Table_calibration_<scenario>.tex   cohorts after averaging seeds; InD row block)
    prediction_error_curves_all.csv, calibration_curves_all.csv
    fig_calibration_<setting>_<h>y.png, fig_prediction_error_<setting>.png

Outputs (both endpoints, in output/results_statistical_comparisons/):
    Table_calibration_all_endpoints_full.csv   all metrics, InD and OOD
    Table_calibration_main.csv / .tex          compact external-cohort table
    Fig_calibration_summary.pdf / .png         Supplementary figure,
                                               drawn by make_summary_figure() below

Usage:
    python calibration_evaluation.py --scenario both --setting both
"""

import argparse
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiment_config import (  # noqa: E402
    BASELINE_MODELS,
    CALIBRATION_HORIZONS,
    MAIN_CALIBRATION_HORIZON,
    PROPOSED_METHOD,
    SEEDS,
    ensure_output_dirs,
    selected_scenarios,
    selected_settings,
)
from survival_metrics import evaluate_survival, load_survival_matrix, survival_matrix_path  # noqa: E402

DEFAULT_MODELS = [PROPOSED_METHOD, *BASELINE_MODELS]
DEFAULT_PLOT_MODELS = [PROPOSED_METHOD, "RSF", "DeepSurv", "ENCox"]
MAIN_HORIZON = MAIN_CALIBRATION_HORIZON


def _fmt(m, s, d=3):
    if m is None or not np.isfinite(m):
        return "NA"
    return f"{m:.{d}f} ± {0.0 if not np.isfinite(s) else s:.{d}f}"


def evaluate_scenario(scenario, settings, models):
    cfg = ensure_output_dirs(scenario)
    pred_dir = cfg["prediction_dir"]
    out_dir = cfg["results_dir"] / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    if "InD" in settings:
        tasks.append(("InD", cfg["training_dataset"]))
    if "OOD" in settings:
        tasks += [("OOD", d) for d in cfg["testing_datasets"]]

    rows, pecs, cals, missing = [], [], [], []
    for setting, dataset in tasks:
        for model in models:
            for seed in SEEDS:
                path = survival_matrix_path(pred_dir, dataset, model, seed)
                if not path.exists():
                    missing.append({"setting": setting, "dataset": dataset, "model": model,
                                    "seed": seed, "file": str(path)})
                    continue
                z = load_survival_matrix(path)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        summ, pec, cal = evaluate_survival(z["S"], z["grid"], z["time"], z["event"])
                except Exception as exc:
                    print(f"[ERROR] {setting} {dataset} {model} seed={seed}: {type(exc).__name__}: {exc}")
                    continue
                key = {"scenario": scenario, "setting": setting, "dataset": dataset,
                       "model": model, "seed": seed}
                rows.append({**key, **summ})
                pecs.append(pec.assign(**key))
                if not cal.empty:
                    cals.append(cal.assign(**key))
                print(f"{setting:3s} {dataset:9s} {model:9s} seed={seed}: IBS={summ['IBS']:.4f} "
                      f"(KM {summ['IBS_KM_reference']:.4f})")

    if missing:
        pd.DataFrame(missing).to_csv(out_dir / "calibration_missing_files.csv", index=False)
        print(f"[WARN] {len(missing)} survival-function files missing; see calibration_missing_files.csv")
    if not rows:
        if missing and not any(True for _ in pred_dir.glob("survival_functions/*_surv.npz")):
            print(f"No survival-function files found for {scenario}. Re-run tabsurv_M.py and baselines.py.")
        else:
            print(f"No survival functions could be evaluated for {scenario}; see the [ERROR] lines above.")
        return None, None

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "calibration_all_seeds.csv", index=False)
    pec_df = pd.concat(pecs, ignore_index=True)
    pec_df.to_csv(out_dir / "prediction_error_curves_all.csv", index=False)
    cal_df = pd.concat(cals, ignore_index=True) if cals else pd.DataFrame()
    cal_df.to_csv(out_dir / "calibration_curves_all.csv", index=False)

    metric_cols = [c for c in raw.columns if c.startswith(("IBS", "tdAUC", "calibration_slope",
                                                            "O_E_ratio", "mean_abs_cal_error"))
                   and c not in ("IBS_from", "IBS_to")]
    by_cohort = (raw.groupby(["setting", "dataset", "model"], sort=False)[metric_cols]
                 .agg(["mean", "std"]))
    by_cohort.columns = [f"{m}_{s}" for m, s in by_cohort.columns]
    by_cohort = by_cohort.reset_index()
    by_cohort.to_csv(out_dir / "calibration_by_cohort.csv", index=False)

    table = _manuscript_table(by_cohort, scenario, models)
    table.to_csv(out_dir / f"Table_calibration_{scenario}.csv", index=False)
    (out_dir / f"Table_calibration_{scenario}.tex").write_text(_latex(table, scenario), encoding="utf-8")
    _plot_pec(pec_df, out_dir, models)
    if not cal_df.empty:
        _plot_calibration(cal_df, out_dir)
    print(f"\nSaved calibration outputs for {scenario} in {out_dir}")
    print(table.to_string(index=False))
    return table, by_cohort


# ---------------------------------------------------------------------------
# Compact manuscript table: external cohorts, both endpoints side by side
# ---------------------------------------------------------------------------
MAIN_METRICS = [
    ("IBS", "IBS", 3),
    (f"tdAUC_{MAIN_CALIBRATION_HORIZON:g}y", f"AUC({MAIN_CALIBRATION_HORIZON:g}y)", 3),
    (f"calibration_slope_{MAIN_CALIBRATION_HORIZON:g}y", f"Slope({MAIN_CALIBRATION_HORIZON:g}y)", 2),
]


def _across_cohorts(sub, col, d):
    v = sub[f"{col}_mean"].dropna() if f"{col}_mean" in sub else pd.Series(dtype=float)
    if v.empty:
        return "NA"
    txt = f"{v.mean():.{d}f} ± {(v.std(ddof=1) if len(v) > 1 else 0.0):.{d}f}"
    return txt + (f" [{len(v)}]" if len(v) < sub.dataset.nunique() else "")


def main_table(by_cohorts, models):
    """Rows = models; for each endpoint: IBS, AUC(5y), calibration slope(5y).

    OOD only: each cohort is first averaged over the five seeds, then
    mean ± SD across the endpoint's external cohorts (same convention as
    Table 3). A Kaplan-Meier (covariate-free) row gives the IBS reference.
    """
    rows = []
    km_row = {"Model": "Kaplan–Meier (reference)"}
    for sc, bc in by_cohorts.items():
        ood = bc[bc.setting == "OOD"]
        ref = ood.groupby("dataset")["IBS_KM_reference_mean"].first()
        km_row[f"{sc} IBS"] = f"{ref.mean():.3f} ± {(ref.std(ddof=1) if len(ref) > 1 else 0):.3f}"
        km_row[f"{sc} AUC({MAIN_CALIBRATION_HORIZON:g}y)"] = "0.500"
        km_row[f"{sc} Slope({MAIN_CALIBRATION_HORIZON:g}y)"] = "–"
    for m in models:
        r = {"Model": m}
        for sc, bc in by_cohorts.items():
            sub = bc[(bc.setting == "OOD") & (bc.model == m)]
            for col, lab, d in MAIN_METRICS:
                r[f"{sc} {lab}"] = _across_cohorts(sub, col, d) if not sub.empty else "NA"
        rows.append(r)
    rows.append(km_row)
    return pd.DataFrame(rows)


def main_table_latex(t, by_cohorts):
    eps = list(by_cohorts)
    k = {sc: bc[bc.setting == "OOD"].dataset.nunique() for sc, bc in by_cohorts.items()}
    h = f"{MAIN_CALIBRATION_HORIZON:g}"
    cap = (
        r"Accuracy and calibration of predicted survival on the external cohorts ("
        + ", ".join(f"{sc}: {k[sc]} cohorts" for sc in eps) + "). "
        r"IBS: integrated Brier score (lower is better), integrated up to 10 years or the 80th "
        r"percentile of follow-up; the Kaplan--Meier row is a covariate-free reference. "
        rf"AUC({h}y): cumulative/dynamic time-dependent AUC at {h} years. "
        rf"Slope({h}y): calibration slope at {h} years (ideal 1; $<1$ indicates over-dispersed, "
        r"$>1$ under-dispersed predictions). Values are mean $\pm$ SD across cohorts after "
        r"averaging over five seeds; [k] marks metrics available in only k cohorts because follow-up "
        rf"did not extend beyond {h} years. Calibration curves and prediction-error curves for "
        r"every cohort are provided in the Supplementary Material."
    )
    n = 3 * len(eps)
    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{cap}}}", r"\label{tab:calibration}",
             r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l" + "c" * n + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{sc} endpoint}}" for sc in eps) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(eps))),
             "Model & " + " & ".join(["IBS", rf"AUC({h}y)", rf"Slope({h}y)"] * len(eps)) + r" \\",
             r"\midrule"]
    for _, r in t.iterrows():
        if r.Model.startswith("Kaplan"):
            lines.append(r"\midrule")
        cells = [str(v).replace("±", r"$\pm$").replace("–", "--") for v in r.values[1:]]
        name = r.Model.replace("_", r"\_").replace("–", "--")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def _manuscript_table(by_cohort, scenario, models):
    """OOD: seed-averaged cohort values, then mean ± SD across cohorts. InD: mean ± SD over seeds."""
    cols = ["IBS", "IBS_KM_reference", *[f"tdAUC_{h:g}y" for h in CALIBRATION_HORIZONS],
            f"calibration_slope_{MAIN_HORIZON:g}y", f"O_E_ratio_{MAIN_HORIZON:g}y"]
    labels = {"IBS": "IBS", "IBS_KM_reference": "IBS (KM ref.)",
              **{f"tdAUC_{h:g}y": f"AUC({h:g}y)" for h in CALIBRATION_HORIZONS},
              f"calibration_slope_{MAIN_HORIZON:g}y": f"Cal. slope ({MAIN_HORIZON:g}y)",
              f"O_E_ratio_{MAIN_HORIZON:g}y": f"O/E ({MAIN_HORIZON:g}y)"}
    out = []
    for setting in ["InD", "OOD"]:
        sub = by_cohort[by_cohort.setting == setting]
        for model in models:
            m = sub[sub.model == model]
            if m.empty:
                continue
            row = {"Endpoint": scenario, "Setting": setting, "Model": model,
                   "n cohorts": int(m.dataset.nunique())}
            for c in cols:
                if f"{c}_mean" not in m.columns:
                    row[labels[c]] = "NA"
                    continue
                if setting == "InD":           # one cohort: SD over seeds
                    row[labels[c]] = _fmt(m[f"{c}_mean"].iloc[0], m[f"{c}_std"].iloc[0])
                else:                            # SD across cohorts
                    v = m[f"{c}_mean"].dropna()
                    row[labels[c]] = (_fmt(v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)
                                      + (f" [{len(v)}]" if len(v) < len(m) else ""))
            out.append(row)
    return pd.DataFrame(out)


def _latex(table, scenario):
    if table.empty:
        return ""
    cols = [c for c in table.columns if c not in ("Endpoint", "Setting", "n cohorts")]
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Accuracy and calibration of predicted survival functions, {scenario} endpoint. "
             r"IBS: integrated Brier score (lower is better; KM ref.: covariate-free Kaplan--Meier model). "
             r"AUC($t$): cumulative/dynamic time-dependent AUC. Calibration slope (ideal 1) and "
             r"observed/expected ratio (ideal 1) at 5 years. OOD: mean $\pm$ SD across external cohorts "
             r"after averaging over five seeds; [k] marks metrics available in only k cohorts because of "
             r"insufficient follow-up. InD: mean $\pm$ SD over five seeds.}",
             rf"\label{{tab:calibration_{scenario.lower()}}}", r"\footnotesize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(c.replace("_", r"\_") for c in cols) + r" \\"]
    for setting in ["InD", "OOD"]:
        sub = table[table.Setting == setting]
        if sub.empty:
            continue
        lines += [r"\midrule", rf"\multicolumn{{{len(cols)}}}{{l}}{{\textit{{{setting}}}}} \\"]
        for _, r in sub.iterrows():
            cells = [str(r[c]).replace("±", r"$\pm$").replace("_", r"\_") for c in cols]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _plot_pec(pec_df, out_dir, models):
    for setting, sub in pec_df.groupby("setting"):
        datasets = list(dict.fromkeys(sub.dataset))
        n = len(datasets)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
        for ax, d in zip(axes.ravel(), datasets):
            dsub = sub[sub.dataset == d]
            ref = dsub.groupby("time")["brier_KM_reference"].mean()
            ax.plot(ref.index, ref.values, color="black", ls="--", lw=1.2, label="Kaplan–Meier")
            for model in models:
                msub = dsub[dsub.model == model]
                if msub.empty:
                    continue
                curve = msub.groupby("time")["brier"].mean()
                ax.plot(curve.index, curve.values, lw=2.0 if model == PROPOSED_METHOD else 1.0,
                        label=model)
            ax.set_title(d)
            ax.set_xlabel("Years")
            ax.set_ylabel("Brier score")
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_prediction_error_{setting}.png", dpi=200)
        plt.close(fig)


def _plot_calibration(cal_df, out_dir, plot_models=DEFAULT_PLOT_MODELS):
    for (setting, h), sub in cal_df.groupby(["setting", "horizon"]):
        datasets = list(dict.fromkeys(sub.dataset))
        n = len(datasets)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.6 * nrow), squeeze=False)
        for ax, d in zip(axes.ravel(), datasets):
            dsub = sub[sub.dataset == d]
            lim = max(0.05, float(np.nanmax(dsub[["pred_risk", "obs_risk_hi"]].to_numpy())) * 1.05)
            ax.plot([0, lim], [0, lim], color="grey", ls=":", lw=1)
            for model in plot_models:
                g = dsub[dsub.model == model].groupby("group")[["pred_risk", "obs_risk"]].mean()
                if g.empty:
                    continue
                ax.plot(g.pred_risk, g.obs_risk, marker="o", ms=4,
                        lw=2.0 if model == PROPOSED_METHOD else 1.0, label=model)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_title(f"{d} ({h:g} y)")
            ax.set_xlabel("Predicted risk")
            ax.set_ylabel("Observed risk (KM)")
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_calibration_{setting}_{h:g}y.png", dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Manuscript summary figure (Section 3.1.6): TabSurv_M vs baselines
# ---------------------------------------------------------------------------
# Dot plot, rows = endpoints (RFS, DMFS), columns = IBS (lower is better;
# dashed = Kaplan-Meier reference), AUC at 5 y (higher is better; dashed =
# chance) and calibration slope at 5 y (ideal 1; open marker = in-distribution,
# filled = external cohorts). Points are means, whiskers +/- 1 SD across
# external cohorts after averaging seeds. TabSurv_M in the accent colour,
# baselines in grey. Drawn from the same table written to
# Table_calibration_all_endpoints_full.csv, so figure and numbers always match.
import re  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ACCENT = "#2a78d6"      # TabSurv_M
BASE = "#8b8a85"        # baselines (de-emphasis grey)
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
FIG_MODELS = ["TabSurv_M", "DeepHS", "DeepSurv", "ENCox", "LH", "MTLR", "PCHazard", "PMF", "RSF"]
FIG_PANELS = [
    ("IBS", "Integrated Brier score", "lower is better"),
    ("AUC(5y)", "Time-dependent AUC (5 years)", "higher is better"),
    ("Cal. slope (5y)", "Calibration slope (5 years)", "ideal = 1"),
]


def _fig_parse(cell):
    """'0.157 ± 0.048 [3]' -> (0.157, 0.048, 3 or None)."""
    m = re.match(r"\s*([-\d.]+)\s*±\s*([-\d.]+)\s*(?:\[(\d+)\])?", str(cell))
    if not m:
        return np.nan, np.nan, None
    return float(m.group(1)), float(m.group(2)), (int(m.group(3)) if m.group(3) else None)


def _fig_style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(axis="x", colors=INK_2, labelsize=8, length=3)
    ax.tick_params(axis="y", length=0, labelsize=8.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_summary_figure(df, out_dir):
    endpoints = [e for e in ("RFS", "DMFS") if e in set(df.Endpoint)]
    fig, axes = plt.subplots(len(endpoints), 3, figsize=(7.2, 2.55 * len(endpoints) + 0.5),
                             sharey=True, squeeze=False)
    y = np.arange(len(FIG_MODELS))[::-1]                   # TabSurv_M on top
    for r, ep in enumerate(endpoints):
        ood = df[(df.Endpoint == ep) & (df.Setting == "OOD")].set_index("Model")
        ind = df[(df.Endpoint == ep) & (df.Setting == "InD")].set_index("Model")
        for c, (col, title, hint) in enumerate(FIG_PANELS):
            ax = axes[r, c]
            _fig_style(ax)
            # reference line
            if col == "IBS":
                km, _, _ = _fig_parse(ood["IBS (KM ref.)"].iloc[0])
                ax.axvline(km, color=INK_2, ls=(0, (3, 2)), lw=1)
                ax.text(km, -0.55, " Kaplan–Meier", color=INK_2, fontsize=7, va="bottom", ha="left")
            elif col == "AUC(5y)":
                ax.axvline(0.5, color=INK_2, ls=(0, (3, 2)), lw=1)
                ax.text(0.5, -0.55, " chance", color=INK_2, fontsize=7, va="bottom")
            else:
                ax.axvline(1.0, color=INK_2, ls=(0, (3, 2)), lw=1)
                ax.text(1.0, -0.55, " ideal", color=INK_2, fontsize=7, va="bottom")
            for yi, m in zip(y, FIG_MODELS):
                color = ACCENT if m == "TabSurv_M" else BASE
                z = 3 if m == "TabSurv_M" else 2
                if m in ood.index:
                    mu, sd, k = _fig_parse(ood.loc[m, col])
                    if np.isfinite(mu):
                        ax.errorbar(mu, yi, xerr=sd, fmt="o", ms=6, color=color, ecolor=color,
                                    elinewidth=1.4, capsize=0, zorder=z,
                                    markeredgecolor="white", markeredgewidth=0.8)
                if col == "Cal. slope (5y)" and m in ind.index:
                    mu, sd, _ = _fig_parse(ind.loc[m, col])
                    if np.isfinite(mu):
                        ax.plot(mu, yi + 0.28, "o", ms=5, mfc="white", mec=color, mew=1.4, zorder=z)
            if r == 0:
                ax.set_title(f"{title}\n", fontsize=9, color=INK, loc="left", pad=4)
                ax.text(0, 1.0, hint, transform=ax.transAxes, fontsize=7.5, color=INK_2,
                        va="bottom", ha="left")
            if col == "AUC(5y)":
                ax.set_xlim(0.45, 0.80)
            if col == "IBS":
                lo = min(ax.get_xlim()[0], 0.08)
                ax.set_xlim(lo, ax.get_xlim()[1])
        axes[r, 0].set_ylabel(f"{ep} endpoint", fontsize=9, color=INK, labelpad=6)
    for ax in axes[:, 0]:
        ax.set_yticks(y)
        labels = ax.set_yticklabels(FIG_MODELS)
        for lab in labels:
            if lab.get_text() == "TabSurv_M":
                lab.set_color(ACCENT)
                lab.set_fontweight("bold")
            else:
                lab.set_color(INK)
        ax.set_ylim(-0.6, len(FIG_MODELS) - 0.1)
    # legend (shape key for the slope panel)
    handles = [
        Line2D([], [], marker="o", ls="", color=ACCENT, ms=6, label="TabSurv_M"),
        Line2D([], [], marker="o", ls="", color=BASE, ms=6, label="Baselines"),
        Line2D([], [], marker="o", ls="", color=INK_2, ms=6, label="External cohorts (mean ± SD)"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=INK_2, mew=1.4, ms=5,
               label="In-distribution (slope panel)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"Fig_calibration_summary.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_dir / "Fig_calibration_summary.pdf"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=["RFS", "DMFS", "both"], default="both")
    p.add_argument("--setting", choices=["InD", "OOD", "both"], default="both")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--skip-figure", action="store_true",
                   help="Do not draw the manuscript summary figure (Fig_calibration_summary).")
    return p.parse_args()


def main():
    args = parse_args()
    settings = selected_settings(args.setting)
    tables, by_cohorts = [], {}
    for sc in selected_scenarios(args.scenario):
        t, bc = evaluate_scenario(sc, settings, args.models)
        if t is not None:
            tables.append(t)
            by_cohorts[sc] = bc
    if tables:
        from pathlib import Path
        out = Path("./output/results_statistical_comparisons")
        out.mkdir(parents=True, exist_ok=True)
        full = pd.concat(tables, ignore_index=True)
        full.to_csv(out / "Table_calibration_all_endpoints_full.csv", index=False)
        if "OOD" in settings:
            t = main_table(by_cohorts, args.models)
            t.to_csv(out / "Table_calibration_main.csv", index=False)
            (out / "Table_calibration_main.tex").write_text(main_table_latex(t, by_cohorts), encoding="utf-8")
            print("\nManuscript calibration table (external cohorts):")
            print(t.to_string(index=False))
            if not args.skip_figure:
                _draw_summary_figure(full, out)
    return 0


def _draw_summary_figure(full, out_dir):
    """Manuscript figure (IBS, AUC(5y), calibration slope; TabSurv_M vs baselines).

    Uses the table just written, so the figure always matches the numbers.
    A plotting failure is reported but never discards the computed metrics.
    """
    try:
        path = make_summary_figure(full, out_dir)
        print(f"Saved manuscript figure: {path} (and .png)")
    except Exception as exc:
        print(f"[WARN] summary figure not drawn: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
