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

Supplementary per-cohort outputs (in output/results_statistical_comparisons/supplementary/):
    Fig_S_calibration_<endpoint>_<h>y.pdf/.png   calibration curves at each horizon
                                                 (one panel per cohort, InD + external)
    Fig_S_prediction_error_<endpoint>.pdf/.png   Brier score over time per cohort
                                                 (Kaplan-Meier reference dashed)
    Table_S_calibration_per_cohort_<endpoint>.csv/.tex
                                                 IBS, AUC, slope, O/E per cohort and model
    Supplementary_calibration_figures.tex        LaTeX figure blocks with captions

    Re-draw from saved CSVs without re-evaluating:
        python calibration_evaluation.py --supp-only

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
    CALIBRATION_SUPP_DIR,
    MAIN_CALIBRATION_HORIZON,
    PROPOSED_METHOD,
    SCENARIOS,
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


def evaluate_scenario(scenario, settings, models, supp=True):
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
    if supp:
        make_supplementary_outputs(scenario, pec_df, cal_df, by_cohort, models)
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


# ---------------------------------------------------------------------------
# Supplementary per-cohort figures and table
# ---------------------------------------------------------------------------
# One figure per endpoint and horizon with a calibration panel for every cohort
# (in-distribution source cohort first, then the external cohorts in config
# order), and one figure per endpoint with the prediction-error (Brier score)
# curve for every cohort. Curves are averaged over the five seeds. TabSurv_M is
# drawn thick in the accent colour; baselines are thin, each in its own colour;
# the colour of every model is the same across all figures.
SUPP_COLORS = {
    "TabSurv_M": ACCENT,
    "TabSurv_P": "#0b3d91",
    "TabSurv_A": "#6aa7e8",
    "DeepHS": "#e8743b",
    "DeepSurv": "#19a979",
    "ENCox": "#c03b5e",
    "LH": "#945ecf",
    "MTLR": "#8c564b",
    "PCHazard": "#d4a017",
    "PMF": "#13a4b4",
    "RSF": "#4c4c4c",
}
SUPP_FALLBACK_COLORS = ["#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ff9896"]


def _supp_style(model, models):
    if model in SUPP_COLORS:
        color = SUPP_COLORS[model]
    else:
        extra = [m for m in models if m not in SUPP_COLORS]
        color = SUPP_FALLBACK_COLORS[extra.index(model) % len(SUPP_FALLBACK_COLORS)]
    main = model == PROPOSED_METHOD
    return dict(color=color, lw=2.4 if main else 1.1, zorder=4 if main else 2,
                alpha=1.0 if main else 0.85)


def _supp_cohorts(scenario, df):
    """Ordered (setting, dataset, label) panels present in df: InD first, then OOD."""
    cfg = SCENARIOS[scenario]
    present = set(zip(df["setting"], df["dataset"]))
    out = []
    if ("InD", cfg["training_dataset"]) in present:
        out.append(("InD", cfg["training_dataset"], f"{cfg['training_dataset']} (in-distribution)"))
    for d in cfg["testing_datasets"]:
        if ("OOD", d) in present:
            out.append(("OOD", d, f"{d} (external)"))
    # anything else present but not in config, keep its order
    for setting, d in dict.fromkeys(zip(df["setting"], df["dataset"])):
        if not any(o[0] == setting and o[1] == d for o in out):
            out.append((setting, d, f"{d} ({'in-distribution' if setting == 'InD' else 'external'})"))
    return out


def _supp_grid(n, ncol=3, w=4.3, h=3.6):
    ncol = max(1, min(ncol, n))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow + 0.9), squeeze=False)
    return fig, axes


def _supp_legend(fig, models, extra=None):
    handles = [Line2D([], [], **{k: v for k, v in _supp_style(m, models).items()
                                 if k in ("color", "lw")}, label=m) for m in models]
    if extra:
        handles += extra
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))


def _supp_save(fig, path_stem):
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Supplementary figure: {path_stem}.pdf (and .png)")


def supp_prediction_error_figure(scenario, pec_df, models, out_dir=CALIBRATION_SUPP_DIR):
    """Brier score over time for every cohort of one endpoint (mean over seeds)."""
    if pec_df is None or pec_df.empty:
        return None
    panels = _supp_cohorts(scenario, pec_df)
    fig, axes = _supp_grid(len(panels))
    for ax, (setting, d, label) in zip(axes.ravel(), panels):
        dsub = pec_df[(pec_df.setting == setting) & (pec_df.dataset == d)]
        ref = dsub.groupby("time")["brier_KM_reference"].mean()
        ax.plot(ref.index, ref.values, color=INK, ls=(0, (4, 2)), lw=1.3, zorder=3)
        for m in models:
            msub = dsub[dsub.model == m]
            if msub.empty:
                continue
            curve = msub.groupby("time")["brier"].mean()
            ax.plot(curve.index, curve.values, **_supp_style(m, models))
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Time (years)", fontsize=9.5)
        ax.set_ylabel("Brier score", fontsize=9.5)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8.5)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Prediction-error curves (Brier score over time), {scenario} endpoint",
                 fontsize=13, fontweight="bold")
    _supp_legend(fig, [m for m in models if m in set(pec_df.model)],
                 extra=[Line2D([], [], color=INK, ls=(0, (4, 2)), lw=1.3,
                               label="Kaplan–Meier (reference)")])
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    stem = Path(out_dir) / f"Fig_S_prediction_error_{scenario}"
    _supp_save(fig, stem)
    return stem


def supp_calibration_figure(scenario, cal_df, models, horizon, out_dir=CALIBRATION_SUPP_DIR):
    """Calibration curves at one horizon for every cohort of one endpoint (mean over seeds).

    Each point is a risk group (as defined in survival_metrics.evaluate_survival):
    mean predicted risk of the event by ``horizon`` against the Kaplan-Meier observed
    risk in that group, averaged over seeds. Cohorts whose follow-up does not
    reach the horizon are shown as an empty panel with a note.
    """
    if cal_df is None or cal_df.empty:
        return None
    hsub = cal_df[np.isclose(cal_df["horizon"].astype(float), float(horizon))]
    # Panels for every cohort evaluated for this endpoint, even if empty at this horizon.
    panels = _supp_cohorts(scenario, cal_df)
    if not panels:
        return None
    fig, axes = _supp_grid(len(panels), w=4.0, h=3.9)
    for ax, (setting, d, label) in zip(axes.ravel(), panels):
        dsub = hsub[(hsub.setting == setting) & (hsub.dataset == d)]
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(f"Predicted {horizon:g}-year risk", fontsize=9.5)
        ax.set_ylabel(f"Observed {horizon:g}-year risk (KM)", fontsize=9.5)
        ax.tick_params(labelsize=8.5)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        if dsub.empty:
            ax.text(0.5, 0.5, f"Follow-up too short\nfor {horizon:g}-year calibration",
                    ha="center", va="center", fontsize=9, color=INK_2, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue
        vals = dsub[["pred_risk", "obs_risk"]].to_numpy(dtype=float)
        lim = float(np.nanmax(vals)) * 1.08 if np.isfinite(vals).any() else 1.0
        lim = min(1.0, max(0.05, lim))
        ax.plot([0, lim], [0, lim], color=INK_2, ls=":", lw=1.1, zorder=1)
        for m in models:
            g = (dsub[dsub.model == m].groupby("group")[["pred_risk", "obs_risk"]].mean()
                 .sort_values("pred_risk"))
            if g.empty:
                continue
            st = _supp_style(m, models)
            ax.plot(g.pred_risk, g.obs_risk, marker="o", ms=4.5 if m == PROPOSED_METHOD else 3, **st)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Calibration at {horizon:g} years, {scenario} endpoint", fontsize=13, fontweight="bold")
    _supp_legend(fig, [m for m in models if m in set(cal_df.model)],
                 extra=[Line2D([], [], color=INK_2, ls=":", lw=1.1, label="Perfect calibration")])
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    stem = Path(out_dir) / f"Fig_S_calibration_{scenario}_{horizon:g}y"
    _supp_save(fig, stem)
    return stem


SUPP_TABLE_METRICS = [
    ("IBS", "IBS", 3),
    ("IBS_KM_reference", "IBS (KM ref.)", 3),
    (f"tdAUC_{MAIN_HORIZON:g}y", f"AUC({MAIN_HORIZON:g}y)", 3),
    (f"calibration_slope_{MAIN_HORIZON:g}y", f"Cal. slope ({MAIN_HORIZON:g}y)", 2),
    (f"O_E_ratio_{MAIN_HORIZON:g}y", f"O/E ({MAIN_HORIZON:g}y)", 2),
]


def supp_per_cohort_table(scenario, by_cohort, models, out_dir=CALIBRATION_SUPP_DIR):
    """Per-cohort table: mean ± SD over seeds for every cohort and model."""
    if by_cohort is None or by_cohort.empty:
        return None
    rows = []
    for setting, d, label in _supp_cohorts(scenario, by_cohort):
        for m in models:
            r = by_cohort[(by_cohort.setting == setting) & (by_cohort.dataset == d)
                          & (by_cohort.model == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            row = {"Endpoint": scenario, "Cohort": label, "Model": m}
            for col, lab, dg in SUPP_TABLE_METRICS:
                mu = r.get(f"{col}_mean", np.nan)
                sd = r.get(f"{col}_std", np.nan)
                row[lab] = _fmt(mu, sd, dg) if pd.notna(mu) else "NA"
            rows.append(row)
    t = pd.DataFrame(rows)
    if t.empty:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir / f"Table_S_calibration_per_cohort_{scenario}.csv", index=False)

    cols = [lab for _, lab, _ in SUPP_TABLE_METRICS]
    h = f"{MAIN_HORIZON:g}"
    lines = [
        r"\begin{longtable}{ll" + "c" * len(cols) + "}",
        rf"\caption{{Accuracy and calibration of predicted survival per cohort, {scenario} endpoint. "
        r"IBS: integrated Brier score (lower is better; KM ref.: covariate-free Kaplan--Meier model). "
        rf"AUC({h}y): time-dependent AUC; Cal.\ slope and O/E: calibration slope and "
        rf"observed/expected ratio at {h} years (ideal 1). Mean $\pm$ SD over five seeds; "
        rf"NA: follow-up does not extend beyond {h} years.}}",
        rf"\label{{tab:supp_calibration_{scenario.lower()}}} \\",
        r"\toprule",
        "Cohort & Model & " + " & ".join(cols) + r" \\",
        r"\midrule", r"\endfirsthead",
        r"\toprule",
        "Cohort & Model & " + " & ".join(cols) + r" \\",
        r"\midrule", r"\endhead",
    ]
    prev = None
    for _, r in t.iterrows():
        if prev is not None and r.Cohort != prev:
            lines.append(r"\midrule")
        cohort = r.Cohort if r.Cohort != prev else ""
        prev = r.Cohort
        cells = [str(r[c]).replace("±", r"$\pm$") for c in cols]
        lines.append(f"{cohort.replace('_', chr(92) + '_')} & {r.Model.replace('_', chr(92) + '_')} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}", ""]
    (out_dir / f"Table_S_calibration_per_cohort_{scenario}.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved Supplementary table: {out_dir / f'Table_S_calibration_per_cohort_{scenario}.csv'} (and .tex)")
    return t


def _supp_captions(out_dir=CALIBRATION_SUPP_DIR):
    """LaTeX figure blocks for every Supplementary figure present in out_dir."""
    out_dir = Path(out_dir)
    blocks = []
    for ep in ("RFS", "DMFS"):
        for h in CALIBRATION_HORIZONS:
            f = out_dir / f"Fig_S_calibration_{ep}_{h:g}y.pdf"
            if f.exists():
                blocks.append(
                    r"\begin{figure*}[t]" "\n" r"\centering" "\n"
                    rf"\includegraphics[width=\textwidth]{{{f.name}}}" "\n"
                    rf"\caption{{Calibration of predicted {h:g}-year {ep} risk in every cohort. "
                    rf"Each point is a risk group: mean predicted risk of an event by {h:g} years "
                    r"against the Kaplan--Meier observed risk in that group, averaged over five seeds. "
                    r"The dotted diagonal marks perfect calibration. TabSurv\_M is shown as a thick blue line. "
                    r"Panels marked as having too short follow-up have no patients at risk beyond "
                    rf"{h:g} years.}}" "\n"
                    rf"\label{{fig:supp_calibration_{ep.lower()}_{h:g}y}}" "\n" r"\end{figure*}")
        f = out_dir / f"Fig_S_prediction_error_{ep}.pdf"
        if f.exists():
            blocks.append(
                r"\begin{figure*}[t]" "\n" r"\centering" "\n"
                rf"\includegraphics[width=\textwidth]{{{f.name}}}" "\n"
                rf"\caption{{Prediction-error curves for the {ep} endpoint in every cohort: Brier score "
                r"over time (lower is better), averaged over five seeds. The dashed black line is the "
                r"covariate-free Kaplan--Meier reference; a model curve below it indicates that the "
                r"covariates improve prediction at that time. TabSurv\_M is shown as a thick blue line.}" "\n"
                rf"\label{{fig:supp_prediction_error_{ep.lower()}}}" "\n" r"\end{figure*}")
    if blocks:
        (out_dir / "Supplementary_calibration_figures.tex").write_text("\n\n".join(blocks) + "\n",
                                                                       encoding="utf-8")


def make_supplementary_outputs(scenario, pec_df, cal_df, by_cohort, models,
                               horizons=CALIBRATION_HORIZONS, out_dir=CALIBRATION_SUPP_DIR):
    """All Supplementary per-cohort figures/tables for one endpoint.

    Plotting failures are reported but never discard computed metrics.
    """
    jobs = [("prediction-error figure", lambda: supp_prediction_error_figure(scenario, pec_df, models, out_dir)),
            ("per-cohort table", lambda: supp_per_cohort_table(scenario, by_cohort, models, out_dir))]
    jobs += [(f"calibration figure {h:g}y",
              (lambda h=h: supp_calibration_figure(scenario, cal_df, models, h, out_dir)))
             for h in horizons]
    for name, job in jobs:
        try:
            job()
        except Exception as exc:
            print(f"[WARN] Supplementary {name} for {scenario} not drawn: {type(exc).__name__}: {exc}")
    _supp_captions(out_dir)


def supplementary_from_saved(scenario, models):
    """Re-draw Supplementary outputs from results_dir/calibration/*.csv (no re-evaluation)."""
    cal_dir = SCENARIOS[scenario]["results_dir"] / "calibration"
    pec_p = cal_dir / "prediction_error_curves_all.csv"
    cal_p = cal_dir / "calibration_curves_all.csv"
    bc_p = cal_dir / "calibration_by_cohort.csv"
    if not pec_p.exists():
        print(f"[WARN] {pec_p} not found; run calibration_evaluation.py without --supp-only first.")
        return 0
    pec_df = pd.read_csv(pec_p)
    try:
        cal_df = pd.read_csv(cal_p)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        cal_df = pd.DataFrame()
    by_cohort = pd.read_csv(bc_p) if bc_p.exists() else pd.DataFrame()
    make_supplementary_outputs(scenario, pec_df, cal_df, by_cohort, models)
    return 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=["RFS", "DMFS", "both"], default="both")
    p.add_argument("--setting", choices=["InD", "OOD", "both"], default="both")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--skip-figure", action="store_true",
                   help="Do not draw the manuscript summary figure (Fig_calibration_summary).")
    p.add_argument("--skip-supp", action="store_true",
                   help="Do not draw the Supplementary per-cohort calibration / prediction-error figures.")
    p.add_argument("--supp-only", action="store_true",
                   help="Only re-draw the Supplementary figures/tables from the saved CSVs "
                        "(results_dir/calibration/*.csv); no re-evaluation.")
    return p.parse_args()


def main():
    args = parse_args()
    settings = selected_settings(args.setting)
    if args.supp_only:
        written = 0
        for sc in selected_scenarios(args.scenario):
            written += supplementary_from_saved(sc, args.models)
        return 0 if written else 1
    tables, by_cohorts = [], {}
    for sc in selected_scenarios(args.scenario):
        t, bc = evaluate_scenario(sc, settings, args.models, supp=not args.skip_supp)
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
