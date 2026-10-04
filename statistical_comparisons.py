#!/usr/bin/env python3
"""Cohort-level effect sizes for TabSurv_M versus each baseline (Revision 2, AE comment 3).

All analyses are run SEPARATELY for each endpoint (RFS: six external cohorts,
model trained on METABRIC; DMFS: four external cohorts, model trained on NKI).
The former pooled 10-cohort analysis is no longer produced.

For each endpoint and each baseline:

1. Cohort-level paired effect size
   Delta C = C(TabSurv_M) - C(baseline), averaged over the five shared seeds,
   with a 95% percentile CI from a paired patient-level bootstrap
   (default 2,000 resamples). In each replicate the SAME resampled patients
   are scored for both models and all seeds, so the interval reflects
   sampling uncertainty rather than seed-to-seed variability only.

2. Pooled effect across the endpoint's cohorts
   Random-effects meta-analysis of the cohort Delta C values
   (DerSimonian-Laird tau^2) with the Hartung-Knapp-Sidik-Jonkman (HKSJ)
   variance correction and a t distribution on k-1 df, which is recommended
   when only a few cohorts are pooled. Reported: pooled Delta C, 95% CI, I^2,
   and exact p-value; Holm-adjusted across the baselines within the endpoint.

3. Exact Wilcoxon signed-rank test on the cohort-level Delta C values
   (exact null distribution), Holm-adjusted across baselines within the
   endpoint. With n cohorts the smallest attainable two-sided exact p is
   2/2^n: 0.03125 for RFS (n=6) and 0.125 for DMFS (n=4). These tests are
   therefore reported as secondary evidence.

4. Sensitivity to influential cohorts
   Leave-one-cohort-out: the pooled Delta C and its p-value are recomputed
   with each cohort removed in turn.

In-distribution (held-out METABRIC / NKI splits) comparisons are reported in a
supplementary table as the seed-averaged Delta C with a patient-level
bootstrap CI and bootstrap p-value, Holm-adjusted across baselines.

Outputs (./output/results_statistical_comparisons/):
    Table4_OOD_effect_sizes.csv / .tex     manuscript Table 4 (main columns)
    OOD_effect_sizes_full.csv              every pooled statistic
    OOD_per_cohort_delta_C.csv / .tex      cohort x baseline Delta C [95% CI]
    OOD_leave_one_cohort_out.csv           LOO detail
    InD_effect_sizes.csv / .tex            supplementary InD comparison

Usage:
    python statistical_comparisons.py --analysis all --setting both --n-boot 2000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evaluations import _harmonise_prediction_file, _read_prediction_file
from experiment_config import (
    BASELINE_MODELS,
    PROPOSED_METHOD,
    SCENARIOS,
    SEEDS,
    ensure_output_dirs,
    selected_settings,
)

ALPHA = 0.05
OUT_DIR = Path("./output/results_statistical_comparisons")


# ---------------------------------------------------------------------------
# Harrell's C for many bootstrap replicates at once
# ---------------------------------------------------------------------------
def _pair_matrices(time, event, risk):
    """Comparable-pair and concordance matrices (identical rules to
    sksurv.metrics.concordance_index_censored: a pair (i, j) is comparable if
    i has an event and T_i < T_j, or T_i == T_j and j is censored; tied risk
    scores count 0.5)."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event).astype(bool)
    r = np.asarray(risk, dtype=float)
    comp = e[:, None] & ((t[None, :] > t[:, None]) | ((t[None, :] == t[:, None]) & ~e[None, :]))
    diff = r[:, None] - r[None, :]
    conc = np.where(np.abs(diff) <= 1e-8, 0.5, (diff > 0).astype(float))
    return comp.astype(np.float64), (comp * conc).astype(np.float64)


def c_index_weighted(time, event, risk, weights):
    """Harrell's C for each row of `weights` (B x n multiplicities); weights of
    ones give the ordinary C-index."""
    comp, conc = _pair_matrices(time, event, risk)
    W = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    num = np.einsum("bi,ij,bj->b", W, conc, W, optimize=True)
    den = np.einsum("bi,ij,bj->b", W, comp, W, optimize=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


# ---------------------------------------------------------------------------
# Loading patient-level predictions
# ---------------------------------------------------------------------------
def _load_model_files(pred_dir, dataset, model, seeds):
    out = {}
    for s in seeds:
        p = Path(pred_dir) / f"{dataset}_{model}_seed{s}_predict.csv"
        if p.exists():
            df = _harmonise_prediction_file(_read_prediction_file(p), model, p)
            df["time"] = pd.to_numeric(df["time"], errors="coerce")
            df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(int)
            out[s] = df.reset_index(drop=True)
    return out


def _align(prop_df, base_df, setting):
    """Return (time, event, risk_prop, risk_base) with patients in the same order."""
    if setting == "InD" and "patient_id" in prop_df and "patient_id" in base_df:
        a = prop_df.set_index(prop_df["patient_id"].astype(str))
        b = base_df.set_index(base_df["patient_id"].astype(str))
        common = a.index.intersection(b.index)
        if len(common) != len(a) or len(common) != len(b):
            raise ValueError(f"InD patient sets differ ({len(a)} vs {len(b)}, {len(common)} shared).")
        b = b.loc[a.index]
        return (a["time"].to_numpy(float), a["event"].to_numpy(int),
                a["risk_score_eval"].to_numpy(float), b["risk_score_eval"].to_numpy(float))
    if len(prop_df) != len(base_df) or not (
        np.allclose(prop_df["time"], base_df["time"]) and np.array_equal(prop_df["event"], base_df["event"])
    ):
        raise ValueError("OOD prediction files are not row-aligned (different time/event order).")
    return (prop_df["time"].to_numpy(float), prop_df["event"].to_numpy(int),
            prop_df["risk_score_eval"].to_numpy(float), base_df["risk_score_eval"].to_numpy(float))


# ---------------------------------------------------------------------------
# Cohort-level paired effect size with patient-level bootstrap
# ---------------------------------------------------------------------------
def cohort_effects(pred_dir, dataset, setting, baselines, n_boot, rng, strict_seeds=False):
    """Seed-averaged Delta C (and bootstrap replicates) for every baseline in one cohort."""
    prop = _load_model_files(pred_dir, dataset, PROPOSED_METHOD, SEEDS)
    if not prop:
        return []
    bases = {b: _load_model_files(pred_dir, dataset, b, SEEDS) for b in baselines}

    # One set of bootstrap multiplicities per seed, shared by ALL models, so
    # the comparison is paired. OOD cohorts contain the same patients for
    # every seed, so the same multiplicities are reused across seeds.
    seeds_all = sorted(prop)
    n0 = len(prop[seeds_all[0]])
    same_patients = setting == "OOD" and all(len(prop[s]) == n0 for s in seeds_all)
    W_common = rng.multinomial(n0, np.full(n0, 1.0 / n0), size=n_boot) if same_patients else None
    W = {s: (W_common if same_patients else
             rng.multinomial(len(prop[s]), np.full(len(prop[s]), 1.0 / len(prop[s])), size=n_boot))
         for s in seeds_all}

    rows = []
    for b in baselines:
        seeds = sorted(set(prop) & set(bases[b]))
        if strict_seeds and set(seeds) != set(SEEDS):
            raise RuntimeError(f"{setting}/{dataset}/{b}: seeds {seeds} != {SEEDS}")
        if not seeds:
            continue
        cp, cb, dp_boot = [], [], []
        for s in seeds:
            t, e, rp, rb = _align(prop[s], bases[b][s], setting)
            ones = np.ones((1, len(t)))
            cp.append(c_index_weighted(t, e, rp, ones)[0])
            cb.append(c_index_weighted(t, e, rb, ones)[0])
            dp_boot.append(c_index_weighted(t, e, rp, W[s]) - c_index_weighted(t, e, rb, W[s]))
        boot = np.nanmean(np.vstack(dp_boot), axis=0)
        boot = boot[np.isfinite(boot)]
        delta = float(np.mean(cp) - np.mean(cb))
        p_boot = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
        rows.append({
            "setting": setting, "dataset": dataset, "baseline": b,
            "C_proposed": float(np.mean(cp)), "C_baseline": float(np.mean(cb)),
            "delta_C": delta,
            "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
            "se_boot": float(boot.std(ddof=1)), "p_boot": p_boot,
            "n_patients": int(np.mean([len(prop[s]) for s in seeds])),
            "n_events": int(np.mean([prop[s]["event"].sum() for s in seeds])),
            "n_seeds": len(seeds), "n_boot": int(len(boot)),
        })
    return rows


# ---------------------------------------------------------------------------
# Pooling, Holm, Wilcoxon
# ---------------------------------------------------------------------------
def random_effects_hksj(est, se):
    """DerSimonian-Laird tau^2 with the HKSJ variance correction (t, k-1 df).

    Uses the conservative variant q* = max(1, q_HK) (Roever et al., 2015), so
    the HKSJ interval is never narrower than the standard DL interval.
    """
    y = np.asarray(est, dtype=float)
    v = np.asarray(se, dtype=float) ** 2
    k = len(y)
    if k == 0:
        return {}
    if k == 1:
        return {"pooled_delta_C": float(y[0]), "ci_lo": np.nan, "ci_hi": np.nan,
                "p_value": np.nan, "tau2": np.nan, "I2": np.nan, "k": 1}
    w = 1.0 / v
    mu_fe = np.sum(w * y) / w.sum()
    q = float(np.sum(w * (y - mu_fe) ** 2))
    c = w.sum() - np.sum(w ** 2) / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / c)
    wr = 1.0 / (v + tau2)
    mu = float(np.sum(wr * y) / wr.sum())
    q_hk = float(np.sum(wr * (y - mu) ** 2) / (k - 1))
    se_hk = np.sqrt(max(1.0, q_hk) / wr.sum())
    tcrit = stats.t.ppf(0.975, k - 1)
    p = float(2 * stats.t.sf(abs(mu / se_hk), k - 1))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return {"pooled_delta_C": mu, "ci_lo": mu - tcrit * se_hk, "ci_hi": mu + tcrit * se_hk,
            "p_value": p, "tau2": tau2, "I2": i2, "k": k}


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok])]
    m, run = len(ok), 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        out[j] = min(1.0, run)
    return out


def exact_wilcoxon(d):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return np.nan, np.nan
    res = stats.wilcoxon(d, alternative="two-sided", method="exact")
    return float(res.statistic), float(res.pvalue)


def pooled_ood(cohort_df, scenario, baselines):
    full, loo = [], []
    for b in baselines:
        sub = cohort_df[cohort_df.baseline == b]
        if sub.empty:
            continue
        re = random_effects_hksj(sub.delta_C, sub.se_boot)
        w_stat, w_p = exact_wilcoxon(sub.delta_C)
        sig_pos = int(((sub.ci_lo > 0)).sum())
        sig_neg = int(((sub.ci_hi < 0)).sum())
        loo_rows = []
        for d in sub.dataset:
            r = random_effects_hksj(sub[sub.dataset != d].delta_C, sub[sub.dataset != d].se_boot)
            loo_rows.append({"scenario": scenario, "baseline": b, "left_out": d, **r})
        loo += loo_rows
        loo_df = pd.DataFrame(loo_rows)
        full.append({
            "scenario": scenario, "baseline": b, "k_cohorts": len(sub),
            "mean_C_proposed": sub.C_proposed.mean(), "mean_C_baseline": sub.C_baseline.mean(),
            **{k if k in ("tau2", "I2") else f"re_{k}": v for k, v in re.items() if k != "k"},
            "wins": int((sub.delta_C > 0).sum()), "losses": int((sub.delta_C < 0).sum()),
            "ties": int((sub.delta_C == 0).sum()),
            "cohorts_CI_above_0": sig_pos, "cohorts_CI_below_0": sig_neg,
            "wilcoxon_W": w_stat, "wilcoxon_exact_p": w_p,
            "wilcoxon_min_attainable_p": 2.0 / 2 ** len(sub),
            "loo_delta_min": loo_df.pooled_delta_C.min(), "loo_delta_max": loo_df.pooled_delta_C.max(),
            "loo_p_max": loo_df.p_value.max(),
            "loo_most_influential": loo_df.loc[
                (loo_df.pooled_delta_C - re["pooled_delta_C"]).abs().idxmax(), "left_out"],
        })
    full = pd.DataFrame(full)
    if not full.empty:
        full["re_p_holm"] = holm(full.re_p_value)
        full["wilcoxon_p_holm"] = holm(full.wilcoxon_exact_p)
    return full, pd.DataFrame(loo)


# ---------------------------------------------------------------------------
# Manuscript tables
# ---------------------------------------------------------------------------
def _p(p):
    if not np.isfinite(p):
        return "NA"
    return f"{p:.4f}" if p >= 0.0001 else "<0.0001"


def table4(full):
    rows = []
    for _, r in full.iterrows():
        rows.append({
            "Endpoint": r.scenario,
            "Baseline": r.baseline,
            "Delta C (95% CI)": f"{r.re_pooled_delta_C:+.4f} ({r.re_ci_lo:+.4f}, {r.re_ci_hi:+.4f})",
            "I2 (%)": f"{100 * r.I2:.0f}",
            "W/L (CI excl. 0)": f"{r.wins}/{r.losses} ({r.cohorts_CI_above_0}/{r.cohorts_CI_below_0})",
            "p_Holm pooled": _p(r.re_p_holm),
            "p_Holm Wilcoxon": _p(r.wilcoxon_p_holm),
            "LOO Delta C range": f"{r.loo_delta_min:+.4f} to {r.loo_delta_max:+.4f}",
        })
    return pd.DataFrame(rows)


def table4_latex(t4, cohorts):
    k_rfs, k_dmfs = cohorts.get("RFS", 0), cohorts.get("DMFS", 0)
    cap = (
        r"Paired comparison of TabSurv\_M with each baseline, reported separately for the RFS "
        rf"({k_rfs} external cohorts) and DMFS ({k_dmfs} external cohorts) endpoints. "
        r"$\Delta$C: pooled difference in C-index (TabSurv\_M minus baseline) from a random-effects "
        r"meta-analysis of cohort-level differences (each averaged over five seeds, with standard errors "
        r"from a paired patient-level bootstrap, 2{,}000 resamples), with Hartung--Knapp 95\% CI; "
        r"$I^2$: between-cohort heterogeneity. W/L: cohorts in which TabSurv\_M has a higher/lower "
        r"C-index; in parentheses, cohorts whose bootstrap CI excludes zero in favour of TabSurv\_M/the "
        r"baseline. $p_{\mathrm{H}}$: Holm-adjusted $p$-values across the eight baselines within each "
        r"endpoint, for the pooled estimate and for the exact Wilcoxon signed-rank test on cohort-level "
        r"differences (smallest attainable unadjusted $p$: 0.031 for RFS, 0.125 for DMFS). "
        r"LOO: range of the pooled $\Delta$C when each cohort is left out in turn."
    )
    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{cap}}}",
             r"\label{tab:statistical_comparison}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llccccc}", r"\toprule",
             r"Baseline & $\Delta$C (95\% CI) & $I^2$ (\%) & W/L & $p_{\mathrm{H}}$ pooled & "
             r"$p_{\mathrm{H}}$ Wilcoxon & LOO $\Delta$C range \\"]
    for ep in ["RFS", "DMFS"]:
        sub = t4[t4.Endpoint == ep]
        if sub.empty:
            continue
        lines += [r"\midrule", rf"\multicolumn{{7}}{{l}}{{\textit{{{ep} endpoint}}}} \\"]
        for _, r in sub.iterrows():
            lines.append(" & ".join([r.Baseline, r["Delta C (95% CI)"], r["I2 (%)"], r["W/L (CI excl. 0)"],
                                     r["p_Holm pooled"].replace("<", "$<$"),
                                     r["p_Holm Wilcoxon"].replace("<", "$<$"),
                                     r["LOO Delta C range"].replace(" to ", " to ")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def per_cohort_wide(cohort_df):
    c = cohort_df.copy()
    c["cell"] = [f"{d:+.3f} ({lo:+.3f}, {hi:+.3f})" for d, lo, hi in zip(c.delta_C, c.ci_lo, c.ci_hi)]
    order = {b: i for i, b in enumerate(BASELINE_MODELS)}
    wide = c.pivot_table(index=["scenario", "dataset"], columns="baseline", values="cell", aggfunc="first")
    wide = wide[[b for b in sorted(wide.columns, key=lambda x: order.get(x, 99))]]
    return wide.reset_index()


def per_cohort_latex(wide):
    base_cols = [c for c in wide.columns if c not in ("scenario", "dataset")]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Cohort-level $\Delta$C (TabSurv\_M minus baseline; mean over five seeds) with 95\% "
             r"paired patient-level bootstrap CIs (2{,}000 resamples).}",
             r"\label{tab:per_cohort_delta}", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{l" + "c" * len(base_cols) + "}", r"\toprule",
             "Cohort & " + " & ".join(base_cols) + r" \\"]
    for ep in ["RFS", "DMFS"]:
        sub = wide[wide.scenario == ep]
        if sub.empty:
            continue
        lines += [r"\midrule", rf"\multicolumn{{{len(base_cols) + 1}}}{{l}}{{\textit{{{ep}}}}} \\"]
        for _, r in sub.iterrows():
            cells = [str(r[c]).replace(" (", r"\newline(") if isinstance(r[c], str) else "NA"
                     for c in base_cols]
            lines.append(f"{r.dataset} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def ind_table(ind_df):
    ind_df = ind_df.copy()
    ind_df["p_boot_holm"] = np.nan
    for sc, idx in ind_df.groupby("scenario").groups.items():
        ind_df.loc[idx, "p_boot_holm"] = holm(ind_df.loc[idx, "p_boot"])
    return ind_df


def ind_latex(ind_df):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{In-distribution comparison (held-out METABRIC/NKI splits). $\Delta$C: TabSurv\_M "
             r"minus baseline, averaged over five seeds, with 95\% paired patient-level bootstrap CI; "
             r"$p_{\mathrm{H}}$: Holm-adjusted bootstrap $p$-value across baselines within each endpoint.}",
             r"\label{tab:ind_effect_sizes}", r"\footnotesize", r"\begin{tabular}{llcc}", r"\toprule",
             r"Endpoint & Baseline & $\Delta$C (95\% CI) & $p_{\mathrm{H}}$ \\", r"\midrule"]
    for _, r in ind_df.iterrows():
        lines.append(f"{r.scenario} & {r.baseline} & {r.delta_C:+.4f} ({r.ci_lo:+.4f}, {r.ci_hi:+.4f}) & "
                     f"{_p(r.p_boot_holm).replace('<', '$<$')}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis", choices=["RFS", "DMFS", "all", "overall-ood"], default="all",
                   help="Endpoint(s). 'overall-ood' is kept for compatibility and now runs RFS and DMFS "
                        "separately (the pooled 10-cohort analysis has been removed).")
    p.add_argument("--setting", choices=["InD", "OOD", "both"], default="both")
    p.add_argument("--baselines", nargs="+", choices=BASELINE_MODELS, default=BASELINE_MODELS)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=2026)
    p.add_argument("--strict-seeds", action="store_true", help="Require all five seeds for every pair.")
    p.add_argument("--strict-ind-pairing", action="store_true",
                   help="Kept for compatibility; InD pairing by patient_id is always enforced.")
    return p.parse_args()


def main():
    args = parse_args()
    settings = selected_settings(args.setting)
    scenarios = ["RFS", "DMFS"] if args.analysis in ("all", "overall-ood") else [args.analysis]
    baselines = list(args.baselines)
    rng = np.random.default_rng(args.boot_seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ood_rows, ind_rows = [], []
    for sc in scenarios:
        cfg = ensure_output_dirs(sc)
        if "OOD" in settings:
            for d in SCENARIOS[sc]["testing_datasets"]:
                print(f"[{sc}] OOD {d}: bootstrap x{args.n_boot} ...", flush=True)
                for r in cohort_effects(cfg["prediction_dir"], d, "OOD", baselines, args.n_boot, rng,
                                        args.strict_seeds):
                    ood_rows.append({"scenario": sc, **r})
        if "InD" in settings:
            d = SCENARIOS[sc]["training_dataset"]
            print(f"[{sc}] InD {d}: bootstrap x{args.n_boot} ...", flush=True)
            for r in cohort_effects(cfg["prediction_dir"], d, "InD", baselines, args.n_boot, rng,
                                    args.strict_seeds):
                ind_rows.append({"scenario": sc, **r})

    if ood_rows:
        cohort_df = pd.DataFrame(ood_rows)
        cohort_df.to_csv(OUT_DIR / "OOD_per_cohort_delta_C.csv", index=False)
        wide = per_cohort_wide(cohort_df)
        (OUT_DIR / "OOD_per_cohort_delta_C.tex").write_text(per_cohort_latex(wide), encoding="utf-8")

        fulls, loos = [], []
        for sc in scenarios:
            f, l = pooled_ood(cohort_df[cohort_df.scenario == sc], sc, baselines)
            fulls.append(f)
            loos.append(l)
        full = pd.concat(fulls, ignore_index=True)
        full.to_csv(OUT_DIR / "OOD_effect_sizes_full.csv", index=False)
        pd.concat(loos, ignore_index=True).to_csv(OUT_DIR / "OOD_leave_one_cohort_out.csv", index=False)
        t4 = table4(full)
        t4.to_csv(OUT_DIR / "Table4_OOD_effect_sizes.csv", index=False)
        k = cohort_df.groupby("scenario").dataset.nunique().to_dict()
        (OUT_DIR / "Table4_OOD_effect_sizes.tex").write_text(table4_latex(t4, k), encoding="utf-8")
        print("\n" + t4.to_string(index=False))

    if ind_rows:
        ind_df = ind_table(pd.DataFrame(ind_rows))
        ind_df.to_csv(OUT_DIR / "InD_effect_sizes.csv", index=False)
        (OUT_DIR / "InD_effect_sizes.tex").write_text(ind_latex(ind_df), encoding="utf-8")

    print(f"\nSaved statistical comparison outputs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
