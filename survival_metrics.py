#!/usr/bin/env python3
"""Survival-function accuracy and calibration metrics.

Every model's individual survival function S(t | x) is stored on the common
grid ``experiment_config.SURV_GRID`` (one compressed .npz per
dataset/model/seed, written next to the existing ``*_predict.csv`` files) and
scored identically by ``calibration_evaluation.py``:

    Integrated Brier score (IBS)          sksurv.metrics.integrated_brier_score
    Prediction-error (Brier) curves        sksurv.metrics.brier_score
    Time-dependent AUC at fixed horizons   sksurv.metrics.cumulative_dynamic_auc
    Calibration curves at fixed horizons   KM observed risk in quintiles of predicted risk
    Calibration slope                      Cox regression on log(-log S(h|x))
                                           (Austin, Harrell & van Klaveren, Stat Med 2020)
    Observed/expected ratio                KM risk at h / mean predicted risk at h

Censoring weights (IPCW) are estimated in each evaluation cohort, because
censoring patterns differ between external cohorts.

This module is used only by the prognosis pipeline (tabsurv_M.py,
baselines.py, calibration_evaluation.py). The treatment-recommendation
scripts do not import it.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# numpy >= 2.4 removed np.trapz, which scikit-survival 0.25 still calls
# (e.g. in cumulative_dynamic_auc). Restore it as an alias of np.trapezoid.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from sksurv.linear_model import CoxPHSurvivalAnalysis  # noqa: E402
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.metrics import brier_score, cumulative_dynamic_auc
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

from experiment_config import CALIBRATION_HORIZONS, IBS_MAX_TIME, SURV_GRID

MIN_EVENTS_BEFORE_HORIZON = 10
N_CAL_GROUPS = 5            # quintiles: several external cohorts have < 50 events
EPS = 1e-4

# Dense quantile levels used to recover each patient's predictive CDF from TabPFN.
TABPFN_QUANTILES = [round(float(q), 2) for q in np.arange(0.01, 1.0, 0.01)]


# ---------------------------------------------------------------------------
# Building survival matrices S[i, j] = S(grid_j | x_i)
# ---------------------------------------------------------------------------
def _quantiles_to_survival(quantile_list, quantile_levels, grid):
    """Convert TabPFN predictive quantiles (list of arrays, one per level) to S."""
    qp = np.asarray(quantile_list, dtype=float)
    if qp.shape[0] == len(quantile_levels):
        qp = qp.T                                    # -> (n_patients, n_levels)
    qs = np.asarray(quantile_levels, dtype=float)
    grid = np.asarray(grid, dtype=float)
    S = np.empty((qp.shape[0], len(grid)), dtype=float)
    for i in range(qp.shape[0]):
        q_vals = np.maximum.accumulate(qp[i])        # enforce a monotone quantile function
        S[i] = 1.0 - np.interp(grid, q_vals, qs, left=0.0, right=1.0)
    return S


def tabpfn_predict_mean_and_survival(models, X, grid=SURV_GRID,
                                     quantile_levels=TABPFN_QUANTILES):
    """One TabPFN forward pass per model returning the mean AND S(t|x).

    ``output_type="main"`` returns the predictive mean together with the
    requested quantiles from the same logits, so the point predictions are
    identical to ``model.predict(X)`` and no extra inference is needed.
    For TabSurv_M the M Stage-2 models (one per imputed dataset) are pooled
    by averaging: the mean of the M means (as before) and the mean of the M
    survival functions, i.e. the mixture of the M predictive distributions.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    means, survs = [], []
    for mdl in models:
        out = mdl.predict(X, output_type="main", quantiles=list(quantile_levels))
        means.append(np.asarray(out["mean"], dtype=float))
        survs.append(_quantiles_to_survival(out["quantiles"], quantile_levels, grid))
    return np.mean(means, axis=0), np.mean(survs, axis=0)


def _step_interp(times, values, grid):
    """Right-continuous step interpolation of S given at `times` (S = 1 before)."""
    times = np.asarray(times, dtype=float)
    values = np.atleast_2d(np.asarray(values, dtype=float))
    idx = np.searchsorted(times, grid, side="right") - 1
    out = values[:, np.clip(idx, 0, len(times) - 1)]
    out[:, idx < 0] = 1.0
    return out


def survival_matrix_from_model(model, X, model_name, grid=SURV_GRID, train=None):
    """S(t|x) for the baseline models on the common grid.

    RSF: predict_survival_function. ENCox: the fitted model is left untouched
    (models.py is not modified); a Breslow baseline hazard is estimated from
    its linear predictor on the TRAINING split, passed as
    ``train=(x_train, times_train, events_train)``. pycox models use
    predict_surv_df; DeepSurv needs its Breslow baseline hazard, which is
    (re)computed from the training data it was fit on.
    """
    grid = np.asarray(grid, dtype=float)
    X = np.asarray(X, dtype=np.float32)
    if model_name == "RSF" and hasattr(model, "unique_times_"):
        vals = model.predict_survival_function(X, return_array=True)
        return _step_interp(model.unique_times_, vals, grid)
    if model_name == "ENCox":
        if train is None:
            raise ValueError("ENCox survival functions need train=(x_train, times, events).")
        x_tr, t_tr, e_tr = train
        breslow = BreslowEstimator().fit(
            np.asarray(model.predict(np.asarray(x_tr, dtype=np.float32)), dtype=float),
            np.asarray(e_tr).astype(bool), np.asarray(t_tr, dtype=float),
        )
        fns = breslow.get_survival_function(np.asarray(model.predict(X), dtype=float))
        return np.vstack([_step_interp(fn.x, fn.y, grid)[0] for fn in fns])
    if hasattr(model, "predict_survival_function"):
        fns = model.predict_survival_function(X)
        return np.vstack([_step_interp(fn.x, fn.y, grid)[0] for fn in fns])
    if model_name == "DeepSurv":
        model.compute_baseline_hazards()
    surv = model.predict_surv_df(X)                  # rows = times, columns = patients
    return _step_interp(surv.index.to_numpy(float), surv.to_numpy().T, grid)


def save_survival_matrix(path, S, times, events, patient_ids=None, grid=SURV_GRID):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(times, dtype=float).reshape(-1)
    if patient_ids is None:
        patient_ids = np.arange(len(times))
    np.savez_compressed(
        path,
        S=np.clip(np.asarray(S, dtype=np.float32), 0.0, 1.0),
        grid=np.asarray(grid, dtype=float),
        time=times,
        event=np.asarray(events, dtype=int).reshape(-1),
        patient_id=np.asarray(patient_ids).astype(str),
    )
    return path


def survival_matrix_path(prediction_dir, dataset, model, seed):
    return Path(prediction_dir) / "survival_functions" / f"{dataset}_{model}_seed{seed}_surv.npz"


def load_survival_matrix(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluation_grid(times, grid=SURV_GRID, upper=IBS_MAX_TIME):
    """Columns of the stored grid inside this cohort's observed follow-up."""
    times = np.asarray(times, dtype=float)
    lo = np.percentile(times, 5)
    hi = min(upper, np.percentile(times, 80), times.max() - EPS)
    keep = (grid >= lo) & (grid <= hi)
    return np.flatnonzero(keep)


def feasible_horizons(times, events, horizons=CALIBRATION_HORIZONS):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    ok, skipped = [], []
    for h in horizons:
        if h >= times.max() or ((times <= h) & events).sum() < MIN_EVENTS_BEFORE_HORIZON:
            skipped.append(h)
        else:
            ok.append(h)
    return ok, skipped


def _km_at(times, events, h):
    t, s = kaplan_meier_estimator(np.asarray(events, bool), np.asarray(times, float))
    j = np.searchsorted(t, h, side="right") - 1
    return 1.0 if j < 0 else float(s[j])


def calibration_at_horizon(S_h, times, events, h, n_groups=N_CAL_GROUPS):
    """Grouped calibration table plus slope, O/E ratio and mean calibration error."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    S_h = np.clip(np.asarray(S_h, dtype=float), EPS, 1 - EPS)
    pred_risk = 1.0 - S_h

    groups = pd.qcut(pd.Series(pred_risk).rank(method="first"), q=n_groups, labels=False)
    rows = []
    for g in np.unique(groups):
        m = (groups == g).to_numpy()
        tt, ss, ci = kaplan_meier_estimator(events[m], times[m], conf_type="log-log")
        j = np.searchsorted(tt, h, side="right") - 1
        s_obs = 1.0 if j < 0 else ss[j]
        lo_s = 1.0 if j < 0 else ci[0][j]
        hi_s = 1.0 if j < 0 else ci[1][j]
        rows.append({
            "group": int(g) + 1, "n": int(m.sum()),
            "pred_risk": float(pred_risk[m].mean()),
            "obs_risk": float(1 - s_obs),
            "obs_risk_lo": float(1 - hi_s), "obs_risk_hi": float(1 - lo_s),
        })
    table = pd.DataFrame(rows)

    obs_all = 1.0 - _km_at(times, events, h)
    stats = {
        "O_E_ratio": float(obs_all / pred_risk.mean()) if pred_risk.mean() > 0 else np.nan,
        "mean_abs_cal_error": float(np.average(np.abs(table.pred_risk - table.obs_risk), weights=table.n)),
        "calibration_slope": np.nan,
    }
    lp = np.log(-np.log(S_h))
    if np.nanstd(lp) > 1e-8:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cox = CoxPHSurvivalAnalysis(alpha=0.0).fit(
                    lp.reshape(-1, 1),
                    Surv.from_arrays(events & (times <= h), np.minimum(times, h)),
                )
            stats["calibration_slope"] = float(cox.coef_[0])
        except Exception as exc:                      # pragma: no cover - reported, not fatal
            warnings.warn(f"Calibration slope failed at h={h}: {exc}")
    return table, stats


def _integrate(values, times):
    values, times = np.asarray(values, float), np.asarray(times, float)
    area = np.sum(np.diff(times) * (values[1:] + values[:-1]) / 2.0)
    return float(area / (times[-1] - times[0]))


def evaluate_survival(S, grid, times, events, horizons=CALIBRATION_HORIZONS):
    """All comment-2 metrics for one model on one cohort (one seed).

    Returns (summary dict, Brier-curve DataFrame, calibration DataFrame).
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    grid = np.asarray(grid, dtype=float)
    S = np.clip(np.asarray(S, dtype=float), 0.0, 1.0)
    y = Surv.from_arrays(events, times)

    cols = evaluation_grid(times, grid)
    g = grid[cols]
    Sg = S[:, cols]
    km = np.array([_km_at(times, events, t) for t in g])
    S_null = np.tile(km, (len(times), 1))

    # IBS = time-averaged IPCW Brier score (trapezoid rule over the grid);
    # identical to sksurv.metrics.integrated_brier_score but independent of
    # numpy.trapz, which was removed in numpy 2.4.
    _, bs = brier_score(y, y, Sg, g)
    _, bs0 = brier_score(y, y, S_null, g)
    out = {
        "IBS": _integrate(bs, g),
        "IBS_KM_reference": _integrate(bs0, g),
        "IBS_from": float(g[0]), "IBS_to": float(g[-1]),
    }
    brier_curve = pd.DataFrame({"time": g, "brier": bs, "brier_KM_reference": bs0})

    ok, skipped = feasible_horizons(times, events, horizons)
    out["horizons_skipped"] = ",".join(f"{h:g}" for h in skipped)
    cal_frames = []
    for h in ok:
        j = int(np.clip(np.searchsorted(grid, h, side="right") - 1, 0, len(grid) - 1))
        S_h = S[:, j]
        try:
            auc, _ = cumulative_dynamic_auc(y, y, 1.0 - S_h, [h])
            out[f"tdAUC_{h:g}y"] = float(auc[0])
        except Exception as exc:
            warnings.warn(f"tdAUC failed at h={h}: {exc}")
            out[f"tdAUC_{h:g}y"] = np.nan
        table, cal = calibration_at_horizon(S_h, times, events, h)
        for k, v in cal.items():
            out[f"{k}_{h:g}y"] = v
        cal_frames.append(table.assign(horizon=h))
    for h in skipped:
        for k in ("tdAUC", "calibration_slope", "O_E_ratio", "mean_abs_cal_error"):
            out[f"{k}_{h:g}y"] = np.nan
    cal_df = pd.concat(cal_frames, ignore_index=True) if cal_frames else pd.DataFrame()
    return out, brier_curve, cal_df
