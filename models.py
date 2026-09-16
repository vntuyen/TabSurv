"""
models.py

Model registry and construction helpers shared across the TabSurv scripts.

Two families of survival model live here:

1. pycox neural-net models + RSF (DeepHitSingle, DeepSurv, LogisticHazard,
   MTLR, PCHazard, PMF, RSF): built via get_model(), which just constructs
   an untrained model object from architecture parameters (in_features,
   out_features, labtrans). Actual training happens in the calling script
   via model.fit(...).

2. ElasticNetCox (CoxnetSurvivalAnalysis, scikit-survival): this does NOT
   fit get_model()'s "build an empty model from architecture params"
   pattern, because selecting its elastic-net mixing parameter and penalty
   strength requires the actual training data (nested cross-validation on
   the training split only -- see fit_coxnet_survival's docstring below).
   So unlike the other 7 models, ElasticNetCox is both constructed AND
   fitted by a single call to fit_coxnet_survival(X_train, durations_train,
   events_train, ...), which returns an already-trained model plus the
   selected hyperparameters. Use coxnet_risk_score(model, X) to score it
   afterwards -- though in practice evaluate_model_sksurv() in utils.py
   already knows to call ElasticNetCox's .predict() directly (like RSF),
   so most calling code doesn't need coxnet_risk_score() explicitly.


"""

import time
import warnings

import numpy as np
import pandas as pd
import torchtuples as tt
from pycox.models import LogisticHazard, PMF, DeepHitSingle, PCHazard, MTLR, CoxPH
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sklearn.model_selection import GridSearchCV, KFold


# =====================================================
# Model registry (pycox + RSF)
# =====================================================

model_dict = {
    "DeepHS": DeepHitSingle,
    "DeepSurv": CoxPH,
    "ENCox": CoxnetSurvivalAnalysis,
    "LH": LogisticHazard,
    "PCHazard": PCHazard,
    "PMF": PMF,
    "MTLR": MTLR,
    "RSF": RandomSurvivalForest,
}


def get_model(model_name, in_features, out_features=None, labtrans=None):
    net = tt.practical.MLPVanilla(in_features, [32, 32], out_features or 1, batch_norm=True, dropout=0.1)
    model_cls = model_dict[model_name]

    if model_name == "DeepSurv":
        model = model_cls(net, tt.optim.Adam(0.01))
    elif model_name == "RSF":
        model = model_cls()
    else:
        model = model_cls(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)

    return model


# =====================================================
# ElasticNetCox (CoxnetSurvivalAnalysis, scikit-survival)
# =====================================================
#





def to_structured_y(durations, events):
    """Build the (event, time) structured array scikit-survival expects."""
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events).astype(bool)
    return Surv.from_arrays(event=events, time=durations)


def fit_coxnet_survival(
    X_train,
    durations_train,
    events_train,
    l1_ratios=(0.5,),
    n_alphas=20,
    alpha_min_ratio=0.05,
    cv_folds=3,
    random_state=42,
    max_iter=5000,
    tol=1e-4,
    n_jobs=1,
    verbose=1,
):
    """
    Fit CoxnetSurvivalAnalysis with nested cross-validation for
    hyperparameter selection, using ONLY the training split passed in (no
    leakage from the held-out/OOD cohorts).

    Parameters
    ----------
    X_train : array-like or DataFrame, shape (n_samples, n_features)
    durations_train, events_train : array-like, shape (n_samples,)
    l1_ratios : iterable of float
        Candidate elastic-net mixing parameters to sweep. Default (0.5,) is
        a standard elastic-net midpoint and keeps the total fit count down;
        pass e.g. (0.1, 0.5, 0.9, 1.0) to also tune the mixing parameter via
        the same nested CV if you have the compute budget (this multiplies
        total runtime by len(l1_ratios)).
    n_alphas : int
        Number of candidate penalty strengths along the regularisation path.
    alpha_min_ratio : float
        Ratio of the smallest to largest alpha in the path (glmnet
        convention). Larger values (e.g. 0.05-0.1) avoid the very weakly
        regularised end of the path, which is both the slowest to fit and,
        for high-dimensional (p >> n) data, rarely the selected optimum.
    cv_folds : int
        Number of folds for the inner (training-only) cross-validation.
    random_state : int
    max_iter : int
        Coordinate-descent iteration cap. 5000 is normally ample once `tol`
        is set to a sane value; raise it only if you see "did not converge"
        warnings you want to chase down.
    tol : float
        Coordinate-descent convergence tolerance -- see the performance
        note above.
    n_jobs : int
        Defaults to 1 (serial). GridSearchCV's default parallel backend
        (loky, spawn-based) re-serialises the full training matrix to every
        worker process, which is often slower than serial execution for a
        single dataset-sized job like this one, especially on Windows.
    verbose : int
        Passed to GridSearchCV; sk-learn prints one line per fit at
        verbose=1+ so a long run stays visibly alive.

    Returns
    -------
    best_model : CoxnetSurvivalAnalysis
        Refit on the FULL training split at the selected (l1_ratio, alpha).
    selection : dict
        {'l1_ratio': ..., 'alpha': ..., 'cv_score': ..., 'n_alphas_tried': ...}
        Report these values alongside the baseline results, per the
        manuscript's Section 3.1.2 placeholder.
    """
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
    y_train = to_structured_y(durations_train, events_train)
    print(f"[ENCox] Fitting on X_train shape={X_train.shape} "
          f"(l1_ratios={list(l1_ratios)}, n_alphas={n_alphas}, cv_folds={cv_folds}, "
          f"tol={tol}, max_iter={max_iter})")

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    best_overall = None
    for l1_ratio in l1_ratios:
        t0 = time.time()
        # Step 1: derive a candidate alpha path from the training data only.
        try:
            path_model = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                alpha_min_ratio=alpha_min_ratio,
                n_alphas=n_alphas,
                max_iter=max_iter,
                tol=tol,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                path_model.fit(X_train, y_train)
            alphas = path_model.alphas_
        except Exception as e:
            print(f"[WARN] Could not derive alpha path for l1_ratio={l1_ratio}: {e}")
            continue

        if alphas is None or len(alphas) == 0:
            continue
        print(f"[ENCox] l1_ratio={l1_ratio}: derived {len(alphas)} candidate "
              f"alphas in {time.time() - t0:.1f}s, running {cv_folds}-fold CV "
              f"({len(alphas) * cv_folds} fits)...")

        # Step 2: nested CV over the alpha path (training data only), scored
        # with the estimator's own default score() -- Harrell's C-index --
        # which does not need a censoring-distribution estimate and so
        # cannot hit the IPCW "censoring survival function is zero" failure.
        t1 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gcv = GridSearchCV(
                    CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, max_iter=max_iter, tol=tol),
                    param_grid={"alphas": [[a] for a in alphas]},
                    cv=cv,
                    error_score=0.5,
                    n_jobs=n_jobs,
                    verbose=verbose,
                ).fit(X_train, y_train)
        except Exception as e:
            print(f"[WARN] Nested CV failed for l1_ratio={l1_ratio}: {e}")
            continue
        print(f"[ENCox] l1_ratio={l1_ratio}: CV done in {time.time() - t1:.1f}s, "
              f"best alpha={gcv.best_params_['alphas'][0]:.6g}, "
              f"best CV C-index={gcv.best_score_:.4f}")

        candidate = {
            "l1_ratio": l1_ratio,
            "alpha": float(gcv.best_params_["alphas"][0]),
            "cv_score": float(gcv.best_score_),
            "n_alphas_tried": len(alphas),
        }
        if best_overall is None or candidate["cv_score"] > best_overall["cv_score"]:
            best_overall = candidate

    if best_overall is None:
        # Robust fallback: a mild, fixed penalty so the pipeline never hard-fails.
        print(f"[WARN] ENCox nested CV selection failed everywhere; "
              "falling back to l1_ratio=0.5, alpha=0.1.")
        best_overall = {"l1_ratio": 0.5, "alpha": 0.1, "cv_score": float("nan"),
                         "n_alphas_tried": 0}

    # Step 3: refit on the FULL training split at the selected hyperparameters.
    best_model = CoxnetSurvivalAnalysis(
        l1_ratio=best_overall["l1_ratio"],
        alphas=[best_overall["alpha"]],
        max_iter=max_iter,
        tol=tol,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_model.fit(X_train, y_train)

    return best_model, best_overall


def coxnet_risk_score(model, X):
    """
    Risk score consistent with the rest of the repo's convention (higher =
    higher risk = shorter expected survival). CoxnetSurvivalAnalysis.predict
    already returns the linear predictor (log partial hazard), which is a
    risk score in this sense -- no sign flip needed (contrast with the
    pycox models, where risk is derived as -E[survival time]).
    """
    X = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    return np.asarray(model.predict(X), dtype=float)
