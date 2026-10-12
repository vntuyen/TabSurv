import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Gadi/PBS jobs are headless. Force a non-interactive backend before importing pyplot.
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pycox.evaluation import EvalSurv
from sksurv.metrics import concordance_index_censored
from tabpfn import TabPFNRegressor


def get_labtrans(model_class, num_durations):
    return model_class.label_transform(num_durations)


import pandas as pd

def manual_c_index_expected_time(df, time_col='real_survival', event_col='event', prediction_col='predicted_survival'):
    """
    Manually compute C-index when the model predicts expected survival time
    (higher predicted value = better survival / lower risk).
    """
    df = df.reset_index(drop=True)  # Ensure row indices are 0, 1, ..., n-1

    n_concordant = 0
    n_discordant = 0
    n_tied = 0
    n_comparable = 0

    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue

            t_i, e_i, p_i = df.loc[i, time_col], df.loc[i, event_col], df.loc[i, prediction_col]
            t_j, e_j, p_j = df.loc[j, time_col], df.loc[j, event_col], df.loc[j, prediction_col]

            # Only compare if i had an event and lived less than j
            if t_i < t_j and e_i == 1:
                n_comparable += 1
                if p_i < p_j:       # Lower predicted survival → shorter real survival → concordant
                    n_concordant += 1
                elif p_i > p_j:
                    n_discordant += 1
                else:
                    n_tied += 1

    c_index_manual = (n_concordant + 0.5 * n_tied) / n_comparable if n_comparable > 0 else None
    # print(f"Manual C-index (manually): {c_index_manual:.4f}")
    print(f"Concordant: {n_concordant}, Discordant: {n_discordant}, Ties: {n_tied}, Comparable Pairs: {n_comparable}")
    return c_index_manual


def evaluate_model_pycox(model, x_test, durations_test, events_test, model_name):
    # Predict survival function
    if model_name == "PCHazard":
        surv = model.predict_surv_df(x_test)
    elif model_name == "DeepSurv":
        _ = model.compute_baseline_hazards()
        surv = model.predict_surv_df(x_test)
    else:
        surv = model.interpolate(10).predict_surv_df(x_test)

    # Create evaluation object
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')

    # Compute predicted risk score: negative expected survival time (area under curve)
    time_grid = surv.index.values
    surv_np = surv.to_numpy().T  # shape: [n_samples, n_times]
    expected_survival_time = np.trapz(surv_np, time_grid, axis=1)
    risk_score = -expected_survival_time  # higher risk = shorter expected survival

    # Convert x_test to DataFrame if needed
    if isinstance(x_test, np.ndarray):
        x_test = pd.DataFrame(x_test)

    # Create results DataFrame
    df_results = x_test.copy().reset_index(drop=True)
    df_results["real_survival"] = durations_test
    df_results["predicted_risk_score"] = risk_score
    df_results["event"] = events_test

    return {
        'c_index': ev.concordance_td('antolini'),
        'df_results': df_results
    }


##Using lifelines:
from lifelines.utils import concordance_index
def evaluate_model_lifelines(model, x_test, durations_test, events_test, model_name):


    if model_name == "RSF":
        risk_score = model.predict(x_test)

    else:
        if model_name == "PCHazard":
            surv = model.predict_surv_df(x_test)
        elif model_name == "DeepSurv":
            _ = model.compute_baseline_hazards()
            surv = model.predict_surv_df(x_test)

        else:
            surv = model.interpolate(10).predict_surv_df(x_test)

        # Compute expected survival time (area under survival curve)
        time_grid = surv.index.values  # time points
        surv_np = surv.to_numpy().T  # shape: [n_samples, n_times]
        exp_surv_time = np.trapz(surv_np, time_grid, axis=1)
        risk_score = -exp_surv_time  # higher risk = lower expected survival

    c_index = concordance_index(durations_test, risk_score, event_observed=events_test)

    return {
        'c_index': c_index
    }

RISK_SCORE_MODELS = ("RSF", "ENCox")


def evaluate_model_sksurv(model, x_test, durations_test, events_test, model_name, feature_names):
    # Ensure x_test is a DataFrame with column names
    if isinstance(x_test, np.ndarray):
        df_results = pd.DataFrame(x_test, columns=feature_names)
    else:
        df_results = x_test.copy()

    # ---- Predict risk ----
    if model_name in RISK_SCORE_MODELS:
        risk_score = model.predict(x_test)

    else:
        # Predict survival curve
        if model_name == "PCHazard":
            surv = model.predict_surv_df(x_test)
        elif model_name == "DeepSurv":
            _ = model.compute_baseline_hazards()
            surv = model.predict_surv_df(x_test)
        else:
            surv = model.predict_surv_df(x_test)

        # Expected survival time = ∫ S(t) dt
        time_grid = surv.index.values
        surv_np = surv.to_numpy().T
        # exp_surv_time = np.trapezoid(surv_np, time_grid, axis=1)
        exp_surv_time = np.trapz(surv_np, time_grid, axis=1)
        risk_score = -exp_surv_time

    # ---- Compute concordance index ----
    from sksurv.util import Surv
    y_test_structured = Surv.from_arrays(events_test.astype(bool), durations_test)

    c_index_result = concordance_index_censored(
        y_test_structured["event"],
        y_test_structured["time"],
        risk_score
    )

    # ---- Add additional columns ----
    df_results = df_results.reset_index(drop=True)
    df_results["time"] = durations_test
    df_results["event"] = events_test
    df_results["risk_score"] = risk_score

    return {
        'c_index': c_index_result,
        'df_results': df_results
    }

###KM curve for 20 years
def survival_curves(data, time_col, event_col,
                    method_name, output_plot_path=None, make_plot=True,
                    show_p_value=True):
    """KM curves (Followed vs Not Followed) up to 20 years plus log-rank p-value.

    With ``make_plot=False`` only the log-rank p-value is computed (no figure),
    which is how the REC scripts call it per seed; the single figure per
    dataset/model is produced by ``plot_km_across_seeds``.

    With ``show_p_value=False`` no log-rank test is run, the title shows only
    ``method_name`` and the function returns ``None``. This is used for the
    pooled multi-seed curves, for which no p-value is reported.
    """

    # Drop rows with missing values
    data = data.dropna(subset=[time_col, event_col])

    followed = data[data["FOLLOW_REC"] == 1]
    not_followed = data[data["FOLLOW_REC"] == 0]

    # Check if groups are empty
    if followed.empty or not_followed.empty:
        print(f"One group is empty (Followed={len(followed)}, NotFollowed={len(not_followed)}). Skipping KM plot.")
        return None

    # Fit Kaplan-Meier models
    km_followed = KaplanMeierFitter().fit(
        followed[time_col],
        followed[event_col],
        label="Followed"
    )
    km_not_followed = KaplanMeierFitter().fit(
        not_followed[time_col],
        not_followed[event_col],
        label="Not Followed"
    )

    # Log-rank test for p-value (skipped when no p-value is to be reported)
    p_value = None
    if show_p_value or not make_plot:
        logrank_res = logrank_test(
            followed[time_col], not_followed[time_col],
            event_observed_A=followed[event_col],
            event_observed_B=not_followed[event_col]
        )
        p_value = logrank_res.p_value

    if not make_plot:
        return p_value

    # Plot curves
    plt.figure(figsize=(10, 6))
    km_followed.plot(ci_show=True, linewidth=2)
    km_not_followed.plot(ci_show=True, linewidth=2)

    # Limit x-axis to 0–25
    # plt.xlim(0, 25)
    plt.xlim(0, 20)

    if show_p_value:
        plt.title(f"{method_name} (p = {p_value:.4f})", fontsize=32)
    else:
        plt.title(method_name, fontsize=32)
    plt.xlabel("Time (Years)", fontsize=20)
    plt.ylabel("Survival Probability", fontsize=18)
    plt.legend(fontsize=16)
    plt.grid()

    # Save or show
    if output_plot_path:
        plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"KM plot saved to {output_plot_path}")
    else:
        plt.show()

    if show_p_value:
        print(f"P-value: {method_name}: {p_value}")

    return p_value


# ------------------------------------------------------------------------
# One pooled KM figure across seeds (20-year horizon)
# ------------------------------------------------------------------------

def _load_seed_prediction_frames(prediction_files, time_col="time", event_col="event"):
    """Read per-seed prediction CSVs. ``prediction_files`` = iterable of (seed, path)."""
    frames = []
    for seed, path in prediction_files:
        if path is None or (isinstance(path, float) and np.isnan(path)):
            continue
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"[KM] Missing prediction file for seed={seed}: {path}")
            continue
        df = df.dropna(subset=[time_col, event_col, "FOLLOW_REC"]).copy()
        df["FOLLOW_REC"] = df["FOLLOW_REC"].astype(int)
        df["seed"] = int(seed)
        frames.append(df)
    return frames


def plot_km_across_seeds(prediction_files, method_name, plot_prefix,
                         time_col="time", event_col="event"):
    """One pooled 20-year KM figure per dataset/model from all seeds.

    Stacks the test-set predictions of every seed and fits one KM curve per group
    (Followed / Not Followed) with ``survival_curves``.

    prediction_files : iterable of (seed, csv_path) with time/event/FOLLOW_REC columns.
    plot_prefix      : path prefix; writes '<prefix>_pooled_KM_plot.png'.
    Returns a dict of summary columns for the results CSV.

    No log-rank p-value is computed or shown for the pooled curves: the log-rank
    test only assesses whether the curves differ, not whether Followed lies above
    Not Followed, and because the 70/30 test sets overlap across splits, a test
    on the pooled data would treat repeated patients as independent and
    overstate significance.
    """
    out = {"km_n_seeds": 0}
    frames = _load_seed_prediction_frames(prediction_files, time_col, event_col)
    out["km_n_seeds"] = len(frames)
    if not frames:
        return out

    pooled = pd.concat(frames, ignore_index=True)
    survival_curves(
        pooled, time_col, event_col,
        f"{method_name}, {len(frames)} seeds pooled",
        f"{plot_prefix}_pooled_KM_plot.png",
        show_p_value=False,
    )
    return out


def plot_km_summary_grid(method_files, suptitle, output_path, ncols=3,
                         max_time=20.0, time_col="time", event_col="event"):
    """Summary figure: one pooled 20-year KM panel per method in a grid.

    method_files : ordered list of (method_name, [(seed, csv_path), ...]).
                   Each panel stacks that method's per-seed test predictions
                   (Followed vs Not Followed), as in ``plot_km_across_seeds``.
    suptitle     : figure title, e.g. "Kaplan–Meier survival curves for ...".
    output_path  : PNG path; a PDF with the same stem is saved alongside.

    No log-rank p-values are computed or shown (see ``plot_km_across_seeds``).
    Methods without any usable prediction file are left out with a warning.
    Returns the list of method names that were plotted.
    """
    import math
    from pathlib import Path

    panels = []
    for method_name, files in method_files:
        frames = _load_seed_prediction_frames(files, time_col, event_col)
        if not frames:
            print(f"[KM summary] No prediction files for {method_name}; panel skipped.")
            continue
        pooled = pd.concat(frames, ignore_index=True)
        followed = pooled[pooled["FOLLOW_REC"] == 1]
        not_followed = pooled[pooled["FOLLOW_REC"] == 0]
        if followed.empty or not_followed.empty:
            print(f"[KM summary] {method_name}: one group is empty; panel skipped.")
            continue
        panels.append((method_name, len(frames), followed, not_followed))

    if not panels:
        print("[KM summary] Nothing to plot.")
        return []

    ncols = max(1, min(ncols, len(panels)))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.0 * ncols, 5.6 * nrows), squeeze=False)

    for ax, (method_name, n_seeds, followed, not_followed) in zip(axes.flat, panels):
        KaplanMeierFitter().fit(followed[time_col], followed[event_col], label="Followed") \
            .plot(ax=ax, ci_show=True, linewidth=2)
        KaplanMeierFitter().fit(not_followed[time_col], not_followed[event_col], label="Not Followed") \
            .plot(ax=ax, ci_show=True, linewidth=2)
        ax.set_xlim(0, max_time)
        ax.set_ylim(-0.02, 1.04)
        ax.set_title(f"{method_name}, {n_seeds} seeds pooled", fontsize=24)
        ax.set_xlabel("Time (Years)", fontsize=17)
        ax.set_ylabel("Survival Probability", fontsize=15)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=14, loc="upper right")
        ax.grid(True)

    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=34, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.5, w_pad=2.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"KM summary figure saved to {output_path} (+ .pdf)")
    return [p[0] for p in panels]




### KM curve for 10 years
# def survival_curves(data, time_col, event_col,
#                     method_name, output_plot_path=None):
#
#     # Drop rows with missing values
#     data = data.dropna(subset=[time_col, event_col])
#
#     followed = data[data["FOLLOW_REC"] == 1]
#     not_followed = data[data["FOLLOW_REC"] == 0]
#
#     # Check if groups are empty
#     if followed.empty or not_followed.empty:
#         print(f" One group is empty (Followed={len(followed)}, NotFollowed={len(not_followed)}). Skipping KM plot.")
#         return None
#
#     # ============================
#     # 1) Fit Kaplan-Meier (full data)
#     # ============================
#     km_followed = KaplanMeierFitter().fit(
#         followed[time_col], followed[event_col], label="Followed"
#     )
#     km_not_followed = KaplanMeierFitter().fit(
#         not_followed[time_col], not_followed[event_col], label="Not Followed"
#     )
#
#     # ============================
#     # 2) Restrict to time ≤ 10 for p-value
#     # ============================
#     followed_10 = followed[followed[time_col] <= 10]
#     not_followed_10 = not_followed[not_followed[time_col] <= 10]
#
#     # If filtered groups become empty → skip
#     if followed_10.empty or not_followed_10.empty:
#         print(" After restricting to time ≤ 10, one group is empty. Cannot compute p-value.")
#         p_value = float('nan')
#     else:
#         logrank_res = logrank_test(
#             followed_10[time_col], not_followed_10[time_col],
#             event_observed_A=followed_10[event_col],
#             event_observed_B=not_followed_10[event_col]
#         )
#         p_value = logrank_res.p_value
#
#     # ============================
#     # 3) Plot KM curves (full data)
#     # ============================
#     plt.figure(figsize=(10, 6))
#     km_followed.plot(ci_show=True, linewidth=2)
#     km_not_followed.plot(ci_show=True, linewidth=2)
#
#     # Show only 0–10 years on x-axis
#     plt.xlim(0, 10.5)
#
#     plt.title(f"{method_name} (p = {p_value:.4f})", fontsize=32)
#     plt.xlabel("Time (Years)", fontsize=20)
#     plt.ylabel("Survival Probability", fontsize=18)
#     plt.legend(fontsize=16)
#     plt.grid()
#
#     # Save or show plot
#     if output_plot_path:
#         plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")tight
#         plt.close()
#         print(f"KM plot saved to {output_plot_path}")
#     else:
#         plt.show()
#
#     print(f"P-value (≤10 years): {method_name}: {p_value:.4f}")
#
#     return p_value






def manual_c_index_risk_score(df, time_col='real_survival', event_col='event', prediction_col='risk_score'):
    """
    Fast manual C-index for risk-score-based survival models.
    Higher predicted risk -> worse survival (shorter time).
    """
    # Convert columns to numpy arrays (much faster)
    times = df[time_col].values
    events = df[event_col].values
    preds = df[prediction_col].values

    n = len(df)

    n_concordant = 0
    n_discordant = 0
    n_tied = 0
    n_comparable = 0

    for i in range(n):
        if events[i] != 1:   # only pairs where i had event
            continue
        for j in range(i + 1, n):  # j > i avoids double counting
            if times[i] == times[j]:
                continue

            # Determine which one is the earlier event
            if times[i] < times[j]:
                # i should have higher risk than j
                n_comparable += 1
                if preds[i] > preds[j]:
                    n_concordant += 1
                elif preds[i] < preds[j]:
                    n_discordant += 1
                else:
                    n_tied += 1

            elif events[j] == 1 and times[j] < times[i]:
                # j had the event earlier, compare reversed
                n_comparable += 1
                if preds[j] > preds[i]:
                    n_concordant += 1
                elif preds[j] < preds[i]:
                    n_discordant += 1
                else:
                    n_tied += 1

    return (n_concordant + 0.5 * n_tied) / n_comparable if n_comparable > 0 else None




# ---------------------------- Recommendation Functions ----------------------------

def recommend_treatment(model, X_test: pd.DataFrame, treatment_plans: list) -> pd.DataFrame:
    """
    Recommend the best treatment plan for each sample based on the lowest predicted outcome.
    """
    predicted_outcomes = pd.DataFrame(index=X_test.index)

    for tp in treatment_plans:
        X_counterfactual = X_test.copy()
        X_counterfactual[treatment_plans] = 0  # Set all TPs to 0
        X_counterfactual[tp] = 1  # Set the current TP to 1
        predicted_outcomes[tp] = model.predict(X_counterfactual)

    # Recommend the TP with the lowest predicted outcome
    predicted_outcomes['REC_TP'] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def compare_recommendations(recommended_df: pd.DataFrame, original_df: pd.DataFrame, tp_cols: list):
    recommended_df['CURRENT_TP'] = original_df[tp_cols].idxmax(axis=1)
    recommended_df['FOLLOW_REC'] = recommended_df['REC_TP'] == recommended_df['CURRENT_TP']
    combined = pd.concat([original_df.reset_index(drop=True), recommended_df.reset_index(drop=True)], axis=1)
    return combined, None, None

def define_models(seed: int = 42, device=None) -> dict:
    """Return treatment-recommendation models using the local TabPFN checkpoint.

    The import is intentionally local to avoid coupling general utility imports to
    project configuration. On Gadi this guarantees that legacy callers of
    ``define_models`` do not fall back to Hugging Face model resolution.
    """
    from experiment_config import tabpfn_regressor_kwargs

    return {
        "TabSurv": TabPFNRegressor(
            ignore_pretraining_limits=True,
            random_state=seed,
            **tabpfn_regressor_kwargs(device=device),
        ),
    }


from sksurv.nonparametric import kaplan_meier_estimator


def mean_survival_time_km(time, event):
    """
    Mean survival time = area under the Kaplan-Meier survival curve up to the
    maximum observed follow-up time (used for the RMST-based Delta mean
    survival reported in Tables 5/6 and the 5-seed sensitivity check).
    """
    time = np.asarray(time)
    event = np.asarray(event).astype(bool)

    if len(time) == 0:
        return np.nan

    surv_time, surv_prob = kaplan_meier_estimator(event, time)

    # Ensure curve starts at (0, 1)
    surv_time = np.insert(surv_time, 0, 0.0)
    surv_prob = np.insert(surv_prob, 0, 1.0)

    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(surv_prob, surv_time))

