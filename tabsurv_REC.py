import os
import time
import numpy as np
import pandas as pd
import torch


from datasets import (load_datafile, load_datafile_gene, preprocess_dataset, get_target,
                      load_datafile_treatment,preprocess_dataset_rec)


from models import get_model, model_dict
from utils import get_labtrans, evaluate_model_sksurv, survival_curves
import torchtuples as tt
from sksurv.ensemble import RandomSurvivalForest

from tabpfn import TabPFNRegressor



# ----------------- CONFIG -----------------
DATASETS = [  'METABRIC_94genes']
#DATASETS = [  'METABRIC_40genes']
#DATASETS = [  'METABRIC_45genes']
#DATASETS = [  'METABRIC_fullgenes']

PREDICTION_MODELS = [ "RSF"]

TIMESTRING = time.strftime("%Y%m%d%H%M")
OUTPUT_PATH = f"./output/TabSurv_REC_risk_{TIMESTRING}/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

np.random.seed(42)
_ = torch.manual_seed(42)

test_size = 0.3
random_state = 42
all_results = []

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

def define_models(seed: int = 42) -> dict:
    # Define and return a dictionary of models with a fixed random seed
    return {
        "TabSurv": TabPFNRegressor(ignore_pretraining_limits=True, random_state=seed),
    }

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

from sksurv.nonparametric import kaplan_meier_estimator

def mean_survival_time_km(time, event):
    """
    Compute mean survival time as area under the Kaplan–Meier curve
    up to the last observed time.
    """
    time = np.asarray(time)
    event = np.asarray(event).astype(bool)

    surv_time, surv_prob = kaplan_meier_estimator(event, time)

    # Ensure curve starts at (0, 1)
    surv_time = np.insert(surv_time, 0, 0.0)
    surv_prob = np.insert(surv_prob, 0, 1.0)

    # Area under survival curve
    mean_survival = np.trapezoid(surv_prob, surv_time)
    return mean_survival


for dataset_name in DATASETS:
    print(f"Processing dataset: {dataset_name}")

    df, cols_standardize, cols_leave, duration_col, event_col, feature_names, treatments = load_datafile_treatment(dataset_name)

    df_train, df_val, df_test, x_train, x_val, x_test = preprocess_dataset_rec(
        df, cols_standardize, cols_leave, duration_col, event_col, test_size, random_state)


    for model_name in PREDICTION_MODELS:
        print(f"  Risk Scores prediction model: {model_name}")
        model_class = model_dict[model_name]
        labtrans = get_labtrans(model_class, 10) if model_name in ["LogisticHazard", "PMF", "DeepHitSingle", "PCHazard", "MTLR"] else None

        if labtrans is not None:
            y_train = labtrans.fit_transform(*get_target(df_train, duration_col, event_col))
            y_val = labtrans.transform(*get_target(df_val, duration_col, event_col))
            durations_train, events_train = get_target(df_train, duration_col, event_col)
            durations_test, events_test = get_target(df_test, duration_col, event_col)
            model = get_model(model_name, x_train.shape[1], labtrans.out_features, labtrans)
        else:
            durations_train, events_train = get_target(df_train, duration_col, event_col)
            durations_val, events_val = get_target(df_val, duration_col, event_col)
            durations_test, events_test = get_target(df_test, duration_col, event_col)
            y_train = (durations_train, events_train)
            y_val = (durations_val, events_val)
            model = get_model(model_name, x_train.shape[1])

        # Train model
        if model_name == "RSF":
            model = RandomSurvivalForest(
                n_estimators=200,
                min_samples_split=10,
                min_samples_leaf=15,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            )
            y_train_rsf = np.array(
                [(bool(e), t) for e, t in zip(df_train[event_col], df_train[duration_col])],
                dtype=[(event_col, 'bool'), (duration_col, 'float')]
            )
            model.fit(x_train, y_train_rsf)
        else:
            callbacks = [tt.cb.EarlyStopping()]
            model.fit(x_train, y_train, batch_size=256, epochs=100,
                      callbacks=callbacks, val_data=(x_val, y_val))

        # Evaluate model
        metrics = evaluate_model_sksurv(model, x_test, durations_test, events_test, model_name, feature_names)
        df_results = metrics["df_results"]

        c_index = metrics["c_index"][0]
        print(f"C-index_test_{dataset_name}_{model_name}: {c_index}")


        # Estimate Risk Score for training data
        risk_scores = model.predict(x_train)

        if model_name == "RSF":
            risk_scores = model.predict(x_train)

        else:
            if model_name == "PCHazard":
                surv = model.predict_surv_df(x_train)
            elif model_name == "DeepSurv":
                _ = model.compute_baseline_hazards()
                surv = model.predict_surv_df(x_train)
            elif model_name == "LogisticHazard":
                surv = model.predict_surv_df(x_train)

            else:
                surv = model.predict_surv_df(x_train)

            # Compute expected survival time (area under survival curve)
            time_grid = surv.index.values  # time points
            surv_np = surv.to_numpy().T  # shape: [n_samples, n_times]
            exp_surv_time = np.trapz(surv_np, time_grid, axis=1)
            risk_scores = -exp_surv_time  # higher risk = lower expected survival


        # Convert to DataFrame but keep labels
        if isinstance(x_train, np.ndarray):
            x_train = pd.DataFrame(x_train, columns=feature_names)

        # Safe copy
        df_scores = x_train.copy()
        df_scores["risk_score"] = risk_scores
        df_scores["time"] = durations_train
        df_scores["event"] = events_train


        df_scores.to_csv(os.path.join(f"{OUTPUT_PATH}/{dataset_name}_{model_name}_score.csv"), index=False)


        X_train_rec = x_train.copy()
        y_train_rec = df_scores["risk_score"]
        models_rec = define_models(random_state)

        for name, model_rec in models_rec.items():
            model_rec = model_rec.fit(X_train_rec, y_train_rec)

            rec_df = recommend_treatment(model_rec, x_test, treatments)
            combined_df, _, _ = compare_recommendations(rec_df, x_test, treatments)

            combined_df["time"] = durations_test
            combined_df["event"] = events_test

            combined_df["risk_score"] = model_rec.predict(x_test)
            manual_c_index = manual_c_index_risk_score(combined_df, time_col='time', event_col='event', prediction_col='risk_score')
            print(f"C-index_REC_{dataset_name}_{name}: {manual_c_index}")


            # Define plot file path
            plot_filename = f"{dataset_name}_{name}_risk_KM_plot.png"
            plot_path = os.path.join(OUTPUT_PATH, plot_filename)

            p_value = survival_curves(combined_df, "time", "event",
                            name, plot_path)

            # ---------------- Mean Survival Time (AUC of KM) ----------------

            follow_df = combined_df[combined_df["FOLLOW_REC"] == True]
            not_follow_df = combined_df[combined_df["FOLLOW_REC"] == False]

            mean_surv_follow = mean_survival_time_km(
                follow_df["time"], follow_df["event"]
            ) if len(follow_df) > 0 else np.nan

            mean_surv_not_follow = mean_survival_time_km(
                not_follow_df["time"], not_follow_df["event"]
            ) if len(not_follow_df) > 0 else np.nan

            print(
                f"Mean Survival Time ({dataset_name}, {name}): "
                f"Followed = {mean_surv_follow:.2f}, "
                f"Not Followed = {mean_surv_not_follow:.2f}"
            )

            # Store results
            all_results.append({
                "dataset": dataset_name,
                "rec_model": name,
                "c_index_rec": manual_c_index,
                "p_value": p_value,
                "mean_survival_followed": mean_surv_follow,
                "mean_survival_not_followed": mean_surv_not_follow,
                "delta_mean_survival": mean_surv_follow - mean_surv_not_follow

            })




df_summary = pd.DataFrame(all_results)
df_summary.to_csv(
    os.path.join(OUTPUT_PATH, "mean_survival_summary.csv"),
    index=False
)

print("\n=== Mean Survival Time Summary ===")
print(df_summary)
