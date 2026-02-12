
import os
import time
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from datasets import load_datafile, load_datafile_gene, preprocess_dataset, get_target
from models import get_model, model_dict
from utils import get_labtrans, evaluate_model_sksurv
import torchtuples as tt
from sksurv.ensemble import RandomSurvivalForest

from datasets import (load_datafile, load_datafile_gene, preprocess_dataset, get_target,
                      load_datafile_treatment,preprocess_dataset_rec)


from models import get_model, model_dict
from utils import get_labtrans, evaluate_model_sksurv, survival_curves
import torchtuples as tt
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored




# ----------------- CONFIG -----------------

DATASETS = [  'METABRIC_94genes']
#DATASETS = [  'METABRIC_fullgenes']
#DATASETS = [  'METABRIC_40genes']
#DATASETS = [  'METABRIC_45genes']



MODELS = ["DeepHitSingle", "DeepSurv", "LogisticHazard", "MTLR", "PCHazard", "PMF",     "RSF"]


TIMESTRING = time.strftime("%Y%m%d%H%M")
OUTPUT_PATH = f"./output/baseline_REC_risk_{TIMESTRING}/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

np.random.seed(42)
_ = torch.manual_seed(42)

test_size = 0.4
random_state = 42

all_results = []

import numpy as np, random, os

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_global_seeds(random_state)


# ---------------------------- Recommendation Functions ----------------------------
def recommend_treatment_baseline(model, X_test, treatment_plans, model_name, train_features):
    """
    train_features = list of columns used to train the model
    """

    # Convert numpy → DataFrame
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=train_features)

    predicted_outcomes = pd.DataFrame(index=X_test.index)

    for tp in treatment_plans:
        # Counterfactual with EXACT feature alignment
        X_cf = X_test.copy()

        # Only modify treatment columns that exist in training data
        for col in treatment_plans:
            if col in X_cf.columns:
                X_cf[col] = 0
        if tp in X_cf.columns:
            X_cf[tp] = 1

        # Enforce trained feature order
        X_cf = X_cf[train_features].astype(np.float32)


        # --- Predict ---
        if model_name == "RSF":
            pred = model.predict(X_cf.values)
        else:
            surv = model.predict_surv_df(X_cf.values)
            t = surv.index.values
            s = surv.to_numpy().T
            exp_surv = np.trapz(s, t, axis=1)
            pred = -exp_surv

        predicted_outcomes[tp] = pred

    predicted_outcomes["REC_TP"] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes



def compare_recommendations(recommended_df: pd.DataFrame,
                            original_df,
                            tp_cols: list,
                            feature_names=None):
    """
    recommended_df : DataFrame with REC_TP
    original_df    : DataFrame or numpy array of test features
    tp_cols        : list of treatment columns
    feature_names  : list of column names for original_df if it is numpy
    """

    # ------------------------------
    # 1. Convert original_df to DataFrame if needed
    # ------------------------------
    if isinstance(original_df, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names must be provided when original_df is numpy.")
        original_df = pd.DataFrame(original_df, columns=feature_names)

    # ------------------------------
    # 2. Ensure treatment columns exist
    # ------------------------------
    for tp in tp_cols:
        if tp not in original_df.columns:
            original_df[tp] = 0

    # ------------------------------
    # 3. Compute CURRENT_TP from max treatment value
    # ------------------------------
    recommended_df = recommended_df.copy()
    recommended_df["CURRENT_TP"] = original_df[tp_cols].idxmax(axis=1)

    # ------------------------------
    # 4. Whether recommendation matches actual treatment
    # ------------------------------
    recommended_df["FOLLOW_REC"] = recommended_df["REC_TP"] == recommended_df["CURRENT_TP"]

    # ------------------------------
    # 5. Combine outputs
    # ------------------------------
    combined = pd.concat(
        [original_df.reset_index(drop=True), recommended_df.reset_index(drop=True)],
        axis=1
    )

    return combined, None, None


from sksurv.nonparametric import kaplan_meier_estimator

def mean_survival_time_km(time, event):
    """
    Mean survival time = area under the Kaplan–Meier survival curve
    up to the maximum observed time.
    """
    time = np.asarray(time)
    event = np.asarray(event).astype(bool)

    if len(time) == 0:
        return np.nan

    surv_time, surv_prob = kaplan_meier_estimator(event, time)

    # Ensure curve starts at S(0)=1
    surv_time = np.insert(surv_time, 0, 0.0)
    surv_prob = np.insert(surv_prob, 0, 1.0)

    return np.trapz(surv_prob, surv_time)



all_results = []

for dataset_name in DATASETS:
    print(f"Processing dataset: {dataset_name}")

    df, cols_standardize, cols_leave, duration_col, event_col, feature_names, treatments = load_datafile_treatment(dataset_name)

    df_train, df_val, df_test, x_train, x_val, x_test, x_mapper = preprocess_dataset(
        df, cols_standardize, cols_leave, duration_col, event_col, test_size, random_state)

    for model_name in MODELS:
        print(f"  Training model: {model_name}")
        model_class = model_dict[model_name]
        labtrans = get_labtrans(model_class, 10) if model_name in ["LogisticHazard", "PMF", "DeepHitSingle",
                                                                   "PCHazard", "MTLR"] else None

        if labtrans is not None:
            y_train = labtrans.fit_transform(*get_target(df_train, duration_col, event_col))
            y_val = labtrans.transform(*get_target(df_val, duration_col, event_col))
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
                n_jobs=1,
                random_state=random_state
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
        print(f"C-index_{dataset_name}_{model_name}: {c_index}")

        df_results.to_csv(os.path.join(f"{OUTPUT_PATH}/{dataset_name}_{model_name}_predict.csv"), index=False)


           # Extract feature names used for training (important for survival models)
        train_features = df_train.drop(columns=[duration_col, event_col]).columns.tolist()

        # Convert DataFrame to numpy
        x_train = df_train[train_features].values.astype(np.float32)
        x_val = df_val[train_features].values.astype(np.float32)
        x_test = df_test[train_features].values.astype(np.float32)

        rec_df = recommend_treatment_baseline(model, x_test, treatments, model_name, train_features)

        combined_df, _, _ = compare_recommendations(rec_df, x_test, treatments,feature_names)

        combined_df["time"] = durations_test
        combined_df["event"] = events_test
        #

        # Define plot file path
        plot_filename = f"{dataset_name}_{model_name}_risk_KM_plot.png"
        plot_path = os.path.join(OUTPUT_PATH, plot_filename)

        p_value = survival_curves(combined_df, "time", "event",
                        model_name, plot_path)


        # ---------------- Mean Survival Time (KM AUC / RMST) ----------------

        follow_df = combined_df[combined_df["FOLLOW_REC"] == True]
        not_follow_df = combined_df[combined_df["FOLLOW_REC"] == False]

        mean_surv_follow = mean_survival_time_km(
            follow_df["time"], follow_df["event"]
        )

        mean_surv_not_follow = mean_survival_time_km(
            not_follow_df["time"], not_follow_df["event"]
        )

        print(
            f"Mean Survival Time ({dataset_name}, {model_name}): "
            f"Followed = {mean_surv_follow:.2f}, "
            f"Not Followed = {mean_surv_not_follow:.2f}"
        )


        all_results.append({
            "dataset": dataset_name,
            "model": model_name,
            "c_index": metrics["c_index"][0],
            "p_value": p_value,
            "mean_survival_followed": mean_surv_follow,
            "mean_survival_not_followed": mean_surv_not_follow,
            "delta_mean_survival": mean_surv_follow - mean_surv_not_follow
        })

# Save global summary

df_all_results = pd.DataFrame(all_results)
global_outfile = os.path.join(OUTPUT_PATH, "baseline_REC_mean_survival.csv")
df_all_results.to_csv(global_outfile, index=False)

print("\n=== Baseline REC Mean Survival Summary ===")
print(df_all_results)

