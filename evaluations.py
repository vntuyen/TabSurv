import os
import time
import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_censored

# =======================
# Configuration
# =======================

MODELS = [
    "TabSurv", "DeepHitSingle", "DeepSurv",
    "LogisticHazard", "MTLR", "PCHazard",
    "PMF", "RSF"
]

ALL_DATASETS = [
    'TCGA500', "GEO", 'GSE6532', "GSE19783",
    'HEL', 'unt', 'nki', "transbig",
    'UK', 'mainz', 'upp', 'METABRIC'
]

OOD_DATASETS = [
    'TCGA500', "GEO", 'GSE6532', "GSE19783",
    'HEL', 'unt', 'nki', "transbig",
    'UK', 'mainz', 'upp'
]

# =======================
# Metric Functions
# =======================

def compute_c_index(times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
    events_bool = events.astype(bool)
    c_index_tuple = concordance_index_censored(events_bool, times, risk_scores)
    c_index = float(c_index_tuple[0])  # first element is c-index
    return c_index

# =======================
# Evaluation Function
# =======================

def evaluate_scenario(
    scenario_name: str,
    datasets: list,
    models: list
):
    """
    Fully isolated evaluation for one scenario (InD or OOD).
    """

    print(f"\n========== Evaluating {scenario_name} ==========")

    TIMESTRING = time.strftime("%Y%m%d%H%M")
    PREDICTION_DIR = f"./output/prediction_{scenario_name}"
    OUTPUT_DIR = f"./output/evaluation_{scenario_name}_{TIMESTRING}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenario_metrics = []  # <-- isolated container

    for dataset in datasets:
        for model_name in models:
            file_path = os.path.join(
                PREDICTION_DIR,
                f"{dataset}_{model_name}_predict.csv"
            )

            if not os.path.exists(file_path):
                print(f"❌ Missing: {dataset}_{model_name}")
                continue

            df = pd.read_csv(file_path)

            # -----------------------
            # Column harmonization
            # -----------------------
            if "time" not in df.columns and "real_survival" in df.columns:
                df.rename(columns={"real_survival": "time"}, inplace=True)

            if "event" not in df.columns and "event_observed" in df.columns:
                df.rename(columns={"event_observed": "event"}, inplace=True)

            # -----------------------
            # Risk score definition
            # -----------------------
            if model_name == "TabSurv":
                if "predicted_survival" in df.columns:
                    df["risk_score"] = -df["predicted_survival"]
                else:
                    df["risk_score"] = -df["predicted"]
            else:
                df["risk_score"] = (
                    df["risk_score"]
                    if "risk_score" in df.columns
                    else df["predicted"]
                )

            times = df["time"].to_numpy()
            events = df["event"].to_numpy(dtype=int)
            risk_scores = df["risk_score"].to_numpy()

            c_index = compute_c_index(times, events, risk_scores)

            print(f"CI_{dataset}_{model_name}: {c_index:.4f}")

            scenario_metrics.append({
                "Dataset": dataset,
                "Model": model_name,
                "C-index": c_index
            })

    # =======================
    # Save raw metrics
    # =======================
    metrics_df = pd.DataFrame(scenario_metrics)
    metrics_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{scenario_name}_c_index_metrics.csv"),
        index=False
    )

    # =======================
    # Pivot table
    # =======================
    pivot_df = metrics_df.pivot_table(
        index="Dataset",
        columns="Model",
        values="C-index",
        aggfunc="mean"
    )

    pivot_df = pivot_df.reindex(datasets)
    pivot_df = pivot_df[[m for m in models if m in pivot_df.columns]]
    pivot_df.reset_index(inplace=True)

    pivot_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{scenario_name}_all_model_c_index_pivot.csv"),
        index=False,
        float_format="%.4f"
    )

    print("\nFinal Results Table (C-index):")
    print(pivot_df)

    # =======================
    # Stability summary
    # =======================
    summary_df = (
        metrics_df
        .groupby("Model")
        .agg(
            mean_C_index=("C-index", "mean"),
            std_C_index=("C-index", "std"),
        )
        .reset_index()
    )

    summary_df["Stability_CI"] = (
        summary_df["mean_C_index"] - summary_df["std_C_index"]
    )

    summary_df["Model"] = pd.Categorical(
        summary_df["Model"],
        categories=models,
        ordered=True
    )
    summary_df = summary_df.sort_values("Model")

    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{scenario_name}_stability_score.csv"),
        index=False,
        float_format="%.4f"
    )

    print("\nStability Summary:")
    print(summary_df)

    return metrics_df, pivot_df, summary_df

# =======================
# Run Evaluations
# =======================

# evaluate_scenario(
#     scenario_name="InD",
#     datasets=ALL_DATASETS,
#     models=MODELS
# )

evaluate_scenario(
    scenario_name="OOD",
    datasets=OOD_DATASETS,
    models=MODELS
)
