from pycox.datasets import metabric, gbsg, support, flchain, nwtco
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

def load_datafile_feature_select(dataset_name, feature):
    file_path = f"input/{dataset_name}{feature}.csv"
    df = pd.read_csv(file_path)
    duration_col = "time"
    event_col = "event"
    df = df.dropna(subset=[duration_col, event_col])  # drop rows with missing target
    cols_standardize = list(df.columns)
    cols_leave = []

    return df, cols_standardize, cols_leave, duration_col, event_col



def load_datafile(dataset_name):
    if dataset_name == "gbsg_cancer":
        file_path = f"input/{dataset_name}.csv"
        df = pd.read_csv(file_path)
        duration_col = "time"
        event_col = "event"
        df = df.dropna(subset=[duration_col, event_col])  # drop rows with missing target
        cols_standardize = list(df.columns)
        cols_leave = []

    elif dataset_name == "brca_metabric_numeric":
        file_path = f"input/{dataset_name}.csv"
        df = pd.read_csv(file_path)
        df = df.drop(columns="Sample ID")
        duration_col = "Overall Survival (Months)"
        event_col = "Overall Survival Status"
        df = df.dropna(subset=[duration_col, event_col])  # drop rows with missing target
        cols_standardize = list(df.columns)
        cols_leave = []


    elif dataset_name == "gbsg":
        # df = gbsg.read_df()
        file_path = f"input/{dataset_name}.csv"
        df = pd.read_csv(file_path)
        duration_col = "duration"
        event_col = "event"
        cols_standardize = ['x3', 'x4', 'x5', 'x6']
        cols_leave = ['x0', 'x1', 'x2']


    elif dataset_name == "support":
        # df = support.read_df()
        file_path = f"input/{dataset_name}.csv"
        df = pd.read_csv(file_path)
        duration_col = "duration"
        event_col = "event"
        cols_standardize = ['x0', 'x2', 'x3', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
        cols_leave = ['x1', 'x4', 'x5']
    elif dataset_name == "flchain":
        # df = flchain.read_df()
        file_path = f"input/{dataset_name}.csv"
        df = pd.read_csv(file_path)
        df = df.drop(columns="rownames")
        duration_col = "futime"
        event_col = "death"
        cols_standardize = ['age', 'sample.yr', 'kappa', 'lambda', 'flc.grp', 'creatinine']
        cols_leave = ['sex', 'mgus']
    elif dataset_name == "nwtco":
        # df = nwtco.read_df()
        file_path = f"input/{dataset_name}.csv"
        df = pd.read_csv(file_path)
        df = df.drop(columns="rownames")
        duration_col = "edrel"
        event_col = "rel"
        cols_standardize = ['stage', 'age', 'in.subcohort']
        cols_leave = ['instit_2', 'histol_2', 'study_4']



    else:
        file_path = f"input/{dataset_name}.csv"
        df = pd.read_csv(file_path)
        # df = df.drop(columns="Sample ID")
        duration_col = "time"
        event_col = "event"
        df = df.dropna(subset=[duration_col, event_col])  # drop rows with missing target
        cols_standardize = list(df.columns)
        cols_leave = []


    return df, cols_standardize, cols_leave, duration_col, event_col

def load_datafile_reduced2(dataset_name):
    file_path = f"input/{dataset_name}_reduced2.csv"
    df = pd.read_csv(file_path)
    # df = df.drop(columns="Sample ID")
    duration_col = "time"
    event_col = "event"
    df = df.dropna(subset=[duration_col, event_col])  # drop rows with missing target
    cols_standardize = list(df.columns)
    cols_leave = []

    return df, cols_standardize, cols_leave, duration_col, event_col

def load_datafile_gene(dataset_name):
    # file_path = f"input/{dataset_name}{feature}.csv"
    file_path = f"input/{dataset_name}.csv"

    df = pd.read_csv(file_path)
    # df = df.drop(columns="Sample ID")
    duration_col = "time"
    event_col = "event"
    df = df.dropna(subset=[duration_col, event_col])  # drop rows with missing target
    cols_standardize = list(df.columns)
    cols_leave = []
    feature_names = df.drop([duration_col, event_col], axis=1).columns.tolist()


    return df, cols_standardize, cols_leave, duration_col, event_col, feature_names

def load_datafile_treatment(dataset_name):
    file_path = f"input/{dataset_name}_treatment.csv"


    df = pd.read_csv(file_path)
    duration_col = "time"
    event_col = "event"


    treatments = ["CHEMOTHERAPY","RADIO_THERAPY"]  # binary 1=treated, 0=untreated

    df = df.dropna(subset=[duration_col, event_col])  # drop rows with missing target
    cols_standardize = list(df.columns)
    cols_leave = []
    feature_names = df.drop([duration_col, event_col], axis=1).columns.tolist()

    return df, cols_standardize, cols_leave, duration_col, event_col, feature_names, treatments


def load_datafile_gene_selection(dataset_name, feature):
    file_path = f"input/{dataset_name}{feature}.csv"

    df = pd.read_csv(file_path)
    duration_col = "time"
    event_col = "event"
    df = df.dropna(subset=[duration_col, event_col])  # drop rows with missing target
    cols_standardize = list(df.columns)
    cols_leave = []

    return df, cols_standardize, cols_leave, duration_col, event_col

def load_dataset(dataset_name):
    if dataset_name == "metabric":
        df = metabric.read_df()
        duration_col = "duration"
        event_col = "event"
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7']
    elif dataset_name == "gbsg":
        df = gbsg.read_df()
        duration_col = "duration"
        event_col = "event"
        cols_standardize = ['x3', 'x4', 'x5', 'x6']
        cols_leave = ['x0', 'x1', 'x2']


    elif dataset_name == "support":
        df = support.read_df()
        duration_col = "duration"
        event_col = "event"
        cols_standardize = ['x0', 'x2', 'x3', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
        cols_leave = ['x1', 'x4', 'x5']
    elif dataset_name == "flchain":
        df = flchain.read_df()
        df = df.drop(columns="rownames")
        duration_col = "futime"
        event_col = "death"
        cols_standardize = ['age', 'sample.yr', 'kappa', 'lambda', 'flc.grp', 'creatinine']
        cols_leave = ['sex', 'mgus']
    elif dataset_name == "nwtco":
        df = nwtco.read_df()
        df = df.drop(columns="rownames")
        duration_col = "edrel"
        event_col = "rel"
        cols_standardize = ['stage', 'age', 'in.subcohort']
        cols_leave = ['instit_2', 'histol_2', 'study_4']

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return df, cols_standardize, cols_leave, duration_col, event_col

def preprocess_dataset_rec(df, cols_standardize, cols_leave, duration_col, event_col, test_size, random_state=42):
    # Remove target columns from transformation list (if mistakenly included)
    cols_standardize = [col for col in cols_standardize if col not in [duration_col, event_col]]
    cols_leave = [col for col in cols_leave if col not in [duration_col, event_col]]

    # Split data
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train, df_val = train_test_split(df_train, test_size=test_size, random_state=random_state)

    drop_cols = [duration_col, event_col]
    X_train = df_train.drop(columns=drop_cols, errors='ignore')
    X_test = df_test.drop(columns=drop_cols, errors='ignore')
    X_val = df_val.drop(columns=drop_cols, errors='ignore')


    return df_train, df_val, df_test, X_train, X_val, X_test


def preprocess_dataset(df, cols_standardize, cols_leave, duration_col, event_col, test_size, random_state=42):
    # Remove target columns from transformation list (if mistakenly included)
    cols_standardize = [col for col in cols_standardize if col not in [duration_col, event_col]]
    cols_leave = [col for col in cols_leave if col not in [duration_col, event_col]]

    # Build transformation mapper
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    x_mapper = DataFrameMapper(standardize + leave)

    # Split data
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train, df_val = train_test_split(df_train, test_size=test_size, random_state=random_state)

    drop_cols = [duration_col, event_col]
    X_train = df_train.drop(columns=drop_cols, errors='ignore')
    X_test = df_test.drop(columns=drop_cols, errors='ignore')
    X_val = df_val.drop(columns=drop_cols, errors='ignore')

    # Apply transformations
    x_train = x_mapper.fit_transform(X_train).astype('float32')
    x_val = x_mapper.transform(X_val).astype('float32')
    x_test = x_mapper.transform(X_test).astype('float32')

    return df_train, df_val, df_test, x_train, x_val, x_test, x_mapper


def preprocess_dataset_train(df, cols_standardize, cols_leave, duration_col, event_col, test_size, random_state=42):
    # Remove target columns from transformation list (if mistakenly included)
    cols_standardize = [col for col in cols_standardize if col not in [duration_col, event_col]]
    cols_leave = [col for col in cols_leave if col not in [duration_col, event_col]]

    # Build transformation mapper
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    x_mapper = DataFrameMapper(standardize + leave)

    # Split data
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train, df_val = train_test_split(df_train, test_size=test_size, random_state=random_state)

    drop_cols = [duration_col, event_col]
    X_train = df_train.drop(columns=drop_cols, errors='ignore')
    X_test = df_test.drop(columns=drop_cols, errors='ignore')
    X_val = df_val.drop(columns=drop_cols, errors='ignore')

    # Apply transformations
    x_train = x_mapper.fit_transform(X_train).astype('float32')
    x_val = x_mapper.transform(X_val).astype('float32')
    x_test = x_mapper.transform(X_test).astype('float32')

    return df_train, df_val, df_test, x_train, x_val, x_test, x_mapper


from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

def preprocess_dataset_test(df, cols_standardize, cols_leave, duration_col, event_col, x_mapper=None, random_state=42):
    """
    Prepare test dataset features using the provided x_mapper (preferred).
    If x_mapper is None, a mapper will be created and fitted on the test data (fallback).
    Returns: df_test, x_test (np.float32), x_mapper (fitted)
    """
    # Ensure we don't include target columns in transformation lists:
    cols_standardize = [col for col in cols_standardize if col not in [duration_col, event_col]]
    cols_leave = [col for col in cols_leave if col not in [duration_col, event_col]]

    # Split - here we create a single df_test (use entire df as test)
    # we'll just treat df as the test set and avoid an unnecessary split
    df_test = df.copy()

    drop_cols = [duration_col, event_col]
    X_test = df_test.drop(columns=drop_cols, errors='ignore')

    # If a mapper was provided (from training), use it to transform test features.
    if x_mapper is not None:
        try:
            x_test = x_mapper.transform(X_test).astype('float32')
        except (NotFittedError, AttributeError):
            # If the provided mapper wasn't fitted (rare), try fitting it on training-like data fallback
            # but emit a warning — the preferred approach is to pass the mapper fitted on training data.
            print("[WARN] Provided x_mapper was not fitted. Fitting x_mapper on test data as fallback.")
            try:
                x_test = x_mapper.fit_transform(X_test).astype('float32')
            except Exception as e:
                raise RuntimeError(f"Failed to fit/transform provided x_mapper on test data: {e}")
        return df_test, x_test, x_mapper

    # If no mapper provided: create one and fit on the test set (backward-compatible behaviour).
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    x_mapper_local = DataFrameMapper(standardize + leave)

    # Fit mapper on test (not ideal for OOD evaluations — prefer training mapper)
    x_test = x_mapper_local.fit_transform(X_test).astype('float32')
    return df_test, x_test, x_mapper_local


def preprocess_dataset_test1(df, cols_standardize, cols_leave, duration_col, event_col,  random_state=42):

    cols_standardize = [col for col in cols_standardize if col not in [duration_col, event_col]]
    cols_leave = [col for col in cols_leave if col not in [duration_col, event_col]]

    # Build transformation mapper
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    x_mapper = DataFrameMapper(standardize + leave)

    # # Split data
    df_train, df_test = train_test_split(df, test_size=1, random_state=random_state)

    drop_cols = [duration_col, event_col]
    X_test = df_test.drop(columns=drop_cols, errors='ignore')

    x_test = x_mapper.transform(X_test).astype('float32')
    return df_test, x_test, x_mapper


def get_target(df, duration_col, event_col):
    return df[duration_col].values, df[event_col].values

#

def _ensure_df_and_cols(df_or_name,  time_col=None, event_col=None):
    """
    Accepts either a pandas DataFrame or a dataset name (str).
    If a name is given, load_datafile_gene(name) must return:
      (df, cols_standardize, cols_leave, duration_col, event_col)
    Returns (df, time_col, event_col).
    """
    if isinstance(df_or_name, str):
        df_loaded, _, _, dcol, ecol, feature_names = load_datafile_gene(df_or_name)
        return df_loaded.copy(), (time_col or dcol), (event_col or ecol)
    elif isinstance(df_or_name, pd.DataFrame):
        if time_col is None or event_col is None:
            raise ValueError("When passing a DataFrame, you must provide time_col and event_col.")
        return df_or_name.copy(), time_col, event_col
    else:
        raise TypeError("df_or_name must be either a DataFrame or a dataset name (str).")

def _ensure_df_and_cols_gene_selection(df_or_name, feature, time_col=None, event_col=None):
    """
    Accepts either a pandas DataFrame or a dataset name (str).
    If a name is given, load_datafile_gene(name) must return:
      (df, cols_standardize, cols_leave, duration_col, event_col)
    Returns (df, time_col, event_col).
    """
    if isinstance(df_or_name, str):
        df_loaded, _, _, dcol, ecol, feature_names = load_datafile_gene(df_or_name, feature)
        return df_loaded.copy(), (time_col or dcol), (event_col or ecol)
    elif isinstance(df_or_name, pd.DataFrame):
        if time_col is None or event_col is None:
            raise ValueError("When passing a DataFrame, you must provide time_col and event_col.")
        return df_or_name.copy(), time_col, event_col
    else:
        raise TypeError("df_or_name must be either a DataFrame or a dataset name (str).")

def load_tab_survival_dataset_censoring(
    df_or_name,
    time_col: str,
    event_col: str,
    test_size: float,
    random_state: int,
):
    """
    Prepare TabSurv dataset:
      1. Split into train_full and test (stratified)
      2. Split train_full into:
            - train_dead   (event == 1)
            - train_alive  (event == 0)
      3. Return:
            X_test, y_test, y_test_event,
            X_train_dead, y_train_dead,
            X_train_alive, y_train_alive
    """

    # --- Load dataframe and verify column names ---
    df, time_col, event_col = _ensure_df_and_cols(df_or_name, time_col, event_col)

    # --- Basic cleaning ---
    df = df.dropna(subset=[time_col, event_col]).reset_index(drop=True)
    df["patient_id"] = np.arange(len(df))

    # --- Stratified split if possible ---
    stratify_col = df[event_col] if df[event_col].nunique() > 1 else None
    train_ids, test_ids = train_test_split(
        df["patient_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    # full train and test (no filtering)
    train_full = df[df["patient_id"].isin(train_ids)].copy()
    test_df    = df[df["patient_id"].isin(test_ids)].copy()

    # --- Split train_full into train_dead and train_alive ---
    train_dead  = train_full[train_full[event_col] == 1].copy()
    train_alive = train_full[train_full[event_col] == 0].copy()

    # --- Prepare X and y for each set ---
    drop_cols = [time_col, event_col, "patient_id"]

    # test set
    X_test       = test_df.drop(columns=drop_cols, errors="ignore")
    y_test       = test_df[time_col].astype(float)
    y_test_event = test_df[event_col].astype(int)

    # train_dead
    X_train_dead  = train_dead.drop(columns=drop_cols, errors="ignore")
    y_train_dead  = train_dead[time_col].astype(float)

    # train_alive
    X_train_alive = train_alive.drop(columns=drop_cols, errors="ignore")
    y_train_alive = train_alive[time_col].astype(float)

    return (
        X_train_dead, y_train_dead,
        X_train_alive, y_train_alive,
        X_test, y_test, y_test_event
    )


def load_tab_survival_dataset(
    df_or_name,
    time_col: str,
    event_col: str,
    test_size: float,
    random_state: int,
):
    """
    Prepare data for TabSurv training/eval (ID split on a single dataset).
    Training uses ONLY uncensored samples (event == 1); test keeps all.

    Args:
        df_or_name: DataFrame OR dataset name (str) resolvable by load_datafile_gene.
        time_col: Optional override for time column (if df passed).
        event_col: Optional override for event column (if df passed).
        test_size: Fraction for test split.
        random_state: Seed.

    Returns:
        X_train: Features for training (event==1).
        y_train: Observed times for training (event==1).
        X_test:  Features for testing (all samples).
        y_test:  Times for testing (may be censored).
        y_test_event: Event indicators for testing.
    """
    df, time_col, event_col = _ensure_df_and_cols(df_or_name, time_col, event_col)

    # basic cleaning
    df = df.dropna(subset=[time_col, event_col]).reset_index(drop=True)
    df["patient_id"] = np.arange(len(df))

    # optional stratification if both classes exist
    stratify_col = df[event_col] if df[event_col].nunique() > 1 else None
    train_ids, test_ids = train_test_split(
        df["patient_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    train_df_raw = df[df["patient_id"].isin(train_ids) & (df[event_col] == 1)]
    test_df_raw  = df[df["patient_id"].isin(test_ids)]

    drop_cols = [time_col, event_col, "patient_id"]
    X_train = train_df_raw.drop(columns=drop_cols, errors="ignore")
    X_test  = test_df_raw.drop(columns=drop_cols, errors="ignore")

    y_train      = train_df_raw[time_col].astype(float)
    y_test       = test_df_raw[time_col].astype(float)
    y_test_event = test_df_raw[event_col].astype(int)

    return X_train, y_train, X_test, y_test, y_test_event



def load_tab_survival_dataset_feature_selection(
    df_or_name,
    feature,
    time_col: str,
    event_col: str,
    test_size: float,
    random_state: int,
):
    """
    Prepare data for TabSurv training/eval (ID split on a single dataset).
    Training uses ONLY uncensored samples (event == 1); test keeps all.

    Args:
        df_or_name: DataFrame OR dataset name (str) resolvable by load_datafile_gene.
        time_col: Optional override for time column (if df passed).
        event_col: Optional override for event column (if df passed).
        test_size: Fraction for test split.
        random_state: Seed.

    Returns:
        X_train: Features for training (event==1).
        y_train: Observed times for training (event==1).
        X_test:  Features for testing (all samples).
        y_test:  Times for testing (may be censored).
        y_test_event: Event indicators for testing.
    """
    df, time_col, event_col = _ensure_df_and_cols(df_or_name, feature, time_col, event_col)

    # basic cleaning
    df = df.dropna(subset=[time_col, event_col]).reset_index(drop=True)
    df["patient_id"] = np.arange(len(df))

    # optional stratification if both classes exist
    stratify_col = df[event_col] if df[event_col].nunique() > 1 else None
    train_ids, test_ids = train_test_split(
        df["patient_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    train_df_raw = df[df["patient_id"].isin(train_ids) & (df[event_col] == 1)]
    test_df_raw  = df[df["patient_id"].isin(test_ids)]

    drop_cols = [time_col, event_col, "patient_id"]
    X_train = train_df_raw.drop(columns=drop_cols, errors="ignore")
    X_test  = test_df_raw.drop(columns=drop_cols, errors="ignore")

    y_train      = train_df_raw[time_col].astype(float)
    y_test       = test_df_raw[time_col].astype(float)
    y_test_event = test_df_raw[event_col].astype(int)

    return X_train, y_train, X_test, y_test, y_test_event


def load_tab_survival_dataset_test(
    df_or_name,
    time_col: str = None,
    event_col: str = None,
    random_state: int = 42,  # kept for signature compatibility (unused)
):
    """
    Prepare a FULL dataset as test-only (for OOD evaluation).

    Args:
        df_or_name: DataFrame OR dataset name (str) resolvable by load_datafile_gene.
        time_col: Optional override (if df passed).
        event_col: Optional override (if df passed).

    Returns:
        X_test:        Features for testing (all samples).
        y_test_time:   Times (may be censored).
        y_test_event:  Event indicators.
    """
    df, time_col, event_col = _ensure_df_and_cols(df_or_name, time_col, event_col)

    df = df.dropna(subset=[time_col, event_col]).reset_index(drop=True)

    drop_cols = [time_col, event_col]
    X_test        = df.drop(columns=drop_cols, errors="ignore")
    y_test_time   = df[time_col].astype(float)
    y_test_event  = df[event_col].astype(int)

    return X_test, y_test_time, y_test_event


def load_tab_survival_dataset_test_feature_selection(
    df_or_name,
    feature,
    time_col: str = None,
    event_col: str = None,
    random_state: int = 42,  # kept for signature compatibility (unused)
):
    """
    Prepare a FULL dataset as test-only (for OOD evaluation).

    Args:
        df_or_name: DataFrame OR dataset name (str) resolvable by load_datafile_gene.
        time_col: Optional override (if df passed).
        event_col: Optional override (if df passed).

    Returns:
        X_test:        Features for testing (all samples).
        y_test_time:   Times (may be censored).
        y_test_event:  Event indicators.
    """
    df, time_col, event_col = _ensure_df_and_cols_gene_selection(df_or_name,feature, time_col, event_col)

    df = df.dropna(subset=[time_col, event_col]).reset_index(drop=True)

    drop_cols = [time_col, event_col]
    X_test        = df.drop(columns=drop_cols, errors="ignore")
    y_test_time   = df[time_col].astype(float)
    y_test_event  = df[event_col].astype(int)

    return X_test, y_test_time, y_test_event


import pandas as pd


def select_high_variance_genes(
        input_csv_path: str,
        output_csv_path: str,
        n_top_genes: int = 500,
        time_col: str = "time",
        event_col: str = "event"
):
    """
    Select top N high-variance genes from a gene expression dataset and keep survival columns.

    Parameters:
        input_csv_path (str): Path to input CSV file.
        output_csv_path (str): Path to save the output CSV with selected features.
        n_top_genes (int): Number of top-variance genes to retain.
        time_col (str): Name of survival time column.
        event_col (str): Name of event/censoring column.
    """
    # Load dataset
    df = pd.read_csv(input_csv_path)

    # Separate gene expression features and survival columns
    feature_cols = df.columns.difference([time_col, event_col])
    X = df[feature_cols]
    y = df[[time_col, event_col]]

    # Compute variance and select top N genes
    top_genes = X.var().sort_values(ascending=False).head(n_top_genes).index
    X_top = X[top_genes]

    # Combine selected features with survival columns
    df_selected = pd.concat([X_top, y], axis=1)

    # Save the reduced dataset
    df_selected.to_csv(output_csv_path, index=False)
    print(f"✅ Saved top {n_top_genes} high-variance genes to: {output_csv_path}")


