from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from litexids.data.audit import leakage_candidates
from litexids.data.loading import load_csv_dataset
from litexids.utils.config import ensure_dir, require_package


ResamplingMethod = Literal["none", "smote", "borderline_smote", "adasyn"]


@dataclass
class PreprocessArtifacts:
    x_train: object
    x_valid: object
    x_test: object
    y_train: object
    y_valid: object
    y_test: object
    feature_names: list[str]
    dropped_columns: list[str]


def normalize_binary_labels(series, positive_labels: list[str] | None = None):
    pd = require_package("pandas")
    positive_labels = positive_labels or []
    if series.dtype.kind in "biufc":
        values = series.astype(int)
        if set(values.unique()) <= {0, 1}:
            return values
    lowered = series.astype(str).str.lower()
    positives = {str(x).lower() for x in positive_labels}
    positives |= {"attack", "malicious", "anomaly", "ddos", "dos", "1", "true"}
    benign = {"benign", "normal", "0", "false"}
    mapped = []
    for value in lowered:
        if value in benign:
            mapped.append(0)
        elif value in positives:
            mapped.append(1)
        else:
            mapped.append(1)
    return pd.Series(mapped, index=series.index, name=series.name)


def clean_dataframe(
    df,
    label_column: str,
    drop_columns: list[str],
    leakage_patterns: list[str],
    remove_duplicates: bool = True,
    categorical_max_unique: int = 30,
):
    np = require_package("numpy")
    pd = require_package("pandas")
    work = df.copy()
    if remove_duplicates:
        work = work.drop_duplicates().reset_index(drop=True)
    work = work.replace([np.inf, -np.inf], np.nan)

    leakage = leakage_candidates(work.columns, leakage_patterns)
    to_drop = set(drop_columns or []) | set(leakage)
    to_drop.discard(label_column)
    existing_drop = [c for c in to_drop if c in work.columns]
    if existing_drop:
        work = work.drop(columns=existing_drop)

    categorical_cols = []
    high_cardinality_cols = []
    for col in list(work.columns):
        if col == label_column:
            continue
        if work[col].dtype == object:
            converted = pd.to_numeric(work[col], errors="coerce")
            if converted.notna().mean() > 0.95:
                work[col] = converted
            else:
                nunique = work[col].nunique(dropna=True)
                if nunique <= categorical_max_unique:
                    categorical_cols.append(col)
                else:
                    high_cardinality_cols.append(col)

    numeric_cols = [c for c in work.select_dtypes(include=["number"]).columns if c != label_column]
    for col in numeric_cols:
        work[col] = work[col].fillna(work[col].median())

    if high_cardinality_cols:
        work = work.drop(columns=high_cardinality_cols)
        existing_drop.extend(high_cardinality_cols)
    if categorical_cols:
        work[categorical_cols] = work[categorical_cols].fillna("missing").astype(str)
        work = pd.get_dummies(work, columns=categorical_cols, drop_first=False, dtype=int)

    non_numeric = [c for c in work.columns if c != label_column and not pd.api.types.is_numeric_dtype(work[c])]
    if non_numeric:
        work = work.drop(columns=non_numeric)
        existing_drop.extend(non_numeric)

    return work, sorted(existing_drop)


def split_and_scale(df, label_column: str, test_size: float, validation_size: float, random_state: int, scaler: str = "minmax"):
    np = require_package("numpy")
    train_test_split = require_package("sklearn.model_selection").train_test_split
    MinMaxScaler = require_package("sklearn.preprocessing").MinMaxScaler
    StandardScaler = require_package("sklearn.preprocessing").StandardScaler

    y = df[label_column]
    x = df.drop(columns=[label_column])
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )
    valid_fraction = validation_size / max(1e-12, 1 - test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full, y_train_full, test_size=valid_fraction, stratify=y_train_full, random_state=random_state
    )

    scaler_obj = StandardScaler() if scaler == "standard" else MinMaxScaler()
    x_train_arr = scaler_obj.fit_transform(x_train)
    x_valid_arr = scaler_obj.transform(x_valid)
    x_test_arr = scaler_obj.transform(x_test)
    return x_train_arr, x_valid_arr, x_test_arr, y_train.to_numpy(), y_valid.to_numpy(), y_test.to_numpy(), list(x.columns)


def apply_resampling(x_train, y_train, method: ResamplingMethod, random_state: int):
    if method == "none":
        return x_train, y_train
    if method == "smote":
        SMOTE = require_package("imblearn.over_sampling").SMOTE
        return SMOTE(random_state=random_state).fit_resample(x_train, y_train)
    if method == "borderline_smote":
        BorderlineSMOTE = require_package("imblearn.over_sampling").BorderlineSMOTE
        return BorderlineSMOTE(random_state=random_state).fit_resample(x_train, y_train)
    if method == "adasyn":
        ADASYN = require_package("imblearn.over_sampling").ADASYN
        return ADASYN(random_state=random_state).fit_resample(x_train, y_train)
    raise ValueError(f"Unknown resampling method: {method}")


def preprocess_from_config(
    config: dict,
    resampling_method: ResamplingMethod = "none",
    binary: bool = True,
    sample_rows: int | None = None,
) -> PreprocessArtifacts:
    pd = require_package("pandas")
    dataset = config["dataset"]
    split = config.get("split", {})
    prep = config.get("preprocessing", {})
    df = load_csv_dataset(dataset["raw_path"], dataset.get("file_pattern", "*.csv"), sample_rows=sample_rows)
    label_column = dataset["label_column"]
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found. Columns: {list(df.columns)[:20]}")
    if binary:
        df[label_column] = normalize_binary_labels(df[label_column], dataset.get("positive_labels", []))

    cleaned, dropped = clean_dataframe(
        df,
        label_column=label_column,
        drop_columns=dataset.get("drop_columns", []),
        leakage_patterns=dataset.get("leakage_patterns", []),
        remove_duplicates=bool(prep.get("duplicate_removal", True)),
        categorical_max_unique=int(prep.get("categorical_max_unique", 30)),
    )
    if prep.get("remove_zero_variance", True):
        numeric = cleaned.drop(columns=[label_column]).select_dtypes(include=["number"])
        zero_var = [c for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1]
        if zero_var:
            cleaned = cleaned.drop(columns=zero_var)
            dropped.extend(zero_var)

    x_train, x_valid, x_test, y_train, y_valid, y_test, features = split_and_scale(
        cleaned,
        label_column=label_column,
        test_size=float(split.get("test_size", 0.2)),
        validation_size=float(split.get("validation_size", 0.15)),
        random_state=int(split.get("random_state", 42)),
        scaler=prep.get("scaler", "minmax"),
    )
    x_train, y_train = apply_resampling(x_train, y_train, resampling_method, int(split.get("random_state", 42)))
    out_dir = ensure_dir(dataset["processed_path"])
    pd.DataFrame(x_train, columns=features).assign(**{label_column: y_train}).to_csv(out_dir / f"train_{resampling_method}.csv", index=False)
    pd.DataFrame(x_valid, columns=features).assign(**{label_column: y_valid}).to_csv(out_dir / "valid.csv", index=False)
    pd.DataFrame(x_test, columns=features).assign(**{label_column: y_test}).to_csv(out_dir / "test.csv", index=False)

    return PreprocessArtifacts(x_train, x_valid, x_test, y_train, y_valid, y_test, features, sorted(set(dropped)))
