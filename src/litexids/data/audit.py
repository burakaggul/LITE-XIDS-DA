from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from litexids.data.loading import load_csv_dataset
from litexids.utils.config import ensure_dir, require_package


@dataclass(frozen=True)
class AuditResult:
    dataset_name: str
    rows: int
    columns: int
    duplicate_rows: int
    missing_values: int
    infinite_values: int
    zero_variance_features: list[str]
    leakage_risk_features: list[str]


def leakage_candidates(columns: Iterable[str], patterns: Iterable[str]) -> list[str]:
    regexes = [re.compile(p, re.IGNORECASE) for p in patterns]
    out = []
    for col in columns:
        if any(r.search(col) for r in regexes):
            out.append(col)
    return sorted(set(out))


def audit_dataframe(df, dataset_name: str, label_column: str, leakage_patterns: list[str]) -> AuditResult:
    np = require_package("numpy")
    numeric = df.select_dtypes(include=["number"])
    zero_variance = []
    for col in numeric.columns:
        if col == label_column:
            continue
        if numeric[col].nunique(dropna=True) <= 1:
            zero_variance.append(col)

    infinite_values = 0
    if not numeric.empty:
        infinite_values = int(np.isinf(numeric.to_numpy(dtype=float, copy=True)).sum())

    return AuditResult(
        dataset_name=dataset_name,
        rows=int(len(df)),
        columns=int(len(df.columns)),
        duplicate_rows=int(df.duplicated().sum()),
        missing_values=int(df.isna().sum().sum()),
        infinite_values=infinite_values,
        zero_variance_features=sorted(zero_variance),
        leakage_risk_features=leakage_candidates(df.columns, leakage_patterns),
    )


def write_audit_outputs(df, result: AuditResult, label_column: str, output_dir: str | Path) -> None:
    pd = require_package("pandas")
    out = ensure_dir(output_dir)

    summary = pd.DataFrame(
        [
            {
                "dataset": result.dataset_name,
                "rows": result.rows,
                "columns": result.columns,
                "duplicate_rows": result.duplicate_rows,
                "missing_values": result.missing_values,
                "infinite_values": result.infinite_values,
                "zero_variance_feature_count": len(result.zero_variance_features),
                "leakage_risk_feature_count": len(result.leakage_risk_features),
            }
        ]
    )
    summary.to_csv(out / "dataset_summary.csv", index=False)

    if label_column in df.columns:
        df[label_column].value_counts(dropna=False).rename_axis("label").reset_index(name="count").to_csv(
            out / "class_distribution.csv", index=False
        )

    pd.DataFrame({"feature": result.zero_variance_features}).to_csv(out / "zero_variance_features.csv", index=False)
    pd.DataFrame({"feature": result.leakage_risk_features}).to_csv(out / "leakage_risk_features.csv", index=False)
    pd.DataFrame([{"duplicate_rows": result.duplicate_rows, "duplicate_ratio": result.duplicate_rows / max(result.rows, 1)}]).to_csv(
        out / "duplicate_report.csv", index=False
    )
    pd.DataFrame([{"missing_values": result.missing_values, "infinite_values": result.infinite_values}]).to_csv(
        out / "missing_inf_report.csv", index=False
    )
    with (out / "audit_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result.__dict__, handle, indent=2)


def audit_from_config(config: dict, sample_rows: int | None = None) -> AuditResult:
    dataset = config["dataset"]
    df = load_csv_dataset(dataset["raw_path"], dataset.get("file_pattern", "*.csv"), sample_rows=sample_rows)
    result = audit_dataframe(
        df,
        dataset_name=dataset["name"],
        label_column=dataset["label_column"],
        leakage_patterns=dataset.get("leakage_patterns", []),
    )
    out = Path(config.get("output", {}).get("root", "results")) / "data_audit" / dataset["name"]
    write_audit_outputs(df, result, dataset["label_column"], out)
    return result
