from __future__ import annotations

from dataclasses import dataclass

from litexids.utils.config import require_package


ROBUST_LOG_FEATURES = {
    "duration",
    "packet_total",
    "bytes_total",
    "bytes_mean",
    "bytes_min",
    "bytes_max",
    "rate",
    "iat_proxy",
}


@dataclass
class RobustTransformArtifacts:
    lower: dict[str, float]
    upper: dict[str, float]
    log_features: list[str]


@dataclass
class DriftGateResult:
    selected_features: list[str]
    stats: object


@dataclass
class CoralTransform:
    source_mean: object
    target_mean: object
    transform: object


def _signed_log1p(values):
    np = require_package("numpy")
    return np.sign(values) * np.log1p(np.abs(values))


def fit_robust_transform(source_df, target_adapt_df, features: list[str], quantile: float = 0.995) -> RobustTransformArtifacts:
    pd = require_package("pandas")
    combined = pd.concat([source_df[features], target_adapt_df[features]], axis=0, ignore_index=True)
    log_features = [feature for feature in features if feature in ROBUST_LOG_FEATURES]
    work = combined.copy()
    for feature in log_features:
        work[feature] = _signed_log1p(work[feature].astype(float))
    lower = work.quantile(1.0 - quantile).to_dict()
    upper = work.quantile(quantile).to_dict()
    return RobustTransformArtifacts(lower=lower, upper=upper, log_features=log_features)


def apply_robust_transform(df, artifacts: RobustTransformArtifacts, features: list[str]):
    work = df[features].copy()
    for feature in artifacts.log_features:
        if feature in work.columns:
            work[feature] = _signed_log1p(work[feature].astype(float))
    for feature in features:
        if feature in work.columns:
            work[feature] = work[feature].clip(artifacts.lower[feature], artifacts.upper[feature])
    return work


def compute_drift_stats(source_df, target_df, features: list[str]):
    pd = require_package("pandas")
    rows = []
    for feature in features:
        source = pd.to_numeric(source_df[feature], errors="coerce").fillna(0.0).astype(float)
        target = pd.to_numeric(target_df[feature], errors="coerce").fillna(0.0).astype(float)
        ks_statistic = _ks_statistic(source.to_numpy(), target.to_numpy())
        rows.append(
            {
                "feature": feature,
                "ks_statistic": float(ks_statistic),
                "source_nonzero_rate": float((source.abs() > 1e-12).mean()),
                "target_nonzero_rate": float((target.abs() > 1e-12).mean()),
                "source_mean": float(source.mean()),
                "target_mean": float(target.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False).reset_index(drop=True)


def _ks_statistic(source, target) -> float:
    try:
        stats = require_package("scipy.stats")
        return float(stats.ks_2samp(source, target).statistic)
    except RuntimeError:
        np = require_package("numpy")
        values = np.sort(np.unique(np.concatenate([source, target])))
        if len(values) == 0:
            return 0.0
        source_sorted = np.sort(source)
        target_sorted = np.sort(target)
        source_cdf = np.searchsorted(source_sorted, values, side="right") / max(len(source_sorted), 1)
        target_cdf = np.searchsorted(target_sorted, values, side="right") / max(len(target_sorted), 1)
        return float(np.max(np.abs(source_cdf - target_cdf)))


def select_drift_gated_features(
    source_df,
    target_adapt_df,
    features: list[str],
    mode: str = "none",
    min_features: int = 3,
) -> DriftGateResult:
    stats = compute_drift_stats(source_df, target_adapt_df, features)
    if mode == "none":
        return DriftGateResult(selected_features=list(features), stats=stats)
    if mode.startswith("ks_le_"):
        threshold = float(mode.replace("ks_le_", ""))
        selected = stats.loc[stats["ks_statistic"] <= threshold, "feature"].tolist()
    elif mode.startswith("stable_top"):
        count = int(mode.replace("stable_top", ""))
        selected = stats.sort_values("ks_statistic", ascending=True).head(count)["feature"].tolist()
    else:
        raise ValueError(f"Unknown drift gate mode: {mode}")
    if len(selected) < min_features:
        selected = stats.sort_values("ks_statistic", ascending=True).head(min_features)["feature"].tolist()
    return DriftGateResult(selected_features=selected, stats=stats)


def fit_coral(source, target, regularization: float = 1e-3) -> CoralTransform:
    np = require_package("numpy")
    source_arr = np.asarray(source, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    source_mean = source_arr.mean(axis=0, keepdims=True)
    target_mean = target_arr.mean(axis=0, keepdims=True)
    source_centered = source_arr - source_mean
    target_centered = target_arr - target_mean
    source_cov = np.cov(source_centered, rowvar=False) + regularization * np.eye(source_arr.shape[1])
    target_cov = np.cov(target_centered, rowvar=False) + regularization * np.eye(target_arr.shape[1])
    source_inv_sqrt = _matrix_power_symmetric(source_cov, -0.5)
    target_sqrt = _matrix_power_symmetric(target_cov, 0.5)
    transform = source_inv_sqrt @ target_sqrt
    return CoralTransform(source_mean=source_mean, target_mean=target_mean, transform=transform)


def apply_coral(source, coral: CoralTransform):
    np = require_package("numpy")
    source_arr = np.asarray(source, dtype=float)
    return (source_arr - coral.source_mean) @ coral.transform + coral.target_mean


def _matrix_power_symmetric(matrix, power: float):
    np = require_package("numpy")
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-12, None)
    return vectors @ np.diag(values**power) @ vectors.T
