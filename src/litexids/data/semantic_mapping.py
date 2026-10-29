from __future__ import annotations

from litexids.data.preprocessing import normalize_binary_labels
from litexids.utils.config import require_package


def _col(df, *names: str):
    lookup = {c.lower(): c for c in df.columns}
    for name in names:
        found = lookup.get(name.lower())
        if found is not None:
            return found
    return None


def _numeric(df, name: str | None, default: float = 0.0):
    pd = require_package("pandas")
    if name is None or name not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default).astype(float)


def _protocol_number(series):
    pd = require_package("pandas")
    mapping = {
        "tcp": 6,
        "udp": 17,
        "icmp": 1,
        "arp": 2054,
        "dns": 17,
        "http": 6,
        "https": 6,
    }
    as_num = pd.to_numeric(series, errors="coerce")
    if as_num.notna().mean() > 0.8:
        return as_num.fillna(0).astype(float)
    return series.astype(str).str.lower().map(mapping).fillna(0).astype(float)


def build_semantic_features(df, label_column: str, dataset_name: str, positive_labels: list[str] | None = None):
    pd = require_package("pandas")
    np = require_package("numpy")
    out = pd.DataFrame(index=df.index)

    duration = _numeric(df, _col(df, "flow_duration", "duration", "Duration"))
    if dataset_name.lower().startswith("cic"):
        protocol = _numeric(df, _col(df, "Protocol Type", "protocol", "proto"))
        bytes_total = _numeric(df, _col(df, "Tot size", "Tot sum"))
        packet_total = _numeric(df, _col(df, "Number"))
        bytes_mean = _numeric(df, _col(df, "AVG"))
        bytes_min = _numeric(df, _col(df, "Min"))
        bytes_max = _numeric(df, _col(df, "Max"))
        rate = _numeric(df, _col(df, "Rate", "Srate"))
        tcp = _numeric(df, _col(df, "TCP"))
        udp = _numeric(df, _col(df, "UDP"))
        icmp = _numeric(df, _col(df, "ICMP"))
        iat_proxy = _numeric(df, _col(df, "IAT"))
    else:
        proto_col = _col(df, "proto", "protocol", "Protocol Type")
        protocol = _protocol_number(df[proto_col]) if proto_col else pd.Series(0.0, index=df.index)
        src_bytes = _numeric(df, _col(df, "src_bytes", "src_ip_bytes"))
        dst_bytes = _numeric(df, _col(df, "dst_bytes", "dst_ip_bytes"))
        bytes_total = src_bytes + dst_bytes
        src_pkts = _numeric(df, _col(df, "src_pkts"))
        dst_pkts = _numeric(df, _col(df, "dst_pkts"))
        packet_total = src_pkts + dst_pkts
        bytes_mean = bytes_total / packet_total.replace(0, np.nan)
        bytes_mean = bytes_mean.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bytes_min = pd.concat([src_bytes, dst_bytes], axis=1).min(axis=1)
        bytes_max = pd.concat([src_bytes, dst_bytes], axis=1).max(axis=1)
        rate = bytes_total / duration.replace(0, np.nan)
        rate = rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        proto_text = df[proto_col].astype(str).str.lower() if proto_col else pd.Series("", index=df.index)
        tcp = proto_text.eq("tcp").astype(float)
        udp = proto_text.eq("udp").astype(float)
        icmp = proto_text.eq("icmp").astype(float)
        iat_proxy = duration / packet_total.replace(0, np.nan)
        iat_proxy = iat_proxy.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["duration"] = duration
    out["protocol_number"] = protocol
    out["packet_total"] = packet_total
    out["bytes_total"] = bytes_total
    out["bytes_mean"] = bytes_mean
    out["bytes_min"] = bytes_min
    out["bytes_max"] = bytes_max
    out["rate"] = rate
    out["tcp"] = tcp
    out["udp"] = udp
    out["icmp"] = icmp
    out["iat_proxy"] = iat_proxy
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[label_column] = normalize_binary_labels(df[label_column], positive_labels)
    return out


def deduplicate_semantic_features(df, label_column: str):
    """Remove duplicate semantic rows before source/target split construction.

    Exact semantic duplicates with the same label are collapsed to one row. If the
    same semantic feature vector appears with conflicting labels, all conflicting
    rows are removed because they cannot support a leakage-safe supervised split.
    """
    feature_cols = [c for c in df.columns if c != label_column]
    exact = df.drop_duplicates().reset_index(drop=True)
    conflict_mask = exact.duplicated(subset=feature_cols, keep=False)
    if not conflict_mask.any():
        return exact
    conflicts = exact.loc[conflict_mask]
    conflicting_keys = (
        conflicts.groupby(feature_cols, dropna=False)[label_column]
        .nunique()
        .reset_index(name="_label_count")
    )
    conflicting_keys = conflicting_keys[conflicting_keys["_label_count"] > 1].drop(columns=["_label_count"])
    if conflicting_keys.empty:
        return exact.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)
    marked = exact.merge(conflicting_keys.assign(_drop_conflict=True), on=feature_cols, how="left")
    return marked[marked["_drop_conflict"].isna()].drop(columns=["_drop_conflict"]).drop_duplicates(
        subset=feature_cols, keep="first"
    ).reset_index(drop=True)
