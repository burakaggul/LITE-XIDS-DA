from __future__ import annotations

from pathlib import Path

from litexids.utils.config import require_package


def find_csv_files(raw_path: str | Path, pattern: str = "*.csv") -> list[Path]:
    path = Path(raw_path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    files = sorted(path.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No data files found in {path} with pattern {pattern}")
    return files


def load_csv_dataset(raw_path: str | Path, pattern: str = "*.csv", sample_rows: int | None = None):
    pd = require_package("pandas")
    frames = []
    for file in find_csv_files(raw_path, pattern):
        if file.suffix.lower() == ".parquet":
            frame = pd.read_parquet(file)
        else:
            frame = pd.read_csv(file)
        if sample_rows is not None and len(frame) > sample_rows:
            frame = frame.sample(n=sample_rows, random_state=42)
        frames.append(frame)
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)
