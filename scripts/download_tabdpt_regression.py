"""Download TabDPT's 29 real regression training tables from OpenML."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import sparse
from sklearn.datasets import fetch_openml


# Intersection of TabDPT's no-leak training split and its regression dataset list:
# https://github.com/layer6ai-labs/TabDPT-training/tree/main/data_splits
DATASET_IDS = (
    3050, 3277, 4549, 41540, 41980, 42563, 42572, 42705, 42724, 42727,
    42728, 42730, 43071, 44055, 44056, 44061, 44063, 44065, 44068, 44069,
    44136, 44137, 44145, 45032, 45041, 45043, 45045, 45046, 45047,
)


def encode_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns = []
    names = []
    for name in frame.columns:
        series = frame[name]
        if is_numeric_dtype(series.dtype):
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=True)
        else:
            values = pd.factorize(series, sort=True, use_na_sentinel=True)[0].astype(np.float64)
            values[values < 0] = np.nan
        values[~np.isfinite(values)] = np.nan
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        values[np.isnan(values)] = np.median(finite)
        if np.unique(values).size < 2:
            continue
        columns.append(values.astype(np.float32))
        names.append(str(name))
    if not columns:
        raise ValueError("table has no non-constant usable features")
    return np.column_stack(columns), names


def select_feature_rows(data, valid: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    if sparse.issparse(data):
        return pd.DataFrame(data[valid].toarray(), columns=feature_names)
    if isinstance(data, pd.DataFrame):
        return data.loc[valid].reset_index(drop=True)
    return pd.DataFrame(np.asarray(data)[valid], columns=feature_names)


def save_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def download_dataset(dataset_id: int, output_dir: Path, cache_dir: Path) -> dict:
    destination = output_dir / str(dataset_id)
    metadata_path = destination / "metadata.json"
    if (destination / "X.npy").is_file() and (destination / "y.npy").is_file() and metadata_path.is_file():
        return json.loads(metadata_path.read_text())

    dataset = fetch_openml(data_id=dataset_id, as_frame="auto", parser="auto", data_home=cache_dir)
    target = dataset.target
    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError(f"dataset {dataset_id} has {target.shape[1]} targets")
        target = target.iloc[:, 0]
    y = np.asarray(pd.to_numeric(target, errors="coerce"), dtype=np.float64)
    valid = np.isfinite(y)
    frame = select_feature_rows(dataset.data, valid, dataset.feature_names)
    X, feature_names = encode_features(frame)
    y = y[valid].astype(np.float32)

    destination.mkdir(parents=True, exist_ok=True)
    save_array(destination / "X.npy", X)
    save_array(destination / "y.npy", y)
    details = getattr(dataset, "details", {}) or {}
    metadata = {
        "dataset_id": dataset_id,
        "name": str(details.get("name", getattr(dataset, "name", dataset_id))),
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "feature_names": feature_names,
        "openml_version": details.get("version"),
        "licence": details.get("licence"),
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="OpenML download cache; defaults to OPENML_CACHE_DIR or a sibling of output_dir.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = Path(os.environ.get("OPENML_CACHE_DIR", output_dir.parent / ".openml-cache"))
    cache_dir = cache_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    failures = []
    for index, dataset_id in enumerate(DATASET_IDS, start=1):
        print(f"[{index:02d}/{len(DATASET_IDS)}] OpenML dataset {dataset_id}", flush=True)
        try:
            records.append(download_dataset(dataset_id, output_dir, cache_dir))
        except Exception as error:
            failures.append((dataset_id, str(error)))
            print(f"  failed: {error}", flush=True)

    manifest_path = output_dir / "manifest.csv"
    temporary = manifest_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset_id", "name", "rows", "features", "openml_version", "licence"),
        )
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in records)
    os.replace(temporary, manifest_path)

    print(f"Downloaded {len(records)}/{len(DATASET_IDS)} datasets to {output_dir}")
    if failures:
        failed = ", ".join(str(dataset_id) for dataset_id, _ in failures)
        raise SystemExit(f"Failed dataset IDs: {failed}; rerun to retry them")


if __name__ == "__main__":
    main()
