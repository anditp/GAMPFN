#!/usr/bin/env python3
"""Compare every local GAMPFN checkpoint with SplineGAM and TabICLv2.

Edit ``ELECTRICITY_DATA`` if needed, then run:

    .venv/bin/python scripts/compare_stage2_models.py

The script evaluates the models on the shared synthetic GAM-family suite and
on two one-year electricity test windows, each with exactly two preceding
years of per-instant context. Results and two summary figures are written to a
timestamped directory under ``results/``.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluate_gampfn import FAMILIES, ReusableTabICLRegressor, make_task, score
from tabicl._model.additive import AdditivePerceiver


ELECTRICITY_DATA = Path(
    "/Users/andrei/Desktop/TSFM/data_preprocessed_final_enhanced_2.csv"
)
CHECKPOINTS = (
    ("TabICL-GAM-v0.0", "tabicl", REPO_ROOT / "checkpoints/v0.0.ckpt"),
    ("GAMPFN-v0.1-stage1", "additive", REPO_ROOT / "checkpoints/v0.1_stage1.ckpt"),
    (
        "GAMPFN-v0.1-stage2-mixed",
        "additive",
        REPO_ROOT / "checkpoints/v0.1_stage2_mixed.ckpt",
    ),
    (
        "GAMPFN-v0.1-stage2-real",
        "additive",
        REPO_ROOT / "checkpoints/v0.1_stage2_real.ckpt",
    ),
    (
        "GAMPFN-v0.1-stage2-tabicl",
        "additive",
        REPO_ROOT / "checkpoints/v0.1_stage2_tabicl.ckpt",
    ),
    (
        "GAMPFN-v1-stage2-mixed",
        "additive",
        REPO_ROOT / "checkpoints/v1_stage2_mixed.ckpt",
    ),
)
TABICLV2_VERSION = "tabicl-regressor-v2-20260212.ckpt"
ELECTRICITY_SPLITS = (
    {
        "name": "2022-2023",
        "context_start": pd.Timestamp("2020-07-01"),
        "test_start": pd.Timestamp("2022-07-01"),
        "test_end": pd.Timestamp("2023-07-01"),
    },
    {
        "name": "2023-2024",
        "context_start": pd.Timestamp("2021-07-01"),
        "test_start": pd.Timestamp("2023-07-01"),
        "test_end": pd.Timestamp("2024-07-01"),
    },
)
EXCLUDED_FEATURES = {"Consommation", "Date_local", "Instant"}
GAM_CATEGORICAL_FEATURES = ("JourFerie", "HeureEte", "DayOfWeek")
GAM_N_KNOTS = 10
GAM_ALPHAS = np.logspace(-4, 4, 17)

# Screening defaults. Increase these two values for the final comparison.
N_ESTIMATORS = 2
SYNTHETIC_SEEDS = 10
SYNTHETIC_TRAIN_SIZES = (128, 512, 800)
SYNTHETIC_FEATURE_COUNTS = (10, 50)
SYNTHETIC_TEST_SIZE = 200
SYNTHETIC_NOISE = 0.2
PREDICTION_CHUNK_SIZE = 256
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def amp_context():
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class AdditiveCheckpointRegressor:
    """Small fit/predict adapter for v0.1 and v1 additive checkpoints."""

    def __init__(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_type") != "additive":
            raise ValueError(f"{path} is not an additive checkpoint")
        self.model = AdditivePerceiver(**checkpoint["config"])
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval().to(DEVICE)
        self.step = checkpoint.get("curr_step")
        self.context = None

    def fit(self, x, y):
        x_tensor = torch.as_tensor(
            np.asarray(x, dtype=np.float32), device=DEVICE
        ).unsqueeze(0)
        y_tensor = torch.as_tensor(
            np.asarray(y, dtype=np.float32), device=DEVICE
        ).unsqueeze(0)
        with torch.inference_mode(), amp_context():
            self.context = self.model.encode_context(x_tensor, y_tensor)
        return self

    def predict(self, x) -> np.ndarray:
        if self.context is None:
            raise RuntimeError("fit must be called before predict")
        x_array = np.asarray(x, dtype=np.float32)
        predictions = []
        with torch.inference_mode(), amp_context():
            for start in range(0, len(x_array), PREDICTION_CHUNK_SIZE):
                query = torch.as_tensor(
                    x_array[start : start + PREDICTION_CHUNK_SIZE], device=DEVICE
                ).unsqueeze(0)
                effects, interaction = self.model.predict_components(self.context, query)
                if self.model.probabilistic:
                    median = self.model.num_quantiles // 2
                    prediction = (
                        self.context.quantile_intercept[..., median]
                        + effects[..., median].sum(dim=-1)
                    )
                else:
                    prediction = self.context.y_mean + effects.sum(dim=-1) + interaction
                predictions.append(prediction[0].float().cpu().numpy())
        return np.concatenate(predictions)


def load_checkpoint_model(kind: str, path: Path):
    if kind == "additive":
        return AdditiveCheckpointRegressor(path)
    if kind == "tabicl":
        return ReusableTabICLRegressor(
            model_path=path,
            allow_auto_download=False,
            n_estimators=N_ESTIMATORS,
            random_state=0,
            device=str(DEVICE),
            kv_cache="repr",
        )
    raise ValueError(f"Unknown checkpoint kind: {kind}")


def load_released_tabicl():
    return ReusableTabICLRegressor(
        model_path=None,
        allow_auto_download=True,
        checkpoint_version=TABICLV2_VERSION,
        n_estimators=N_ESTIMATORS,
        random_state=0,
        device=str(DEVICE),
        kv_cache="repr",
    )


def make_synthetic_spline_gam():
    return make_pipeline(
        SplineTransformer(
            n_knots=GAM_N_KNOTS,
            degree=3,
            include_bias=False,
            extrapolation="linear",
        ),
        RidgeCV(alphas=GAM_ALPHAS),
    )


def make_electricity_spline_gam(features: list[str]):
    categorical = [name for name in GAM_CATEGORICAL_FEATURES if name in features]
    numeric = [name for name in features if name not in categorical]
    transformers = [
        (
            "smooths",
            SplineTransformer(
                n_knots=GAM_N_KNOTS,
                degree=3,
                include_bias=False,
                extrapolation="linear",
            ),
            numeric,
        )
    ]
    if categorical:
        transformers.append(
            (
                "categoricals",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    return make_pipeline(
        ColumnTransformer(transformers, sparse_threshold=0.0),
        RidgeCV(alphas=GAM_ALPHAS),
    )


def load_electricity_tasks() -> tuple[list[dict[str, object]], list[str]]:
    data = pd.read_csv(ELECTRICITY_DATA, parse_dates=["Date_local"])
    begin = min(split["context_start"] for split in ELECTRICITY_SPLITS)
    end = max(split["test_end"] for split in ELECTRICITY_SPLITS)
    data = data[(data["Date_local"] >= begin) & (data["Date_local"] < end)].copy()
    if "JourFerie" in data:
        data["JourFerie"] = data["JourFerie"].astype(int)

    features = [column for column in data.columns if column not in EXCLUDED_FEATURES]
    non_numeric = data[features].select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise TypeError(f"Electricity features must be numeric; found {non_numeric}")
    if data[features + ["Consommation"]].isna().any().any():
        raise ValueError("Electricity data contain missing feature or target values")

    tasks = []
    for split in ELECTRICITY_SPLITS:
        for hour in range(24):
            for minute in (0, 15, 30, 45):
                instant = (data["Date_local"].dt.hour == hour) & (
                    data["Date_local"].dt.minute == minute
                )
                train = data[
                    instant
                    & (data["Date_local"] >= split["context_start"])
                    & (data["Date_local"] < split["test_start"])
                ]
                test = data[
                    instant
                    & (data["Date_local"] >= split["test_start"])
                    & (data["Date_local"] < split["test_end"])
                ]
                if train.empty or test.empty:
                    raise ValueError(
                        f"Empty {split['name']} electricity split at {hour:02d}:{minute:02d}"
                    )
                tasks.append(
                    {
                        "split": split["name"],
                        "hour": hour,
                        "minute": minute,
                        "x_train": train[features],
                        "y_train": train["Consommation"].to_numpy(dtype=np.float32),
                        "x_test": test[features],
                        "y_test": test["Consommation"].to_numpy(dtype=np.float32),
                        "dates": test["Date_local"].to_numpy(),
                    }
                )
    return tasks, features


def evaluate_electricity(
    name: str,
    model,
    tasks: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows = []
    prediction_rows = []
    for index, task in enumerate(tasks, start=1):
        started = time.perf_counter()
        prediction = model.fit(task["x_train"], task["y_train"]).predict(task["x_test"])
        elapsed = time.perf_counter() - started
        actual = task["y_test"]
        error = prediction - actual
        rmse = float(np.sqrt(np.mean(error**2)))
        metric_rows.append(
            {
                "split": task["split"],
                "model": name,
                "hour": task["hour"],
                "minute": task["minute"],
                "n_train": len(task["y_train"]),
                "n_test": len(actual),
                "rmse": rmse,
                "mae": float(np.mean(np.abs(error))),
                "nrmse": rmse / max(float(np.std(actual, ddof=1)), 1e-8),
                "seconds": elapsed,
            }
        )
        prediction_rows.extend(
            {
                "split": task["split"],
                "model": name,
                "date": date,
                "hour": task["hour"],
                "minute": task["minute"],
                "actual": target,
                "prediction": estimate,
            }
            for date, target, estimate in zip(task["dates"], actual, prediction)
        )
        print(
            f"  electricity [{index:03d}/{len(tasks)}] {task['split']} "
            f"{task['hour']:02d}:{task['minute']:02d} RMSE={rmse:.2f}",
            flush=True,
        )
    return metric_rows, prediction_rows


def evaluate_synthetic(name: str, model) -> list[dict[str, object]]:
    rows = []
    total = (
        len(FAMILIES)
        * len(SYNTHETIC_TRAIN_SIZES)
        * len(SYNTHETIC_FEATURE_COUNTS)
        * SYNTHETIC_SEEDS
    )
    index = 0
    for family in FAMILIES:
        for n_train in SYNTHETIC_TRAIN_SIZES:
            for n_features in SYNTHETIC_FEATURE_COUNTS:
                for seed in range(SYNTHETIC_SEEDS):
                    index += 1
                    x_train, y_train, x_test, y_test, clean, correlated = make_task(
                        family,
                        seed,
                        n_train,
                        SYNTHETIC_TEST_SIZE,
                        n_features,
                        SYNTHETIC_NOISE,
                    )
                    started = time.perf_counter()
                    prediction = model.fit(x_train, y_train).predict(x_test)
                    elapsed = time.perf_counter() - started
                    nrmse_f, nrmse_y, nmae_f = score(prediction, clean, y_test)
                    rows.append(
                        {
                            "model": name,
                            "family": family,
                            "seed": seed,
                            "n_train": n_train,
                            "n_features": n_features,
                            "correlated": correlated,
                            "nrmse_f": nrmse_f,
                            "nrmse_y": nrmse_y,
                            "nmae_f": nmae_f,
                            "seconds": elapsed,
                        }
                    )
                    print(
                        f"  synthetic [{index:03d}/{total}] {family} n={n_train} "
                        f"d={n_features} seed={seed} NRMSE={nrmse_f:.3f}",
                        flush=True,
                    )
    return rows


def electricity_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, name), frame in predictions.groupby(["split", "model"], sort=False):
        error = frame["prediction"] - frame["actual"]
        rmse = float(np.sqrt(np.mean(error**2)))
        rows.append(
            {
                "split": split,
                "model": name,
                "rmse": rmse,
                "mae": float(np.mean(np.abs(error))),
                "nrmse": rmse / max(float(frame["actual"].std(ddof=1)), 1e-8),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "rmse"]).reset_index(drop=True)


def synthetic_summary(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["family", "seed", "n_train", "n_features"]
    ranked = results.copy()
    ranked["rank"] = ranked.groupby(keys)["nrmse_f"].rank(method="average")
    overall = (
        ranked.groupby("model", as_index=False)
        .agg(
            median_nrmse_f=("nrmse_f", "median"),
            mean_rank=("rank", "mean"),
            win_rate=("rank", lambda values: float(np.mean(values == 1))),
        )
        .sort_values(["mean_rank", "median_nrmse_f"])
        .reset_index(drop=True)
    )
    by_family = (
        ranked.groupby(["family", "model"], as_index=False)["nrmse_f"]
        .median()
        .rename(columns={"nrmse_f": "median_nrmse_f"})
    )
    return overall, by_family


def pairwise_wins(
    results: pd.DataFrame,
    task_keys: list[str],
    score_column: str,
) -> pd.DataFrame:
    scores = results.pivot(index=task_keys, columns="model", values=score_column)
    rows = []
    for model in scores.columns:
        for opponent in scores.columns:
            if model == opponent:
                continue
            valid = scores[[model, opponent]].dropna()
            rows.append(
                {
                    "model": model,
                    "opponent": opponent,
                    "win_rate": float((valid[model] < valid[opponent]).mean()),
                    "median_improvement_pct": float(
                        np.median(
                            100
                            * (valid[opponent] - valid[model])
                            / valid[opponent].clip(lower=1e-12)
                        )
                    ),
                    "n_tasks": len(valid),
                }
            )
    return pd.DataFrame(rows)


def make_plots(
    electricity_metrics: pd.DataFrame,
    electricity_overall: pd.DataFrame,
    synthetic_by_family: pd.DataFrame,
    output_dir: Path,
) -> None:
    pivot = synthetic_by_family.pivot(
        index="family", columns="model", values="median_nrmse_f"
    )
    axis = pivot.plot.bar(figsize=(15, 6), rot=20)
    axis.set(
        title="Synthetic suite: median clean-signal NRMSE",
        xlabel="Family",
        ylabel="NRMSE",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    axis.figure.tight_layout()
    axis.figure.savefig(output_dir / "synthetic_summary.png", dpi=160)
    plt.close(axis.figure)

    fig, axes = plt.subplots(
        len(ELECTRICITY_SPLITS), 2, figsize=(16, 10), constrained_layout=True
    )
    for row, split in enumerate(item["name"] for item in ELECTRICITY_SPLITS):
        overall = electricity_overall[electricity_overall["split"] == split]
        axes[row, 0].bar(overall["model"], overall["rmse"])
        axes[row, 0].tick_params(axis="x", rotation=35, labelsize=8)
        axes[row, 0].set(title=f"{split}: overall RMSE", ylabel="RMSE (MW)")
        metrics = electricity_metrics[electricity_metrics["split"] == split]
        for name, frame in metrics.groupby("model", sort=False):
            instant = frame["hour"] + frame["minute"] / 60
            axes[row, 1].plot(instant, frame["rmse"], label=name, linewidth=1.2)
        axes[row, 1].set(
            title=f"{split}: RMSE by instant",
            xlabel="Hour",
            ylabel="RMSE (MW)",
            xlim=(0, 24),
        )
        axes[row, 1].grid(alpha=0.2)
        axes[row, 1].legend(fontsize=7, ncol=2)
    fig.savefig(output_dir / "electricity_summary.png", dpi=160)
    plt.close(fig)


def write_tables(
    output_dir: Path,
    synthetic_rows: list[dict[str, object]],
    electricity_metric_rows: list[dict[str, object]],
    electricity_prediction_rows: list[dict[str, object]],
) -> None:
    pd.DataFrame(synthetic_rows).to_csv(output_dir / "synthetic_tasks.csv", index=False)
    pd.DataFrame(electricity_metric_rows).to_csv(
        output_dir / "electricity_by_instant.csv", index=False
    )
    pd.DataFrame(electricity_prediction_rows).to_csv(
        output_dir / "electricity_predictions.csv", index=False
    )


def self_check() -> None:
    task = make_task("smooth", 0, 32, 16, 10, SYNTHETIC_NOISE)
    assert task[0].shape == (32, 10) and task[2].shape == (16, 10)
    assert len(CHECKPOINTS) == 6
    assert EXCLUDED_FEATURES == {"Consommation", "Date_local", "Instant"}
    for split in ELECTRICITY_SPLITS:
        assert split["context_start"] == split["test_start"] - pd.DateOffset(years=2)
        assert split["test_end"] == split["test_start"] + pd.DateOffset(years=1)
    print("Static comparison setup passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return

    missing = [
        path
        for path in [ELECTRICITY_DATA, *(item[2] for item in CHECKPOINTS)]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(map(str, missing)))

    output_dir = REPO_ROOT / "results" / datetime.now().strftime(
        "checkpoint_comparison_%Y%m%d_%H%M%S"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    electricity_tasks, features = load_electricity_tasks()
    print(
        f"Device: {DEVICE}; electricity features: {len(features)}; "
        f"tasks: {len(electricity_tasks)}; output: {output_dir}",
        flush=True,
    )

    synthetic_rows = []
    electricity_metric_rows = []
    electricity_prediction_rows = []
    model_specs = [*CHECKPOINTS, ("TabICLv2", "released", None)]
    for name, kind, path in model_specs:
        model = (
            load_released_tabicl()
            if kind == "released"
            else load_checkpoint_model(kind, path)
        )
        print(f"\n{name}: {path or TABICLV2_VERSION}", flush=True)
        synthetic_rows.extend(evaluate_synthetic(name, model))
        metrics, predictions = evaluate_electricity(name, model, electricity_tasks)
        electricity_metric_rows.extend(metrics)
        electricity_prediction_rows.extend(predictions)
        write_tables(
            output_dir,
            synthetic_rows,
            electricity_metric_rows,
            electricity_prediction_rows,
        )
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("\nSplineGAM", flush=True)
    synthetic_rows.extend(evaluate_synthetic("SplineGAM", make_synthetic_spline_gam()))
    metrics, predictions = evaluate_electricity(
        "SplineGAM", make_electricity_spline_gam(features), electricity_tasks
    )
    electricity_metric_rows.extend(metrics)
    electricity_prediction_rows.extend(predictions)
    write_tables(
        output_dir,
        synthetic_rows,
        electricity_metric_rows,
        electricity_prediction_rows,
    )

    electricity_metrics = pd.DataFrame(electricity_metric_rows)
    electricity_predictions = pd.DataFrame(electricity_prediction_rows)
    synthetic_results = pd.DataFrame(synthetic_rows)
    electricity_overall = electricity_summary(electricity_predictions)
    synthetic_overall, synthetic_by_family = synthetic_summary(synthetic_results)

    electricity_overall.to_csv(output_dir / "electricity_summary.csv", index=False)
    synthetic_overall.to_csv(output_dir / "synthetic_summary.csv", index=False)
    synthetic_by_family.to_csv(output_dir / "synthetic_by_family.csv", index=False)
    pairwise_wins(
        electricity_metrics,
        ["split", "hour", "minute"],
        "rmse",
    ).to_csv(output_dir / "electricity_pairwise_wins.csv", index=False)
    pairwise_wins(
        synthetic_results,
        ["family", "seed", "n_train", "n_features"],
        "nrmse_f",
    ).to_csv(output_dir / "synthetic_pairwise_wins.csv", index=False)
    make_plots(electricity_metrics, electricity_overall, synthetic_by_family, output_dir)

    print("\nElectricity leaderboard (lower is better):")
    print(electricity_overall.to_string(index=False))
    print("\nSynthetic leaderboard (lower rank and NRMSE are better):")
    print(synthetic_overall.to_string(index=False))
    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
