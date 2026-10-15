#!/usr/bin/env python3
"""Compare Stage 1 and the three Stage 2 GAMPFN mixtures.

Edit the paths below, then run:

    .venv/bin/python scripts/compare_stage2_models.py

The script evaluates the checkpoints on the EDF electricity task and on the
shared synthetic family suite. It writes timestamped CSVs and two summary
figures under ``results/``.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluate_gampfn import FAMILIES, make_task, score
from tabicl._model.additive import AdditivePerceiver


# Edit only these paths if your cluster layout differs.
ELECTRICITY_DATA = REPO_ROOT / "data/data_preprocessed_final_enhanced_2.csv"
CHECKPOINTS = {
    "stage1_65k": REPO_ROOT / "results/gampfn/additive_stage1/step-65000.ckpt",
    "gam_real_20k": REPO_ROOT / "results/gampfn/additive_stage2_gam_real/step-20000.ckpt",
    "gam_tabicl_20k": REPO_ROOT / "results/gampfn/additive_stage2_gam_tabicl/step-20000.ckpt",
    "gam_tabicl_real_20k": REPO_ROOT
    / "results/gampfn/additive_stage2_gam_tabicl_real/step-20000.ckpt",
}

BEGIN_TRAIN = pd.Timestamp("2021-07-01")
BEGIN_TEST = pd.Timestamp("2023-07-01")
END_TEST = pd.Timestamp("2024-07-01")
EXCLUDED_FEATURES = {"Consommation", "Date_local", "Instant"}

# Screening defaults. Increase SYNTHETIC_SEEDS only for the final comparison.
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


def load_model(path: Path) -> tuple[AdditivePerceiver, int | None]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "additive":
        raise ValueError(f"{path} is not an additive checkpoint")
    model = AdditivePerceiver(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().to(DEVICE), checkpoint.get("curr_step")


def predict(
    model: AdditivePerceiver,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    x_context = torch.as_tensor(x_train, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    y_context = torch.as_tensor(y_train, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    predictions = []
    with torch.inference_mode(), amp_context():
        context = model.encode_context(x_context, y_context)
        for start in range(0, len(x_test), PREDICTION_CHUNK_SIZE):
            query = torch.as_tensor(
                x_test[start : start + PREDICTION_CHUNK_SIZE],
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)
            effects = model.predict_effects(context, query)
            predictions.append((context.y_mean + effects.sum(dim=-1))[0].float().cpu().numpy())
    return np.concatenate(predictions)


def load_electricity_tasks() -> tuple[list[dict[str, object]], list[str]]:
    data = pd.read_csv(ELECTRICITY_DATA, parse_dates=["Date_local"])
    data = data[(data["Date_local"] >= BEGIN_TRAIN) & (data["Date_local"] < END_TEST)].copy()
    if "JourFerie" in data:
        data["JourFerie"] = data["JourFerie"].astype(int)
    features = [column for column in data.columns if column not in EXCLUDED_FEATURES]
    non_numeric = data[features].select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise TypeError(f"Electricity features must be numeric; found {non_numeric}")
    if data[features + ["Consommation"]].isna().any().any():
        raise ValueError("Electricity data contain missing feature or target values")

    tasks = []
    instants = sorted(
        {(timestamp.hour, timestamp.minute) for timestamp in data["Date_local"]}
    )
    for hour, minute in instants:
        mask = (data["Date_local"].dt.hour == hour) & (data["Date_local"].dt.minute == minute)
        train = data[mask & (data["Date_local"] < BEGIN_TEST)]
        test = data[mask & (data["Date_local"] >= BEGIN_TEST)]
        if train.empty or test.empty:
            raise ValueError(f"Empty electricity split at {hour:02d}:{minute:02d}")
        tasks.append(
            {
                "hour": hour,
                "minute": minute,
                "x_train": train[features].to_numpy(dtype=np.float32),
                "y_train": train["Consommation"].to_numpy(dtype=np.float32),
                "x_test": test[features].to_numpy(dtype=np.float32),
                "y_test": test["Consommation"].to_numpy(dtype=np.float32),
                "dates": test["Date_local"].to_numpy(),
            }
        )
    return tasks, features


def evaluate_electricity(
    name: str,
    model: AdditivePerceiver,
    tasks: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows = []
    prediction_rows = []
    for index, task in enumerate(tasks, start=1):
        started = time.perf_counter()
        prediction = predict(model, task["x_train"], task["y_train"], task["x_test"])
        elapsed = time.perf_counter() - started
        actual = task["y_test"]
        error = prediction - actual
        rmse = float(np.sqrt(np.mean(error**2)))
        metric_rows.append(
            {
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
            f"  electricity [{index:02d}/{len(tasks)}] {task['hour']:02d}:"
            f"{task['minute']:02d} RMSE={rmse:.2f}",
            flush=True,
        )
    return metric_rows, prediction_rows


def evaluate_synthetic(name: str, model: AdditivePerceiver) -> list[dict[str, object]]:
    rows = []
    total = len(FAMILIES) * len(SYNTHETIC_TRAIN_SIZES) * len(SYNTHETIC_FEATURE_COUNTS) * SYNTHETIC_SEEDS
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
                    prediction = predict(model, x_train, y_train, x_test)
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
    for name, frame in predictions.groupby("model", sort=False):
        error = frame["prediction"] - frame["actual"]
        rmse = float(np.sqrt(np.mean(error**2)))
        rows.append(
            {
                "model": name,
                "rmse": rmse,
                "mae": float(np.mean(np.abs(error))),
                "nrmse": rmse / max(float(frame["actual"].std(ddof=1)), 1e-8),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


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
                        np.median(100 * (valid[opponent] - valid[model]) / valid[opponent].clip(lower=1e-12))
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].bar(electricity_overall["model"], electricity_overall["rmse"])
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set(title="Electricity: overall RMSE", ylabel="RMSE (MW)")
    for name, frame in electricity_metrics.groupby("model", sort=False):
        instant = frame["hour"] + frame["minute"] / 60
        axes[1].plot(instant, frame["rmse"], label=name, linewidth=1.5)
    axes[1].set(title="Electricity: RMSE by instant", xlabel="Hour", ylabel="RMSE (MW)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.savefig(output_dir / "electricity_summary.png", dpi=160)
    plt.close(fig)

    pivot = synthetic_by_family.pivot(index="family", columns="model", values="median_nrmse_f")
    axis = pivot.plot.bar(figsize=(12, 5), rot=20)
    axis.set(title="Synthetic suite: median clean-signal NRMSE", xlabel="Family", ylabel="NRMSE")
    axis.grid(axis="y", alpha=0.2)
    axis.figure.tight_layout()
    axis.figure.savefig(output_dir / "synthetic_summary.png", dpi=160)
    plt.close(axis.figure)


def self_check() -> None:
    task = make_task("smooth", 0, 32, 16, 10, SYNTHETIC_NOISE)
    assert task[0].shape == (32, 10) and task[2].shape == (16, 10)
    assert set(CHECKPOINTS) == {
        "stage1_65k",
        "gam_real_20k",
        "gam_tabicl_20k",
        "gam_tabicl_real_20k",
    }
    print("Static comparison setup passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return

    missing = [path for path in [ELECTRICITY_DATA, *CHECKPOINTS.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(map(str, missing)))

    output_dir = REPO_ROOT / "results" / datetime.now().strftime("stage2_comparison_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    electricity_tasks, features = load_electricity_tasks()
    print(f"Device: {DEVICE}; electricity features: {len(features)}; output: {output_dir}")

    electricity_metrics_rows = []
    electricity_prediction_rows = []
    synthetic_rows = []
    for name, path in CHECKPOINTS.items():
        model, step = load_model(path)
        print(f"\n{name}: {path} (checkpoint step={step})", flush=True)
        metrics, predictions = evaluate_electricity(name, model, electricity_tasks)
        electricity_metrics_rows.extend(metrics)
        electricity_prediction_rows.extend(predictions)
        synthetic_rows.extend(evaluate_synthetic(name, model))
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    electricity_metrics = pd.DataFrame(electricity_metrics_rows)
    electricity_predictions = pd.DataFrame(electricity_prediction_rows)
    synthetic_results = pd.DataFrame(synthetic_rows)
    electricity_overall = electricity_summary(electricity_predictions)
    synthetic_overall, synthetic_by_family = synthetic_summary(synthetic_results)

    electricity_metrics.to_csv(output_dir / "electricity_by_instant.csv", index=False)
    electricity_predictions.to_csv(output_dir / "electricity_predictions.csv", index=False)
    electricity_overall.to_csv(output_dir / "electricity_summary.csv", index=False)
    synthetic_results.to_csv(output_dir / "synthetic_tasks.csv", index=False)
    synthetic_overall.to_csv(output_dir / "synthetic_summary.csv", index=False)
    synthetic_by_family.to_csv(output_dir / "synthetic_by_family.csv", index=False)
    pairwise_wins(
        electricity_metrics,
        ["hour", "minute"],
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
