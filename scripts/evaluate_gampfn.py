#!/usr/bin/env python3
"""Paired synthetic evaluation of a GAMPFN checkpoint against TabICLv2.

Example screening run:

    uv run python scripts/evaluate_gampfn.py \
        --gampfn /path/to/step-80000.ckpt \
        --tabiclv2 /path/to/tabicl-regressor-v2-20260212.ckpt \
        --output gampfn_eval.csv

Use ``--seeds 50 --n-estimators 8`` for the slower confirmation run.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from tabicl import TabICLRegressor


FAMILIES = ("smooth", "monotone", "discontinuous", "ga2m", "nonadditive")
FIELDS = (
    "family",
    "seed",
    "n_train",
    "n_test",
    "n_features",
    "correlated",
    "model",
    "nrmse_f",
    "nrmse_y",
    "nmae_f",
    "seconds",
)


class ReusableTabICLRegressor(TabICLRegressor):
    """Avoid re-reading the same checkpoint for every synthetic task."""

    def _load_model(self) -> None:
        if not hasattr(self, "model_"):
            super()._load_model()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gampfn", type=Path, help="GAMPFN checkpoint")
    parser.add_argument("--tabiclv2", type=Path, help="Released TabICLv2 regressor checkpoint")
    parser.add_argument("--output", type=Path, default=Path("gampfn_eval.csv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[128, 512, 800])
    parser.add_argument("--n-features", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.2, help="Noise standard deviation after signal scaling")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--self-check", action="store_true", help="Check task generation without loading models")
    return parser.parse_args()


def sample_features(rng: np.random.Generator, n_samples: int, n_features: int, correlated: bool) -> np.ndarray:
    noise = rng.standard_normal((n_samples, n_features))
    if not correlated:
        return noise
    signs = rng.choice((-1.0, 1.0), size=n_features)
    latent = rng.standard_normal((n_samples, 1))
    return np.sqrt(0.6) * latent * signs + np.sqrt(0.4) * noise


def active_features(rng: np.random.Generator, n_features: int) -> np.ndarray:
    return rng.choice(n_features, size=min(n_features, int(rng.integers(2, 6))), replace=False)


def smooth_signal(rng: np.random.Generator, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = active_features(rng, X.shape[1])
    signal = np.zeros(len(X))
    for feature in active:
        amplitude = rng.uniform(0.5, 2.0) * rng.choice((-1.0, 1.0))
        scale = rng.uniform(0.5, 2.5)
        shift = rng.uniform(-1.0, 1.0)
        if rng.random() < 0.5:
            signal += amplitude * np.arctan(scale * X[:, feature] + shift)
        else:
            signal += amplitude * np.logaddexp(0.0, scale * (X[:, feature] - shift)) / scale
    return signal, active


def monotone_signal(rng: np.random.Generator, X: np.ndarray) -> np.ndarray:
    signal = np.zeros(len(X))
    for feature in active_features(rng, X.shape[1]):
        knots = np.sort(rng.uniform(-1.5, 1.5, size=4))
        weights = rng.uniform(0.2, 1.5, size=4)
        direction = rng.choice((-1.0, 1.0))
        signal += direction * (np.maximum(X[:, feature, None] - knots, 0.0) * weights).sum(axis=1)
    return signal


def discontinuous_signal(rng: np.random.Generator, X: np.ndarray) -> np.ndarray:
    signal = np.zeros(len(X))
    for feature in active_features(rng, X.shape[1]):
        cuts = np.sort(rng.uniform(-1.5, 1.5, size=int(rng.integers(2, 6))))
        levels = rng.normal(size=len(cuts) + 1)
        signal += levels[np.digitize(X[:, feature], cuts)]
    return signal


def ga2m_signal(rng: np.random.Generator, X: np.ndarray) -> np.ndarray:
    signal, active = smooth_signal(rng, X)
    if len(active) < 2:
        return signal
    all_pairs = np.array([(left, right) for i, left in enumerate(active) for right in active[i + 1 :]])
    pairs = all_pairs[rng.choice(len(all_pairs), size=min(len(all_pairs), int(rng.integers(1, 4))), replace=False)]
    for left, right in pairs:
        signal += rng.uniform(0.5, 1.5) * np.tanh(X[:, left]) * np.tanh(1.5 * X[:, right])
    return signal


def nonadditive_signal(rng: np.random.Generator, X: np.ndarray) -> np.ndarray:
    active = rng.choice(X.shape[1], size=min(X.shape[1], 6), replace=False)
    weights = rng.normal(size=len(active))
    ridge = np.sin(X[:, active] @ weights)
    product = np.prod(np.tanh(X[:, active[: min(4, len(active))]]), axis=1)
    return ridge + product


def make_task(
    family: str,
    seed: int,
    n_train: int,
    n_test: int,
    n_features: int,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    family_id = FAMILIES.index(family)
    rng = np.random.default_rng(np.random.SeedSequence((seed, family_id, n_train, n_features)))
    correlated = bool(seed % 2)
    X = sample_features(rng, n_train + n_test, n_features, correlated)

    if family == "smooth":
        raw, _ = smooth_signal(rng, X)
    elif family == "monotone":
        raw = monotone_signal(rng, X)
    elif family == "discontinuous":
        raw = discontinuous_signal(rng, X)
    elif family == "ga2m":
        raw = ga2m_signal(rng, X)
    else:
        raw = nonadditive_signal(rng, X)

    center = raw[:n_train].mean()
    scale = max(raw[:n_train].std(ddof=1), 1e-8)
    clean = (raw - center) / scale
    observed = clean + noise_scale * rng.standard_normal(len(clean))
    return (
        X[:n_train].astype(np.float32),
        observed[:n_train].astype(np.float32),
        X[n_train:].astype(np.float32),
        observed[n_train:].astype(np.float32),
        clean[n_train:].astype(np.float32),
        correlated,
    )


def classical_models(seed: int) -> dict[str, object]:
    return {
        "spline_ridge": make_pipeline(
            SplineTransformer(n_knots=5, degree=3, include_bias=False),
            StandardScaler(),
            RidgeCV(alphas=(0.1, 1.0, 10.0)),
        ),
        "hist_gb": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def score(prediction: np.ndarray, clean: np.ndarray, observed: np.ndarray) -> tuple[float, float, float]:
    if not np.isfinite(prediction).all():
        raise ValueError("Model produced non-finite predictions")
    denominator = max(float(np.std(clean, ddof=1)), 1e-8)
    nrmse_f = np.sqrt(np.mean((prediction - clean) ** 2)) / denominator
    nrmse_y = np.sqrt(np.mean((prediction - observed) ** 2)) / denominator
    nmae_f = np.mean(np.abs(prediction - clean)) / denominator
    return float(nrmse_f), float(nrmse_y), float(nmae_f)


def self_check(args: argparse.Namespace) -> None:
    for family in FAMILIES:
        task = make_task(family, 0, 32, 16, 10, args.noise)
        X_train, y_train, X_test, y_test, clean, _ = task
        assert X_train.shape == (32, 10) and X_test.shape == (16, 10)
        assert y_train.shape == (32,) and y_test.shape == clean.shape == (16,)
        assert all(np.isfinite(value).all() for value in task[:5])
    print("Synthetic task generation passed.")


def summarize(rows: list[dict[str, object]]) -> None:
    paired: dict[tuple[object, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row["family"], row["seed"], row["n_train"], row["n_features"])
        paired[key][str(row["model"])] = float(row["nrmse_f"])

    groups: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for key, scores in paired.items():
        if "gampfn" not in scores or "tabiclv2" not in scores:
            continue
        family, _, n_train, n_features = key
        baseline = max(scores["tabiclv2"], 1e-12)
        groups[(str(family), int(n_train), int(n_features))].append(100 * (baseline - scores["gampfn"]) / baseline)

    print("\nPaired GAMPFN improvement over TabICLv2 in clean-signal NRMSE (higher is better):")
    print(f"{'family':<14} {'train':>6} {'d':>4} {'median':>9} {'95% CI':>22} {'wins':>8}")
    bootstrap_rng = np.random.default_rng(0)
    for (family, n_train, n_features), values in sorted(groups.items()):
        improvements = np.asarray(values)
        draws = bootstrap_rng.choice(improvements, size=(2000, len(improvements)), replace=True)
        low, high = np.percentile(np.median(draws, axis=1), (2.5, 97.5))
        wins = np.mean(improvements > 0)
        print(f"{family:<14} {n_train:>6} {n_features:>4} {np.median(improvements):>8.2f}% "
              f"[{low:>7.2f}%, {high:>7.2f}%] {wins:>7.0%}")


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check(args)
        return
    if args.gampfn is None or args.tabiclv2 is None:
        raise ValueError("--gampfn and --tabiclv2 are required unless --self-check is used")
    for checkpoint in (args.gampfn, args.tabiclv2):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if args.seeds < 1 or args.n_estimators < 1 or args.n_test < 2 or args.noise < 0:
        raise ValueError("seeds, n-estimators, and n-test must be positive; noise must be non-negative")
    if any(size < 2 for size in args.train_sizes) or any(count < 1 for count in args.n_features):
        raise ValueError("train sizes must be at least 2 and feature counts must be positive")

    pfn_models = {
        "gampfn": ReusableTabICLRegressor(
            model_path=args.gampfn,
            allow_auto_download=False,
            n_estimators=args.n_estimators,
            random_state=0,
            device=args.device,
        ),
        "tabiclv2": ReusableTabICLRegressor(
            model_path=args.tabiclv2,
            allow_auto_download=False,
            n_estimators=args.n_estimators,
            random_state=0,
            device=args.device,
        ),
    }

    rows: list[dict[str, object]] = []
    total = len(args.families) * len(args.train_sizes) * len(args.n_features) * args.seeds
    task_number = 0
    with args.output.open("x", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS)
        writer.writeheader()
        for family in args.families:
            for n_train in args.train_sizes:
                for n_features in args.n_features:
                    for seed in range(args.seeds):
                        task_number += 1
                        print(
                            f"[{task_number}/{total}] {family} seed={seed} n={n_train} d={n_features}",
                            flush=True,
                        )
                        X_train, y_train, X_test, y_test, clean, correlated = make_task(
                            family, seed, n_train, args.n_test, n_features, args.noise
                        )
                        models = {**pfn_models, **classical_models(seed)}
                        for name, model in models.items():
                            started = time.perf_counter()
                            prediction = model.fit(X_train, y_train).predict(X_test)
                            elapsed = time.perf_counter() - started
                            nrmse_f, nrmse_y, nmae_f = score(prediction, clean, y_test)
                            row = {
                                "family": family,
                                "seed": seed,
                                "n_train": n_train,
                                "n_test": args.n_test,
                                "n_features": n_features,
                                "correlated": correlated,
                                "model": name,
                                "nrmse_f": nrmse_f,
                                "nrmse_y": nrmse_y,
                                "nmae_f": nmae_f,
                                "seconds": elapsed,
                            }
                            rows.append(row)
                            writer.writerow(row)
                        output_file.flush()

    summarize(rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
