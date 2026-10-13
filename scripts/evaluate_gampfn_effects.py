#!/usr/bin/env python3
"""Evaluate an additive GAMPFN checkpoint against known feature effects.

Example:

    python scripts/evaluate_gampfn_effects.py \
        --checkpoint /path/to/step-40000.ckpt \
        --output-dir effect-eval
"""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tabicl._model.additive import AdditiveContext, AdditivePerceiver


BASE_FAMILIES = (
    "linear",
    "saturating",
    "spline",
    "fourier",
    "rbf",
    "piecewise_linear",
    "step",
    "categorical",
)
PROXY_SOURCE_FAMILIES = ("linear", "spline", "fourier")
FAMILIES = (*BASE_FAMILIES, "mixed", "proxies")
NUM_CATEGORIES = 5
MAX_PROXIES_PER_FEATURE = 3
MIN_PROXY_CORRELATION = 0.95
MAX_PROXY_CORRELATION = 0.999
FIELDS = (
    "family",
    "seed",
    "context_size",
    "n_features",
    "feature",
    "effect_family",
    "active",
    "effect_rmse",
    "effect_nrmse",
    "effect_correlation",
    "effect_rms",
    "effect_rank",
    "support_at_k",
    "proxy",
    "proxy_source",
    "input_correlation",
    "proxy_leakage_ratio",
    "group_effect_nrmse",
    "group_duplication_gain",
    "prediction_nrmse",
)


@dataclass(frozen=True)
class Task:
    x_context: np.ndarray
    y_context: np.ndarray
    x_query: np.ndarray
    clean_query: np.ndarray
    true_effects_query: np.ndarray
    effect_families: tuple[str | None, ...]
    effect_centers: np.ndarray
    signal_scale: float
    proxy_sources: tuple[int | None, ...]
    proxy_correlations: tuple[float | None, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("effect-eval"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--context-sizes", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--n-features", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--n-query", type=int, default=512)
    parser.add_argument("--n-grid", type=int, default=256)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument(
        "--plot-seeds",
        type=int,
        default=1,
        help="Number of seeds plotted per family/context/dimension; all seeds are scored",
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def effect_value(family: str, x: np.ndarray, variant: int) -> np.ndarray:
    """Evaluate one deterministic member of an effect family."""
    sign = -1.0 if variant % 2 else 1.0
    phase = 0.35 * variant
    if family == "linear":
        return sign * (1.1 + 0.15 * variant) * x
    if family == "saturating":
        return sign * 1.8 * np.tanh((0.8 + 0.15 * variant) * x + phase)
    if family == "spline":
        # A cubic truncated-power spline with fixed internal knots.
        return sign * (
            0.35 * x
            - 0.12 * x**2
            + 0.20 * np.maximum(x + 1.0, 0.0) ** 3
            - 0.32 * np.maximum(x, 0.0) ** 3
            + 0.14 * np.maximum(x - 1.0, 0.0) ** 3
        )
    if family == "fourier":
        return sign * (np.sin((1.2 + 0.15 * variant) * x + phase) + 0.45 * np.cos(2.7 * x))
    if family == "rbf":
        return sign * (
            1.5 * np.exp(-0.5 * ((x - 0.7) / 0.55) ** 2)
            - 1.1 * np.exp(-0.5 * ((x + 0.9) / 0.8) ** 2)
        )
    if family == "piecewise_linear":
        knots = np.array([-3.0, -1.2, -0.2, 0.9, 3.0])
        values = sign * np.array([-0.8, 1.2, -0.5, 1.5, 0.2])
        return np.interp(x, knots, values)
    if family == "step":
        return sign * np.select(
            [x < -0.8, x < 0.4, x < 1.4],
            [-1.2, 0.5, 1.4],
            default=-0.3,
        )
    if family == "categorical":
        levels = np.roll(np.array([-1.3, 0.8, -0.2, 1.5, -0.7]), variant % NUM_CATEGORIES)
        return sign * levels[np.asarray(x, dtype=int)]
    raise ValueError(f"Unknown effect family: {family}")


def sample_features(
    rng: np.random.Generator,
    n_rows: int,
    effect_families: tuple[str | None, ...],
) -> np.ndarray:
    x = np.clip(rng.standard_normal((n_rows, len(effect_families))), -3.0, 3.0)
    for feature, family in enumerate(effect_families):
        if family == "categorical":
            x[:, feature] = rng.integers(0, NUM_CATEGORIES, size=n_rows)
    return x


def assign_effects(family: str, n_features: int, rng: np.random.Generator) -> tuple[str | None, ...]:
    effect_families: list[str | None] = [None] * n_features
    if family == "proxies":
        active_families = PROXY_SOURCE_FAMILIES[: min(3, max(1, n_features // 2))]
    elif family == "mixed":
        active_families = BASE_FAMILIES[: min(n_features, len(BASE_FAMILIES))]
    else:
        active_families = (family,) * min(3, n_features)
    active_features = rng.choice(n_features, size=len(active_families), replace=False)
    for feature, active_family in zip(active_features, active_families):
        effect_families[int(feature)] = active_family
    return tuple(effect_families)


def inject_proxies(
    rng: np.random.Generator,
    x_context: np.ndarray,
    x_query: np.ndarray,
    effect_families: tuple[str | None, ...],
) -> tuple[tuple[int | None, ...], tuple[float | None, ...]]:
    """Turn inactive columns into near-duplicates of active numeric features."""
    proxy_sources: list[int | None] = [None] * len(effect_families)
    proxy_correlations: list[float | None] = [None] * len(effect_families)
    sources = np.array(
        [feature for feature, name in enumerate(effect_families) if name is not None],
        dtype=int,
    )
    unused = np.array(
        [feature for feature, name in enumerate(effect_families) if name is None],
        dtype=int,
    )
    rng.shuffle(sources)
    rng.shuffle(unused)
    combined = np.concatenate((x_context, x_query), axis=0)
    next_unused = 0
    for source in sources:
        remaining = len(unused) - next_unused
        if remaining == 0:
            break
        count = int(rng.integers(1, min(MAX_PROXIES_PER_FEATURE, remaining) + 1))
        source_values = combined[:, source]
        source_values = (source_values - source_values.mean()) / max(source_values.std(), 1e-8)
        for proxy in unused[next_unused : next_unused + count]:
            correlation_value = float(rng.uniform(MIN_PROXY_CORRELATION, MAX_PROXY_CORRELATION))
            noise = rng.standard_normal(len(combined))
            noise = (noise - noise.mean()) / max(noise.std(), 1e-8)
            noise -= source_values * np.dot(noise, source_values) / max(
                np.dot(source_values, source_values), 1e-8
            )
            noise /= max(noise.std(), 1e-8)
            combined[:, proxy] = (
                correlation_value * source_values
                + np.sqrt(1 - correlation_value**2) * noise
            )
            proxy_sources[int(proxy)] = int(source)
            proxy_correlations[int(proxy)] = correlation_value
        next_unused += count
    x_context[:] = combined[: len(x_context)]
    x_query[:] = combined[len(x_context) :]
    return tuple(proxy_sources), tuple(proxy_correlations)


def raw_effects(x: np.ndarray, effect_families: tuple[str | None, ...]) -> np.ndarray:
    effects = np.zeros_like(x, dtype=np.float64)
    for feature, family in enumerate(effect_families):
        if family is not None:
            effects[:, feature] = effect_value(family, x[:, feature], feature)
    return effects


def make_task(
    family: str,
    seed: int,
    context_size: int,
    n_query: int,
    n_features: int,
    noise: float,
) -> Task:
    family_id = FAMILIES.index(family)
    rng = np.random.default_rng(np.random.SeedSequence((seed, family_id, context_size, n_features)))
    effect_families = assign_effects(family, n_features, rng)
    x_context = sample_features(rng, context_size, effect_families)
    x_query = sample_features(rng, n_query, effect_families)
    if family == "proxies":
        proxy_sources, proxy_correlations = inject_proxies(
            rng, x_context, x_query, effect_families
        )
    else:
        proxy_sources = (None,) * n_features
        proxy_correlations = (None,) * n_features
    context_effects = raw_effects(x_context, effect_families)
    query_effects = raw_effects(x_query, effect_families)

    context_signal = context_effects.sum(axis=1)
    signal_center = float(context_signal.mean())
    signal_scale = max(float(context_signal.std()), 1e-8)
    effect_centers = context_effects.mean(axis=0)
    true_effects_query = (query_effects - effect_centers) / signal_scale
    clean_context = (context_signal - signal_center) / signal_scale
    clean_query = (query_effects.sum(axis=1) - signal_center) / signal_scale
    y_context = clean_context + noise * rng.standard_normal(context_size)

    return Task(
        x_context=x_context.astype(np.float32),
        y_context=y_context.astype(np.float32),
        x_query=x_query.astype(np.float32),
        clean_query=clean_query.astype(np.float32),
        true_effects_query=true_effects_query.astype(np.float32),
        effect_families=effect_families,
        effect_centers=effect_centers,
        signal_scale=signal_scale,
        proxy_sources=proxy_sources,
        proxy_correlations=proxy_correlations,
    )


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[AdditivePerceiver, int | None]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "additive":
        raise ValueError(f"Expected an additive checkpoint, found {checkpoint.get('model_type')!r}")
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain 'config' and 'state_dict'")
    model = AdditivePerceiver(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(device)
    return model, checkpoint.get("curr_step")


def amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def encode_task(model: AdditivePerceiver, task: Task, device: torch.device) -> AdditiveContext:
    x_context = torch.from_numpy(task.x_context).unsqueeze(0).to(device)
    y_context = torch.from_numpy(task.y_context).unsqueeze(0).to(device)
    with torch.inference_mode(), amp_context(device):
        return model.encode_context(x_context, y_context)


def predict_effects(
    model: AdditivePerceiver,
    context: AdditiveContext,
    x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x_tensor = torch.from_numpy(x.astype(np.float32, copy=False)).unsqueeze(0).to(device)
    with torch.inference_mode(), amp_context(device):
        effects = model.predict_effects(context, x_tensor)
        prediction = context.y_mean + effects.sum(dim=-1)
    return prediction[0].float().cpu().numpy(), effects[0].float().cpu().numpy()


def effect_grid(task: Task, n_grid: int) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    query = np.empty((n_grid, len(task.effect_families)), dtype=np.float32)
    grids: list[np.ndarray] = []
    for feature, family in enumerate(task.effect_families):
        if family == "categorical":
            grid = np.resize(np.arange(NUM_CATEGORIES, dtype=np.float32), n_grid)
        else:
            grid = np.linspace(-3.0, 3.0, n_grid, dtype=np.float32)
        query[:, feature] = grid
        grids.append(grid)
    return query, tuple(grids)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.std() < 1e-8 or right.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def score_task(
    family: str,
    seed: int,
    task: Task,
    prediction: np.ndarray,
    predicted_effects: np.ndarray,
) -> list[dict[str, object]]:
    prediction_scale = max(float(task.clean_query.std()), 1e-8)
    prediction_nrmse = float(np.sqrt(np.mean((prediction - task.clean_query) ** 2)) / prediction_scale)
    effect_rms = np.sqrt(np.mean(predicted_effects**2, axis=0))
    order = np.argsort(-effect_rms)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    active_mask = np.array([name is not None for name in task.effect_families])
    num_active = int(active_mask.sum())
    support_at_k = float(active_mask[order[:num_active]].mean())
    proxy_groups: dict[int, list[int]] = {}
    for proxy, source in enumerate(task.proxy_sources):
        if source is not None:
            proxy_groups.setdefault(source, []).append(proxy)
    group_metrics: dict[int, tuple[float, float]] = {}
    for source, proxies in proxy_groups.items():
        truth = task.true_effects_query[:, source]
        source_estimate = predicted_effects[:, source]
        group_estimate = source_estimate + predicted_effects[:, proxies].sum(axis=1)
        group_nrmse = float(
            np.sqrt(np.mean((group_estimate - truth) ** 2)) / max(float(truth.std()), 1e-8)
        )
        group_gain = float(
            np.sqrt(np.mean(group_estimate**2))
            / max(float(np.sqrt(np.mean(source_estimate**2))), 1e-8)
        )
        group_metrics[source] = (group_nrmse, group_gain)
    rows = []
    for feature, effect_family in enumerate(task.effect_families):
        truth = task.true_effects_query[:, feature]
        estimate = predicted_effects[:, feature]
        rmse = float(np.sqrt(np.mean((estimate - truth) ** 2)))
        effect_scale = float(truth.std())
        proxy_source = task.proxy_sources[feature]
        if proxy_source is None:
            proxy_leakage_ratio: float | str = ""
        else:
            source_scale = float(np.sqrt(np.mean(task.true_effects_query[:, proxy_source] ** 2)))
            proxy_leakage_ratio = float(effect_rms[feature]) / max(source_scale, 1e-8)
        group_nrmse, group_gain = group_metrics.get(feature, ("", ""))
        rows.append(
            {
                "family": family,
                "seed": seed,
                "context_size": len(task.x_context),
                "n_features": task.x_context.shape[1],
                "feature": feature,
                "effect_family": effect_family or "inactive",
                "active": effect_family is not None,
                "effect_rmse": rmse,
                "effect_nrmse": rmse / max(effect_scale, 1e-8) if effect_family else "",
                "effect_correlation": correlation(truth, estimate) if effect_family else "",
                "effect_rms": float(effect_rms[feature]),
                "effect_rank": int(ranks[feature]),
                "support_at_k": support_at_k,
                "proxy": proxy_source is not None,
                "proxy_source": proxy_source if proxy_source is not None else "",
                "input_correlation": (
                    task.proxy_correlations[feature] if proxy_source is not None else ""
                ),
                "proxy_leakage_ratio": proxy_leakage_ratio,
                "group_effect_nrmse": group_nrmse,
                "group_duplication_gain": group_gain,
                "prediction_nrmse": prediction_nrmse,
            }
        )
    return rows


def plot_task(
    output_path: Path,
    family: str,
    seed: int,
    task: Task,
    model: AdditivePerceiver,
    context: AdditiveContext,
    device: torch.device,
    n_grid: int,
    query_effects: np.ndarray,
) -> None:
    active = [feature for feature, name in enumerate(task.effect_families) if name is not None]
    grid_query, grids = effect_grid(task, n_grid)
    _, predicted = predict_effects(model, context, grid_query, device)
    ncols = min(3, len(active))
    nrows = (len(active) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, 3.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, feature in zip(axes.flat, active):
        effect_family = task.effect_families[feature]
        assert effect_family is not None
        x = grids[feature]
        truth = (
            effect_value(effect_family, x, feature) - task.effect_centers[feature]
        ) / task.signal_scale
        if effect_family == "categorical":
            categories = np.arange(NUM_CATEGORIES)
            estimate = np.array([predicted[x == category, feature].mean() for category in categories])
            truth = effect_value(effect_family, categories, feature)
            truth = (truth - task.effect_centers[feature]) / task.signal_scale
            axis.plot(categories, truth, "o-", label="true effect", color="tab:blue")
            axis.plot(categories, estimate, "s--", label="predicted effect", color="tab:orange")
            axis.set_xticks(categories)
        else:
            order = np.argsort(x)
            axis.plot(x[order], truth[order], label="true effect", color="tab:blue", linewidth=2)
            axis.plot(
                x[order],
                predicted[order, feature],
                label="predicted effect",
                color="tab:orange",
                linewidth=2,
            )
            axis.scatter(
                task.x_context[:, feature],
                np.full(len(task.x_context), axis.get_ylim()[0]),
                marker="|",
                s=18,
                alpha=0.25,
                color="black",
                label="context x",
            )
        axis.axhline(0.0, color="0.6", linewidth=0.8)
        axis.set(title=f"feature {feature}: {effect_family}", xlabel="feature value", ylabel="centered effect")
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    for axis in axes.flat[len(active) :]:
        axis.remove()
    fig.suptitle(
        f"{family}; seed={seed}; context={len(task.x_context)}; D={task.x_context.shape[1]}"
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    true_rms = np.sqrt(np.mean(task.true_effects_query**2, axis=0))
    predicted_rms = np.sqrt(np.mean(query_effects**2, axis=0))
    active_mask = np.array([name is not None for name in task.effect_families])
    order = np.argsort(-predicted_rms)
    support_at_k = active_mask[order[: int(active_mask.sum())]].mean()
    features = np.arange(len(active_mask))
    fig, axis = plt.subplots(figsize=(max(8.0, 0.22 * len(features)), 4.2), constrained_layout=True)
    axis.bar(features, predicted_rms, color="tab:orange", alpha=0.75, label="predicted effect RMS")
    axis.scatter(features, true_rms, color="black", marker="x", zorder=3, label="true effect RMS")
    axis.scatter(
        features[active_mask],
        predicted_rms[active_mask],
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=1.5,
        s=65,
        zorder=3,
        label="truly active feature",
    )
    tick_step = 1 if len(features) <= 20 else 5
    axis.set_xticks(features[::tick_step])
    axis.set(
        xlabel="feature index",
        ylabel="effect RMS on query distribution",
        title=f"Feature relevance; support@k={support_at_k:.0%}",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="best")
    fig.savefig(output_path.with_name(f"relevance_{output_path.name}"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    proxy_groups: dict[int, list[int]] = {}
    for proxy, source in enumerate(task.proxy_sources):
        if source is not None:
            proxy_groups.setdefault(source, []).append(proxy)
    if not proxy_groups:
        return
    ncols = min(2, len(proxy_groups))
    nrows = (len(proxy_groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.0 * ncols, 4.0 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (source, proxies) in zip(axes.flat, proxy_groups.items()):
        order = np.argsort(task.x_query[:, source])
        x = task.x_query[order, source]
        truth = task.true_effects_query[order, source]
        source_estimate = query_effects[order, source]
        proxy_sum = query_effects[order][:, proxies].sum(axis=1)
        axis.plot(x, truth, color="black", linewidth=2, label="true canonical effect")
        axis.plot(x, source_estimate, color="tab:blue", label="predicted canonical effect")
        axis.plot(x, proxy_sum, color="tab:red", linestyle="--", label="summed proxy effects")
        axis.plot(
            x,
            source_estimate + proxy_sum,
            color="tab:green",
            linewidth=2,
            label="predicted group total",
        )
        correlations = [task.proxy_correlations[proxy] for proxy in proxies]
        axis.set(
            title=(
                f"source {source} -> proxies {proxies}; "
                f"corr {min(correlations):.3f}-{max(correlations):.3f}"
            ),
            xlabel=f"source feature {source}",
            ylabel="centered effect",
        )
        axis.axhline(0.0, color="0.6", linewidth=0.8)
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    for axis in axes.flat[len(proxy_groups) :]:
        axis.remove()
    fig.suptitle(
        f"Proxy attribution; seed={seed}; context={len(task.x_context)}; "
        f"D={task.x_context.shape[1]}"
    )
    fig.savefig(output_path.with_name(f"proxies_{output_path.name}"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["family"]), int(row["context_size"]), int(row["n_features"]))
        groups.setdefault(key, []).append(row)
    print(
        f"{'family':<18} {'context':>7} {'D':>4} {'effect NRMSE':>14} "
        f"{'effect corr':>12} {'support@k':>10} {'leak med':>10} {'leak p95':>10} "
        f"{'leak max':>10} {'pred NRMSE':>12}"
    )
    for (family, context_size, n_features), group in sorted(groups.items()):
        active = [row for row in group if row["active"]]
        inactive = [row for row in group if not row["active"]]
        pred_by_task = {(row["seed"], row["prediction_nrmse"]) for row in group}
        support_by_task = {(row["seed"], row["support_at_k"]) for row in group}
        leakage = np.array([float(row["effect_rms"]) for row in inactive])
        leakage_median = float(np.median(leakage)) if len(leakage) else float("nan")
        leakage_p95 = float(np.percentile(leakage, 95)) if len(leakage) else float("nan")
        leakage_max = float(np.max(leakage)) if len(leakage) else float("nan")
        print(
            f"{family:<18} {context_size:>7} {n_features:>4} "
            f"{np.median([float(row['effect_nrmse']) for row in active]):>14.3f} "
            f"{np.nanmedian([float(row['effect_correlation']) for row in active]):>12.3f} "
            f"{np.median([float(value) for _, value in support_by_task]):>10.1%} "
            f"{leakage_median:>10.3f} "
            f"{leakage_p95:>10.3f} "
            f"{leakage_max:>10.3f} "
            f"{np.median([float(value) for _, value in pred_by_task]):>12.3f}"
        )

    proxy_rows = [row for row in rows if row["family"] == "proxies"]
    if not proxy_rows:
        return
    proxy_groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in proxy_rows:
        key = (int(row["context_size"]), int(row["n_features"]))
        proxy_groups.setdefault(key, []).append(row)
    print(
        f"\n{'proxy attribution':<18} {'context':>7} {'D':>4} {'source NRMSE':>14} "
        f"{'proxy/source':>13} {'group NRMSE':>12} {'group gain':>11} {'pred NRMSE':>12}"
    )
    for (context_size, n_features), group in sorted(proxy_groups.items()):
        sources = [row for row in group if row["group_effect_nrmse"] != ""]
        proxies = [row for row in group if row["proxy"]]
        pred_by_task = {(row["seed"], row["prediction_nrmse"]) for row in group}
        print(
            f"{'proxies':<18} {context_size:>7} {n_features:>4} "
            f"{np.median([float(row['effect_nrmse']) for row in sources]):>14.3f} "
            f"{np.median([float(row['proxy_leakage_ratio']) for row in proxies]):>13.3f} "
            f"{np.median([float(row['group_effect_nrmse']) for row in sources]):>12.3f} "
            f"{np.median([float(row['group_duplication_gain']) for row in sources]):>11.3f} "
            f"{np.median([float(value) for _, value in pred_by_task]):>12.3f}"
        )


def self_check() -> None:
    for family in FAMILIES:
        task = make_task(family, seed=0, context_size=32, n_query=16, n_features=10, noise=0.2)
        assert task.x_context.shape == (32, 10)
        assert task.true_effects_query.shape == (16, 10)
        assert np.allclose(task.true_effects_query.sum(axis=1), task.clean_query, atol=1e-5)
        assert np.isfinite(task.y_context).all()
        rows = score_task(family, 0, task, task.clean_query, task.true_effects_query)
        assert all(float(row["support_at_k"]) == 1.0 for row in rows)
        if family == "proxies":
            proxies = [feature for feature, source in enumerate(task.proxy_sources) if source is not None]
            assert proxies
            combined = np.concatenate((task.x_context, task.x_query), axis=0)
            for proxy in proxies:
                source = task.proxy_sources[proxy]
                assert source is not None
                assert correlation(combined[:, source], combined[:, proxy]) >= MIN_PROXY_CORRELATION - 1e-6
                assert np.count_nonzero(task.true_effects_query[:, proxy]) == 0
            source_rows = [row for row in rows if row["group_effect_nrmse"] != ""]
            assert all(float(row["group_effect_nrmse"]) < 1e-6 for row in source_rows)
            assert all(float(row["proxy_leakage_ratio"]) == 0.0 for row in rows if row["proxy"])
    print("Effect-task generation passed.")


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --self-check is used")
    if args.seeds < 1 or min(args.context_sizes) < 2 or min(args.n_features) < 1:
        raise SystemExit("seeds, context sizes, and feature counts must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model, step = load_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    print(f"Loaded additive checkpoint at step {step if step is not None else 'unknown'}")

    rows: list[dict[str, object]] = []
    for family in args.families:
        for context_size in args.context_sizes:
            for n_features in args.n_features:
                for seed_index in range(args.seeds):
                    seed = args.seed_offset + seed_index
                    task = make_task(
                        family,
                        seed,
                        context_size,
                        args.n_query,
                        n_features,
                        args.noise,
                    )
                    context = encode_task(model, task, device)
                    prediction, effects = predict_effects(model, context, task.x_query, device)
                    rows.extend(score_task(family, seed, task, prediction, effects))
                    if seed_index < args.plot_seeds:
                        plot_task(
                            plot_dir / f"{family}_c{context_size}_d{n_features}_seed{seed}.png",
                            family,
                            seed,
                            task,
                            model,
                            context,
                            device,
                            args.n_grid,
                            effects,
                        )
                    print(f"finished family={family} context={context_size} D={n_features} seed={seed}")

    metrics_path = args.output_dir / "effect_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved plots to {plot_dir}\n")
    summarize(rows)


if __name__ == "__main__":
    main()
