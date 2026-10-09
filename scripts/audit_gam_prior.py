"""Audit the rich GAM prior without training a model."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

import torch

from tabicl.prior import GAMConfig
from tabicl.prior._gam import GAM, _evaluate_effect, _evaluate_rich_signal


def _frequencies(counter: Counter[str], total: int) -> dict[str, float]:
    return {key: value / total for key, value in sorted(counter.items())}


def _benchmark(config: GAMConfig, count: int, seq_len: int, features: int) -> float:
    torch.manual_seed(123)
    start = time.perf_counter()
    for _ in range(count):
        GAM(seq_len=seq_len, num_features=features, max_features=features, config=config)()
    return (time.perf_counter() - start) / count


def _plot_examples(examples: dict[str, tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]], output: Path) -> None:
    import matplotlib.pyplot as plt

    families = (
        "linear",
        "saturating",
        "spline",
        "fourier",
        "rbf",
        "piecewise_linear",
        "step",
        "categorical",
    )
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for family, axis in zip(families, axes.flat):
        if family not in examples:
            axis.set_visible(False)
            continue
        values, categories, spec = examples[family]
        if categories is None:
            grid = torch.linspace(float(values.min()), float(values.max()), 500)
            effect = _evaluate_effect(spec, grid)
            axis.plot(grid.numpy(), effect.numpy())
        else:
            codes = torch.arange(len(spec["lookup"]))
            effect = _evaluate_effect(spec, codes.float(), codes)
            axis.plot(codes.numpy(), effect.numpy(), marker=".", linewidth=1)
        axis.set_title(family)
        axis.grid(alpha=0.2)
    figure.savefig(output / "effects.png", dpi=160)
    plt.close(figure)


def _plot_interaction(example: tuple[dict[str, Any], torch.Tensor, torch.Tensor], output: Path) -> None:
    import matplotlib.pyplot as plt

    interaction, left_values, right_values = example
    left_grid = torch.linspace(float(left_values.min()), float(left_values.max()), 80)
    right_grid = torch.linspace(float(right_values.min()), float(right_values.max()), 80)
    surface = torch.zeros((len(left_grid), len(right_grid)))
    for weight, left_spec, right_spec in interaction["rank_components"]:
        left_effect = _evaluate_effect(left_spec, left_grid)
        right_effect = _evaluate_effect(right_spec, right_grid)
        surface += weight * left_effect[:, None] * right_effect[None, :]
    surface = (surface - interaction["normalization_center"]) / interaction["normalization_scale"]

    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(
        surface.numpy(),
        origin="lower",
        aspect="auto",
        extent=(float(right_grid.min()), float(right_grid.max()), float(left_grid.min()), float(left_grid.max())),
        cmap="coolwarm",
    )
    axis.set_title(f"rank-{interaction['rank']} interaction")
    figure.colorbar(image, ax=axis)
    figure.savefig(output / "interaction.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=int, default=1000)
    parser.add_argument("--benchmark-datasets", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.datasets < 1 or args.benchmark_datasets < 1:
        parser.error("dataset counts must be positive")
    torch.set_num_threads(1)
    config = GAMConfig(rich=True, heteroscedastic_probability=0.3)
    structures: Counter[str] = Counter()
    sparsities: Counter[str] = Counter()
    contributions: Counter[str] = Counter()
    noises: Counter[str] = Counter()
    families: Counter[str] = Counter()
    examples: dict[str, tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]] = {}
    interaction_example = None
    maximum_reconstruction_error = 0.0
    pure_interaction_datasets = 0
    maximum_family_probabilities = []
    entropies = []

    for index in range(args.datasets):
        torch.manual_seed(args.seed + index)
        X, y, metadata = GAM(
            seq_len=args.seq_len,
            num_features=args.features,
            max_features=args.features,
            config=config,
        )(return_metadata=True)
        if not torch.isfinite(X).all() or not torch.isfinite(y).all():
            raise RuntimeError(f"Non-finite dataset at seed {args.seed + index}")
        reconstructed = _evaluate_rich_signal(X, metadata["category_codes"], metadata)
        error = float((reconstructed - metadata["clean_signal"]).abs().max())
        maximum_reconstruction_error = max(maximum_reconstruction_error, error)

        structures[metadata["structural_regime"]] += 1
        sparsities[metadata["sparsity_regime"]] += 1
        contributions[metadata["contribution_regime"]] += 1
        noises[metadata["noise"]["noise_regime"]] += 1
        probabilities = metadata["family_probabilities"]
        maximum_family_probabilities.append(float(probabilities.max()))
        entropies.append(float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()))

        active = {effect["feature"] for effect in metadata["main_effects"]}
        if any(not set(item["features"]) <= active for item in metadata["interactions"]):
            pure_interaction_datasets += 1
        for effect in metadata["main_effects"]:
            feature, spec = effect["feature"], effect["spec"]
            family = spec["family"]
            families[family] += 1
            if family not in examples:
                categories = metadata["category_codes"][:, feature] if family == "categorical" else None
                examples[family] = (X[:, feature], categories, spec)
        if interaction_example is None:
            for interaction in metadata["interactions"]:
                left, right = interaction["features"]
                if all(
                    left_spec["family"] != "categorical" and right_spec["family"] != "categorical"
                    for _, left_spec, right_spec in interaction["rank_components"]
                ):
                    interaction_example = (interaction, X[:, left], X[:, right])
                    break

    if maximum_reconstruction_error >= 1e-4:
        raise RuntimeError(f"Signal reconstruction error is {maximum_reconstruction_error:.3g}")

    legacy_seconds = _benchmark(GAMConfig(), args.benchmark_datasets, args.seq_len, args.features)
    rich_seconds = _benchmark(config, args.benchmark_datasets, args.seq_len, args.features)
    summary = {
        "datasets": args.datasets,
        "structural_regimes": _frequencies(structures, args.datasets),
        "sparsity_regimes": _frequencies(sparsities, args.datasets),
        "contribution_regimes": _frequencies(contributions, args.datasets),
        "noise_regimes": _frequencies(noises, args.datasets),
        "main_effect_families": _frequencies(families, sum(families.values())),
        "mean_maximum_family_probability": sum(maximum_family_probabilities) / len(maximum_family_probabilities),
        "mean_family_entropy": sum(entropies) / len(entropies),
        "pure_interaction_dataset_fraction": pure_interaction_datasets / args.datasets,
        "maximum_reconstruction_error": maximum_reconstruction_error,
        "legacy_seconds_per_dataset": legacy_seconds,
        "rich_seconds_per_dataset": rich_seconds,
        "rich_to_legacy_runtime_ratio": rich_seconds / legacy_seconds,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _plot_examples(examples, args.output_dir)
        if interaction_example is not None:
            _plot_interaction(interaction_example, args.output_dir)


if __name__ == "__main__":
    main()
