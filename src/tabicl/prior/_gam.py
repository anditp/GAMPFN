from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ._reg2cls import outlier_removing, standard_scaling


@dataclass(frozen=True)
class GAMConfig:
    """High-impact controls for the regression-only GA2M prior."""

    rich: bool = False
    strong_interaction_heredity: bool = False
    categorical_probability: float = 0.2
    min_active_fraction: float = 0.3
    interaction_probability: float = 0.75
    max_interactions: int = 5
    max_interaction_fraction: float = 0.5
    min_noise_scale: float = 0.01
    max_noise_scale: float = 1.0
    heteroscedastic_probability: float = 0.5
    outlier_probability: float = 0.1

    def __post_init__(self) -> None:
        probabilities = (
            self.categorical_probability,
            self.interaction_probability,
            self.heteroscedastic_probability,
            self.outlier_probability,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("Probabilities must lie between 0 and 1.")
        if not 0 < self.min_active_fraction <= 1:
            raise ValueError("min_active_fraction must lie in (0, 1].")
        if self.max_interactions < 0:
            raise ValueError("max_interactions must be non-negative.")
        if not 0 <= self.max_interaction_fraction <= 0.5:
            raise ValueError("max_interaction_fraction must lie between 0 and 0.5.")
        if not 0 < self.min_noise_scale <= self.max_noise_scale:
            raise ValueError("Noise scales must satisfy 0 < min <= max.")


def _standardize(values: Tensor) -> Tensor:
    """Center one vector and scale it to unit sample variance."""
    values = values - values.mean()
    return values / values.std().clamp_min(1e-6)


def _copula_features(seq_len: int, num_features: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Generate correlated Gaussian features and their uniform percentiles."""
    rank = min(num_features, int(torch.randint(1, 4, (), device=device)))
    factors = torch.randn(seq_len, rank, device=device)
    loadings = torch.randn(rank, num_features, device=device) / math.sqrt(rank)
    strength = torch.empty((), device=device).uniform_(0.2, 0.9)
    dependent = _standardize_columns(factors @ loadings)
    normal = strength * dependent + torch.sqrt(1 - strength.square()) * torch.randn(
        seq_len, num_features, device=device
    )
    normal = _standardize_columns(normal)
    uniform = (0.5 * (1 + torch.erf(normal / math.sqrt(2)))).clamp_(1e-6, 1 - 1e-6)
    return normal, uniform


def _standardize_columns(values: Tensor) -> Tensor:
    """Standardize every feature column independently."""
    return (values - values.mean(dim=0)) / values.std(dim=0).clamp_min(1e-6)


def _mixed_marginals(
    normal: Tensor, uniform: Tensor, categorical_probability: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Give each correlated latent column a random numeric or categorical marginal."""
    seq_len, num_features = normal.shape
    device = normal.device
    # Categorical columns override the numeric marginal sampled below.
    categorical = torch.rand(num_features, device=device) < categorical_probability
    marginal = torch.randint(0, 4, (num_features,), device=device)
    features = torch.empty_like(normal)

    # Numeric marginals: Gaussian, uniform, Cauchy-like, and log-normal.
    features[:, marginal == 0] = normal[:, marginal == 0]
    features[:, marginal == 1] = 2 * uniform[:, marginal == 1] - 1
    features[:, marginal == 2] = torch.tan(math.pi * (uniform[:, marginal == 2] - 0.5)).clamp_(-20, 20)
    features[:, marginal == 3] = torch.exp(normal[:, marginal == 3].clamp_(-4, 4))

    # Split percentiles into 2-100 bins and retain the codes for lookup effects.
    category_sizes = torch.randint(2, 101, (num_features,), device=device)
    category_codes = torch.floor(uniform * category_sizes).long().clamp_min_(0)
    features[:, categorical] = category_codes[:, categorical].to(features.dtype)
    category_codes[:, ~categorical] = -1

    # Reuse TabICL's clipping and scaling while preserving each marginal's shape.
    features = standard_scaling(outlier_removing(features.float(), threshold=4))
    return features, category_codes, category_sizes


def _effect(values: Tensor, categories: Tensor | None = None) -> Tensor:
    """Sample one categorical lookup or numeric GAM main-effect function."""
    device = values.device
    if categories is not None:
        lookup = torch.randn(int(categories.max()) + 1, device=device)
        return _standardize(lookup[categories])

    kind = int(torch.randint(0, 5, (), device=device))
    if kind == 0:
        direction = torch.where(torch.rand((), device=device) < 0.5, -1.0, 1.0)
        effect = direction * values
    elif kind == 1:
        direction = torch.where(torch.rand((), device=device) < 0.5, -1.0, 1.0)
        effect = direction * torch.tanh(torch.empty((), device=device).uniform_(0.5, 2.5) * values)
    elif kind == 2:
        num_knots = int(torch.randint(3, 7, (), device=device))
        knots = torch.empty(num_knots, device=device).uniform_(-1.5, 1.5)
        coefficients = torch.randn(num_knots + 3, device=device)
        polynomial = torch.stack((values, values.square(), values.pow(3)), dim=1) @ coefficients[:3]
        effect = polynomial + (F.relu(values[:, None] - knots).pow(3) * coefficients[3:]).sum(dim=1)
    elif kind == 3:
        num_frequencies = int(torch.randint(2, 6, (), device=device))
        base_frequency = torch.empty((), device=device).uniform_(0.5, 2.0)
        frequencies = base_frequency * torch.arange(1, num_frequencies + 1, device=device)
        phases = torch.empty(num_frequencies, device=device).uniform_(0, 2 * math.pi)
        amplitudes = torch.randn(num_frequencies, device=device) / math.sqrt(num_frequencies)
        effect = (amplitudes * torch.sin(values[:, None] * frequencies + phases)).sum(dim=1)
    else:
        centers = torch.empty(3, device=device).uniform_(-2, 2)
        widths = torch.empty(3, device=device).uniform_(0.3, 1.5)
        weights = torch.randn(3, device=device)
        effect = (weights * torch.exp(-0.5 * ((values[:, None] - centers) / widths).square())).sum(dim=1)
    return _standardize(effect)


def _component(features: Tensor, category_codes: Tensor, feature: int) -> Tensor:
    """Evaluate one feature with the appropriate categorical or numeric effect."""
    categories = category_codes[:, feature]
    return _effect(features[:, feature], None if categories[0] < 0 else categories)


def _signal(features: Tensor, category_codes: Tensor, config: GAMConfig) -> Tensor:
    """Combine active main effects and optional strong-hierarchy interactions."""
    num_features = features.shape[1]
    active_fraction = torch.empty((), device=features.device).uniform_(config.min_active_fraction, 1.0)
    num_active = max(1, math.ceil(num_features * float(active_fraction)))
    active = torch.randperm(num_features, device=features.device)[:num_active]

    main_effects = torch.stack([_component(features, category_codes, int(index)) for index in active])
    # Log-uniform magnitudes allow occasional dominant features without favoring large scales.
    weights = torch.empty(num_active, device=features.device).uniform_(math.log(0.1), math.log(10.0)).exp_()
    weights *= torch.where(torch.rand(num_active, device=features.device) < 0.5, -1.0, 1.0)
    main = _standardize((weights[:, None] * main_effects).sum(dim=0))

    use_interactions = (
        num_active > 1
        and config.max_interactions > 0
        and config.max_interaction_fraction > 0
        and torch.rand((), device=features.device) < config.interaction_probability
    )
    if not use_interactions:
        return main

    pairs = torch.combinations(active, r=2)
    num_pairs = int(torch.randint(1, min(config.max_interactions, len(pairs)) + 1, (), device=features.device))
    pairs = pairs[torch.randperm(len(pairs), device=features.device)[:num_pairs]]
    interactions = []
    for left, right in pairs:
        interactions.append(
            _component(features, category_codes, int(left)) * _component(features, category_codes, int(right))
        )
    interaction = _standardize(torch.stack(interactions).sum(dim=0))
    interaction = _standardize(interaction - (interaction @ main) / (main @ main).clamp_min(1e-6) * main)
    fraction = torch.empty((), device=features.device).uniform_(0, config.max_interaction_fraction)
    return _standardize(torch.sqrt(1 - fraction) * main + torch.sqrt(fraction) * interaction)


def _add_noise(signal: Tensor, features: Tensor, config: GAMConfig) -> Tensor:
    """Add sampled observation noise, optional heteroscedasticity, and rare outliers."""
    device = signal.device
    noise_type = int(torch.randint(0, 3, (), device=device))
    if noise_type == 0:
        noise = torch.randn_like(signal)
    elif noise_type == 1:
        noise = torch.distributions.Laplace(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)).sample(
            signal.shape
        )
    else:
        degrees = torch.empty((), device=device).uniform_(2.5, 10)
        noise = torch.distributions.StudentT(degrees).sample(signal.shape)

    if torch.rand((), device=device) < config.heteroscedastic_probability:
        driver = features[:, int(torch.randint(features.shape[1], (), device=device))].abs()
        noise *= 0.25 + driver / driver.mean().clamp_min(1e-6)
    noise = _standardize(noise)
    log_scale = torch.empty((), device=device).uniform_(math.log(config.min_noise_scale), math.log(config.max_noise_scale))
    target = signal + log_scale.exp() * noise

    max_outliers = int(0.02 * len(target))
    if max_outliers and torch.rand((), device=device) < config.outlier_probability:
        count = int(torch.randint(1, max_outliers + 1, (), device=device))
        rows = torch.randperm(len(target), device=device)[:count]
        magnitude = torch.empty(count, device=device).uniform_(4, 8)
        signs = torch.where(torch.rand(count, device=device) < 0.5, -1.0, 1.0)
        target[rows] += signs * magnitude * target.std().clamp_min(1e-6)
    return target


_NUMERIC_FAMILIES = (
    "linear",
    "saturating",
    "spline",
    "fourier",
    "rbf",
    "piecewise_linear",
    "step",
)


def _rand_sign(device: torch.device) -> Tensor:
    return torch.where(torch.rand((), device=device) < 0.5, -1.0, 1.0)


def _log_uniform(low: float, high: float, device: torch.device) -> Tensor:
    return torch.empty((), device=device).uniform_(math.log(low), math.log(high)).exp_()


def _center_scale(values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    center = values.mean()
    scale = values.std().clamp_min(1e-6)
    return (values - center) / scale, center, scale


def _reference_values(values: Tensor, size: int = 256) -> Tensor:
    quantiles = torch.linspace(0.001, 0.999, min(size, max(32, len(values))), device=values.device)
    return torch.quantile(values.float(), quantiles)


def _separated_percentiles(count: int, separation: float, device: torch.device) -> Tensor:
    """Sample sorted percentiles with a small minimum separation."""
    for _ in range(100):
        points = torch.empty(count, device=device).uniform_(0.05, 0.95).sort().values
        if count < 2 or bool(torch.all(points[1:] - points[:-1] >= separation)):
            return points
    # Deterministic fallback for the rare case where rejection sampling fails.
    return torch.linspace(0.05, 0.95, count + 2, device=device)[1:-1]


def _bspline_basis(values: Tensor, knots: Tensor, degree: int = 3) -> Tensor:
    """Evaluate an open B-spline basis with Cox-de Boor recursion."""
    x = values[:, None]
    basis = ((x >= knots[:-1]) & (x < knots[1:])).to(values.dtype)
    basis[values == knots[-1], -1] = 1
    for order in range(1, degree + 1):
        columns = len(knots) - order - 1
        next_basis = values.new_zeros((len(values), columns))
        for index in range(columns):
            left_denominator = knots[index + order] - knots[index]
            right_denominator = knots[index + order + 1] - knots[index + 1]
            if float(left_denominator.abs()) > 1e-12:
                next_basis[:, index] += (values - knots[index]) / left_denominator * basis[:, index]
            if float(right_denominator.abs()) > 1e-12:
                next_basis[:, index] += (knots[index + order + 1] - values) / right_denominator * basis[:, index + 1]
        basis = next_basis
    return basis


def _raw_spline(spec: dict[str, Any], values: Tensor) -> Tensor:
    knots = spec["knots"]
    coefficients = spec["coefficients"]
    left, right = knots[0], knots[-1]
    inside = values.clamp(float(left), float(right))
    result = _bspline_basis(inside, knots) @ coefficients

    # Continue linearly outside the boundary knots instead of cubic extrapolation.
    step = ((right - left) * 1e-3).clamp_min(1e-4)
    edge_values = torch.stack((left, left + step, right - step, right))
    edge_result = _bspline_basis(edge_values, knots) @ coefficients
    left_slope = (edge_result[1] - edge_result[0]) / step
    right_slope = (edge_result[3] - edge_result[2]) / step
    result = torch.where(values < left, edge_result[0] + left_slope * (values - left), result)
    return torch.where(values > right, edge_result[3] + right_slope * (values - right), result)


def _evaluate_raw_effect(spec: dict[str, Any], values: Tensor, categories: Tensor | None = None) -> Tensor:
    family = spec["family"]
    if family == "categorical":
        if categories is None:
            raise ValueError("Categorical effects require integer category codes.")
        return spec["lookup"][categories.long().clamp(0, len(spec["lookup"]) - 1)]
    if family == "linear":
        return spec["direction"] * values
    if family == "saturating":
        argument = spec["steepness"] * spec["reflection"] * (values - spec["location"])
        if spec["subtype"] == "tanh":
            return spec["direction"] * torch.tanh(argument)
        return spec["direction"] * F.softplus(argument) / spec["steepness"]
    if family == "spline":
        return _raw_spline(spec, values)
    if family == "fourier":
        argument = values[:, None] * spec["frequencies"] + spec["phases"]
        return (spec["amplitudes"] * torch.sin(argument)).sum(dim=1)
    if family == "rbf":
        distances = (values[:, None] - spec["centers"]) / spec["widths"]
        return (spec["weights"] * torch.exp(-0.5 * distances.square())).sum(dim=1)
    if family == "piecewise_linear":
        return spec["initial_slope"] * values + (
            spec["slope_changes"] * F.relu(values[:, None] - spec["thresholds"])
        ).sum(dim=1)
    if family == "step":
        return (
            spec["jumps"] * (values[:, None] > spec["thresholds"]).to(values.dtype)
        ).sum(dim=1)
    raise ValueError(f"Unknown effect family: {family}")


def _evaluate_effect(spec: dict[str, Any], values: Tensor, categories: Tensor | None = None) -> Tensor:
    raw = _evaluate_raw_effect(spec, values, categories)
    return (raw - spec["normalization_center"]) / spec["normalization_scale"]


def _finish_numeric_spec(spec: dict[str, Any], values: Tensor) -> dict[str, Any]:
    reference = _reference_values(values)
    raw = _evaluate_raw_effect(spec, reference)
    spec["normalization_center"] = raw.mean()
    spec["normalization_scale"] = raw.std().clamp_min(1e-6)
    return spec


def _sample_numeric_effect(values: Tensor, family_probabilities: Tensor) -> dict[str, Any]:
    device = values.device
    family = _NUMERIC_FAMILIES[int(torch.multinomial(family_probabilities, 1))]

    if family == "linear":
        spec: dict[str, Any] = {"family": family, "direction": _rand_sign(device)}
    elif family == "saturating":
        spec = {
            "family": family,
            "subtype": "tanh" if torch.rand((), device=device) < 0.5 else "softplus",
            "direction": _rand_sign(device),
            "reflection": _rand_sign(device),
            "location": torch.empty((), device=device).uniform_(-1.5, 1.5),
            "steepness": _log_uniform(0.5, 3.0, device),
        }
    elif family == "spline":
        number_internal = int(torch.randint(6, 17, (), device=device))
        boundary_probabilities = torch.tensor((0.01, 0.99), device=device)
        boundaries = torch.quantile(values.float(), boundary_probabilities)
        base = torch.arange(1, number_internal + 1, device=device) / (number_internal + 1)
        jitter = (torch.rand(number_internal, device=device) - 0.5) * 0.5 / (number_internal + 1)
        internal = torch.quantile(values.float(), (base + jitter).clamp(0.01, 0.99).sort().values)
        knots = torch.cat((boundaries[:1].repeat(4), internal, boundaries[1:].repeat(4)))
        number_coefficients = number_internal + 4
        roughness = _log_uniform(0.03, 1.0, device)
        coefficients = torch.empty(number_coefficients, device=device)
        coefficients[:2] = torch.randn(2, device=device)
        for index in range(2, number_coefficients):
            coefficients[index] = (
                2 * coefficients[index - 1]
                - coefficients[index - 2]
                + roughness * torch.randn((), device=device)
            )
        spec = {
            "family": family,
            "knots": knots,
            "coefficients": coefficients,
            "roughness": roughness,
        }
    elif family == "fourier":
        harmonics = int(torch.randint(1, 7, (), device=device))
        bounds = torch.quantile(values.float(), torch.tensor((0.01, 0.99), device=device))
        central_range = (bounds[1] - bounds[0]).clamp_min(1e-3)
        cycles = _log_uniform(0.5, 5.0, device)
        base_frequency = 2 * math.pi * cycles / central_range
        orders = torch.arange(1, harmonics + 1, device=device, dtype=values.dtype)
        decay = torch.empty((), device=device).uniform_(0.5, 2.5)
        spec = {
            "family": family,
            "frequencies": base_frequency * orders,
            "phases": torch.empty(harmonics, device=device).uniform_(0, 2 * math.pi),
            "amplitudes": torch.randn(harmonics, device=device) / orders.pow(decay),
            "cycles": cycles,
            "spectral_decay": decay,
        }
    elif family == "rbf":
        components = int(torch.randint(1, 7, (), device=device))
        center_probabilities = torch.empty(components, device=device).uniform_(0.05, 0.95)
        radii = torch.empty(components, device=device).uniform_(math.log(0.03), math.log(0.25)).exp_()
        centers = torch.quantile(values.float(), center_probabilities)
        lower = (center_probabilities - radii).clamp(0.01, 0.99)
        upper = (center_probabilities + radii).clamp(0.01, 0.99)
        widths = (torch.quantile(values.float(), upper) - torch.quantile(values.float(), lower)).abs() / 2
        spec = {
            "family": family,
            "centers": centers,
            "widths": widths.clamp_min(1e-3),
            "weights": torch.randn(components, device=device),
            "probability_radii": radii,
        }
    elif family == "piecewise_linear":
        change_points = int(torch.randint(1, 9, (), device=device))
        percentiles = _separated_percentiles(change_points, 0.03, device)
        spec = {
            "family": family,
            "thresholds": torch.quantile(values.float(), percentiles),
            "threshold_percentiles": percentiles,
            "initial_slope": torch.randn((), device=device),
            "slope_changes": torch.randn(change_points, device=device),
        }
    else:
        change_points = 1 + min(int(torch.poisson(torch.tensor(1.5, device=device))), 5)
        percentiles = _separated_percentiles(change_points, 0.05, device)
        jumps = torch.randn(change_points, device=device)
        monotone = bool(torch.rand((), device=device) < 0.3)
        if monotone:
            jumps = _rand_sign(device) * jumps.abs()
        spec = {
            "family": family,
            "thresholds": torch.quantile(values.float(), percentiles),
            "threshold_percentiles": percentiles,
            "jumps": jumps,
            "monotone": monotone,
        }

    return _finish_numeric_spec(spec, values)


def _sample_categorical_effect(number_categories: int, device: torch.device) -> dict[str, Any]:
    draw = float(torch.rand((), device=device))
    if draw < 0.6:
        subtype = "nominal"
        lookup = torch.randn(number_categories, device=device)
    elif draw < 0.85:
        subtype = "ordinal"
        increment_scale = _log_uniform(0.1, 1.0, device)
        lookup = torch.randn(number_categories, device=device).mul_(increment_scale).cumsum(0)
    else:
        subtype = "grouped"
        groups = int(torch.randint(2, min(8, number_categories) + 1, (), device=device))
        assignments = torch.randperm(number_categories, device=device).remainder(groups)
        lookup = torch.randn(groups, device=device)[assignments]
    _, center, scale = _center_scale(lookup)
    return {
        "family": "categorical",
        "subtype": subtype,
        "lookup": lookup,
        "normalization_center": center,
        "normalization_scale": scale,
    }


def _sample_effect_spec(
    features: Tensor,
    category_codes: Tensor,
    category_sizes: Tensor,
    feature: int,
    family_probabilities: Tensor,
) -> dict[str, Any]:
    if category_codes[0, feature] >= 0:
        return _sample_categorical_effect(int(category_sizes[feature]), features.device)
    return _sample_numeric_effect(features[:, feature], family_probabilities)


def _effect_from_feature(
    spec: dict[str, Any], features: Tensor, category_codes: Tensor, feature: int
) -> Tensor:
    categories = category_codes[:, feature] if spec["family"] == "categorical" else None
    return _evaluate_effect(spec, features[:, feature], categories)


def _sample_active_features(num_features: int, device: torch.device) -> tuple[Tensor, str]:
    draw = float(torch.rand((), device=device))
    if draw < 0.45:
        regime = "very_sparse"
        number = int(torch.randint(1, min(5, num_features) + 1, (), device=device))
    elif draw < 0.85:
        regime = "proportional_sparse"
        fraction = _log_uniform(0.05, 0.40, device)
        number = max(1, math.ceil(num_features * float(fraction)))
    else:
        regime = "dense"
        fraction = torch.empty((), device=device).uniform_(0.40, 1.0)
        number = max(1, math.ceil(num_features * float(fraction)))
    return torch.randperm(num_features, device=device)[:number], regime


def _sample_contribution_shares(count: int, device: torch.device) -> tuple[Tensor, str]:
    draw = float(torch.rand((), device=device))
    if draw < 0.30:
        concentration, regime = 10.0, "balanced"
    elif draw < 0.80:
        concentration, regime = 1.0, "mixed"
    else:
        concentration, regime = 0.15, "dominant"
    alpha = torch.full((count,), concentration, device=device)
    return torch.distributions.Dirichlet(alpha).sample(), regime


def _sample_structure(num_features: int, device: torch.device) -> str:
    if num_features < 2:
        return "additive"
    draw = float(torch.rand((), device=device))
    if draw < 0.50:
        return "additive"
    if draw < 0.90:
        return "sparse_ga2m"
    return "rich_ga2m"


def _interaction_values(
    interaction: dict[str, Any], features: Tensor, category_codes: Tensor
) -> Tensor:
    left, right = interaction["features"]
    values = features.new_zeros(len(features))
    for weight, left_spec, right_spec in interaction["rank_components"]:
        left_values = _effect_from_feature(left_spec, features, category_codes, left)
        right_values = _effect_from_feature(right_spec, features, category_codes, right)
        values += weight * left_values * right_values
    return (values - interaction["normalization_center"]) / interaction["normalization_scale"]


def _sample_interaction(
    pair: Tensor,
    rank: int,
    features: Tensor,
    category_codes: Tensor,
    category_sizes: Tensor,
    family_probabilities: Tensor,
) -> tuple[Tensor, dict[str, Any]]:
    left, right = int(pair[0]), int(pair[1])
    rank_shares = torch.distributions.Dirichlet(torch.ones(rank, device=features.device)).sample()
    rank_weights = rank_shares.sqrt() * torch.stack([_rand_sign(features.device) for _ in range(rank)])
    components: list[tuple[Tensor, dict[str, Any], dict[str, Any]]] = []
    raw = features.new_zeros(len(features))
    for weight in rank_weights:
        left_spec = _sample_effect_spec(features, category_codes, category_sizes, left, family_probabilities)
        right_spec = _sample_effect_spec(features, category_codes, category_sizes, right, family_probabilities)
        raw += weight * _effect_from_feature(left_spec, features, category_codes, left) * _effect_from_feature(
            right_spec, features, category_codes, right
        )
        components.append((weight, left_spec, right_spec))
    normalized, center, scale = _center_scale(raw)
    return normalized, {
        "features": (left, right),
        "rank": rank,
        "rank_components": components,
        "normalization_center": center,
        "normalization_scale": scale,
    }


def _rich_signal(
    features: Tensor,
    category_codes: Tensor,
    category_sizes: Tensor,
    config: GAMConfig,
) -> tuple[Tensor, dict[str, Any]]:
    device = features.device
    num_features = features.shape[1]
    family_probabilities = torch.distributions.Dirichlet(
        torch.full((len(_NUMERIC_FAMILIES),), 0.15, device=device)
    ).sample()
    active, sparsity_regime = _sample_active_features(num_features, device)
    structure = _sample_structure(num_features, device)
    if config.strong_interaction_heredity and structure != "additive" and len(active) < 2:
        missing = torch.tensor(
            [index for index in range(num_features) if index not in active.tolist()], device=device
        )
        if len(missing):
            active = torch.cat((active, missing[torch.randint(len(missing), (), device=device)].view(1)))

    main_shares, contribution_regime = _sample_contribution_shares(len(active), device)
    main_effects: list[dict[str, Any]] = []
    main_raw = features.new_zeros(len(features))
    for feature, share in zip(active.tolist(), main_shares):
        spec = _sample_effect_spec(features, category_codes, category_sizes, feature, family_probabilities)
        weight = share.sqrt()
        main_raw += weight * _effect_from_feature(spec, features, category_codes, feature)
        main_effects.append({"feature": feature, "weight": weight, "spec": spec})
    main, main_center, main_scale = _center_scale(main_raw)

    interactions: list[dict[str, Any]] = []
    interaction_fraction = features.new_zeros(())
    interaction_center = features.new_zeros(())
    interaction_scale = features.new_ones(())
    interaction = features.new_zeros(len(features))
    if structure != "additive":
        candidates = active if config.strong_interaction_heredity else torch.arange(num_features, device=device)
        pairs = torch.combinations(candidates, r=2)
        if len(pairs):
            upper = 3 if structure == "sparse_ga2m" else 8
            lower = 1 if structure == "sparse_ga2m" else min(2, len(pairs))
            number = int(torch.randint(lower, min(upper, len(pairs)) + 1, (), device=device))
            chosen = pairs[torch.randperm(len(pairs), device=device)[:number]]
            pair_values = []
            for pair in chosen:
                rank = 1 if structure == "sparse_ga2m" else int(torch.randint(1, 4, (), device=device))
                values, spec = _sample_interaction(
                    pair, rank, features, category_codes, category_sizes, family_probabilities
                )
                pair_values.append(values)
                interactions.append(spec)
            pair_shares = torch.distributions.Dirichlet(torch.ones(len(interactions), device=device)).sample()
            interaction_raw = features.new_zeros(len(features))
            for share, values, spec in zip(pair_shares, pair_values, interactions):
                spec["weight"] = share.sqrt()
                interaction_raw += spec["weight"] * values
            interaction, interaction_center, interaction_scale = _center_scale(interaction_raw)
            bounds = (0.05, 0.30) if structure == "sparse_ga2m" else (0.20, 0.60)
            interaction_fraction = torch.empty((), device=device).uniform_(*bounds)
        else:
            structure = "additive"

    if interactions:
        combined = torch.sqrt(1 - interaction_fraction) * main + torch.sqrt(interaction_fraction) * interaction
    else:
        combined = main
    signal, signal_center, signal_scale = _center_scale(combined)
    metadata: dict[str, Any] = {
        "structural_regime": structure,
        "sparsity_regime": sparsity_regime,
        "contribution_regime": contribution_regime,
        "family_probabilities": family_probabilities,
        "main_effects": main_effects,
        "main_center": main_center,
        "main_scale": main_scale,
        "interactions": interactions,
        "interaction_center": interaction_center,
        "interaction_scale": interaction_scale,
        "interaction_fraction": interaction_fraction,
        "signal_center": signal_center,
        "signal_scale": signal_scale,
        "strong_interaction_heredity": config.strong_interaction_heredity,
    }
    return signal, metadata


def _evaluate_rich_signal(features: Tensor, category_codes: Tensor, metadata: dict[str, Any]) -> Tensor:
    main_raw = features.new_zeros(len(features))
    for effect in metadata["main_effects"]:
        main_raw += effect["weight"] * _effect_from_feature(
            effect["spec"], features, category_codes, effect["feature"]
        )
    main = (main_raw - metadata["main_center"]) / metadata["main_scale"]

    if metadata["interactions"]:
        interaction_raw = features.new_zeros(len(features))
        for interaction in metadata["interactions"]:
            interaction_raw += interaction["weight"] * _interaction_values(
                interaction, features, category_codes
            )
        interaction = (interaction_raw - metadata["interaction_center"]) / metadata["interaction_scale"]
        combined = (
            torch.sqrt(1 - metadata["interaction_fraction"]) * main
            + torch.sqrt(metadata["interaction_fraction"]) * interaction
        )
    else:
        combined = main
    return (combined - metadata["signal_center"]) / metadata["signal_scale"]


def _rich_noise(signal: Tensor, features: Tensor, category_codes: Tensor, config: GAMConfig) -> tuple[Tensor, dict[str, Any]]:
    device = signal.device
    draw = float(torch.rand((), device=device))
    if draw < 0.60:
        noise_type = "gaussian"
        noise = torch.randn_like(signal)
    elif draw < 0.85:
        noise_type = "student_t"
        degrees = torch.empty((), device=device).uniform_(3, 10)
        noise = torch.distributions.StudentT(degrees).sample(signal.shape)
    else:
        noise_type = "laplace"
        noise = torch.distributions.Laplace(
            torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        ).sample(signal.shape)

    heteroscedastic = False
    heteroscedastic_type: str | None = None
    numeric = torch.nonzero(category_codes[0] < 0).flatten()
    if len(numeric) and torch.rand((), device=device) < config.heteroscedastic_probability:
        heteroscedastic = True
        driver = int(numeric[torch.randint(len(numeric), (), device=device)])
        values = features[:, driver]
        if torch.rand((), device=device) < 0.5:
            heteroscedastic_type = "tail"
            noise *= 0.5 + values.abs()
        else:
            heteroscedastic_type = "directional"
            strength = torch.empty((), device=device).uniform_(0.2, 0.8)
            noise *= torch.exp(_rand_sign(device) * strength * values.clamp(-2, 2))
    noise = _standardize(noise)

    regime_draw = float(torch.rand((), device=device))
    if regime_draw < 0.20:
        noise_regime, bounds = "clean", (0.95, 0.999)
    elif regime_draw < 0.85:
        noise_regime, bounds = "ordinary", (0.50, 0.95)
    else:
        noise_regime, bounds = "noisy", (0.10, 0.50)
    requested_r2 = torch.empty((), device=device).uniform_(*bounds)
    noise_scale = torch.sqrt((1 - requested_r2) / requested_r2)
    target = signal + noise_scale * noise

    outlier_count = 0
    if torch.rand((), device=device) < config.outlier_probability:
        fraction = _log_uniform(0.001, 0.02, device)
        outlier_count = max(1, min(len(target), math.ceil(len(target) * float(fraction))))
        rows = torch.randperm(len(target), device=device)[:outlier_count]
        magnitude = torch.empty(outlier_count, device=device).uniform_(4, 8)
        signs = torch.where(torch.rand(outlier_count, device=device) < 0.5, -1.0, 1.0)
        target[rows] += signs * magnitude * target.std().clamp_min(1e-6)

    realized_r2 = signal.var() / (signal.var() + (target - signal).var().clamp_min(1e-6))
    return target, {
        "noise_type": noise_type,
        "noise_regime": noise_regime,
        "requested_r2": requested_r2,
        "realized_r2": realized_r2,
        "heteroscedastic": heteroscedastic,
        "heteroscedastic_type": heteroscedastic_type,
        "outlier_count": outlier_count,
    }


class GAM:
    """Generate one padded regression dataset from a robust GA2M prior."""

    def __init__(
        self,
        regression: bool = True,
        seq_len: int = 1024,
        num_features: int = 100,
        max_features: int = 100,
        permute_features: bool = True,
        config: GAMConfig | None = None,
        device: str = "cpu",
        **_: object,
    ) -> None:
        if not regression:
            raise ValueError("The GAM prior supports regression only.")
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2.")
        if not 1 <= num_features <= max_features:
            raise ValueError("num_features must lie between 1 and max_features.")
        self.seq_len = seq_len
        self.num_features = num_features
        self.max_features = max_features
        self.permute_features = permute_features
        self.config = config or GAMConfig()
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(
        self, return_metadata: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Any]]:
        """Generate, normalize, permute, and pad one feature/target pair."""
        normal, uniform = _copula_features(self.seq_len, self.num_features, self.device)
        features, category_codes, category_sizes = _mixed_marginals(
            normal, uniform, self.config.categorical_probability
        )
        metadata: dict[str, Any] | None = None
        if self.config.rich:
            signal, metadata = _rich_signal(features, category_codes, category_sizes, self.config)
            target, noise_metadata = _rich_noise(signal, features, category_codes, self.config)
            target, target_center, target_scale = _center_scale(
                outlier_removing(target[:, None].float(), threshold=4).view(-1)
            )
            metadata.update(
                {
                    "clean_signal": signal,
                    "noise": noise_metadata,
                    "target_center": target_center,
                    "target_scale": target_scale,
                }
            )
        else:
            target = _add_noise(_signal(features, category_codes, self.config), features, self.config)
            target = standard_scaling(outlier_removing(target[:, None].float(), threshold=4)).view(-1)

        if self.permute_features:
            permutation = torch.randperm(self.num_features, device=self.device)
            features = features[:, permutation]
            category_codes = category_codes[:, permutation]
            category_sizes = category_sizes[permutation]
            if metadata is not None:
                inverse = torch.argsort(permutation)
                for effect in metadata["main_effects"]:
                    effect["feature"] = int(inverse[effect["feature"]])
                for interaction in metadata["interactions"]:
                    left, right = interaction["features"]
                    interaction["features"] = (int(inverse[left]), int(inverse[right]))
                metadata["feature_permutation"] = permutation
        if self.num_features < self.max_features:
            features = F.pad(features, (0, self.max_features - self.num_features))

        features, target = torch.nan_to_num(features), torch.nan_to_num(target)
        if return_metadata:
            if metadata is None:
                raise ValueError("Metadata is available only when GAMConfig.rich is true.")
            metadata["category_codes"] = category_codes
            metadata["category_sizes"] = category_sizes
            metadata["num_features"] = self.num_features
            return features, target, metadata
        return features, target
