from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ._reg2cls import outlier_removing, standard_scaling


@dataclass(frozen=True)
class GAMConfig:
    """High-impact controls for the regression-only GA2M prior."""

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


def _mixed_marginals(normal: Tensor, uniform: Tensor, categorical_probability: float) -> tuple[Tensor, Tensor]:
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
    return features, category_codes


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
    def __call__(self) -> tuple[Tensor, Tensor]:
        """Generate, normalize, permute, and pad one feature/target pair."""
        normal, uniform = _copula_features(self.seq_len, self.num_features, self.device)
        features, category_codes = _mixed_marginals(normal, uniform, self.config.categorical_probability)
        target = _add_noise(_signal(features, category_codes, self.config), features, self.config)
        target = standard_scaling(outlier_removing(target[:, None].float(), threshold=4)).view(-1)

        if self.permute_features:
            permutation = torch.randperm(self.num_features, device=self.device)
            features = features[:, permutation]
        if self.num_features < self.max_features:
            features = F.pad(features, (0, self.max_features - self.num_features))

        return torch.nan_to_num(features), torch.nan_to_num(target)
