import numpy as np
import pytest
import torch

pytest.importorskip("xgboost", reason="GAM prior tests require tabicl[pretrain]")

from tabicl.prior import GAMConfig, PriorDataset
from tabicl.prior._gam import (
    GAM,
    _evaluate_effect,
    _evaluate_rich_signal,
    _sample_numeric_effect,
)


def test_gam_generates_finite_padded_categorical_data_deterministically():
    config = GAMConfig(
        categorical_probability=1.0,
        interaction_probability=1.0,
        max_interactions=2,
        min_noise_scale=0.1,
        max_noise_scale=0.1,
        heteroscedastic_probability=1.0,
        outlier_probability=1.0,
    )

    torch.manual_seed(7)
    first = GAM(seq_len=128, num_features=5, max_features=8, config=config)()
    torch.manual_seed(7)
    second = GAM(seq_len=128, num_features=5, max_features=8, config=config)()

    X, y = first
    assert X.shape == (128, 8)
    assert y.shape == (128,)
    assert X.dtype == y.dtype == torch.float32
    assert torch.isfinite(X).all() and torch.isfinite(y).all()
    assert torch.equal(X, second[0]) and torch.equal(y, second[1])
    assert torch.count_nonzero(X[:, 5:]) == 0
    cardinalities = [torch.unique(X[:, column]).numel() for column in range(5)]
    assert max(cardinalities) <= 100
    assert max(cardinalities) > 20
    assert y.var() > 0.5


def test_gam_supports_interaction_free_data_and_rejects_classification():
    config = GAMConfig(interaction_probability=0.0, max_interaction_fraction=0.0)
    X, y = GAM(seq_len=64, num_features=3, max_features=3, config=config)()

    assert X.shape == (64, 3) and y.shape == (64,)
    with pytest.raises(ValueError, match="regression only"):
        GAM(regression=False)


def test_prior_dataset_gam_batch_contract():
    np.random.seed(11)
    torch.manual_seed(11)
    prior = PriorDataset(
        prior_type="gam",
        regression=True,
        gam_config=GAMConfig(interaction_probability=0.0),
        batch_size=3,
        batch_size_per_gp=2,
        min_features=2,
        max_features=6,
        max_seq_len=64,
        min_train_size=0.25,
        max_train_size=0.75,
        n_jobs=1,
    )

    X, y, d, seq_lens, train_sizes = prior.get_batch()

    assert X.shape == (3, 64, 6) and y.shape == (3, 64)
    assert d.shape == seq_lens.shape == train_sizes.shape == (3,)
    assert torch.all((0 < d) & (d <= 6))
    assert torch.all(seq_lens == 64)
    assert torch.all((16 <= train_sizes) & (train_sizes < 48))
    assert torch.isfinite(X).all() and torch.isfinite(y).all()
    assert all(torch.count_nonzero(X[index, :, d[index] :]) == 0 for index in range(3))


def test_prior_dataset_gam_reuses_variable_length_batching():
    np.random.seed(3)
    torch.manual_seed(3)
    prior = PriorDataset(
        prior_type="gam",
        regression=True,
        batch_size=3,
        batch_size_per_gp=1,
        min_features=2,
        max_features=4,
        min_seq_len=48,
        max_seq_len=65,
        seq_len_per_gp=True,
        n_jobs=1,
    )

    X, y, d, seq_lens, train_sizes = prior.get_batch()

    assert X.is_nested and y.is_nested
    assert len(torch.unique(seq_lens)) > 1
    assert torch.all((48 <= seq_lens) & (seq_lens < 65))
    assert torch.all((0 < train_sizes) & (train_sizes < seq_lens))
    assert all(x.shape == (length, 4) for x, length in zip(X.unbind(), seq_lens))
    assert all(target.shape == (length,) for target, length in zip(y.unbind(), seq_lens))

    with pytest.raises(ValueError, match="regression only"):
        PriorDataset(prior_type="gam", regression=False)


def test_rich_gam_metadata_reconstructs_the_clean_signal():
    config = GAMConfig(
        rich=True,
        categorical_probability=0.3,
        heteroscedastic_probability=0.0,
        outlier_probability=0.0,
    )
    torch.manual_seed(19)
    X, y, metadata = GAM(
        seq_len=128,
        num_features=8,
        max_features=10,
        config=config,
    )(return_metadata=True)

    reconstructed = _evaluate_rich_signal(
        X[:, :8], metadata["category_codes"], metadata
    )

    assert X.shape == (128, 10) and y.shape == (128,)
    assert torch.isfinite(X).all() and torch.isfinite(y).all()
    assert torch.allclose(reconstructed, metadata["clean_signal"], atol=1e-5, rtol=1e-5)
    assert metadata["structural_regime"] in {"additive", "sparse_ga2m", "rich_ga2m"}
    assert metadata["sparsity_regime"] in {"very_sparse", "proportional_sparse", "dense"}
    assert torch.allclose(metadata["family_probabilities"].sum(), torch.tensor(1.0))
    assert abs(float(metadata["clean_signal"].std()) - 1.0) < 1e-5


def test_every_rich_numeric_family_is_finite_and_normalized():
    values = torch.linspace(-3, 3, 256)
    for family in range(7):
        torch.manual_seed(100 + family)
        probabilities = torch.zeros(7)
        probabilities[family] = 1
        spec = _sample_numeric_effect(values, probabilities)
        effect = _evaluate_effect(spec, values)
        assert torch.isfinite(effect).all()
        assert effect.std() > 0.1


def test_rich_gam_can_sample_pure_interactions_and_enforce_heredity():
    pure_interaction_seen = False
    for seed in range(30):
        torch.manual_seed(seed)
        _, _, metadata = GAM(
            seq_len=64,
            num_features=8,
            max_features=8,
            permute_features=False,
            config=GAMConfig(rich=True, strong_interaction_heredity=False),
        )(return_metadata=True)
        active = {effect["feature"] for effect in metadata["main_effects"]}
        if any(not set(interaction["features"]) <= active for interaction in metadata["interactions"]):
            pure_interaction_seen = True
            break
    assert pure_interaction_seen

    for seed in range(10):
        torch.manual_seed(seed)
        _, _, metadata = GAM(
            seq_len=64,
            num_features=8,
            max_features=8,
            permute_features=False,
            config=GAMConfig(rich=True, strong_interaction_heredity=True),
        )(return_metadata=True)
        active = {effect["feature"] for effect in metadata["main_effects"]}
        assert all(set(interaction["features"]) <= active for interaction in metadata["interactions"])


def test_rich_gam_is_deterministic_and_metadata_requires_rich_mode():
    config = GAMConfig(rich=True)
    torch.manual_seed(23)
    first = GAM(seq_len=64, num_features=5, max_features=5, config=config)()
    torch.manual_seed(23)
    second = GAM(seq_len=64, num_features=5, max_features=5, config=config)()
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])

    with pytest.raises(ValueError, match="Metadata is available only"):
        GAM(seq_len=64, num_features=3, max_features=3)(return_metadata=True)
