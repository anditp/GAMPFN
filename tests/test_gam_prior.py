import numpy as np
import pytest
import torch

pytest.importorskip("xgboost", reason="GAM prior tests require tabicl[pretrain]")

from tabicl.prior import GAMConfig, PriorDataset
from tabicl.prior._gam import GAM


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
