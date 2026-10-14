import numpy as np
import pytest
import torch

pytest.importorskip("xgboost", reason="Training tests require tabicl[pretrain]")

from tabicl._model.additive import AdditivePerceiver
from tabicl.prior._real import MixedRegressionDataset, RealRegressionDataset
from tabicl.train._run import Trainer
from tabicl.train._train_config import parse_args


def _config(**overrides):
    values = {
        "regression_method": "additive",
        "prior_type": "gam",
        "gam_rich": "true",
        "gam_interaction_probability": "0",
        "device": "cpu",
        "embed_dim": "16",
        "col_num_blocks": "1",
        "col_nhead": "4",
        "col_num_inds": "4",
        "row_num_blocks": "1",
        "row_nhead": "4",
        "additive_num_memory_tokens": "4",
        "additive_num_decoder_blocks": "1",
        "ff_factor": "2",
        "zero_init": "false",
    }
    values.update({key: str(value) for key, value in overrides.items()})
    argv = [item for key, value in values.items() for item in (f"--{key}", value)]
    return parse_args(argv)


def _trainer(config):
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.ddp = False
    trainer.master_process = False
    trainer.build_model()
    trainer.configure_amp()
    return trainer


def test_additive_trainer_runs_one_backward_step():
    torch.manual_seed(5)
    trainer = _trainer(_config())
    assert isinstance(trainer.model, AdditivePerceiver)

    batch_size, seq_len, max_features, train_size = 2, 10, 4, 6
    feature_counts = torch.tensor([3, 4])
    X = torch.randn(batch_size, seq_len, max_features)
    X[0, :, 3] = 0
    y = torch.randn(batch_size, seq_len)
    clean_target = torch.randn(batch_size, seq_len)
    effect_target = torch.randn(batch_size, seq_len, max_features)
    effect_target[0, :, 3] = 0
    micro_batch = (
        X,
        y,
        feature_counts,
        torch.full((batch_size,), seq_len),
        torch.full((batch_size,), train_size),
        clean_target,
        effect_target,
    )

    results = trainer.run_micro_batch(micro_batch, micro_batch_idx=0, num_micro_batches=1)

    assert results.keys() == {"loss", "target_mse", "effect_loss"}
    assert all(torch.isfinite(torch.tensor(value)) for value in results.values())
    gradients = [parameter.grad for parameter in trainer.model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_additive_trainer_rejects_interaction_supervision():
    trainer = Trainer.__new__(Trainer)
    trainer.config = _config(gam_interaction_probability=0.5)
    trainer.ddp = False
    trainer.master_process = False

    with pytest.raises(ValueError, match="interactions to be disabled"):
        trainer.build_model()


def test_real_batch_uses_prediction_loss_only():
    torch.manual_seed(5)
    trainer = _trainer(_config())
    batch_size, seq_len, num_features, train_size = 2, 10, 4, 6
    X = torch.randn(batch_size, seq_len, num_features)
    y = torch.randn(batch_size, seq_len)
    micro_batch = (
        X,
        y,
        torch.full((batch_size,), num_features),
        torch.full((batch_size,), seq_len),
        torch.full((batch_size,), train_size),
        y.clone(),
        torch.full((batch_size, seq_len, num_features), 1e6),
        torch.zeros(batch_size, dtype=torch.bool),
    )

    results = trainer.run_micro_batch(micro_batch, micro_batch_idx=0, num_micro_batches=1)

    assert results["effect_loss"] == 0.0
    assert results["target_mse"] > 0.0


def test_real_regression_dataset_shapes_and_uses_gam_target_normalization(tmp_path):
    table_dir = tmp_path / "123"
    table_dir.mkdir()
    np.save(table_dir / "X.npy", np.arange(60, dtype=np.float32).reshape(20, 3))
    np.save(table_dir / "y.npy", np.linspace(-3, 5, 20, dtype=np.float32))
    dataset = RealRegressionDataset(
        data_dir=tmp_path,
        batch_size=2,
        min_seq_len=None,
        max_seq_len=8,
        min_train_size=0.5,
        max_train_size=0.5,
        max_features=3,
    )

    X, y, d, seq_lens, train_sizes, clean, effects, supervised = dataset.get_batch()

    assert X.shape == (2, 8, 3)
    assert y.shape == clean.shape == (2, 8)
    assert effects.shape == (2, 8, 3)
    assert torch.equal(d, torch.tensor([3, 3]))
    assert torch.equal(seq_lens, torch.tensor([8, 8]))
    assert torch.equal(train_sizes, torch.tensor([4, 4]))
    assert torch.allclose(y.mean(dim=1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(y.std(dim=1), torch.ones(2), atol=1e-6)
    assert not supervised.any()


def test_real_regression_dataset_clips_extreme_targets(tmp_path):
    table_dir = tmp_path / "123"
    table_dir.mkdir()
    np.save(table_dir / "X.npy", np.arange(200, dtype=np.float32).reshape(100, 2))
    targets = np.zeros(100, dtype=np.float32)
    targets[-1] = 1e6
    np.save(table_dir / "y.npy", targets)
    dataset = RealRegressionDataset(
        data_dir=tmp_path,
        batch_size=1,
        min_seq_len=None,
        max_seq_len=100,
        min_train_size=0.5,
        max_train_size=0.5,
        max_features=2,
    )

    _, y, *_ = dataset.get_batch()

    assert torch.isfinite(y).all()
    assert y.abs().max() <= 100


class _FixedDataset:
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        return self.batch


def test_mixed_regression_dataset_routes_all_three_sources(monkeypatch):
    X = torch.ones(2, 5, 3)
    y = torch.ones(2, 5)
    d = torch.full((2,), 3)
    seq_lens = torch.full((2,), 5)
    train_sizes = torch.full((2,), 3)
    base = (X, y, d, seq_lens, train_sizes)
    gam = _FixedDataset((*base, X.clone(), X.clone()))
    tabicl = _FixedDataset(base)
    real = _FixedDataset((*base, y.clone(), torch.zeros_like(X), torch.zeros(2, dtype=torch.bool)))
    mixed = MixedRegressionDataset(
        synthetic=gam,
        real=real,
        real_probability=0.25,
        tabicl=tabicl,
        tabicl_probability=0.25,
    )

    monkeypatch.setattr(np.random, "random", lambda: 0.1)
    assert not next(mixed)[-1].any()
    monkeypatch.setattr(np.random, "random", lambda: 0.3)
    assert not next(mixed)[-1].any()
    monkeypatch.setattr(np.random, "random", lambda: 0.8)
    assert next(mixed)[-1].all()

    with pytest.raises(ValueError, match="must not exceed"):
        MixedRegressionDataset(gam, real, 0.6, tabicl, 0.5)
