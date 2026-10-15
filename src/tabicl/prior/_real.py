"""Real regression tasks and synthetic/real batch mixing for additive training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from ._dataset import Prior
from ._reg2cls import outlier_removing, standard_scaling


class RealRegressionDataset(IterableDataset):
    """Sample context/query tasks uniformly from downloaded regression tables."""

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int,
        min_seq_len: int | None,
        max_seq_len: int,
        min_train_size: int | float,
        max_train_size: int | float,
        max_features: int,
        log_seq_len: bool = False,
        replay_small: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.max_features = max_features
        self.log_seq_len = log_seq_len
        self.replay_small = replay_small

        self.tables = []
        for x_path in sorted(self.data_dir.glob("*/X.npy")):
            y_path = x_path.with_name("y.npy")
            if not y_path.is_file():
                continue
            x = np.load(x_path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r")
            if x.ndim == 2 and y.ndim == 1 and len(x) == len(y) and len(y) >= 2 and x.shape[1] > 0:
                self.tables.append((x_path, y_path, x.shape))
        if not self.tables:
            raise ValueError(f"No complete regression tables found in {self.data_dir}")

    @torch.no_grad()
    def get_batch(self) -> tuple[Tensor, ...]:
        x_path, y_path, shape = self.tables[np.random.randint(len(self.tables))]
        x_table = np.load(x_path, mmap_mode="r")
        y_table = np.load(y_path, mmap_mode="r")

        requested_len = Prior.sample_seq_len(
            self.min_seq_len,
            self.max_seq_len,
            log=self.log_seq_len,
            replay_small=self.replay_small,
        )
        seq_len = min(requested_len, shape[0])
        train_size = Prior.sample_train_size(self.min_train_size, self.max_train_size, seq_len)
        feature_limit = Prior.adjust_max_features(seq_len, self.max_features)
        num_features = min(shape[1], feature_limit)
        feature_ids = np.random.choice(shape[1], num_features, replace=False)

        x_batch = []
        y_batch = []
        for _ in range(self.batch_size):
            row_ids = np.random.choice(shape[0], seq_len, replace=False)
            x = np.asarray(x_table[row_ids])[:, feature_ids].astype(np.float32, copy=False)
            y = torch.from_numpy(np.asarray(y_table[row_ids], dtype=np.float32))
            y = standard_scaling(outlier_removing(y[:, None], threshold=4)).view(-1)
            x_batch.append(torch.from_numpy(x))
            y_batch.append(y)

        X = torch.stack(x_batch)
        y = torch.stack(y_batch)
        d = torch.full((self.batch_size,), num_features, dtype=torch.long)
        seq_lens = torch.full((self.batch_size,), seq_len, dtype=torch.long)
        train_sizes = torch.full((self.batch_size,), train_size, dtype=torch.long)
        effect_targets = torch.zeros(self.batch_size, seq_len, num_features)
        interaction_targets = torch.zeros(self.batch_size, seq_len)
        effect_supervision = torch.zeros(self.batch_size, dtype=torch.bool)
        return (
            X,
            y,
            d,
            seq_lens,
            train_sizes,
            y.clone(),
            effect_targets,
            interaction_targets,
            effect_supervision,
        )

    def __iter__(self) -> "RealRegressionDataset":
        return self

    def __next__(self) -> tuple[Tensor, ...]:
        return self.get_batch()

    def __repr__(self) -> str:
        return f"RealRegressionDataset(tables={len(self.tables)}, data_dir={self.data_dir!s})"


class MixedRegressionDataset(IterableDataset):
    """Mix effect-supervised GAM batches with prediction-only batches."""

    def __init__(
        self,
        synthetic,
        real: RealRegressionDataset | None = None,
        real_probability: float = 0.0,
        tabicl=None,
        tabicl_probability: float = 0.0,
    ) -> None:
        super().__init__()
        if real_probability < 0.0 or tabicl_probability < 0.0:
            raise ValueError("Mixture probabilities must be non-negative.")
        if real_probability + tabicl_probability > 1.0:
            raise ValueError("real_probability + tabicl_probability must not exceed 1.")
        if real_probability and real is None:
            raise ValueError("A real dataset is required when real_probability is non-zero.")
        if tabicl_probability and tabicl is None:
            raise ValueError("A TabICL prior is required when tabicl_probability is non-zero.")
        self.synthetic = synthetic
        self.real = real
        self.real_probability = real_probability
        self.tabicl = tabicl
        self.tabicl_probability = tabicl_probability

    @staticmethod
    def _prediction_only(batch: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        X, y, d, seq_lens, train_sizes = batch
        effect_targets = torch.zeros_like(X)
        interaction_targets = torch.zeros_like(y)
        effect_supervision = torch.zeros(d.shape[0], dtype=torch.bool, device=d.device)
        return (
            X,
            y,
            d,
            seq_lens,
            train_sizes,
            y.clone(),
            effect_targets,
            interaction_targets,
            effect_supervision,
        )

    def __iter__(self) -> "MixedRegressionDataset":
        return self

    def __next__(self) -> tuple[Tensor, ...]:
        draw = np.random.random()
        if draw < self.real_probability:
            return next(self.real)
        if draw < self.real_probability + self.tabicl_probability:
            return self._prediction_only(next(self.tabicl))
        batch = next(self.synthetic)
        supervised = torch.ones(batch[2].shape[0], dtype=torch.bool, device=batch[2].device)
        return *batch, supervised

    def __repr__(self) -> str:
        return (
            f"MixedRegressionDataset(real_probability={self.real_probability}, "
            f"tabicl_probability={self.tabicl_probability}, synthetic={self.synthetic!r}, "
            f"tabicl={self.tabicl!r}, real={self.real!r})"
        )
