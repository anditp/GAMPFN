"""CLI entry point: python -m tabicl.train"""
from tabicl.train._train_config import parse_args
from tabicl.train._run import Trainer
from torch.multiprocessing import set_start_method

if __name__ == "__main__":
    config = parse_args()

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    trainer = Trainer(config)
    trainer.train()
