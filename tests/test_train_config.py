import pytest

yaml = pytest.importorskip("yaml", reason="Training config tests require tabicl[pretrain]")
pytest.importorskip("xgboost", reason="Training config tests require tabicl[pretrain]")

from tabicl.train._train_config import build_parser, parse_args


def test_yaml_values_are_loaded_and_cli_overrides_them(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "batch_size": 32,
                "prior_type": "gam",
                "gam_max_interactions": 3,
                "gam_rich": True,
                "gam_strong_interaction_heredity": False,
                "use_flash_attn3": False,
            }
        )
    )

    config = parse_args(["--config", str(path), "--batch_size", "64"])

    assert config.batch_size == 64
    assert config.prior_type == "gam"
    assert config.gam_max_interactions == 3
    assert config.gam_rich is True
    assert config.gam_strong_interaction_heredity is False
    assert config.use_flash_attn3 is False


def test_unknown_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text("not_an_argument: 1\n")

    with pytest.raises(SystemExit, match="2"):
        parse_args(["--config", str(path)])


def test_legacy_cli_is_unchanged_without_config():
    legacy = build_parser().parse_args(["--batch_size", "16", "--use_flash_attn3", "True"])
    parsed = parse_args(["--batch_size", "16", "--use_flash_attn3", "True"])

    assert vars(parsed) == vars(legacy)
