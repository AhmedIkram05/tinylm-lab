"""Training smoke tests: overfit-one-batch and the yaml -> Config round-trip."""

import pathlib
from dataclasses import asdict

import torch
import yaml

from tinylm.data import generate_splits, make_train_generator, train_batch
from tinylm.model import TinyLM
from tinylm.train import causal_lm_loss
from tinylm.utils import Config, config_hash, load_config, set_seeds


def test_overfit_one_batch() -> None:
    cfg = load_config("configs/copy.yaml")
    set_seeds(cfg.seed)
    splits = generate_splits(cfg)
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg.weight_decay,  # pinned 0.0, same as train.py - AdamW defaults to 0.01
    )
    batch = train_batch(splits, cfg, make_train_generator(cfg))

    def batch_loss() -> float:
        return causal_lm_loss(model(batch), batch).item()

    model.train()
    initial = batch_loss()

    for _ in range(50):
        opt.zero_grad()
        causal_lm_loss(model(batch), batch).backward()
        opt.step()
    final = batch_loss()

    assert final < 0.3 * initial, (
        f"{initial=} {final=}"
    )  # relative drop, not an absolute bar


def test_yaml_round_trip() -> None:
    cfg = load_config("configs/copy.yaml")
    assert (cfg.vocab_size, cfg.seq_len, cfg.period) == (8, 32, 4)
    assert (cfg.n_train, cfg.n_val) == (512, 512)
    assert (cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff) == (128, 4, 2, 512)


def test_config_hash_stable_across_key_order() -> None:
    """config_hash must identify a run, so YAML key order cannot change it."""
    raw = yaml.safe_load(pathlib.Path("configs/copy.yaml").read_text())
    reordered = Config(**dict(sorted(raw.items(), reverse=True)))
    assert config_hash(reordered) == config_hash(load_config("configs/copy.yaml"))


def test_config_hash_changes_with_any_value() -> None:
    """...and any value change must change it, or runs get mislabelled in the logs."""
    baseline = config_hash(load_config("configs/copy.yaml"))
    for field, value in (("lr", 2e-3), ("seed", 7), ("d_model", 256), ("dropout", 0.1)):
        assert (
            config_hash(
                Config(**{**asdict(load_config("configs/copy.yaml")), field: value})
            )
            != baseline
        )
