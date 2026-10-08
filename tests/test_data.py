"""Data pipeline tests: determinism, disjointness, period purity, coverage, batches."""

import torch

from tinylm.data import (
    PROBE_SIZE,
    _has_true_period,
    generate_splits,
    make_train_generator,
    train_batch,
)
from tinylm.utils import Config, load_config


def demo_cfg() -> Config:
    return load_config("configs/copy.yaml")


def test_deterministic_under_seed() -> None:
    cfg = demo_cfg()
    a, b = generate_splits(cfg), generate_splits(cfg)
    assert torch.equal(a.train, b.train) and torch.equal(a.val, b.val)
    assert torch.equal(a.train_probe, b.train_probe)


def test_split_disjoint() -> None:
    s = generate_splits(demo_cfg())
    assert s.train_patterns.isdisjoint(s.val_patterns)
    assert s.train.shape[0] == 512 and s.val.shape[0] == 512
    assert s.train_probe.shape == (PROBE_SIZE, s.train.shape[1])


def test_period_purity() -> None:
    s = generate_splits(demo_cfg())
    for p in list(s.train_patterns) + list(s.val_patterns):
        assert _has_true_period(p, 4)
    assert not _has_true_period((0, 0, 0, 0), 4)  # period 1
    assert not _has_true_period((1, 2, 1, 2), 4)  # period 2


def test_vocab_coverage_and_novelty() -> None:
    s = generate_splits(demo_cfg())
    assert {t for p in s.train_patterns for t in p} == set(range(8))
    for row, p in zip(s.val.tolist(), sorted(s.val_patterns), strict=True):
        assert row == list(p) * (s.val.shape[1] // 4)  # rule-consistent, novel content


def test_batches() -> None:
    cfg = demo_cfg()
    s = generate_splits(cfg)
    b = train_batch(s, cfg, make_train_generator(cfg))
    assert b.dtype == torch.long and b.shape == (cfg.batch_size, cfg.seq_len)
    b2 = train_batch(s, cfg, make_train_generator(cfg))  # same seed -> same first batch
    assert torch.equal(b, b2)
