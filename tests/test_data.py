"""Data pipeline tests: determinism, disjointness, period purity, coverage, batches."""

import dataclasses

import torch

from tinylm.data import (
    COMMON_VAL_SIZE,
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


def test_small_train_probe_rule() -> None:
    """n_train < PROBE_SIZE: probe shrinks to min(128, n_train) (default path)."""
    cfg = dataclasses.replace(load_config("configs/copy.yaml"), n_train=64)
    assert cfg.common_val is False and cfg.n_val == 512

    s = generate_splits(cfg)
    assert s.train.shape == (64, 32)
    assert s.train_probe.shape == (64, 32)
    assert torch.equal(s.train_probe, s.train)  # probe == train[:64] == whole train

    for seed in range(10):
        a = generate_splits(dataclasses.replace(cfg, seed=seed))
        b = generate_splits(dataclasses.replace(cfg, seed=seed))
        assert torch.equal(a.train, b.train)
        assert torch.equal(a.train_probe, b.train_probe)
        assert a.train_patterns.isdisjoint(a.val_patterns)
        assert {t for p in a.train_patterns for t in p} == set(range(cfg.vocab_size))


def test_common_val_fixed_across_sizes() -> None:
    """common_val: val + representative probe identical across sizes; trains nested."""
    sizes = (64, 256, 512, 1024)

    for seed in range(3):
        splits = {
            n: generate_splits(
                dataclasses.replace(
                    load_config(f"configs/copy_n{n:03d}.yaml"), seed=seed
                )
            )
            for n in sizes
        }

        assert all(splits[n].train.shape == (n, 32) for n in sizes)
        assert all(splits[n].val.shape == (COMMON_VAL_SIZE, 32) for n in sizes)

        base = splits[sizes[0]]
        for n in sizes[1:]:  # identical val across every size
            assert torch.equal(splits[n].val, base.val)
            assert splits[n].val_patterns == base.val_patterns

        for i in range(len(sizes) - 1):  # strictly nested train sets
            small, big = sizes[i], sizes[i + 1]
            assert splits[small].train_patterns < splits[big].train_patterns

        # representative probe identical across sizes >= PROBE_SIZE
        big_probes = [splits[n].train_probe for n in (256, 512, 1024)]
        for probe in big_probes[1:]:
            assert torch.equal(probe, big_probes[0])
