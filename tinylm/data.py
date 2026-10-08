"""Period-4 repeated-token copy task. Deterministic given Config.seed."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch

from .utils import Config

PROBE_SIZE = 128  # fixed train-subset probe; never the live batch


@dataclass
class Splits:
    train: torch.Tensor  # (n_train, seq_len) long
    val: torch.Tensor  # (n_val, seq_len) long
    train_probe: torch.Tensor  # (PROBE_SIZE, seq_len) long, fixed subset of train
    train_patterns: set[tuple[int, ...]]
    val_patterns: set[tuple[int, ...]]


def _has_true_period(pattern: tuple[int, ...], period: int) -> bool:
    if len(set(pattern)) == 1:
        return False  # period 1

    for p in range(1, period):
        if period % p == 0 and all(pattern[i] == pattern[i % p] for i in range(period)):
            return False

    return True


def _expand(pattern: tuple[int, ...], seq_len: int) -> list[int]:
    assert seq_len % len(pattern) == 0, (
        "seq_len must be a multiple of the pattern length"
    )

    return list(pattern) * (seq_len // len(pattern))


def generate_splits(cfg: Config) -> Splits:
    assert cfg.n_train >= PROBE_SIZE, (
        f"train probe needs >= {PROBE_SIZE} train patterns"
    )
    rng = np.random.default_rng(cfg.seed)
    valid = [
        p
        for p in itertools.product(range(cfg.vocab_size), repeat=cfg.period)
        if _has_true_period(p, cfg.period)
    ]
    if len(valid) < cfg.n_train + cfg.n_val:
        raise ValueError(
            f"not enough period-{cfg.period} patterns: {len(valid)} < {cfg.n_train + cfg.n_val}"
        )

    order = rng.permutation(len(valid))
    chosen = [valid[i] for i in order[: cfg.n_train + cfg.n_val]]
    train_patterns = set(chosen[: cfg.n_train])
    val_patterns = set(chosen[cfg.n_train :])
    assert train_patterns.isdisjoint(val_patterns), (
        "train/val pattern sets must be disjoint"
    )

    train_tokens = {t for p in train_patterns for t in p}
    assert train_tokens == set(range(cfg.vocab_size)), (
        "every vocab token must appear in training"
    )

    train = torch.tensor(
        [_expand(p, cfg.seq_len) for p in sorted(train_patterns)], dtype=torch.long
    )
    val = torch.tensor(
        [_expand(p, cfg.seq_len) for p in sorted(val_patterns)], dtype=torch.long
    )
    probe = train[:PROBE_SIZE]

    return Splits(
        train=train,
        val=val,
        train_probe=probe,
        train_patterns=train_patterns,
        val_patterns=val_patterns,
    )


def make_train_generator(cfg: Config) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    return g


def train_batch(splits: Splits, cfg: Config, g: torch.Generator) -> torch.Tensor:
    idx = torch.randint(splits.train.shape[0], (cfg.batch_size,), generator=g)

    return splits.train[idx]
