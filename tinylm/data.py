"""Copy and modular-arithmetic tasks. Deterministic given Config.seed."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch

from .utils import Config

PROBE_SIZE = 128  # fixed train-subset probe cap; never the live batch
COMMON_VAL_SIZE = 512  # size-independent val pool under common_val


@dataclass
class Splits:
    train: torch.Tensor  # (n_train, seq_len) long
    val: torch.Tensor  # (n_val, seq_len) long
    train_probe: (
        torch.Tensor
    )  # (min(PROBE_SIZE, n_train), seq_len) long, fixed train subset
    # Key type is task-dependent: copy holds period-length token tuples, modadd holds
    # ordered (a, b) operand pairs - check cfg.task before assuming token tuples.
    train_patterns: set[tuple[int, ...]]
    val_patterns: set[tuple[int, ...]]


def _has_true_period(pattern: tuple[int, ...], period: int) -> bool:
    """True when ``pattern``'s period is exactly ``period``, not a divisor of it."""
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


def _copy_splits(cfg: Config) -> Splits:
    rng = np.random.default_rng(cfg.seed)
    valid = [
        p
        for p in itertools.product(range(cfg.vocab_size), repeat=cfg.period)
        if _has_true_period(p, cfg.period)
    ]
    probe_size = min(PROBE_SIZE, cfg.n_train)
    # under common_val the val pool is the fixed COMMON_VAL_SIZE, not cfg.n_val
    n_val = COMMON_VAL_SIZE if cfg.common_val else cfg.n_val

    if len(valid) < cfg.n_train + n_val:
        raise ValueError(
            f"not enough period-{cfg.period} patterns: {len(valid)} < {cfg.n_train + n_val}"
        )

    order = [valid[i] for i in rng.permutation(len(valid))]

    if cfg.common_val:
        # size-independent split: val is a fixed first window and train starts at the
        # same index for every size, so val is identical across sizes and the train
        # sets are strictly nested (train64 ⊂ train256 ⊂ train512 ⊂ train1024)
        val_patterns = set(order[:n_val])
        train_patterns = set(order[n_val : n_val + cfg.n_train])
        # representative probe: the first probe_size SELECTED train patterns (a random
        # draw via the shuffled order), sorted into the tensor like train/val are. It is
        # a random sample of the train distribution and identical across sizes >= PROBE_SIZE
        probe = torch.tensor(
            [
                _expand(p, cfg.seq_len)
                for p in sorted(order[n_val : n_val + probe_size])
            ],
            dtype=torch.long,
        )
    else:
        chosen = order[: cfg.n_train + n_val]
        train_patterns = set(chosen[: cfg.n_train])
        val_patterns = set(chosen[cfg.n_train :])
        probe = None  # the frozen default path: train[:probe_size] below

    assert train_patterns.isdisjoint(val_patterns), (
        "train/val pattern sets must be disjoint"
    )

    train_tokens = {t for p in train_patterns for t in p}
    if train_tokens != set(range(cfg.vocab_size)):
        # ValueError (not a bare assert) so the message survives -O and reaches the
        # sweep runner's stderr_tail
        missing = sorted(set(range(cfg.vocab_size)) - train_tokens)
        raise ValueError(f"vocab coverage: tokens {missing} missing from training")

    train = torch.tensor(
        [_expand(p, cfg.seq_len) for p in sorted(train_patterns)], dtype=torch.long
    )
    val = torch.tensor(
        [_expand(p, cfg.seq_len) for p in sorted(val_patterns)], dtype=torch.long
    )

    if probe is None:
        probe = train[:probe_size]

    return Splits(
        train=train,
        val=val,
        train_probe=probe,
        train_patterns=train_patterns,
        val_patterns=val_patterns,
    )


def _modadd_splits(cfg: Config) -> Splits:
    """Modular-arithmetic splits: sequences [a, PLUS, b, EQ, (a+b) % p], T=5.

    Pairs are ORDERED, so (a, b) != (b, a): commutativity means a pair and its
    mirror share an answer, which is standard random-fraction grokking design
    (Power et al.), not a leak - logged, not "fixed".
    """
    p = cfg.modulus
    rng = np.random.default_rng(cfg.seed)
    pairs = [(a, b) for a in range(p) for b in range(p)]

    if cfg.n_train + cfg.n_val > len(pairs):
        raise ValueError(
            f"not enough ordered pairs: {len(pairs)} < {cfg.n_train + cfg.n_val}"
        )

    perm = rng.permutation(len(pairs))
    train_patterns = {pairs[i] for i in perm[: cfg.n_train]}
    val_patterns = {pairs[i] for i in perm[cfg.n_train : cfg.n_train + cfg.n_val]}

    assert train_patterns.isdisjoint(val_patterns), (
        "train/val pair sets must be disjoint"
    )

    def seq(pair: tuple[int, int]) -> list[int]:
        a, b = pair

        return [a, p, b, p + 1, (a + b) % p]

    train = torch.tensor([seq(pr) for pr in sorted(train_patterns)], dtype=torch.long)
    val = torch.tensor([seq(pr) for pr in sorted(val_patterns)], dtype=torch.long)
    probe_size = min(PROBE_SIZE, cfg.n_train)

    return Splits(
        train=train,
        val=val,
        train_probe=train[:probe_size],
        train_patterns=train_patterns,
        val_patterns=val_patterns,
    )


def generate_splits(cfg: Config) -> Splits:
    """Generate train/val/probe splits for the configured task."""
    if cfg.task == "modadd":
        return _modadd_splits(cfg)

    return _copy_splits(cfg)


def make_train_generator(cfg: Config) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    return g


def train_batch(splits: Splits, cfg: Config, g: torch.Generator) -> torch.Tensor:
    idx = torch.randint(splits.train.shape[0], (cfg.batch_size,), generator=g)

    return splits.train[idx]
