"""Tiny config, seeding, and device helpers. Must keep this module small."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]


@dataclass
class Config:
    vocab_size: int = 8
    seq_len: int = 32
    period: int = 4
    n_train: int = 512
    n_val: int = 512
    common_val: bool = False  # size-independent val pool (dataset-size series)
    task: str = "copy"  # "copy" | "modadd"
    modulus: int = 11  # modadd: value-token count; vocab is modulus + PLUS + EQ
    train_frac: float | None = None  # modadd: fraction of the ordered pair-pool
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    steps: int = 500
    eval_every: int = 25
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.train_frac is not None:
            # Checked here, before deriving, so a bad train_frac fails as a train_frac
            # error rather than as a downstream int() crash or a confusing n_train message.
            if self.task != "modadd":
                raise ValueError(
                    "train_frac is modadd-only (the mod-arith pair-pool knob)"
                )

            if isinstance(self.train_frac, bool) or not isinstance(
                self.train_frac, (int, float)
            ):
                raise TypeError(f"train_frac must be a number, got {self.train_frac!r}")

            if not 0.0 < self.train_frac < 1.0:
                raise ValueError("train_frac must be in (0, 1)")

            # Modadd grokking knob: fraction of the ordered pair-pool. Derived here (not in
            # main()) so the sweep runner and the child derive identically - and therefore
            # hash identically, which the registry's hash_match rule depends on.
            n_train = int(self.train_frac * self.modulus * self.modulus)

            if n_train < 1:
                raise ValueError("train_frac too small: derives n_train < 1")

            self.n_train = n_train
            self.n_val = self.modulus * self.modulus - n_train

        self.validate()

    def validate(self) -> None:
        """Fail fast on bad config values."""
        int_fields = (
            "vocab_size",
            "seq_len",
            "period",
            "n_train",
            "n_val",
            "modulus",
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
            "batch_size",
            "steps",
            "eval_every",
            "seed",
        )

        for name in int_fields:
            if isinstance(getattr(self, name), bool) or not isinstance(
                getattr(self, name), int
            ):
                raise TypeError(f"{name} must be an int, got {getattr(self, name)!r}")

        for name in ("dropout", "lr", "weight_decay"):
            if isinstance(getattr(self, name), bool) or not isinstance(
                getattr(self, name), (int, float)
            ):
                raise TypeError(f"{name} must be a number, got {getattr(self, name)!r}")

        if self.task not in ("copy", "modadd"):
            raise ValueError(f"unknown task: {self.task!r} (expected copy|modadd)")

        # train_frac is validated in __post_init__, before it is used to derive
        # n_train/n_val, so validate() only sees derived (already checked) values.
        if self.task == "copy":
            # period is the copy task's repeated-token length; under modadd it only
            # anchors the accuracy window (period == seq_len - 1), so these two rules
            # are copy-only.
            if self.period >= self.seq_len:
                raise ValueError(
                    "period must be less than seq_len (accuracy window is empty)"
                )

            if self.seq_len % self.period != 0:
                raise ValueError("seq_len must be a multiple of period")

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        if self.task == "modadd":
            # pins the answer position, so "val_acc IS answer accuracy" is enforced
            if self.seq_len != 5:
                raise ValueError("modadd requires seq_len == 5")

            if self.vocab_size != self.modulus + 2:
                raise ValueError("modadd requires vocab_size == modulus + 2")

            if self.period != self.seq_len - 1:
                raise ValueError("modadd requires period == seq_len - 1")

        for name in int_fields:
            # seed 0 is valid, everything else must be positive
            if name != "seed" and getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        # seed is exempt from the positivity check above (0 is valid) but must still be a
        # valid numpy seed, or set_seeds raises deep inside training - and a sweep runner
        # triages exit codes, so it would look like a training crash, not a config error.
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")

        if self.lr <= 0:
            raise ValueError("lr must be positive")

        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping, got {type(raw)}")  # noqa: TRY004

    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known

    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")

    return Config(**raw)


def config_hash(cfg: Config) -> str:
    """Short stable id for a config (log column / run naming) - not a security hash."""
    payload = json.dumps(asdict(cfg), sort_keys=True)

    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # warn_only: a few ops have no deterministic implementation, so warn rather than crash
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        # "auto" resolves to CPU: it is the canonical, bit-exact, CI-compatible backend.
        # MPS is opt-in (--device mps)
        return torch.device("cpu")

    if pref in ("cpu", "mps"):
        if pref == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("device 'mps' requested but not available")
        return torch.device(pref)

    raise ValueError(f"unknown device: {pref!r} (expected auto|cpu|mps)")
