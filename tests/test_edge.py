"""Edge-shape tests: odd-but-legal configs work, illegal input fails loudly."""

import dataclasses

import pytest
import torch

from tinylm.model import CausalSelfAttention, TinyLM, sinusoidal_pe
from tinylm.utils import Config, load_config


def demo_cfg() -> Config:
    return load_config("configs/copy.yaml")


def test_sinusoidal_pe_odd_d_model() -> None:
    pe = sinusoidal_pe(32, 7)

    assert pe.shape == (32, 7)
    # row 0 is sin(0)=0 in the even columns and cos(0)=1 in the odd ones; the odd
    # column count is one short, which is exactly the case the div[:...] trim handles
    assert pe[0].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_odd_divisible_model_builds_and_forwards() -> None:
    cfg = dataclasses.replace(demo_cfg(), d_model=96, n_heads=3, d_ff=384)
    model = TinyLM(cfg)
    model.eval()

    logits = model(torch.zeros(2, cfg.seq_len, dtype=torch.long))

    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)


def test_indivisible_d_model_raises_in_both_places() -> None:
    # 100 % 3 == 1: genuinely indivisible (100 % 4 == 0 would raise nothing)
    with pytest.raises(ValueError, match="divisible"):
        dataclasses.replace(demo_cfg(), d_model=100, n_heads=3)

    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(d_model=100, n_heads=3, dropout=0.0)


# (overrides, expected message) - the message pins WHICH rule fired, so a case cannot
# keep passing later by raising for some unrelated reason
BAD_VALUES = {
    "negative_lr": ({"lr": -1e-3}, "lr must be positive"),
    "dropout_one": ({"dropout": 1.0}, r"dropout must be in \[0, 1\)"),
    "period_above_seq_len": (
        {"period": 64},
        "period must be less than seq_len",
    ),
    "period_equals_seq_len": (
        {"period": 32},  # legal to dataclass shape rules, but leaves an empty window
        "period must be less than seq_len",
    ),
    "seq_len_not_multiple_of_period": (
        {"period": 5},
        "seq_len must be a multiple of period",
    ),
    "zero_steps": ({"steps": 0}, "steps must be positive"),
    "unknown_task": ({"task": "bogus"}, "unknown task"),
    "modadd_seq_len_not_5": (
        {"task": "modadd", "seq_len": 6, "vocab_size": 13, "period": 5},
        "modadd requires seq_len == 5",
    ),
    "modadd_vocab_not_modulus_plus_2": (
        {"task": "modadd", "seq_len": 5, "vocab_size": 14, "period": 4},
        r"modadd requires vocab_size == modulus \+ 2",
    ),
    "train_frac_zero": (
        {
            "task": "modadd",
            "seq_len": 5,
            "vocab_size": 13,
            "period": 4,
            "train_frac": 0.0,
        },
        r"train_frac must be in \(0, 1\)",
    ),
    "seed_negative": ({"seed": -1}, r"seed must be in \[0, 2\*\*32\)"),
    "seed_above_uint32": ({"seed": 2**32}, r"seed must be in \[0, 2\*\*32\)"),
}


@pytest.mark.parametrize(
    ("overrides", "message"),
    BAD_VALUES.values(),
    ids=list(BAD_VALUES),
)
def test_bad_config_values_raise(overrides: dict, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        dataclasses.replace(demo_cfg(), **overrides)


def test_unknown_yaml_key_rejected(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("vocab_size: 8\nbogus_key: 1\n")

    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(bad)


def test_missing_config_file_raises_oserror() -> None:
    # an OSError subclass, so the Step-1 handler in main() reports it as a config error
    with pytest.raises(OSError):
        load_config("configs/does-not-exist.yaml")


def test_eval_every_above_steps_is_legal() -> None:
    """`eval_every > steps` must stay legal (one eval, at the final step).

    The construction below IS the assertion - it fails if a future validation rule
    starts rejecting this shape, which is the regression this guards.
    """
    cfg = dataclasses.replace(demo_cfg(), steps=10, eval_every=100)

    assert cfg.eval_every > cfg.steps
