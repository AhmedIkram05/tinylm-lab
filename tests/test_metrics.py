"""Metrics tests: loss sanity, position ordering, aggregation, subprocess sidecar smoke."""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tinylm.data import generate_splits
from tinylm.metrics import (
    compute_metrics,
    per_example_span_metrics,
    per_position_metrics,
)
from tinylm.model import TinyLM
from tinylm.train import evaluate
from tinylm.utils import load_config

ROOT = Path(__file__).resolve().parent.parent


def test_uniform_logits_loss_is_ln_vocab() -> None:
    cfg = load_config("configs/copy.yaml")
    logits = torch.zeros(4, cfg.seq_len, cfg.vocab_size)
    targets = torch.zeros(4, cfg.seq_len, dtype=torch.long)

    per_pos_loss, per_pos_acc = per_position_metrics(logits, targets)

    assert len(per_pos_loss) == len(per_pos_acc) == cfg.seq_len - 1
    assert all(
        math.isclose(v, math.log(cfg.vocab_size), rel_tol=1e-5) for v in per_pos_loss
    )
    assert all(v == 1.0 for v in per_pos_acc)  # targets all 0; argmax of zeros is 0


def test_position_ordering() -> None:
    # T=5; only target position 3 (array index 2) is perfectly predicted
    targets = torch.tensor([[1, 2, 3, 4, 5]])
    logits = torch.zeros(1, 5, 8)
    logits[0, 2] = -10.0
    logits[0, 2, 4] = 10.0

    per_pos_loss, per_pos_acc = per_position_metrics(logits, targets)

    assert len(per_pos_loss) == len(per_pos_acc) == 4  # T-1
    assert per_pos_acc[2] == 1.0
    assert per_pos_acc[1] < 1.0 and per_pos_acc[3] < 1.0


def test_aggregation_matches_compute_metrics() -> None:
    cfg = load_config("configs/copy.yaml")
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, cfg.seq_len, cfg.vocab_size, generator=g)
    targets = torch.randint(0, cfg.vocab_size, (4, cfg.seq_len), generator=g)

    loss, acc = compute_metrics(logits, targets, cfg.period)
    per_pos_loss, per_pos_acc = per_position_metrics(logits, targets)

    assert torch.allclose(
        torch.tensor(per_pos_loss).mean(), torch.tensor(loss), atol=1e-6
    )
    assert torch.allclose(
        torch.tensor(per_pos_acc[cfg.period - 1 :]).mean(),
        torch.tensor(acc),
        atol=1e-6,
    )


def test_per_example_span_metrics() -> None:
    # T=5, period=4 (modadd): the span is the answer token alone
    targets = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]])
    logits = torch.zeros(2, 5, 8)
    logits[0, 3, 5] = 10.0  # example 0 predicts the answer; example 1 does not
    logits[1, 3, 7] = 10.0
    span_ce, reproduced = per_example_span_metrics(logits, targets, 4)
    assert len(span_ce) == len(reproduced) == 2
    assert reproduced == [1, 0]
    assert span_ce[0] < span_ce[1]

    # T=4, period=2 (copy-shaped): the span is 2 positions and reproduction is
    # conjunctive, so a miss at EITHER position fails it - including a miss at the
    # first span position only, which a last-position-only reading would accept
    targets = torch.tensor([[1, 2, 3, 4]] * 3)
    logits = torch.zeros(3, 4, 8)
    logits[0, 1, 3] = 10.0
    logits[0, 2, 4] = 10.0  # both span positions correct -> reproduced
    logits[1, 1, 3] = 10.0
    logits[1, 2, 0] = 10.0  # first span position correct, second wrong
    logits[2, 1, 0] = 10.0
    logits[2, 2, 4] = 10.0  # first wrong, second right -> still not reproduced
    span_ce, reproduced = per_example_span_metrics(logits, targets, 2)
    assert reproduced == [1, 0, 0]
    assert all(type(v) is int for v in reproduced)

    # the span is exactly the accuracy window: mean CE equals per-position tail
    per_pos_loss, _ = per_position_metrics(logits, targets)
    tail = per_pos_loss[2 - 1 :]
    assert sum(span_ce) / len(span_ce) == pytest.approx(sum(tail) / len(tail))


def test_eval_positions_sidecar_written() -> None:
    run = "evalx-smoke"
    run_dir = ROOT / "logs" / run
    shutil.rmtree(run_dir, ignore_errors=True)  # a killed run must not poison this one

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tinylm.train",
                "--steps",
                "50",
                "--device",
                "cpu",
                "--run-name",
                run,
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        with (run_dir / "eval_positions.jsonl").open() as f:
            rows = [json.loads(line) for line in f if line.strip()]

        assert len(rows) == 2  # --steps 50, eval_every 25 -> steps 25 and 50
        assert rows[-1]["step"] == 50
        assert rows[0].keys() == {
            "step",
            "seed",
            "config_hash",
            "train_per_pos_loss",
            "train_per_pos_acc",
            "val_per_pos_loss",
            "val_per_pos_acc",
        }

        cfg = load_config("configs/copy.yaml")
        assert len(rows[-1]["train_per_pos_loss"]) == cfg.seq_len - 1
        assert len(rows[-1]["val_per_pos_acc"]) == cfg.seq_len - 1

        with (run_dir / "metrics.jsonl").open() as f:
            metrics_rows = [json.loads(line) for line in f if line.strip()]

        # one per-example row per eval, train probe
        # and full val, exact-reproduction indicator + span CE
        with (run_dir / "eval_examples.jsonl").open() as f:
            examples = [json.loads(line) for line in f if line.strip()]

        assert len(examples) == len(rows) == len(metrics_rows) == 2
        assert [r["step"] for r in metrics_rows] == [25, 50]
        assert [r["step"] for r in rows] == [r["step"] for r in examples] == [25, 50]
        assert examples[0].keys() == {
            "step",
            "seed",
            "config_hash",
            "train_reproduced",
            "train_span_ce",
            "val_reproduced",
            "val_span_ce",
        }
        for m, e in zip(metrics_rows, examples, strict=True):
            assert e["seed"] == m["seed"]
            assert e["config_hash"] == m["config_hash"]

        last = examples[-1]
        probe_size = min(128, cfg.n_train)
        assert len(last["train_reproduced"]) == len(last["train_span_ce"]) == probe_size
        assert len(last["val_reproduced"]) == len(last["val_span_ce"]) == cfg.n_val
        assert all(type(v) is int and v in (0, 1) for v in last["val_reproduced"])
        assert all(
            math.isfinite(v) for v in last["train_span_ce"] + last["val_span_ce"]
        )
        # conjunctive reproduction is stricter than position-wise accuracy
        assert sum(last["val_reproduced"]) / len(last["val_reproduced"]) <= (
            metrics_rows[-1]["val_acc"] + 1e-9
        )

        # insurance: the state saved is the FINAL one, loadable and scoring the same
        state = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
        assert state["token_emb.weight"].shape == (cfg.vocab_size, cfg.d_model)
        model = TinyLM(cfg)
        assert len(state) == len(model.state_dict())  # full state dict, not a fragment
        model.load_state_dict(state)
        _, reloaded_acc, *_ = evaluate(
            model, generate_splits(cfg).val, cfg, torch.device("cpu")
        )
        assert reloaded_acc == pytest.approx(metrics_rows[-1]["val_acc"], abs=1e-6)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
