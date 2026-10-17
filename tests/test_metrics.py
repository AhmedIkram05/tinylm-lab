"""Metrics tests: loss sanity, position ordering, aggregation, subprocess sidecar smoke."""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from tinylm.metrics import compute_metrics, per_position_metrics
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
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
