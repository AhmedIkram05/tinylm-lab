"""Mod-arith task tests: determinism, pair disjointness, answers, sidecar agreement,
overfit smoke, and the copy-path regression guard."""

import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tinylm.data import generate_splits, make_train_generator, train_batch
from tinylm.metrics import answer_acc, causal_lm_loss
from tinylm.model import TinyLM
from tinylm.utils import Config, load_config, set_seeds

ROOT = Path(__file__).resolve().parent.parent


def modadd_cfg() -> Config:
    return load_config("configs/modadd.yaml")


def test_modadd_deterministic_under_seed() -> None:
    cfg = modadd_cfg()
    a, b = generate_splits(cfg), generate_splits(cfg)
    assert torch.equal(a.train, b.train) and torch.equal(a.val, b.val)
    assert torch.equal(a.train_probe, b.train_probe)
    assert a.train_patterns == b.train_patterns


def test_modadd_split_and_shapes() -> None:
    cfg = modadd_cfg()
    assert (cfg.n_train, cfg.n_val) == (48, 73)  # derived from train_frac 0.4 * 121
    s = generate_splits(cfg)

    assert s.train.shape == (48, 5) and s.val.shape == (73, 5)
    assert s.train_probe.shape == (48, 5)  # min(PROBE_SIZE, n_train) == whole train
    assert s.train_patterns.isdisjoint(s.val_patterns)
    # disjointness is on ORDERED pairs: a pair and its mirror are distinct patterns
    assert s.train_patterns | s.val_patterns == {
        (a, b) for a in range(11) for b in range(11)
    }


def test_modadd_answer_correctness() -> None:
    cfg = modadd_cfg()
    s = generate_splits(cfg)
    p = cfg.modulus

    for row in torch.cat([s.train, s.val]).tolist():
        a, plus, b, eq, ans = row
        assert (plus, eq) == (p, p + 1)
        assert a < p and b < p
        assert ans == (a + b) % p


def test_modadd_val_acc_matches_sidecar_answer_acc() -> None:
    run = "modadd-smoke"
    run_dir = ROOT / "logs" / run
    shutil.rmtree(run_dir, ignore_errors=True)  # a killed run must not poison this one

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tinylm.train",
                "--config",
                "configs/modadd.yaml",
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

        with (run_dir / "metrics.jsonl").open() as f:
            val_acc = [json.loads(line) for line in f if line.strip()][-1]["val_acc"]

        with (run_dir / "eval_positions.jsonl").open() as f:
            sidecar = [json.loads(line) for line in f if line.strip()][-1]

        cfg = modadd_cfg()
        assert len(sidecar["val_per_pos_acc"]) == cfg.seq_len - 1  # 4, never 31
        # modadd's accuracy window is the answer position alone, so these are the
        # same number on two code paths
        assert abs(val_acc - answer_acc(sidecar["val_per_pos_acc"], cfg.seq_len)) < 1e-6

        with (run_dir / "eval_examples.jsonl").open() as f:
            examples = [json.loads(line) for line in f if line.strip()][-1]

        # modadd's span is the answer token alone, so exact reproduction IS accuracy
        assert sum(examples["val_reproduced"]) / len(examples["val_reproduced"]) == (
            pytest.approx(val_acc, abs=1e-9)  # float32 mean vs exact division
        )
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_modadd_overfit_one_batch() -> None:
    cfg = modadd_cfg()  # WD=0.01; T=5/V=13 dynamics differ from copy, hence the 0.5 bar
    set_seeds(cfg.seed)
    splits = generate_splits(cfg)
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
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

    assert final < 0.5 * initial, f"{initial=} {final=}"


def test_copy_path_unchanged() -> None:
    """The modadd dispatch must not perturb the copy path: pin the split tensors."""
    s = generate_splits(load_config("configs/copy.yaml"))
    h = hashlib.sha256()
    h.update(s.train.numpy().tobytes())
    h.update(s.val.numpy().tobytes())
    h.update(s.train_probe.numpy().tobytes())
    h.update(repr(sorted(s.train_patterns)).encode())
    h.update(repr(sorted(s.val_patterns)).encode())

    assert h.hexdigest() == (
        "5dfb8f87fc0b62e0f00d00c0facc18182037e928794cef2bb4929776f0394509"
    )


def test_train_frac_derivation_and_overrides() -> None:
    """train_frac is derived in __post_init__, so every load_config agrees."""
    cfg = modadd_cfg()
    assert (cfg.n_train, cfg.n_val) == (48, 73)
    assert dataclasses.replace(cfg, train_frac=0.5).n_train == 60
    assert (
        dataclasses.replace(cfg, modulus=7, vocab_size=9, train_frac=0.5).n_train == 24
    )
    assert dataclasses.replace(cfg, train_frac=None, n_train=10, n_val=10).n_val == 10

    # a bad train_frac must fail as a train_frac error, before it reaches int()
    with pytest.raises(TypeError):
        dataclasses.replace(cfg, train_frac="0.4")
    with pytest.raises(ValueError, match="train_frac"):
        dataclasses.replace(cfg, train_frac=1.5)
    with pytest.raises(ValueError, match="train_frac"):
        dataclasses.replace(load_config("configs/copy.yaml"), train_frac=0.4)
