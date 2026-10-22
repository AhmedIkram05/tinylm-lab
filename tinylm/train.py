"""Seeded training loop for the copy and mod-arith tasks. Logs JSONL + sidecar + one plot."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from .data import generate_splits, make_train_generator, train_batch
from .metrics import (
    causal_lm_loss,
    compute_metrics,
    per_example_span_metrics,
    per_position_metrics,
)
from .model import TinyLM
from .palette import TRAIN_BLUE, VAL_VERMILION
from .utils import Config, config_hash, get_device, load_config, set_seeds


def evaluate(
    model: TinyLM, seqs: torch.Tensor, cfg: Config, device: torch.device
) -> tuple[float, float, list[float], list[float], list[float], list[int]]:
    """Loss/acc, per-position, and per-example diagnostics on a fixed set.

    One forward, four consumers: scalar metrics, the per-position sidecar and the
    per-example span sidecar (the memorisation read) all share the same logits.
    """
    model.eval()

    with torch.no_grad():
        targets = seqs.to(device)
        logits = model(targets)
        loss, acc = compute_metrics(logits, targets, cfg.period)
        per_pos_loss, per_pos_acc = per_position_metrics(logits, targets)
        span_ce, reproduced = per_example_span_metrics(logits, targets, cfg.period)

        return loss, acc, per_pos_loss, per_pos_acc, span_ce, reproduced


def plot_learning_curve(rows: list[dict], out: Path, vocab_size: int) -> None:
    """One figure, two panels (loss, accuracy) + dashed chance baseline at 1/vocab."""
    steps = [r["step"] for r in rows]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(
        steps, [r["train_loss"] for r in rows], color=TRAIN_BLUE, label="train"
    )
    ax_loss.plot(steps, [r["val_loss"] for r in rows], color=VAL_VERMILION, label="val")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()
    ax_acc.plot(steps, [r["train_acc"] for r in rows], color=TRAIN_BLUE, label="train")
    ax_acc.plot(steps, [r["val_acc"] for r in rows], color=VAL_VERMILION, label="val")
    ax_acc.axhline(1.0 / vocab_size, linestyle="--", color="black", label="chance")
    ax_acc.set_xlabel("step")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def run_training(cfg: Config, run_dir: Path) -> list[dict]:
    set_seeds(cfg.seed)
    device = get_device(cfg.device)
    splits = generate_splits(cfg)
    model = TinyLM(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )
    g = make_train_generator(cfg)
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a run
    rows: list[dict] = []
    eval_rows: list[dict] = []
    example_rows: list[dict] = []
    t0 = time.perf_counter()
    h, tv = config_hash(cfg), torch.__version__

    def log_row(step: int) -> None:
        (
            train_loss,
            train_acc,
            train_ploss,
            train_pacc,
            train_xce,
            train_xrep,
        ) = evaluate(model, splits.train_probe, cfg, device)
        val_loss, val_acc, val_ploss, val_pacc, val_xce, val_xrep = evaluate(
            model, splits.val, cfg, device
        )

        rows.append(
            {
                "step": step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": cfg.lr,
                "elapsed": time.perf_counter() - t0,
                "seed": cfg.seed,
                "config_hash": h,
                "torch_version": tv,
            }
        )
        eval_rows.append(
            {
                "step": step,
                "seed": cfg.seed,
                "config_hash": h,
                "train_per_pos_loss": train_ploss,
                "train_per_pos_acc": train_pacc,
                "val_per_pos_loss": val_ploss,
                "val_per_pos_acc": val_pacc,
            }
        )
        example_rows.append(
            {
                "step": step,
                "seed": cfg.seed,
                "config_hash": h,
                "train_reproduced": train_xrep,
                "train_span_ce": train_xce,
                "val_reproduced": val_xrep,
                "val_span_ce": val_xce,
            }
        )

    metrics_path = run_dir / "metrics.jsonl"
    positions_path = run_dir / "eval_positions.jsonl"
    examples_path = run_dir / "eval_examples.jsonl"

    with (
        metrics_path.open("w") as f,
        positions_path.open("w") as fx,
        examples_path.open("w") as fe,
    ):
        for step in range(1, cfg.steps + 1):
            model.train()
            batch = train_batch(splits, cfg, g).to(device)
            loss = causal_lm_loss(model(batch), batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % cfg.eval_every == 0 or step == cfg.steps:
                log_row(step)
                f.write(json.dumps(rows[-1]) + "\n")
                f.flush()
                fx.write(json.dumps(eval_rows[-1]) + "\n")
                fx.flush()
                fe.write(json.dumps(example_rows[-1]) + "\n")
                fe.flush()

    torch.save(
        {k: v.cpu() for k, v in model.state_dict().items()}, run_dir / "model.pt"
    )
    plot_learning_curve(rows, run_dir / "learning_curve.png", cfg.vocab_size)
    last = rows[-1]

    print(
        f"step {last['step']}: train_loss={last['train_loss']:.4f} "
        f"train_acc={last['train_acc']:.4f} val_loss={last['val_loss']:.4f} "
        f"val_acc={last['val_acc']:.4f} -> {run_dir}"
    )

    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the TinyLM prototype on a synthetic task (--config selects copy or modadd)."
    )
    p.add_argument("--config", default="configs/copy.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        cfg = load_config(args.config)

        if args.seed is not None:
            cfg.seed = args.seed

        if args.steps is not None:
            cfg.steps = args.steps

        if args.device is not None:
            cfg.device = args.device

        cfg.validate()
        get_device(cfg.device)

    except (OSError, ValueError, TypeError, RuntimeError, yaml.YAMLError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        sys.exit(2)

    run_name = args.run_name or f"copy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"  # noqa: DTZ005
    run_training(cfg, Path("logs") / run_name)


if __name__ == "__main__":
    main()
