"""Seeded training loop for the period-copy task. Logs JSONL + one plot."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from .data import generate_splits, make_train_generator, train_batch
from .model import TinyLM
from .utils import Config, config_hash, get_device, load_config, set_seeds

TRAIN_BLUE = "#0072B2"  # Okabe-Ito, colour-blind safe colours palette
VAL_VERMILION = "#D55E00"


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """The project's one loss: next-token CE over positions 1..T-1."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), targets[:, 1:].reshape(-1)
    )


def compute_metrics(
    logits: torch.Tensor, targets: torch.Tensor, period: int
) -> tuple[float, float]:
    """Mean cross-entropy loss over positions 1..T-1; accuracy over copyable positions (period..T-1)."""
    loss = causal_lm_loss(logits, targets).item()

    preds = logits[:, :-1].argmax(dim=-1)  # (B, T-1): predictions for positions 1..T-1
    acc = (preds[:, period - 1 :] == targets[:, period:]).float().mean().item()

    return loss, acc


def evaluate(
    model: TinyLM, seqs: torch.Tensor, cfg: Config, device: torch.device
) -> tuple[float, float]:
    """Loss/acc on a fixed sequence set, in eval mode under no_grad."""
    model.eval()

    with torch.no_grad():
        targets = seqs.to(device)
        return compute_metrics(model(targets), targets, cfg.period)


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
    t0 = time.perf_counter()
    h, tv = config_hash(cfg), torch.__version__

    def log_row(step: int) -> None:
        train_loss, train_acc = evaluate(model, splits.train_probe, cfg, device)
        val_loss, val_acc = evaluate(model, splits.val, cfg, device)

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

    with (run_dir / "metrics.jsonl").open("w") as f:
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
        description="Train the TinyLM prototype on the period-copy task."
    )
    p.add_argument("--config", default="configs/copy.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", default=None)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)

    if args.steps is not None:
        cfg.steps = args.steps

    if args.device is not None:
        cfg.device = args.device

    cfg.validate()
    run_name = args.run_name or f"copy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"  # noqa: DTZ005
    run_training(cfg, Path("logs") / run_name)


if __name__ == "__main__":
    main()
