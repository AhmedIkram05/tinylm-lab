"""The project's loss and metric definitions, kept out of the frozen log schema."""

from __future__ import annotations

import torch
from torch.nn import functional as F


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
    # preds index j-1 predicts target j (same convention as per_position_metrics), so
    # the accuracy window starts one earlier on preds than on targets
    acc = (preds[:, period - 1 :] == targets[:, period:]).float().mean().item()

    return loss, acc


def per_position_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[list[float], list[float]]:
    """CE loss and argmax accuracy at every predicted position.

    Returns ``(per_pos_loss, per_pos_acc)``, each of length ``T-1``: index
    ``j-1`` holds the metric for target position ``j`` (``j = 1..T-1``), whose
    prediction is ``logits[:, j-1]``. ``targets[:, 0]`` is never predicted.
    Loss is the per-position mean over the batch, accuracy the per-position
    argmax-match rate. Positions before the copy window sit at chance - that
    is the diagnostic, so every position is reported.
    """
    logits = logits[:, :-1]  # (B, T-1, V): distribution for targets[:, 1:]
    tgt = targets[:, 1:]  # (B, T-1)

    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none"
    ).reshape(tgt.shape)

    per_pos_loss = losses.mean(dim=0).tolist()
    per_pos_acc = (logits.argmax(dim=-1) == tgt).float().mean(dim=0).tolist()

    return per_pos_loss, per_pos_acc


def answer_acc(per_pos_acc: list[float], seq_len: int) -> float:
    """Accuracy at the answer position: the last predicted position, index ``seq_len - 2``.

    Modadd's ``metrics.jsonl`` accuracy window covers only that position, so this
    must agree with ``val_acc``. Asserts the sidecar array length rather than
    assuming ``T - 1 == 4``.
    """
    assert len(per_pos_acc) == seq_len - 1, (
        f"per_pos_acc has {len(per_pos_acc)} entries, expected {seq_len - 1}"
    )

    return per_pos_acc[seq_len - 2]
