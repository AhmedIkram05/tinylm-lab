"""Metric-vs-step trajectory from a sweep registry: per-run lines under a mean ± std band.

Grokking is a time phenomenon, so a final-row gap cannot show it: every seed's run
is drawn as a thin line under the aggregate band (a mean-only curve would smear
"3/10 seeds grokked"). Readers come from `tinylm/plotting.py`.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tinylm.palette import TRAIN_BLUE, VAL_VERMILION
from tinylm.plotting import (
    aligned_mean_std,
    load_metrics_rows,
    load_registry,
    repo_path,
)

GROK_THRESHOLD = 0.9
METRIC_COLOURS = {"val_acc": VAL_VERMILION, "train_acc": TRAIN_BLUE}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot a metric against step for a sweep.")
    p.add_argument("--registry", required=True, help="sweep registry.jsonl")
    p.add_argument("--metric", default="val_acc", choices=sorted(METRIC_COLOURS))
    p.add_argument("--out", required=True, help="output PNG path")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    colour = METRIC_COLOURS[args.metric]
    series: list[tuple[list[int], list[float]]] = []
    skipped = 0

    for row in load_registry(Path(args.registry)):
        if not isinstance(row, dict) or row.get("status") != "ok":
            skipped += 1
            continue

        try:
            metrics = load_metrics_rows(repo_path(row["run_dir"]))
            steps = [m["step"] for m in metrics]
            values = [m[args.metric] for m in metrics]

            # None/NaN/inf must not reach statistics.mean (and must not poison the band)
            if not steps or not all(math.isfinite(v) for v in values):
                raise ValueError("no finite metric values")
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
            skipped += 1
            continue

        series.append((steps, values))

    if not series:
        print(f"no plottable rows (skipped {skipped})", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(7, 4))
    many = len(series) > 1

    for steps, values in series:
        ax.plot(
            steps,
            values,
            color=colour,
            alpha=0.25 if many else 1.0,
            linewidth=0.8 if many else 1.5,
        )

    band = aligned_mean_std(series) if many else None

    if band is not None:
        steps, means, stds = band
        ax.plot(steps, means, color=colour, linewidth=2.0, label="mean")
        ax.fill_between(
            steps,
            [m - s for m, s in zip(means, stds, strict=True)],
            [m + s for m, s in zip(means, stds, strict=True)],
            color=colour,
            alpha=0.2,
            label="mean ± 1 sample std",
        )

    if args.metric == "val_acc":
        ax.axhline(
            GROK_THRESHOLD,
            linestyle="--",
            color="black",
            linewidth=0.8,
            label=f"{GROK_THRESHOLD} grok threshold",
        )

    ax.set_xlabel("step")
    ax.set_ylabel(args.metric)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)

    print(f"wrote {args.out}: {len(series)} run(s), skipped {skipped} row(s)")


if __name__ == "__main__":
    main()
