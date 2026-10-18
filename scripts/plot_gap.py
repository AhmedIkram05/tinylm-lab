"""Gap-vs-factor plot from a sweep registry: train_acc - val_acc, 10-seed error bars.

Reads only the registry columns it needs; a row whose config or metrics cannot be
read is skipped with a printed count, never a crash. Single-seed groups plot as
points with no fake error bars.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from tinylm.palette import TRAIN_BLUE, VAL_VERMILION
from tinylm.plotting import load_metrics_rows, load_registry, repo_path
from tinylm.utils import Config, load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot final-row gap against a config factor."
    )
    p.add_argument("--registry", required=True, help="sweep registry.jsonl")
    p.add_argument("--x-factor", required=True, help="a Config field, e.g. n_train")
    p.add_argument("--out", required=True, help="output PNG path")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.x_factor not in {f.name for f in dataclasses.fields(Config)}:
        print(f"unknown x-factor: {args.x_factor!r}", file=sys.stderr)
        sys.exit(2)

    groups: dict[object, list[float]] = defaultdict(list)
    skipped = 0

    for row in load_registry(Path(args.registry)):
        if not isinstance(row, dict) or row.get("status") != "ok":
            skipped += 1
            continue

        try:
            # load_config (not the raw YAML) so a config that omits the factor still fills it
            factor = getattr(load_config(repo_path(row["config"])), args.x_factor)
            last = load_metrics_rows(repo_path(row["run_dir"]))[-1]
            gap = last["train_acc"] - last["val_acc"]

            if not math.isfinite(gap):  # a diverged run must not poison the mean
                raise ValueError("non-finite gap")
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
            yaml.YAMLError,
        ):
            skipped += 1
            continue

        groups[factor].append(gap)

    if not groups:
        print(f"no plottable rows (skipped {skipped})", file=sys.stderr)
        sys.exit(1)

    xs = sorted(groups, key=lambda x: (x is None, x))
    multi = [x for x in xs if len(groups[x]) > 1]
    single = [x for x in xs if len(groups[x]) == 1]

    fig, ax = plt.subplots(figsize=(6, 4))

    if multi:
        ax.errorbar(
            multi,
            [statistics.mean(groups[x]) for x in multi],
            yerr=[statistics.stdev(groups[x]) for x in multi],
            fmt="o-",
            color=VAL_VERMILION,
            ecolor=VAL_VERMILION,
            capsize=4,
            label="mean ± sample std",
        )

    if single:
        ax.plot(
            single,
            [groups[x][0] for x in single],
            "o",
            color=TRAIN_BLUE,
            label="single seed",
        )

    ax.set_xlabel(args.x_factor)
    ax.set_ylabel("gap = train_acc - val_acc")
    ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)

    print(f"wrote {args.out}: {len(xs)} factor value(s), skipped {skipped} row(s)")


if __name__ == "__main__":
    main()
