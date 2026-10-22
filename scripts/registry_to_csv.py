"""Registry -> CSV evidence: per-run final rows, per-factor aggregates, per-step curves.

Stdlib `csv` only, reading exactly what `plot_gap.py` reads. A `hash_match: false`
row is flagged and exits non-zero, so a config-hash breach cannot pass silently;
a missing `hash_match` key (a registry written before the runner amendment) is
treated as unknown, not as a breach.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from tinylm.plotting import load_ok_metrics, load_registry, repo_path
from tinylm.utils import Config, load_config

TRAIN_SATURATION = 0.99
LOW_VAL = 0.5
FINAL_VAL = 0.9

RUN_FIELDS = (
    "sweep",
    "config",
    "seed",
    "config_hash",
    "hash_match",
    "git_commit",
    "run_dir",
    "status",
    "backend",
    "wall_s",
    "torch_version",
    "final_train_acc",
    "final_val_acc",
    "val_acc_at_train_saturation",
    "max_val_acc",
    "grokked",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write sweep registry evidence as CSV.")
    p.add_argument("--registry", required=True, help="sweep registry.jsonl")
    p.add_argument("--x-factor", required=True, help="a Config field, e.g. n_train")
    p.add_argument(
        "--out-dir",
        default=None,
        help="default: docs/evidence/<sweep>-v1/ (repo-relative)",
    )

    return p.parse_args(argv)


def grok_stats(metrics: list[dict]) -> tuple[float | None, float, bool]:
    """(val_acc at first train-saturation, max val_acc, grokked).

    Grokked = some step has train_acc >= 0.99 with val_acc < 0.5, and the final
    val_acc is >= 0.9. Saturation val is None when train_acc never reaches 0.99.
    """
    saturated = next((m for m in metrics if m["train_acc"] >= TRAIN_SATURATION), None)
    grokked = (
        any(
            m["train_acc"] >= TRAIN_SATURATION and m["val_acc"] < LOW_VAL
            for m in metrics
        )
        and metrics[-1]["val_acc"] >= FINAL_VAL
    )

    return (
        saturated["val_acc"] if saturated is not None else None,
        max(m["val_acc"] for m in metrics),
        grokked,
    )


def write_csv(path: Path, fieldnames: Sequence[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.x_factor not in {f.name for f in dataclasses.fields(Config)}:
        print(f"unknown x-factor: {args.x_factor!r}", file=sys.stderr)
        sys.exit(2)

    registry_rows = load_registry(Path(args.registry))
    sweep = next(
        (
            r.get("sweep")
            for r in registry_rows
            if isinstance(r, dict) and r.get("sweep")
        ),
        Path(args.registry).parent.name,
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else repo_path(Path("docs") / "evidence" / f"{sweep}-v1")
    )

    runs: list[dict] = []
    trajectory: list[dict] = []
    by_factor: dict[object, list[float]] = {}
    # A genuine mismatch (both hashes present and different) is the only breach; a
    # missing hash_match key (pre-amendment registry) or None (unable to compare) is
    # unknown, not a breach.
    breaches = [
        f"{r.get('run_dir')} (seed {r.get('seed')})"
        for r in registry_rows
        if isinstance(r, dict) and r.get("hash_match") is False
    ]
    pairs, skipped = load_ok_metrics(Path(args.registry))

    for row, metrics in pairs:
        try:
            factor = getattr(load_config(repo_path(row["config"])), args.x_factor)

            if not all(
                "step" in m
                and math.isfinite(m["train_acc"])
                and math.isfinite(m["val_acc"])
                for m in metrics
            ):
                raise ValueError("no finite metrics")
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

        last = metrics[-1]
        sat_val, max_val, grokked = grok_stats(metrics)
        runs.append(
            {
                "sweep": row.get("sweep"),
                "config": row.get("config"),
                "seed": row.get("seed"),
                "config_hash": row.get("config_hash"),
                "hash_match": row.get("hash_match"),
                "git_commit": row.get("git_commit"),
                "run_dir": row.get("run_dir"),
                "status": row.get("status"),
                "backend": row.get("backend"),
                "wall_s": row.get("wall_s"),
                "torch_version": row.get("torch_version"),
                "final_train_acc": last["train_acc"],
                "final_val_acc": last["val_acc"],
                "val_acc_at_train_saturation": sat_val,
                "max_val_acc": max_val,
                "grokked": grokked,
            }
        )
        by_factor.setdefault(factor, []).append(last["val_acc"])
        trajectory.extend(
            {
                "seed": row.get("seed"),
                "step": m["step"],
                "train_acc": m["train_acc"],
                "val_acc": m["val_acc"],
            }
            for m in metrics
        )

    if not runs:
        print(f"no usable rows (skipped {skipped})", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "runs.csv", RUN_FIELDS, runs)
    write_csv(
        out_dir / f"by_{args.x_factor}.csv",
        (args.x_factor, "n", "mean_val_acc", "stdev_val_acc"),
        [
            {
                args.x_factor: factor,
                "n": len(values),
                "mean_val_acc": statistics.mean(values),
                "stdev_val_acc": (
                    statistics.stdev(values) if len(values) > 1 else None
                ),
            }
            for factor, values in sorted(
                by_factor.items(), key=lambda kv: (kv[0] is None, kv[0])
            )
        ],
    )
    write_csv(
        out_dir / "trajectory.csv",
        ("seed", "step", "train_acc", "val_acc"),
        trajectory,
    )

    print(
        f"wrote {out_dir}: {len(runs)} run(s), {len(by_factor)} factor value(s), "
        f"{len(trajectory)} trajectory row(s), skipped {skipped}"
    )

    if breaches:
        print(
            f"HASH MISMATCH in {len(breaches)} row(s): {'; '.join(breaches)}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
