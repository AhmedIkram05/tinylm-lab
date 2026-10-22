"""Shared artefact readers for the sweep tooling (plots, CSV export).

The readers are stdlib-only so both `scripts/*.py` entry points and any future
consumer import them as first-party code instead of reaching sideways into a
sibling script. Colours live in `tinylm/palette.py`.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_path(value: str | Path) -> Path:
    """Resolve a registry path: absolute paths stay put, relatives are repo-rooted."""
    path = Path(value)

    return path if path.is_absolute() else REPO_ROOT / path


def load_jsonl(path: Path) -> list[dict]:
    """JSONL rows; blank or malformed lines are skipped with a stderr warning."""
    rows: list[dict] = []

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"warning: {path}:{lineno}: skipping malformed row", file=sys.stderr
                )
                print(f"  ({exc})", file=sys.stderr)

    return rows


def load_registry(path: Path) -> list[dict]:
    """Every row of a sweep registry.jsonl."""
    return load_jsonl(path)


def load_metrics_rows(run_dir: Path) -> list[dict]:
    """Every logged row for a run; [] when the file is absent (callers skip, never crash)."""
    path = run_dir / "metrics.jsonl"

    if not path.exists():
        return []

    return load_jsonl(path)


def load_ok_metrics(registry_path: Path) -> tuple[list[tuple[dict, list[dict]]], int]:
    """(row, metrics) for each status==ok run with a non-empty metrics.jsonl.

    Failed, malformed, and metrics-less runs are counted and skipped (never crash);
    the returned ``skipped`` count is what callers report. Callers still apply their
    own per-factor extraction and finite-metric checks.
    """
    pairs: list[tuple[dict, list[dict]]] = []
    skipped = 0

    for row in load_registry(registry_path):
        if not isinstance(row, dict) or row.get("status") != "ok":
            skipped += 1
            continue

        try:
            metrics = load_metrics_rows(repo_path(row["run_dir"]))

            if not metrics:
                raise ValueError("no metric rows")
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
            skipped += 1
            continue

        pairs.append((row, metrics))

    return pairs, skipped


def aligned_mean_std(
    series: list[tuple[list[int], list[float]]],
) -> tuple[list[int], list[float], list[float]] | None:
    """Mean ± sample std at steps present in at least two series.

    Aligns by step (not zip-index) so a short run cannot silently truncate the
    band. Returns None when no step has two or more values — no fake band.
    """
    by_step: dict[int, list[float]] = defaultdict(list)

    for steps, values in series:
        for step, value in zip(steps, values, strict=True):
            by_step[step].append(value)

    steps = [s for s in sorted(by_step) if len(by_step[s]) > 1]

    if not steps:
        return None

    return (
        steps,
        [statistics.mean(by_step[s]) for s in steps],
        [statistics.stdev(by_step[s]) for s in steps],
    )
