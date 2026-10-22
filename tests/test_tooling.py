"""Tooling tests: plot stubs + registry_to_csv on hand-written registries.

One test per contract: band alignment, malformed-line tolerance, the plot
smoke, skip-with-count, the hash-breach flag, and the grok rule. Fixtures
go in pytest's tmp dir, never `logs/`, and no training run is spawned.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tinylm.plotting import aligned_mean_std, load_metrics_rows

ROOT = Path(__file__).resolve().parent.parent


def write_metrics(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def write_registry(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def metric_rows(
    seed: int, train_accs: list[float], val_accs: list[float]
) -> list[dict]:
    return [
        {
            "step": (i + 1) * 25,
            "train_loss": 0.1,
            "train_acc": train,
            "val_loss": 0.2,
            "val_acc": val,
            "lr": 1e-3,
            "elapsed": float(i),
            "seed": seed,
            "config_hash": "deadbeef0000",
            "torch_version": "2.14.1",
        }
        for i, (train, val) in enumerate(zip(train_accs, val_accs, strict=True))
    ]


def ok_row(run_dir: Path, seed: int = 0, **extra) -> dict:
    return {
        "sweep": "t",
        "config": "configs/copy_n064.yaml",
        "seed": seed,
        "run_dir": str(run_dir),
        "status": "ok",
        **extra,
    }


def run_script(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_aligned_mean_std_matches_by_step() -> None:
    """A short series must not zip-truncate the band; n=1 steps get no fake std."""
    long = ([25, 50, 75], [0.2, 0.4, 0.6])
    short = ([25, 50], [0.4, 0.6])
    band = aligned_mean_std([long, short])

    assert band is not None
    steps, means, stds = band
    assert steps == [25, 50]
    assert means == [pytest.approx(0.3), pytest.approx(0.5)]
    assert stds[0] > 0 and stds[1] > 0
    assert aligned_mean_std([long]) is None  # single series: no band


def test_load_metrics_rows_skips_malformed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.jsonl").write_text('{"step": 1, "train_acc": 1.0}\nnot-json\n')

    rows = load_metrics_rows(run)

    assert len(rows) == 1
    assert rows[0]["step"] == 1
    assert load_metrics_rows(tmp_path / "missing") == []


def test_plot_stubs_write_pngs(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    write_metrics(run, metric_rows(0, [0.5, 1.0], [0.4, 0.8]))
    registry = tmp_path / "registry.jsonl"
    write_registry(registry, [ok_row(run)])

    gap_png, traj_png = tmp_path / "gap.png", tmp_path / "traj.png"

    for script, extra in (
        ("plot_gap.py", ("--x-factor", "n_train")),
        ("plot_trajectory.py", ("--metric", "val_acc")),
    ):
        out = gap_png if script == "plot_gap.py" else traj_png
        proc = run_script(
            script, "--registry", str(registry), *extra, "--out", str(out)
        )
        assert proc.returncode == 0, proc.stderr

    assert gap_png.exists() and traj_png.exists()


def test_plot_stubs_skip_unreadable_rows(tmp_path: Path) -> None:
    """The never-crash contract: a run with no metrics, and a failed row, are skipped."""
    good = tmp_path / "run-good"
    write_metrics(good, metric_rows(0, [1.0], [0.8]))
    registry = tmp_path / "registry.jsonl"
    write_registry(
        registry,
        [
            ok_row(good, seed=0),
            ok_row(tmp_path / "gone", seed=1),  # missing metrics.jsonl
            {
                "sweep": "t",
                "config": "configs/nope.yaml",
                "seed": 2,
                "run_dir": str(tmp_path / "failed"),
                "status": "failed",
            },
        ],
    )

    for script, extra in (
        ("plot_gap.py", ("--x-factor", "n_train")),
        ("plot_trajectory.py", ("--metric", "val_acc")),
    ):
        out = tmp_path / f"{script}.png"
        proc = run_script(
            script, "--registry", str(registry), *extra, "--out", str(out)
        )
        assert proc.returncode == 0, proc.stderr
        assert "skipped 2" in proc.stdout
        assert out.exists()


def test_registry_to_csv_flags_a_hash_breach(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    write_metrics(run, metric_rows(0, [1.0], [0.9]))
    registry = tmp_path / "registry.jsonl"
    write_registry(registry, [ok_row(run, hash_match=False)])

    out = tmp_path / "evidence"
    proc = run_script(
        "registry_to_csv.py",
        "--registry",
        str(registry),
        "--x-factor",
        "n_train",
        "--out-dir",
        str(out),
    )

    assert proc.returncode == 1
    assert "HASH MISMATCH" in proc.stderr
    assert (out / "runs.csv").exists()  # evidence is written before flagging


def test_registry_to_csv_writes_evidence_and_grok(tmp_path: Path) -> None:
    """One real fixture: the three CSVs, the pre-amendment key, and the grok rule."""
    grok = tmp_path / "grok"  # saturates with low val, then val >= 0.9 at the end
    write_metrics(grok, metric_rows(0, [1.0, 1.0, 1.0], [0.1, 0.3, 0.95]))
    plateau = tmp_path / "plateau"  # memorises, but ends in 0.5-0.9: NOT grokked
    write_metrics(plateau, metric_rows(1, [1.0, 1.0, 1.0], [0.1, 0.4, 0.7]))
    flat = tmp_path / "flat"  # never reaches 0.99 -> no saturation step
    write_metrics(flat, metric_rows(2, [0.4, 0.5], [0.2, 0.3]))
    registry = tmp_path / "registry.jsonl"
    write_registry(
        registry,
        [
            ok_row(grok, seed=0, hash_match=True),  # post-amendment true path
            ok_row(plateau, seed=1),  # no hash_match key: must not error
            {
                "sweep": "t",
                "config": "configs/copy.yaml",
                "seed": 2,
                "run_dir": str(flat),
                "status": "ok",
            },
        ],
    )

    out = tmp_path / "evidence"
    proc = run_script(
        "registry_to_csv.py",
        "--registry",
        str(registry),
        "--x-factor",
        "n_train",
        "--out-dir",
        str(out),
    )
    assert proc.returncode == 0, proc.stderr

    for name in ("runs.csv", "by_n_train.csv", "trajectory.csv"):
        assert (out / name).exists()

    by_seed = {r["seed"]: r for r in csv.DictReader((out / "runs.csv").open())}
    assert by_seed["0"]["hash_match"] == "True"
    assert by_seed["1"]["hash_match"] == ""  # missing key -> null, never a breach
    assert by_seed["0"]["grokked"] == "True"
    assert by_seed["0"]["val_acc_at_train_saturation"] == "0.1"
    assert by_seed["0"]["max_val_acc"] == "0.95"
    # memorised but never generalised past 0.9 -> the partial plateau is NOT grokked
    assert by_seed["1"]["grokked"] == "False"
    assert by_seed["1"]["val_acc_at_train_saturation"] == "0.1"
    assert by_seed["1"]["max_val_acc"] == "0.7"
    # never saturates -> no saturation step recorded
    assert by_seed["2"]["grokked"] == "False"
    assert by_seed["2"]["val_acc_at_train_saturation"] == ""

    # two factor values (64 from copy_n064, 512 from copy.yaml) and one row per step
    assert len(list(csv.DictReader((out / "by_n_train.csv").open()))) == 2
    assert len(list(csv.DictReader((out / "trajectory.csv").open()))) == 8
