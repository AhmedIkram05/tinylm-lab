"""CPU bit-exact repro: two subprocess runs, identical rows except `elapsed`."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_rows(run: str) -> list[dict]:
    rows = []

    with (ROOT / "logs" / run / "metrics.jsonl").open() as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def test_cpu_repro() -> None:
    runs = ["repro-a", "repro-b"]

    for run in runs:
        shutil.rmtree(
            ROOT / "logs" / run, ignore_errors=True
        )  # a killed run must not poison this one
    try:
        for run in runs:
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

        a, b = (read_rows(r) for r in runs)
        assert len(a) == len(b) == 2  # --steps 50, eval_every 25 -> steps 25 and 50
        assert a[-1]["step"] == b[-1]["step"] == 50

        for ra, rb in zip(a, b, strict=True):
            assert ra.keys() == rb.keys()
            assert {k: v for k, v in ra.items() if k != "elapsed"} == {
                k: v for k, v in rb.items() if k != "elapsed"
            }
    finally:
        for run in runs:
            shutil.rmtree(ROOT / "logs" / run, ignore_errors=True)
